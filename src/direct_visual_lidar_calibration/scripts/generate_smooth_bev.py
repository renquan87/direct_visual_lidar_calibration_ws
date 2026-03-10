#!/usr/bin/env python3
"""
generate_smooth_bev.py — 生成平滑连续的彩色 BEV 俯视图

不同于简单的点投影，这里：
1. 每个点用圆形渲染（可调半径）
2. 用最近邻插值填充空隙
3. 高斯模糊平滑
4. 可选中值滤波去噪

结果看起来像连续的面，而不是离散的点。
"""
import os, sys, argparse
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import NearestNDInterpolator

import open3d as o3d

CAPTURE_DIR = "/data/projects/radar/direct_visual_lidar_calibration_ws/colored_captures"


def points_to_bev_smooth(points, colors, resolution=0.01, point_radius=3,
                          blur_size=5, fill_holes=True, max_fill_dist=20):
    """
    将3D彩色点云投影为平滑的BEV图像

    Args:
        points: (N,3) 点坐标
        colors: (N,3) RGB [0,1]
        resolution: 米/像素
        point_radius: 每个点的渲染半径（像素）
        blur_size: 高斯模糊核大小
        fill_holes: 是否填充空隙
        max_fill_dist: 最大填充距离（像素）
    """
    x, y = points[:, 0], points[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # 加边距
    margin = 0.5  # 米
    x_min -= margin; x_max += margin
    y_min -= margin; y_max += margin

    w = int((x_max - x_min) / resolution) + 1
    h = int((y_max - y_min) / resolution) + 1

    print(f"  BEV 尺寸: {w}x{h} (分辨率={resolution}m/px)", flush=True)

    # 像素坐标
    px = ((x - x_min) / resolution).astype(int)
    py = ((y_max - y) / resolution).astype(int)  # Y轴翻转

    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)

    # 创建图像和掩码
    img = np.zeros((h, w, 3), dtype=np.float64)
    count = np.zeros((h, w), dtype=np.float64)

    colors_u8 = (np.clip(colors, 0, 1) * 255).astype(np.float64)

    # 用圆形渲染每个点
    print(f"  渲染 {len(points)} 点 (半径={point_radius}px)...", flush=True)

    if point_radius <= 1:
        # 快速路径：直接像素赋值
        for i in range(len(points)):
            img[py[i], px[i]] += colors_u8[i]
            count[py[i], px[i]] += 1
    else:
        # 用 OpenCV 画圆
        for i in range(len(points)):
            cv2.circle(img, (px[i], py[i]), point_radius, colors_u8[i].tolist(), -1)
            cv2.circle(count, (px[i], py[i]), point_radius, 1.0, -1)

    # 平均化重叠区域
    mask = count > 0
    for c in range(3):
        img[:, :, c][mask] /= count[mask]

    img = img.astype(np.uint8)
    mask_u8 = mask.astype(np.uint8) * 255

    # 填充空隙
    if fill_holes:
        print(f"  填充空隙...", flush=True)

        # 方法：用最近邻插值填充
        empty_mask = ~mask

        if empty_mask.any() and mask.any():
            # 计算每个空像素到最近有值像素的距离
            dist, indices = distance_transform_edt(empty_mask, return_indices=True)

            # 只填充距离在阈值内的
            fill_mask = empty_mask & (dist <= max_fill_dist)

            if fill_mask.any():
                # 从最近的有值像素复制颜色
                nearest_y = indices[0][fill_mask]
                nearest_x = indices[1][fill_mask]
                img[fill_mask] = img[nearest_y, nearest_x]
                mask_u8[fill_mask] = 255

                filled = fill_mask.sum()
                print(f"    填充了 {filled} 像素", flush=True)

    # 高斯模糊平滑
    if blur_size > 0:
        print(f"  高斯模糊 (kernel={blur_size})...", flush=True)
        # 只对有值区域模糊
        blurred = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
        # 用掩码混合：有值区域用模糊结果，无值区域保持
        mask_3ch = np.stack([mask_u8, mask_u8, mask_u8], axis=-1) > 0
        img[mask_3ch] = blurred[mask_3ch]

    # 背景色
    bg_mask = mask_u8 == 0
    img[bg_mask] = [40, 40, 40]  # 深灰背景

    return img, (x_min, x_max, y_min, y_max)


def add_grid_overlay(img, bounds, resolution, grid_spacing=1.0):
    """添加网格线"""
    x_min, x_max, y_min, y_max = bounds
    h, w = img.shape[:2]

    # 垂直线
    for x in np.arange(np.ceil(x_min / grid_spacing) * grid_spacing, x_max, grid_spacing):
        px = int((x - x_min) / resolution)
        if 0 <= px < w:
            cv2.line(img, (px, 0), (px, h - 1), (80, 80, 80), 1)

    # 水平线
    for y in np.arange(np.ceil(y_min / grid_spacing) * grid_spacing, y_max, grid_spacing):
        py = int((y_max - y) / resolution)
        if 0 <= py < h:
            cv2.line(img, (0, py), (w - 1, py), (80, 80, 80), 1)

    return img


def add_scale_bar(img, resolution, bar_length_m=5.0):
    """添加比例尺"""
    h, w = img.shape[:2]
    bar_px = int(bar_length_m / resolution)
    x0, y0 = 30, h - 40
    cv2.rectangle(img, (x0, y0), (x0 + bar_px, y0 + 8), (255, 255, 255), -1)
    cv2.putText(img, f"{bar_length_m:.0f}m", (x0, y0 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=None)
    parser.add_argument('--input-dir', '-i', default=CAPTURE_DIR)
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--resolution', type=float, default=0.01, help='米/像素')
    parser.add_argument('--point-radius', type=int, default=3, help='点渲染半径(像素)')
    parser.add_argument('--blur', type=int, default=5, help='高斯模糊核大小(0=不模糊)')
    parser.add_argument('--max-fill', type=int, default=15, help='最大填充距离(像素)')
    parser.add_argument('--grid', type=float, default=1.0, help='网格间距(米, 0=无网格)')
    parser.add_argument('--z-min', type=float, default=None, help='Z轴最小值(过滤)')
    parser.add_argument('--z-max', type=float, default=None, help='Z轴最大值(过滤)')
    parser.add_argument('--no-fill', action='store_true', help='不填充空隙')
    args = parser.parse_args()

    input_path = args.input or os.path.join(args.input_dir, "merged_full_colored.ply")
    output = args.output or os.path.join(args.input_dir, "bev_smooth.png")

    print(f"读取: {input_path}", flush=True)
    pcd = o3d.io.read_point_cloud(input_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    print(f"  {len(points)} 点", flush=True)

    # Z轴过滤
    if args.z_min is not None or args.z_max is not None:
        z = points[:, 2]
        mask = np.ones(len(z), dtype=bool)
        if args.z_min is not None:
            mask &= z >= args.z_min
        if args.z_max is not None:
            mask &= z <= args.z_max
        points = points[mask]
        colors = colors[mask]
        print(f"  Z过滤后: {len(points)} 点", flush=True)

    print(f"\n生成平滑BEV...", flush=True)
    img, bounds = points_to_bev_smooth(
        points, colors,
        resolution=args.resolution,
        point_radius=args.point_radius,
        blur_size=args.blur,
        fill_holes=not args.no_fill,
        max_fill_dist=args.max_fill)

    # 添加网格
    if args.grid > 0:
        img = add_grid_overlay(img, bounds, args.resolution, args.grid)

    # 添加比例尺
    img = add_scale_bar(img, args.resolution)

    # BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output, img_bgr)
    print(f"\n✓ {output} ({img.shape[1]}x{img.shape[0]})", flush=True)

    # 也保存一个无网格版本
    output_clean = output.rsplit('.', 1)[0] + '_clean.png'
    img_clean, _ = points_to_bev_smooth(
        points, colors,
        resolution=args.resolution,
        point_radius=args.point_radius,
        blur_size=args.blur,
        fill_holes=not args.no_fill,
        max_fill_dist=args.max_fill)
    cv2.imwrite(output_clean, cv2.cvtColor(img_clean, cv2.COLOR_RGB2BGR))
    print(f"  无网格版: {output_clean}", flush=True)


if __name__ == '__main__':
    main()
