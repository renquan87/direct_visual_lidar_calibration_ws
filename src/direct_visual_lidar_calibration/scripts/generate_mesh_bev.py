#!/usr/bin/env python3
"""
generate_mesh_bev.py — 用 Delaunay 三角化生成面片式 BEV

1. 将点云投影到 XY 平面
2. 2D Delaunay 三角化，点之间连成三角面
3. 过滤过长的三角形边（避免跨区域连接）
4. 光栅化三角形，每个三角形内部颜色插值
5. 提高饱和度和亮度
"""
import os, sys, argparse
import numpy as np
import cv2
from scipy.spatial import Delaunay

import open3d as o3d

CAPTURE_DIR = "/data/projects/radar/direct_visual_lidar_calibration_ws/colored_captures"


def enhance_colors_hsv(colors, sat_boost=1.6, val_boost=1.4):
    """提高饱和度和亮度"""
    # colors: (N,3) float [0,1] RGB
    img = (np.clip(colors, 0, 1) * 255).astype(np.uint8).reshape(1, -1, 3)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[0, :, 1] = np.clip(hsv[0, :, 1] * sat_boost, 0, 255)
    hsv[0, :, 2] = np.clip(hsv[0, :, 2] * val_boost, 0, 255)
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return rgb.reshape(-1, 3).astype(np.float64) / 255.0


def bilateral_smooth(img, mask, d=9, sigma_color=40, sigma_space=15, iterations=2):
    """双边滤波平滑颜色（保边去噪，让相邻区域颜色统一）"""
    result = img.copy()
    for _ in range(iterations):
        filtered = cv2.bilateralFilter(result, d, sigma_color, sigma_space)
        # 只在有值区域应用
        result[mask > 0] = filtered[mask > 0]
    return result


def sharpen_edges(img, amount=0.5):
    """Unsharp mask 锐化边缘"""
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return sharpened


def generate_mesh_bev(points, colors, resolution=0.01, max_edge_len=0.15,
                       bg_color=(30, 30, 30)):
    """
    Delaunay 三角化 + 光栅化 BEV

    Args:
        points: (N,3)
        colors: (N,3) RGB [0,1]
        resolution: 米/像素
        max_edge_len: 最大三角形边长（米），超过的三角形被剔除
    """
    x, y = points[:, 0], points[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    margin = 0.3
    x_min -= margin; x_max += margin
    y_min -= margin; y_max += margin

    w = int((x_max - x_min) / resolution) + 1
    h = int((y_max - y_min) / resolution) + 1
    print(f"  BEV: {w}x{h} ({resolution}m/px)", flush=True)

    # 像素坐标
    px = ((x - x_min) / resolution).astype(np.float32)
    py = ((y_max - y) / resolution).astype(np.float32)

    # 2D Delaunay
    pts_2d = np.stack([x, y], axis=1)
    print(f"  Delaunay 三角化 ({len(pts_2d)} 点)...", flush=True)
    tri = Delaunay(pts_2d)
    simplices = tri.simplices
    print(f"  {len(simplices)} 三角形", flush=True)

    # 过滤过长边的三角形
    print(f"  过滤边长 > {max_edge_len}m...", flush=True)
    keep = []
    for s in simplices:
        p0, p1, p2 = pts_2d[s[0]], pts_2d[s[1]], pts_2d[s[2]]
        e0 = np.linalg.norm(p1 - p0)
        e1 = np.linalg.norm(p2 - p1)
        e2 = np.linalg.norm(p0 - p2)
        if max(e0, e1, e2) <= max_edge_len:
            keep.append(s)
    simplices = np.array(keep)
    print(f"  保留 {len(simplices)} 三角形", flush=True)

    # 光栅化
    print(f"  光栅化...", flush=True)
    img = np.full((h, w, 3), bg_color, dtype=np.uint8)
    colors_u8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

    # 批量处理三角形
    batch = 50000
    for start in range(0, len(simplices), batch):
        end = min(start + batch, len(simplices))
        for s in simplices[start:end]:
            i0, i1, i2 = s

            # 三角形顶点像素坐标
            pts_tri = np.array([
                [px[i0], py[i0]],
                [px[i1], py[i1]],
                [px[i2], py[i2]]
            ], dtype=np.int32)

            # 三角形平均颜色
            avg_color = (colors_u8[i0].astype(int) +
                        colors_u8[i1].astype(int) +
                        colors_u8[i2].astype(int)) // 3

            cv2.fillConvexPoly(img, pts_tri, avg_color.tolist())

        if end % 200000 == 0 or end == len(simplices):
            print(f"    {end}/{len(simplices)}", flush=True)

    # 创建掩码（有三角形覆盖的区域）
    mask = np.any(img != np.array(bg_color, dtype=np.uint8), axis=2).astype(np.uint8) * 255

    # 双边滤波平滑颜色（保边，让相邻区域颜色统一）
    print(f"  双边滤波平滑...", flush=True)
    img = bilateral_smooth(img, mask, d=9, sigma_color=40, sigma_space=15, iterations=2)

    # 锐化边缘
    print(f"  边缘锐化...", flush=True)
    img = sharpen_edges(img, amount=0.4)

    # 恢复背景
    img[mask == 0] = list(bg_color)

    return img, (x_min, x_max, y_min, y_max)


def add_grid(img, bounds, resolution, spacing=1.0):
    x_min, x_max, y_min, y_max = bounds
    h, w = img.shape[:2]
    for x in np.arange(np.ceil(x_min / spacing) * spacing, x_max, spacing):
        px = int((x - x_min) / resolution)
        if 0 <= px < w:
            cv2.line(img, (px, 0), (px, h - 1), (60, 60, 60), 1)
    for y in np.arange(np.ceil(y_min / spacing) * spacing, y_max, spacing):
        py = int((y_max - y) / resolution)
        if 0 <= py < h:
            cv2.line(img, (0, py), (w - 1, py), (60, 60, 60), 1)
    return img


def add_scale(img, resolution, length=5.0):
    h, w = img.shape[:2]
    bar = int(length / resolution)
    x0, y0 = 30, h - 40
    cv2.rectangle(img, (x0, y0), (x0 + bar, y0 + 8), (255, 255, 255), -1)
    cv2.putText(img, f"{length:.0f}m", (x0, y0 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=None)
    parser.add_argument('--input-dir', '-i', default=CAPTURE_DIR)
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--resolution', type=float, default=0.01)
    parser.add_argument('--max-edge', type=float, default=0.15,
                        help='最大三角形边长(米)')
    parser.add_argument('--sat-boost', type=float, default=1.6, help='饱和度增强')
    parser.add_argument('--val-boost', type=float, default=1.4, help='亮度增强')
    parser.add_argument('--grid', type=float, default=1.0)
    parser.add_argument('--z-min', type=float, default=None)
    parser.add_argument('--z-max', type=float, default=2.0,
                        help='Z轴最大值(去天花板，默认2.0)')
    args = parser.parse_args()

    input_path = args.input or os.path.join(args.input_dir, "merged_full_colored.ply")
    output = args.output or os.path.join(args.input_dir, "bev_mesh.png")

    print(f"读取: {input_path}", flush=True)
    pcd = o3d.io.read_point_cloud(input_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    print(f"  {len(points)} 点", flush=True)

    # Z过滤
    if args.z_min is not None or args.z_max is not None:
        z = points[:, 2]
        mask = np.ones(len(z), dtype=bool)
        if args.z_min is not None: mask &= z >= args.z_min
        if args.z_max is not None: mask &= z <= args.z_max
        points, colors = points[mask], colors[mask]
        print(f"  Z过滤: {len(points)} 点", flush=True)

    # 增强颜色
    print(f"\n增强颜色 (饱和度x{args.sat_boost}, 亮度x{args.val_boost})...", flush=True)
    colors = enhance_colors_hsv(colors, args.sat_boost, args.val_boost)

    # 生成 mesh BEV
    print(f"\n生成 Mesh BEV...", flush=True)
    img, bounds = generate_mesh_bev(
        points, colors,
        resolution=args.resolution,
        max_edge_len=args.max_edge)

    # 网格 + 比例尺
    if args.grid > 0:
        img = add_grid(img, bounds, args.resolution, args.grid)
    img = add_scale(img, args.resolution)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output, img_bgr)
    print(f"\n✓ {output} ({img.shape[1]}x{img.shape[0]})", flush=True)

    # 无网格版
    out_clean = output.rsplit('.', 1)[0] + '_clean.png'
    img2, _ = generate_mesh_bev(points, colors, args.resolution, args.max_edge)
    cv2.imwrite(out_clean, cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
    print(f"  无网格: {out_clean}", flush=True)


if __name__ == '__main__':
    main()
