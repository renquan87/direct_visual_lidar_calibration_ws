#!/usr/bin/env python3
"""
cloud_to_bev.py — 从彩色点云生成俯视图

读取合并后的彩色 PLY，可选去天花板，生成俯视图 PNG。
同时保存处理后的完整彩色点云。

用法：
  # 默认去天花板
  python3 scripts/cloud_to_bev.py --input .../merged.ply --output .../map/bev.png

  # 不去天花板
  python3 scripts/cloud_to_bev.py --input .../merged.ply --no-remove-ceiling

  # 自定义分辨率
  python3 scripts/cloud_to_bev.py --input .../merged.ply --resolution 0.005
"""

import os
import json
import argparse
import numpy as np
import cv2

try:
    import open3d as o3d
except ImportError:
    print("需要 open3d: pip install open3d")
    exit(1)


def remove_ceiling(points, colors, threshold=0.15):
    z_range = points[:, 2].max() - points[:, 2].min()
    z_70 = points[:, 2].min() + z_range * 0.7
    upper = points[:, 2] > z_70
    if upper.sum() < 100:
        return points, colors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[upper])
    try:
        model, inliers = pcd.segment_plane(distance_threshold=threshold, ransac_n=3, num_iterations=1000)
    except Exception:
        return points, colors

    if abs(model[2]) < 0.8 or len(inliers) < upper.sum() * 0.15:
        return points, colors

    upper_idx = np.where(upper)[0]
    ceiling_z = points[upper_idx[inliers], 2].mean()
    keep = points[:, 2] < (ceiling_z - threshold * 2)
    removed = (~keep).sum()
    print(f"  去除天花板: Z={ceiling_z:.2f}m, 去除 {removed} 点")
    return points[keep], colors[keep]


def generate_bev(points, colors_rgb, resolution=0.01, bg=(30, 30, 30), margin=0.3, grid=True):
    if len(points) == 0:
        return None, None

    xmin, xmax = points[:, 0].min() - margin, points[:, 0].max() + margin
    ymin, ymax = points[:, 1].min() - margin, points[:, 1].max() + margin
    W = int((xmax - xmin) / resolution)
    H = int((ymax - ymin) / resolution)
    if W <= 0 or H <= 0 or W > 20000 or H > 20000:
        return None, None

    bev = np.full((H, W, 3), bg, dtype=np.uint8)
    zbuf = np.full((H, W), -np.inf)

    px = ((points[:, 0] - xmin) / resolution).astype(int)
    py = (H - 1 - ((points[:, 1] - ymin) / resolution).astype(int))
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px, py, z = px[ok], py[ok], points[ok, 2]
    c = (colors_rgb[ok] * 255).astype(np.uint8)

    for i in np.argsort(z):
        if z[i] > zbuf[py[i], px[i]]:
            zbuf[py[i], px[i]] = z[i]
            bev[py[i], px[i]] = c[i][::-1]  # RGB -> BGR

    # 填充小间隙
    mask = (zbuf > -np.inf).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=2)
    filled = cv2.medianBlur(bev, 3)
    gap = (dilated > 0) & (mask == 0)
    bev[gap] = filled[gap]
    bev[dilated == 0] = bg

    # 裁剪
    rows = np.any(dilated > 0, axis=1)
    cols = np.any(dilated > 0, axis=0)
    if np.any(rows) and np.any(cols):
        mp = int(margin / resolution)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        r0, r1 = max(0, r0 - mp), min(H - 1, r1 + mp)
        c0, c1 = max(0, c0 - mp), min(W - 1, c1 + mp)
        bev = bev[r0:r1+1, c0:c1+1].copy()
        xmin += c0 * resolution
        xmax = xmin + (c1 - c0 + 1) * resolution
        ymax = ymin + (H - r0) * resolution
        ymin = ymin + (H - r1 - 1) * resolution
        H, W = bev.shape[:2]

    fw, fh = xmax - xmin, ymax - ymin

    if grid:
        sz = max(fw, fh)
        step = 0.5 if sz <= 5 else (1.0 if sz <= 15 else (2.0 if sz <= 30 else 5.0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = max(0.3, min(1.0, W / 2000))
        th = max(1, int(W / 1500))
        x = np.ceil(xmin / step) * step
        while x <= xmax:
            px_ = int((x - xmin) / resolution)
            if 0 <= px_ < W:
                cv2.line(bev, (px_, 0), (px_, H - 1), (100, 100, 100), 1)
                cv2.putText(bev, f"{x:.1f}m", (px_ + 3, H - 10), font, fs, (200, 200, 200), th)
            x += step
        y = np.ceil(ymin / step) * step
        while y <= ymax:
            py_ = H - 1 - int((y - ymin) / resolution)
            if 0 <= py_ < H:
                cv2.line(bev, (0, py_), (W - 1, py_), (100, 100, 100), 1)
                cv2.putText(bev, f"{y:.1f}m", (5, py_ - 5), font, fs, (200, 200, 200), th)
            y += step
        # 比例尺
        sl = int(step / resolution)
        m, by_ = 30, H - 30
        cv2.rectangle(bev, (m - 5, by_ - 25), (m + sl + 5, by_ + 10), (0, 0, 0), -1)
        cv2.line(bev, (m, by_), (m + sl, by_), (255, 255, 255), 2)
        cv2.putText(bev, f"{step:.1f}m", (m, by_ - 10), font, fs, (255, 255, 255), th)
        # 场地尺寸
        st = f"Field: {fw:.1f}m x {fh:.1f}m"
        tsz = cv2.getTextSize(st, font, fs, th)[0]
        tx = W - tsz[0] - 10
        cv2.rectangle(bev, (tx - 5, 0), (tx + tsz[0] + 5, 30), (0, 0, 0), -1)
        cv2.putText(bev, st, (tx, 22), font, fs, (255, 255, 255), th)

    meta = {
        'x_range': [float(xmin), float(xmax)],
        'y_range': [float(ymin), float(ymax)],
        'resolution': resolution,
        'image_size': [W, H],
        'field_width_m': float(fw),
        'field_height_m': float(fh),
        'num_points': len(points),
    }
    return bev, meta


def main():
    parser = argparse.ArgumentParser(description='从彩色点云生成俯视图')
    parser.add_argument('--input', '-i', required=True, help='输入 PLY 文件')
    parser.add_argument('--output', '-o', default=None, help='输出 BEV PNG')
    parser.add_argument('--resolution', '-r', type=float, default=0.01, help='BEV 分辨率 m/px')
    parser.add_argument('--no-remove-ceiling', action='store_true', help='不去天花板')
    parser.add_argument('--save-processed-ply', default=None, help='保存处理后的点云')
    parser.add_argument('--show', action='store_true', help='3D 预览')
    args = parser.parse_args()

    print(f"\n读取: {args.input}")
    pcd = o3d.io.read_point_cloud(args.input)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    print(f"  {len(points)} 点")

    if args.show:
        print("  3D 预览（关闭窗口继续）...")
        o3d.visualization.draw_geometries([pcd], window_name="原始点云", width=1280, height=720)

    # 去天花板
    if not args.no_remove_ceiling:
        points, colors = remove_ceiling(points, colors)
        print(f"  去天花板后: {len(points)} 点")

    # 保存处理后的点云
    ply_out = args.save_processed_ply
    if ply_out is None:
        base = os.path.splitext(args.input)[0]
        ply_out = base + "_processed.ply"
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(points)
    pcd_out.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(ply_out, pcd_out)
    print(f"  处理后点云: {ply_out}")

    # 生成 BEV
    output = args.output or os.path.join(os.path.dirname(args.input), "bev.png")
    print(f"\n生成俯视图 (分辨率: {args.resolution} m/px)...")
    bev, meta = generate_bev(points, colors, resolution=args.resolution)

    if bev is not None:
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        cv2.imwrite(output, bev)

        # 干净版（无网格）
        bev_clean, _ = generate_bev(points, colors, resolution=args.resolution, grid=False)
        clean_path = output.rsplit('.', 1)[0] + '_clean.png'
        if bev_clean is not None:
            cv2.imwrite(clean_path, bev_clean)

        # 元数据
        meta_path = output.rsplit('.', 1)[0] + '_meta.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"\n输出:")
        print(f"  带网格: {output}")
        print(f"  干净版: {clean_path}")
        print(f"  元数据: {meta_path}")
        print(f"  场地:   {meta['field_width_m']:.2f}m x {meta['field_height_m']:.2f}m")
        print(f"  图像:   {meta['image_size'][0]}x{meta['image_size'][1]}")
        print(f"  点数:   {meta['num_points']}")
    else:
        print("  生成失败")

    print("\n完成！")


if __name__ == '__main__':
    main()
