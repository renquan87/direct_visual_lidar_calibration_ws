#!/usr/bin/env python3
"""
colorize_planes.py — 按平面分割着色点云
类似 Direct Visual LiDAR Calibration 的风格：
每个平面一个鲜明的颜色，非平面点用 intensity/高度着色

用法:
  python3 colorize_planes.py
  python3 colorize_planes.py --input merged.ply --output colored.ply
"""

import os, sys, argparse
import numpy as np

os.environ['PYTHONUNBUFFERED'] = '1'

import open3d as o3d

CAPTURE_DIR = "/data/projects/radar/direct_visual_lidar_calibration_ws/colored_captures"

# 鲜明的调色板（类似 DVL / matplotlib tab20）
BRIGHT_COLORS = [
    [0.12, 0.47, 0.71],  # 蓝
    [1.00, 0.50, 0.05],  # 橙
    [0.17, 0.63, 0.17],  # 绿
    [0.84, 0.15, 0.16],  # 红
    [0.58, 0.40, 0.74],  # 紫
    [0.55, 0.34, 0.29],  # 棕
    [0.89, 0.47, 0.76],  # 粉
    [0.50, 0.50, 0.50],  # 灰
    [0.74, 0.74, 0.13],  # 黄绿
    [0.09, 0.75, 0.81],  # 青
    [0.40, 0.76, 0.65],  # 薄荷
    [0.99, 0.75, 0.44],  # 浅橙
    [0.62, 0.85, 0.90],  # 浅蓝
    [0.96, 0.61, 0.58],  # 浅红
    [0.78, 0.92, 0.55],  # 浅绿
    [0.85, 0.73, 0.93],  # 浅紫
]


def turbo_colormap(t):
    """Turbo colormap (类似 DVL 用的 glk::COLORMAP::TURBO)"""
    t = np.clip(t, 0, 1)
    r = np.clip(0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943)))), 0, 1)
    g = np.clip(0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604)))), 0, 1)
    b = np.clip(0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973)))), 0, 1)
    return np.stack([r, g, b], axis=-1)


def detect_planes_ransac(pcd, distance_threshold=0.05, min_points=500, max_planes=15):
    """用 RANSAC 检测多个平面"""
    planes = []
    remaining = pcd
    
    print(f"  RANSAC 平面检测 (阈值={distance_threshold}m, 最少={min_points}点)...", flush=True)
    
    for i in range(max_planes):
        if len(remaining.points) < min_points:
            break
            
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000)
        
        if len(inliers) < min_points:
            break
        
        plane_pcd = remaining.select_by_index(inliers)
        remaining = remaining.select_by_index(inliers, invert=True)
        
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        planes.append({
            'pcd': plane_pcd,
            'model': plane_model,
            'normal': normal,
            'count': len(inliers)
        })
        
        print(f"    平面 {i}: {len(inliers)} 点, "
              f"法向=[{a:.2f},{b:.2f},{c:.2f}]", flush=True)
    
    print(f"  检测到 {len(planes)} 个平面, 剩余 {len(remaining.points)} 点", flush=True)
    return planes, remaining


def colorize_by_height(points, z_min=None, z_max=None):
    """按高度用 Turbo colormap 着色"""
    pts = np.asarray(points.points)
    z = pts[:, 2]
    if z_min is None: z_min = np.percentile(z, 2)
    if z_max is None: z_max = np.percentile(z, 98)
    
    t = (z - z_min) / max(z_max - z_min, 0.01)
    colors = turbo_colormap(t)
    return colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=None)
    parser.add_argument('--input-dir', '-i', default=CAPTURE_DIR)
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--distance', type=float, default=0.08, help='RANSAC 平面距离阈值')
    parser.add_argument('--min-points', type=int, default=1000, help='最小平面点数')
    parser.add_argument('--max-planes', type=int, default=20, help='最大平面数')
    parser.add_argument('--mode', choices=['planes', 'turbo', 'both'], default='both',
                        help='planes=仅平面着色, turbo=仅高度着色, both=平面+高度')
    parser.add_argument('--brightness', type=float, default=1.0, help='亮度系数')
    args = parser.parse_args()

    input_path = args.input or os.path.join(args.input_dir, "merged.ply")
    output = args.output or os.path.join(args.input_dir, "merged_colorized.ply")

    print(f"读取: {input_path}", flush=True)
    pcd = o3d.io.read_point_cloud(input_path)
    print(f"  {len(pcd.points)} 点", flush=True)

    if args.mode == 'turbo':
        # 纯 Turbo 高度着色
        print(f"\nTurbo 高度着色...", flush=True)
        colors = colorize_by_height(pcd)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    elif args.mode in ('planes', 'both'):
        # 平面检测 + 着色
        print(f"\n平面检测...", flush=True)
        
        # 先计算法线
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
        
        planes, remaining = detect_planes_ransac(
            pcd, 
            distance_threshold=args.distance,
            min_points=args.min_points,
            max_planes=args.max_planes)
        
        # 给每个平面分配颜色
        all_points = []
        all_colors = []
        
        for i, plane in enumerate(planes):
            color = BRIGHT_COLORS[i % len(BRIGHT_COLORS)]
            # 稍微加点变化让同一平面内有层次感
            pts = np.asarray(plane['pcd'].points)
            n = len(pts)
            
            # 基础颜色 + 轻微随机变化
            base_color = np.array(color)
            noise = np.random.normal(0, 0.03, (n, 3))
            colors = np.clip(base_color + noise, 0, 1)
            
            all_points.append(pts)
            all_colors.append(colors)
        
        # 非平面点
        if len(remaining.points) > 0:
            if args.mode == 'both':
                # 用 Turbo 高度着色
                rem_colors = colorize_by_height(remaining)
            else:
                # 用灰色
                rem_colors = np.full((len(remaining.points), 3), 0.6)
            
            all_points.append(np.asarray(remaining.points))
            all_colors.append(rem_colors)
        
        # 合并
        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    # 亮度调整
    if args.brightness != 1.0:
        colors = np.asarray(pcd.colors)
        colors = np.clip(colors * args.brightness, 0, 1)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.io.write_point_cloud(output, pcd)
    print(f"\n✓ {output} ({len(pcd.points)} 点)", flush=True)
    print(f"\n查看: python3 -c \"import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('{output}')])\"", flush=True)


if __name__ == '__main__':
    main()
