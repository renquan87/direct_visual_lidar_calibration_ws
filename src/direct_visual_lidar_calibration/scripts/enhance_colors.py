#!/usr/bin/env python3
"""
enhance_colors.py — 优化合并点云的颜色
- 重叠区域颜色混合（消除接缝）
- 亮度/对比度均衡
- 饱和度增强
- 去除暗淡/灰色噪点
"""
import os, sys, json, copy, glob, argparse
import numpy as np

import open3d as o3d

CAPTURE_DIR = "/data/projects/radar/direct_visual_lidar_calibration_ws/colored_captures"


def rgb_to_hsv(rgb):
    """批量 RGB→HSV, rgb shape (N,3) 范围 [0,1]"""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    diff = maxc - minc

    h = np.zeros_like(maxc)
    mask = diff > 1e-6

    # R is max
    m = mask & (maxc == r)
    h[m] = (60 * ((g[m] - b[m]) / diff[m]) + 360) % 360
    # G is max
    m = mask & (maxc == g)
    h[m] = (60 * ((b[m] - r[m]) / diff[m]) + 120) % 360
    # B is max
    m = mask & (maxc == b)
    h[m] = (60 * ((r[m] - g[m]) / diff[m]) + 240) % 360

    s = np.where(maxc > 1e-6, diff / maxc, 0)
    v = maxc
    return np.stack([h, s, v], axis=1)


def hsv_to_rgb(hsv):
    """批量 HSV→RGB"""
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    h = h / 60.0
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    rgb = np.zeros((len(h), 3))
    for idx, (r, g, b) in enumerate([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]):
        mask = (i == idx)
        rgb[mask, 0] = r[mask]
        rgb[mask, 1] = g[mask]
        rgb[mask, 2] = b[mask]
    return np.clip(rgb, 0, 1)


def enhance_point_cloud(pcd, saturation_boost=1.3, brightness_target=0.55,
                        contrast=1.2, gamma=0.9, remove_dark=True):
    """增强点云颜色"""
    colors = np.asarray(pcd.colors).copy()
    n = len(colors)
    print(f"  原始点数: {n}", flush=True)

    # 1. 去除过暗的点（通常是噪点或无效区域）
    if remove_dark:
        brightness = colors.mean(axis=1)
        bright_mask = brightness > 0.05
        dark_count = n - bright_mask.sum()
        if dark_count > 0:
            print(f"  去除暗点: {dark_count}", flush=True)
            pcd = pcd.select_by_index(np.where(bright_mask)[0])
            colors = np.asarray(pcd.colors).copy()

    # 2. 转 HSV
    hsv = rgb_to_hsv(colors)

    # 3. 饱和度增强
    hsv[:, 1] = np.clip(hsv[:, 1] * saturation_boost, 0, 1)
    print(f"  饱和度增强: x{saturation_boost}", flush=True)

    # 4. 亮度均衡 — 将平均亮度调整到目标值
    mean_v = hsv[:, 2].mean()
    if mean_v > 0.01:
        scale = brightness_target / mean_v
        hsv[:, 2] = np.clip(hsv[:, 2] * scale, 0, 1)
        print(f"  亮度调整: {mean_v:.3f} → {brightness_target:.3f} (x{scale:.2f})", flush=True)

    # 5. 转回 RGB
    colors = hsv_to_rgb(hsv)

    # 6. 对比度增强
    mean_color = colors.mean()
    colors = np.clip((colors - mean_color) * contrast + mean_color, 0, 1)
    print(f"  对比度增强: x{contrast}", flush=True)

    # 7. Gamma 校正
    colors = np.power(colors, gamma)
    print(f"  Gamma: {gamma}", flush=True)

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def blend_overlap_colors(pcds_with_transforms, voxel=0.03):
    """
    在重叠区域混合颜色，消除接缝
    """
    print(f"\n颜色混合（消除接缝）...", flush=True)

    # 合并所有点，记录来源
    all_points = []
    all_colors = []
    all_sources = []

    for i, (pcd, T) in enumerate(pcds_with_transforms):
        p = copy.deepcopy(pcd)
        p.transform(T)
        pts = np.asarray(p.points)
        cols = np.asarray(p.colors)
        all_points.append(pts)
        all_colors.append(cols)
        all_sources.append(np.full(len(pts), i))

    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    all_sources = np.concatenate(all_sources)

    # 体素化，在每个体素内混合颜色
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(all_points)
    merged.colors = o3d.utility.Vector3dVector(all_colors)

    # 用体素降采样自动混合（取平均）
    merged = merged.voxel_down_sample(voxel)
    print(f"  混合后: {len(merged.points)} 点", flush=True)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=None, help='输入 merged.ply')
    parser.add_argument('--input-dir', '-i', default=CAPTURE_DIR)
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--saturation', type=float, default=1.3)
    parser.add_argument('--brightness', type=float, default=0.55)
    parser.add_argument('--contrast', type=float, default=1.2)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--blend', action='store_true', help='从原始帧重建并混合颜色')
    parser.add_argument('--blend-voxel', type=float, default=0.02)
    args = parser.parse_args()

    input_path = args.input or os.path.join(args.input_dir, "merged.ply")
    output = args.output or os.path.join(args.input_dir, "merged_enhanced.ply")

    if args.blend:
        # 从原始帧重建，混合重叠区域颜色
        state_file = os.path.join(args.input_dir, "merge_state.json")
        if not os.path.exists(state_file):
            print(f"需要 {state_file}"); return
        with open(state_file) as f:
            transforms = json.load(f)

        pcds_with_T = []
        for name, T_list in transforms.items():
            path = os.path.join(args.input_dir, name)
            if os.path.exists(path):
                pcd = o3d.io.read_point_cloud(path)
                T = np.array(T_list)
                pcds_with_T.append((pcd, T))
                print(f"  {name}: {len(pcd.points)} 点", flush=True)

        pcd = blend_overlap_colors(pcds_with_T, args.blend_voxel)
    else:
        print(f"读取: {input_path}", flush=True)
        pcd = o3d.io.read_point_cloud(input_path)

    print(f"\n颜色增强...", flush=True)
    pcd = enhance_point_cloud(
        pcd,
        saturation_boost=args.saturation,
        brightness_target=args.brightness,
        contrast=args.contrast,
        gamma=args.gamma)

    # 最终去离群
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"  最终: {len(pcd.points)} 点", flush=True)

    o3d.io.write_point_cloud(output, pcd)
    print(f"\n✓ {output} ({len(pcd.points)} 点)", flush=True)


if __name__ == '__main__':
    main()
