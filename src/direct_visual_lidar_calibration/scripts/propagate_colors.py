#!/usr/bin/env python3
"""
propagate_colors.py — 将图像颜色传播到未着色的点

1. 区分有图像颜色的点（低饱和度）和纯Turbo的点（高饱和度）
2. 用 KNN 找每个未着色点最近的 K 个有色邻居
3. 按距离加权平均颜色，赋给未着色点
4. 输出全部点都有图像颜色的点云
"""
import os, sys, json, copy, glob, argparse
import numpy as np
import open3d as o3d

CAPTURE_DIR = "/data/projects/radar/direct_visual_lidar_calibration_ws/colored_captures"


def compute_saturation(colors):
    maxc = colors.max(axis=1)
    minc = colors.min(axis=1)
    diff = maxc - minc
    return np.where(maxc > 1e-6, diff / maxc, 0)


def propagate_knn(points, colors, colored_mask, k=8, max_dist=0.5):
    """
    用 KNN 将有色点的颜色传播到无色点
    """
    colored_idx = np.where(colored_mask)[0]
    uncolored_idx = np.where(~colored_mask)[0]

    if len(colored_idx) == 0 or len(uncolored_idx) == 0:
        return colors

    print(f"  构建 KD-Tree ({len(colored_idx)} 有色点)...", flush=True)

    # 用有色点建 KD-Tree
    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = o3d.utility.Vector3dVector(points[colored_idx])
    tree = o3d.geometry.KDTreeFlann(colored_pcd)

    colored_colors = colors[colored_idx]
    new_colors = colors.copy()

    print(f"  传播颜色到 {len(uncolored_idx)} 个无色点 (K={k})...", flush=True)

    batch_size = 50000
    for batch_start in range(0, len(uncolored_idx), batch_size):
        batch_end = min(batch_start + batch_size, len(uncolored_idx))
        batch = uncolored_idx[batch_start:batch_end]

        for idx in batch:
            pt = points[idx]
            [count, nn_idx, nn_dist] = tree.search_knn_vector_3d(pt, k)

            if count == 0:
                continue

            nn_idx = np.array(nn_idx[:count])
            nn_dist = np.sqrt(np.array(nn_dist[:count]))  # o3d 返回的是距离平方

            # 距离过滤
            valid = nn_dist < max_dist
            if not valid.any():
                # 用最近的一个
                nearest = nn_idx[0]
                new_colors[idx] = colored_colors[nearest]
                continue

            nn_idx = nn_idx[valid]
            nn_dist = nn_dist[valid]

            # 距离加权（反距离权重）
            weights = 1.0 / (nn_dist + 1e-6)
            weights /= weights.sum()

            blended = (colored_colors[nn_idx] * weights[:, np.newaxis]).sum(axis=0)
            new_colors[idx] = blended

        if batch_end % 100000 == 0 or batch_end == len(uncolored_idx):
            print(f"    {batch_end}/{len(uncolored_idx)}", flush=True)

    return new_colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-i', default=CAPTURE_DIR)
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--sat-threshold', type=float, default=0.35)
    parser.add_argument('--knn', type=int, default=8, help='KNN 邻居数')
    parser.add_argument('--max-dist', type=float, default=0.5, help='最大传播距离')
    parser.add_argument('--final-voxel', type=float, default=0.02)
    args = parser.parse_args()

    output = args.output or os.path.join(args.input_dir, "merged_full_colored.ply")

    state_file = os.path.join(args.input_dir, "merge_state.json")
    if not os.path.exists(state_file):
        print(f"需要 {state_file}"); return

    with open(state_file) as f:
        transforms = json.load(f)

    print(f"帧数: {len(transforms)}", flush=True)
    print(f"饱和度阈值: {args.sat_threshold}, KNN: {args.knn}, 最大距离: {args.max_dist}m", flush=True)

    # 先合并所有点（带原始颜色）
    print(f"\n合并所有帧...", flush=True)
    all_points = []
    all_colors = []

    for name, T_list in transforms.items():
        path = os.path.join(args.input_dir, name)
        if not os.path.exists(path):
            continue
        pcd = o3d.io.read_point_cloud(path)
        T = np.array(T_list)
        pcd.transform(T)
        all_points.append(np.asarray(pcd.points))
        all_colors.append(np.asarray(pcd.colors))
        print(f"  {name}: {len(pcd.points)} 点", flush=True)

    points = np.vstack(all_points)
    colors = np.vstack(all_colors)
    print(f"\n总点数: {len(points)}", flush=True)

    # 先降采样减少计算量
    print(f"降采样...", flush=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.voxel_down_sample(args.final_voxel)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    print(f"  降采样后: {len(points)} 点", flush=True)

    # 区分有色/无色
    sat = compute_saturation(colors)
    colored_mask = sat < args.sat_threshold
    n_colored = colored_mask.sum()
    n_uncolored = (~colored_mask).sum()
    print(f"\n有图像颜色: {n_colored} ({100*n_colored/len(points):.0f}%)", flush=True)
    print(f"纯Turbo: {n_uncolored} ({100*n_uncolored/len(points):.0f}%)", flush=True)

    # KNN 颜色传播
    print(f"\nKNN 颜色传播...", flush=True)
    new_colors = propagate_knn(points, colors, colored_mask, k=args.knn, max_dist=args.max_dist)

    # 构建结果
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(points)
    result.colors = o3d.utility.Vector3dVector(new_colors)

    # 去离群
    result, _ = result.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"去离群: {len(result.points)} 点", flush=True)

    o3d.io.write_point_cloud(output, result)
    print(f"\n✓ {output} ({len(result.points)} 点)", flush=True)


if __name__ == '__main__':
    main()
