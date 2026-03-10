#!/usr/bin/env python3
"""
filter_colored.py — 从合并点云中去除未着色（纯Turbo）的点，只保留有图像映射的点

原理：capture_colored.py 中：
  - 有图像的点 = 0.7*灰度 + 0.3*turbo → 低饱和度（偏灰）
  - 无图像的点 = 纯turbo → 高饱和度（彩虹色）

通过饱和度阈值过滤：保留低饱和度的点（有图像颜色的）
"""
import os, sys, json, copy, glob, argparse
import numpy as np
import open3d as o3d

CAPTURE_DIR = "/data/projects/radar/direct_visual_lidar_calibration_ws/colored_captures"


def compute_saturation(colors):
    """计算每个点的 HSV 饱和度"""
    maxc = colors.max(axis=1)
    minc = colors.min(axis=1)
    diff = maxc - minc
    sat = np.where(maxc > 1e-6, diff / maxc, 0)
    return sat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-i', default=CAPTURE_DIR)
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--sat-threshold', type=float, default=0.35,
                        help='饱和度阈值：低于此值=有图像颜色，高于=纯Turbo')
    parser.add_argument('--final-voxel', type=float, default=0.02)
    parser.add_argument('--show-stats', action='store_true', default=True)
    args = parser.parse_args()

    output = args.output or os.path.join(args.input_dir, "merged_camera_only.ply")

    # 从原始帧重建，逐帧过滤
    state_file = os.path.join(args.input_dir, "merge_state.json")
    if not os.path.exists(state_file):
        print(f"需要 {state_file}"); return

    with open(state_file) as f:
        transforms = json.load(f)

    print(f"状态文件: {state_file}", flush=True)
    print(f"已对齐帧数: {len(transforms)}", flush=True)
    print(f"饱和度阈值: {args.sat_threshold}", flush=True)

    merged = o3d.geometry.PointCloud()
    total_original = 0
    total_kept = 0

    for name, T_list in transforms.items():
        path = os.path.join(args.input_dir, name)
        if not os.path.exists(path):
            print(f"  跳过 {name} (文件不存在)", flush=True)
            continue

        pcd = o3d.io.read_point_cloud(path)
        colors = np.asarray(pcd.colors)
        n_orig = len(colors)
        total_original += n_orig

        # 计算饱和度
        sat = compute_saturation(colors)

        # 保留低饱和度的点（有图像颜色的）
        keep_mask = sat < args.sat_threshold
        n_kept = keep_mask.sum()
        total_kept += n_kept

        if args.show_stats:
            print(f"  {name}: {n_orig} → {n_kept} 点 "
                  f"({100*n_kept/max(n_orig,1):.0f}% 保留, "
                  f"平均饱和度: 保留={sat[keep_mask].mean():.3f} 去除={sat[~keep_mask].mean():.3f})",
                  flush=True)

        if n_kept == 0:
            continue

        # 过滤
        filtered = pcd.select_by_index(np.where(keep_mask)[0])

        # 应用变换
        T = np.array(T_list)
        filtered.transform(T)
        merged += filtered

    print(f"\n总计: {total_original} → {total_kept} 点 ({100*total_kept/max(total_original,1):.0f}%)", flush=True)

    # 降采样
    if args.final_voxel > 0:
        merged = merged.voxel_down_sample(args.final_voxel)
        print(f"降采样: {len(merged.points)} 点", flush=True)

    # 去离群
    merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"去离群: {len(merged.points)} 点", flush=True)

    o3d.io.write_point_cloud(output, merged)
    print(f"\n✓ {output} ({len(merged.points)} 点)", flush=True)


if __name__ == '__main__':
    main()
