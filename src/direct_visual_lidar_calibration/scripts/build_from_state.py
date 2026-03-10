#!/usr/bin/env python3
"""
从 merge_state.json 构建合并点云
"""
import os, sys, glob, copy, json, argparse
import numpy as np
import open3d as o3d

CAPTURE_DIR = "/data/projects/radar/direct_visual_lidar_calibration_ws/colored_captures"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-i', default=CAPTURE_DIR)
    parser.add_argument('--state', '-s', default=None)
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--final-voxel', type=float, default=0.02)
    args = parser.parse_args()

    state_file = args.state or os.path.join(args.input_dir, "merge_state.json")
    output = args.output or os.path.join(args.input_dir, "merged.ply")

    with open(state_file) as f:
        transforms = json.load(f)

    print(f"状态文件: {state_file}", flush=True)
    print(f"已对齐帧数: {len(transforms)}", flush=True)

    merged = o3d.geometry.PointCloud()
    for name, T_list in transforms.items():
        path = os.path.join(args.input_dir, name)
        if not os.path.exists(path):
            print(f"  跳过 {name} (文件不存在)", flush=True)
            continue
        pcd = o3d.io.read_point_cloud(path)
        T = np.array(T_list)
        pcd.transform(T)
        merged += pcd
        print(f"  {name}: {len(pcd.points)} 点", flush=True)

    print(f"\n总点数: {len(merged.points)}", flush=True)

    if args.final_voxel > 0:
        merged = merged.voxel_down_sample(args.final_voxel)
        print(f"降采样: {len(merged.points)}", flush=True)

    merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"去离群: {len(merged.points)}", flush=True)

    o3d.io.write_point_cloud(output, merged)
    print(f"\n✓ {output} ({len(merged.points)} 点)", flush=True)

if __name__ == '__main__':
    main()
