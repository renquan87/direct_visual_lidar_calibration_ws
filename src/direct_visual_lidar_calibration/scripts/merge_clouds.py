#!/usr/bin/env python3
"""
merge_clouds.py — 多帧彩色点云配准合并（缓存优化版）
"""

import os, sys
os.environ['PYTHONUNBUFFERED'] = '1'

import glob, copy, time, json, argparse
import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("需要 open3d: pip install open3d"); exit(1)

CAPTURE_DIR = "/data/projects/radar/direct_visual_lidar_calibration_ws/colored_captures"
LOG_FILE = os.path.join(CAPTURE_DIR, "merge_log.txt")

_log_fh = None
def log(msg):
    global _log_fh
    if _log_fh is None:
        _log_fh = open(LOG_FILE, 'w')
    text = str(msg)
    sys.stderr.write(text + '\n')
    sys.stderr.flush()
    _log_fh.write(text + '\n')
    _log_fh.flush()

def compute_fpfh(pcd, voxel_size):
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=20))
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=50))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-i', default=CAPTURE_DIR)
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--voxel', type=float, default=0.3, help='配准体素')
    parser.add_argument('--final-voxel', type=float, default=0.02)
    parser.add_argument('--max-ref', type=int, default=30000)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "capture_*.ply")))
    if not files:
        log("没有 capture_*.ply"); return
    output = args.output or os.path.join(args.input_dir, "merged.ply")
    V = args.voxel

    log(f"\n{len(files)} 个点云, voxel={V}m")
    log("读取...")
    pcds = []
    for f in files:
        pcd = o3d.io.read_point_cloud(f)
        if len(pcd.points) > 100:
            pcds.append((os.path.basename(f), pcd))
            log(f"  {os.path.basename(f)}: {len(pcd.points)}")
    if len(pcds) < 2:
        if pcds: o3d.io.write_point_cloud(output, pcds[0][1])
        return

    transforms = [None]*len(pcds)
    transforms[0] = np.eye(4)
    registered = [0]
    failed = []

    # 初始化参考（降采样 + 缓存 FPFH）
    ref_down = pcds[0][1].voxel_down_sample(V)
    log(f"\n初始参考: {len(ref_down.points)} 点")
    log("计算参考 FPFH...")
    ref_fpfh = compute_fpfh(ref_down, V)
    ref_dirty = False  # 参考是否需要重算 FPFH
    log("就绪")

    t_total = time.time()

    for i in range(1, len(pcds)):
        name = pcds[i][0]
        log(f"\n[{i}/{len(pcds)-1}] {name}")

        # source 降采样 + FPFH
        t0 = time.time()
        src_down = pcds[i][1].voxel_down_sample(V)
        log(f"  src: {len(src_down.points)} 点, ref: {len(ref_down.points)} 点")
        src_fpfh = compute_fpfh(src_down, V)
        t1 = time.time()
        log(f"  src FPFH: {t1-t0:.1f}s")

        # 如果参考被更新了，重算 FPFH
        if ref_dirty:
            log(f"  更新 ref FPFH...")
            ref_fpfh = compute_fpfh(ref_down, V)
            ref_dirty = False
            log(f"  ref FPFH: {time.time()-t1:.1f}s")
            t1 = time.time()

        # FGR
        log(f"  FGR...")
        dist = V * 0.5
        fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            src_down, ref_down, src_fpfh, ref_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=dist))
        t2 = time.time()
        log(f"  FGR: {t2-t1:.1f}s, fit={fgr.fitness:.3f}")

        # ICP
        log(f"  ICP...")
        src_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=V*2, max_nn=20))
        ref_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=V*2, max_nn=20))
        icp = o3d.pipelines.registration.registration_icp(
            src_down, ref_down, dist, fgr.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
        t3 = time.time()
        log(f"  ICP: {t3-t2:.1f}s, fit={icp.fitness:.3f}, rmse={icp.inlier_rmse:.4f}")

        fit = icp.fitness
        T = icp.transformation

        if fit >= 0.02:
            transforms[i] = T
            registered.append(i)
            # 更新参考
            t_src = copy.deepcopy(src_down)
            t_src.transform(T)
            ref_down += t_src
            if len(ref_down.points) > args.max_ref:
                ref_down = ref_down.voxel_down_sample(V)
            ref_dirty = True
            log(f"  ✓ ({time.time()-t0:.1f}s 总)")
        else:
            failed.append(i)
            log(f"  ✗ 跳过")

    elapsed = time.time() - t_total
    log(f"\n{'='*50}")
    log(f"配准: {len(registered)}/{len(pcds)} 帧, {elapsed:.0f}s")
    if failed:
        log(f"失败: {[pcds[i][0] for i in failed]}")

    # 合并原始点云
    log(f"\n合并...")
    merged = o3d.geometry.PointCloud()
    for idx, i in enumerate(registered):
        p = copy.deepcopy(pcds[i][1])
        p.transform(transforms[i])
        merged += p
        if (idx+1) % 5 == 0:
            log(f"  {idx+1}/{len(registered)}")
    log(f"  {len(merged.points)} 点")

    if args.final_voxel > 0:
        merged = merged.voxel_down_sample(args.final_voxel)
        log(f"  降采样: {len(merged.points)}")
    merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    log(f"  去离群: {len(merged.points)}")

    o3d.io.write_point_cloud(output, merged)
    log(f"\n✓ {output} ({len(merged.points)} 点)")

    tf_path = output.rsplit('.', 1)[0] + '_transforms.json'
    with open(tf_path, 'w') as f:
        json.dump({pcds[i][0]: transforms[i].tolist() for i in registered}, f, indent=2)
    log(f"  变换: {tf_path}")

if __name__ == '__main__':
    main()
