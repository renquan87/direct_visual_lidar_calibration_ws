#!/usr/bin/env python3
"""
online_bev.py — 在线多位置采集着色点云，生成俯视图

工作流程：
  1. 订阅雷达点云和相机图像话题
  2. 每次按 Enter，采集当前帧的点云+图像
  3. 用 T_lidar_camera 外参将点云投影到相机，混合着色
  4. 可以搬动雷达到不同位置，多次采集（解决遮挡问题）
  5. 按 q 结束采集，合并所有着色点云，生成俯视图

用法：
  python3 scripts/online_bev.py \
      --calib-json /home/rq/radar/calib_ws/preprocessed/calib.json \
      --pcl-topic /livox/lidar \
      --img-topic /image_raw \
      --output /data/projects/radar/hnurm_radar/map/online_bev.png

  # 每次采集累积多帧（更密集）
  python3 scripts/online_bev.py \
      --calib-json /home/rq/radar/calib_ws/preprocessed/calib.json \
      --accumulate 20

  # 纯 intensity（不需要相机）
  python3 scripts/online_bev.py \
      --calib-json /home/rq/radar/calib_ws/preprocessed/calib.json \
      --blend 0.0
"""

import os
import sys
import json
import time
import threading
import argparse
import numpy as np
import cv2

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge


# ─── 着色函数 ──────────────────────────────────────────────

def intensity_to_turbo(intensity):
    """TURBO 色表"""
    i_u8 = (np.clip(intensity, 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(i_u8.reshape(-1, 1), cv2.COLORMAP_TURBO)
    return colored.reshape(-1, 3)[:, ::-1].astype(np.float64) / 255.0


def histogram_equalize(intensity):
    """intensity 直方图均衡化"""
    n = len(intensity)
    if n == 0:
        return intensity
    indices = np.argsort(intensity)
    eq = np.zeros(n, dtype=np.float64)
    bins = 256
    for i, idx in enumerate(indices):
        eq[idx] = np.floor(bins * i / n) / bins
    return eq


def project_and_color(points, intensity, image_gray, T_lidar_camera, intrinsics, dist_coeffs, blend):
    """
    对一帧点云做完整着色：
    1. intensity 直方图均衡 → TURBO
    2. 投影到相机 → 混合
    返回 colors (N,3) RGB [0,1]
    """
    # 均衡化
    int_eq = histogram_equalize(intensity)
    turbo = intensity_to_turbo(int_eq)

    if image_gray is None or T_lidar_camera is None or blend <= 0:
        return turbo

    # 投影
    pts_h = np.hstack([points, np.ones((len(points), 1))])
    T_cam_lidar = np.linalg.inv(T_lidar_camera)
    pts_cam = (T_cam_lidar @ pts_h.T).T[:, :3]

    valid = pts_cam[:, 2] > 0.1
    fx, fy, cx, cy = intrinsics[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array(dist_coeffs, dtype=np.float64)

    pts_for_proj = pts_cam[valid].astype(np.float64)
    if len(pts_for_proj) == 0:
        return turbo

    img_pts, _ = cv2.projectPoints(pts_for_proj, np.zeros(3), np.zeros(3), K, dist)
    img_pts = img_pts.reshape(-1, 2).astype(int)

    h, w = image_gray.shape[:2]
    in_bounds = (img_pts[:, 0] >= 0) & (img_pts[:, 0] < w) & \
                (img_pts[:, 1] >= 0) & (img_pts[:, 1] < h)

    colors = turbo.copy()
    valid_idx = np.where(valid)[0]
    ok_idx = valid_idx[in_bounds]
    ok_px = img_pts[in_bounds]

    for j in range(len(ok_idx)):
        i = ok_idx[j]
        px, py = ok_px[j]
        g = image_gray[py, px] / 255.0
        colors[i] = np.array([g, g, g]) * blend + turbo[i] * (1.0 - blend)

    return colors


# ─── 天花板去除 ─────────────────────────────────────────────

def remove_ceiling(points, colors, threshold=0.15):
    if not HAS_O3D:
        return points, colors

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
    print(f"  去除天花板: Z={ceiling_z:.2f}m, 去除 {(~keep).sum()} 点")
    return points[keep], colors[keep]


# ─── BEV 生成 ──────────────────────────────────────────────

def generate_bev(points, colors_rgb, resolution=0.01, bg=(30, 30, 30), margin=0.3, grid=True):
    if len(points) == 0:
        return None, None

    xmin, xmax = points[:, 0].min() - margin, points[:, 0].max() + margin
    ymin, ymax = points[:, 1].min() - margin, points[:, 1].max() + margin
    W, H = int((xmax - xmin) / resolution), int((ymax - ymin) / resolution)
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
            bev[py[i], px[i]] = c[i][::-1]

    mask = (zbuf > -np.inf).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=2)
    filled = cv2.medianBlur(bev, 3)
    gap = (dilated > 0) & (mask == 0)
    bev[gap] = filled[gap]
    bev[dilated == 0] = bg

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
        sl = int(step / resolution)
        m, by_ = 30, H - 30
        cv2.rectangle(bev, (m - 5, by_ - 25), (m + sl + 5, by_ + 10), (0, 0, 0), -1)
        cv2.line(bev, (m, by_), (m + sl, by_), (255, 255, 255), 2)
        cv2.putText(bev, f"{step:.1f}m", (m, by_ - 10), font, fs, (255, 255, 255), th)
        st = f"Field: {fw:.1f}m x {fh:.1f}m"
        tsz = cv2.getTextSize(st, font, fs, th)[0]
        tx = W - tsz[0] - 10
        cv2.rectangle(bev, (tx - 5, 0), (tx + tsz[0] + 5, 30), (0, 0, 0), -1)
        cv2.putText(bev, st, (tx, 22), font, fs, (255, 255, 255), th)

    meta = {
        'x_range': [xmin, xmax], 'y_range': [ymin, ymax],
        'resolution': resolution, 'image_size': [W, H],
        'field_width_m': fw, 'field_height_m': fh,
        'num_points': len(points),
    }
    return bev, meta


# ─── ROS2 采集节点 ─────────────────────────────────────────

class Collector(Node):
    def __init__(self, pcl_topic, img_topic):
        super().__init__('online_bev_collector')
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_points = None
        self.latest_intensity = None
        self.latest_image = None

        if 'livox' in pcl_topic:
            pcl_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                 history=HistoryPolicy.KEEP_LAST, depth=10)
        else:
            pcl_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 history=HistoryPolicy.KEEP_LAST, depth=5)
        img_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=5)

        self.create_subscription(PointCloud2, pcl_topic, self._pcl_cb, pcl_qos)
        self.create_subscription(Image, img_topic, self._img_cb, img_qos)

    def _pcl_cb(self, msg):
        fields = [f.name for f in msg.fields]
        i_field = None
        for c in ['intensity', 'reflectivity']:
            if c in fields:
                i_field = c
                break

        if i_field:
            pts = list(pc2.read_points(msg, field_names=("x", "y", "z", i_field), skip_nans=True))
            arr = np.array(pts, dtype=[("x", np.float32), ("y", np.float32),
                                       ("z", np.float32), ("i", np.float32)])
            xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=-1).astype(np.float64)
            intensity = arr["i"].astype(np.float64)
        else:
            pts = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            arr = np.array(pts, dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)])
            xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=-1).astype(np.float64)
            intensity = np.zeros(len(xyz))

        with self.lock:
            self.latest_points = xyz
            self.latest_intensity = intensity

    def _img_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception:
            try:
                img = self.bridge.imgmsg_to_cv2(msg)
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception:
                return
        img = cv2.equalizeHist(img)
        with self.lock:
            self.latest_image = img

    def grab(self):
        with self.lock:
            p = self.latest_points.copy() if self.latest_points is not None else None
            i = self.latest_intensity.copy() if self.latest_intensity is not None else None
            img = self.latest_image.copy() if self.latest_image is not None else None
        return p, i, img


# ─── 主程序 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='在线多位置采集着色点云生成俯视图')
    parser.add_argument('--calib-json', required=True, help='calib.json 路径')
    parser.add_argument('--pcl-topic', default='/livox/lidar', help='点云话题')
    parser.add_argument('--img-topic', default='/image', help='图像话题')
    parser.add_argument('--output', '-o', default=None, help='输出路径')
    parser.add_argument('--blend', type=float, default=0.7, help='相机混合权重')
    parser.add_argument('--resolution', '-r', type=float, default=0.01, help='BEV 分辨率 m/px')
    parser.add_argument('--accumulate', type=int, default=10, help='每次采集累积帧数')
    parser.add_argument('--max-dist', type=float, default=15.0, help='最大距离')
    parser.add_argument('--min-dist', type=float, default=0.3, help='最小距离')
    parser.add_argument('--no-remove-ceiling', action='store_true')
    parser.add_argument('--show-3d', action='store_true', help='最后 3D 预览')
    parser.add_argument('--save-ply', default=None, help='保存合并的着色 PLY')
    parser.add_argument('--voxel-size', type=float, default=0.02, help='降采样体素大小')
    args = parser.parse_args()

    # 读取标定
    with open(args.calib_json) as f:
        calib = json.load(f)
    intrinsics = calib['camera']['intrinsics']
    dist_coeffs = calib['camera']['distortion_coeffs']

    T_lidar_camera = None
    results = calib.get('results', {})
    for key in ['T_lidar_camera', 'init_T_lidar_camera', 'init_T_lidar_camera_auto']:
        if key in results:
            v = results[key]
            from scipy.spatial.transform import Rotation
            R = Rotation.from_quat([v[3], v[4], v[5], v[6]]).as_matrix()
            T_lidar_camera = np.eye(4)
            T_lidar_camera[:3, :3] = R
            T_lidar_camera[:3, 3] = [v[0], v[1], v[2]]
            break

    # 启动 ROS2
    rclpy.init()
    node = Collector(args.pcl_topic, args.img_topic)
    spin_t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_t.start()

    # 存储所有采集
    captures = []  # list of (points, colors)

    print()
    print("=" * 60)
    print("  在线多位置采集工具")
    print("=" * 60)
    print(f"  点云: {args.pcl_topic}")
    print(f"  图像: {args.img_topic}")
    print(f"  混合: {args.blend}")
    print(f"  累积: {args.accumulate} 帧/次")
    print()
    print("  操作说明:")
    print("    Enter  — 在当前位置采集")
    print("    p      — 预览当前已采集的合并点云")
    print("    s      — 保存当前进度（不退出）")
    print("    q      — 结束采集，生成俯视图")
    print()
    print("  你可以随时搬动雷达到新位置，然后按 Enter 采集")
    print("  多个位置的点云会自动合并，解决遮挡问题")
    print("=" * 60)

    # 等待数据
    print("\n等待数据...")
    for _ in range(200):
        p, i, img = node.grab()
        if p is not None and len(p) > 100:
            print(f"  点云就绪: {len(p)} 点/帧")
            if img is not None:
                print(f"  图像就绪: {img.shape[1]}x{img.shape[0]}")
            else:
                print("  图像: 暂无（将仅用 intensity 着色）")
            break
        time.sleep(0.1)
    else:
        print("  超时，未收到数据。请检查话题名称。")
        node.destroy_node()
        rclpy.shutdown()
        return

    capture_idx = 0
    while True:
        total_pts = sum(len(c[0]) for c in captures)
        try:
            cmd = input(f"\n[已采集 {len(captures)} 次, {total_pts} 点] 操作 (Enter/p/s/q): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            cmd = 'q'

        if cmd == 'q':
            break

        elif cmd == 'p':
            # 预览
            if not captures:
                print("  还没有采集数据")
                continue
            if not HAS_O3D:
                print("  没有 open3d")
                continue
            all_pts = np.vstack([c[0] for c in captures])
            all_clr = np.vstack([c[1] for c in captures])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(all_pts)
            pcd.colors = o3d.utility.Vector3dVector(all_clr)
            print(f"  预览 {len(all_pts)} 点（关闭窗口继续）...")
            o3d.visualization.draw_geometries([pcd], window_name=f"已采集 {len(all_pts)} 点",
                                              width=1280, height=720)
            continue

        elif cmd == 's':
            # 中途保存
            if not captures:
                print("  没有数据")
                continue
            save_path = args.output or 'online_bev_progress.png'
            all_pts = np.vstack([c[0] for c in captures])
            all_clr = np.vstack([c[1] for c in captures])
            if not args.no_remove_ceiling:
                all_pts, all_clr = remove_ceiling(all_pts, all_clr)
            bev, meta = generate_bev(all_pts, all_clr, resolution=args.resolution)
            if bev is not None:
                cv2.imwrite(save_path, bev)
                print(f"  已保存进度: {save_path}")
            continue

        # Enter — 采集
        capture_idx += 1
        print(f"\n  === 采集 #{capture_idx} ===")

        # 累积多帧
        pt_frames = []
        int_frames = []
        image_for_blend = None

        for frame_i in range(args.accumulate):
            time.sleep(0.05)
            p, intensity, img = node.grab()
            if p is not None and len(p) > 50:
                pt_frames.append(p)
                int_frames.append(intensity)
                if img is not None:
                    image_for_blend = img
            if (frame_i + 1) % 5 == 0:
                print(f"    累积 {frame_i + 1}/{args.accumulate} 帧...")

        if not pt_frames:
            print("  没有收到点云")
            continue

        pts = np.vstack(pt_frames)
        ints = np.concatenate(int_frames)
        print(f"  累积 {len(pt_frames)} 帧: {len(pts)} 点")

        # 距离滤波
        dist = np.linalg.norm(pts, axis=1)
        mask = (dist >= args.min_dist) & (dist <= args.max_dist)
        pts, ints = pts[mask], ints[mask]
        print(f"  距离滤波后: {len(pts)} 点")

        # 降采样
        if HAS_O3D and args.voxel_size > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd_down = pcd.voxel_down_sample(args.voxel_size)
            # 用最近邻保留 intensity
            down_pts = np.asarray(pcd_down.points)
            tree = o3d.geometry.KDTreeFlann(pcd)
            new_ints = np.zeros(len(down_pts))
            for j in range(len(down_pts)):
                _, nn_idx, _ = tree.search_knn_vector_3d(down_pts[j], 1)
                new_ints[j] = ints[nn_idx[0]]
            pts, ints = down_pts, new_ints
            print(f"  降采样后: {len(pts)} 点")

        # 着色
        colors = project_and_color(pts, ints, image_for_blend, T_lidar_camera,
                                   intrinsics, dist_coeffs, args.blend)

        in_cam = 0
        if image_for_blend is not None and T_lidar_camera is not None and args.blend > 0:
            # 统计投影到相机的点数
            pts_h = np.hstack([pts, np.ones((len(pts), 1))])
            T_cam = np.linalg.inv(T_lidar_camera)
            pts_cam = (T_cam @ pts_h.T).T[:, :3]
            valid = pts_cam[:, 2] > 0.1
            if valid.sum() > 0:
                fx, fy, cx, cy = intrinsics[:4]
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                img_pts, _ = cv2.projectPoints(pts_cam[valid], np.zeros(3), np.zeros(3),
                                               K, np.array(dist_coeffs))
                img_pts = img_pts.reshape(-1, 2).astype(int)
                h, w = image_for_blend.shape[:2]
                in_cam = ((img_pts[:, 0] >= 0) & (img_pts[:, 0] < w) &
                          (img_pts[:, 1] >= 0) & (img_pts[:, 1] < h)).sum()
            print(f"  相机覆盖: {in_cam}/{len(pts)} 点 ({100*in_cam/max(1,len(pts)):.1f}%)")

        captures.append((pts, colors))
        total = sum(len(c[0]) for c in captures)
        print(f"  ✓ 采集完成！累计 {len(captures)} 次, {total} 点")

    if not captures:
        print("\n没有采集数据")
        node.destroy_node()
        rclpy.shutdown()
        return

    # 合并所有采集
    print(f"\n合并 {len(captures)} 次采集...")
    all_pts = np.vstack([c[0] for c in captures])
    all_clr = np.vstack([c[1] for c in captures])
    print(f"  总计: {len(all_pts)} 点")

    # 3D 预览
    if args.show_3d and HAS_O3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts)
        pcd.colors = o3d.utility.Vector3dVector(all_clr)
        print("  3D 预览（关闭窗口继续）...")
        o3d.visualization.draw_geometries([pcd], window_name=f"合并 {len(all_pts)} 点",
                                          width=1280, height=720)

    # 去天花板
    if not args.no_remove_ceiling:
        all_pts, all_clr = remove_ceiling(all_pts, all_clr)
        print(f"  去天花板后: {len(all_pts)} 点")

    # 生成 BEV
    output = args.output or os.path.join(os.path.dirname(args.calib_json), 'online_bev.png')
    print(f"\n  生成俯视图 (分辨率: {args.resolution} m/px)...")
    bev, meta = generate_bev(all_pts, all_clr, resolution=args.resolution)

    if bev is not None:
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        cv2.imwrite(output, bev)

        bev_clean, _ = generate_bev(all_pts, all_clr, resolution=args.resolution, grid=False)
        clean_path = output.rsplit('.', 1)[0] + '_clean.png'
        if bev_clean is not None:
            cv2.imwrite(clean_path, bev_clean)

        meta['blend_weight'] = args.blend
        meta['num_captures'] = len(captures)
        meta_path = output.rsplit('.', 1)[0] + '_meta.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"\n  输出:")
        print(f"    带网格: {output}")
        print(f"    干净版: {clean_path}")
        print(f"    场地:   {meta['field_width_m']:.2f}m x {meta['field_height_m']:.2f}m")
        print(f"    图像:   {meta['image_size'][0]}x{meta['image_size'][1]}")
        print(f"    采集:   {len(captures)} 次, {meta['num_points']} 点")

    # 保存着色点云
    if args.save_ply and HAS_O3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts)
        pcd.colors = o3d.utility.Vector3dVector(all_clr)
        os.makedirs(os.path.dirname(args.save_ply) or '.', exist_ok=True)
        o3d.io.write_point_cloud(args.save_ply, pcd)
        print(f"    着色点云: {args.save_ply}")

    node.destroy_node()
    rclpy.shutdown()
    print("\n  完成！")


if __name__ == '__main__':
    main()
