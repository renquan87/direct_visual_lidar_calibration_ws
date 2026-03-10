#!/usr/bin/env python3
"""
capture_colored.py — 采集彩色点云

每按 Enter 采集一次（累积多帧），着色后保存为 PLY 文件。
可以多次运行，每次运行的文件自动编号，不会覆盖之前的。

用法：
  python3 scripts/capture_colored.py \
      --calib-json /data/projects/radar/direct_visual_lidar_calibration_ws/preprocessed/calib.json

  # 自定义参数
  python3 scripts/capture_colored.py \
      --calib-json .../calib.json \
      --accumulate 30 --blend 0.7 --max-dist 15
"""

import os
import sys
import json
import time
import glob
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

# 尝试导入海康相机
HAS_HKCAM = False
try:
    import sys as _sys
    _cam_path = "/data/projects/radar/hnurm_radar/src/hnurm_radar/hnurm_radar/Camera"
    _sys.path.insert(0, _cam_path)
    _sys.path.insert(0, os.path.join(_cam_path, "MvImport"))
    from HKCam import HKCam
    HAS_HKCAM = True
except Exception:
    HAS_HKCAM = False


SAVE_DIR = "/data/projects/radar/direct_visual_lidar_calibration_ws/colored_captures"


def intensity_to_turbo(intensity):
    i_u8 = (np.clip(intensity, 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(i_u8.reshape(-1, 1), cv2.COLORMAP_TURBO)
    return colored.reshape(-1, 3)[:, ::-1].astype(np.float64) / 255.0


def histogram_equalize(intensity):
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
    int_eq = histogram_equalize(intensity)
    turbo = intensity_to_turbo(int_eq)

    if image_gray is None or T_lidar_camera is None or blend <= 0:
        return turbo

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


class Collector(Node):
    def __init__(self, pcl_topic, img_topic, use_hkcam=False):
        super().__init__('capture_colored')
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_points = None
        self.latest_intensity = None
        self.latest_image = None
        self.use_hkcam = use_hkcam
        self.hkcam = None

        if 'livox' in pcl_topic:
            pcl_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                 history=HistoryPolicy.KEEP_LAST, depth=10)
        else:
            pcl_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 history=HistoryPolicy.KEEP_LAST, depth=5)

        self.create_subscription(PointCloud2, pcl_topic, self._pcl_cb, pcl_qos)

        if use_hkcam and HAS_HKCAM:
            # 直接用海康 SDK 采集图像
            self.get_logger().info("使用海康工业相机 (HKCam) 直接采集图像")
            try:
                self.hkcam = HKCam(0)
                self._cam_thread = threading.Thread(target=self._hkcam_loop, daemon=True)
                self._cam_thread.start()
            except Exception as e:
                self.get_logger().error(f"HKCam 初始化失败: {e}")
                self.hkcam = None
                self._setup_img_sub(img_topic)
        else:
            self._setup_img_sub(img_topic)

    def _setup_img_sub(self, img_topic):
        """通过 ROS2 话题订阅图像"""
        img_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(Image, img_topic, self._img_cb, img_qos)

    def _hkcam_loop(self):
        """海康相机采集循环"""
        while rclpy.ok():
            try:
                frame = self.hkcam.getFrame()
                if frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.equalizeHist(gray)
                    with self.lock:
                        self.latest_image = gray
            except Exception:
                pass
            time.sleep(0.05)

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


def get_next_index(save_dir):
    """找到已有文件的最大编号，返回下一个"""
    existing = glob.glob(os.path.join(save_dir, "capture_*.ply"))
    if not existing:
        return 0
    nums = []
    for f in existing:
        base = os.path.basename(f)
        try:
            n = int(base.replace("capture_", "").replace(".ply", ""))
            nums.append(n)
        except ValueError:
            pass
    return max(nums) + 1 if nums else 0


def save_colored_ply(points, colors, filepath):
    """保存带颜色的 PLY"""
    if HAS_O3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(filepath, pcd)
    else:
        # 手动写 PLY
        n = len(points)
        c_u8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
        with open(filepath, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for i in range(n):
                f.write(f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f} "
                        f"{c_u8[i,0]} {c_u8[i,1]} {c_u8[i,2]}\n")


def main():
    parser = argparse.ArgumentParser(description='采集彩色点云，每按 Enter 保存一个 PLY')
    parser.add_argument('--calib-json', required=True, help='calib.json 路径')
    parser.add_argument('--pcl-topic', default='/livox/lidar', help='点云话题')
    parser.add_argument('--img-topic', default='/image', help='图像话题')
    parser.add_argument('--blend', type=float, default=0.7, help='相机混合权重')
    parser.add_argument('--accumulate', type=int, default=20, help='每次采集累积帧数')
    parser.add_argument('--max-dist', type=float, default=15.0, help='最大距离')
    parser.add_argument('--min-dist', type=float, default=0.3, help='最小距离')
    parser.add_argument('--save-dir', default=SAVE_DIR, help='保存目录')
    parser.add_argument('--voxel-size', type=float, default=0.02, help='降采样体素大小（0=不降采样）')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

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
    use_hkcam = HAS_HKCAM  # 有海康 SDK 就直接用
    node = Collector(args.pcl_topic, args.img_topic, use_hkcam=use_hkcam)
    spin_t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_t.start()

    next_idx = get_next_index(args.save_dir)

    print()
    print("=" * 60)
    print("  彩色点云采集工具")
    print("=" * 60)
    print(f"  点云: {args.pcl_topic}")
    print(f"  图像: {args.img_topic}")
    print(f"  混合: {args.blend}")
    print(f"  累积: {args.accumulate} 帧/次")
    print(f"  保存: {args.save_dir}")
    print(f"  已有: {next_idx} 个采集文件")
    print()
    print("  Enter — 采集并保存")
    print("  q     — 退出")
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
                print("  图像: 暂无")
            break
        time.sleep(0.1)
    else:
        print("  超时，未收到数据")
        node.destroy_node()
        rclpy.shutdown()
        return

    while True:
        try:
            cmd = input(f"\n[下一个: capture_{next_idx:03d}.ply] Enter=采集, q=退出: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == 'q':
            break

        if cmd and cmd != '':
            continue

        # 采集
        print(f"  采集中...")
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

        # 降采样
        if HAS_O3D and args.voxel_size > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd_down = pcd.voxel_down_sample(args.voxel_size)
            down_pts = np.asarray(pcd_down.points)
            tree = o3d.geometry.KDTreeFlann(pcd)
            new_ints = np.zeros(len(down_pts))
            for j in range(len(down_pts)):
                _, nn_idx, _ = tree.search_knn_vector_3d(down_pts[j], 1)
                new_ints[j] = ints[nn_idx[0]]
            pts, ints = down_pts, new_ints

        print(f"  处理后: {len(pts)} 点")

        # 着色
        colors = project_and_color(pts, ints, image_for_blend, T_lidar_camera,
                                   intrinsics, dist_coeffs, args.blend)

        # 保存
        filename = f"capture_{next_idx:03d}.ply"
        filepath = os.path.join(args.save_dir, filename)
        save_colored_ply(pts, colors, filepath)
        print(f"  ✓ 已保存: {filepath} ({len(pts)} 点)")

        next_idx += 1

    node.destroy_node()
    rclpy.shutdown()
    total = get_next_index(args.save_dir)
    print(f"\n  总共 {total} 个采集文件在 {args.save_dir}")
    print("  下一步: python3 scripts/merge_clouds.py")


if __name__ == '__main__':
    main()
