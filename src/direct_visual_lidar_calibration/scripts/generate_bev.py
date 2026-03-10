#!/usr/bin/env python3
"""
generate_bev.py — 基于 direct_visual_lidar_calibration 预处理数据生成俯视图

完全复用 DVL 的预处理结果：
  - PLY 点云（intensity 已做直方图均衡化）
  - 相机图像（已做 equalizeHist）
  - calib.json（T_lidar_camera + 相机内参）

着色流程（和 DVL viewer 的 PointsColorUpdater 一致）：
  1. 均衡化后的 intensity → TURBO 色表
  2. 用 T_lidar_camera 投影到相机图像获取灰度
  3. 混合：final = gray * blend + turbo * (1 - blend)
  4. 去天花板 → 生成俯视图

用法：
  # 基本用法
  python3 scripts/generate_bev.py /home/rq/radar/calib_ws/preprocessed

  # 指定参数
  python3 scripts/generate_bev.py /home/rq/radar/calib_ws/preprocessed \
      --blend 0.7 --resolution 0.01 --output bev.png

  # 3D 预览
  python3 scripts/generate_bev.py /home/rq/radar/calib_ws/preprocessed --show-3d

  # 不去天花板
  python3 scripts/generate_bev.py /home/rq/radar/calib_ws/preprocessed --no-remove-ceiling

  # 纯 intensity 着色（不混合相机）
  python3 scripts/generate_bev.py /home/rq/radar/calib_ws/preprocessed --blend 0.0
"""

import os
import sys
import json
import struct
import argparse
import numpy as np
import cv2

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


# ─── PLY 读取（直接解析 binary_little_endian，不依赖 plyfile） ──

def read_ply_with_intensity(path):
    """
    读取 DVL 生成的 PLY 文件（binary_little_endian）。
    返回 (points: Nx3 float64, intensity: N float64)
    """
    with open(path, 'rb') as f:
        # 解析 header
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        n_vertices = 0
        properties = []
        for line in header_lines:
            if line.startswith('element vertex'):
                n_vertices = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                prop_type = parts[1]
                prop_name = parts[2]
                properties.append((prop_name, prop_type))

        # 构建 struct 格式
        type_map = {'float': 'f', 'double': 'd', 'uchar': 'B', 'int': 'i', 'uint': 'I'}
        fmt = '<'  # little endian
        for _, ptype in properties:
            fmt += type_map.get(ptype, 'f')

        record_size = struct.calcsize(fmt)
        data = f.read(n_vertices * record_size)

    # 解析数据
    prop_names = [p[0] for p in properties]
    points = np.zeros((n_vertices, 3), dtype=np.float64)
    intensity = np.zeros(n_vertices, dtype=np.float64)

    x_idx = prop_names.index('x') if 'x' in prop_names else 0
    y_idx = prop_names.index('y') if 'y' in prop_names else 1
    z_idx = prop_names.index('z') if 'z' in prop_names else 2
    i_idx = prop_names.index('intensity') if 'intensity' in prop_names else -1

    for j in range(n_vertices):
        record = struct.unpack_from(fmt, data, j * record_size)
        points[j, 0] = record[x_idx]
        points[j, 1] = record[y_idx]
        points[j, 2] = record[z_idx]
        if i_idx >= 0:
            intensity[j] = record[i_idx]

    return points, intensity


# ─── 着色函数 ──────────────────────────────────────────────

def intensity_to_turbo(intensity):
    """TURBO 色表映射（和 glk::COLORMAP::TURBO 一致）"""
    i_u8 = (np.clip(intensity, 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(i_u8.reshape(-1, 1), cv2.COLORMAP_TURBO)
    return colored.reshape(-1, 3)[:, ::-1].astype(np.float64) / 255.0  # BGR→RGB, [0,1]


def project_to_camera(points, T_lidar_camera, intrinsics, dist_coeffs, img_shape):
    """
    将点云投影到相机图像。
    T_lidar_camera: 4x4, 从 lidar 坐标系到 camera 坐标系的变换。
    注意：DVL 的 calib.json 存的是 T_lidar_camera，
    viewer 里用 T_camera_lidar = T_lidar_camera.inverse()。
    投影时需要先把点从 lidar 变换到 camera 坐标系。
    """
    # 点变换到相机坐标系
    pts_h = np.hstack([points, np.ones((len(points), 1))])
    # T_lidar_camera 把 lidar 坐标变换到 camera 坐标
    # 但实际上 DVL 的命名是：T_lidar_camera 表示 "lidar frame 中 camera 的位姿"
    # 所以要把点从 lidar 变到 camera，需要用 T_lidar_camera 的逆
    T_cam_lidar = np.linalg.inv(T_lidar_camera)
    pts_cam = (T_cam_lidar @ pts_h.T).T[:, :3]

    # 只保留相机前方
    valid = pts_cam[:, 2] > 0.1

    fx, fy, cx, cy = intrinsics[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array(dist_coeffs, dtype=np.float64)

    pts_for_proj = pts_cam[valid].astype(np.float64)
    if len(pts_for_proj) == 0:
        return np.zeros((len(points), 2), dtype=int), np.zeros(len(points), dtype=bool)

    img_pts, _ = cv2.projectPoints(pts_for_proj, np.zeros(3), np.zeros(3), K, dist)
    img_pts = img_pts.reshape(-1, 2).astype(int)

    h, w = img_shape[:2]
    in_bounds = (img_pts[:, 0] >= 0) & (img_pts[:, 0] < w) & \
                (img_pts[:, 1] >= 0) & (img_pts[:, 1] < h)

    pixel_coords = np.zeros((len(points), 2), dtype=int)
    final_valid = np.zeros(len(points), dtype=bool)
    valid_idx = np.where(valid)[0]
    ok_idx = valid_idx[in_bounds]
    pixel_coords[ok_idx] = img_pts[in_bounds]
    final_valid[ok_idx] = True

    return pixel_coords, final_valid


def blend_colors(turbo_rgb, image_gray, pixel_coords, valid_mask, blend_weight):
    """
    复现 PointsColorUpdater::update：
    color = gray * blend + turbo * (1 - blend)
    """
    n = len(turbo_rgb)
    colors = turbo_rgb.copy()  # 默认用 turbo

    for i in range(n):
        if valid_mask[i]:
            px, py = pixel_coords[i]
            g = image_gray[py, px] / 255.0
            colors[i] = np.array([g, g, g]) * blend_weight + turbo_rgb[i] * (1.0 - blend_weight)

    return colors


# ─── 天花板去除 ─────────────────────────────────────────────

def remove_ceiling(points, colors, threshold=0.15):
    """RANSAC 检测并去除天花板（最高的大水平面）"""
    if not HAS_OPEN3D:
        print("  警告：没有 open3d，跳过天花板去除")
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
    removed = (~keep).sum()
    print(f"  去除天花板: Z={ceiling_z:.2f}m, 去除 {removed} 点")
    return points[keep], colors[keep]


# ─── BEV 生成 ──────────────────────────────────────────────

def generate_bev(points, colors_rgb, resolution=0.01, bg=(30, 30, 30),
                 grid=True, margin=0.3):
    """俯视图生成，colors_rgb 是 [0,1] 的 Nx3"""
    if len(points) == 0:
        return None, None

    xmin, xmax = points[:, 0].min() - margin, points[:, 0].max() + margin
    ymin, ymax = points[:, 1].min() - margin, points[:, 1].max() + margin
    W = int((xmax - xmin) / resolution)
    H = int((ymax - ymin) / resolution)

    if W <= 0 or H <= 0 or W > 20000 or H > 20000:
        print(f"  图像尺寸异常: {W}x{H}")
        return None, None

    bev = np.full((H, W, 3), bg, dtype=np.uint8)
    zbuf = np.full((H, W), -np.inf)

    px = ((points[:, 0] - xmin) / resolution).astype(int)
    py = (H - 1 - ((points[:, 1] - ymin) / resolution).astype(int))
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px, py = px[ok], py[ok]
    z = points[ok, 2]
    c = (colors_rgb[ok] * 255).astype(np.uint8)

    for i in np.argsort(z):
        if z[i] > zbuf[py[i], px[i]]:
            zbuf[py[i], px[i]] = z[i]
            bev[py[i], px[i]] = c[i][::-1]  # RGB→BGR

    # 填充空洞
    mask = (zbuf > -np.inf).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=2)
    filled = cv2.medianBlur(bev, 3)
    gap = (dilated > 0) & (mask == 0)
    bev[gap] = filled[gap]
    mask = dilated
    bev[mask == 0] = bg

    # 裁切
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if np.any(rows) and np.any(cols):
        mp = int(margin / resolution)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        r0, r1 = max(0, r0 - mp), min(H - 1, r1 + mp)
        c0, c1 = max(0, c0 - mp), min(W - 1, c1 + mp)
        bev = bev[r0:r1+1, c0:c1+1].copy()
        xmin += c0 * resolution
        xmax = xmin + (c1 - c0 + 1) * resolution
        ymax_n = ymin + (H - r0) * resolution
        ymin_n = ymin + (H - r1 - 1) * resolution
        ymin, ymax = ymin_n, ymax_n
        H, W = bev.shape[:2]

    fw, fh = xmax - xmin, ymax - ymin

    if grid:
        bev = draw_grid(bev, xmin, xmax, ymin, ymax, resolution)

    meta = {
        'x_range': [xmin, xmax], 'y_range': [ymin, ymax],
        'resolution': resolution, 'image_size': [W, H],
        'field_width_m': fw, 'field_height_m': fh,
        'num_points': len(points),
    }
    return bev, meta


def draw_grid(bev, xmin, xmax, ymin, ymax, res):
    H, W = bev.shape[:2]
    sz = max(xmax - xmin, ymax - ymin)
    step = 0.5 if sz <= 5 else (1.0 if sz <= 15 else (2.0 if sz <= 30 else 5.0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.3, min(1.0, W / 2000))
    th = max(1, int(W / 1500))

    x = np.ceil(xmin / step) * step
    while x <= xmax:
        px = int((x - xmin) / res)
        if 0 <= px < W:
            cv2.line(bev, (px, 0), (px, H - 1), (100, 100, 100), 1)
            cv2.putText(bev, f"{x:.1f}m", (px + 3, H - 10), font, fs, (200, 200, 200), th)
        x += step

    y = np.ceil(ymin / step) * step
    while y <= ymax:
        py = H - 1 - int((y - ymin) / res)
        if 0 <= py < H:
            cv2.line(bev, (0, py), (W - 1, py), (100, 100, 100), 1)
            cv2.putText(bev, f"{y:.1f}m", (5, py - 5), font, fs, (200, 200, 200), th)
        y += step

    # 比例尺
    sl = int(step / res)
    m, by = 30, H - 30
    cv2.rectangle(bev, (m - 5, by - 25), (m + sl + 5, by + 10), (0, 0, 0), -1)
    cv2.line(bev, (m, by), (m + sl, by), (255, 255, 255), 2)
    cv2.line(bev, (m, by - 5), (m, by + 5), (255, 255, 255), 2)
    cv2.line(bev, (m + sl, by - 5), (m + sl, by + 5), (255, 255, 255), 2)
    cv2.putText(bev, f"{step:.1f}m", (m, by - 10), font, fs, (255, 255, 255), th)

    st = f"Field: {xmax-xmin:.1f}m x {ymax-ymin:.1f}m"
    tsz = cv2.getTextSize(st, font, fs, th)[0]
    tx = W - tsz[0] - 10
    cv2.rectangle(bev, (tx - 5, 0), (tx + tsz[0] + 5, 30), (0, 0, 0), -1)
    cv2.putText(bev, st, (tx, 22), font, fs, (255, 255, 255), th)
    return bev


# ─── 3D 预览（复现 DVL viewer） ───────────────────────────────

def show_3d_preview(points, colors_rgb, title="DVL colored point cloud"):
    """用 Open3D 显示着色后的点云（复现 DVL viewer 的效果）"""
    if not HAS_OPEN3D:
        print("  没有 open3d，跳过 3D 预览")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

    # 设置和 DVL viewer 类似的视角
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=720)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = 1.5
    opt.background_color = np.array([0.15, 0.15, 0.15])

    vis.run()
    vis.destroy_window()


# ─── 主程序 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='基于 DVL 预处理数据生成着色俯视图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 scripts/generate_bev.py /home/rq/radar/calib_ws/preprocessed
  python3 scripts/generate_bev.py /home/rq/radar/calib_ws/preprocessed --show-3d
  python3 scripts/generate_bev.py /home/rq/radar/calib_ws/preprocessed --blend 0.0  # 纯 intensity
  python3 scripts/generate_bev.py /home/rq/radar/calib_ws/preprocessed --blend 1.0  # 纯相机
        """
    )
    parser.add_argument('data_path', help='DVL 预处理数据目录（含 calib.json, *.ply, *.png）')
    parser.add_argument('--output', '-o', default=None, help='输出路径（默认: data_path/bev.png）')
    parser.add_argument('--blend', type=float, default=0.7,
                        help='相机混合权重 (0=纯TURBO intensity, 1=纯相机灰度, 默认0.7)')
    parser.add_argument('--resolution', '-r', type=float, default=0.01, help='BEV 分辨率 m/px')
    parser.add_argument('--max-dist', type=float, default=15.0, help='最大距离')
    parser.add_argument('--min-dist', type=float, default=0.3, help='最小距离')
    parser.add_argument('--no-remove-ceiling', action='store_true', help='不去天花板')
    parser.add_argument('--z-range', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'),
                        help='手动指定 Z 范围')
    parser.add_argument('--show-3d', action='store_true', help='3D 预览（复现 DVL viewer）')
    parser.add_argument('--save-colored-ply', default=None, help='保存着色后的 PLY')
    parser.add_argument('--no-grid', action='store_true', help='不画网格')
    args = parser.parse_args()

    data_path = args.data_path
    print(f"\n{'='*60}")
    print(f"  DVL BEV 生成工具")
    print(f"{'='*60}")
    print(f"  数据目录: {data_path}")

    # 读取 calib.json
    calib_path = os.path.join(data_path, 'calib.json')
    if not os.path.exists(calib_path):
        print(f"  错误：{calib_path} 不存在")
        return 1

    with open(calib_path) as f:
        calib = json.load(f)

    camera_model = calib['camera']['camera_model']
    intrinsics = calib['camera']['intrinsics']
    dist_coeffs = calib['camera']['distortion_coeffs']
    bag_names = calib['meta']['bag_names']
    print(f"  相机模型: {camera_model}")
    print(f"  内参: fx={intrinsics[0]:.1f} fy={intrinsics[1]:.1f} cx={intrinsics[2]:.1f} cy={intrinsics[3]:.1f}")
    print(f"  数据包: {bag_names}")

    # 解析 T_lidar_camera
    results = calib.get('results', {})
    T_lidar_camera = None
    for key in ['T_lidar_camera', 'init_T_lidar_camera', 'init_T_lidar_camera_auto']:
        if key in results:
            v = results[key]
            from scipy.spatial.transform import Rotation
            R = Rotation.from_quat([v[3], v[4], v[5], v[6]]).as_matrix()
            T_lidar_camera = np.eye(4)
            T_lidar_camera[:3, :3] = R
            T_lidar_camera[:3, 3] = [v[0], v[1], v[2]]
            print(f"  使用变换: {key}")
            break

    if T_lidar_camera is None:
        print("  警告：calib.json 中没有 T_lidar_camera，仅用 intensity 着色")

    # 读取所有数据包
    all_points = []
    all_intensity = []
    image = None

    for bag_name in bag_names:
        ply_path = os.path.join(data_path, bag_name + '.ply')
        img_path = os.path.join(data_path, bag_name + '.png')

        if not os.path.exists(ply_path):
            print(f"  警告：{ply_path} 不存在")
            continue

        print(f"\n  读取: {ply_path}")
        pts, ints = read_ply_with_intensity(ply_path)
        print(f"    {len(pts)} 点, intensity 范围: [{ints.min():.3f}, {ints.max():.3f}]")
        all_points.append(pts)
        all_intensity.append(ints)

        if os.path.exists(img_path):
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            print(f"    图像: {img_path} ({image.shape[1]}x{image.shape[0]})")

    if not all_points:
        print("  没有数据")
        return 1

    points = np.vstack(all_points)
    intensity = np.concatenate(all_intensity)
    print(f"\n  总计: {len(points)} 点")

    # 距离滤波
    dist = np.linalg.norm(points, axis=1)
    mask = (dist >= args.min_dist) & (dist <= args.max_dist)
    points, intensity = points[mask], intensity[mask]
    print(f"  距离滤波后: {len(points)} 点")

    # Z 范围
    if args.z_range:
        z0, z1 = args.z_range
        mask = (points[:, 2] >= z0) & (points[:, 2] <= z1)
        points, intensity = points[mask], intensity[mask]
        print(f"  Z 过滤后: {len(points)} 点")

    # TURBO 着色（intensity 已经是均衡化后的 [0,1]）
    print(f"  TURBO 着色...")
    turbo_colors = intensity_to_turbo(intensity)  # Nx3 RGB [0,1]

    # 相机混合
    if image is not None and T_lidar_camera is not None and args.blend > 0:
        print(f"  投影到相机 (blend={args.blend})...")
        pixel_coords, valid_mask = project_to_camera(
            points, T_lidar_camera, intrinsics, dist_coeffs, image.shape
        )
        in_img = valid_mask.sum()
        print(f"    {in_img}/{len(points)} 点在图像内 ({100*in_img/len(points):.1f}%)")
        colors = blend_colors(turbo_colors, image, pixel_coords, valid_mask, args.blend)
    else:
        if args.blend > 0 and (image is None or T_lidar_camera is None):
            print("  无相机数据或标定结果，仅用 intensity 着色")
        colors = turbo_colors

    # 3D 预览
    if args.show_3d:
        print("  3D 预览（关闭窗口继续）...")
        show_3d_preview(points, colors, f"DVL colored — {len(points)} pts (blend={args.blend})")

    # 去天花板
    if not args.no_remove_ceiling and args.z_range is None:
        points, colors = remove_ceiling(points, colors)
        print(f"  去天花板后: {len(points)} 点")

    # 生成 BEV
    output = args.output or os.path.join(data_path, 'bev.png')
    print(f"\n  生成俯视图 (分辨率: {args.resolution} m/px)...")
    bev, meta = generate_bev(points, colors, resolution=args.resolution, grid=not args.no_grid)

    if bev is not None:
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        cv2.imwrite(output, bev)

        # 干净版
        bev_clean, _ = generate_bev(points, colors, resolution=args.resolution, grid=False)
        clean = output.rsplit('.', 1)
        clean_path = clean[0] + '_clean.' + (clean[1] if len(clean) > 1 else 'png')
        if bev_clean is not None:
            cv2.imwrite(clean_path, bev_clean)

        # 元数据
        meta['blend_weight'] = args.blend
        meta['data_path'] = data_path
        meta_path = output.rsplit('.', 1)[0] + '_meta.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"\n  输出:")
        print(f"    带网格: {output}")
        print(f"    干净版: {clean_path}")
        print(f"    元数据: {meta_path}")
        print(f"    场地:   {meta['field_width_m']:.2f}m x {meta['field_height_m']:.2f}m")
        print(f"    图像:   {meta['image_size'][0]}x{meta['image_size'][1]}")
        print(f"    点数:   {meta['num_points']}")

    # 保存着色点云
    if args.save_colored_ply and HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        save_path = args.save_colored_ply
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"    着色点云: {save_path}")

    print("\n  完成！")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
