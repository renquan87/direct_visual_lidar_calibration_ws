#!/usr/bin/env python3
"""
interactive_merge.py — 交互式手动对齐 + ICP 精配准

所有操作都在 3D 窗口中完成（键盘快捷键）：

移动:  W/S=Y轴  A/D=X轴  Q/E=Z轴
旋转:  R/F=绕Z轴
步长:  1=0.1m  2=0.5m  3=1m  4=2m  5=5m
旋转步长: 6=5°  7=15°  8=45°  9=90°
ICP:   Space=执行ICP精配准
确认:  Enter=确认合并  Backspace=撤销  P=跳过  X=退出保存
"""

import os, sys, glob, copy, json, argparse, time
import numpy as np

os.environ['PYTHONUNBUFFERED'] = '1'

try:
    import open3d as o3d
except ImportError:
    print("需要 open3d: pip install open3d"); exit(1)

CAPTURE_DIR = "/data/projects/radar/direct_visual_lidar_calibration_ws/colored_captures"


def icp_refine(source, target, init_T, voxel=0.1):
    src = source.voxel_down_sample(voxel)
    tgt = target.voxel_down_sample(voxel)
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2, max_nn=30))
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2, max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        src, tgt, voxel*2, init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
    return result.transformation, result.fitness, result.inlier_rmse


class InteractiveMerger:
    def __init__(self, pcds, args):
        self.pcds = pcds  # [(name, pcd), ...]
        self.args = args
        self.transforms = {}
        self.transforms[pcds[0][0]] = np.eye(4)

        self.current_idx = 1
        self.current_T = np.eye(4)
        self.step_trans = 1.0
        self.step_rot = np.radians(15)

        self.action = None  # 'confirm', 'skip', 'exit', 'icp'

        # 构建已合并点云
        self.merged = copy.deepcopy(pcds[0][1])
        self.merged_down = self.merged.voxel_down_sample(args.voxel_display)

        # 可视化对象
        self.vis = None
        self.ref_geo = None
        self.cur_geo = None

    def log(self, msg):
        print(msg, flush=True)

    def update_current_display(self):
        """更新当前点云的显示"""
        if self.cur_geo is not None and self.vis is not None:
            self.vis.remove_geometry(self.cur_geo, reset_bounding_box=False)

        cur = copy.deepcopy(self.pcds[self.current_idx][1])
        cur.transform(self.current_T)
        self.cur_geo = cur.voxel_down_sample(self.args.voxel_display)

        if self.vis is not None:
            self.vis.add_geometry(self.cur_geo, reset_bounding_box=False)
            self.vis.update_renderer()

    def update_ref_display(self):
        """更新参考点云显示"""
        if self.ref_geo is not None and self.vis is not None:
            self.vis.remove_geometry(self.ref_geo, reset_bounding_box=False)

        self.merged_down = self.merged.voxel_down_sample(self.args.voxel_display)
        self.ref_geo = copy.deepcopy(self.merged_down)
        self.ref_geo.paint_uniform_color([0.5, 0.5, 0.5])

        if self.vis is not None:
            self.vis.add_geometry(self.ref_geo, reset_bounding_box=False)
            self.vis.update_renderer()

    def apply_delta(self, delta):
        self.current_T = delta @ self.current_T
        self.update_current_display()
        pos = self.current_T[:3, 3]
        self.log(f"  位置: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] 步长={self.step_trans:.2f}m")

    def register_keys(self, vis):
        # 移动
        vis.register_key_callback(ord('W'), lambda v: self._move(v, 1, self.step_trans))
        vis.register_key_callback(ord('S'), lambda v: self._move(v, 1, -self.step_trans))
        vis.register_key_callback(ord('A'), lambda v: self._move(v, 0, -self.step_trans))
        vis.register_key_callback(ord('D'), lambda v: self._move(v, 0, self.step_trans))
        vis.register_key_callback(ord('Q'), lambda v: self._move(v, 2, self.step_trans))
        vis.register_key_callback(ord('E'), lambda v: self._move(v, 2, -self.step_trans))

        # 旋转
        vis.register_key_callback(ord('R'), lambda v: self._rotate(v, self.step_rot))
        vis.register_key_callback(ord('F'), lambda v: self._rotate(v, -self.step_rot))

        # 步长 - 数字键
        vis.register_key_callback(ord('1'), lambda v: self._set_step(v, 0.1))
        vis.register_key_callback(ord('2'), lambda v: self._set_step(v, 0.5))
        vis.register_key_callback(ord('3'), lambda v: self._set_step(v, 1.0))
        vis.register_key_callback(ord('4'), lambda v: self._set_step(v, 2.0))
        vis.register_key_callback(ord('5'), lambda v: self._set_step(v, 5.0))

        # 旋转步长
        vis.register_key_callback(ord('6'), lambda v: self._set_rot(v, 5))
        vis.register_key_callback(ord('7'), lambda v: self._set_rot(v, 15))
        vis.register_key_callback(ord('8'), lambda v: self._set_rot(v, 45))
        vis.register_key_callback(ord('9'), lambda v: self._set_rot(v, 90))

        # ICP
        vis.register_key_callback(32, lambda v: self._do_icp(v))  # Space

        # 确认/跳过/退出
        vis.register_key_callback(257, lambda v: self._confirm(v))  # Enter (some systems)
        vis.register_key_callback(13, lambda v: self._confirm(v))   # Enter
        vis.register_key_callback(10, lambda v: self._confirm(v))   # Enter (Linux)
        vis.register_key_callback(ord('Y'), lambda v: self._confirm(v))
        vis.register_key_callback(ord('P'), lambda v: self._skip(v))
        vis.register_key_callback(ord('X'), lambda v: self._exit(v))
        vis.register_key_callback(ord('N'), lambda v: self._reset(v))

    def _move(self, vis, axis, amount):
        delta = np.eye(4)
        delta[axis, 3] = amount
        self.apply_delta(delta)
        return False

    def _rotate(self, vis, angle):
        delta = np.eye(4)
        c, s = np.cos(angle), np.sin(angle)
        delta[:2, :2] = [[c, -s], [s, c]]
        self.apply_delta(delta)
        return False

    def _set_step(self, vis, val):
        self.step_trans = val
        self.log(f"  平移步长: {val}m")
        return False

    def _set_rot(self, vis, deg):
        self.step_rot = np.radians(deg)
        self.log(f"  旋转步长: {deg}°")
        return False

    def _do_icp(self, vis):
        self.log(f"  执行 ICP...")
        icp_T, fitness, rmse = icp_refine(
            self.pcds[self.current_idx][1], self.merged,
            self.current_T, self.args.voxel_icp)
        self.log(f"  ICP: fitness={fitness:.4f}, RMSE={rmse:.4f}")
        self.current_T = icp_T
        self.update_current_display()
        return False

    def _confirm(self, vis):
        self.log(f"  ✓ 确认合并")
        self.action = 'confirm'
        vis.close()
        return False

    def _skip(self, vis):
        self.log(f"  跳过")
        self.action = 'skip'
        vis.close()
        return False

    def _exit(self, vis):
        self.log(f"  退出保存")
        self.action = 'exit'
        vis.close()
        return False

    def _reset(self, vis):
        self.log(f"  重置位置")
        self.current_T = np.eye(4)
        self.update_current_display()
        return False

    def align_one(self, idx):
        """对齐一帧"""
        self.current_idx = idx
        self.current_T = np.eye(4)
        self.action = None
        name = self.pcds[idx][0]

        self.log(f"\n{'='*50}")
        self.log(f"[{idx}/{len(self.pcds)-1}] {name} ({len(self.pcds[idx][1].points)} 点)")
        self.log(f"  灰色=已合并  彩色=当前帧")
        self.log(f"  WASDQE=移动 RF=旋转 1-5=步长 Space=ICP Enter/Y=确认 P=跳过 X=退出")

        # 创建可视化
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            window_name=f"对齐 [{idx}/{len(self.pcds)-1}] {name}",
            width=1280, height=720)

        self.register_keys(self.vis)

        # 添加参考
        self.ref_geo = copy.deepcopy(self.merged_down)
        self.ref_geo.paint_uniform_color([0.5, 0.5, 0.5])
        self.vis.add_geometry(self.ref_geo)

        # 添加当前
        self.cur_geo = copy.deepcopy(self.pcds[idx][1]).voxel_down_sample(self.args.voxel_display)
        self.vis.add_geometry(self.cur_geo)

        # 运行
        self.vis.run()
        self.vis.destroy_window()
        self.vis = None

        return self.action

    def merge_frame(self, idx):
        """合并一帧"""
        name = self.pcds[idx][0]
        self.transforms[name] = self.current_T.copy()
        t_pcd = copy.deepcopy(self.pcds[idx][1])
        t_pcd.transform(self.current_T)
        self.merged += t_pcd
        self.merged_down = self.merged.voxel_down_sample(self.args.voxel_display)
        self.log(f"  已合并 (总计 {len(self.merged.points)} 点)")

    def save_state(self, state_file):
        save = {k: v.tolist() for k, v in self.transforms.items()}
        with open(state_file, 'w') as f:
            json.dump(save, f, indent=2)

    def run(self):
        args = self.args
        output = args.output or os.path.join(args.input_dir, "merged.ply")
        state_file = os.path.join(args.input_dir, "merge_state.json")

        # 恢复状态
        start_idx = 1
        if args.resume and os.path.exists(state_file):
            with open(state_file) as f:
                saved = json.load(f)
            self.transforms = {k: np.array(v) for k, v in saved.items()}
            # 重建 merged
            self.merged = o3d.geometry.PointCloud()
            for name, T in self.transforms.items():
                for n, p in self.pcds:
                    if n == name:
                        t = copy.deepcopy(p)
                        t.transform(T)
                        self.merged += t
                        break
            self.merged_down = self.merged.voxel_down_sample(args.voxel_display)
            for idx in range(1, len(self.pcds)):
                if self.pcds[idx][0] not in self.transforms:
                    start_idx = idx
                    break
            else:
                start_idx = len(self.pcds)
            self.log(f"恢复: 已处理 {len(self.transforms)} 帧，从第 {start_idx} 帧继续")

        # 逐帧对齐
        for idx in range(start_idx, len(self.pcds)):
            action = self.align_one(idx)

            if action == 'confirm':
                self.merge_frame(idx)
                self.save_state(state_file)
            elif action == 'exit':
                self.save_state(state_file)
                self.log(f"状态已保存: {state_file}")
                self.log(f"下次用 --resume 继续")
                return
            # skip: do nothing

        # 最终保存
        self.log(f"\n{'='*50}")
        self.log(f"合并完成: {len(self.transforms)} 帧")

        final = o3d.geometry.PointCloud()
        for name, T in self.transforms.items():
            for n, p in self.pcds:
                if n == name:
                    t = copy.deepcopy(p)
                    t.transform(T)
                    final += t
                    break

        self.log(f"  总点数: {len(final.points)}")
        if args.final_voxel > 0:
            final = final.voxel_down_sample(args.final_voxel)
            self.log(f"  降采样: {len(final.points)}")
        final, _ = final.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        self.log(f"  去离群: {len(final.points)}")

        o3d.io.write_point_cloud(output, final)
        self.log(f"\n✓ 保存: {output} ({len(final.points)} 点)")

        tf_path = output.rsplit('.', 1)[0] + '_transforms.json'
        self.save_state(tf_path)
        self.log(f"  变换: {tf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-i', default=CAPTURE_DIR)
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--voxel-display', type=float, default=0.05)
    parser.add_argument('--voxel-icp', type=float, default=0.1)
    parser.add_argument('--final-voxel', type=float, default=0.02)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "capture_*.ply")))
    if not files:
        print("没有找到 capture_*.ply"); return

    print(f"\n找到 {len(files)} 个点云文件", flush=True)
    print("读取点云...", flush=True)
    pcds = []
    for f in files:
        pcd = o3d.io.read_point_cloud(f)
        if len(pcd.points) > 100:
            pcds.append((os.path.basename(f), pcd))
            print(f"  {os.path.basename(f)}: {len(pcd.points)} 点", flush=True)

    if len(pcds) < 2:
        print("点云不足"); return

    print(f"\n操作说明:", flush=True)
    print(f"  WASDQE=移动  RF=旋转  1-5=平移步长  6-9=旋转步长", flush=True)
    print(f"  Space=ICP精配准  Enter/Y=确认合并  P=跳过  N=重置  X=退出保存", flush=True)

    merger = InteractiveMerger(pcds, args)
    merger.run()


if __name__ == '__main__':
    main()
