# 雷达-相机外参标定指南

> 面向 hnurm_radar 雷达站项目 · 标定工具：[direct_visual_lidar_calibration](https://github.com/koide3/direct_visual_lidar_calibration)

---

## 一、外参基础

外参（Extrinsic Parameters）描述激光雷达与相机之间的刚性空间变换，用 4×4 齐次矩阵表示：

```
T = [ R  t ]    R: 3×3 旋转矩阵（正交，det=1）
    [ 0  1 ]    t: 3×1 平移向量（单位：米）
```

### 1.1 两个方向

| 符号 | 含义 | 用途 |
|------|------|------|
| **T_camera_lidar** | 雷达坐标 → 相机坐标 | 投影点云到图像（hnurm_radar 使用） |
| **T_lidar_camera** | 相机坐标 → 雷达坐标 | 标定工具输出（calib.json 存储） |

互为逆矩阵：`T_camera_lidar = inv(T_lidar_camera)`

### 1.2 坐标系

**Livox HAP 激光雷达**（原点 O 在光学窗口表面中心）：

| 轴 | 方向 | 说明 |
|----|------|------|
| X | 向前 | 垂直于光学窗口向外 |
| Y | 向左 | |
| Z | 向上 | |

> O 点距底面 23mm，距定位基准面 O' 前方 8mm。

**相机**（标准 OpenCV 相机模型）：

| 轴 | 方向 |
|----|------|
| X | 向右 |
| Y | 向下 |
| Z | 向前（光轴方向） |

### 1.3 旋转含义速查

当相机装在雷达上方、朝前看时，T_camera_lidar 的旋转矩阵约为：

```
         雷达X(前)  雷达Y(左)  雷达Z(上)
相机X(右) [  ~0      ~-1       ~0    ]    雷达Y(左) → 相机-X(右的反方向) ✓
相机Y(下) [  ~0       ~0       ~-1   ]    雷达Z(上) → 相机-Y(下的反方向) ✓
相机Z(前) [  ~1       ~0       ~0    ]    雷达X(前) → 相机Z(前)          ✓
```

理想情况旋转矩阵很接近上面的模式，只有微小角度偏差（几度以内）。

### 1.4 平移含义

T_camera_lidar 的平移向量 t 表示**雷达原点在相机坐标系中的坐标**。

要得到**相机在雷达坐标系中的位置**：`p_cam_in_lidar = -R^T @ t`

典型值（传感器紧凑安装时每个分量 < 15cm，总距离 < 20cm）。

---

## 二、标定流程概览

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────┐     ┌──────────┐
│  录制 rosbag │ ──▶ │  preprocess 预处理│ ──▶ │ 初始外参估计    │ ──▶ │ calibrate│
│  (30~60秒)   │     │  累积点云+图像   │     │ (三选一，见下)  │     │ NID 精调 │
└─────────────┘     └──────────────────┘     └────────────────┘     └──────────┘
```

初始外参估计三种方式（按推荐优先级）：

| 方式 | 命令 | 适用场景 |
|------|------|----------|
| **手动对齐** | `initial_guess_manual` | ✅ 最可靠，任何场景都能用 |
| **注入旧值** | 编辑 calib.json | ✅ 传感器未拆装时直接复用 |
| **自动估计** | `find_matches_superglue` + `initial_guess_auto` | ⚠️ 需要纹理丰富场景 |

---

## 三、前置条件

- ✅ 相机内参已标定（`ost.yaml` 已生成）
- ✅ 相机和激光雷达刚性固定，相对位置不会变
- ✅ `direct_visual_lidar_calibration_ws` 已编译

```bash
cd /data/projects/radar/direct_visual_lidar_calibration_ws
colcon build --symlink-install
source install/setup.bash
```

---

## 四、操作步骤

### 4.1 启动传感器（三个终端）

```bash
# 终端 1 — 激光雷达
cd /data/projects/radar/hnurm_radar && source install/setup.bash
ros2 launch livox_ros_driver2 rviz_HAP_launch.py

# 终端 2 — 相机
cd /data/projects/radar/lidar_camera_calib_utils && source install/setup.bash
ros2 launch hnurm_camera hnurm_camera.launch.py

# 终端 3 — 相机内参发布
cd /data/projects/radar/lidar_camera_calib_utils && source install/setup.bash
python3 publish_camera_info.py
```

确认话题：`ros2 topic list` 应包含 `/image`、`/livox/lidar`、`/camera_info`

### 4.2 录制数据

```bash
cd /data/projects/radar/direct_visual_lidar_calibration_ws
source install/setup.bash
mkdir -p livox

ros2 bag record /camera_info /image /livox/lidar \
    -o livox/rosbag2_$(date +%Y_%m_%d-%H_%M_%S)
```

**录制要求**：
- 传感器**完全静止**
- 录制 **30~60 秒**（HAP 非重复扫描需要足够时间积累稠密点云）
- **场景选择**（对自动方式很重要，手动方式要求较低）：
  - ✅ 多深度层次（近处桌椅 + 远处墙壁）
  - ✅ 明显边缘（门框、柱子、设备棱角）
  - ✅ 纹理图案（海报、文字、标识牌）
  - ❌ 白墙/空地/大面积均匀区域

### 4.3 预处理

```bash
ros2 run direct_visual_lidar_calibration preprocess livox preprocessed -av
```

产出 `preprocessed/` 目录：

| 文件 | 内容 |
|------|------|
| `calib.json` | 元数据（内参、话题名等） |
| `*.ply` | 累积稠密点云 |
| `*.png` | 关键帧图像（**原始畸变图像**，仅做直方图均衡化） |
| `*_intensities.png` | 点云强度图 |
| `*_point_indices.png` | 像素→3D点索引图 |

> **注意**：preprocess 保存的是原始畸变图像，不做去畸变。calibrate 在投影时会施加畸变模型，与原始图像匹配。这是设计上的一致性。

### 4.4 初始外参估计

#### 方式一：手动对齐（推荐）

```bash
ros2 run direct_visual_lidar_calibration initial_guess_manual preprocessed
```

GUI 操作：
1. 在图像上和点云上各选取 **≥ 3 个对应点**
2. 点击 **Estimate** 估计初始位姿
3. 用 **6-DOF Gizmo** 微调，使点云投影与图像边缘对齐
4. 点击 **Save** → 写入 `calib.json` 的 `init_T_lidar_camera`

#### 方式二：注入已有标定值

适用于传感器未拆装、有旧的好结果可用的情况：

```bash
python3 << 'PYEOF'
import json

calib_path = "preprocessed/calib.json"
with open(calib_path) as f:
    config = json.load(f)

# 填入已知的 T_lidar_camera (TUM 格式: tx,ty,tz,qx,qy,qz,qw)
config.setdefault("results", {})["init_T_lidar_camera"] = [
    0.078105, -0.051296, 0.049243,         # tx, ty, tz
    -0.505595, 0.493719, -0.496659, 0.503929  # qx, qy, qz, qw
]

with open(calib_path, 'w') as f:
    json.dump(config, f, indent=2)
print("已写入 init_T_lidar_camera")
PYEOF
```

#### 方式三：SuperGlue 自动估计

```bash
# 特征匹配
ros2 run direct_visual_lidar_calibration find_matches_superglue.py preprocessed

# RANSAC 估计
ros2 run direct_visual_lidar_calibration initial_guess_auto preprocessed
```

**关键**：检查终端输出 `num_inliers: XX / YY`

| 内点率 | 状态 | 操作 |
|--------|------|------|
| > 30% | ✅ 好 | 继续 calibrate |
| 10~30% | ⚠️ 勉强 | 可尝试，注意观察 |
| < 10% | ❌ 失败 | 换用方式一或方式二 |

同时检查平移量：如果某分量 > 0.3m（30cm），基本不可信。

### 4.5 精细标定（calibrate）

```bash
ros2 run direct_visual_lidar_calibration calibrate preprocessed
```

**可选参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--registration_type` | `nid_bfgs` | 优化器类型，可选 `nid_nelder_mead` |
| `--nid_bins` | 16 | NID 直方图 bin 数 |
| `--nelder_mead_init_step` | 0.001 | Nelder-Mead 初始步长 |
| `--disable_culling` | false | 禁用 Z-buffer 隐藏点去除 |
| `--background` | false | 无头后台运行 |

**读取初始值的优先级**：
1. 🥇 `init_T_lidar_camera` — 手动/外部注入
2. 🥈 `init_T_lidar_camera_auto` — RANSAC 自动估计
3. ❌ 都没有 → 报错

**观察收敛**：
- ✅ 正常：cost 从 ~0.98 明显下降到 0.5~0.7
- ❌ 异常：cost 几乎不动、反复出现 `Wolfe zoom failed`

**硬编码约束**（源码中不可配置）：
- 单步最大平移 0.2m、最大旋转 2°
- 外层最多 10 次迭代、内层最多 256 次
- 外层收敛阈值：平移 0.1m、旋转 0.5°

> ⚠️ **如果 NID 优化器"跑偏"**（初始值不错但结果更差），建议跳过 calibrate，直接用手动对齐结果。当场景纹理不足时，NID 代价函数几乎平坦，梯度无法正确引导优化方向。

### 4.6 查看结果

```bash
ros2 run direct_visual_lidar_calibration viewer preprocessed
```

GUI 中可切换查看：
- `INIT_GUESS (AUTO)` — RANSAC 结果
- `INIT_GUESS (MANUAL)` — 手动对齐结果
- `CALIBRATION_RESULT` — calibrate 优化结果

**对比判断**：选择点云边缘与图像物体边缘最吻合的那个作为最终结果。

---

## 五、产出与格式转换

### 5.1 calib.json 结构

```json
{
  "results": {
    "init_T_lidar_camera_auto": [tx, ty, tz, qx, qy, qz, qw],
    "init_T_lidar_camera":      [tx, ty, tz, qx, qy, qz, qw],
    "T_lidar_camera":           [tx, ty, tz, qx, qy, qz, qw]
  }
}
```

**TUM 格式**：`[tx, ty, tz, qx, qy, qz, qw]` — 平移在前，四元数（qw 实部）在最后。

### 5.2 选择最终结果

根据 viewer 的目视判断，确定使用哪个字段的值：

| 来源 | 字段 | 何时使用 |
|------|------|----------|
| 手动对齐 | `init_T_lidar_camera` | viewer 中 MANUAL 效果最好 |
| calibrate | `T_lidar_camera` | viewer 中 RESULT 效果最好 |
| 注入的旧值 | `init_T_lidar_camera` | 跳过 calibrate 直接复用 |

### 5.3 转换为 T_camera_lidar
访问这个网站，粘贴四元数，依次点击TUM..., Inverse。得到目标矩阵。
https://staff.aist.go.jp/k.koide/workspace/matrix_converter/matrix_converter.html


### 5.4 写入 converter_config.yaml

将上一步输出的 R 和 T 填入 `hnurm_radar/configs/converter_config.yaml`：

```yaml
calib:
  extrinsic:
    R:
      rows: 3
      cols: 3
      dt: d
      data: [R11, R12, R13, R21, R22, R23, R31, R32, R33]  # 行优先
    T:
      rows: 3
      cols: 1
      dt: d
      data: [t1, t2, t3]
```

### 5.5 写入 lidar_camera.txt

同一个 T_camera_lidar 的 4×4 矩阵也写入：

```
/data/projects/radar/lidar_camera_calib_utils/parameters/lidar_camera.txt
```

格式：空格分隔的 4×4 矩阵（含最后一行 `0 0 0 1`）。

---

## 六、完整数据流

```
录制 rosbag (livox/)
      │
      ▼  preprocess -av
preprocessed/calib.json  (内参 + 元数据)
      │
      ▼  initial_guess_manual / auto / 注入旧值
calib.json.results.init_T_lidar_camera  (T_lidar_camera, TUM格式)
      │
      ▼  calibrate (可选，场景纹理足够时)
calib.json.results.T_lidar_camera  (T_lidar_camera, TUM格式)
      │
      ▼  viewer 目视对比 → 选最好的
      │
      ▼  Python inv() 求逆
T_camera_lidar 4×4 矩阵
      │
      ├──▶ hnurm_radar/configs/converter_config.yaml  (拆成 R + T)
      └──▶ lidar_camera_calib_utils/parameters/lidar_camera.txt  (完整 4×4)
```

---

## 七、验证

### 7.1 viewer 投影检查

在 calibrate/initial_guess 之后，直接用 viewer 查看。

### 7.2 crossValidation.py

```bash
cd /data/projects/radar/lidar_camera_calib_utils
python3 crossValidation.py
```

> ⚠️ 该工具对平移误差不敏感（齐次除法后平移被归一化）。主要验证旋转正确性。

### 7.3 运行 hnurm_radar

最终验证：直接运行雷达站程序，观察点云在图像上的投影对齐效果。

---

## 八、何时需要重新标定？

| 情况 | 需要？ | 原因 |
|------|:---:|------|
| 相机/雷达安装位置或角度变了 | ✅ | 相对位姿改变 |
| 换了相机或雷达硬件 | ✅ | 坐标系原点不同 |
| 传感器拆下重新安装 | ✅ | 拆装无法保证一致 |
| 整体搬动（传感器未拆） | ❌ | 相对位姿不变 |
| 换了场地但传感器没动 | ❌ | 外参只取决于安装关系 |
| 重新标定了内参 | ❌ | 内参外参互相独立 |

---

## 九、常见问题

### Q: RANSAC 内点数很低（< 10%）怎么办？

1. **换场景重录**：选择纹理丰富、边缘清晰的场景
2. **改用手动对齐**：`initial_guess_manual` 最可靠
3. **注入旧值**：传感器没拆过就直接复用

### Q: calibrate 的 cost 不下降？

初始值太差或场景纹理不足。方案：
- 先用 `viewer` 检查初始值投影质量
- 换用 `--registration_type nid_nelder_mead`（无导数方法，更稳健）
- 减小步长 `--nelder_mead_init_step 0.0005`
- 如果手动对齐已经够好，跳过 calibrate

### Q: calibrate 从好的初始值"跑偏"了？

这说明场景的 NID 代价函数几乎平坦，梯度无法引导优化。直接使用初始值（手动对齐结果），不要 calibrate。

### Q: TUM 格式是什么顺序？

`[tx, ty, tz, qx, qy, qz, qw]` — 平移 3 个 + 四元数 4 个（qw 实部在最后）。

### Q: T_camera_lidar 矩阵怎么快速检查对不对？

检查旋转矩阵的列向量映射（第三行第一列应接近 1，表示雷达 X→相机 Z）：
```
第 3 行 ≈ [1, 0, 0, *]   → 雷达X(前) 映射到相机Z(前)   ✓
第 1 行 ≈ [0, -1, 0, *]  → 雷达Y(左) 映射到相机-X(左)  ✓
第 2 行 ≈ [0, 0, -1, *]  → 雷达Z(上) 映射到相机-Y(上)  ✓
```
平移列（第 4 列）每个分量应 < 0.15m。

### Q: 标定结果的平移值异常大？

每个分量 > 0.3m 基本说明标定有问题。检查初始值是否合理。

---

## 十、目录结构

```
direct_visual_lidar_calibration_ws/
├── src/direct_visual_lidar_calibration/   # 标定算法源码
├── livox/                                  # 录制的 rosbag
│   └── rosbag2_YYYY_MM_DD-HH_MM_SS/
├── preprocessed/                           # 预处理输出 + 标定结果
│   ├── calib.json                         #   ← 标定结果
│   ├── *.ply / *.png                      #   点云 / 图像
│   └── backup/                            #   历史备份
├── docs/                                   # 文档
├── build/ install/ log/
```
