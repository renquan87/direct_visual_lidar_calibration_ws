# 雷达-相机外参标定指南（direct_visual_lidar_calibration）

> 本文档面向 hnurm_radar 雷达站项目，详细解释外参标定的原理、流程、产出和应用。
> 
> 上游项目：https://github.com/koide3/direct_visual_lidar_calibration

---

## 一、什么是外参？

外参（Extrinsic Parameters）描述的是激光雷达和相机之间的空间关系——它们之间差了多少旋转和平移。

用数学表示就是一个 4×4 的齐次变换矩阵：

```
T = [ R  t ]    R: 3×3 旋转矩阵
    [ 0  1 ]    t: 3×1 平移向量
```

有两个方向：
- **T_lidar_camera**：从相机坐标系到激光雷达坐标系的变换（"相机看到的点，在雷达坐标系里是什么位置"）
- **T_camera_lidar**：从激光雷达坐标系到相机坐标系的变换（"雷达扫到的点，在相机坐标系里是什么位置"）

两者互为逆矩阵：`T_camera_lidar = T_lidar_camera⁻¹`

hnurm_radar 项目中 `converter_config.yaml` 里的 R 和 T 用的是 **T_camera_lidar**（雷达→相机），因为代码需要把雷达点云投影到相机图像上。

---

## 二、标定原理

`direct_visual_lidar_calibration` 的工作原理：

1. **数据采集**：同时录制相机图像和激光雷达点云
2. **预处理**：将多帧点云累积成一个稠密点云，提取关键帧图像
3. **初始估计**：用 SuperGlue（深度学习特征匹配）找到图像和点云强度图之间的对应点，通过 RANSAC 估计初始外参
4. **精细优化**：基于直接法（Direct Method），优化点云投影到图像上的光度一致性，迭代求解精确外参

这个方法的优点是不需要标定板，利用场景中的自然特征即可完成标定。

---

## 三、前置条件

- ✅ 相机内参已标定完成（`ost.yaml` 已生成）
- ✅ 相机和激光雷达已刚性固定，相对位置不会变
- ✅ 场景中有足够的纹理和结构特征（不要对着白墙标定）

---

## 四、完整操作流程

### 4.1 构建项目

```bash
cd /data/projects/radar/direct_visual_lidar_calibration_ws
rm -rf build/ log/ install/
colcon build --symlink-install
source install/setup.bash
```

### 4.2 启动传感器

需要三个终端：

**终端 1：启动激光雷达**
```bash
cd /data/projects/radar/hnurm_radar
source install/setup.bash
ros2 launch livox_ros_driver2 rviz_HAP_launch.py
```

**终端 2：启动相机**
```bash
cd /data/projects/radar/lidar_camera_calib_utils
source install/setup.bash
ros2 launch hnurm_camera hnurm_camera.launch.py
```

**终端 3：发布相机内参**
```bash
cd /data/projects/radar/lidar_camera_calib_utils
source install/setup.bash
python3 publish_camera_info.py
```

### 4.3 检查话题

```bash
ros2 topic list
# 应该能看到：
# /image          - 相机图像
# /livox/lidar    - 激光雷达点云
# /camera_info    - 相机内参信息
```

### 4.4 录制数据

```bash
cd /data/projects/radar/direct_visual_lidar_calibration_ws
source install/setup.bash
mkdir -p livox

# 录制数据（保持传感器静止，录制约 30 秒）
ros2 bag record /camera_info /image /livox/lidar -o livox/rosbag2_$(date +%Y_%m_%d-%H_%M_%S)
```

> 录制时保持相机和雷达静止不动。场景中应有丰富的结构特征（桌椅、墙角、柱子等）。

### 4.5 数据预处理

```bash
cd /data/projects/radar/direct_visual_lidar_calibration_ws
source install/setup.bash

# 预处理（-a 自动检测话题，-v 可视化）
ros2 run direct_visual_lidar_calibration preprocess livox preprocessed -av
```

预处理会生成 `preprocessed/` 目录，包含：
- 累积的稠密点云（`.ply`）
- 关键帧图像（`.png`）
- 激光雷达强度图和索引图
- 元数据文件（`calib.json`）

### 4.6 初始变换估计

```bash
# 用 SuperGlue 找对应点
ros2 run direct_visual_lidar_calibration find_matches_superglue.py preprocessed

# 基于 RANSAC 估计初始外参
ros2 run direct_visual_lidar_calibration initial_guess_auto preprocessed
```

### 4.7 精细标定

```bash
ros2 run direct_visual_lidar_calibration calibrate preprocessed
```

标定完成后 `Ctrl+C` 保存结果。

---

## 五、产出与转换

### 5.1 标定产出

结果保存在 `preprocessed/calib.json`，其中包含：

```json
{
  "T_lidar_camera": [qx, qy, qz, qw, tx, ty, tz]
}
```

这是四元数 + 平移的格式（从相机到雷达的变换）。

### 5.2 四元数转矩阵

calib.json 中只有四元数，需要转成 4×4 矩阵：

1. 访问在线转换工具：https://staff.aist.go.jp/k.koide/workspace/matrix_converter/matrix_converter.html
2. 将 `T_lidar_camera` 的 7 个数值填入
3. 得到 4×4 的 `T_lidar_camera` 矩阵

### 5.3 求逆矩阵

hnurm_radar 需要的是 **T_camera_lidar**（雷达→相机），所以要对 T_lidar_camera 求逆：

在同一个在线工具中点击 "Inverse" 按钮，得到 `T_camera_lidar` 的 4×4 矩阵。

### 5.4 写入参数文件

将求逆后的 4×4 矩阵写入：

```
/data/projects/radar/lidar_camera_calib_utils/parameters/lidar_camera.txt
```

格式示例（当前值）：

```
0.01914   -0.99981  0.00462   -0.05301
0.00132   -0.00459  -0.99999  0.04890
0.99982   0.01915   0.00123   -0.07717
0.00000   0.00000   0.00000   1.00000
```

其中：
- 左上 3×3 是旋转矩阵 R
- 右侧 3×1 是平移向量 T（单位：米）
- 最后一行固定为 `0 0 0 1`

### 5.5 应用到 hnurm_radar

将 `lidar_camera.txt` 中的数值填入 `hnurm_radar/configs/converter_config.yaml`：

```yaml
calib:
  extrinsic:
    R:
      rows: 3
      cols: 3
      dt: d
      data: [0.01914, -0.99981, 0.00462,     # ← 矩阵第 1 行前 3 个
             0.00132, -0.00459, -0.99999,     # ← 矩阵第 2 行前 3 个
             0.99982, 0.01915, 0.00123]       # ← 矩阵第 3 行前 3 个
    T:
      rows: 3
      cols: 1
      dt: d
      data: [-0.05301, 0.04890, -0.07717]    # ← 矩阵第 1/2/3 行第 4 个
```

---

## 六、完整数据流总结

```
direct_visual_lidar_calibration_ws 标定
       │
       ▼
preprocessed/calib.json
(T_lidar_camera 四元数: [qx, qy, qz, qw, tx, ty, tz])
       │
       ▼  在线工具转换
T_lidar_camera 4×4 矩阵
       │
       ▼  求逆 (Inverse)
T_camera_lidar 4×4 矩阵
       │
       ▼  手动写入
lidar_camera_calib_utils/parameters/lidar_camera.txt
       │
       ▼  手动提取 R 和 T 填入
hnurm_radar/configs/converter_config.yaml → extrinsic.R 和 extrinsic.T
       │
       ▼  代码运行时读取
radar.py → PointCloudConverter 用 R、T 将点云投影到图像
```

---

## 七、验证标定结果

标定完成后，回到 `lidar_camera_calib_utils` 运行验证：

```bash
cd /data/projects/radar/lidar_camera_calib_utils
source install/setup.bash

# 交叉验证：将点云投影到图像上，检查是否对齐
python3 crossValidation.py

# PnP 验证
python3 pnp_demo.py
```

如果点云投影和图像中的物体边缘对齐良好，说明标定成功。

---

## 八、什么时候需要重新做外参标定？

| 情况 | 需要重新标定？ |
|------|:---:|
| 相机或雷达的安装位置/角度变了 | ✅ |
| 换了相机或雷达硬件 | ✅ |
| 从赛场搬到实验室，重新安装了传感器 | ✅ |
| 传感器刚性固定在支架上，整体搬动 | ❌ |
| 换了场地但传感器没动 | ❌ |
| 换了 YOLO 模型 | ❌ |
| 改了软件配置参数 | ❌ |

**关键原则**：只要相机和雷达之间的相对位置没变，就不需要重新标定外参。

---

## 九、目录结构

```
direct_visual_lidar_calibration_ws/
├── src/
│   └── direct_visual_lidar_calibration/   # 标定算法源码
├── livox/                                  # 录制的 rosbag 数据
├── preprocessed/                           # 预处理后的数据 + calib.json 结果
├── lidar_camera_calibration_data/          # 标定数据存档
├── build/                                  # 编译产物
├── install/                                # 安装产物
└── log/                                    # 日志
```

---

## 十、常见问题

**Q: SuperGlue 找不到足够的匹配点？**
A: 场景纹理不够丰富。换一个有更多结构特征的场景重新录制数据。避免对着白墙或空旷区域。

**Q: 初始估计明显不对（点云投影完全偏移）？**
A: 可以尝试手动初始估计。参考 direct_visual_lidar_calibration 的文档使用 `initial_guess_manual` 工具。

**Q: 标定结果的精度如何判断？**
A: 运行 `crossValidation.py`，观察点云投影到图像上是否与物体边缘对齐。误差应在几个像素以内。

**Q: 为什么 calib.json 用四元数而不是矩阵？**
A: 四元数是一种紧凑的旋转表示（4 个数 vs 矩阵的 9 个数），且没有万向锁问题。这是 direct_visual_lidar_calibration 的默认输出格式。
