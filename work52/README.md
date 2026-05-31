# Ray Tracing Demo - Work 5.2

## 项目简介

这是一个基于 Taichi 的光线追踪（Ray Tracing）演示程序，实现了折射与玻璃材质以及 MSAA 抗锯齿功能。

## 功能特性

### 1. 折射与玻璃材质 (+15%)
- 实现了斯涅尔定律（Snell's Law）计算折射光线方向
- 支持玻璃材质的折射效果
- 处理全内反射（Total Internal Reflection）情况
- 使用 Schlick 菲涅尔近似实现真实的菲涅尔反射效果

### 2. MSAA 抗锯齿 (+10%)
- 在每个像素内进行多次随机采样
- 对多次采样的颜色进行平均，实现平滑的边缘过渡
- 可调节的采样次数（1-8次）

## 场景组成

- **左侧玻璃球**：应用折射和菲涅尔反射效果
- **右侧镜面球**：纯反射材质
- **棋盘格地板**：漫反射材质，带阴影效果
- **可调光源**：支持实时调整位置

## 运行要求

- Python 3.12
- Taichi >= 1.7.0
- 支持 Vulkan 或 CUDA 的 GPU（程序会自动选择可用后端）

## 安装依赖

```bash
pip install taichi
```

## 运行方式

```bash
py -3.12 ray_tracing_glass.py
```

或

```bash
python ray_tracing_glass.py
```

## 交互控制

程序运行后会显示控制面板，可调节以下参数：

| 参数 | 说明 | 范围 |
|------|------|------|
| Light X | 光源 X 坐标 | -5.0 ~ 5.0 |
| Light Y | 光源 Y 坐标 | 1.0 ~ 8.0 |
| Light Z | 光源 Z 坐标 | -5.0 ~ 5.0 |
| Max Bounces | 光线最大弹射次数 | 1 ~ 8 |
| Glass IOR | 玻璃折射率 | 1.0 ~ 2.5 |
| MSAA Samples | 抗锯齿采样次数 | 1 ~ 8 |

### 常见折射率参考

- 空气：1.0
- 水：1.33
- 普通玻璃：1.5
- 重火石玻璃：1.7
- 钻石：2.42

## 技术实现

### 斯涅尔定律

$$n_1 \sin(\theta_1) = n_2 \sin(\theta_2)$$

折射光线的计算公式：

$$\mathbf{t} = \eta \mathbf{i} + (\eta \cos\theta_i - \cos\theta_t) \mathbf{n}$$

### Schlick 菲涅尔近似

$$R(\theta) = R_0 + (1 - R_0)(1 - \cos\theta)^5$$

其中 $R_0 = \left(\frac{n_1 - n_2}{n_1 + n_2}\right)^2$

### MSAA 抗锯齿

在每个像素内发射多条带随机偏移的射线，对结果进行平均：

$$C_{final} = \frac{1}{N} \sum_{i=1}^{N} C_i$$

## 项目结构

```
ray_tracing_glass.py  - 主程序文件
README.md              - 项目说明文档
```

## 效果展示

运行程序后，将看到：
1. 玻璃球的折射效果（可透过玻璃看到扭曲的背景和其他物体）
2. 光滑无锯齿的物体边缘
3. 真实的阴影效果
4. 镜面反射

## 作者

基于 Taichi 官方教程扩展开发

## 许可证

MIT License
