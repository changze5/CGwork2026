# 可微渲染 (Differentiable Rendering) - 作业6

本项目使用 Taichi 实现了一个完整的可微渲染系统，包括场景构建、正向渲染、自动微分、参数优化等功能。

## 📁 项目结构

```
work6/
├── basic_differentiable_rendering.py    # 基础版本 - 仅光源优化 (任务1-4)
├── complete_differentiable_rendering.py # 完整版本 - Blinn-Phong + 多参数优化 (所有任务)
├── advanced_differentiable_rendering.py # 高级版本 - 扩展功能 (可选)
├── differentiable_rendering.py          # 初始版本
├── output/                              # 基础版本输出图像
│   ├── target_image.png
│   ├── final_result.png
│   └── iter_*.png
├── output_advanced/                     # 完整版本输出图像
│   ├── target_image.png
│   ├── final_result.png
│   └── iter_*.png
└── README.md
```

## 🚀 快速开始

### 环境要求
- Python 3.9-3.12
- Taichi 1.7.0+
- Pillow

### 安装依赖
```bash
pip install taichi pillow
```

### 运行程序

**基础版本（仅光源优化）**：
```bash
python basic_differentiable_rendering.py
```

**完整版本（Blinn-Phong + 多参数优化）**：
```bash
python complete_differentiable_rendering.py
```

## ✅ 已完成任务

### 任务1：场景构建与目标图像生成
- ✅ 半径 0.3，中心点 (0.5, 0.5, 0.5) 的三维球体
- ✅ 目标光源位置 (0.8, 0.8, 0.2)
- ✅ 正向渲染核函数生成 Ground Truth

### 任务2：可微渲染管线
- ✅ `light_pos` 声明为 `needs_grad=True` 的 Vector.field
- ✅ Leaky Lambertian 光照模型（0.1 泄漏系数）
- ✅ MSE Loss 计算，保留负值用于梯度传播

### 任务3：反向传播与参数更新
- ✅ 初始光源位置 (0.2, 0.2, 0.8) - 位于球体偏背面
- ✅ `ti.ad.Tape(loss)` 记录计算图
- ✅ Adam 优化器自动寻优

### 任务4：实时可视化
- ✅ GUI 窗口并排显示：左=目标图像，右=当前渲染
- ✅ 迭代过程图像自动保存

### 选做任务1：多参数联合优化
- ✅ 同时优化光源位置 + 漫反射强度参数
- ✅ 使用不同学习率避免梯度冲突

### 选做任务2：Blinn-Phong 光照模型
- ✅ 高光项：`spec = dot(n, h)^shininess`
- ✅ 漫反射 + 高光混合：`0.5 * diffuse + 0.3 * specular`
- ✅ Shininess 参数固定为 16.0

## 📊 优化结果

### 基础版本 (Lambertian)
- **初始光源**: [0.200, 0.200, 0.800]
- **最终光源**: [0.800, 0.800, 0.198]
- **最终 Loss**: 0.000356

### 完整版本 (Blinn-Phong)
- **初始光源**: [0.200, 0.200, 0.800]
- **最终光源**: [0.800, 0.800, 0.198]
- **最终 Loss**: 0.000089 (更低的 Loss)

## 🎯 技术亮点

1. **梯度保留机制**：Leaky Lambertian 保留负值，确保阴影区域也能获得梯度
2. **自适应学习率**：Adam 优化器自动调整每个参数的学习率
3. **数值稳定性**：梯度裁剪和参数边界约束避免 NaN
4. **实时反馈**：GUI 和保存图像双重可视化

## 📷 效果展示

程序运行后会生成以下图像：
- `output/target_image.png` - 目标图像
- `output/final_result.png` - 最终渲染结果
- `output/iter_*.png` - 迭代过程中间结果（每 30 步保存一张）

所有图像左侧为目标图像，右侧为当前渲染结果。

## 📝 作者与日期

- 日期: 2026年5月31日
