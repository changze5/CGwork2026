# Cloth Simulation - Taichi GPU Accelerated

## 项目概述

基于 Taichi 的 GPU 加速布料模拟器，支持三种积分方法的实时切换。

## 环境要求

- Python 3.12
- Taichi >= 1.7.0
- NumPy

## 安装依赖

```bash
pip install taichi numpy
```

## 运行程序

```bash
python work7.py
```

## 功能特性

### 任务1：场景初始化
- 20x20 网格布料模拟
- 分离式 `@ti.kernel` 初始化架构（位置、弹簧、渲染索引）
- 支持 GPU 同步

### 任务2：力学计算
- 重力、阻尼力、弹簧力计算
- `@ti.func` 内联优化
- 速度限制防爆处理

### 任务3：积分求解器
- **显式欧拉 (Explicit Euler)** - 基础但可能不稳定
- **半隐式欧拉 (Semi-implicit Euler)** - 速度先更新，更稳定
- **隐式迭代 (Implicit)** - 迭代优化，稳定性最好

### 任务4：交互控制
- 3D 可视化界面
- GUI 控制面板：
  - 实时切换积分方法
  - 暂停/继续模拟
  - 重置布料

## 操作说明

| 按钮 | 功能 |
|------|------|
| Explicit Euler | 显式欧拉积分 |
| Semi-implicit Euler | 半隐式欧拉积分（推荐） |
| Implicit | 隐式迭代积分 |
| Pause | 暂停/继续 |
| Reset | 重置布料 |

鼠标移动控制相机视角。

## 物理参数

| 参数 | 值 | 说明 |
|------|-----|------|
| N | 20 | 网格尺寸 |
| REST_LENGTH | 0.05 | 弹簧自然长度 |
| STIFFNESS | 500.0 | 弹簧刚度 |
| DAMPING | 0.99 | 阻尼系数 |
| GRAVITY | -9.8 | 重力加速度 |
| MAX_VELOCITY | 5.0 | 最大速度限制 |
| DT | 0.016 | 时间步长 |

## 文件结构

```
work7/
├── work7.py          # 主程序
├── 效果展示1.gif     # 运行效果截图1
├── 效果展示2.gif     # 运行效果截图2
└── README.md         # 说明文档
```

## 技术架构

- **初始化**：分离式 Kernel 设计，保证多线程 GPU 同步
- **计算**：ti.func 内联到调用 Kernel，减少函数调用开销
- **积分**：三种方法各有特点，可实时切换对比效果
- **渲染**：Taichi GGUI 3D 可视化

## 效果展示

见 `效果展示1.gif` 和 `效果展示2.gif`
