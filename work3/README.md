# 贝塞尔曲线交互式绘制程序

## 项目简介

本项目实现了一个基于 Python + Taichi 的贝塞尔曲线交互式绘制程序，用于计算机图形学实验教学。

## 功能特性

- ✅ 实现 De Casteljau 算法计算贝塞尔曲线
- ✅ 支持鼠标交互添加控制点
- ✅ GPU/CPU 加速渲染，流畅显示
- ✅ 实时曲线绘制与更新

## 数学原理

### De Casteljau 算法

贝塞尔曲线由一组控制点决定。对于给定参数 \( t \in [0, 1] \)，算法通过递归线性插值计算曲线上的点：

1. **第一层插值**：对相邻控制点 \( P_i \) 和 \( P_{i+1} \)，计算 \( P'_i = (1-t)P_i + tP_{i+1} \)
2. **递归**：对新生成的点重复上述操作
3. **终止**：当只剩一个点时，即为贝塞尔曲线在参数 \( t \) 处的坐标

## 技术实现

### 性能优化策略

1. **GPU 缓冲区预分配**：使用 `ti.Vector.field` 在 GPU 显存中预分配曲线点缓冲区
2. **批量数据传输**：CPU 计算完成后一次性将所有曲线点发送到 GPU
3. **并行像素绘制**：使用 `@ti.kernel` 装饰器实现 GPU 并行像素渲染

### 光栅化过程

1. 创建 800x800 的像素缓冲区模拟屏幕显存
2. 将归一化坐标 \([0,1]\) 映射到像素索引 \([0, 799]\)
3. 直接操作像素缓冲区点亮对应像素

## 环境配置

```bash
# 创建 Conda 环境
conda create -n cg_env python=3.12 -y

# 激活环境
conda activate cg_env

# 安装依赖
pip install taichi numpy
```

## 运行方式

```bash
cd work3
python bezier_curve.py
```

## 交互操作

| 操作 | 说明 |
|------|------|
| **鼠标左键点击** | 添加红色控制点 |
| **按 c 键** | 清空所有控制点 |

## 效果展示

### 运行截图

程序运行时会弹出一个 800x800 的窗口：

- **红色圆点**：用户添加的控制点
- **灰色线段**：控制点之间的连接线
- **绿色曲线**：根据控制点计算出的贝塞尔曲线

### 示例效果

当用户添加 3 个控制点时，程序会绘制一条二次贝塞尔曲线；添加更多控制点时，会绘制更高阶的贝塞尔曲线。曲线会随着控制点的位置实时更新。

## 文件结构

```
work3/
├── bezier_curve.py    # 主程序文件
└── README.md          # 项目说明文档
```

## 核心代码

### De Casteljau 算法实现

```python
def de_casteljau(points, t):
    if len(points) == 1:
        return points[0]
    next_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i+1]
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        next_points.append([x, y])
    return de_casteljau(next_points, t)
```

### GPU 绘制内核

```python
@ti.kernel
def draw_curve_kernel(n: ti.i32):
    for i in range(n):
        pt = curve_points_field[i]
        x_pixel = ti.cast(pt[0] * WIDTH, ti.i32)
        y_pixel = ti.cast(pt[1] * HEIGHT, ti.i32)
        if 0 <= x_pixel < WIDTH and 0 <= y_pixel < HEIGHT:
            pixels[x_pixel, y_pixel] = ti.Vector([0.0, 1.0, 0.0])
```

## 实验目标

1. 理解贝塞尔曲线的几何意义
2. 掌握 De Casteljau 算法的实现
3. 理解光栅化的基础概念
4. 掌握图形界面的交互事件处理

## 许可证

MIT License
