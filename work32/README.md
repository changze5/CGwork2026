# 反走样贝塞尔曲线与 B 样条曲线绘制

本项目实现了具有平滑边缘的反走样贝塞尔曲线和均匀三次 B 样条曲线的绘制功能。

## 功能特性

### 反走样渲染
- 使用 3×3 像素邻域分析
- 基于距离衰减模型的权重计算
- 立方权重函数实现平滑颜色混合
- 亚像素级抗锯齿效果

### B 样条曲线
- 均匀三次 B 样条曲线实现
- 分段多项式基函数
- 局部控制特性（修改一个控制点只影响曲线的一小段）
- 支持无限增加控制点而不增加阶数

### 交互功能
- 左键点击添加控制点
- B 键切换曲线模式（贝塞尔/B样条）
- C 键清除画布

## 文件说明

| 文件 | 描述 | 依赖 |
|------|------|------|
| `bezier_bspline.html` | HTML/JavaScript 版本 | 浏览器（推荐） |
| `bezier_bspline_antialiased.py` | Taichi GPU 加速版本 | Taichi |
| `bezier_bspline_pygame.py` | Pygame 版本 | Pygame, NumPy |

## 使用方法

### HTML 版本（推荐）
```bash
# 直接在浏览器中打开
start bezier_bspline.html
```

### Taichi 版本
```bash
# 安装依赖
pip install taichi numpy

# 运行
python bezier_bspline_antialiased.py
```

### Pygame 版本
```bash
# 安装依赖
pip install pygame numpy

# 运行
python bezier_bspline_pygame.py
```

## 操作说明

1. **添加控制点**：使用鼠标左键点击画布
2. **切换模式**：按 `B` 键在贝塞尔曲线和 B 样条曲线之间切换
3. **清除画布**：按 `C` 键清除所有控制点

## 两种曲线对比

| 特性 | 贝塞尔曲线 | B样条曲线 |
|------|-----------|----------|
| 控制特性 | 全局控制 | 局部控制 |
| 控制点关系 | 所有控制点影响整条曲线 | 每4个控制点影响一段 |
| 移动控制点 | 整条曲线改变 | 只影响局部段 |
| 数学基础 | De Casteljau 算法 | Cox-de Boor 递归公式 |

## 反走样算法原理

对于曲线上的每个精确浮点坐标 (x, y)，考察其 3×3 邻域内的像素：

1. 计算每个像素中心点与精确坐标的欧氏距离
2. 使用距离衰减函数计算权重：`weight = max(0, 1 - distance / 1.2)^3`
3. 根据权重混合颜色，实现平滑的边缘过渡

## 技术栈

- HTML5 Canvas / JavaScript
- Taichi（GPU 加速）
- Pygame
- NumPy

## 许可证

MIT License
