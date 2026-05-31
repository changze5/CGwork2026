# 3D 变换演示 (Taichi)

基于 Taichi 的 3D 图形变换演示程序，实现 MVP（Model-View-Projection）变换矩阵的完整流程。

## 功能特性

- 模型变换：绕 Z 轴旋转
- 视图变换：相机位置调整
- 透视投影：将 3D 坐标投影到 2D 屏幕

## 环境配置

### 推荐方法（使用 Conda）

```bash
# 创建并激活环境
conda create -n cg_env python=3.12 -y
conda activate cg_env

# 安装 Taichi
pip install taichi
```

### 直接安装

```bash
# 确保使用 Python 3.8-3.13
pip install taichi
```

## 运行方法

```bash
python 3d_transformation.py
```

## 操作说明

| 按键 | 功能 |
|------|------|
| A | 顺时针旋转 10° |
| D | 逆时针旋转 10° |
| Esc | 退出程序 |

## 项目结构

```
work2/
├── 3d_transformation.py    # 主程序
├── README.md               # 项目说明
├── python-3.12.5-amd64.exe # Python 安装程序
└── 效果展示.gif             # 演示效果
```

## 技术实现

程序实现了完整的 MVP 变换流程：

1. **Model 矩阵**：绕 Z 轴旋转
2. **View 矩阵**：将相机移动到原点
3. **Projection 矩阵**：透视投影 + 正交投影组合

三角形顶点经过 MVP 变换后，通过透视除法转换到 NDC 坐标，最后映射到屏幕空间进行渲染。
