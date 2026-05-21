# CG-Lab: Taichi Gravity Swarm

基于 Taichi 的 GPU 粒子物理模拟实验项目，实现鼠标引力交互效果。

## 项目架构

```
wrok1/
├── pyproject.toml          # uv 项目配置文件
├── .gitignore              # Git 忽略配置
└── src/
    └── Work0/              # 实验零专属包
        ├── __init__.py     # Python 包标识
        ├── config.py       # 参数配置中心
        ├── physics.py      # GPU 并行计算与物理逻辑
        └── main.py         # 程序入口与 GUI 渲染
```

## 代码逻辑

### 1. config.py - 参数配置中心

集中管理所有可调参数，便于快速调整：
- **物理系统参数**：粒子总数、引力强度、空气阻力系数、边界反弹系数
- **渲染系统参数**：窗口分辨率、粒子半径、粒子颜色

### 2. physics.py - GPU 核心逻辑

包含 Taichi 数据结构定义和 `@ti.kernel` 修饰的并行算子：
- `pos` / `vel`: GPU 显存中的粒子位置和速度向量场
- `init_particles()`: 初始化粒子随机位置
- `update_particles()`: GPU 并行执行的物理更新（引力计算、阻力、边界碰撞）

### 3. main.py - 程序入口

负责初始化环境、读取用户输入、渲染画面：
- 初始化 Taichi GPU 环境
- 创建 GUI 窗口
- 主循环：读取鼠标位置 → 驱动 GPU 计算 → 绘制粒子 → 显示画面

## 实现功能

- **GPU 并行计算**：利用 Taichi 在 GPU 上并行更新数万粒子
- **鼠标引力交互**：粒子会被鼠标位置吸引
- **物理模拟**：包含引力、空气阻力、边界反弹效果
- **实时渲染**：流畅的粒子群动画展示

## 运行方式

确保终端当前路径在项目根目录，执行：

```bash
uv run -m src.Work0.main
```

## 效果展示

粒子群会跟随鼠标移动，呈现引力吸引效果：

![粒子引力效果](https://via.placeholder.com/800x450?text=Taichi+Gravity+Swarm+Demo)

**交互说明**：运行程序后，在弹出的窗口中移动鼠标，粒子会被鼠标吸引并跟随移动。

## 性能优化建议

如果运行时卡顿，可在 `src/Work0/config.py` 中调整参数：

| 参数 | 说明 | 默认值 | 建议范围 |
|------|------|--------|----------|
| NUM_PARTICLES | 粒子总数 | 10000 | 2000-20000 |
| GRAVITY_STRENGTH | 引力强度 | 0.001 | 0.0005-0.005 |
| DRAG_COEF | 空气阻力 | 0.98 | 0.95-0.99 |

## 技术栈

- Python 3.13+
- Taichi 1.7.4
- uv (Python 包管理器)

## 许可证

MIT License