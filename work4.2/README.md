# Blinn-Phong Shading Demo with Hard Shadows

基于 Taichi 实现的 Blinn-Phong 光照模型演示程序，支持实时交互调节材质参数。

## 功能特性

### 1. Blinn-Phong 光照模型
- **环境光 (Ambient)**：模拟全局光照，使物体在阴影中也能可见
- **漫反射 (Diffuse)**：模拟光线在粗糙表面的均匀散射
- **镜面反射 (Specular)**：使用半程向量 H 计算高光，产生更平滑的高光效果

### 2. 硬阴影 (Hard Shadow)
- 实现暗影射线算法，准确计算物体间的遮挡关系
- 阴影区域仅显示环境光，形成明显的明暗分界

### 3. 交互控制面板
- **Ka**：环境光系数 (0.0 - 1.0)
- **Kd**：漫反射系数 (0.0 - 1.0)
- **Ks**：镜面反射系数 (0.0 - 1.0)
- **N**：高光指数 (1.0 - 128.0)，控制高光区域大小

## 场景说明

- **红球**：位于场景左侧，展示球体的光照效果
- **紫色圆锥**：位于场景右侧，展示圆锥体的光照效果
- **点光源**：位于场景右上方，产生真实的阴影投射

## Phong vs Blinn-Phong 对比

| 特性 | Phong 模型 | Blinn-Phong 模型 |
|------|-----------|------------------|
| 高光计算 | 使用反射向量 R | 使用半程向量 H |
| 高光边缘 | 较锐利 | 更柔和平滑 |
| 大入射角 | 高光较窄 | 高光更宽 |
| 计算效率 | 较低（需计算反射） | 较高（仅需归一化） |

## 运行方式

```bash
# 安装依赖
pip install taichi

# 运行程序
python phong_shading.py
```

## 技术实现

### Blinn-Phong 高光公式
```
H = normalize(L + V)
specular = Ks * (N · H)^shininess * light_color
```

### 硬阴影算法
1. 从交点向光源发射暗影射线
2. 检测射线与场景中其他物体的相交
3. 如果交点到光源之间有遮挡，则该点处于阴影中

## 注意事项

- 所有参与点乘的向量（N、L、V、H）均已归一化
- 使用 `ti.max(0.0, dot_product)` 处理背面光照
- 最终颜色使用 `ti.math.clamp(color, 0.0, 1.0)` 限制在合法范围

## 效果展示

程序运行后将显示一个 800x600 的窗口，包含：
- 左侧红色球体和右侧紫色圆锥
- 右上角材质参数调节面板
- 实时渲染的光照和阴影效果

## 参考资料

- Blinn-Phong 光照模型：https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_reflection_model
- Taichi 编程指南：https://docs.taichi-lang.org/