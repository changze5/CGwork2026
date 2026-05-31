import taichi as ti
import math
import os
from PIL import Image

os.environ['TAICHI_CACHE_DIR'] = 'e:/code1/work6/.taichi_cache'
ti.init(arch=ti.cpu, offline_cache=False, debug=False, fast_math=True)

res = 256

target_pixels = ti.field(dtype=ti.f32, shape=(res, res))
display_pixels = ti.field(dtype=ti.f32, shape=(res * 2, res))

loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

light_pos = ti.Vector.field(3, dtype=ti.f32, shape=(), needs_grad=True)

sphere_center = ti.Vector([0.5, 0.5, 0.5])
sphere_radius = 0.3
TARGET_LIGHT = [0.8, 0.8, 0.2]


@ti.kernel
def generate_target():
    for i, j in target_pixels:
        x = (i + 0.5) / res
        y = (j + 0.5) / res
        dx = x - sphere_center[0]
        dy = y - sphere_center[1]
        dist_sq = dx**2 + dy**2

        if dist_sq < sphere_radius**2:
            dz = ti.sqrt(sphere_radius**2 - dist_sq)
            z = sphere_center[2] - dz
            p = ti.Vector([x, y, z])
            n = (p - sphere_center).normalized()
            
            target_light_vec = ti.Vector(TARGET_LIGHT)
            l_dir = (target_light_vec - p).normalized()

            dot_val = n.dot(l_dir)
            intensity = ti.max(0.0, ti.min(1.0, dot_val))
            target_pixels[i, j] = intensity
        else:
            target_pixels[i, j] = 0.0


@ti.kernel
def render_and_compute_loss():
    for i, j in target_pixels:
        x = (i + 0.5) / res
        y = (j + 0.5) / res
        dx = x - sphere_center[0]
        dy = y - sphere_center[1]
        dist_sq = dx**2 + dy**2

        intensity = 0.0
        if dist_sq < sphere_radius**2:
            dz = ti.sqrt(sphere_radius**2 - dist_sq)
            z = sphere_center[2] - dz
            p = ti.Vector([x, y, z])
            n = (p - sphere_center).normalized()
            l_dir = (light_pos[None] - p).normalized()

            dot_val = n.dot(l_dir)
            intensity = ti.max(0.1 * dot_val, dot_val)
        
        diff = intensity - target_pixels[i, j]
        loss[None] += (1.0 / (res * res)) * (diff ** 2)
        
        display_pixels[i, j] = target_pixels[i, j]
        display_pixels[i + res, j] = ti.max(0.0, ti.min(1.0, intensity))


def main():
    generate_target()
    
    light_pos[None] = [0.2, 0.2, 0.8]  
    
    m = [0.0, 0.0, 0.0]
    v = [0.0, 0.0, 0.0]
    beta1 = 0.9
    beta2 = 0.999
    lr = 0.02
    eps = 1e-8

    print("=" * 80)
    print("任务1-3: 基本可微渲染 - 光源位置优化")
    print("=" * 80)
    print(f"目标光源位置: {TARGET_LIGHT}")
    print(f"初始光源位置: [{light_pos[None][0]:.3f}, {light_pos[None][1]:.3f}, {light_pos[None][2]:.3f}]")
    print("-" * 80)
    
    os.makedirs('e:/code1/work6/output', exist_ok=True)
    img_target = Image.fromarray((target_pixels.to_numpy() * 255).astype('uint8'), mode='L')
    img_target.save('e:/code1/work6/output/target_image.png')
    print("已保存目标图像: output/target_image.png")
    
    gui = None
    try:
        gui = ti.GUI("可微渲染 - 任务4: 实时可视化\n左: 目标图像 | 右: 当前渲染", res=(res * 2, res))
    except Exception as e:
        print(f"GUI不可用，仅保存图像文件")
    
    for iter in range(1, 301):
        loss[None] = 0.0
        
        with ti.ad.Tape(loss=loss):
            render_and_compute_loss()
        
        grad = light_pos.grad[None]

        for c in range(3):
            m[c] = beta1 * m[c] + (1 - beta1) * grad[c]
            v[c] = beta2 * v[c] + (1 - beta2) * grad[c] * grad[c]
            
            m_hat = m[c] / (1 - beta1**iter)
            v_hat = v[c] / (1 - beta2**iter)
            
            light_pos[None][c] -= lr * m_hat / (math.sqrt(v_hat) + eps)

        if iter % 10 == 0:
            pos = light_pos[None]
            print(f"Iter {iter:03d} | Loss: {loss[None]:.6f} | Light Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        
        if iter % 30 == 0:
            img_array = display_pixels.to_numpy()
            img = Image.fromarray((img_array * 255).astype('uint8'), mode='L')
            img.save(f'e:/code1/work6/output/iter_{iter:03d}.png')
        
        if gui is not None:
            gui.set_image(display_pixels)
            gui.show()
    
    print("=" * 80)
    print("优化完成!")
    print(f"最终光源位置: [{light_pos[None][0]:.3f}, {light_pos[None][1]:.3f}, {light_pos[None][2]:.3f}]")
    print(f"最终Loss: {loss[None]:.6f}")
    print("=" * 80)
    
    img_final = Image.fromarray((display_pixels.to_numpy() * 255).astype('uint8'), mode='L')
    img_final.save('e:/code1/work6/output/final_result.png')
    print("已保存最终结果: output/final_result.png")


if __name__ == "__main__":
    main()
