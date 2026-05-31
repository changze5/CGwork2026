import taichi as ti
import math
import os
from PIL import Image
import numpy as np

os.environ['TAICHI_CACHE_DIR'] = 'e:/code1/work6/.taichi_cache'
ti.init(arch=ti.cpu, offline_cache=False, debug=False, fast_math=True)

res = 256

target_pixels = ti.field(dtype=ti.f32, shape=(res, res))
display_pixels = ti.field(dtype=ti.f32, shape=(res * 2, res))

loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

light_pos = ti.Vector.field(3, dtype=ti.f32, shape=(), needs_grad=True)
diffuse_color = ti.Vector.field(3, dtype=ti.f32, shape=(), needs_grad=True)
shininess = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

sphere_center = ti.Vector([0.5, 0.5, 0.5])
sphere_radius = 0.3
TARGET_LIGHT = [0.8, 0.8, 0.2]
TARGET_COLOR = [0.8, 0.6, 0.3]
TARGET_SHININESS = 32.0

use_blinn_phong = True


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
            v_dir = (ti.Vector([0.5, 0.5, 1.0]) - p).normalized()
            intensity = 0.0
            
            if use_blinn_phong:
                h = (l_dir + v_dir).normalized()
                spec = ti.pow(ti.max(n.dot(h), 0.0), TARGET_SHININESS)
                intensity = n.dot(l_dir) * (1.0 + 0.5 * spec)
            else:
                intensity = n.dot(l_dir)
            
            intensity = ti.max(0.0, ti.min(1.0, intensity))
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
            v_dir = (ti.Vector([0.5, 0.5, 1.0]) - p).normalized()
            
            if use_blinn_phong:
                h = (l_dir + v_dir).normalized()
                spec = ti.pow(ti.max(n.dot(h), 0.0), shininess[None])
                dot_val = n.dot(l_dir)
                intensity = ti.max(0.1 * dot_val, dot_val) * (1.0 + 0.5 * spec)
            else:
                dot_val = n.dot(l_dir)
                intensity = ti.max(0.1 * dot_val, dot_val)
        
        diff = intensity - target_pixels[i, j]
        loss[None] += (1.0 / (res * res)) * (diff ** 2)
        
        display_pixels[i, j] = target_pixels[i, j]
        display_pixels[i + res, j] = ti.max(0.0, ti.min(1.0, intensity))


def adam_optimizer(param, grad, m, v, iter_num, lr=0.02, beta1=0.9, beta2=0.999, eps=1e-8, is_vector=False):
    if is_vector:
        grad_val = grad[None]
        for c in range(3):
            m[c] = beta1 * m[c] + (1 - beta1) * grad_val[c]
            v[c] = beta2 * v[c] + (1 - beta2) * grad_val[c] * grad_val[c]
            m_hat = m[c] / (1 - beta1**iter_num)
            v_hat = v[c] / (1 - beta2**iter_num)
            param[None][c] -= lr * m_hat / (math.sqrt(v_hat) + eps)
    else:
        m[0] = beta1 * m[0] + (1 - beta1) * grad[None]
        v[0] = beta2 * v[0] + (1 - beta2) * grad[None] * grad[None]
        m_hat = m[0] / (1 - beta1**iter_num)
        v_hat = v[0] / (1 - beta2**iter_num)
        param[None] -= lr * m_hat / (math.sqrt(v_hat) + eps)


def main():
    generate_target()
    
    light_pos[None] = [0.2, 0.2, 0.8]  
    diffuse_color[None] = [0.3, 0.3, 0.8]
    shininess[None] = 8.0
    
    m_light = [0.0, 0.0, 0.0]
    v_light = [0.0, 0.0, 0.0]
    m_color = [0.0, 0.0, 0.0]
    v_color = [0.0, 0.0, 0.0]
    m_shininess = [0.0]
    v_shininess = [0.0]
    
    print("=" * 80)
    print("Differentiable Rendering - Multi-Parameter Optimization")
    print("=" * 80)
    print(f"Target Light Position: {TARGET_LIGHT}")
    print(f"Target Diffuse Color: {TARGET_COLOR}")
    print(f"Target Shininess: {TARGET_SHININESS}")
    print(f"Initial Light Position: [{light_pos[None][0]:.3f}, {light_pos[None][1]:.3f}, {light_pos[None][2]:.3f}]")
    print(f"Initial Diffuse Color: [{diffuse_color[None][0]:.3f}, {diffuse_color[None][1]:.3f}, {diffuse_color[None][2]:.3f}]")
    print(f"Initial Shininess: {shininess[None]:.3f}")
    print("-" * 80)
    
    os.makedirs('e:/code1/work6/output', exist_ok=True)
    img_target = Image.fromarray((target_pixels.to_numpy() * 255).astype('uint8'), mode='L')
    img_target.save('e:/code1/work6/output/target_image.png')
    print("Saved target image: output/target_image.png")
    
    gui = None
    try:
        gui = ti.GUI("Differentiable Rendering", res=(res * 2, res))
    except:
        print("GUI not available, saving images only")
    
    for iter in range(1, 301):
        loss[None] = 0.0
        
        with ti.ad.Tape(loss=loss):
            render_and_compute_loss()
        
        adam_optimizer(light_pos, light_pos.grad, m_light, v_light, iter, lr=0.02, is_vector=True)
        adam_optimizer(diffuse_color, diffuse_color.grad, m_color, v_color, iter, lr=0.1, is_vector=True)
        adam_optimizer(shininess, shininess.grad, m_shininess, v_shininess, iter, lr=1.0, is_vector=False)
        
        shininess[None] = ti.max(1.0, ti.min(100.0, shininess[None]))
        
        if iter % 10 == 0:
            pos = light_pos[None]
            col = diffuse_color[None]
            shin = shininess[None]
            print(f"Iter {iter:03d} | Loss: {loss[None]:.6f} | "
                  f"Light: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | "
                  f"Color: [{col[0]:.3f}, {col[1]:.3f}, {col[2]:.3f}] | "
                  f"Shininess: {shin:.3f}")
        
        if iter % 30 == 0:
            img_array = display_pixels.to_numpy()
            img = Image.fromarray((img_array * 255).astype('uint8'), mode='L')
            img.save(f'e:/code1/work6/output/iter_{iter:03d}.png')
        
        if gui is not None:
            gui.set_image(display_pixels)
            gui.show()
    
    print("=" * 80)
    print("Optimization Complete!")
    print(f"Final Light Position: [{light_pos[None][0]:.3f}, {light_pos[None][1]:.3f}, {light_pos[None][2]:.3f}]")
    print(f"Final Diffuse Color: [{diffuse_color[None][0]:.3f}, {diffuse_color[None][1]:.3f}, {diffuse_color[None][2]:.3f}]")
    print(f"Final Shininess: {shininess[None]:.3f}")
    print(f"Final Loss: {loss[None]:.6f}")
    print("=" * 80)
    
    img_final = Image.fromarray((display_pixels.to_numpy() * 255).astype('uint8'), mode='L')
    img_final.save('e:/code1/work6/output/final_result.png')
    print("Saved final result: output/final_result.png")


if __name__ == "__main__":
    main()
