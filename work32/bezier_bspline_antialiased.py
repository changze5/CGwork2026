import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)

curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)

curve_mode = ti.field(dtype=ti.i32, shape=())
curve_mode[None] = 0

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

def cox_de_boor(i, k, t, knots):
    if k == 0:
        return 1.0 if knots[i] <= t < knots[i + 1] else 0.0
    result = 0.0
    denom1 = knots[i + k] - knots[i]
    if denom1 != 0:
        result += (t - knots[i]) / denom1 * cox_de_boor(i, k - 1, t, knots)
    denom2 = knots[i + k + 1] - knots[i + 1]
    if denom2 != 0:
        result += (knots[i + k + 1] - t) / denom2 * cox_de_boor(i + 1, k - 1, t, knots)
    return result

def uniform_cubic_bspline_basis(t):
    t2 = t * t
    t3 = t2 * t
    b0 = (-t3 + 3*t2 - 3*t + 1) / 6.0
    b1 = (3*t3 - 6*t2 + 4) / 6.0
    b2 = (-3*t3 + 3*t2 + 3*t + 1) / 6.0
    b3 = t3 / 6.0
    return [b0, b1, b2, b3]

def evaluate_bspline_segment(control_points, t):
    bases = uniform_cubic_bspline_basis(t)
    x = 0.0
    y = 0.0
    for i in range(4):
        x += bases[i] * control_points[i][0]
        y += bases[i] * control_points[i][1]
    return [x, y]

def generate_bspline_points(control_points, segments_per_curve=250):
    if len(control_points) < 4:
        return []
    
    all_points = []
    num_segments = len(control_points) - 3
    
    for seg in range(num_segments):
        segment_points = control_points[seg:seg + 4]
        for i in range(segments_per_curve + 1):
            t = i / segments_per_curve
            pt = evaluate_bspline_segment(segment_points, t)
            all_points.append(pt)
    
    return all_points

@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def draw_curve_kernel(n: ti.i32):
    for i in range(n):
        pt = curve_points_field[i]
        x_pixel = ti.cast(pt[0] * WIDTH, ti.i32)
        y_pixel = ti.cast(pt[1] * HEIGHT, ti.i32)
        if 0 <= x_pixel < WIDTH and 0 <= y_pixel < HEIGHT:
            pixels[x_pixel, y_pixel] = ti.Vector([0.0, 1.0, 0.0])

@ti.kernel
def draw_antialiased_curve_kernel(n: ti.i32):
    for i in range(n):
        pt = curve_points_field[i]
        x_center = pt[0] * WIDTH
        y_center = pt[1] * HEIGHT
        
        x0 = ti.cast(ti.floor(x_center - 1.5), ti.i32)
        y0 = ti.cast(ti.floor(y_center - 1.5), ti.i32)
        
        for dx in range(3):
            for dy in range(3):
                px = x0 + dx
                py = y0 + dy
                
                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    pixel_center_x = px + 0.5
                    pixel_center_y = py + 0.5
                    
                    dist_x = x_center - pixel_center_x
                    dist_y = y_center - pixel_center_y
                    distance = ti.sqrt(dist_x * dist_x + dist_y * dist_y)
                    
                    weight = ti.max(0.0, 1.0 - distance / 1.2)
                    weight = weight * weight * weight
                    
                    old_color = pixels[px, py]
                    contribution = ti.Vector([0.0, weight, 0.0])
                    pixels[px, py] = old_color + contribution * 0.7

def main():
    window = ti.ui.Window("Bezier & B-Spline Curves with Antialiasing", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []
    
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)
                    print(f"Added control point: {pos}")
            elif e.key == 'c':
                control_points = []
                print("Canvas cleared.")
            elif e.key == 'b':
                curve_mode[None] = 1 - curve_mode[None]
                mode_name = "B-Spline" if curve_mode[None] == 1 else "Bezier"
                print(f"Switched to {mode_name} mode")
        
        clear_pixels()
        
        current_count = len(control_points)
        
        if curve_mode[None] == 0 and current_count >= 2:
            curve_points_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
            for t_int in range(NUM_SEGMENTS + 1):
                t = t_int / NUM_SEGMENTS
                curve_points_np[t_int] = de_casteljau(control_points, t)
            curve_points_field.from_numpy(curve_points_np)
            draw_antialiased_curve_kernel(NUM_SEGMENTS + 1)
        elif curve_mode[None] == 1 and current_count >= 4:
            bspline_pts = generate_bspline_points(control_points, segments_per_curve=250)
            if bspline_pts:
                total_points = min(len(bspline_pts), NUM_SEGMENTS + 1)
                curve_points_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
                for i in range(total_points):
                    curve_points_np[i] = bspline_pts[i]
                curve_points_field.from_numpy(curve_points_np)
                draw_antialiased_curve_kernel(total_points)
        
        canvas.set_image(pixels)
        
        if current_count > 0:
            np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_points[:current_count] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_points)
            canvas.circles(gui_points, radius=0.006, color=(1.0, 0.0, 0.0))
            
            if current_count >= 2:
                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                indices = []
                for i in range(current_count - 1):
                    indices.extend([i, i + 1])
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_indices)
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.5, 0.5, 0.5))
        
        mode_name = "B-Spline" if curve_mode[None] == 1 else "Bezier"
        canvas.text(f"Mode: {mode_name} (Press 'b' to switch)", (0.02, 0.95), font_size=20)
        canvas.text(f"Press 'c' to clear", (0.02, 0.90), font_size=16)
        canvas.text(f"Left click to add control points", (0.02, 0.85), font_size=16)
        
        window.show()

if __name__ == '__main__':
    main()
