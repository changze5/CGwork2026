import pygame
import numpy as np

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bezier & B-Spline Curves with Antialiasing")
font = pygame.font.Font(None, 24)

def de_casteljau(points, t):
    if len(points) == 1:
        return points[0]
    next_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        next_points.append([x, y])
    return de_casteljau(next_points, t)

def uniform_cubic_bspline_basis(t):
    t2 = t * t
    t3 = t2 * t
    b0 = (-t3 + 3 * t2 - 3 * t + 1) / 6.0
    b1 = (3 * t3 - 6 * t2 + 4) / 6.0
    b2 = (-3 * t3 + 3 * t2 + 3 * t + 1) / 6.0
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

def draw_antialiased_curve(surface, curve_points, color=(0, 255, 0)):
    for pt in curve_points:
        x_center, y_center = pt[0], pt[1]
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                px = int(x_center + dx)
                py = int(y_center + dy)
                
                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    pixel_center_x = px + 0.5
                    pixel_center_y = py + 0.5
                    
                    dist_x = x_center - pixel_center_x
                    dist_y = y_center - pixel_center_y
                    distance = np.sqrt(dist_x * dist_x + dist_y * dist_y)
                    
                    weight = max(0.0, 1.0 - distance / 1.2)
                    weight = weight ** 3
                    
                    current_color = surface.get_at((px, py))
                    new_color = (
                        min(255, int(current_color[0] + color[0] * weight * 0.7)),
                        min(255, int(current_color[1] + color[1] * weight * 0.7)),
                        min(255, int(current_color[2] + color[2] * weight * 0.7))
                    )
                    surface.set_at((px, py), new_color)

def main():
    running = True
    control_points = []
    curve_mode = "bezier"
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if len(control_points) < MAX_CONTROL_POINTS:
                        pos = event.pos
                        control_points.append(pos)
                        print(f"Added control point: {pos}")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    control_points = []
                    print("Canvas cleared.")
                elif event.key == pygame.K_b:
                    curve_mode = "bspline" if curve_mode == "bezier" else "bezier"
                    mode_name = "B-Spline" if curve_mode == "bspline" else "Bezier"
                    print(f"Switched to {mode_name} mode")
        
        screen.fill((0, 0, 0))
        
        current_count = len(control_points)
        
        if curve_mode == "bezier" and current_count >= 2:
            curve_points = []
            for t_int in range(NUM_SEGMENTS + 1):
                t = t_int / NUM_SEGMENTS
                pt = de_casteljau(control_points, t)
                curve_points.append(pt)
            draw_antialiased_curve(screen, curve_points, (0, 255, 0))
        elif curve_mode == "bspline" and current_count >= 4:
            bspline_pts = generate_bspline_points(control_points, segments_per_curve=250)
            if bspline_pts:
                draw_antialiased_curve(screen, bspline_pts, (0, 255, 0))
        
        if current_count > 0:
            for pt in control_points:
                pygame.draw.circle(screen, (255, 0, 0), (int(pt[0]), int(pt[1])), 5)
            
            if current_count >= 2:
                pygame.draw.lines(screen, (128, 128, 128), False, 
                                [(int(pt[0]), int(pt[1])) for pt in control_points], 2)
        
        mode_text = f"Mode: {'B-Spline' if curve_mode == 'bspline' else 'Bezier'} (Press 'b' to switch)"
        screen.blit(font.render(mode_text, True, (255, 255, 255)), (10, 10))
        screen.blit(font.render("Press 'c' to clear", True, (255, 255, 255)), (10, 35))
        screen.blit(font.render("Left click to add control points", True, (255, 255, 255)), (10, 60))
        
        pygame.display.flip()
        pygame.time.Clock().tick(60)
    
    pygame.quit()

if __name__ == '__main__':
    main()
