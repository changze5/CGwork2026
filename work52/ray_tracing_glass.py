import taichi as ti

ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())
glass_ior = ti.field(ti.f32, shape=())
msaa_samples = ti.field(ti.i32, shape=())

MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2

@ti.func
def normalize(v):
    return v / v.norm(1e-5)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

@ti.func
def refract(I, N, eta):
    cosi = ti.max(-1.0, ti.min(1.0, I.dot(N)))
    etai = 1.0
    etat = eta
    n = N
    inside = False
    if cosi < 0.0:
        cosi = -cosi
        etai, etat = etat, etai
        n = -N
        inside = True
    eta_ratio = etai / etat
    k = 1.0 - eta_ratio * eta_ratio * (1.0 - cosi * cosi)
    refract_dir = eta_ratio * I + (eta_ratio * cosi - ti.sqrt(ti.abs(k))) * n
    success = k >= 0.0
    return refract_dir, success, inside

@ti.func
def total_internal_reflection(I, N, eta):
    cosi = I.dot(N)
    return cosi < 0.0 and (1.0 - cosi * cosi) > (1.0 / eta) * (1.0 / eta)

@ti.func
def intersect_sphere(ro, rd, center, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal

@ti.func
def intersect_plane(ro, rd, plane_y):
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 0:
            t = t1
    return t, normal

@ti.func
def scene_intersect(ro, rd):
    min_t = 1e10
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE
    hit_inside = False

    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.95, 1.0])
        hit_mat = MAT_GLASS
        oc = ro - ti.Vector([-1.2, 0.0, 0.0])
        if oc.dot(rd) > 0:
            hit_inside = True
            hit_n = -n

    t, n = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.9, 0.9])
        hit_mat = MAT_MIRROR

    t, n = intersect_plane(ro, rd, -1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE
        p = ro + rd * t
        grid_scale = 2.0
        ix = ti.floor(p.x * grid_scale)
        iz = ti.floor(p.z * grid_scale)
        if (ix + iz) % 2 == 0:
            hit_c = ti.Vector([0.3, 0.3, 0.3])
        else:
            hit_c = ti.Vector([0.8, 0.8, 0.8])

    return min_t, hit_n, hit_c, hit_mat, hit_inside

@ti.func
def trace_ray(ro, rd, light_pos, bg_color, max_bounces):
    final_color = ti.Vector([0.0, 0.0, 0.0])
    throughput = ti.Vector([1.0, 1.0, 1.0])
    current_ior = 1.0
    
    for bounce in range(max_bounces):
        t, N, obj_color, mat_id, inside = scene_intersect(ro, rd)
        
        if t > 1e9:
            final_color += throughput * bg_color
            break
            
        p = ro + rd * t
        
        if mat_id == MAT_MIRROR:
            ro = p + N * 1e-4
            rd = normalize(reflect(rd, N))
            throughput *= 0.8 * obj_color
            
        elif mat_id == MAT_GLASS:
            ior = glass_ior[None]
            if inside:
                ior = 1.0 / ior
            
            eta = current_ior / ior
            refract_dir, success, refract_inside = refract(rd, N, ior)
            
            if success:
                if total_internal_reflection(rd, N, ior):
                    rd = normalize(reflect(rd, N))
                    ro = p + N * 1e-4
                else:
                    rd = normalize(refract_dir)
                    if refract_inside:
                        current_ior = 1.0
                    else:
                        current_ior = ior
                    ro = p - N * 1e-4
                    schlick = 0.04 + 0.96 * ti.pow(1.0 - ti.abs(rd.dot(N)), 5.0)
                    throughput *= (1.0 - schlick) * obj_color
            else:
                rd = normalize(reflect(rd, N))
                ro = p + N * 1e-4
            
        elif mat_id == MAT_DIFFUSE:
            L = normalize(light_pos - p)
            shadow_ray_orig = p + N * 1e-4
            shadow_t, _, _, _, _ = scene_intersect(shadow_ray_orig, L)
            
            dist_to_light = (light_pos - p).norm()
            in_shadow = 0.0
            if shadow_t < dist_to_light:
                in_shadow = 1.0
                
            ambient = 0.2 * obj_color
            direct_light = ambient
            
            if in_shadow == 0.0:
                diff = ti.max(0.0, N.dot(L))
                diffuse = 0.8 * diff * obj_color
                direct_light += diffuse
            
            final_color += throughput * direct_light
            break
    
    return final_color

@ti.kernel
def render():
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    bg_color = ti.Vector([0.05, 0.15, 0.2])
    samples = msaa_samples[None]
    
    for i, j in pixels:
        pixel_color = ti.Vector([0.0, 0.0, 0.0])
        
        for s in range(samples):
            u_base = (i - res_x / 2.0) / res_y * 2.0
            v_base = (j - res_y / 2.0) / res_y * 2.0
            
            u_offset = (ti.random() - 0.5) / res_y * 2.0 if samples > 1 else 0.0
            v_offset = (ti.random() - 0.5) / res_y * 2.0 if samples > 1 else 0.0
            
            u = u_base + u_offset
            v = v_base + v_offset
            
            ro = ti.Vector([0.0, 1.0, 5.0])
            rd = normalize(ti.Vector([u, v - 0.2, -1.0]))
            
            pixel_color += trace_ray(ro, rd, light_pos, bg_color, max_bounces[None])
        
        pixels[i, j] = ti.math.clamp(pixel_color / float(samples), 0.0, 1.0)

def main():
    window = ti.ui.Window("Ray Tracing Demo", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    light_pos_x[None] = 2.0
    light_pos_y[None] = 4.0
    light_pos_z[None] = 3.0
    max_bounces[None] = 5
    glass_ior[None] = 1.5
    msaa_samples[None] = 4

    while window.running:
        render()
        canvas.set_image(pixels)
        
        with gui.sub_window("Controls", 0.75, 0.05, 0.23, 0.35):
            light_pos_x[None] = gui.slider_float('Light X', light_pos_x[None], -5.0, 5.0)
            light_pos_y[None] = gui.slider_float('Light Y', light_pos_y[None], 1.0, 8.0)
            light_pos_z[None] = gui.slider_float('Light Z', light_pos_z[None], -5.0, 5.0)
            max_bounces[None] = gui.slider_int('Max Bounces', max_bounces[None], 1, 8)
            glass_ior[None] = gui.slider_float('Glass IOR', glass_ior[None], 1.0, 2.5)
            gui.text('')
            msaa_samples[None] = gui.slider_int('MSAA Samples', msaa_samples[None], 1, 8)

        window.show()

if __name__ == '__main__':
    main()
