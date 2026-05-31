import taichi as ti

ti.init(arch=ti.gpu)

N = 20
mass = 1.0
dt = 5e-4
k_s = 8000.0
k_shear = 4000.0
k_bend = 2000.0
k_d = 1.0
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0

x = ti.Vector.field(3, dtype=float, shape=N * N)
v = ti.Vector.field(3, dtype=float, shape=N * N)
f = ti.Vector.field(3, dtype=float, shape=N * N)
is_fixed = ti.field(dtype=int, shape=N * N)

x_next = ti.Vector.field(3, dtype=float, shape=N * N)
v_next = ti.Vector.field(3, dtype=float, shape=N * N)
f_next = ti.Vector.field(3, dtype=float, shape=N * N)

max_springs = N * N * 8
spring_indices = ti.field(dtype=int, shape=max_springs * 2)
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)
spring_lengths = ti.field(dtype=float, shape=max_springs)
spring_types = ti.field(dtype=int, shape=max_springs)
num_springs = ti.field(dtype=int, shape=())

ball_pos = ti.Vector.field(3, dtype=float, shape=())
ball_radius = ti.field(dtype=float, shape=())
collision_enabled = ti.field(dtype=int, shape=())
shear_enabled = ti.field(dtype=int, shape=())
bend_enabled = ti.field(dtype=int, shape=())

@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0

@ti.kernel
def init_springs():
    num_springs[None] = 0
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        if i < N - 1:
            idx_right = (i + 1) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_right])
            spring_lengths[c] = (x[idx] - x[idx_right]).norm()
            spring_types[c] = 0
        if j < N - 1:
            idx_down = i * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_down])
            spring_lengths[c] = (x[idx] - x[idx_down]).norm()
            spring_types[c] = 0
        if shear_enabled[None]:
            if i < N - 1 and j < N - 1:
                idx_diag = (i + 1) * N + (j + 1)
                c = ti.atomic_add(num_springs[None], 1)
                spring_pairs[c] = ti.Vector([idx, idx_diag])
                spring_lengths[c] = (x[idx] - x[idx_diag]).norm()
                spring_types[c] = 1
            if i > 0 and j < N - 1:
                idx_diag = (i - 1) * N + (j + 1)
                c = ti.atomic_add(num_springs[None], 1)
                spring_pairs[c] = ti.Vector([idx, idx_diag])
                spring_lengths[c] = (x[idx] - x[idx_diag]).norm()
                spring_types[c] = 1
        if bend_enabled[None]:
            if i < N - 2:
                idx_skip = (i + 2) * N + j
                c = ti.atomic_add(num_springs[None], 1)
                spring_pairs[c] = ti.Vector([idx, idx_skip])
                spring_lengths[c] = (x[idx] - x[idx_skip]).norm()
                spring_types[c] = 2
            if j < N - 2:
                idx_skip = i * N + (j + 2)
                c = ti.atomic_add(num_springs[None], 1)
                spring_pairs[c] = ti.Vector([idx, idx_skip])
                spring_lengths[c] = (x[idx] - x[idx_skip]).norm()
                spring_types[c] = 2

@ti.kernel
def init_spring_indices():
    for i in range(num_springs[None]):
        spring_indices[i * 2] = spring_pairs[i][0]
        spring_indices[i * 2 + 1] = spring_pairs[i][1]

def init_cloth():
    num_springs[None] = 0
    init_positions()
    init_springs()
    init_spring_indices()

@ti.func
def get_spring_stiffness(stype: int) -> float:
    if stype == 0:
        return k_s
    elif stype == 1:
        return k_shear
    else:
        return k_bend

@ti.func
def get_spring_color(stype: int) -> ti.Vector:
    if stype == 0:
        return ti.Vector([0.8, 0.8, 0.8])
    elif stype == 1:
        return ti.Vector([1.0, 0.5, 0.2])
    else:
        return ti.Vector([0.5, 1.0, 0.5])

@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template()):
    for i in range(N * N):
        force[i] = gravity * mass - k_d * vel[i]
    for i in range(num_springs[None]):
        idx_a = spring_pairs[i][0]
        idx_b = spring_pairs[i][1]
        pos_a = pos[idx_a]
        pos_b = pos[idx_b]
        d = pos_a - pos_b
        dist = d.norm()
        if dist > 1e-6:
            d_normalized = d / dist
            k = get_spring_stiffness(spring_types[i])
            f_spring = -k * (dist - spring_lengths[i]) * d_normalized
            ti.atomic_add(force[idx_a], f_spring)
            ti.atomic_add(force[idx_b], -f_spring)

@ti.func
def handle_collisions(pos: ti.template(), vel: ti.template()):
    if collision_enabled[None] == 0:
        return
    bp = ball_pos[None]
    br = ball_radius[None]
    for i in range(N * N):
        if is_fixed[i] == 0:
            diff = pos[i] - bp
            dist = diff.norm()
            if dist < br + 0.015:
                if dist > 1e-6:
                    normal = diff / dist
                    pos[i] = bp + normal * (br + 0.015)
                    v_proj = vel[i].dot(normal)
                    if v_proj < 0:
                        vel[i] -= 1.2 * v_proj * normal

@ti.func
def clamp_velocity(vel: ti.template(), idx: int):
    vel_norm = vel[idx].norm()
    if vel_norm > max_velocity:
        vel[idx] = vel[idx] / vel_norm * max_velocity

@ti.kernel
def step_explicit():
    compute_forces_on(x, v, f)
    handle_collisions(x, v)
    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)

@ti.kernel
def step_semi_implicit():
    compute_forces_on(x, v, f)
    handle_collisions(x, v)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt

@ti.kernel
def step_implicit_iter():
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]
    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next)
        handle_collisions(x_next, v_next)
        for i in range(N * N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i] / mass) * dt
                clamp_velocity(v_next, i)
                x_next[i] = x[i] + v_next[i] * dt
    for i in range(N * N):
        v[i] = v_next[i]
        x[i] = x_next[i]

def main():
    ball_pos[None] = ti.Vector([0.0, 0.3, 0.0])
    ball_radius[None] = 0.15
    collision_enabled[None] = 1
    shear_enabled[None] = 1
    bend_enabled[None] = 1
    
    init_cloth()

    window = ti.ui.Window("Games101 - Enhanced Mass Spring System", (800, 800))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.5, 2.0)
    camera.lookat(0.0, 0.0, 0.0)

    current_method = 1
    paused = False

    while window.running:
        window.GUI.begin("Control Panel", 0.02, 0.02, 0.42, 0.55)
        
        window.GUI.text("Integration Method:")
        
        prefix_0 = "[*] " if current_method == 0 else "[ ] "
        prefix_1 = "[*] " if current_method == 1 else "[ ] "
        prefix_2 = "[*] " if current_method == 2 else "[ ] "

        if window.GUI.button(prefix_0 + "Explicit Euler"):
            current_method = 0
            init_cloth()
        if window.GUI.button(prefix_1 + "Semi-Implicit Euler"):
            current_method = 1
            init_cloth()
        if window.GUI.button(prefix_2 + "Implicit Euler"):
            current_method = 2
            init_cloth()
        
        window.GUI.text("")
        window.GUI.text("Spring Types:")
        
        struct_on = shear_enabled[None] == 1
        if window.GUI.checkbox("Structural Springs", struct_on):
            shear_enabled[None] = 1 - shear_enabled[None]
            init_cloth()
        
        shear_on = shear_enabled[None] == 1
        if window.GUI.checkbox("Shear Springs", shear_on):
            shear_enabled[None] = 1 - shear_enabled[None]
            init_cloth()
        
        bend_on = bend_enabled[None] == 1
        if window.GUI.checkbox("Bending Springs", bend_on):
            bend_enabled[None] = 1 - bend_enabled[None]
            init_cloth()
        
        window.GUI.text("")
        window.GUI.text("Collision:")
        
        coll_on = collision_enabled[None] == 1
        if window.GUI.checkbox("Sphere Collision", coll_on):
            collision_enabled[None] = 1 - collision_enabled[None]
        
        window.GUI.text(f"Ball Radius: {ball_radius[None]:.3f}")
        ball_radius[None] = window.GUI.slider_float(" ", ball_radius[None], 0.05, 0.3)
        
        window.GUI.text("")
        pause_label = "Resume" if paused else "Pause"
        if window.GUI.button(pause_label):
            paused = not paused
        
        if window.GUI.button("Reset"):
            init_cloth()
        
        window.GUI.end()

        if not paused:
            for _ in range(40):
                if current_method == 0:
                    step_explicit()
                elif current_method == 1:
                    step_semi_implicit()
                elif current_method == 2:
                    step_implicit_iter()

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(x, radius=0.015, color=(0.2, 0.6, 1.0))
        
        if shear_enabled[None]:
            struct_count = 0
            shear_count = 0
            bend_count = 0
            for i in range(num_springs[None]):
                stype = spring_types[i]
                if stype == 0:
                    struct_count += 1
                elif stype == 1:
                    shear_count += 1
                else:
                    bend_count += 1
            
            struct_indices = ti.field(dtype=int, shape=struct_count * 2)
            shear_indices_arr = ti.field(dtype=int, shape=shear_count * 2)
            bend_indices_arr = ti.field(dtype=int, shape=bend_count * 2)
            
            si = 0
            shi = 0
            bi = 0
            for i in range(num_springs[None]):
                idx_a = spring_pairs[i][0]
                idx_b = spring_pairs[i][1]
                stype = spring_types[i]
                if stype == 0:
                    struct_indices[si] = idx_a
                    struct_indices[si + 1] = idx_b
                    si += 2
                elif stype == 1:
                    shear_indices_arr[shi] = idx_a
                    shear_indices_arr[shi + 1] = idx_b
                    shi += 2
                else:
                    bend_indices_arr[bi] = idx_a
                    bend_indices_arr[bi + 1] = idx_b
                    bi += 2
            
            if struct_count > 0:
                scene.lines(x, indices=struct_indices, width=1.5, color=(0.8, 0.8, 0.8))
            if shear_count > 0:
                scene.lines(x, indices=shear_indices_arr, width=1.2, color=(1.0, 0.5, 0.2))
            if bend_count > 0:
                scene.lines(x, indices=bend_indices_arr, width=0.8, color=(0.5, 1.0, 0.5))
        else:
            scene.lines(x, indices=spring_indices, width=1.5, color=(0.8, 0.8, 0.8))
        
        if collision_enabled[None]:
            scene.particles(ball_pos, radius=ball_radius[None], color=(0.9, 0.3, 0.3))
        
        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()
