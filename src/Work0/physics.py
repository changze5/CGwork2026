import taichi as ti
from .config import *

pos = ti.Vector.field(2, dtype=float, shape=NUM_PARTICLES)
vel = ti.Vector.field(2, dtype=float, shape=NUM_PARTICLES)

@ti.kernel
def init_particles():
    for i in range(NUM_PARTICLES):
        pos[i] = [ti.random(), ti.random()]
        vel[i] = [0.0, 0.0]

@ti.kernel
def update_particles(mouse_x: float, mouse_y: float):
    for i in range(NUM_PARTICLES):
        mouse_pos = ti.Vector([mouse_x, mouse_y])
        dir = mouse_pos - pos[i]
        dist = dir.norm()
        
        if dist > 0.05:
            vel[i] += dir.normalized() * GRAVITY_STRENGTH
        
        vel[i] *= DRAG_COEF
        pos[i] += vel[i]

        for j in ti.static(range(2)):
            if pos[i][j] < 0:
                pos[i][j] = 0.0
                vel[i][j] *= BOUNCE_COEF
            elif pos[i][j] > 1:
                pos[i][j] = 1.0
                vel[i][j] *= BOUNCE_COEF