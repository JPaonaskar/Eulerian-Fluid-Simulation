'''
MAIN
by Josha Paonaskar

Main file to run fluid simulator
'''

import cv2
import time
import numpy as np

import fluid

sim = fluid.Fluid(100, 100)
dt = 0.01
t = time.time()

while True:
    sim.gravity(dt)
    sim.walls()
    sim.incompressibility(dt, overrelax=1.0, n=50)
    sim.advection(dt)

    # plot pressures
    p = sim.p.copy()
    p = p - p.min()
    p = p / p.max() * 255
    p = p.astype(np.uint8)

    p = cv2.applyColorMap(p, cv2.COLORMAP_JET)
    cv2.imshow('Pressure', cv2.resize(p, (500, 500), cv2.INTER_NEAREST))

    # key binds
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

    # time step
    dt = time.time() - t
    t += dt