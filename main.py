'''
MAIN
by Josha Paonaskar

Main file to run fluid simulator
'''

import cv2
import time
import numpy as np

import fluid

sim = fluid.Fluid(200, 200, boundries='nsw')
sim.circle((70, 100), 20)
dt = 0.01
t = time.time()

while True:
    #sim.gravity(dt)
    sim.walls()
    sim.incompressibility(dt, overrelax=1.0, n=50)
    sim.advection(dt)

    # plot pressures
    p = sim.p.copy()
    p = p / np.abs(p).max() * 127.5 + 127.5
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