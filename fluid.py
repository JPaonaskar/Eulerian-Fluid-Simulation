'''
FLUID
by Josha Paonaskar

Manages fluid prperties and time step
'''
import numpy as np
import matplotlib.pyplot as plt

from typing import Union

# walls
BOUNDS_N = 'n'
BOUNDS_E = 'e'
BOUNDS_S = 's'
BOUNDS_W = 'w'

class Fluid():
    '''
    Fluid class
    '''
    def __init__(self, width:int, height:int, boundries:str='nesw', dtype:type=np.float16):
        '''
        Initialize Fluid

        Parameters:
        -----------
        width : int
            fluid width
        height : int
            fluid height
        boundries : str = 'nesw'
            fluid boundries
        dtype : type = np.float16
            datatype to use
        '''
        # store shape
        self.w = width
        self.h = height
        self.dtype = dtype

        # create bounds
        self.s = np.ones((height + 2, width + 2), dtype=bool) # false is solid

        # add walls
        if BOUNDS_N in boundries:
            self.s[-1, :] = False
        if BOUNDS_E in boundries:
            self.s[:, -1] = False
        if BOUNDS_S in boundries:
            self.s[0, :] = False
        if BOUNDS_W in boundries:
            self.s[:, 0] = False

        # initialize velocity field
        self.u = np.zeros((height, width + 1), dtype=dtype)
        self.v = np.zeros((height + 1, width), dtype=dtype)

        # initilize pressure field
        self.p = np.zeros((height, width), dtype=dtype)

    def gravity(self, dt:float, g:Union[tuple, list, np.ndarray]=(0, -9.81)):
        '''
        Apply gravity

        Parameters:
        -----------
        dt : float
            timestep size
        g : tuple | list | np.ndarray
            g vector
        '''
        self.u += g[0] * dt
        self.v += g[1] * dt

    def walls(self):
        '''
        Apply boundry conditions at walls
        '''
        # apply east walls
        self.u[:, 1:][~self.s[1:-1, 2:]] = 0.0

        # apply west walls
        self.u[:, :-1][~self.s[1:-1, :-2]] = 0.0

        # apply north walls
        self.v[1:, :][~self.s[2:, 1:-1]] = 0.0

        # apply south walls
        self.v[:-1, :][~self.s[:-2, 1:-1]] = 0.0

    def divergence(self) -> np.ndarray:
        '''
        Compute divergence of cells

        Returns:
        --------
        np.ndarray
            cell divergences
        '''
        # initialize divergence
        d = np.zeros((self.h, self.w), dtype=self.dtype)

        # add east velocity
        d += self.u[:, 1:]

        # subtract west velocity
        d -= self.u[:, :-1]

        # add north velocity
        d += self.v[1:, :]

        # subtract south velocity
        d -= self.v[:-1, :]

        # return
        return d

    def _count_walls(self) -> np.ndarray:
        '''
        Compute number of neighboring walls cells

        Returns:
        --------
        np.ndarray
            wall count
        '''
        # initialize wall count
        s = np.zeros((self.h, self.w), dtype=np.uint8)

        # add north walls
        s += self.s[2:, 1:-1]

        # add east walls
        s += self.s[1:-1, 2:]

        # add south walls
        s += self.s[:-2, 1:-1]

        # add west walls
        s += self.s[1:-1, :-2]

        # return
        return s

    def incompressibility(self, dt:float, overrelax:float=1.0, n:int=10, rho:float=1.225, ds:float=0.01):
        '''
        Force incompressiblity and compute pressure

        Parameters:
        -----------
        dt : float
            timestep size
        overrelax : float = 1.0
            overrelaxation factor
        n : int = 10
            number of itterations
        rho : float = 1.225
            denisty (kg / m3)
        ds : float = 0.01
            grid spacing (m)
        '''
        # zero pressure
        self.p = np.zeros((self.h, self.w), dtype=self.dtype)

        # Use Gause-Seidel Method to solve grid
        for i in range(n):
            # compute divergence
            d = self.divergence()

            # count walls
            s = self._count_walls()

            # normalize divergence
            d = d / s * overrelax

            # update pressure
            self.p += d * rho * ds / dt

            # update east velocity
            self.u[:, 1:] -= d * self.s[1:-1, 2:]

            # update west velocity
            self.u[:, :-1] += d * self.s[1:-1, :-2]

            # update north velocity
            self.v[1:, :] -= d * self.s[2:, 1:-1]

            # update south velocity
            self.v[:-1, :] += d * self.s[:-2, 1:-1]

    def advection(self, dt:float):
        '''
        Move velocity field (Semi-Legrangian)
        '''
        pass

    def plot_velocites(self):
        '''
        Plot velocity vectors
        '''
        # plot u
        # plot v

    def plot_bounds(self):
        '''
        Plot walls
        '''
        plt.imshow(self.s, cmap='gray')
        plt.show()

    def plot_pressure(self):
        '''
        Plot pressure
        '''
        plt.imshow(self.p, cmap='jet')
        plt.colorbar()
        plt.show()

if __name__ == '__main__':
    fluid = Fluid(50, 50)
    dt = 0.05
    for i in range(50):
        fluid.gravity(dt)
        fluid.walls()
        fluid.incompressibility(dt, n=50)
        fluid.advection(dt)
    fluid.plot_pressure()