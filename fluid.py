'''
FLUID
by Josha Paonaskar

Manages fluid properties and motion
'''
import numpy as np
import matplotlib.pyplot as plt

from typing import Union

import utils

# walls
BOUNDS_N = 'n'
BOUNDS_E = 'e'
BOUNDS_S = 's'
BOUNDS_W = 'w'

class Fluid():
    '''
    Fluid class
    '''
    def __init__(self, width:int, height:int, boundries:str='nesw', dtype:type=np.float32):
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
        dtype : type = np.float32
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

    def circle(self, center:tuple[float], radius:float):
        '''
        Draw circle of walls

        Parameters:
        -----------
        center : tuple[float]
            circle center
        radius : float
            circle radius
        '''
        # create axes
        x, y = np.meshgrid(np.arange(self.w), np.arange(self.h))

        # circle equation for mask
        mask = np.square(x - center[0]) + np.square(y - center[1]) < radius * radius

        # apply mask
        self.s[1:-1, 1:-1][mask] = False

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
        self.u[:, 1:][~self.s[1:-1, 2:]] = -2.0

        # apply west walls
        self.u[:, :-1][~self.s[1:-1, :-2]] = -2.0

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

            # ignore solid cellss
            d[~self.s[1:-1, 1:-1]] = 0

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

    def average_v(self) -> np.ndarray:
        '''
        Get average v values for u

        Returns:
        --------
        np.ndarray
            average v values
        '''
        # copy v
        v = self.v.copy()

        # pad v
        pad_left = np.expand_dims(v[:, 0], axis=1)
        pad_right = np.expand_dims(v[:, -1], axis=1)
        v = np.hstack([pad_left, v, pad_right])

        # average values
        v = 0.25 * (v[:-1, :-1] + v[1:, :-1] + v[:-1, 1:] + v[1:, 1:])

        # return
        return v

    def average_u(self) -> np.ndarray:
        '''
        Get average u values for v

        Returns:
        --------
        np.ndarray
            average u values
        '''
        # copy v
        u = self.u.copy()

        # pad v
        pad_bottom = np.expand_dims(u[0, :], axis=0)
        pad_top = np.expand_dims(u[-1, :], axis=0)
        u = np.vstack([pad_bottom, u, pad_top])

        # average values
        u = 0.25 * (u[:-1, :-1] + u[1:, :-1] + u[:-1, 1:] + u[1:, 1:])

        # return
        return u
    
    def advection_for(self, dt:float):
        '''
        Move velocity field (Semi-Legrangian) using for loop

        Parameters:
        -----------
        dt : float
            timestep size
        '''
        for i in range(self.w + 1):
            for j in range(self.h + 1):
                # u component
                if (self.s[i, j+1] and self.s[i+1, j+1] and j < self.h):
                    # x, y in v space
                    x = i - 0.5
                    y = j + 0.5

                    # get velocity at u
                    u = self.u[i, j]
                    v = utils.sample_2D_field(self.v, x, y)

                    # advect

                # v component
                if (self.s[i+1, j+1] and self.s[i+1, j] and i < self.w):
                    pass

    def advection(self, dt:float):
        '''
        Move velocity field (Semi-Legrangian)

        Parameters:
        -----------
        dt : float
            timestep size
        '''
        # get averages
        avg_v = self.average_v()
        avg_u = self.average_u()

        # create coordinates
        ux, uy = np.meshgrid(np.arange(self.w + 1), np.arange(self.h))
        vx, vy = np.meshgrid(np.arange(self.w), np.arange(self.h + 1))

        # convert to types
        ux = ux.astype(self.dtype)
        uy = uy.astype(self.dtype)
        vx = vx.astype(self.dtype)
        vy = vy.astype(self.dtype)

        # step
        ux -= self.u * dt
        uy -= avg_v * dt
        
        vx -= avg_u * dt
        vy -= self.v * dt

        # sample
        u = utils.sample_2D_field(self.u, ux, uy)
        v = utils.sample_2D_field(self.v, vx, vy)

        # update velocity
        self.u = u
        self.v = v

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
    sim = Fluid(5, 5)
    sim.gravity(0.1)
    sim.walls()
    sim.incompressibility(0.1)
    sim.advection_for(0.1)