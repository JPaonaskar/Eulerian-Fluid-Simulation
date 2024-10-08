'''
UTILS
by Josha Paonaskar

Useful methods
'''
import numpy as np

def sample_2D_field(field:np.ndarray, x:np.ndarray, y:np.ndarray) -> np.ndarray:
    '''
    Sample points from an 2D field

    Parameters:
    -----------
    field : np.ndarray
        field to sample
    x : np.ndarray
        x index (column)
    y : np.ndarray
        y index (row)

    Returns:
    --------
    np.ndarray
        sample
    '''
    h, w = field.shape

    # clamp values
    x[x < 0.0] = 0.0
    y[y < 0.0] = 0.0

    x[x > w - 1.0] = w - 1.0
    y[y > h - 1.0] = h - 1.0

    # compress field
    field = field.flatten()
    
    # get corner indexes
    a = np.floor(y) * w + np.floor(x) # bottom left
    b = np.floor(y) * w +  np.ceil(x) # bottom right
    c =  np.ceil(y) * w + np.floor(x) # top left
    d =  np.ceil(y) * w +  np.ceil(x) # top right

    #print(h, w, w*h-1, y.max(), x.max(), (y * w + x).max(), ((h-1) * w + w-1), a.max()) ### ROUNDING ISSUE FOR LARGE FLOATS!!

    # slice
    a = field[a.astype(int)]
    b = field[b.astype(int)]
    c = field[c.astype(int)]
    d = field[d.astype(int)]

    # get interpolation values
    px = np.mod(x, 1.0)
    py = np.mod(y, 1.0)

    # interpolate
    x = a + (b - a) * px + (c - a) * py + (a - b - c + d) * px * py

    # output
    return x