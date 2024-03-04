import numpy as np


class WrongDimentionError(Exception):
    def __init__(self, dim) -> None:
        super().__init__(f"{dim} is incorrect dimension")


def value(args) -> float:
    '''
    Returns function value of x,[y,..] coordinates
    '''
    try:
        x = args[0]
        y = args[1]
    except IndexError:
        raise WrongDimentionError(len(args)+1)
    return 1 - 0.6*np.exp(-x**2 - y**2) - 0.4*np.exp(-(x+1.75)**2 - (y-1)**2)


def gradient_vector(args) -> np.array:
    '''
    Returns numpy array of gradient coordinates (gradient = vector)
    x for 1 dim; x,y for 2 dims; x,y,z for 3 dims
    '''
    try:
        x = args[0]
        y = args[1]
    except IndexError:
        raise WrongDimentionError(len(args)+1)
    grad_x = 1.2*x*np.exp(-x**2-y**2) + 0.8*(x+1.75)*np.exp(-(x+1.75)**2 - (y-1)**2)
    grad_y = 1.2*y*np.exp(-x**2-y**2) + 0.8*(y-1)*np.exp(-(x+1.75)**2 - (y-1)**2)
    return np.array([grad_x, grad_y])


def dimensions():
    return 3
