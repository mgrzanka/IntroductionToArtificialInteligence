import numpy as np


class WrongDimentionError(Exception):
    def __init__(self, dim) -> None:
        super().__init__(f"{dim} is incorrect dimension")


def value(args) -> float:
    '''
    Sample function_value function to use in gradient_descent()
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
    Sample generate_gradient function to use in gradient_descent()
    Returns numpy array of gradient coordinates (gradient = vector)
    '''
    try:
        x = args[0]
        y = args[1]
    except IndexError:
        raise WrongDimentionError(len(args)+1)
    grad_x = 1.2*x*np.exp(-x**2-y**2) + 0.8*(x+1.75)*np.exp(-(x+1.75)**2 - (y-1)**2)
    grad_y = 1.2*y*np.exp(-x**2-y**2) + 0.8*(y-1)*np.exp(-(x+1.75)**2 - (y-1)**2)
    return np.array([grad_x, grad_y])
