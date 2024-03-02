import numpy as np
from config import value, gradient_vector, dimensions


def gradient_descent():
    dim = dimensions()
    starting_point = np.random.uniform(-10, 10, size=dim-1)  # array([x/x,y/...])
    gradient = gradient_vector(starting_point)  # array([x/x,y/...])
    step = 0.01  # float
    # for _ in range(100000):
    while not np.linalg.norm(gradient) < 0.0000000001:
        new_point = starting_point - step*gradient  # array[x,y,...]
        starting_point = new_point  # array([x/x,y/...])
        gradient = gradient_vector(starting_point)
    point = np.append(starting_point, value(starting_point))
    return point  # array([x,y/x,y,z/...])


def examine_step_value():
    pass
