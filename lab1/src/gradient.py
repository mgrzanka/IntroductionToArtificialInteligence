import numpy as np
import json
from .config import value, gradient_vector, dimensions


def gradient_descent(step=0.01, starting_point=None):
    dim = dimensions()
    if starting_point is None:
        starting_point = np.random.uniform(-10, 10, size=dim-1)  # array([x/x,y/...])
    gradient = gradient_vector(starting_point)  # array([x/x,y/...])
    # for _ in range(100000):
    while not np.linalg.norm(gradient) < 0.000000000001:
        new_point = starting_point - step*gradient  # array[x,y,...]
        starting_point = new_point  # array([x/x,y/...])
        gradient = gradient_vector(starting_point)
    point = np.append(starting_point, value(starting_point))
    return point  # array([x,y/x,y,z/...])


def examine_step_value(steps=[0.1, 0.05, 0.02, 0.005]):
    dim = dimensions()
    starting_point = np.random.uniform(-10, 10, size=dim-1)
    result_dict = {
        "starting point": tuple(starting_point),
        "results": []
    }
    for step in steps:
        point = gradient_descent(step, starting_point)
        point_dict = {"step": step, "point": tuple(point)}
        result_dict["results"].append(point_dict)

    # Write to json
    with open("results.json", 'w') as json_file:
        json.dump(result_dict, json_file, indent=2)
