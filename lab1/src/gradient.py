from typing import Optional
import numpy as np
import json
import time
from matplotlib import pyplot as plt


def gradient_descent(function_value: callable, generate_gradient: callable,
                     dimensions: int, starting_point: Optional[np.array] = None,
                     eps=0.000000000001, step=0.01, draw=False,):
    '''
    Returns local minimum od the given function.
    ----------
    :function_value: function for which minimum is being searched
    :generate_gradient: function that returns gradient of the function from function_value
    :dimensions: degree of the function form function_value
    :starting_point: point from where the search starts
    :eps: precision for the search
    :step: coefficient for the algorithm
    :draw: flag to enable drawing founded points during minimum searching
    '''
    # generate random starting point if none is given
    if starting_point is None:
        starting_point = np.random.uniform(-10, 10, size=dimensions-1)
    gradient = generate_gradient(starting_point)  # gradient=array([x||x,y||x,y,z||...]) - min 1 dim

    # setting up plot if draw flag is set
    if draw:
        if dimensions == 2:
            setup_2D_plot(function_value)
        elif dimensions == 3:
            ax = setup_3D_plot(function_value)

    # main loop
    while not np.linalg.norm(gradient) < eps:
        # Draw current point if draw flag is set
        if draw:
            if dimensions == 2:
                point_x = starting_point[0]
                point_y = function_value(starting_point)
                plt.scatter(point_x, point_y, color='red', s=100)
            elif dimensions == 3:
                point_x, point_y = starting_point[0], starting_point[1]
                point_z = function_value(starting_point)
                ax.scatter(point_x, point_y, point_z, color='red', s=100)

        new_point = starting_point - step*gradient  # new_point=array[x||x,y||x,y,z||...] - min 1 dim
        starting_point = new_point
        gradient = generate_gradient(starting_point)

    # add minimum coordinates adn show the plot
    if draw:
        if dimensions == 2:
            point_x = starting_point[0]
            point_y = function_value(starting_point)
            plt.text(point_x, point_y,
                     f'({point_x:.3f}, {point_y:.3f}', color='black')
        elif dimensions == 3:
            point_x, point_y = starting_point[0], starting_point[1]
            point_z = function_value(starting_point)
            ax.text(point_x, point_y, point_z,
                    f'({point_x:.3f}, {point_y:.3f}, {point_z:.3f})', color='black')
        plt.show()

    # Append starting point array with its corresponding value
    point = np.append(starting_point, function_value(starting_point))
    return point  # return array([x,y||x,y,z||...]) - min 2 dim


def examine_step_value(function_value: callable, generate_gradient: callable,
                       dimensions, eps=0.000000000001, steps=[0.1, 0.05, 0.02, 0.005]):
    '''
    Creates a json file with gradient descent algorithm results for different
    steps and random starting point
    -------
    :function_value: function for which minimum is being searched
    :generate_gradient: function that returns gradient of the function from function_value
    :dimensions: degree of the function form function_value
    :eps: precision for the search
    :steps: list of coefficients for the algorithm to try
    '''
    starting_point = np.random.uniform(-10, 10, size=dimensions-1)

    # The structure of the json file
    result_dict = {
        "starting point": tuple(starting_point),
        "epsilon": eps,
        "results": []
    }

    # main loop
    for step in steps:
        start_time = time.time()
        point = gradient_descent(function_value, generate_gradient, dimensions,
                                 starting_point, eps, step)
        end_time = time.time()
        gradient_time = end_time - start_time
        # append results fot each step to the reuslt list
        point_dict = {"step": step, "point": tuple(point),
                      "time": gradient_time}
        result_dict["results"].append(point_dict)

    # Write to json
    with open("results.json", 'w') as json_file:
        json.dump(result_dict, json_file, indent=2)


def setup_3D_plot(function_value: callable):
    '''
    supp function for gradient_descent function
    :function_value: function for which minimum is being searched
    '''
    x = np.arange(-10, 10, 0.1)
    y = np.arange(-10, 10, 0.1)
    surface = np.meshgrid(x, y)
    z = function_value(surface)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(*surface, z, cmap='viridis')
    fig.colorbar(surf)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return ax


def setup_2D_plot(function_value: callable):
    '''
    supp function for gradient_descent function
    :function_value: function for which minimum is being searched
    '''
    x = np.arange(-10, 10, 0.1)
    y = []
    for element in x:
        y.append(function_value([element]))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y)
