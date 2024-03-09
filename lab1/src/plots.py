import numpy as np
import matplotlib.pyplot as plt

from .gradient import gradient_descent


def create_3D_plot(function_value: callable, generate_gradient: callable,
                   dimensions: int, eps=0.0001, step=0.01):
    '''
    Creates a 3D plot of 3-dimentional function with local minimum marked. Uses gradient descent function
    :function_value: function for which minimum is being searched
    :generate_gradient: function that returns gradient of the function from function_value
    :dimensions: degree of the function form function_value
    '''
    x = np.arange(-10, 10, 0.1)
    y = np.arange(-10, 10, 0.1)
    surface = np.meshgrid(x, y)
    z = function_value(surface)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    surf = ax.plot_surface(*surface, z, cmap='viridis')

    point = gradient_descent(function_value, generate_gradient, dimensions,
                             eps=eps, step=step)
    point_x, point_y, point_z = point[0], point[1], point[2]
    ax.scatter(point_x, point_y, point_z, color='red', s=100)
    ax.text(point_x, point_y, point_z,
            f'({point_x:.3f}, {point_y:.3f}, {point_z:.3f})', color='black')

    fig.colorbar(surf)
    plt.show()


def create_2D_plot(function_value: callable, generate_gradient: callable,
                   dimensions: int, step=0.01):
    '''
    Creates a 2D plot of 2-dimensional function with local minimum marked. Uses gradient descent function
    :function_value: function for which minimum is being searched
    :generate_gradient: function that returns gradient of the function from function_value
    :dimensions: degree of the function form function_value
    '''
    x = np.arange(-10, 10, 0.1)
    y = []
    for element in x:
        y.append(function_value([element]))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y)
    point = gradient_descent(function_value, generate_gradient, dimensions, step=step)
    point_x, point_y = point[0], point[1]
    plt.scatter(point_x, point_y, color='red', s=100)
    plt.text(point_x, point_y,
             f'({point_x:.3f}, {point_y:.3f}', color='black')
    plt.show()
