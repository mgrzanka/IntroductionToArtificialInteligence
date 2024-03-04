import numpy as np
import matplotlib.pyplot as plt

from .gradient import gradient_descent
from .config import value


def create_3D_plot():
    x = np.arange(-10, 10, 0.1)
    y = np.arange(-10, 10, 0.1)
    surface = np.meshgrid(x, y)
    z = value(surface)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(*surface, z, cmap='viridis')

    point = gradient_descent()
    point_x, point_y, point_z = point[0], point[1], point[2]
    ax.scatter(point_x, point_y, point_z, color='red', s=100)
    ax.text(point_x, point_y, point_z,
            f'({point_x:.3f}, {point_y:.3f}, {point_z:.3f})', color='black')

    fig.colorbar(surf)
    plt.show()


def create_2D_plot():
    x = np.arange(-10, 10, 0.1)
    y = []
    for element in x:
        y.append(value([element]))
    plt.plot(x, y)
    point = gradient_descent()
    point_x, point_y = point[0], point[1]
    plt.scatter(point_x, point_y, color='red', s=100)
    plt.text(point_x, point_y,
             f'({point_x:.3f}, {point_y:.3f}', color='black')
    plt.show()
