import numpy as np
import matplotlib.pyplot as plt
from random import randrange


def function(arguments):
    x = arguments[0]
    y = arguments[1]
    return x**2 + y**2


def function_gradient(arguments):
    x = arguments[0]
    y = arguments[1]
    return np.array([2*x, 2*y])


def create_points(dimention):
    points = np.meshgrid(*[np.arange(-50, 50, 0.1) for _ in range(dimention)])
    y_values = function(points)
    return points, y_values


def find_minimum(dimentions):
    step, eps = 0.01, 0.01
    gradient = eps + 1
    current_points = [randrange(-50, 50) for _ in range(dimentions)]
    # current_value = function(current_points)
    # current_possition = np.array(*current_points, current_value)

    while np.linalg.norm(gradient) > eps:
        gradient = function_gradient(current_points)
        new_points = current_points - step*gradient
        current_points = new_points
    current_value = function(current_points)
    return current_points, current_value


def main():
    # print(np.meshgrid(*[np.arange(-50, 50, 0.1)]))
    print(np.linalg.norm(np.array([2])))


if __name__ == '__main__':
    main()




# Rysowanie później

def plot(dimentions):
    points = create_points(dimentions)
    p = [*points[0]]
    plt.plot(*points[0], points[1])
    plt.show()


def plot_chart(x, y):
    x_values, y_values = create_points()
    plt.plot(x_values, y_values)
    plt.scatter(x, y, color="red")
    plt.show()