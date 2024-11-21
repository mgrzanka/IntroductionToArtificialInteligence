import numpy as np
import matplotlib.pyplot as plt
from random import uniform
from GradientDescend.GradientDescend import GradientDescent


def setup2D():
    dimentions = 2

    def func(args):
        x = args[0]
        return 2*x**2 + 3*x - 1

    def grad(args):
        x = args[0]
        return np.array([4*x + 3])

    return func, grad, dimentions


def setup3D():
    dimensions = 3

    def func(args):
        x = args[0]
        y = args[1]
        return 1 - 0.6*np.exp(-x**2 - y**2) - 0.4*np.exp(-(x+1.75)**2 - (y-1)**2)

    def grad(args):
        x = args[0]
        y = args[1]
        grad_x = 1.2*x*np.exp(-x**2-y**2) + 0.8*(x+1.75)*np.exp(-(x+1.75)**2 - (y-1)**2)
        grad_y = 1.2*y*np.exp(-x**2-y**2) + 0.8*(y-1)*np.exp(-(x+1.75)**2 - (y-1)**2)
        return np.array([grad_x, grad_y])

    return func, grad, dimensions


def test_gradient():
    gradient1 = GradientDescent(0.01, eps=0.01)
    gradient2 = GradientDescent(0.01, starting_point=np.array([1.0, 1.0]))
    gradient1.solve(*setup2D(), draw=True)
    gradient2.solve(*setup3D(), draw=True)


def test_plot():
    gradient1 = GradientDescent(0.01, eps=0.01)
    gradient2 = GradientDescent(0.01, starting_point=np.array([1.0, 1.0]))
    gradient1.plot_function(*setup2D())
    gradient2.plot_function(*setup3D())


def test_starting_point():
    gradient = GradientDescent(0.01)
    iterations2D_values = []
    # iterations3D_values = []
    start_points_values = []
    for _ in range(100):
        start = np.random.uniform(-10, 10, 1)
        gradient.set_starting_point(start)
        iterations = gradient.solve(*setup2D())[1]
        iterations2D_values.append(iterations)
        start_points_values.append(start[0])

        # start = np.random.uniform(-10, 10, 2)
        # gradient.set_starting_point(start)
        # iterations = gradient.solve(*setup3D())[1]
        # iterations3D_values.append(iterations)

    plt.xlabel('x coordnate of start point')
    plt.ylabel('iterations')
    plt.title("Iterations for different starting points for f(x)")
    real_x = gradient.solve(*setup2D())[0][0]
    plt.plot(start_points_values, iterations2D_values, 'o')
    plt.scatter(real_x, min(iterations2D_values)-10, color="red")
    plt.text(real_x, min(iterations2D_values)-10,
             "real x cordinate", color="red")
    plt.show()


def test_step_value_fx():
    iterations_values = []
    steps = []
    gradient = GradientDescent(0.01, starting_point=np.array([-10]))
    for _ in range(50):
        step = uniform(0.001, 0.49)
        gradient.set_step(step)
        iterations = gradient.solve(*setup2D())[1]
        iterations_values.append(iterations)
        steps.append(step)
    plt.xlabel("step")
    plt.ylabel("iterations")
    plt.title("Iterations for different steps for f(x)")
    plt.plot(steps, iterations_values, 'o')
    plt.show()


def test_step_value_gx():
    iterations_values = []
    steps = []
    gradient = GradientDescent(0.01, starting_point=np.array([1.0, 1.0]), eps=0.001)
    for _ in range(75):
        step = uniform(11, 30)
        gradient.set_step(step)
        iterations = gradient.solve(*setup3D())[1]
        iterations_values.append(iterations)
        steps.append(step)
    plt.xlabel("step")
    plt.ylabel("iterations")
    plt.title("Iterations for steps 11-30 for g(x)\nStarting point (1.0, 1.0)")
    plt.plot(steps, iterations_values, 'o')
    plt.show()


def test_eps_value_fx():
    eps = 0.01
    values = []
    gradient = GradientDescent(0.01)
    for _ in range(10):
        gradient.set_eps(eps)
        minimum = gradient.solve(*setup2D())[0]
        values.append(tuple(minimum))
        eps /= 10
    print(values)


# test_eps_value_fx()

def test_eps_value_gx():
    eps = 0.01
    values = []
    gradient = GradientDescent(0.01, starting_point=np.array([1.0, 1.0]))
    for _ in range(10):
        gradient.set_eps(eps)
        minimum = gradient.solve(*setup3D())[0]
        values.append(tuple(minimum))
        eps /= 10
    print(values)


test_eps_value_gx()
