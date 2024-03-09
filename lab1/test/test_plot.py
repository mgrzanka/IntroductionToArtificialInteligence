import pytest
import numpy as np
from src.plots import create_2D_plot, create_3D_plot


def setup2D():
    dimensions = 2

    def func(args):
        x = args[0]
        return 2*x**2 + 3*x - 1

    def grad(args):
        x = args[0]
        return np.array([4*x + 3])

    return func, grad, dimensions


def test_2D_plot():
    create_2D_plot(*setup2D())


def setup3D():
    dimentions = 3

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

    return func, grad, dimentions


@pytest.fixture
# Forcing starting point, so the function won't be stuck on flat surface
def force_start_point(monkeypatch):
    def start_point(low, high, size=None):
        return np.array([-4.5, -8.3])
    monkeypatch.setattr("src.gradient.np.random.uniform", start_point)


def test_3D_plot():
    create_3D_plot(*setup3D(), step=0.01)


def test_3D_plot_force(force_start_point):
    create_3D_plot(*setup3D(), step=0.01)


def setup3D_v2():
    dimensions = 3

    def func(args):
        x = args[0]
        y = args[1]
        return x**2 + y**2

    def grad(args):
        x = args[0]
        y = args[1]
        grad_x = 2*x
        grad_y = 2*y
        return np.array([grad_x, grad_y])

    return func, grad, dimensions


def test_3D_plot_v2():
    create_3D_plot(*setup3D_v2())
