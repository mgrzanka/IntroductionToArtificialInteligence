import numpy as np
import pytest
from src.gradient import gradient_descent, examine_step_value


def setup2D():
    dimentions = 2

    def func(args):
        x = args[0]
        return 2*x**2 + 3*x - 1

    def grad(args):
        x = args[0]
        return np.array([4*x + 3])

    return func, grad, dimentions


def test_2D_function():
    minimum = gradient_descent(*setup2D(), starting_point=np.array([-10]),
                               step=0.001, draw=True)
    assert -0.75 == pytest.approx(minimum[0])
    assert -2.125 == pytest.approx(minimum[1])


def test_examine_step_value_2D():
    examine_step_value(*setup2D(), steps=[0.01], eps=0.00000000000001)  # Check results.json


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


@pytest.fixture
def force_start_point(monkeypatch):
    # Forcing starting point, so the function won't be stuck on flat surface
    def start_point(low, high, size=None):
        return np.array([1.0, 1.0])
    monkeypatch.setattr("src.gradient.np.random.uniform", start_point)


def test_3D_function(force_start_point):
    minimum = gradient_descent(*setup3D(), step=5)
    pass  # Pass for checking the output


def test_examine_step_value_3D():
    examine_step_value(*setup3D(), eps=0.001)


def test_examine_step_value_3D_forced(force_start_point):
    examine_step_value(*setup3D(), eps=0.0000001, steps=[0.001, 0.01, 0.1, 1, 15])


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


def test_dim_func():
    minimum = gradient_descent(*setup3D_v2(), draw=True)
    assert 0 == pytest.approx(minimum[0])
    assert 0 == pytest.approx(minimum[1])
    assert 0 == pytest.approx(minimum[2])


def test_examine_step_value_3Dv2():
    examine_step_value(*setup3D_v2())
