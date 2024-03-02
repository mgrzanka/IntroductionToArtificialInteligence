import numpy as np
import pytest
from gradient import gradient_descent


def test_func(monkeypatch):
    # Patching config functions
    def dim():
        return 2
    monkeypatch.setattr("gradient.dimensions", dim)

    def grad(args):
        x = args[0]
        return np.array([4*x + 3])
    monkeypatch.setattr("gradient.gradient_vector", grad)

    def func(args):
        x = args[0]
        return 2*x**2 + 3*x - 1
    monkeypatch.setattr("gradient.value", func)
    # Assertions
    minimum = gradient_descent()
    assert -0.75 == pytest.approx(minimum[0])
    assert -2.125 == pytest.approx(minimum[1])


def test_dim_func(monkeypatch):
    # Patching config functions
    def dim():
        return 3
    monkeypatch.setattr("gradient.dimensions", dim)

    def grad(args):
        x = args[0]
        y = args[1]
        grad_x = 2*x
        grad_y = 2*y
        return np.array([grad_x, grad_y])
    monkeypatch.setattr("gradient.gradient_vector", grad)

    def func(args):
        x = args[0]
        y = args[1]
        return x**2 + y**2
    monkeypatch.setattr("gradient.value", func)
    # Assertions
    minimum = gradient_descent()
    assert 0 == pytest.approx(minimum[0])
    assert 0 == pytest.approx(minimum[1])
    assert 0 == pytest.approx(minimum[2])


def test_three_dim_func(monkeypatch):
    # Patching config functions
    def dim():
        return 3
    monkeypatch.setattr("gradient.dimensions", dim)

    def grad(args):
        x = args[0]
        y = args[1]
        grad_x = 1.2*x*np.exp(-x**2-y**2) + 0.8*(x+1.75)*np.exp(-(x+1.75)**2 - (y-1)**2)
        grad_y = 1.2*y*np.exp(-x**2-y**2) + 0.8*(y-1)*np.exp(-(x+1.75)**2 - (y-1)**2)
        return np.array([grad_x, grad_y])
    monkeypatch.setattr("gradient.gradient_vector", grad)

    def func(args):
        x = args[0]
        y = args[1]
        return 1 - 0.6*np.exp(-x**2 - y**2) - 0.4*np.exp(-(x+1.75)**2 - (y-1)**2)
    monkeypatch.setattr("gradient.value", func)

    # Forcing starting point, so the function won't be stuck on flat surface
    def start_point(low, high, size=None):
        return np.array([1, 1])
    monkeypatch.setattr("gradient.np.random.uniform", start_point)
    # Assertions
    gradient = gradient_descent()
    pass  # Pass for cking the output
