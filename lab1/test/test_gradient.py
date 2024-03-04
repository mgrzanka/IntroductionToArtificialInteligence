import numpy as np
import pytest
from src.gradient import gradient_descent, examine_step_value


@pytest.fixture
def setup2D(monkeypatch):
    def dim():
        return 2
    monkeypatch.setattr("src.gradient.dimensions", dim)

    def grad(args):
        x = args[0]
        return np.array([4*x + 3])
    monkeypatch.setattr("src.gradient.gradient_vector", grad)

    def func(args):
        x = args[0]
        return 2*x**2 + 3*x - 1
    monkeypatch.setattr("src.gradient.value", func)


def test_2D_function(setup2D):
    minimum = gradient_descent()
    assert -0.75 == pytest.approx(minimum[0])
    assert -2.125 == pytest.approx(minimum[1])


def test_examine_step_value_2D(setup2D):
    examine_step_value()  # Check results.json


@pytest.fixture
def setup3D(monkeypatch):
    def dim():
        return 3
    monkeypatch.setattr("src.gradient.dimensions", dim)

    def grad(args):
        x = args[0]
        y = args[1]
        grad_x = 1.2*x*np.exp(-x**2-y**2) + 0.8*(x+1.75)*np.exp(-(x+1.75)**2 - (y-1)**2)
        grad_y = 1.2*y*np.exp(-x**2-y**2) + 0.8*(y-1)*np.exp(-(x+1.75)**2 - (y-1)**2)
        return np.array([grad_x, grad_y])
    monkeypatch.setattr("src.gradient.gradient_vector", grad)

    def func(args):
        x = args[0]
        y = args[1]
        return 1 - 0.6*np.exp(-x**2 - y**2) - 0.4*np.exp(-(x+1.75)**2 - (y-1)**2)
    monkeypatch.setattr("src.gradient.value", func)

    # Forcing starting point, so the function won't be stuck on flat surface
    def start_point(low, high, size=None):
        return np.array([1.0, 1.0])
    monkeypatch.setattr("src.gradient.np.random.uniform", start_point)


def test_3D_function(setup3D):
    minimum = gradient_descent()
    pass  # Pass for cheking the output


def test_examine_step_value_3D(setup3D):
    examine_step_value()


@pytest.fixture
def setup3D_v2(monkeypatch):
    def dim():
        return 3
    monkeypatch.setattr("src.gradient.dimensions", dim)

    def grad(args):
        x = args[0]
        y = args[1]
        grad_x = 2*x
        grad_y = 2*y
        return np.array([grad_x, grad_y])
    monkeypatch.setattr("src.gradient.gradient_vector", grad)

    def func(args):
        x = args[0]
        y = args[1]
        return x**2 + y**2
    monkeypatch.setattr("src.gradient.value", func)


def test_dim_func(setup3D_v2):
    minimum = gradient_descent()
    assert 0 == pytest.approx(minimum[0])
    assert 0 == pytest.approx(minimum[1])
    assert 0 == pytest.approx(minimum[2])


def test_examine_step_value_3Dv2(setup3D_v2):
    examine_step_value()
