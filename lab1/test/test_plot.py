import pytest
import numpy as np
from src.plots import create_2D_plot, create_3D_plot


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
    monkeypatch.setattr("src.plots.value", func)


def test_2D_plot(setup2D):
    create_2D_plot()


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
    monkeypatch.setattr("src.plots.value", func)

    # Forcing starting point, so the function won't be stuck on flat surface
    def start_point(low, high, size=None):
        return np.array([1.0, 1.0])
    monkeypatch.setattr("src.gradient.np.random.uniform", start_point)


def test_3D_plot(setup3D):
    create_3D_plot()


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
    monkeypatch.setattr("src.plots.value", func)


def test_3D_plot_v2(setup3D_v2):
    create_3D_plot()
