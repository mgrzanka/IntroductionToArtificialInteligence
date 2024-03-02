import numpy as np
from plots import create_3D_plot, create_2D_plot


def test_3D_plot(monkeypatch):
    # Patching
    def func(args):
        x = args[0]
        y = args[1]
        return 1 - 0.6*np.exp(-x**2 - y**2) - 0.4*np.exp(-(x+1.75)**2 - (y-1)**2)
    monkeypatch.setattr("plots.value", func)
    monkeypatch.setattr("gradient.value", func)

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

    def start_point(low, high, size=None):
        return np.array([1, 1])
    monkeypatch.setattr("gradient.np.random.uniform", start_point)

    # Testing
    create_3D_plot()


def test_2D_plot(monkeypatch):
    # Patching
    def func(args):
        x = args[0]
        return 2*x**2 + 3*x - 1
    monkeypatch.setattr("plots.value", func)
    monkeypatch.setattr("gradient.value", func)

    def dim():
        return 2
    monkeypatch.setattr("gradient.dimensions", dim)

    def grad(args):
        x = args[0]
        return np.array([4*x + 3])
    monkeypatch.setattr("gradient.gradient_vector", grad)

    # Testing
    create_2D_plot()
