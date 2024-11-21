import numpy as np
from matplotlib import pyplot as plt

from solver import Solver


class GradientDescent(Solver):
    def __init__(self, step: float, starting_point=None,
                 eps=0.00000001) -> None:
        '''Hyperparameters
        :starting_point: point from where the search starts
        :eps: precision for the search
        :step: coefficient for the algorithm
        '''
        super().__init__()
        self._step = step
        self._starting_point = starting_point
        self._eps = eps

    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        return {
            "step": self._step,
            "starting point": self._starting_point,
            "eps": self._eps
        }

    def set_step(self, new_step):
        self._step = new_step

    def set_eps(self, new_eps):
        self._eps = new_eps

    def set_starting_point(self, new_starting_point):
        self._starting_point = new_starting_point

    def solve(self, function_value: callable, generate_gradient: callable,
              dimensions: int, draw=False):
        '''
        Returns local minimum od the given function and number of iteration of the algorithm.
        ----------
        :function_value: function for which minimum is being searched
        :generate_gradient: function that returns gradient of the function from function_value
        :dimensions: degree of the function form function_value
        :draw: flag to enable drawing founded points during minimum searching
        '''
        iterations = 0
        # generate random starting point if none is given
        if self._starting_point is None:
            self.set_starting_point(np.random.uniform(-10, 10, size=dimensions-1))
        gradient = generate_gradient(self._starting_point)  # gradient=array([x||x,y||x,y,z||...]) - min 1 dim
        point = self._starting_point

        # setting up plot if draw flag is set
        if draw:
            if dimensions == 2:
                self._setup_2D_plot(function_value)
            elif dimensions == 3:
                ax = self._setup_3D_plot(function_value)
            else:
                raise WrongDimensions(dimensions)

        # main loop
        while not np.linalg.norm(gradient) < self._eps:
            iterations += 1
            # Draw current point if draw flag is set
            if draw:
                if dimensions == 2:
                    point_x = point[0]
                    point_y = function_value(point)
                    plt.scatter(point_x, point_y, color='red', s=100)
                elif dimensions == 3:
                    point_x, point_y = point[0], point[1]
                    point_z = function_value(point)
                    ax.scatter(point_x, point_y, point_z, color='red', s=100)

            new_point = point - self._step*gradient  # new_point=array[x||x,y||x,y,z||...] - min 1 dim
            point = new_point
            gradient = generate_gradient(point)

        # add minimum coordinates adn show the plot
        if draw:
            if dimensions == 2:
                point_x = point[0]
                point_y = function_value(point)
                plt.text(point_x, point_y,
                         f'({point_x:.3f}, {point_y:.3f}', color='black')
            elif dimensions == 3:
                point_x, point_y = point[0], point[1]
                point_z = function_value(point)
                ax.text(point_x, point_y, point_z,
                        f'({point_x:.3f}, {point_y:.3f}, {point_z:.3f})',
                        color='black')
            plt.show()

        # Append starting point array with its corresponding value
        full_point = np.append(point, function_value(point))
        return full_point, iterations  # return array([x,y||x,y,z||...]) - min 2 dim

    def plot_function(self, function_value: callable, generate_gradient: callable,
                      dimensions: int):
        """plot function with minimum highlighted"""

        # generate random starting point if none is given
        if self._starting_point is None:
            self.set_starting_point(np.random.uniform(-10, 10, size=dimensions-1))

        if dimensions == 2:
            self._setup_2D_plot(function_value)
            point = self.solve(function_value, generate_gradient, dimensions)[0]
            point_x, point_y = point[0], point[1]
            plt.scatter(point_x, point_y, color='red', s=100)
            plt.text(point_x, point_y,
                     f'({point_x:.3f}, {point_y:.3f}', color='black')
            plt.show()
        elif dimensions == 3:
            ax = self._setup_3D_plot(function_value)
            point = self.solve(function_value, generate_gradient, dimensions)[0]
            point_x, point_y, point_z = point[0], point[1], point[2]
            ax.scatter(point_x, point_y, point_z, color='red', s=100)
            ax.text(point_x, point_y, point_z,
                    f'({point_x:.3f}, {point_y:.3f}, {point_z:.3f})',
                    color='black')
            plt.show()
        else:
            raise WrongDimensions(dimensions)

    def _setup_3D_plot(self, function_value: callable):
        '''supp function for gradient_descent function
        :function_value: function for which minimum is being searched
        '''
        x = np.arange(-10, 10, 0.1)
        y = np.arange(-10, 10, 0.1)
        surface = np.meshgrid(x, y)
        z = function_value(surface)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(*surface, z, cmap='viridis')
        fig.colorbar(surf)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        point_x = self._starting_point[0]
        point_y = self._starting_point[1]
        ax.set_title(f"Starting point: (x,y)=({point_x},{point_y})")

        return ax

    def _setup_2D_plot(self, function_value: callable):
        '''
        supp function for gradient_descent function
        :function_value: function for which minimum is being searched
        '''
        x = np.arange(-10, 10, 0.1)
        y = []
        for element in x:
            y.append(function_value([element]))
        plt.xlabel('x')
        plt.ylabel('y')
        point_x = self._starting_point[0]
        plt.title(f"Starting point: x={point_x}")
        plt.plot(x, y)


class WrongDimensions(Exception):
    def __init__(self, dim) -> None:
        super().__init__(f"Can't plot {dim}D chart")
