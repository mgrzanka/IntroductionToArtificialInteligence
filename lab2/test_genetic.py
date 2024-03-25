from genetic import GeneticAlgorithm
import numpy as np
import matplotlib.pyplot as plt
from evaluate import evaluate


def test_generic_algorithm():
    # hiperparameters
    pm = 0.05
    pc = 0.95
    t_max = 800
    individuals_num = 50
    population = np.random.choice([0, 1], size=(50, 100), p=[0.5, 0.5])

    # algorithm initation
    solver = GeneticAlgorithm(pm, pc, t_max)
    best_individual = solver.solve(evaluate, individuals_num, population)
    print(solver.get_parameters())
    print("Individuals number:", individuals_num)
    print("Result: ", best_individual[1])


test_generic_algorithm()


def test_individuals_number():
    # hiperparameters
    pm = 0.1
    pc = 0.8
    t_max = 500
    individuals_num = 5
    population = np.random.choice([0, 1], size=(5, 100), p=[0.5, 0.5])
    solver = GeneticAlgorithm(pm, pc, t_max)
    results = []
    # 20 tests
    for x in range(20):
        y = solver.solve(evaluate, individuals_num, population)[1]
        results.append(y)
        plt.scatter(x, y, color='blue', marker='o')
        plt.text(x, y+0.1, f"({x},{y})")
    iterations = np.arange(0, 20)
    mean = np.mean(np.array(results))
    mean_plot = np.full(iterations.shape, mean)
    plt.plot(iterations, mean_plot, label="Mean")
    plt.xlabel('Iterations')
    plt.ylabel('Founded maximum')
    plt.title("Individuals number impact")
    plt.legend()
    plt.show()



def test_crossover_mutation_probability():
    # hiperparameters
    pm = 0
    pc = 1
    t_max = 500
    individuals_num = 50
    population = np.random.choice([0, 1], size=(50, 100), p=[0.5, 0.5])
    results = []
    solver = GeneticAlgorithm(pm, pc, t_max)
    # 20 tests
    for x in range(20):
        y = solver.solve(evaluate, individuals_num, population)[1]
        results.append(y)
        plt.scatter(x, y, color='red', marker='o')
        plt.text(x, y+0.1, f"({x},{y})")
    iterations = np.arange(0, 20)
    mean = np.mean(results)
    plt.plot(iterations, np.full(iterations.shape, mean), label = "Mean")
    plt.xlabel('Iteration')
    plt.ylabel('Founded maximum')
    plt.title("Corssover probability impact")
    plt.legend()
    plt.show()



def test_t_max_small():
    # const hiperparameters
    pc = 0.9
    individuals_num = 25
    population = np.random.choice([0, 1], size=(25, 100), p=[0.5, 0.5])
    pm = 0.1
    # small t_max
    t_max0 = 25
    results_small = []
    solver = GeneticAlgorithm(pm, pc, t_max0)
    for x in range(20):
        y = solver.solve(evaluate, individuals_num, population)[1]
        results_small.append(y)
        plt.scatter(x, y, color='red', marker='o')
        plt.text(x, y+0.1, f"({x},{y})")
    iterations = np.arange(20)
    mean = np.mean(results_small)
    mean_plot = np.full(iterations.shape, mean)
    plt.plot(iterations, mean_plot, label="Mean")
    plt.xlabel('Iteration')
    plt.ylabel('Founded maximum')
    plt.title("Maximum iterations number impact")
    plt.legend()
    plt.show()


def test_t_max_medium():
    # const hiperparameters
    pc = 0.9
    individuals_num = 25
    population = np.random.choice([0, 1], size=(25, 100), p=[0.5, 0.5])
    pm = 0.1
    # medium t_max
    t_max1 = 500
    results_medium = []
    solver = GeneticAlgorithm(pm, pc, t_max1)
    for x in range(20):
        y = solver.solve(evaluate, individuals_num, population)[1]
        results_medium.append(y)
        plt.scatter(x, y, color='red', marker='o')
        plt.text(x, y+0.1, f"({x},{y})")
    iterations = np.arange(20)
    mean = np.mean(results_medium)
    mean_plot = np.full(iterations.shape, mean)
    plt.plot(iterations, mean_plot, label="Mean")
    plt.xlabel('Iteration')
    plt.ylabel('Founded maximum')
    plt.title("Maximum iterations number impact")
    plt.legend()
    plt.show()


def test_t_max_big():
    # const hiperparameters
    pc = 0.9
    individuals_num = 25
    population = np.random.choice([0, 1], size=(25, 100), p=[0.5, 0.5])
    pm = 0.1
    # medium t_max
    t_max1 = 900
    results_big = []
    solver = GeneticAlgorithm(pm, pc, t_max1)
    for x in range(20):
        y = solver.solve(evaluate, individuals_num, population)[1]
        results_big.append(y)
        plt.scatter(x, y, color='red', marker='o')
        plt.text(x, y+0.1, f"({x},{y})")
    iterations = np.arange(20)
    mean = np.mean(results_big)
    mean_plot = np.full(iterations.shape, mean)
    plt.plot(iterations, mean_plot, label="Mean")
    plt.xlabel('Iteration')
    plt.ylabel('Founded maximum')
    plt.title("Maximum iterations number impact")
    plt.legend()
    plt.show()
