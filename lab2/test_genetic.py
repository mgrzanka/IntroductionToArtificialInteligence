from genetic import GeneticAlgorithm
import numpy as np
import matplotlib.pyplot as plt
from random import random

from evaluate import evaluate


# def test_generic_algorithm():
#     # hiperparameters
#     pm = 0.1
#     pc = 0.8
#     t_max = 500
#     individuals_num = 500
#     population = np.random.choice([0, 1], size=(500, 100), p=[0.5, 0.5])

#     # algorithm initation
#     solver = GeneticAlgorithm(pm, pc, t_max)
#     best_individual = solver.solve(evaluate, individuals_num, population)
#     print(solver.get_parameters())
#     print("Individuals number:", individuals_num)
#     print("Result: ", best_individual[1])


# test_generic_algorithm()


# def test_individuals_number_small():
#     # constnt hiperparameters
#     pm = 0.1
#     pc = 0.8
#     t_max = 500
#     # small poplation
#     individuals_num = 5
#     population = np.random.choice([0, 1], size=(5, 100), p=[0.5, 0.5])
#     solver = GeneticAlgorithm(pm, pc, t_max)
#     results_small = []
#     for _ in range(20):
#         results_small.append(solver.solve(evaluate, individuals_num, population)[1])
#     iterations = np.arange(0, 20)
#     plt.plot(iterations, np.array(results_small),
#              label=f"Population {individuals_num} individuals")
#     mean = np.mean(np.array(results_small))
#     mean_plot = np.full(iterations.shape, mean)
#     plt.plot(iterations, mean_plot, label="Mean")
#     plt.xlabel('Iterations')
#     plt.ylabel('Founded maximum')
#     plt.title("Individuals number impact")
#     plt.legend()
#     plt.show()


# def test_test_individuals_number_medium():
#     # constnt hiperparameters
#     pm = 0.1
#     pc = 0.8
#     t_max = 500
#     individuals_num = 250
#     population = np.random.choice([0, 1], size=(250, 100), p=[0.5, 0.5])
#     solver = GeneticAlgorithm(pm, pc, t_max)
#     results_medium = []
#     for _ in range(20):
#         results_medium.append(solver.solve(evaluate, individuals_num, population)[1])
#     iterations = np.arange(0, 20)
#     plt.plot(iterations, np.array(results_medium),
#              label=f'Population, {individuals_num} individuals')
#     mean = np.mean(np.array(results_medium))
#     mean_plot = np.full(iterations.shape, mean)
#     plt.plot(iterations, mean_plot, label="Mean")
#     plt.xlabel('Iterations')
#     plt.ylabel('Founded maximum')
#     plt.title("Individuals number impact")
#     plt.legend()
#     plt.show()


# def test_test_individuals_number_big():
#     # constnt hiperparameters
#     pm = 0.1
#     pc = 0.8
#     t_max = 500
#     individuals_num2 = 1000
#     population = np.random.choice([0, 1], size=(1000, 100), p=[0.5, 0.5])
#     solver = GeneticAlgorithm(pm, pc, t_max)
#     results_big = []
#     for _ in range(20):
#         results_big.append(solver.solve(evaluate, individuals_num2, population))
#     iterations = np.arange(0, 20)
#     # plotting
#     plt.plot(iterations, np.array(results_big),
#              label=f'Population {individuals_num2} individuals')
#     plt.xlabel('Iterations')
#     plt.ylabel('Founded maximum')
#     plt.title("Individuals number impact")
#     plt.legend()
#     plt.show()


def test_600_individuals():
    # constnt hiperparameters
    pm = 0.1
    pc = 0.8
    t_max = 500
    # very big poplation
    individuals_num = 600
    population = np.random.choice([0, 1], size=(600, 100), p=[0.5, 0.5])
    solver = GeneticAlgorithm(pm, pc, t_max)
    result = solver.solve(evaluate, individuals_num, population)
    print(result[1])


test_600_individuals()


# def test_crossover_probability():
#     # const hiperparameters
#     pm = 0.1
#     t_max = 500
#     individuals_num = 500
#     population = np.random.choice([0, 1], size=(500, 100), p=[0.5, 0.5])
#     # small pc
#     pc0 = 0.1
#     results_small = []
#     solver = GeneticAlgorithm(pm, pc0, t_max)
#     for _ in range(20):
#         results_small.append(solver.solve(evaluate, individuals_num, population))
#     # medium pc
#     pc1 = 0.5
#     results_medium = []
#     solver = GeneticAlgorithm(pm, pc1, t_max)
#     for _ in range(20):
#         results_medium.append(solver.solve(evaluate, individuals_num, population))
#     # big pc
#     pc2 = 0.95
#     results_big = []
#     solver = GeneticAlgorithm(pm, pc2, t_max)
#     for _ in range(20):
#         results_big.append(solver.solve(evaluate, individuals_num, population))
#     # plotting
#     iterations = np.arange(20)
#     plt.plot(iterations, np.array(results_small), label=f"Probability {pc0}")
#     plt.plot(iterations, np.array(results_medium), label=f"Probability {pc1}")
#     plt.plot(iterations, np.array(results_big), label=f"Probability {pc2}")
#     plt.xlabel('Iteration')
#     plt.ylabel('Founded maximum')
#     plt.title("Corssover probability impact")
#     plt.legend()
#     plt.show()


# def test_mutation_probability():
#     # const hiperparameters
#     pc = 0.8
#     t_max = 500
#     individuals_num = 500
#     population = np.random.choice([0, 1], size=(500, 100), p=[0.5, 0.5])
#     # small pm
#     pm0 = 0.05
#     results_small = []
#     solver = GeneticAlgorithm(pm0, pc, t_max)
#     for _ in range(20):
#         results_small.append(solver.solve(evaluate, individuals_num, population))
#     # medium pm
#     pm1 = 0.3
#     results_medium = []
#     solver = GeneticAlgorithm(pm1, pc, t_max)
#     for _ in range(20):
#         results_medium.append(solver.solve(evaluate, individuals_num, population))
#     # big pc
#     pm2 = 0.8
#     results_big = []
#     solver = GeneticAlgorithm(pm2, pc, t_max)
#     for _ in range(20):
#         results_big.append(solver.solve(evaluate, individuals_num, population))
#     # plotting
#     iterations = np.arange(20)
#     plt.plot(iterations, np.array(results_small), label=f"Probability {pm0}")
#     plt.plot(iterations, np.array(results_medium), label=f"Probability {pm1}")
#     plt.plot(iterations, np.array(results_big), label=f"Probability {pm2}")
#     plt.xlabel('Iteration')
#     plt.ylabel('Founded maximum')
#     plt.title("Mutation probability impact")
#     plt.legend()
#     plt.show()


# def test_t_max():
#     # const hiperparameters
#     pc = 0.8
#     individuals_num = 500
#     population = np.random.choice([0, 1], size=(500, 100), p=[0.5, 0.5])
#     pm = 0.1
#     # small t_max
#     t_max0 = 50
#     results_small = []
#     solver = GeneticAlgorithm(pm, pc, t_max0)
#     for _ in range(20):
#         results_small.append(solver.solve(evaluate, individuals_num, population))
#     # medium t_max
#     t_max1 = 250
#     results_medium = []
#     solver = GeneticAlgorithm(pm, pc, t_max1)
#     for _ in range(20):
#         results_medium.append(solver.solve(evaluate, individuals_num, population))
#     # big t_max
#     t_max2 = 1000
#     results_big = []
#     solver = GeneticAlgorithm(pm, pc, t_max2)
#     for _ in range(20):
#         results_big.append(solver.solve(evaluate, individuals_num, population))
#     # plotting
#     iterations = np.arange(20)
#     plt.plot(iterations, np.array(results_small), label=f"Iterations {t_max0}")
#     plt.plot(iterations, np.array(results_medium), label=f"Iterations {t_max1}")
#     plt.plot(iterations, np.array(results_big), label=f"Iterations {t_max2}")
#     plt.xlabel('Iteration')
#     plt.ylabel('Founded maximum')
#     plt.title("Maximum iterations number impact")
#     plt.legend()
#     plt.show()


# def test_select():
#     solver = GeneticAlgorithm(0.01, 0.8, 100)
#     evaluation = np.array([1,2,3,4])
#     n = 4
#     population = np.random.choice([0, 1], size=(4, 3), p=[0.5, 0.5])
#     new_population = solver._select(evaluation, n, population)


# def test_genetic_operations():
#     solver = GeneticAlgorithm(0.8, 0.8, 100)
#     population = np.random.choice([0, 1], size=(4, 5), p=[0.5, 0.5])
#     crossovered = solver._crossover(population)
#     mutated = solver._mutate(population)


# def evalueate(x):
#     ratings = [random() for _ in x]
#     sumary = 0
#     for indx, item in enumerate(x):
#         if item == 1:
#             sumary += ratings[indx]
#     if sumary > 2:
#         return 0
#     else:
#         return sumary


# def test_algorithm():
#     solver = GeneticAlgorithm(0.1, 0.8, 500)
#     population = np.random.choice([0, 1], size=(20, 10), p=[0.5, 0.5])
#     best = solver.solve(evalueate, 20, population)
#     pass
