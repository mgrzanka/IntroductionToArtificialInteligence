from genetic import GeneticAlgorithm
import numpy as np
import matplotlib.pyplot as plt
from random import random

from evaluate import evaluate


def test_generic_algorithm():
    # hiperparameters
    pm = 0.1
    pc = 0.8
    t_max = 500
    individuals_num = 25
    population = np.random.choice([0, 1], size=(25, 100), p=[0.5, 0.5])

    # algorithm initation
    solver = GeneticAlgorithm(pm, pc, t_max)
    best_individual = solver.solve(evaluate, individuals_num, population)
    print(solver.get_parameters())
    print("Individuals number:", individuals_num)
    print("Result: ", best_individual[1])


test_generic_algorithm()


def test_select():
    solver = GeneticAlgorithm(0.01, 0.8, 100)
    evaluation = np.array([1,2,3,4])
    n = 4
    population = np.random.choice([0, 1], size=(4, 3), p=[0.5, 0.5])
    new_population = solver._select(evaluation, n, population)


def test_genetic_operations():
    solver = GeneticAlgorithm(0.8, 0.8, 100)
    population = np.random.choice([0, 1], size=(4, 5), p=[0.5, 0.5])
    crossovered = solver._crossover(population)
    mutated = solver._mutate(population)


def evalueate(x):
    ratings = [random() for _ in x]
    sumary = 0
    for indx, item in enumerate(x):
        if item == 1:
            sumary += ratings[indx]
    if sumary > 2:
        return 0
    else:
        return sumary


def test_algorithm():
    solver = GeneticAlgorithm(0.1, 0.8, 500)
    population = np.random.choice([0, 1], size=(20, 10), p=[0.5, 0.5])
    best = solver.solve(evalueate, 20, population)
    pass
