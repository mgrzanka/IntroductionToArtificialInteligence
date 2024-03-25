from solver import Solver

import numpy as np


class GeneticAlgorithm(Solver):
    def __init__(self, pm: float, pc: float, t_max: int) -> None:
        self._mutation_probability = pm
        self._crossover_probability = pc
        self._max_iterations = t_max

    def get_parameters(self):
        return {
            "mutation probability": self._mutation_probability,
            "crossover probability": self._crossover_probability,
            "max iterations": self._max_iterations
        }

    def solve(self, evaluate: callable, individuals_num: int,
              population: np.array):
        # setup
        iteration = 0
        evaluation = np.array([evaluate(x) for x in population])
        best_individual = self.__find_the_best(evaluation, population)  # array, int value
        # loop
        while iteration < self._max_iterations:
            selected = self._select(evaluation, individuals_num, population)
            crossovered = self._crossover(selected)
            crossovered_and_mutated = self._mutate(crossovered)

            evaluation = np.array([evaluate(x)
                                   for x in crossovered_and_mutated])
            new_best_individual = self.__find_the_best(evaluation,
                                                       crossovered_and_mutated)
            if new_best_individual[1] > best_individual[1]:
                best_individual = new_best_individual

            population = crossovered_and_mutated  # generational succession
            iteration += 1
        return best_individual

    def _select(self, evaluation: np.array, individuals_num: int,
                starting_population: np.array):  # roulette selection
        # scaling
        minimum = np.min(evaluation)
        maximum = np.max(evaluation)
        if minimum == maximum:  # all values are the same, selection not necessary
            return starting_population
        scaled_evaluation = (evaluation - minimum)/(maximum-minimum)

        omega = np.sum(scaled_evaluation)
        probabilities = scaled_evaluation / omega
        new_population = []

        for _ in range(individuals_num):
            draw = np.random.choice(np.arange(individuals_num),
                                    p=probabilities)
            new_population.append(starting_population[draw])

        return np.array(new_population)

    def _crossover(self, individuals):
        # one-point crossover
        result = []
        not_used = individuals.copy()
        while (len(not_used) > 0):
            parent1, not_used = self.__choose_parent(not_used)
            # odd number of individuals in population
            if len(not_used) == 0:
                result.append(parent1)
                return np.array(result)

            parent2, not_used = self.__choose_parent(not_used)
            draw = np.random.rand()
            if draw < self._crossover_probability:
                indx = np.random.choice(np.arange(1, parent1.shape[0]))
                tmp = parent1[indx:].copy()
                parent1[indx:] = parent2[indx:]
                parent2[indx:] = tmp
            result.append(parent1)
            result.append(parent2)
        return np.array(result)

    def _mutate(self, individuals):
        for indx, individual in enumerate(individuals):
            draw = np.random.rand()
            if draw < self._mutation_probability:
                negated_individual = np.logical_not(individual).astype(int)
                individuals[indx] = negated_individual
        return np.array(individuals)

    def __find_the_best(self, evaluation: np.array, population: np.array):
        indx = np.argmax(evaluation)
        rating = evaluation[indx]
        individual = population[indx]

        return individual, rating

    def __choose_parent(self, individuals):
        indx = np.random.choice(np.arange(individuals.shape[0]))
        parent = individuals[indx]
        individuals = np.delete(individuals, indx, axis=0)
        return parent, individuals
