from LearningModel import LearningModel
import matplotlib.pyplot as plt
import numpy as np


def test_model():
    model = LearningModel(0.95, 1, 1000, 500, 6, 1)
    model.fit()
    model.play()
    pass


def experiment_episodes_iterations():
    episodes = [100, 300, 700, 1000, 4000, 10000, 20000]
    discount_factors = np.linspace(0, 1, 10)
    Ts = np.linspace(0.1, 5, 8)
    learning_rates = np.linspace(0.01, 1.2, 8)
    iterations = []
    for t in Ts:
        iterations_sum = 0
        model = LearningModel(0.95, t, 3000, 500, 6, 0.5)
        model.fit()

        for _ in range(25):
            result, i = model.play()
            iterations_sum += i

        iterations.append(iterations_sum / 25)

    plt.plot(Ts, iterations)
    plt.title("The impact of T")
    plt.xlabel("T")
    plt.ylabel("Iterations")
    plt.show()


def experiment_episodes_efficiency():
    episodes = [100, 300, 700, 1000, 4000, 10000]
    discount_factors = np.linspace(0, 1, 10)
    Ts = np.linspace(0.1, 5, 8)
    learning_rates = np.linspace(0.01, 1.2, 8)
    efficiencies = []
    for t in Ts:
        results_sum = 0
        model = LearningModel(0.95, t, 3000, 500, 6, 0.5)
        model.fit()

        for _ in range(25):
            result, i = model.play()
            efficiency = 1 if result == 20 else 0
            results_sum += efficiency

        efficiencies.append(results_sum / 25)

    plt.plot(Ts, efficiencies)
    plt.title("The impact of T")
    plt.xlabel("T")
    plt.ylabel("Efficiency")
    plt.show()


if __name__ == "__main__":
    test_model()
