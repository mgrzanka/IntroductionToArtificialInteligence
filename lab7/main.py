from sklearn.datasets import load_breast_cancer
from NaiveBayesClassificator import NaiveBayes
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt


def get_test_split(data, target):
    return train_test_split(data, target, test_size=0.15, random_state=42)


def evaluate_model(predictions, target):
    correct_predictions = sum(pred == targ for pred, targ in zip(predictions, target))
    accuracy = correct_predictions / len(target)
    return accuracy


def standard_validation(data, target, random_state):
    data, validation, target, validation_target = train_test_split(data, target, test_size=0.2, random_state=random_state)
    model = NaiveBayes()
    model.fit(data, target)
    predictions = model.predict(validation)
    return evaluate_model(predictions, validation_target)


def cross_validation(data, target, k):
    model = NaiveBayes()
    accurancies = np.zeros(k)

    # split dataset on k subsets
    splitted_data = np.array_split(data, k)
    splitted_target = np.array_split(target, k)

    # each one of the subsets can be a validation set
    for i in range(k):
        # split data
        validation_data = splitted_data[i]
        validation_target = splitted_target[i]
        training_data = [np.array(single_data) for indx, single_data in
                         enumerate(splitted_data) if indx != i]
        training_target = [np.array(single_data) for indx, single_data in
                           enumerate(splitted_target) if indx != i]
        training_data = np.concatenate(training_data)
        training_target = np.concatenate(training_target)

        # train model
        model.fit(training_data, training_target)

        # predict
        predictions = model.predict(validation_data)
        accurancies[i] = evaluate_model(predictions, validation_target)

    return np.mean(accurancies)


def test_validation(data, target):
    stds = np.zeros((20, 2))
    for i in range(20):
        stds[i][0] = standard_validation(data, target, np.random.randint(120))
        stds[i][1] = cross_validation(data, target, np.random.randint(3, 10))
        print(i+1)

    categories = ["stndardowy podział", "walidacja krzyżowa"]

    plt.bar(categories, [np.std(stds[:, 0]), np.std(stds[:, 1])])
    plt.title("Średnie odchylenia standardowe dla oceny jakości predykcji różnymi metodami")
    plt.xlabel("Sposób oceny jakości")
    plt.ylabel("Średnia wartość odchylenia standardowego")
    plt.show()


def get_accurancy_on_test_data(data, target, test_data, test_target):
    model = NaiveBayes()
    model.fit(data, target)
    predictions = model.predict(test_data)
    return evaluate_model(predictions, test_target)


if __name__ == "__main__":
    d = load_breast_cancer()
    data, test_data, target, test_target = train_test_split(d.data, d.target, test_size=0.1, random_state=42)
    # test_validation(data, target)
    print(get_accurancy_on_test_data(data, target, test_data, test_target))
