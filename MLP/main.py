import pandas as pd
import numpy as np

from mlp import NeuralNetwork


def scale_data(X_train: np.ndarray, X_test: np.ndarray):
    x_min = np.min(X_train, axis=0)
    x_max = np.max(X_train, axis=0)
    range_ = x_max - x_min
    range_[range_ == 0] = 1
    X_train_scaled = (X_train - x_min) / range_
    X_test_scaled = (X_test - x_min) / range_
    return X_train_scaled, X_test_scaled


def get_accurancy(preds: np.ndarray, y_test: np.ndarray):
    s = 0
    for pred, y in zip(preds, y_test):
        s = s + 1 if y == pred else s
    return s/len(y_test)


if __name__ == "__main__":
    network = NeuralNetwork()
    df_train = pd.read_csv('MNIST_dataset/mnist_train.csv')
    df_test = pd.read_csv('MNIST_dataset/mnist_test.csv')

    X_train = df_train.drop(labels="label", axis=1).to_numpy()
    y_train = df_train["label"].to_numpy()
    X_test = df_test.drop(labels="label", axis=1).to_numpy()
    y_test = df_test["label"].to_numpy()
    X_train, X_test = scale_data(X_train, X_test)

    network.fit(X_train, y_train)
    preds = network.predict(X_test)
    accurancy = get_accurancy(preds, y_test)
    print(accurancy)
