import numpy as np
from scipy.stats import norm
from solver import Solver


class NaiveBayes(Solver):
    def __init__(self) -> None:
        self.class_probabilities = None
        self.means = None
        self.standard_deviations = None

        self.samples_number = None
        self.sample_length = None
        self.classes = None

    def _separate_classes(self, X: np.ndarray, Y: np.ndarray) -> dict[int, list[np.ndarray]]:
        """
        Split dataset into parts with corresponding, one class.
        dict: class - key, samples with this class - value
        """
        X_Y_splitted = {}

        # Giving each class y corresponding data
        for x, y in zip(X, Y):
            if y not in X_Y_splitted.keys():
                X_Y_splitted[y] = []
            X_Y_splitted[y].append(x)

        return X_Y_splitted

    def _get_class_probabilities(self, X_Y_splitted: dict[int, list[np.ndarray]]) -> np.ndarray:
        """
        Get probability of each class from dataset
        """
        classes = list(X_Y_splitted.keys())
        classes_probabilities = np.zeros(len(self.classes))  # initialize classes probabilities

        # for each class calculate number of elements with this class / number of elements in dataset
        for indx, cls in enumerate(classes):
            samples_in_class_number = len(X_Y_splitted[cls])
            classes_probabilities[indx] = samples_in_class_number / self.samples_number

        return classes_probabilities

    def _get_mean_std(self, X_Y_splitted: dict[int, list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        """
        Get matrix of means and standard deviations for normal distribution
        """
        # row represents class, column - a xi feature
        means = np.zeros(shape=(len(self.classes), self.sample_length))
        standard_deviations = np.zeros(shape=(len(self.classes), self.sample_length))

        # get arrays of mean and standard deviation for each class
        for indx, cls in enumerate(self.classes):
            class_means = np.zeros(self.sample_length)
            class_std = np.zeros(self.sample_length)

            data = X_Y_splitted[cls]

            for column_index in range(self.sample_length):
                column_data = np.array([row[column_index] for row in data])
                class_means[column_index] = np.mean(column_data)
                class_std[column_index] = np.std(column_data)

            # add new row (one row = one class)
            means[indx] = class_means
            standard_deviations[indx] = class_std

        return means, standard_deviations

    def fit(self, X, y):
        """
        Train the model for given dataset
        """
        self.samples_number = len(X)
        self.sample_length = len(X[0])

        X_Y_splitted = self._separate_classes(X, y)
        self.classes = list(X_Y_splitted.keys())

        self.class_probabilities = self._get_class_probabilities(X_Y_splitted)
        self.means, self.standard_deviations = self._get_mean_std(X_Y_splitted)

    def _single_class_probability(self, x: np.ndarray, class_indx):
        """
        Get probability of x being class with given index
        """
        log_probs = np.zeros(x.shape)
        for xi_indx, xi in enumerate(x):
            mean = self.means[class_indx][xi_indx]
            std = self.standard_deviations[class_indx][xi_indx]
            log_probs[xi_indx] = norm.logpdf(xi, loc=mean, scale=std)  # log to eliminate very small numbers

        return np.sum(log_probs)

    def _single_predict(self, x):
        the_best_log_probability = float("-inf")
        the_best_class = None

        for indx, clc in enumerate(self.classes):
            log_probability = np.log(self.class_probabilities[indx]) + \
                self._single_class_probability(x, indx)

            if log_probability > the_best_log_probability:
                the_best_log_probability = log_probability
                the_best_class = clc

        return the_best_class

    def predict(self, X: np.ndarray):
        predictions = np.zeros(len(X))
        for indx, x in enumerate(X):
            predicted_class = self._single_predict(x)
            predictions[indx] = predicted_class

        return predictions
