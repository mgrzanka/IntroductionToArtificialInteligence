import numpy as np
from activation_functions import ActivationFunctions


class NeuralNetwork:
    def __init__(self, mini_batch_length=100, learning_rate=0.1, epochs=10, layers_sizes=None, activation_functions=None):
        """
        weights = list of matrixes (one matrix is weights for one layer)
        biases = list of vectors (one vector is bias for one layer)
        """
        self.weights = []
        self.biases = []

        self.a_calculator = ActivationFunctions()
        self._mini_batch_len = mini_batch_length
        self.learning_rate = learning_rate
        self.epochs = epochs

        if layers_sizes is None and activation_functions is None:
            self._a_functions = ["sigmoid", "sigmoid", "sigmoid"]
            self._layers_sizes = [16, 16, 10]
        else:
            self._a_functions = activation_functions
            self._layers_sizes = layers_sizes

    def initialize_params(self, n_features: int) -> None:
        previous = n_features
        for layer_size in self._layers_sizes:
            self.biases.append(np.zeros(layer_size))
            self.weights.append(np.random.randn(layer_size, previous) * np.sqrt(2 / previous))
            previous = layer_size

    def split_to_minibatches(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray]:
        """Split dataset to some number of minibatches (with given length)"""
        indices = np.random.permutation(X.shape[0])
        X, y = X[indices], y[indices]
        mini_batch_num = X.shape[0] / self._mini_batch_len
        start = 0
        mini_batches = []
        mini_batches_y = []
        for i in range(int(mini_batch_num)):
            end = start + self._mini_batch_len
            end = len(y) if end > len(y) else end
            mini_batches.append(X[start:end, :])
            mini_batches_y.append(y[start:end])
            start = end
        return mini_batches, mini_batches_y

    def _get_activations(self, x: np.ndarray) -> list[np.ndarray]:
        """calculates activations in hidden layers"""
        activations = []
        activations.append(x)
        for w, b, a_func in zip(self.weights[:-1], self.biases[:-1], self._a_functions):
            x = self.a_calculator.calculate(np.dot(w, x) + b, a_func)
            activations.append(x)
        return activations

    def get_desired_output(self, y: np.ndarray) -> np.ndarray:
        output = np.zeros(10)
        output[y] = 1
        return output

    def backprop(self, x: np.ndarray, y: int) -> tuple[list[np.ndarray]]:
        """Calculates gradients for weights and biases for one sample"""
        activations = self._get_activations(x)
        dCda = 2 * (self.feedforward(x) - self.get_desired_output(y))
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        for i in range(1, len(self.weights) + 1):
            w = self.weights[-i]
            b = self.biases[-i]
            a = activations[-i]
            a_func = self._a_functions[-i]
            z = np.dot(w, a) + b
            delta = self.a_calculator.calculate(z, a_func, derivitave=True) * dCda
            grad_b[-i] = delta
            grad_w[-i] = np.outer(delta, a)
            dCda = np.dot(w.T, delta)

        return grad_w, grad_b

    def gradient_for_minibatch(self, minibatch: np.ndarray, minibatch_y: np.ndarray):
        """Calculates gradients for weights and biases for minibatch"""
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in zip(minibatch, minibatch_y):
            dw, db = self.backprop(x, y)
            grad_b = [gb + d for gb, d in zip(grad_b, db)]
            grad_w = [gw + d for gw, d in zip(grad_w, dw)]
        return grad_w, grad_b

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        """Return output of neural network with given input a"""
        for w, b, a_func in zip(self.weights, self.biases, self._a_functions):
            a = self.a_calculator.calculate(np.dot(w, a) + b, a_func)
        return a

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """fits model to given data"""
        n_samples, n_features = X.shape
        self.initialize_params(n_features)
        for epoch_number in range(self.epochs):
            minibatches, minibatches_y = self.split_to_minibatches(X, y)
            for minibatch, minibatch_y in zip(minibatches, minibatches_y):
                grad_w, grad_b = self.gradient_for_minibatch(minibatch, minibatch_y)
                self.weights = [current_w - self.learning_rate/len(minibatch) * w for current_w, w in zip(self.weights, grad_w)]
                self.biases = [current_b - self.learning_rate/len(minibatch) * b for current_b, b in zip(self.biases, grad_b)]
            print(f'{epoch_number+1}/{self.epochs} epochs completed')

    def predict_proba(self, x: np.ndarray) -> int:
        """predict one sample"""
        result = self.feedforward(x)
        return np.argmax(result)

    def predict(self, X: np.ndarray) -> list[int]:
        """make prediction for trained model"""
        preds = []
        for x in X:
            preds.append(self.predict_proba(x))
        return preds
