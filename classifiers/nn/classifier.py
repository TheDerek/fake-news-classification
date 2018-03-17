import numpy as np


class NeuralNetwork:
    def __init__(self, shape):
        """

        :param shape: asdfsdf
        """
        self.shape = shape
        self.weights = [np.random.randn(l, shape[i]) for i, l in enumerate(shape[1:])]
        self.biases = [np.random.rand(l, 1) for l in shape[1:]]

    def sigmoid(self, x, derivative=False):
        return x*(1-x) if derivative else 1/(1+np.exp(-x))

    def feed_forward(self, X):
        X = X.T
        for layer, biases in zip(self.weights, self.biases):
            X = self.sigmoid((np.dot(layer, X) + biases))
        return X.T

    def error(self, X, y):
        z = self.feed_forward(X)
        return np.sum(0.5 * ((y - z) ** 2))

    def back_propagation(self, X, y, iterations=10):
        for i in range(iterations):
            error = self.error(X, y)
            print('Error: {}'.format(error))
        pass


if __name__ == '__main__':
    y = np.array([[1, 1, 0, 0]], dtype=np.float64).T

    X = np.array([[0, 0, 1], [1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=np.float64)

    network = NeuralNetwork([X.shape[1], 5, 2])
    network.back_propagation(X, y)

    print(network.feed_forward(X))
