import numpy as np

class MultiClassLogisticRegression:

    def __init__(self, learning_rate=0.001, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weight = None
        self.intercept = None

    def fit(self, X, y):
        i, n = X.shape
        j, k = y.shape
        self.weight = np.zeros((n, k))
        self.intercept = 0

        for _ in range(self.n_iter):

            Z = X @ self.weight
            y_hat = self.softmax(Z)

            dbdw = (1/i) * np.dot(X.T, ( y_hat - y))

            self.weight = self.weight - self.learning_rate * dbdw

    def softmax(self, Z):

        max_value = np.amax(Z)
        exp_ti = np.exp(Z - max_value)
        return exp_ti / np.sum(exp_ti)

    def predict(self, X):
        Z = X @ self.weight
        return self.softmax(Z)

