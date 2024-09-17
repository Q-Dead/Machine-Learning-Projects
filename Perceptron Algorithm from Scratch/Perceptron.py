import numpy as np

class Perceptron_algo:

    def __init__(self, learning_rate=0.001, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weight = None
        self.intercept = None

    def perceptron_function(self, Z):
        return [1 if 0 <= i else 0 for i in Z]

    def theta_T_X(self, X):
        return self.intercept + np.dot(X, self.weight)

    def fit(self, X, y):
        i, n = X.shape
        self.weight = np.zeros(n)
        self.intercept = 0

        for _ in range(self.n_iter):

            lin_line = self.theta_T_X(X)
            y_hat = self.perceptron_function(lin_line)

            dbda = (1/i) * np.dot(X.T, (y_hat - y))
            dbdb = (1/i) * sum(y_hat - y)

            self.weight = self.weight - (self.learning_rate * dbda)
            self.intercept = self.intercept - (self.learning_rate * dbdb)
        

    def predict(self, X):
        lin_line = self.theta_T_X(X)
        return self.perceptron_function(lin_line)
