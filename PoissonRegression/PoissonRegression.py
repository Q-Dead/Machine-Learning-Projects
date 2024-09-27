import numpy as np

class PoissonRegressionAlgo:

    def __init__(self, learning_rate=0.001, n_iter=10000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weight = None
        self.intercept = None

    def fit(self, X, y):
        i , n = X.shape
        self.weight = np.zeros(n)
        self.intercept = 0.0
        
        for _ in range(self.n_iter):
            
            eta = self.thetaTX(X)
            lmbda = self.poisson(eta)

            dbda = (1/i) * np.dot(X.T, (lmbda - y))
            dbdb = (1/i) * np.sum(lmbda - y)

            self.weight = self.weight - self.learning_rate * dbda
            self.intercept = self.intercept - self.learning_rate * dbdb

    def predict(self, X):

        eta = self.thetaTX(X)
        return self.poisson(eta)

    def thetaTX(self, X):
        return X @ self.weight + self.intercept

    def poisson(self, Z):
        return np.exp(Z)