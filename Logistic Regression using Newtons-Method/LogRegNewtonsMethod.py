import numpy as np


class NMLogisticRegression:

    def __init__(self, n_iter=50):
        self.n_iter = n_iter
        self.weight = None
        self.intercept = None

    def fit(self, X, y):
        i, n = X.shape
        self.weight = np.zeros(n)
        self.intercept = 0

        for _ in range(self.n_iter):

            # TODO make and Augmented X and weight so that there is a bisa
            Z = np.dot(X, self.weight) + self.intercept
            yhat = self.sigmoid(Z)
            
            gradient = (1/n) *  np.dot(X.T, (yhat - y))

            diag = np.diag(yhat * (1 - yhat))
            hessian  = (1/n) * np.dot(np.dot(X.T, diag), X)

            self.weight = self.weight - np.dot(np.linalg.pinv(hessian), gradient)


    def predict(self, X):
        Z = np.dot(X, self.weight) + self.intercept
        probability = self.sigmoid(Z)
        return np.array([1 if i > 0.5 else 0 for i in probability])

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    