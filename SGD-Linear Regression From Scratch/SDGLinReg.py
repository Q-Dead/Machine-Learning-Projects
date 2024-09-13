import numpy as np
from sklearn import datasets

class SGDLinearRegression:

    def __init__(self, learning_rate=0.001, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.intercept = None
        self.slope = None
    
    def gradient_descent(self, X, y):

        yhat = self.intercept + np.dot(X, self.slope)

        dbda = (1/len(X)) * np.dot(X.T, ( yhat - y))
        dbdb = (1/len(X)) * sum(yhat - y)

        return dbda, dbdb

    def fit(self, X, y):
        self.intercept = 0
        self.slope = 0
        for _ in range(self.n_iter):
            random_index = np.random.randint(0, len(X))

            random_X = X[random_index]
            random_y = y[random_index]

            dbda, dbdb = self.gradient_descent(np.array(random_X), np.array(random_y))

            self.slope = self.slope - (self.learning_rate * dbda)
            self.intercept = self.intercept - (self.learning_rate * dbdb)

    def predict(self, X):
        return self.intercept + np.dot(X, self.slope)
