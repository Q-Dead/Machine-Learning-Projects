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
        self.intercept = np.zeros(k)

        for _ in range(self.n_iter):

            Z = np.dot(X, self.weight) + self.intercept
            y_hat = self.softmax(Z)

            dbda = (1/i) * np.dot(X.T, ( y_hat - y))
            dbdb = (1/i) * sum(y_hat - y)

            self.weight = self.weight - self.learning_rate * dbda
            self.intercept = self.intercept - self.learning_rate * dbdb

    def softmax(self, Z):
        i, n = Z.shape
        Z_out_container = np.empty((i, n))
        for i in range(Z.shape[0]):

            max_value = np.amax(Z[i])
            exp_ti = np.exp(Z[i] - max_value)
            Z_out_container[i] = exp_ti / np.sum(exp_ti)

        return Z_out_container
        

    def predict(self, X):
        Z = np.dot(X, self.weight) + self.intercept
        probability_pred = self.softmax(Z)
        return (probability_pred == probability_pred.max(axis=1, keepdims=True)).astype(float)
            

