import numpy as np


class SDGBatchLinearRegression:

    def __init__(self, learning_rate=0.01, n_iter=1000, batch_size=32):

        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.intercept = None
        self.slope = None

    def gradientDescent(self, X, y):
        
        yhat = self.intercept + np.dot(X, self.slope)

        dbda = (1/len(X)) * np.dot(X.T, (yhat - y))
        dbdb = (1/len(X)) * sum(yhat - y)

        return dbda, dbdb

    def fit(self, X, y):
        n_items, n_features = X.shape
        self.intercept = 0
        self.slope = np.zeros(n_features)
        for _ in range(self.n_iter):

            random_index = np.random.choice(range(len(X)), self.batch_size)
            random_X_batch = X[random_index]
            random_y_batch = y[random_index]

            dbda, dbdb = self.gradientDescent(random_X_batch, random_y_batch)

            self.slope = self.slope - (self.learning_rate * dbda)
            self.intercept = self.intercept - (self.learning_rate * dbdb)
            

    def predict(self, X):
        print(self.slope)
        return self.intercept + np.dot(X, self.slope)