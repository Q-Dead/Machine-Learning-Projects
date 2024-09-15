import numpy as np


class LogisticRegression:

    def __init__(self, learning_rate=0.001, n_iter=1000):
        self.linear_rate = learning_rate
        self.n_iter = n_iter
        self.weight = None
        self.intercept = None

    def predict(self, X):
        threshold = .5
        linear_hypothesis = self.linear_line(X)
        sigmoid_f = self.sigmoid_function(linear_hypothesis)
        return  [1 if i > threshold else 0 for i in sigmoid_f], sigmoid_f
    
    # TODO seperate the predicted probability and predicted Class
    def predict_prob(self):
        pass

    def fit(self, X, y):
        i, n = X.shape
        self.weight = np.zeros(n)
        self.intercept = 0

        for _ in range(self.n_iter): # Gradient Ascent

            linear_hypothesis = self.linear_line(X)
            sigmoid_f = self.sigmoid_function(linear_hypothesis)

            abda = (1/i) * np.dot(X.T, (sigmoid_f - y))
            abdb = (1/i) * sum( sigmoid_f - y)

            self.weight = self.weight - self.linear_rate * abda
            self.intercept = self.intercept - self.linear_rate * abdb

    def sigmoid_function(self, z): # Sigmoid Function
        return 1/( 1 + np.exp(-z))

    def linear_line(self, X): # Linear regression Hypothesis
        return self.intercept + np.dot(X, self.weight) 
