import numpy as np


class loss:
    def __init__(self, y):
        self.y = y

    def cross_entropy(self, y_pred):
        self.y_pred = y_pred
        print(self.y_pred[0])
        print(np.log10(self.y_pred[0]))
        return -np.sum(self.y * np.log10(self.y_pred))

    def cross_entropy_prime(self, y_pred):
        dLdY = y_pred - self.y
        return dLdY

    def mean_squared_error(self, y_pred, n): # y_pred should be an array of all the predicted values in one epoch and y should be their corresponding actual values
        return (np.sum((self.y-y_pred)**2, 0))/n

    def mean_squared_error_prime(self, y_pred, n):
        return (2*np.sum(self.y-y_pred, 0))/n