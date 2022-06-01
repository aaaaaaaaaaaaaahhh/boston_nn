import numpy as np
import math

class loss:

    def cross_entropy(self, y_pred):
        self.y_pred = y_pred
        return -np.sum(self.y * np.log10(self.y_pred))

    def cross_entropy_prime(self, y_pred):
        dLdY = y_pred - self.y
        return dLdY

    def root_mean_squared_error(self, y_pred, y, n):
        #return 1 / (2 * len(y)) * np.sum(np.square(y_pred - y))
        return math.sqrt(np.mean(np.sum((y-y_pred)**2, 0)))

    def mean_squared_error(self, y_pred, y): # y_pred should be an array of all the predicted values in one epoch and y should be their corresponding actual values
        return 1 / (2 * len(y)) * np.sum(np.square(y_pred - y))
        #return np.mean(np.power(y - y_pred, 2))

    def mean_squared_error_prime(self, y_pred, y):
        return 2 * (y_pred - y) / np.size(y)