import numpy as np
import math


class CrossEntropy:

    def cross_entropy(self, y_pred, y):
        self.y = y
        return -np.sum(self.y * np.log10(y_pred))

    def cross_entropy_prime(self, y_pred):
        dLdY = y_pred - self.y
        return dLdY


class Mse:

    def root_mse(self, y_pred, y):
        return math.sqrt(np.mean(np.sum((y - y_pred) ** 2, 0)))

    def mse(self, y_pred, y):  # y_pred should be an array of all the predicted values in one epoch and y should be their corresponding actual values
        return 1 / (2 * len(y)) * np.sum(np.square(y_pred - y))

    def mse_prime(self, y_pred, y):
        return 2 * (y_pred - y) / np.size(y)
