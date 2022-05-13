import numpy as np


class relu:
    def forward(self, input):
        self.input = input
        return np.maximum(0.0001, input)

    def backward(self, dLdA):
        return np.where(self.input>= 0, 1, 0) * dLdA


class softmax:
    def forward(self, input):
        out = np.zeros((np.shape(input)))
        self.n, self.k = np.shape(input)
        for j in range(self.k):
            out[:, j] = np.exp(input[:, j]) / np.sum(np.exp(input), 0)

    def backward(self, dLdZ):
        return dLdZ

class linear:
    def forward(self, input):
        return input

    def backward(self):
        return 1
