import numpy as np


class relu:
    def forward(self, input, lr):
        self.input = input
        self.A = np.maximum(0, input)
        return self.A

    def backward(self, dLdA, lr):
        return dLdA * (self.A != 0)


class softmax:
    def forward(self, input, lr):
        out = np.zeros((np.shape(input)))
        self.n, self.k = np.shape(input)
        '''for j in range(self.k):
            out[:, j] = np.exp(input[:, j]) / np.sum(np.exp(input), 0)
'''
        return np.exp(input) / np.sum(np.exp(input), axis=0)
    def backward(self, dLdZ, lr):
        return dLdZ

class linear:
    def forward(self, input, lr):
        return input

    def backward(self, input, lr):
        return 1
