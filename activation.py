import numpy as np


class Relu:

    def forward(self, input, epoch, layer, lr):
        self.A = np.maximum(0, input)
        #print('a_relu', epoch, layer, self.A)
        return self.A

    def backward(self, dLdA, epoch, layer, lr):
        return np.multiply(dLdA, np.where(self.A>=0, 1, 0))


class Softmax:

    def forward(self, input, epoch, layer, lr):
        out = np.zeros((np.shape(input)))
        self.n, self.k = np.shape(input)

        return np.exp(input) / np.sum(np.exp(input), axis=0)
    def backward(self, dLdZ, epoch, layer, lr):
        return dLdZ

class Linear:

    def forward(self, input, epoch, layer, lr):
        return input

    def backward(self, input, epoch, layer, lr):
        return np.ones(np.shape(input), dtype=int)


class Tanh:

    def forward(self, input, epoch, layer, lr):
        return np.tanh(input)

    def backward(self, input, epoch, layer, lr):
        return 1 - np.tanh(input) ** 2