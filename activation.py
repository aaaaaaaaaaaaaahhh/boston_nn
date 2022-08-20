import numpy as np


class relu:
    def forward(self, input, epoch, layer, lr):
        self.input = input
        self.A = np.maximum(0, input)
        #print('a_relu', epoch, layer, self.A)
        return self.A

    def backward(self, dLdA, epoch, layer_num, network, y_train_shaped, lr):
        return np.multiply(dLdA, np.where(self.A>=0, 1, 0))


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
    def forward(self, input, epoch, layer, lr):
        return input

    def backward(self, input, epoch, layer_num, network, y_train_shaped, lr):
        #return input
        return [[1]]


class tanh:
    def forward(self, input, lr):
        return np.tanh(input)

    def backward(self, input, lr):
        return 1 - np.tanh(input) ** 2