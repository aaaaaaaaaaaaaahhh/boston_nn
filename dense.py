import numpy as np

# if you dont get any of the derivation, dont worry i dont really get them either.
# go to https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week7/week7_homework/?activate_block_id=block-v1%3AMITx%2B6.036%2B1T2019%2Btype%40sequential%2Bblock%40week7_homework

np.random.seed(6)

x = np.array([[1, 2, 3, 2.5],
              [2.0, 5.0, -1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]])


class FC_layer:
    def __init__(self, input_size, nodes):
        # d is number of dimensions to the data and n is the number of data points
        # being fed through(batch size)
        self.m = nodes  # number of nodes
        self.weights = np.random.randn(nodes, input_size) - .5# d by m
        self.biases = np.random.randn(nodes, 1) - .5

    def forward(self, input, lr):
        self.x = input
        self.z = np.dot(self.weights, self.x) + self.biases # the reshape is to make the array compatible with the biases. only
        return self.z

    def backward(self, output_error, epoch, learning_rate=.03):  # dLdZ is of size m(l+1) by n(l+1)
        #print("shape of output error, in linear", np.shape(output_error))
        #print("shape of biases, in linear", np.shape(self.biases))
        #print("shape of weights, in linear", np.shape(self.weights))
        #print("shape of x, in linear", np.shape(self.x))
        self.dLdA = np.dot(self.weights.T, output_error)
        self.dLdW = np.dot(output_error, self.x.T)
        self.dLdW0 = np.sum(output_error, axis=1, keepdims=True)# m by n (same size as dLdZ)
        print("dLdW", epoch, self.dLdW)
        print("dLdA", epoch, self.dLdA)
        print("dLdW0", epoch, self.dLdW0)
        #self.dLdW0 = output_error
        #print("dldw", self.dLdW)
        #print("shape of dLdA, in linear", np.shape(self.dLdA))
        #print("shape of dLdW0, in linear", np.shape(self.dLdW0))
        #print("shape of dLdW, in linear", np.shape(self.dLdW))
        self.weights = self.weights - learning_rate * self.dLdW
        self.biases = self.biases - learning_rate * self.dLdW0
        return self.dLdA  # d by n

'''
layer_1 = FC_layer(x.T, 5)

Z_1 = layer_1.linear()
print(Z_1)

A_1 = layer_1.reLU(Z_1)
print(A_1)

layer_2 = FC_layer(A_1, 4)

Z_2 = layer_2.linear()
print(Z_2)

A_2 = layer_2.reLU(Z_2)
print(A_2)



Z_1 = layer_1.linear(forward=False)
print(Z_1)

A_1 = layer_1.reLU(forward=False)
print(A_1)


Z_2 = layer_2.linear(forward=False)
print(Z_2)

A_2 = layer_2.reLU(forward=Z_2)
print(A_2)


'''
