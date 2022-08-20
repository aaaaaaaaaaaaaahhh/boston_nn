import numpy as np
import activation

# if you dont get any of the derivation, dont worry i dont really get them either.
# go to https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week7/week7_homework/?activate_block_id=block-v1%3AMITx%2B6.036%2B1T2019%2Btype%40sequential%2Bblock%40week7_homework


x = np.array([[1, 2, 3, 2.5],
              [2.0, 5.0, -1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]])


class FC_layer:
    def __init__(self, input_size, nodes):
        # d is number of dimensions to the data and n is the number of data points
        # being fed through(batch size)
        self.m = nodes  # number of nodes
        self.weights = np.random.randn(nodes, input_size)*.01# d by m
        self.biases = np.random.randn(nodes, 1)*.01

    def forward(self, input, epoch, layer, lr):
        self.x = input
        self.z = np.dot(self.weights, self.x) + self.biases # the reshape is to make the array compatible with the biases. only

        '''
        self.x = input
        self.z = np.dot(self.weights, self.x) + self.biases
        ''' # this is code from before it was working
        #print('x', epoch, layer, input)
        #print("shape of x, in linear", np.shape(input))
        #print('z', epoch, layer, self.z)
        #print("shape of z, in linear", np.shape(self.z))
        #print('w', epoch, layer, self.weights)
        #print("shape of w, in linear", np.shape(self.weights))
        #print('b', epoch, layer, self.biases)
        #print("shape of b, in linear", np.shape(self.biases))
        #print(" ")
        return self.z

    def backward(self, output_error, epoch, layer, network, y_train, learning_rate=.03):  # dLdZ is of size m(l+1) by n(l+1)
        #print("shape of output error, in linear", np.shape(output_error))
        #print("shape of biases, in linear", np.shape(self.biases))
        #print("shape of weights, in linear", np.shape(self.weights))
        #print("shape of x, in linear", np.shape(self.x))
        '''
        layers = len(network)
        m = len(y_train)
        #print("dLdA test", 1 / m * (self.z - y_train))
        if layer == layers:
            print("output error shape", epoch, layer, np.shape(output_error))
            self.dLdA = 1/m * (self.a - y_train.T)
            self.dLdA = np.dot(self.weights.T, output_error)
            output_error = self.dLdA
            #print("dL d A", epoch, layer, self.dLdA)
            print("shape of dL d A", epoch, layer, np.shape(self.dLdA))
            #print('output error', epoch, layer, output_error)
        else:
            print(np.shape(output_error))
            #print(layers, layer)
            self.dLdA = np.dot(self.weights.T, output_error)
            print("shape of dLd A", epoch, layer, np.shape(self.dLdA))
            output_error = np.multiply(self.dLdA, np.where(self.a>=0, 1, 0))
            #print('output error', epoch, layer, output_error)
            #print("dLd A", epoch, layer, self.dLdA)

        if layer == 1:
            self.dLdW = 1 / m * np.dot(output_error, x_train)
        else:
            self.dLdW = 1 / m * np.dot(output_error, network[layer - 1].a.T)

        self.dLdW0 = 1/m * np.sum(output_error, axis=1, keepdims=True)# m by n (same size as dLdZ)
        '''
        self.dLdA = np.dot(self.weights.T, output_error)
        self.dLdW = np.dot(output_error, self.x.T)
        self.dLdW0 = np.sum(output_error, axis=1, keepdims=True)
         # code from before it was working, still had the weights and biases reassignment at the end
        #print("dLdW", epoch, layer, self.dLdW)
        #print("dLdW0", epoch, layer, self.dLdW0)
        #print("dLdA", epoch, layer, self.dLdA)
        #self.dLdW0 = output_error
        #print("dldw", self.dLdW)
        #print("shape of dLdA, in linear", np.shape(self.dLdA))
        #print("shape of dLdW0, in linear", np.shape(self.dLdW0))
        #print("shape of dLdW, in linear", np.shape(self.dLdW))
        #print("shape of weights, in linear", np.shape(self.weights))
        #print("shape of biases, in linear", np.shape(self.biases))
        threshold = 1
        if np.linalg.norm(self.dLdW):
            self.dLdW = threshold * (self.dLdW/np.linalg.norm(self.dLdW))
        if np.linalg.norm(self.dLdW0):
            self.dLdW0 = threshold * (self.dLdW0 / np.linalg.norm(self.dLdW0))
        self.weights = self.weights - learning_rate * self.dLdW
        self.biases = self.biases - learning_rate * self.dLdW0
        #print("shape of x, in linear", np.shape(self.x))
        #print(" ")
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
