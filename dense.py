import numpy as np




class FCLayer:
    def __init__(self, input_size, nodes):
        # d is number of dimensions to the data and n is the number of data points
        # being fed through(batch size)
        self.m = nodes  # number of nodes
        self.weights = np.random.randn(nodes, input_size)*.01# d by m
        self.biases = np.random.randn(nodes, 1)*.01

    def forward(self, input, epoch, layer, lr): # epoch and layer are for troubleshooting
        self.x = input
        self.z = np.dot(self.weights, self.x) + self.biases # the reshape is to make the array compatible with the biases. only

 # this is code from before it was working
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

    def backward(self, output_error, epoch, layer, lr=.03):  # dLdZ is of size m(l+1) by n(l+1)
        #print("shape of output error, in linear", np.shape(output_error))
        #print("shape of biases, in linear", np.shape(self.biases))
        #print("shape of weights, in linear", np.shape(self.weights))
        #print("shape of x, in linear", np.shape(self.x))

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
        self.weights = self.weights - lr * self.dLdW
        self.biases = self.biases - lr * self.dLdW0
        #print("shape of x, in linear", np.shape(self.x))
        #print(" ")
        return self.dLdA  # d by n


