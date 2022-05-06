import numpy as np
import dense
import activation
import loss

from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt


#----------------------------------------------Preprocessing data-------------------------------------------------------


(x_train, y_train),(x_test, y_test) = boston_housing.load_data()
ptratio = x_train[:, 10]
lstat = x_train[:, 12]
x_train = np.c_[ptratio, lstat]#getting only the features that we need
#x_train = x_train.T

x_test = np.c_[x_test[:, 10], x_test[:, 12]]
#x_test = x_test.T
y_train_shaped = np.reshape(y_train, (404, 1))
#-----------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------Training network------------------------------------------------------


layer_1 = dense.FC_layer(2, 15)
layer_1_activation = activation.relu()
layer_2 = dense.FC_layer(15, 13)
layer_2_activation = activation.relu()
layer_3 = dense.FC_layer(13, 1)
layer_3_activation = activation.linear()
final_loss = loss.loss(y_train_shaped)

e = 1000
y_preds = []
losses = []
epochs = []
l1_w = layer_3.weights
for i in range(e):

    layer_1_out = layer_1.forward(x_train)
    layer_1_w = layer_1.weights
    layer_1_activation_out = layer_1_activation.forward(layer_1_out)
    layer_2_out = layer_2.forward(layer_1_activation_out)
    layer_2_activation_out = layer_2_activation.forward(layer_2_out)
    layer_3_out = layer_3.forward(layer_2_activation_out)
    prediction = layer_3_activation.forward(layer_3_out)

    y_preds.append(prediction)
    loss = final_loss.root_mean_squared_error(prediction, 404)
    losses.append(loss)
    epochs.append(i)

    l_prime = final_loss.mean_squared_error_prime(prediction, 404)

    layer_3_output = l_prime
    layer_3_input = layer_3.x
    print(np.shape(layer_3_input), np.shape(layer_3_output))

    layer_3_back = layer_3.backward(l_prime, .001)
    layer_2_activation_back = layer_2_activation.backward(layer_3_back)
    layer_2_activation_back = np.reshape(layer_2_activation_back, (1, 13))
    layer_2_back = layer_2.backward(layer_2_activation_back, .001)
    layer_1_activation_back = layer_1_activation.backward(layer_2_back)
    layer_1_back = layer_1.backward(layer_1_activation_back, .001)
    '''print("")
    print("dLdZ ", i, " ", layer_3.dLdZ)
    print("dLdA ", i, " ", layer_3.dLdA)
    print("z ", i, " ", layer_3.z)
    print("x ", i, " ", layer_3.x)
    print("weights ", i, " ", layer_3.weights)
    print("biases ", i, " ", layer_3.biases)
    print("dLdW ", i, " ", layer_3.dLdW)'''

print(np.shape(layer_3_input), np.shape(layer_3_output))

plt.figure()
plt.scatter(lstat, ye_preds)

plt.figure()
plt.plot(epochs, losses)
print(losses)

plt.figure()
plt.scatter(lstat, y_train)


plt.figure()
plt.scatter(ptratio, y_train)

plt.figure()
plt.scatter(ptratio, ye_preds, c='Red')

plt.show()




