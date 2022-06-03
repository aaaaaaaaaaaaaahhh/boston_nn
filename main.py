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

network = [
    dense.FC_layer(2, 5),
    activation.relu(),
    dense.FC_layer(5, 5),
    activation.relu(),
    dense.FC_layer(5, 1),
    activation.linear()
]
layer_1 = dense.FC_layer(2, 5)
layer_1_activation = activation.relu()
layer_2 = dense.FC_layer(5, 5)
layer_2_activation = activation.relu()
layer_3 = dense.FC_layer(5, 1)
layer_3_activation = activation.linear()
final_loss = loss.loss()

e = 1000
y_preds = []
losses = []
epochs = []
l1_w = layer_3.weights
n = len(y_train)

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output, .03)
    return output


for i in range(e):
    error = 0
    epochs.append(i)
    '''
    for x in range(np.shape(x_train)[0]):
        x_initial = np.reshape(x_train[x, :], (2, 1))
        y = y_train[x]'''

    output = predict(network, x_train.T)

    # error
    error += final_loss.mean_squared_error(output, y_train.T)
    losses.append(error)

    # backward
    grad = final_loss.mean_squared_error_prime(output, y_train.T)
    layer_num = len(network)
    for layer in reversed(network):
        grad = layer.backward(grad, i, layer_num, network, y_train_shaped, .03)
        layer_num = layer_num - 1


        '''
        #print("beginning x", x_initial)
        layer_1_out = layer_1.forward(x_initial)
        #print("layer1 out", layer_1_out)
        layer_1_w = layer_1.weights
        layer_1_activation_out = layer_1_activation.forward(layer_1_out)
        #print("layer1 activation out", layer_1_activation_out)
        layer_2_out = layer_2.forward(layer_1_activation_out)
        #print("layer2 out", layer_2_out)
        layer_2_activation_out = layer_2_activation.forward(layer_2_out)
        #print("layer2 activation out", layer_2_activation_out)
        layer_3_out = layer_3.forward(layer_2_activation_out)
        #print("layer3 out", layer_3_out)
        prediction = layer_3_activation.forward(layer_3_out)
        #print("layer3 activation out", prediction)

        loss = final_loss.root_mean_squared_error(prediction, y, 404)
        losses.append(loss)
        epochs.append(i)

        l_prime = final_loss.mean_squared_error_prime(prediction, y)

        #layer_3_output = l_prime
        #layer_3_input = layer_3.x

        layer_3_back = layer_3.backward(l_prime, .03)
        #print("layer3 weight changes", layer_3.dLdW)
        #print("layer3 x", layer_3.x)
        layer_2_activation_back = layer_2_activation.backward(layer_3_back)
        #print("layer2 back", layer_2_activation_back)
        layer_2_back = layer_2.backward(layer_2_activation_back, .03)
        #print("layer2 weight changes", layer_2.dLdW)
        layer_1_activation_back = layer_1_activation.backward(layer_2_back)
        layer_1_back = layer_1.backward(layer_1_activation_back, .03)
        #print("layer1 weight changes", layer_1.dLdW)
        '''

for x in x_train:
    x = np.reshape(x, (2, 1))
    prediction = predict(network, x)
    y_preds.append(prediction)


print(np.shape(y_preds))

plt.figure()
plt.scatter(lstat, y_preds, c="Red")
plt.scatter(lstat, y_train)

print(epochs)
plt.figure()
plt.plot(epochs, losses)
print(losses)


plt.figure()
plt.scatter(ptratio, y_train)
plt.scatter(ptratio, y_preds, c='Red')

plt.show()




