import numpy as np
import dense
import activation
import loss

from tensorflow.keras.datasets import boston_housing



#----------------------------------------------Preprocessing data-------------------------------------------------------


(x_train, y_train),(x_test, y_test) = boston_housing.load_data()
x_train = np.c_[x_train[:, 10], x_train[:, 12]] #getting only the features that we need
x_test = np.c_[x_test[:, 10], x_test[:, 12]]


#-----------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------Training network------------------------------------------------------


layer_1 = dense.FC_layer((2, 1), 15)
layer_1_activation = activation.relu()
layer_2 = dense.FC_layer((15, 1), 13)
layer_2_activation = activation.relu()
layer_3 = dense.FC_layer((13, 1), 1)
layer_3_activation = activation.linear()

e = 10
y_preds = []
losses = []

for i in range(e):
    for l in range(np.shape(x_train)[0]):
        x_l = np.reshape(x_train[l, :], (2, 1))
        layer_1_out = layer_1.forward(x_l)
        layer_1_activation_out = layer_1_activation.forward(layer_1_out)
        layer_2_out = layer_2.forward(layer_1_activation_out)
        layer_2_activation_out = layer_2_activation.forward(layer_2_out)
        layer_3_out = layer_3.forward(layer_2_activation_out)
        prediction = layer_3_activation.forward(layer_3_out)
        prediction = prediction.item()

        y_preds.append(prediction)

    loss = loss.

        # do backwards pass. how with no loss?



print(y_preds)






