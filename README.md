# Project Description

This repository is a school project in which I made a Neural 
Network model from scratch. I used OOP to make it usable to 
other network dimensions, as well as provided some activation
and loss functions. It's current purpose is to learn the correct
price of a house from the popular boston housing dataset which you
can go learn more about 
**[here](https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset/notebook)**.

# Project Overview

- ## Goal
  - The goal of this project was to use a Neural Network to predict house prices
  of a dataset of house in Boston using the 13 variables about each house in
  the dataset
  - To test if my model was learning correctly, I chose 2 variables that
  had a correlation with the target variable `mdev`, which is the median
  value of the homes in thousands, or the price. These 2 variables are 
  `lstat` and `ptratio`. I set each one on the x-axis of 2 different graphs 
  and then made each y-axis `mdev`. Once ran my model I did the same
  except with the predictions the model made on the y-axis instead of 
  `mdev`. I also stored the losses of the network as it trained and plotted
  those against the epochs to make sure they were going down.
- ## Model
  - My model consists of only 2 types of layers, Fully Connected layers
  and Relu activation layers. In addition to these layers I used the
  Mean Squared Error function for my loss.
  - ![neural net](https://github.com/aaaaaaaaaaaaaahhh/boston_nn/blob/master/neural%20net.png) This 
  first layer has 13 nodes for the 13 parameters for each datapoint. It gets
  multiplied with the weights and added to the bias, and then goes to 
  the second layer which is Relu. The second and third rows of nodes
  represent the Relu activation function, with the nodes not connected
  to the previous layer being the biases added in the Fully Connected
  Layers. The final node is the output, which will be compared to the
  actual value in the training set and give an error for backpropagation.
  - ### Fully Connected Layer
    - Each Fully Connected Layer(FC Layer for short) consists of a 
    `forward` and `backward` function, the first of which finds the 
    dot product of the input and the weights and adds the bias $W^{T}X+b$, similar
    to the equation of a line $m*x+b$. The backward function finds the 
    partial derivative of the loss with respect to the weights and 
    biases(`dLdW` and `dLdW0`), as well as passing the partial derivative of the loss 
    with respect to the input(`dLdA`) back to the layer behind.
  - ### Relu Activation Layer
    - The Activation Layer takes a linear vector or matrix and 
    applies a non-linear function to it, in this case, the Rectified
    Linear function. ![Relu](https://miro.medium.com/max/1400/1*DfMRHwxY1gyyDmrIAd-gjQ.png)
    This function maps negative inputs to 0 and lets positive inputs
    remain the same.
    
- ## Result
  - After running the model through forward and backward propagation
  50000 times, it has successfully learned the data set. The loss on the
  training data(data only used for the training of the model) is 6.53
  and the loss on the test data is 7.87. These numbers are good and all
  but to show you that it really has learned the data set, I took the
  models predictions and compared them to 2 of the parameters in the dataset
  - Below is a graph of the predictions vs the actual data where
  the x-axis is the lstat parameter and the y-axis is the actual house
  prices(blue) or the models predictions(red).
  ![lstat results](https://github.com/aaaaaaaaaaaaaahhh/boston_nn/blob/master/lstat%20results.png)
  - Another indicator of a properly learning neural net is a gradual
  downward curve of the loss over the epochs
  ![loss vs epochs](https://github.com/aaaaaaaaaaaaaahhh/boston_nn/blob/master/losses%20vs%20epochs.png)
  


# Sources

1. https://medium.com/@thevie/predicting-boston-housing-prices-step-by-step-linear-regression-tutorial-from-scratch-in-python-c50a09b70b22
2. https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset/notebook
3. http://alexlenail.me/NN-SVG/index.html


