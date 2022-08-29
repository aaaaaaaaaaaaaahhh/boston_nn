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
  - ### Fully Connected Layer
    - Each Fully Connected Layer(FC Layer for short) consists of a 
    `forward` and `backward` function, the first of which finds the 
    dot product of the input and the weights and adds the bias, similar
    to the equation of a line. $this_{is}$
- ## Result



# Sources

1. https://medium.com/@thevie/predicting-boston-housing-prices-step-by-step-linear-regression-tutorial-from-scratch-in-python-c50a09b70b22
2. https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset/notebook
3. http://alexlenail.me/NN-SVG/index.html

$thing$
