# -*- coding: utf-8 -*-
"""
Deep Learning from Scratch

This is an example demonstrating how to use the NNFS package.
"""
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from neuralnetwork import NeuralNetwork
from block import Dense
from linear import Linear
from sigmoid import Sigmoid
from loss import MeanSquaredError
from optimizer import SGD
from trainer import Trainer
from helper import evaluate


# random seed
seed = 42

# load sample dataset
boston = load_boston()

# extract data
data = boston.data     # features -> X
target = boston.target # targets  -> y

# load a preprocessing class
s = StandardScaler()

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

# reshape the 1d arrays: 1d -> 2d
y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

# scale the data: zero mean and unit variance
# z = (x - mean(x)) / std(x)
s_train = s.fit(X_train)
X_train_scaled = s_train.transform(X_train)
X_test_scaled = s_train.transform(X_test)

# initialize three models to evaluate:
# linear regression model
lr = NeuralNetwork(layers=[Dense(neurons=1, activation=Linear())],
                   loss=MeanSquaredError(), seed=seed)

# a simple neural network
nn = NeuralNetwork(layers=[Dense(neurons=13, activation=Sigmoid()),
                           Dense(neurons=1, activation=Linear())],
                   loss=MeanSquaredError(), seed=seed)

# a deep neural network
dl = NeuralNetwork(layers=[Dense(neurons=13, activation=Sigmoid()),
                           Dense(neurons=13, activation=Sigmoid()),
                           Dense(neurons=1, activation=Linear())],
                   loss=MeanSquaredError(), seed=seed)

# initialize an optimizer:
# Stochastic Gradient Descent
optimizer = SGD(lr=0.01)

# initialize a trainer for each model:
# fit and evaluate each model
trainer = Trainer(lr, optimizer)
trainer.fit(X_train_scaled, y_train, X_test_scaled, y_test, epochs=50, eval_every=10, seed=seed)
evaluate(lr, X_test, y_test)

trainer = Trainer(nn, optimizer)
trainer.fit(X_train_scaled, y_train, X_test_scaled, y_test, epochs=50, eval_every=10, seed=seed)
evaluate(nn, X_test, y_test)

trainer = Trainer(dl, optimizer)
trainer.fit(X_train_scaled, y_train, X_test_scaled, y_test, epochs=50, eval_every=10, seed=seed)
evaluate(dl, X_test, y_test)
