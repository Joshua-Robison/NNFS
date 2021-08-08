# -*- coding: utf-8 -*-
"""
Deep Learning from Scratch

Helper Functions:
----------------
shuffle
mae
rmse
evaluate
"""
import numpy as np


def shuffle(X, y):
    """
    This function shuffles the input features (X) and labels (y).
    """
    perm = np.random.permutation(X.shape[0])

    return X[perm], y[perm]


def mae(y_true, y_pred):
    """
    This function computes the mean absolute error for a neural network.
    """
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """
    This function computes the root mean squared error for a neural network.
    """
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))


def evaluate(model, X_test, y_test):
    """
    This function computes the mae and rmse for a neural network.
    """
    preds = model.forward(X_test)
    preds = preds.reshape(-1, 1)

    print('Mean absolute error: {:.2f}'.format(mae(preds, y_test)))
    print('Root mean squared error: {:.2f}\n'.format(rmse(preds, y_test)))
