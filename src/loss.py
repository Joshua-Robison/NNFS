# -*- coding: utf-8 -*-
"""
Deep Learning from Scratch

Class Definition:
----------------
Loss
MeanSquaredError
"""
import numpy as np


class Loss:
    """
    This class defines the loss of a neural network.
    """
    def __init__(self):
        pass

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """
        This function computes the actual loss value.

        Parameters:
        ----------
        prediction : np.ndarray
        target : np.ndarray

        Returns:
        -------
        float
        """
        self.prediction = prediction
        self.target = target

        return self._output()

    def backward(self) -> np.ndarray:
        """
        This function computes the gradient of the
        loss w.r.t. the input to the loss function.

        Parameters:
        ----------
        None

        Returns:
        -------
        self.input_grad : np.ndarray
        """
        self.input_grad = self._input_grad()

        return self.input_grad

    def _output(self) -> float:
        raise NotImplementedError()

    def _input_grad(self) -> np.ndarray:
        raise NotImplementedError()


class MeanSquaredError(Loss):
    """
    This class defines the mean squared loss of a neural network.
    """
    def __init__(self) -> None:
        super().__init__()

    def _output(self) -> float:
        """
        This function computes the per-observation squared error loss.

        Parameters:
        ----------
        None

        Returns:
        -------
        float
        """
        return np.sum(np.power(self.prediction - self.target, 2)) / self.prediction.shape[0]

    def _input_grad(self) -> np.ndarray:
        """
        This function computes the loss gradient
        w.r.t. the input for MSE loss.

        Parameters:
        ----------
        None

        Returns:
        -------
        np.ndarray
        """
        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]
