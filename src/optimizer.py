# -*- coding: utf-8 -*-
"""
Deep Learning from Scratch

Class Definition:
----------------
Optimizer
SGD
"""
class Optimizer:
    """
    This is the base class for a neural network optimizer.
    """
    def __init__(self, lr: float=0.01):
        self.lr = lr

    def step(self):
        pass


class SGD(Optimizer):
    """
    This is the stochastic gradient descent optimizer.
    """
    def __init__(self, lr: float=0.01):
        super().__init__(lr)

    def step(self):
        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            param -= self.lr * param_grad
