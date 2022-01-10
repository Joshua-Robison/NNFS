import numpy as np
from typing import List
from block import Layer, Dense
from linear import Linear
from loss import Loss, MeanSquaredError


class NeuralNetwork:
    """This class defines a neural network."""
    def __init__(self, layers: List[Layer], loss: Loss, seed: int=1):
        self.layers = layers
        self.loss = loss
        self.seed = seed
        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)

    def forward(self, x_batch: np.ndarray) -> np.ndarray:
        out = x_batch
        for layer in self.layers:
            out = layer.forward(out)

        return out

    def backward(self, loss_grad: np.ndarray):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        predictions = self.forward(x_batch)
        loss = self.loss.forward(predictions, y_batch)
        self.backward(self.loss.backward())

        return loss

    def params(self):
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        for layer in self.layers:
            yield from layer.param_grads


if __name__ == '__main__':
    pass
