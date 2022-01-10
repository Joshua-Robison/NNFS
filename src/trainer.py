import numpy as np
from copy import deepcopy
from typing import Tuple
from helper import shuffle
from optimizer import Optimizer
from neuralnetwork import NeuralNetwork
from linear import Linear
from block import Dense
from loss import MeanSquaredError


class Trainer:
    """This class trains a neural network."""
    def __init__(self, net: NeuralNetwork, optim: Optimizer):
        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        setattr(self.optim, 'net', self.net)

    def generate_batches(self, X: np.ndarray, y: np.ndarray, size: int=32) -> Tuple[np.ndarray]:
        N = X.shape[0]
        for ii in range(0, N, size):
            X_batch, y_batch = X[ii:ii+size], y[ii:ii+size]
            yield X_batch, y_batch

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
            epochs: int=100, eval_every: int=10, batch_size: int=32, seed: int=1, restart: bool=True):
        np.random.seed(seed)
        if restart:
            for layer in self.net.layers:
                layer.first = True

            self.best_loss = 1e9

        for e in range(epochs):
            if (e + 1) % eval_every == 0:
                # early stopping
                last_model = deepcopy(self.net)

            X_train, y_train = shuffle(X_train, y_train)
            batch_generator = self.generate_batches(X_train, y_train, batch_size)
            for ii, (X_batch, y_batch) in enumerate(batch_generator):
                self.net.train_batch(X_batch, y_batch)
                self.optim.step()

            if (e + 1) % eval_every == 0:
                test_preds = self.net.forward(X_test)
                loss = self.net.loss.forward(test_preds, y_test)
                if loss < self.best_loss:
                    print(f"Validation loss after {e+1} epochs is {loss:.3f}")
                    self.best_loss = loss
                else:
                    print(f"Loss increased after epoch {e+1}, final loss was {self.best_loss:.3f}, using the model from epoch {e+1-eval_every}")
                    self.net = last_model
                    # ensure self.optim is still updating self.net
                    setattr(self.optim, 'net', self.net)
                    break


if __name__ == '__main__':
    pass
