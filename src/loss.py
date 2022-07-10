import numpy as np


class Loss:
    """This class defines the loss of a neural network."""

    def __init__(self):
        pass

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        self.prediction = prediction
        self.target = target

        return self._output()

    def backward(self) -> np.ndarray:
        self.input_grad = self._input_grad()

        return self.input_grad

    def _output(self) -> float:
        raise NotImplementedError()

    def _input_grad(self) -> np.ndarray:
        raise NotImplementedError()


class MeanSquaredError(Loss):
    """This class defines the mean squared loss of a neural network."""

    def __init__(self) -> None:
        super().__init__()

    def _output(self) -> float:
        return (
            np.sum(np.power(self.prediction - self.target, 2))
            / self.prediction.shape[0]
        )

    def _input_grad(self) -> np.ndarray:
        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]


if __name__ == "__main__":
    pass
