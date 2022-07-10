import numpy as np
from operation import ParamOperation


class WeightMultiply(ParamOperation):
    """This class defines a weight multiplication operation for a neural network."""

    def __init__(self, W: np.ndarray):
        super().__init__(W)

    def _output(self) -> np.ndarray:
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return np.dot(np.transpose(self.input_, (1, 0)), output_grad)


if __name__ == "__main__":
    pass
