import numpy as np
from operation import ParamOperation


class BiasAdd(ParamOperation):
    """This class is the bias addition operation for a neural network."""

    def __init__(self, B: np.ndarray):
        assert B.shape[0] == 1
        super().__init__(B)

    def _output(self) -> np.ndarray:
        return self.input_ + self.param

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])


if __name__ == "__main__":
    pass
