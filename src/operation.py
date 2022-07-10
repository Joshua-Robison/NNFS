import numpy as np


class Operation:
    """This is the base class for an operation in a neural network."""

    def __init__(self):
        pass

    def forward(self, input_: np.ndarray) -> np.ndarray:
        self.input_ = input_
        self.output = self._output()

        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        self.input_grad = self._input_grad(output_grad)

        return self.input_grad

    def _output(self) -> np.ndarray:
        raise NotImplementedError()

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class ParamOperation(Operation):
    """This class is an operation with parameters."""

    def __init__(self, param: np.ndarray):
        super().__init__()
        self.param = param

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        return self.input_grad

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


if __name__ == "__main__":
    pass
