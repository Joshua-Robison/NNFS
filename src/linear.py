import numpy as np
from operation import Operation


class Linear(Operation):
    """This class is the identity activation function."""

    def __init__(self):
        super().__init__()

    def _output(self) -> np.ndarray:
        return self.input_

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad


if __name__ == "__main__":
    pass
