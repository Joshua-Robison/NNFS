import warnings
import numpy as np
from operation import Operation


class Sigmoid(Operation):
    """This class is the sigmoid activation function."""

    def __init__(self) -> None:
        super().__init__()

    def _output(self) -> np.ndarray:
        # suppress overflow warning due to lack of precision:
        #   Windows does not support float128
        #   warnings.warn('RuntimeWarning')
        warnings.simplefilter("ignore")

        return 1.0 / (1.0 + np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        sigmoid_backward = self.output * (1.0 - self.output)

        return sigmoid_backward * output_grad


if __name__ == "__main__":
    pass
