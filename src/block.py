import numpy as np
from typing import List
from operation import Operation, ParamOperation
from weights import WeightMultiply
from bias import BiasAdd
from sigmoid import Sigmoid


class Layer:
    """This class is a layer of neurons in a neural network."""
    def __init__(self, neurons: int):
        self.neurons = neurons
        self.first = True
        self.params: List[np.ndarray] = []
        self.param_grads: List[np.ndarray] = []
        self.operations: List[Operation] = []

    def _setup_layer(self, num_in: int):
        raise NotImplementedError()

    def forward(self, input_: np.ndarray) -> np.ndarray:
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_
        for operation in self.operations:
            input_ = operation.forward(input_)

        self.output = input_

        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        input_grad = output_grad
        self._param_grads()

        return input_grad

    def _param_grads(self):
        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self):
        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)


class Dense(Layer):
    """This class is a fully connected layer of neurons."""
    def __init__(self, neurons: int, activation: Operation=Sigmoid()):
        super().__init__(neurons)
        self.activation = activation

    def _setup_layer(self, input_: np.ndarray):
        if self.seed:
            np.random.seed(self.seed)

        self.params = []
        self.params.append(np.random.randn(input_.shape[1], self.neurons))
        self.params.append(np.random.randn(1, self.neurons))
        self.operations = [WeightMultiply(self.params[0]), BiasAdd(self.params[1]), self.activation]


if __name__ == '__main__':
    pass
