import numpy as np
from numpy.typing import NDArray


class ActivationFunction:
    def __init__(self):
        self.input = np.array([])
        self.output = np.array([])
        self.d_inputs = np.array([])

    def __str__(self):
        return self.__class__.__name__


class ReLU(ActivationFunction):
    def forward(self, inputs: NDArray) -> NDArray:
        self.input = inputs
        self.output = np.maximum(0, inputs)

        return self.output

    def backward(self, d_values: NDArray) -> None:
        self.d_inputs = d_values.copy()
        self.d_inputs[self.input <= 0] = 0


class SoftMax(ActivationFunction):
    # Only for output layer
    def forward(self, inputs: NDArray) -> NDArray:
        # To avoid exploding gradients max value in each row is substracted
        # Source: https://jamesmccaffrey.wordpress.com/2016/03/04/the-max-trick-when-computing-softmax/
        exponential_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exponential_values / np.sum(
            exponential_values, axis=1, keepdims=True
        )

        return self.output

    def backward(self, d_values: NDArray) -> None:
        d_inputs = np.empty_like(d_values)

        for idx, (output, d_value) in enumerate(zip(self.output, d_values)):
            output = output.reshape(-1, 1)
            # each entry represents the partial derivative of the softmax
            # output with respect to the input values
            jacobian_matrix = np.diagflat(output) - np.dot(output, output.T)
            d_inputs[idx] = np.dot(jacobian_matrix, d_value)

        self.d_inputs = d_inputs


class Sigmoid(ActivationFunction):
    # For regression tasks
    def forward(self, inputs: NDArray) -> NDArray:
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, d_values: NDArray) -> NDArray:
        # Compute the derivative of the sigmoid function
        sigmoid_derivative = self.output * (1 - self.output)
        # Compute the gradient of the loss with respect to the input
        d_inputs = d_values * sigmoid_derivative
        
        self.d_inputs = d_inputs


class Linear(ActivationFunction):
    def forward(self, values: NDArray) -> NDArray:
        self.output = values
        return self.output

    def backward(self, d_values: NDArray) -> NDArray:
        self.d_inputs = d_values.copy()
        self.d_inputs = NDArray
