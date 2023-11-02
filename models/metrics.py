import numpy as np
from numpy.typing import NDArray


class Loss:
    def __init__(self):
        self.d_inputs = np.array([])

    def forward(self, values: NDArray, expected: NDArray) -> NDArray:
        raise NotImplemented

    def backward(self, d_values: NDArray, expected: NDArray):
        raise NotImplemented


class CategoricalCrossEntropyLoss(Loss):
    # From Neural Networks from Scratch in Python
    # For classification
    def calculate(self, values: NDArray, expected: NDArray) -> float:
        sample = self.forward(values, expected)
        data_loss = np.mean(sample)

        return data_loss

    def forward(self, values: NDArray, expected: NDArray) -> NDArray:
        clipped_values = np.clip(values, 1e-7, 1 - 1e-7)

        confidences = self._calc_correct_values(clipped_values, expected)
        log_of_confidences = np.log(confidences)

        return log_of_confidences * -1

    def backward(self, d_values: NDArray, expected: NDArray):
        samples = len(d_values)
        labels = len(d_values[0])
        expected_values = expected

        if len(expected_values.shape) == 1:
            expected_values = np.eye(labels)[expected_values]

        # Input values gradient
        d_inputs = -expected_values / d_values

        # Normalize the gradient
        self.d_inputs = d_inputs / samples

    @staticmethod
    def _calc_correct_values(values: NDArray, expected: NDArray) -> NDArray:
        samples = len(values)
        shape_length = len(expected.shape)

        if shape_length == 1:
            return values[range(samples), expected]

        return np.sum(values * expected, axis=1)


class MeanSquaredError(Loss):
    # From Neural Networks from Scratch in Python
    # For regression
    def forward(self, values: NDArray, expected: NDArray) -> NDArray:
        return np.mean((expected - values) ** 2, axis=-1)

    def backward(self, d_values: NDArray, expected: NDArray):
        samples = len(d_values)
        outputs = len(d_values[0])

        self.d_inputs = -2 * (expected - d_values) / outputs
        self.d_inputs = self.d_inputs / samples


class Accuracy:
    @staticmethod
    def calculate(values: NDArray, expected: NDArray):
        predictions = np.argmax(values, axis=1)

        if len(expected.shape) == 2:
            targets = np.argmax(expected, axis=1)

            return np.mean(predictions == targets)

        return np.mean(predictions == expected)
