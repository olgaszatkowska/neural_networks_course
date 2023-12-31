from typing import Optional
from numpy.typing import NDArray
import numpy as np
import copy

from models.neural_network import BaseNeuralNetwork
from models.metrics import Loss, Accuracy
from models.layers import DenseLayer, Conv2D


class Optimizer:
    def __init__(
        self,
        network: BaseNeuralNetwork,
        loss: Loss,
        learning_rate: float,
        batch_size: int,
        epochs: int,
        reshape: bool = False,
        accuracy: Optional[Accuracy] = None,
        early_stopping: bool = False,
    ):
        self.network = network
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.accuracy_fn = accuracy
        self.loss_fn = loss

        self.accuracy = []
        self.loss = []
        self.validation_loss = []

        self.reshape = reshape

        self.count_accuracy = accuracy != None
        self.early_stopping = early_stopping

    def fit(
        self, x: NDArray, y: NDArray, x_valid: NDArray = None, y_valid: NDArray = None
    ):
        batch_count = x.shape[0] // self.batch_size
        validation_loss = None
        saved_model = None

        for epoch in range(self.epochs):
            if self.early_stopping:
                saved_model = copy.deepcopy(self.network)

            acc, loss = self._fit_batch(x, y, batch_count)
            self._print_metrics(epoch, acc, loss)

            if not self.early_stopping:
                continue

            predictions = self.network.forward(x_valid)
            validation_loss_new = self.loss_fn.calculate(predictions, y_valid)

            if self.early_stopping and validation_loss != None:
                self.validation_loss.append(validation_loss_new)

                if (validation_loss_new - validation_loss) > 0:
                    self.network = saved_model
                    print(f"Early stopping at {validation_loss_new}")
                    return

            validation_loss = validation_loss_new

    def _fit_batch(self, x: NDArray, y: NDArray, batch_count: int):
        acc, loss = 0.0, 0.0
        batch_acc, batch_loss = [], []

        for batch_no in range(batch_count):
            x_window, y_window = self._get_window(batch_no, x, y)
            predictions = self.network.forward(x_window)

            if self.count_accuracy:
                acc = self.accuracy_fn.calculate(predictions, y_window)

            loss = self.loss_fn.calculate(predictions, y_window)

            batch_acc.append(acc)
            batch_loss.append(loss)

            self._backward(predictions, y_window)

        avg_acc = np.average(batch_acc)
        avg_loss = np.average(batch_loss)

        self._append_metrics(avg_acc, avg_loss)

        return avg_acc, avg_loss

    def _get_window(self, batch_no: int, x: NDArray, y: NDArray):
        start_idx = batch_no * self.batch_size
        end_idx = start_idx + self.batch_size

        y_window = y[start_idx:end_idx]

        if self.reshape:
            return x[start_idx:end_idx], y_window.reshape(-1, 1)

        return x[start_idx:end_idx], y_window

    def _backward(self, predictions: NDArray, y_window: NDArray):
        self.loss_fn.backward(predictions, y_window)
        self.network.backwards(self.loss_fn.d_inputs)
        for layer in self.network.layers:
            layer.weights += -self.learning_rate * layer.d_weights
            layer.biases += -self.learning_rate * layer.d_bias

    def _append_metrics(self, acc: float, loss: float):
        self.accuracy.append(acc)
        self.loss.append(loss)

    def _print_metrics(self, epoch: int, acc: float, loss: float) -> None:
        acc_str = f"accuracy {acc:.3f} -" if self.count_accuracy else ""
        print(f"Epoch {epoch}  -- {acc_str} loss {loss:.3f}")


class ConvolutionOptimizer:
    def __init__(self, network, *, learning_rate: float):
        self.network = network
        self.learning_rate = learning_rate
        self.loss = []

    def fit(self):
        for layer in self.network.layers:
            if isinstance(layer, DenseLayer):
                layer.weights += -self.learning_rate * layer.d_weights
                layer.biases += -self.learning_rate * layer.d_bias
            elif isinstance(layer, Conv2D):
                layer.kernel += -self.learning_rate * layer.d_weights
                layer.biases += -self.learning_rate * layer.d_bias
                
    def save_loss(self, loss):
        self.loss.append(loss)
