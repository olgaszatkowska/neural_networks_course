from numpy.typing import NDArray

from models.neural_network import BaseNeuralNetwork
from models.metrics import Loss, Accuracy


class Optimizer:
    def __init__(
        self,
        network: BaseNeuralNetwork,
        loss: Loss,
        accuracy: Accuracy,
        learning_rate: float,
        batch_size: int,
        epochs: int,
    ):
        self.network = network
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.accuracy_fn = accuracy
        self.loss_fn = loss
        
        self.accuracy = []
        self.loss = []

    def fit(self, x: NDArray, y: NDArray):
        batch_count = x.shape[0] // self.batch_size

        for epoch in range(self.epochs):
            acc, loss = self._fit_batch(x, y, batch_count)
            print(f"Epoch {epoch}  --  accuracy {acc:.3f} - loss {loss:.3f}")

    def _fit_batch(self, x: NDArray, y: NDArray, batch_count: int):
        acc, loss = 0.0, 0.0

        for batch_no in range(batch_count):
            x_window, y_window = self._get_window(batch_no, x, y)

            predictions = self.network.forward(x_window)

            acc = self.accuracy_fn.calculate(predictions, y_window)
            loss = self.loss_fn.calculate(predictions, y_window)
                
            self._append_metrics(acc, loss)
            self._backward(predictions, y_window)
            
        return acc, loss

    def _get_window(self, batch_no: int, x: NDArray, y: NDArray):
        start_idx = batch_no * self.batch_size
        end_idx = start_idx + self.batch_size

        return x[start_idx:end_idx], y[start_idx:end_idx]

    def _backward(self, predictions: NDArray, y_window: NDArray):
        self.loss_fn.backward(predictions, y_window)
        self.network.backwards(self.loss_fn.d_inputs)
        for layer in self.network.layers:
            layer.weights += -self.learning_rate * layer.d_weights
            layer.biases += -self.learning_rate * layer.d_bias

    def _append_metrics(self, acc: float, loss: float):
        self.accuracy.append(acc)
        self.loss.append(loss)