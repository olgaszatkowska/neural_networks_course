from numpy.typing import NDArray

from models.layers import DenseLayer
from models.weights_initializer import Initializer, RandomInitializer, XavierInitializer
from models.activation_functions import ActivationFunction, SoftMax, ReLU, Sigmoid


class BaseNeuralNetwork:
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        number_of_hidden_layers: int,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.number_of_hidden_layers = number_of_hidden_layers

        self.layers = self._layers()
        self.activation_fns = self._activation_functions()

    def forward(self, inputs: NDArray) -> NDArray:
        vector = inputs

        for layer, activation_fn in zip(self.layers, self.activation_fns):
            layer.forward(vector)
            activation_fn.forward(layer.output)

            vector = activation_fn.output

        return vector

    def backwards(self, d_loss: NDArray):
        gradient = d_loss

        for layer, activation_fn in zip(
            reversed(self.layers),
            reversed(self.activation_fns),
        ):
            activation_fn.backward(gradient)
            layer.backward(activation_fn.d_inputs)

            gradient = layer.d_inputs

    def _layers(self):
        raise NotImplemented

    def _create_layers(self, *initializers: list[Initializer]):
        input_init, hidden_init, output_init, *_ = initializers
        layers = [
            DenseLayer(self.input_dim, self.hidden_dim, weights_initializer=input_init)
        ]

        for _ in range(self.number_of_hidden_layers):
            layer = DenseLayer(
                self.hidden_dim, self.hidden_dim, weights_initializer=hidden_init
            )
            layers.append(layer)

        layers.append(
            DenseLayer(
                self.hidden_dim, self.output_dim, weights_initializer=output_init
            )
        )

        return layers

    def _activation_functions(self):
        raise NotImplemented

    def _create_activation_functions(self, *functions: list[ActivationFunction]):
        input_fn, hidden_fn, output_fn, *_ = functions

        fns = [input_fn]

        for _ in range(self.number_of_hidden_layers):
            fns.append(hidden_fn)

        fns.append(output_fn)

        return fns

    def __str__(self):
        value = ""
        for layer, activation_fn in zip(self.layers, self.activation_fns):
            value += f"{layer} -> {activation_fn}\n"

        return value


class ClassificationNeuralNetwork(BaseNeuralNetwork):
    def _layers(self) -> list:
        return self._create_layers(
            RandomInitializer, XavierInitializer, RandomInitializer
        )

    def _activation_functions(self):
        return self._create_activation_functions(ReLU(), ReLU(), SoftMax())


class RegressionNeuralNetwork(BaseNeuralNetwork):
    def _layers(self) -> list:
        return self._create_layers(
            RandomInitializer, XavierInitializer, RandomInitializer
        )

    def _activation_functions(self):
        return self._create_activation_functions(ReLU(), ReLU(), Sigmoid())
