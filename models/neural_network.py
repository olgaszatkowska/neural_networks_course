from numpy.typing import NDArray

from models.layers import DenseLayer
from models.weights_initializer import Initializer, RandomInitializer, XavierInitializer
from models.activation_functions import ActivationFunction, SoftMax, ReLU, Sigmoid


class BaseNeuralNetwork:
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

    def __str__(self):
        value = ""
        for layer, activation_fn in zip(self.layers, self.activation_fns):
            value += f"{layer} -> {activation_fn} \n"

        return value


class CustomizableNeuralNetwork(BaseNeuralNetwork):
    def __init__(
        self,
        input_dim: int,
    ):
        self.input_dim = input_dim

        self.layers = []
        self.activation_fns = []

    def add_layer(
        self,
        neurons_count: int,
        activation_fn: ActivationFunction,
        initializer: Initializer,
    ):
        if self.layers != []:
            input_dim = self.layers[-1].neurons_count
        else:
            input_dim = self.input_dim
        self.layers.append(DenseLayer(input_dim, neurons_count, initializer))
        self.activation_fns.append(activation_fn)


class PredefinedNeuralNetwork(BaseNeuralNetwork):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        number_of_hidden_layers: int,
        activation_fns: list[ActivationFunction],
        weight_initializers: list[Initializer],
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.number_of_hidden_layers = number_of_hidden_layers

        self.layers = self._layers(weight_initializers)
        self.activation_fns = self._activation_fns(activation_fns)

    def _layers(self, initializers: list[Initializer]):
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

    def _activation_fns(self, functions: list[ActivationFunction]):
        input_fn, hidden_fn, output_fn, *_ = functions

        fns = [input_fn]

        for _ in range(self.number_of_hidden_layers):
            fns.append(hidden_fn)

        fns.append(output_fn)

        return fns


class ClassificationNeuralNetwork(PredefinedNeuralNetwork):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        number_of_hidden_layers: int,
    ):
        super().__init__(
            input_dim,
            hidden_dim,
            output_dim,
            number_of_hidden_layers,
            activation_fns=[ReLU(), ReLU(), SoftMax()],
            weight_initializers=[
                RandomInitializer,
                XavierInitializer,
                RandomInitializer,
            ],
        )


class RegressionNeuralNetwork(PredefinedNeuralNetwork):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        number_of_hidden_layers: int,
    ):
        super().__init__(
            input_dim,
            hidden_dim,
            output_dim,
            number_of_hidden_layers,
            activation_fns=[ReLU(), ReLU(), Sigmoid()],
            weight_initializers=[
                RandomInitializer,
                XavierInitializer,
                RandomInitializer,
            ],
        )
