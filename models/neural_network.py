from numpy.typing import NDArray
from typing import Type

from models.layers import DenseLayer
from models.weights_initializer import Initializer, RandomInitializer, XavierInitializer
from models.activation_functions import (
    ActivationFunction,
    SoftMax,
    ReLU,
    Linear,
    Sigmoid,
)


class BaseNeuralNetwork:
    layers: list[DenseLayer]
    activation_fns: list[ActivationFunction]

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


class CustomizedNeuralNetwork(BaseNeuralNetwork):
    def __init__(
        self,
        *,
        input_dim: int,
    ):
        self.input_dim = input_dim
        self.layers: list[DenseLayer] = []
        self.activation_fns: list[ActivationFunction] = []

    def add_layer(
        self,
        neurons_count: int,
        activation_fn: ActivationFunction,
        initializer: Type[Initializer],
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
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        number_of_hidden_layers: int,
        activation_fns: list[ActivationFunction],
        weight_initializers: list[Type[Initializer]],
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.number_of_hidden_layers = number_of_hidden_layers

        self.layers = self._generate_layers(weight_initializers)
        self.activation_fns = self._generate_activation_fns(activation_fns)

    def _generate_layers(self, initializers: list[Type[Initializer]]):
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

    def _generate_activation_fns(self, functions: list[ActivationFunction]):
        input_fn, hidden_fn, output_fn, *_ = functions

        fns = [input_fn]

        for _ in range(self.number_of_hidden_layers):
            fns.append(hidden_fn)

        fns.append(output_fn)

        return fns


class ClassificationNeuralNetwork(PredefinedNeuralNetwork):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        number_of_hidden_layers: int,
    ):

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            number_of_hidden_layers=number_of_hidden_layers,
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
        *,
        input_dim: int,
        hidden_dim: int,
        number_of_hidden_layers: int,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            number_of_hidden_layers=number_of_hidden_layers,
            activation_fns=[ReLU(), ReLU(), Linear()],
            weight_initializers=[
                RandomInitializer,
                XavierInitializer,
                RandomInitializer,
            ],
        )


class Autoencoder(CustomizedNeuralNetwork):
    def __init__(self,  *, input_dim: int, code_dim: int, encoder_hidden_count: int, reduce_by: int,) -> None:
        super().__init__(input_dim=input_dim)
        self.code_dim = code_dim
        self._add_layers(encoder_hidden_count, reduce_by)
       
    def _add_layers(self, encoder_hidden_count: int, reduce_by: int):
        self.add_layer(self.input_dim, ReLU(), XavierInitializer)

        hidden_dim = self.input_dim // reduce_by
        saved_dims = []
        
        for _ in range(encoder_hidden_count):
            hidden_dim = int(hidden_dim // reduce_by)
            saved_dims.append(hidden_dim)
            self.add_layer(hidden_dim, ReLU(), RandomInitializer)
            
        self.add_layer(self.code_dim, ReLU(), XavierInitializer)
        
        for dim in reversed(saved_dims):
            self.add_layer(dim, ReLU(), RandomInitializer)
        
        self.add_layer(self.input_dim, Linear(), XavierInitializer)
        
    def get_encoded(self, inputs: NDArray, slice: int):
        vector = inputs

        for layer, activation_fn in zip(self.layers[:slice], self.activation_fns[:slice]):
            layer.forward(vector)
            activation_fn.forward(layer.output)

            vector = activation_fn.output

        return vector
