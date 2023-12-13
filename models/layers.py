import math
import numpy as np
from numpy.typing import NDArray
from numpy.lib.stride_tricks import as_strided

from models.weights_initializer import Initializer, RandomInitializer


class DenseLayer:
    def __init__(
        self,
        input_count: int,
        neurons_count: int,
        weights_initializer: Initializer = RandomInitializer,
    ):
        self.neurons_count = neurons_count
        self.input_count = input_count
        self.initializer = weights_initializer(
            np.array([self.input_count, self.neurons_count])
        )

        self.weights = self.initializer.initialize()
        self.biases = np.zeros((1, self.neurons_count))

        self.input = np.zeros((1, self.input_count))
        self.output = np.zeros((1, self.neurons_count))

        self.d_weights = np.zeros((1, self.neurons_count))
        self.d_bias = np.zeros((1, self.neurons_count))
        self.d_inputs = np.zeros((1, self.input_count))

    def forward(self, inputs: NDArray) -> NDArray:
        self.input = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

        return self.output

    def backward(self, d_values: NDArray) -> None:
        # Set gradient on layer params
        self.d_weights = np.dot(self.input.T, d_values)
        self.d_bias = np.sum(d_values, axis=0, keepdims=True)

        # Set gradient on inputs
        self.d_inputs = np.dot(d_values, self.weights.T)

    def __str__(self):
        return f"DenseLayer: {self.input_count} x {self.neurons_count} -> {self.initializer}"


class Conv2D:
    def __init__(
        self,
        in_channels: int,
        kernel_side: int,
        kernel_count: int,
        stride=1,
        padding=0,
    ):
        self.in_channels = in_channels
        self.kernel_side = kernel_side
        self.stride = stride
        self.padding = padding

        self.kernel_count = kernel_count
        self.kernel = self._get_kernel()
        self.biases = np.zeros(self.kernel_count)

        self.input = None

    def _get_kernel(self):
        return np.random.randn(
            self.kernel_count, self.in_channels, self.kernel_side, self.kernel_side
        ) * np.sqrt(2.0 / (self.in_channels * self.kernel_side * self.kernel_side))

    def forward(self, inputs: NDArray) -> NDArray:
        self.input = np.pad(
            inputs,
            (
                (0, 0),
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
            ),
            mode="constant",
        )
        in_strides = self.input.strides

        batch_size, in_channels, input_h, input_w = self.input.shape

        output_h, output_w = self._get_output_shape(input_h, input_w)

        self.output = np.zeros((batch_size, self.kernel_count, output_h, output_w))

        stride_view = self._get_forward_stride_view(
            batch_size, in_channels, (output_h, output_w), in_strides
        )

        for kernel_idx, kernel in enumerate(self.kernel):
            self._add_bias(kernel_idx, stride_view, kernel)

        return self.output

    def _get_forward_stride_view(
        self, batch_size, in_channels, output_shape, in_strides
    ):
        output_h, output_w = output_shape
        stride_shape = (
            batch_size,
            in_channels,
            output_h,
            output_w,
            self.kernel_side,
            self.kernel_side,
        )
        return as_strided(
            self.input,
            shape=stride_shape,
            strides=(
                *in_strides[:2],
                in_strides[2] * self.stride,
                in_strides[3] * self.stride,
                *in_strides[2:],
            ),
        )

    def _add_bias(self, kernel_idx, stride_view, kernel):
        self.output[:, kernel_idx, :, :] = np.tensordot(
            stride_view,
            kernel,
            axes=([1, 4, 5], [0, 1, 2]),
        )

    def _get_output_shape(self, input_h, input_w):
        output_h = math.ceil((input_h - self.kernel_side) / self.stride)
        output_w = math.ceil(input_w - self.kernel_side / self.stride)

        return output_h, output_w

    def backward(self, d_values: NDArray) -> None:

        batch_size, input_channels, input_h, input_w = self.input.shape
        _, _, output_h, output_w = d_values.shape
        in_strides = self.input.strides

        self._init_d_as_zeros()

        stride_view_kernel = as_strided(
            self.input,
            shape=(
                batch_size,
                input_channels,
                output_h,
                output_w,
                self.kernel_side,
                self.kernel_side,
            ),
            strides=(
                *in_strides[:2],
                in_strides[2] * self.stride,
                in_strides[3] * self.stride,
                *in_strides[2:],
            ),
        )

        for idx in range(self.kernel_count):
            kernel_d_values = d_values[:, idx, :, :].reshape(
                batch_size, 1, output_h, output_w, 1, 1
            )
            self.d_weights[idx] = np.sum(
                stride_view_kernel * kernel_d_values, axis=(0, 2, 3)
            )
            self.d_bias[idx] = np.sum(d_values[:, idx, :, :], axis=(0, 1, 2))

        inv_kernel = self.kernel[:, :, ::-1, ::-1]
        d_values_with_pad = np.pad(
            d_values,
            (
                (0, 0),
                (0, 0),
                (self.kernel_side - 1, self.kernel_side - 1),
                (self.kernel_side - 1, self.kernel_side - 1),
            ),
            mode="constant",
        )
        d_values_with_pad_strides = d_values_with_pad.strides

        out_stride = self._get_backward_stride_view(
            batch_size, (input_h, input_w), d_values_with_pad, d_values_with_pad_strides
        )

        self.d_inputs = np.tensordot(
            out_stride, inv_kernel, axes=((1, 4, 5), (0, 2, 3))
        )

        has_padding = self.padding != 0

        if has_padding:
            self.d_inputs = self.d_inputs[
                :, :, self.padding : -self.padding, self.padding : -self.padding
            ]

        self.d_inputs = self.d_inputs.transpose((0, 3, 1, 2))

    def _init_d_as_zeros(self):
        self.d_weights = np.zeros_like(self.kernel)
        self.d_bias = np.zeros_like(self.biases)
        self.d_inputs = np.zeros_like(self.input)

    def _get_backward_stride_view(
        self, batch_size, input_shape, d_values_with_pad, d_values_with_pad_strides
    ):
        input_h, input_w = input_shape
        stride_shape = (
            batch_size,
            self.kernel_count,
            input_h,
            input_w,
            self.kernel_side,
            self.kernel_side,
        )

        return as_strided(
            d_values_with_pad,
            shape=stride_shape,
            strides=(
                *d_values_with_pad_strides[:2],
                d_values_with_pad_strides[2] * self.stride,
                d_values_with_pad_strides[3] * self.stride,
                *d_values_with_pad_strides[2:],
            ),
        )


class MaxPool2D:
    def __init__(
        self,
        kernel_side: int,
        stride=1,
    ):
        self.kernel_side = kernel_side
        self.stride = stride

        self.input = None

    def forward(self, inputs: NDArray) -> NDArray:
        self.input = inputs
        in_strides = self.input.strides

        batch_size, kernels, input_h, input_w = inputs.shape

        output_h = math.ceil((input_h - self.kernel_side) / self.stride)
        output_w = math.ceil((input_w - self.kernel_side) / self.stride)

        stride_view = as_strided(
            self.input,
            shape=(
                batch_size,
                kernels,
                output_h,
                output_w,
                self.kernel_side,
                self.kernel_side,
            ),
            strides=(
                *in_strides[:2],
                in_strides[2] * self.stride,
                in_strides[3] * self.stride,
                *in_strides[2:],
            ),
        )

        self.output = np.max(stride_view, axis=(4, 5))

        return self.output

    def backward(self, d_values: NDArray) -> None:
        batch_size, kernels, input_h, input_w = self.input.shape
        in_strides = self.input.strides

        output_h = d_values.shape[2]
        output_w = d_values.shape[3]

        stride_view = as_strided(
            self.input,
            shape=(
                batch_size,
                kernels,
                output_h,
                output_w,
                self.kernel_side,
                self.kernel_side,
            ),
            strides=(
                *in_strides[:2],
                in_strides[2] * self.stride,
                in_strides[3] * self.stride,
                *in_strides[2:],
            ),
        )

        self.d_inputs = np.zeros_like(self.input)

        for h in range(output_h):
            for w in range(output_w):
                max_indices = self._get_max_indices(
                    stride_view, h, w, batch_size, kernels
                )
                max_coords = self._get_max_coords(max_indices)

                for batch in range(batch_size):
                    for kernel in range(kernels):
                        coord_1 = max_coords[0][batch, kernel]
                        coord_2 = max_coords[1][batch, kernel]

                        h_start = h * self.stride
                        w_start = w * self.stride

                        local_err = d_values[batch, kernel, h, w]
                        self.d_inputs[
                            batch, kernel, h_start + coord_1, w_start + coord_2
                        ] += local_err

    def _get_max_indices(self, stride_view, h, w, batch_size, kernels):
        return np.argmax(
            stride_view[:, :, h, w].reshape(batch_size, kernels, -1), axis=2
        )

    def _get_max_coords(self, max_indices):
        return np.unravel_index(max_indices, (self.kernel_side, self.kernel_side))


class Flatten:
    def __init__(self):
        self.input = None
        self.shape = None

    def forward(self, inputs: NDArray) -> NDArray:
        self.input = inputs
        self.shape = inputs.shape
        self.output = inputs.reshape(inputs.shape[0], -1)

        return self.output

    def backward(self, d_values: NDArray) -> None:
        self.d_inputs = d_values.reshape(self.shape)
