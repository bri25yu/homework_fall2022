from typing import Tuple

try:  # `prod` is a native python function starting with python 3.8
    from math import prod
except ImportError:
    # For all previous versions, we define a custom prod function
    from functools import reduce
    from operator import mul


    prod = lambda t: reduce(mul, t)

import torch
import torch.nn as nn

from dataclasses import dataclass


__all__ = ["TORCH_DEVICE", "FFNConfig", "CNNConfig", "build_ffn", "build_cnn", "build_log_std"]


TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_FLOAT_DTYPE = torch.float32


def get_activation(activation_name: str) -> nn.Module:
    activation_function = getattr(nn, activation_name)()  
    return activation_function


@dataclass
class FFNConfig:
    in_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]
    n_layers: int = 2
    hidden_dim: int = 64


@dataclass
class CNNConfig:
    in_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]
    kernel_size: int
    n_layers: int
    hidden_dim: int


class ReshapeLayer(nn.Module):
    def __init__(self, in_shape: Tuple[int, ...], out_shape: Tuple[int, ...]) -> None:
        super().__init__()

        assert prod(in_shape) == prod(out_shape)

        self.in_shape = in_shape
        self.out_shape = out_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size()[-len(self.in_shape):] == self.in_shape

        if self.in_shape == self.out_shape:
            return x

        preceding_dims = x.size()[:-len(self.in_shape)]
        final_shape = preceding_dims + self.out_shape

        return x.reshape(final_shape)


def build_ffn(ffn_config: FFNConfig) -> nn.Module:
    flattened_in_dim = prod(ffn_config.in_shape)
    flattened_out_dim = prod(ffn_config.out_shape)

    in_dims = [flattened_in_dim] + [ffn_config.hidden_dim] * (ffn_config.n_layers + 1)
    out_dims = [ffn_config.hidden_dim] * (ffn_config.n_layers + 1) + [flattened_out_dim]
    activations = ["GELU"] * (ffn_config.n_layers + 1) + ["Identity"]

    layers = []
    for in_dim, out_dim, activation in zip(in_dims, out_dims, activations):
        layers.extend((nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim), get_activation(activation)))

    input_reshape = ReshapeLayer(ffn_config.in_shape, (flattened_in_dim,))
    output_reshape = ReshapeLayer((flattened_out_dim,), ffn_config.out_shape)

    return nn.Sequential(input_reshape, *layers, output_reshape)


def build_cnn(cnn_config: CNNConfig) -> nn.Module:
    raise NotImplementedError


def build_log_std(shape: Tuple[int, ...]) -> nn.Parameter:
    return nn.Parameter(torch.zeros(shape))
