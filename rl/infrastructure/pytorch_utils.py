from typing import Tuple, Union

from dataclasses import dataclass

import copy

from numpy import prod

import torch
import torch.nn as nn

from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Stack, T5PreTrainedModel


__all__ = [
    "TORCH_DEVICE",
    "TORCH_FLOAT_DTYPE",
    "to_numpy",
    "build_ffn",
    "build_log_std",
    "normalize",
    "FFNConfig",
    "T5ForReinforcementLearning",
    "build_transformer",
]


TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_FLOAT_DTYPE = torch.float32


def to_numpy(t: torch.Tensor) -> float:
    return t.detach().to("cpu").numpy()


def get_activation(activation_name: str) -> nn.Module:
    activation_function = getattr(nn, activation_name)()  
    return activation_function


def normalize(t: torch.Tensor) -> torch.Tensor:
    return (t - t.mean()) / (t.std() + 1e-8)


@dataclass
class FFNConfig:
    in_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]
    n_layers: int = 2
    hidden_dim: int = 64


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
        layers.extend((nn.Linear(in_dim, out_dim), get_activation(activation)))

    input_reshape = ReshapeLayer(ffn_config.in_shape, (flattened_in_dim,))
    output_reshape = ReshapeLayer((flattened_out_dim,), ffn_config.out_shape)

    return nn.Sequential(input_reshape, *layers, output_reshape)


def build_log_std(shape: Tuple[int, ...]) -> nn.Parameter:
    return nn.Parameter(torch.zeros(shape))


class T5ForReinforcementLearning(T5PreTrainedModel):
    def __init__(self, config: T5Config) -> None:
        super().__init__(config)

        # This is an exact copy of `T5Model.__init__`
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        encoder_outputs = self.encoder(inputs_embeds=trajectory)
        last_hidden_state = encoder_outputs.last_hidden_state

        decoder_outputs = self.decoder(inputs_embeds=last_hidden_state)
        return decoder_outputs.last_hidden_state


def build_transformer(transformer_config: Union[None, T5Config]=None, model_dim: int=None) -> nn.Module:
    assert transformer_config or model_dim, "Must provide either a config or a model dimension"

    if transformer_config is None:  # Default model
        transformer_config = T5Config(
            d_model=model_dim,
            d_kv=64,
            dff=64,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=4,
        )

    return T5ForReinforcementLearning(transformer_config)
