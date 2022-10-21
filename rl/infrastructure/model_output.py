from dataclasses import dataclass

import torch


__all__ = ["ModelOutput"]


@dataclass
class ModelOutput:
    actions: torch.Tensor    # a tensor of shape (batch_size, max_sequence_length, *action_shape)
    loss: torch.Tensor      # a torch scalar
