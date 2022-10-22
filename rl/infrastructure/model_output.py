from typing import Any, Dict, Union

from dataclasses import dataclass

import torch


__all__ = ["ModelOutput"]


@dataclass
class ModelOutput:
    actions: torch.Tensor                       # A tensor of shape (batch_size, max_sequence_length, *action_shape)
    loss: torch.Tensor                          # A torch scalar
    logs: Union[None, Dict[str, Any]] = None    # Optional logs
