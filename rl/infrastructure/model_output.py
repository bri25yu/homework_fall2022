from typing import Any, Dict, Union

from dataclasses import dataclass

import torch


__all__ = ["ModelOutput"]


@dataclass
class ModelOutput:
    actions: Union[None, torch.Tensor] = None  # Of shape (L, *action_shape)
    loss: Union[None, torch.Tensor] = None
    logs: Union[None, Dict[str, Any]] = None
