from dataclasses import dataclass

import torch


__all__ = ["ModelOutput"]


@dataclass
class ModelOutput:
    action: torch.Tensor
    loss: torch.Tensor
