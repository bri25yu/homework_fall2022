from typing import Any, Dict, Union

from dataclasses import dataclass

import torch


__all__ = ["ModelOutput"]


@dataclass
class ModelOutput:
    action: Union[None, torch.Tensor] = None
    loss: Union[None, torch.Tensor] = None
    logs: Union[None, Dict[str, Any]] = None
