from typing import Tuple

from dataclasses import dataclass

import numpy as np


__all__ = ["EnvironmentInfo"]


@dataclass
class EnvironmentInfo:
    observation_shape: Tuple[int, ...]
    action_shape: Tuple[int, ...]
    max_episode_steps: int

    @property
    def observation_dim(self) -> int:
        return np.prod(self.observation_shape)

    @property
    def action_dim(self) -> int:
        return np.prod(self.action_shape)
