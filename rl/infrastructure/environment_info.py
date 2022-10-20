from typing import Tuple

from dataclasses import dataclass


__all__ = ["EnvironmentInfo"]


@dataclass
class EnvironmentInfo:
    max_trajectory_length: int
    observation_shape: Tuple[int, ...]
    action_shape: Tuple[int, ...]
