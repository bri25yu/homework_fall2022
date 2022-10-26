from typing import Tuple

from dataclasses import dataclass

import numpy as np

from gym import Env


__all__ = ["EnvironmentInfo"]


@dataclass
class EnvironmentInfo:
    observation_shape: Tuple[int, ...]
    action_shape: Tuple[int, ...]
    max_episode_steps: int
    action_range: Tuple[np.ndarray, np.ndarray]

    @classmethod
    def from_env(cls, env: Env) -> "EnvironmentInfo":
        return EnvironmentInfo(
            observation_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            max_episode_steps=env.spec.max_episode_steps,
            action_range=(env.action_space.low, env.action_space.high),
        )

    @property
    def observation_dim(self) -> int:
        return np.prod(self.observation_shape)

    @property
    def action_dim(self) -> int:
        return np.prod(self.action_shape)
