from typing import Tuple, Union

from dataclasses import dataclass

import numpy as np

from gym import Env
from gym.spaces import Discrete


__all__ = ["EnvironmentInfo"]


@dataclass
class EnvironmentInfo:
    observation_shape: Tuple[int, ...]
    action_shape: Tuple[int, ...]
    max_episode_steps: int
    is_discrete: bool
    action_range: Union[None, Tuple[np.ndarray, np.ndarray]]  # None if is_discrete

    @classmethod
    def from_env(cls, env: Env) -> "EnvironmentInfo":
        is_discrete = isinstance(env.action_space, Discrete)
        return EnvironmentInfo(
            observation_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            max_episode_steps=env.spec.max_episode_steps,
            is_discrete=is_discrete,
            action_range=(env.action_space.low, env.action_space.high) if not is_discrete else None,
        )

    @property
    def observation_dim(self) -> int:
        return np.prod(self.observation_shape)

    @property
    def action_dim(self) -> int:
        return np.prod(self.action_shape)
