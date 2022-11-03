import numpy as np

import torch

from gym import Env
from gym.spaces import Discrete

from rl.infrastructure.pytorch_utils import TORCH_FLOAT_DTYPE, TORCH_DEVICE


__all__ = ["Trajectory"]


class Trajectory:
    def __init__(self, L: int, env: Env) -> None:
        self.L = L

        obs_shape = env.observation_space.shape
        ac_shape = () if isinstance(env.action_space, Discrete) else env.action_space.shape

        create = lambda shape: torch.zeros((self.L, *shape), dtype=TORCH_FLOAT_DTYPE, device=TORCH_DEVICE)

        self.observations = create(obs_shape)
        self.actions = create(ac_shape)
        self.next_observations = create(obs_shape)
        self.rewards = create((1,))
        self.terminals = torch.ones((L, 1), dtype=torch.bool, device=TORCH_DEVICE)

    def update_observations_from_numpy(self, index: int, observation: np.ndarray) -> None:
        self.observations[index] = torch.from_numpy(observation)

    def update_consequences_from_numpy(
        self, index: int, action: np.ndarray, next_observation: np.ndarray, reward: float, terminal: bool
    ) -> None:
        next_observation_pt = torch.from_numpy(next_observation)
        action_pt = torch.from_numpy(action)

        self.actions[index] = action_pt
        self.next_observations[index] = next_observation_pt
        self.rewards[index] = reward
        self.terminals[index] = terminal

        if index + 1 < self.L:
            self.observations[index+1] = next_observation_pt
