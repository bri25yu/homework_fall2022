from typing import Union

from dataclasses import dataclass

import numpy as np

import torch

from rl.infrastructure.environment_info import EnvironmentInfo
from rl.infrastructure.pytorch_utils import TORCH_FLOAT_DTYPE


__all__ = ["BatchTrajectory"]


@dataclass
class BatchTrajectory:
    environment_info: EnvironmentInfo
    batch_size: int
    device: str
    observations: torch.Tensor        # Shape: (batch_size, trajectory_length, *observation_shape)
    actions: torch.Tensor             # Shape: (batch_size, trajectory_length, *action_shape)
    next_observations: torch.Tensor   # Shape: (batch_size, trajectory_length, *observation_shape)
    rewards: torch.Tensor             # Shape: (batch_size, trajectory_length, 1)
    terminals: torch.Tensor           # Shape: (batch_size, trajectory_length, 1). 1 if terminal, 0 otherwise

    def __post_init__(self) -> None:
        self.to_device(self.device)

        batch_size = self.batch_size
        max_trajectory_length = self.environment_info.max_trajectory_length
        observation_shape = self.environment_info.observation_shape
        action_shape = self.environment_info.action_shape

        assert self.observations.size() == (batch_size, max_trajectory_length, *observation_shape)
        assert self.actions.size() == (batch_size, max_trajectory_length, *action_shape)
        assert self.next_observations.size() == (batch_size, max_trajectory_length, *observation_shape)
        assert self.rewards.size() == (batch_size, max_trajectory_length, 1)
        assert self.terminals.size() == (batch_size, max_trajectory_length, 1)

    @classmethod
    def create(
        cls, environment_info: EnvironmentInfo, batch_size: int, device: str, initial_observation: Union[None, np.ndarray]=None
    ) -> "BatchTrajectory":
        max_trajectory_length = environment_info.max_trajectory_length
        observation_shape = environment_info.observation_shape
        action_shape = environment_info.action_shape

        trajectory = BatchTrajectory(
            environment_info=environment_info,
            batch_size=batch_size,
            device=device,
            observations=torch.zeros((batch_size, max_trajectory_length, *observation_shape), dtype=TORCH_FLOAT_DTYPE),
            actions=torch.zeros((batch_size, max_trajectory_length, *action_shape), dtype=TORCH_FLOAT_DTYPE),
            next_observations=torch.zeros((batch_size, max_trajectory_length, *observation_shape), dtype=TORCH_FLOAT_DTYPE),
            rewards=torch.zeros((batch_size, max_trajectory_length, 1), dtype=TORCH_FLOAT_DTYPE),
            terminals=torch.ones((batch_size, max_trajectory_length, 1), dtype=torch.bool),
        )

        if initial_observation is not None:
            assert batch_size == 1
            trajectory.observations[0, 0] = torch.from_numpy(initial_observation)

        return trajectory

    def update_from_numpy(
        self, index: int, action: np.ndarray, next_observation: np.ndarray, reward: float, terminal: bool
    ) -> None:
        """
        Assumes this trajectory has been initialized from `create` with an `initial_observation`.
        """
        assert self.batch_size == 1
        assert 0 <= index < self.environment_info.max_trajectory_length
        assert action.shape == self.environment_info.action_shape
        assert next_observation.shape == self.environment_info.observation_shape

        next_observation_pt = torch.from_numpy(next_observation)

        self.actions[0, index] = torch.from_numpy(action)
        self.next_observations[0, index] = next_observation_pt
        self.rewards[0, index] = reward
        self.terminals[0, index] = terminal

        if index + 1 < self.environment_info.max_trajectory_length:
            self.observations[0, index+1] = next_observation_pt

    def to_device(self, device: str) -> None:
        self.device = device
        self.observations = self.observations.to(device=device)
        self.actions = self.actions.to(device=device)
        self.next_observations = self.next_observations.to(device=device)
        self.rewards = self.rewards.to(device=device)
        self.terminals = self.terminals.to(device=device)
