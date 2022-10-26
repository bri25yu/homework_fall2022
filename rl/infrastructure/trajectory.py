import numpy as np

import torch

from rl.infrastructure.environment_info import EnvironmentInfo
from rl.infrastructure.pytorch_utils import TORCH_FLOAT_DTYPE, TORCH_DEVICE


__all__ = ["Trajectory"]


class Trajectory:
    @classmethod
    def create(cls, environment_info: EnvironmentInfo, device: str, L: int) -> "Trajectory":
        observation_dim = environment_info.observation_dim
        action_dim = environment_info.action_dim

        # Technically we can combine observation and next observation data
        # but it's unnecessarily complicated so we just make a copy of the data
        # !TODO reduce memory footprint of observation/next_observation
        # We store the observations, actions, next_observations, and rewards in this array
        data_dim = observation_dim + action_dim + observation_dim + 1

        _data = torch.zeros((L, data_dim), device=device, dtype=TORCH_FLOAT_DTYPE)
        _terminals = torch.ones((L, 1), device=device, dtype=torch.bool)

        return Trajectory(environment_info=environment_info, L=L, _data=_data, _terminals=_terminals)

    def __init__(
        self, environment_info: EnvironmentInfo, L: int, _data: torch.Tensor, _terminals: torch.Tensor
    ) -> None:
        self.environment_info = environment_info
        self.L = L

        observation_dim = environment_info.observation_dim
        action_dim = environment_info.action_dim

        self._data = _data
        self._terminals = _terminals

        self.observations_slice = slice(0, observation_dim)
        self.actions_slice = slice(self.observations_slice.stop, self.observations_slice.stop + action_dim)
        self.next_observations_slice = slice(self.actions_slice.stop, self.actions_slice.stop + observation_dim)
        self.rewards_slice = slice(self.next_observations_slice.stop, self.next_observations_slice.stop + 1)

    @property
    def observations(self) -> torch.Tensor:  # (L, *observation_shape)
        return self._data[:, self.observations_slice].view(self.L, *self.environment_info.observation_shape)

    @property
    def actions(self) -> torch.Tensor:  # (L, *action_shape)
        return self._data[:, self.actions_slice].view(self.L, *self.environment_info.action_shape)

    @property
    def next_observations(self) -> torch.Tensor:  # (L, observation_dim)
        return self._data[:, self.next_observations_slice]

    @property
    def rewards(self) -> torch.Tensor:  # (L, 1)
        return self._data[:, self.rewards_slice]

    @property
    def terminals(self) -> torch.Tensor:  # (L, 1)
        return self._terminals

    def initialize_from_numpy(self, index: int, initial_observation: np.ndarray) -> None:
        self.observations[index] = torch.from_numpy(initial_observation)

    def update_from_numpy(
        self, index: int, action: np.ndarray, next_observation: np.ndarray, reward: float, terminal: bool
    ) -> None:
        """
        Assumes this trajectory has been initialized from initialize_from_numpy.
        """
        assert 0 <= index < self.L
        assert action.shape == self.environment_info.action_shape
        assert next_observation.shape == self.environment_info.observation_shape

        next_observation_pt = torch.from_numpy(next_observation)
        action_pt = torch.from_numpy(action)

        self.actions[index] = action_pt
        self.next_observations[index] = next_observation_pt
        self.rewards[index] = reward
        self.terminals[index] = terminal

        if index + 1 < self.L:
            self.observations[index+1] = next_observation_pt

    def to_device(self, device: str) -> None:
        self._data = self._data.to(device=device)
        self._terminals = self._terminals.to(device=device)

    def cpu(self) -> None:
        self.to_device("cpu")

    def cuda(self) -> None:
        self.to_device(TORCH_DEVICE)

    def take(self, indices: torch.Tensor) -> "Trajectory":
        return Trajectory(
            environment_info=self.environment_info,
            L=indices.shape[0],
            _data=self._data[indices].clone(),
            _terminals=self._terminals[indices].clone(),
        )

    def overwrite_indices(self, indices: torch.Tensor, trajectories: "Trajectory") -> None:
        self._data[indices] = trajectories._data
        self.terminals[indices] = trajectories._terminals
