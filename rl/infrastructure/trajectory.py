from typing import List, Union

from dataclasses import dataclass

import numpy as np

import torch

from rl.infrastructure.environment_info import EnvironmentInfo
from rl.infrastructure.constants import NP_FLOAT_DTYPE, TORCH_FLOAT_DTYPE


__all__ = ["Trajectory", "BatchTrajectories", "BatchTrajectoriesPyTorch"]


@dataclass
class Trajectory:
    environment_info: EnvironmentInfo
    observations: np.ndarray        # Shape: (trajectory_length, *observation_shape)
    actions: np.ndarray             # Shape: (trajectory_length, *action_shape)
    next_observations: np.ndarray   # Shape: (trajectory_length, *observation_shape)
    rewards: np.ndarray             # Shape: (trajectory_length, 1)
    terminals: np.ndarray           # Shape: (trajectory_length, 1). 1 if terminal, 0 otherwise

    def __post_init__(self) -> None:
        max_trajectory_length = self.environment_info.max_trajectory_length
        observation_shape = self.environment_info.observation_shape
        action_shape = self.environment_info.action_shape

        assert self.observations.shape == (max_trajectory_length, *observation_shape)
        assert self.actions.shape == (max_trajectory_length, *action_shape)
        assert self.next_observations.shape == (max_trajectory_length, *observation_shape)
        assert self.rewards.shape == (max_trajectory_length, 1)
        assert self.terminals.shape == (max_trajectory_length, 1)

        assert self.observations.dtype == NP_FLOAT_DTYPE
        assert self.actions.dtype == NP_FLOAT_DTYPE
        assert self.next_observations.dtype == NP_FLOAT_DTYPE
        assert self.rewards.dtype == NP_FLOAT_DTYPE
        assert self.terminals.dtype == bool

    @classmethod
    def create(cls, environment_info: EnvironmentInfo, initial_observation: Union[None, np.ndarray]=None) -> "Trajectory":
        trajectory_length = environment_info.max_trajectory_length
        observation_shape = environment_info.observation_shape
        action_shape = environment_info.action_shape

        trajectory = Trajectory(
            environment_info=environment_info,
            observations=np.zeros((trajectory_length, *observation_shape), dtype=NP_FLOAT_DTYPE),
            actions=np.zeros((trajectory_length, *action_shape), dtype=NP_FLOAT_DTYPE),
            next_observations=np.zeros((trajectory_length, *observation_shape), dtype=NP_FLOAT_DTYPE),
            rewards=np.zeros((trajectory_length, 1), dtype=NP_FLOAT_DTYPE),
            terminals=np.ones((trajectory_length, 1), dtype=bool),
        )

        if initial_observation is not None:
            trajectory.observations[0] = initial_observation

        return trajectory

    def update(
        self, index: int, observation: np.ndarray, action: np.ndarray, next_observation: np.ndarray, reward: float, terminal: bool
    ) -> None:
        assert 0 <= index < self.environment_info.max_trajectory_length
        assert observation.shape == self.environment_info.observation_shape
        assert action.shape == self.environment_info.action_shape
        assert next_observation.shape == self.environment_info.observation_shape

        self.observations[index] = observation
        self.actions[index] = action
        self.next_observations[index] = next_observation
        self.rewards[index] = reward
        self.terminals[index] = terminal


@dataclass
class BatchTrajectories:
    batch_size: int
    environment_info: EnvironmentInfo
    observations: np.ndarray        # Shape: (batch_size, trajectory_length, *observation_shape)
    actions: np.ndarray             # Shape: (batch_size, trajectory_length, *action_shape)
    next_observations: np.ndarray   # Shape: (batch_size, trajectory_length, *observation_shape)
    rewards: np.ndarray             # Shape: (batch_size, trajectory_length, 1)
    terminals: np.ndarray           # Shape: (batch_size, trajectory_length, 1). 1 if terminal, 0 otherwise

    def __post_init__(self) -> None:
        batch_size = self.batch_size
        max_trajectory_length = self.environment_info.max_trajectory_length
        observation_shape = self.environment_info.observation_shape
        action_shape = self.environment_info.action_shape

        assert self.observations.shape == (batch_size, max_trajectory_length, *observation_shape)
        assert self.actions.shape == (batch_size, max_trajectory_length, *action_shape)
        assert self.next_observations.shape == (batch_size, max_trajectory_length, *observation_shape)
        assert self.rewards.shape == (batch_size, max_trajectory_length, 1)
        assert self.terminals.shape == (batch_size, max_trajectory_length, 1)

        assert self.observations.dtype == NP_FLOAT_DTYPE
        assert self.actions.dtype == NP_FLOAT_DTYPE
        assert self.next_observations.dtype == NP_FLOAT_DTYPE
        assert self.rewards.dtype == NP_FLOAT_DTYPE
        assert self.terminals.dtype == bool

    @classmethod
    def from_trajectories(cls, trajectories: List[Trajectory]) -> "BatchTrajectories":
        get_trajectory_property = lambda s: np.array([getattr(t, s) for t in trajectories])
        batch_size = len(trajectories)
        environment_info = trajectories[0].environment_info

        return BatchTrajectories(
            batch_size=batch_size,
            environment_info=environment_info,
            observations=get_trajectory_property("observations"),
            actions=get_trajectory_property("actions"),
            next_observations=get_trajectory_property("next_observations"),
            rewards=get_trajectory_property("rewards"),
            terminals=get_trajectory_property("terminals"),
        )


@dataclass
class BatchTrajectoriesPyTorch:
    batch_size: int
    environment_info: EnvironmentInfo
    observations: torch.Tensor        # Shape: (batch_size, trajectory_length, *observation_shape)
    actions: torch.Tensor             # Shape: (batch_size, trajectory_length, *action_shape)
    next_observations: torch.Tensor   # Shape: (batch_size, trajectory_length, *observation_shape)
    rewards: torch.Tensor             # Shape: (batch_size, trajectory_length, 1)
    terminals: torch.Tensor           # Shape: (batch_size, trajectory_length, 1). 1 if terminal, 0 otherwise

    def __post_init__(self) -> None:
        batch_size = self.batch_size
        max_trajectory_length = self.environment_info.max_trajectory_length
        observation_shape = self.environment_info.observation_shape
        action_shape = self.environment_info.action_shape

        assert self.observations.size() == (batch_size, max_trajectory_length, *observation_shape)
        assert self.actions.size() == (batch_size, max_trajectory_length, *action_shape)
        assert self.next_observations.size() == (batch_size, max_trajectory_length, *observation_shape)
        assert self.rewards.size() == (batch_size, max_trajectory_length, 1)
        assert self.terminals.size() == (batch_size, max_trajectory_length, 1)

        assert self.observations.dtype == TORCH_FLOAT_DTYPE
        assert self.actions.dtype == TORCH_FLOAT_DTYPE
        assert self.next_observations.dtype == TORCH_FLOAT_DTYPE
        assert self.rewards.dtype == TORCH_FLOAT_DTYPE
        assert self.terminals.dtype == torch.bool

    @classmethod
    def from_batch_trajectories(cls, batch_trajectories: BatchTrajectories, device: torch.device) -> "BatchTrajectoriesPyTorch":
        def convert_property_to_pytorch(property_name: str) -> torch.nn.Module:
            property_np = getattr(batch_trajectories, property_name)
            property_pt = torch.from_numpy(property_np).to(device)
            property_pt.requires_grad = False
            return property_pt

        return BatchTrajectoriesPyTorch(
            batch_size=batch_trajectories.batch_size,
            environment_info=batch_trajectories.environment_info,
            observations=convert_property_to_pytorch("observations"),
            actions=convert_property_to_pytorch("actions"),
            next_observations=convert_property_to_pytorch("next_observations"),
            rewards=convert_property_to_pytorch("rewards"),
            terminals=convert_property_to_pytorch("terminals"),
        )

    @classmethod
    def from_trajectory(self, trajectory: Trajectory, device: torch.device) -> "BatchTrajectoriesPyTorch":
        batch_trajectories = BatchTrajectories.from_trajectories([trajectory])
        return BatchTrajectoriesPyTorch.from_batch_trajectories(batch_trajectories, device)