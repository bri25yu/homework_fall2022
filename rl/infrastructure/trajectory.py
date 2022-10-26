from typing import Union

import numpy as np

import torch

from rl.infrastructure.pytorch_utils import TORCH_FLOAT_DTYPE, TORCH_DEVICE


__all__ = ["Trajectory"]


class Trajectory:
    def __init__(self, L: int) -> None:
        self.L = L

        self.observations: Union[None, torch.Tensor] = None
        self.actions: Union[None, torch.Tensor] = None
        self.next_observations: Union[None, torch.Tensor] = None
        self.rewards: Union[None, torch.Tensor] = None
        self.terminals: Union[None, torch.Tensor] = None

    def update_observations_from_numpy(self, index: int, observation: np.ndarray) -> None:
        if self.observations is None:
            self.observations = self.batch_pt_from_np(observation)

        self.observations[index] = torch.from_numpy(observation)

    def update_consequences_from_numpy(
        self, index: int, action: np.ndarray, next_observation: np.ndarray, reward: float, terminal: bool
    ) -> None:
        if self.actions is None:  # Initialize block data arrays to improve performance using torch
            self.actions = self.batch_pt_from_np(action)
            self.next_observations = self.batch_pt_from_np(next_observation)
            self.rewards = self.batch_pt_from_np(reward)
            self.terminals = self.batch_pt_from_np(terminal)

        next_observation_pt = torch.from_numpy(next_observation)
        action_pt = torch.from_numpy(action)

        self.actions[index] = action_pt
        self.next_observations[index] = next_observation_pt
        self.rewards[index] = reward
        self.terminals[index] = terminal

        if index + 1 < self.L:
            self.observations[index+1] = next_observation_pt

    def batch_pt_from_np(self, arr_np: Union[np.ndarray, float, bool]) -> torch.Tensor:
        if isinstance(arr_np, bool):
            return torch.zeros((self.L, 1), dtype=torch.bool, device=TORCH_DEVICE)
        elif isinstance(arr_np, float):
            return torch.zeros((self.L, 1), dtype=TORCH_FLOAT_DTYPE, device=TORCH_DEVICE)
        elif isinstance(arr_np, np.ndarray):
            return torch.zeros((self.L, *arr_np.shape), dtype=TORCH_FLOAT_DTYPE, device=TORCH_DEVICE)
        else:
            raise ValueError(f"Unrecognized input array type {type(arr_np)}")
