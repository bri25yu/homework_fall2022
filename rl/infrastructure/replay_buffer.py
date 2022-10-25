from typing import List

import torch

from rl.infrastructure.environment_info import EnvironmentInfo
from rl.infrastructure.trajectory import BatchTrajectory


class ReplayBuffer:
    DEVICE = "cpu"

    def __init__(self, environment_info: EnvironmentInfo, size: int=10000) -> None:
        self.num_examples_stored = 0

        self.indices = torch.arange(size)
        self.trajectories = BatchTrajectory.create(environment_info, size, self.DEVICE)

    @property
    def size(self) -> int:
        return self.trajectories.batch_size

    def sample(self, batch_size: int) -> BatchTrajectory:
        assert batch_size <= self.num_examples_stored

        batch_indices = torch.randint(0, self.num_examples_stored, (batch_size,))

        return BatchTrajectory(
            batch_size=batch_size,
            environment_info=self.trajectories.environment_info,
            device=self.DEVICE,
            observations=torch.Tensor(self.trajectories.observations[batch_indices]),
            actions=torch.Tensor(self.trajectories.actions[batch_indices]),
            next_observations=torch.Tensor(self.trajectories.next_observations[batch_indices]),
            rewards=torch.Tensor(self.trajectories.rewards[batch_indices]),
            mask=torch.Tensor(self.trajectories.mask[batch_indices]),
        )

    def add_trajectories_to_buffer(self, trajectories: List[BatchTrajectory]) -> None:
        batch_size = sum(t.batch_size for t in trajectories)

        self.num_examples_stored = min(self.size, self.num_examples_stored + batch_size)
        self.indices = torch.roll(self.indices, batch_size)

        current_index = 0
        for trajectory in trajectories:
            next_index = current_index + trajectory.batch_size

            # These are the actual indices we use to update our array
            start, end = self.indices[current_index], self.indices[next_index]

            self.trajectories.observations[start: end] = trajectory.observations
            self.trajectories.actions[start: end] = trajectory.actions
            self.trajectories.next_observations[start: end] = trajectory.next_observations
            self.trajectories.rewards[start: end] = trajectory.rewards
            self.trajectories.mask[start: end] = trajectory.mask

            current_index = next_index
