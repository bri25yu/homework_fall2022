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

        replay_buffer_index = 0  # Our location in indices
        finished_adding = False
        for trajectory in trajectories:
            if finished_adding:
                break

            for trajectory_batch_idx in range(trajectory.batch_size):
                if finished_adding:
                    break

                index_to_update = self.indices[replay_buffer_index]

                self.trajectories.observations[index_to_update] = trajectory.observations[trajectory_batch_idx]
                self.trajectories.actions[index_to_update] = trajectory.actions[trajectory_batch_idx]
                self.trajectories.next_observations[index_to_update] = trajectory.next_observations[trajectory_batch_idx]
                self.trajectories.rewards[index_to_update] = trajectory.rewards[trajectory_batch_idx]
                self.trajectories.mask[index_to_update] = trajectory.mask[trajectory_batch_idx]

                replay_buffer_index += 1
                finished_adding = replay_buffer_index >= self.size
