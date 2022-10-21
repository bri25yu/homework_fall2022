from typing import List

import numpy as np

from rl.infrastructure.environment_info import EnvironmentInfo
from rl.infrastructure.trajectory import Trajectory, BatchTrajectories


class ReplayBuffer:
    def __init__(self, environment_info: EnvironmentInfo, size: int=10000) -> None:
        self.environment_info = environment_info
        self.size = size
        self.num_examples_stored = 0

        trajectory_length = environment_info.max_trajectory_length
        observation_shape = environment_info.observation_shape
        action_shape = environment_info.action_shape

        self.indices = np.arange(size)
        # We can use np.empty here without worrying about accessing empty data
        # due to the specifics of our sampling alg
        self.observations = np.empty((size, trajectory_length, *observation_shape), dtype=np.float16)
        self.actions = np.empty((size, trajectory_length, *action_shape), dtype=np.float16)
        self.next_observations = np.empty((size, trajectory_length, *observation_shape), dtype=np.float16)
        self.rewards = np.empty((size, trajectory_length, 1), dtype=np.float16)
        self.terminals = np.ones((size, trajectory_length, 1), dtype=bool)

    def sample(self, batch_size: int) -> BatchTrajectories:
        assert batch_size <= self.num_examples_stored

        batch_indices = np.random.randint(0, self.num_examples_stored, batch_size)

        return BatchTrajectories(
            batch_size=batch_size,
            environment_info=self.environment_info,
            observations=np.array(self.observations.take(batch_indices)),
            actions=np.array(self.actions.take(batch_indices)),
            next_observations=np.array(self.next_observations.take(batch_indices)),
            rewards=np.array(self.rewards.take(batch_indices)),
            terminals=np.array(self.terminals.take(batch_indices)),
        )

    def add_trajectories_to_buffer(self, trajectories: List[Trajectory]) -> None:
        batch_size = len(trajectories)

        self.num_examples_stored = min(self.size, self.num_examples_stored + batch_size)
        self.indices = np.roll(self.indices, batch_size)

        batch_trajectories = BatchTrajectories.from_trajectories(trajectories)

        indices_to_update = self.indices[:batch_size]

        self.observations[indices_to_update] = batch_trajectories.observations
        self.actions[indices_to_update] = batch_trajectories.actions
        self.next_observations[indices_to_update] = batch_trajectories.next_observations
        self.rewards[indices_to_update] = batch_trajectories.rewards
        self.terminals[indices_to_update] = batch_trajectories.terminals
