import torch

from rl.infrastructure.environment_info import EnvironmentInfo
from rl.infrastructure.trajectory import Trajectory


class ReplayBuffer:
    DEVICE = "cpu"

    def __init__(self, environment_info: EnvironmentInfo, size: int=int(1e6)) -> None:
        self.size = size
        self.num_examples_stored = 0

        self.indices = torch.arange(size)
        self.trajectories = Trajectory.create(environment_info, self.DEVICE, size)

    def sample(self, batch_size: int) -> Trajectory:
        assert batch_size <= self.num_examples_stored

        if batch_size == self.num_examples_stored:
            start = 0
        else:
            start = torch.randint(self.num_examples_stored - batch_size, (1,)).item()

        batch_indices = self.indices[torch.arange(start, start + batch_size)]
        return self.trajectories.take(batch_indices)

    def add_trajectories_to_buffer(self, trajectories: Trajectory) -> None:
        self.num_examples_stored = min(self.size, self.num_examples_stored + trajectories.L)
        self.indices = torch.roll(self.indices, trajectories.L)

        indices_to_overwrite = self.indices[:trajectories.L]
        self.trajectories.overwrite_indices(indices_to_overwrite, trajectories)
