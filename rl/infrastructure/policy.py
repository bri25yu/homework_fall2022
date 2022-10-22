from torch.nn import Module

from rl.infrastructure.model_output import ModelOutput
from rl.infrastructure.trajectory import BatchTrajectory


class PolicyBase(Module):
    def forward(self, trajectories: BatchTrajectory) -> ModelOutput:
        raise NotImplementedError
