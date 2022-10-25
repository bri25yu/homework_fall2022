from torch.nn import Module

from rl.infrastructure.model_output import ModelOutput
from rl.infrastructure.trajectory import Trajectory


class PolicyBase(Module):
    def forward(self, trajectory: Trajectory) -> ModelOutput:
        raise NotImplementedError
