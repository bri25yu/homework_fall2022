from torch.nn import Module

from rl.infrastructure.model_output import ModelOutput
from rl.infrastructure.trajectory import BatchTrajectoriesPyTorch


class PolicyBase(Module):
    def forward(self, trajectories: BatchTrajectoriesPyTorch) -> ModelOutput:
        raise NotImplementedError
