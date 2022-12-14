from typing import Dict, Union
from torch.optim import AdamW, Optimizer

from rl.infrastructure import PolicyBase
from rl.training_pipelines.off_policy import OffPolicyTrainingPipelineBase


__all__ = ["OffPolicyWithOptimizerParamsTrainingPipelineBase"]


class OffPolicyWithOptimizerParamsTrainingPipelineBase(OffPolicyTrainingPipelineBase):
    OPTIMIZER_KWARGS: Union[None, Dict[str, Union[str, float]]] = None

    def setup_optimizer(self, policy: PolicyBase) -> Optimizer:
        learning_rate = self.LEARNING_RATE
        optimizer_kwargs = self.OPTIMIZER_KWARGS

        return AdamW(policy.parameters(), lr=learning_rate, **optimizer_kwargs)
