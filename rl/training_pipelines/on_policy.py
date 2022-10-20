from typing import Any, Dict, Tuple

import torch.nn as nn

from gym import Env

from rl.infrastructure import EnvironmentInfo, ModelOutput
from rl.training_pipelines.base import TrainingPipelineBase


__all__ = ["OnPolicyTrainingPipelineBase"]


class OnPolicyTrainingPipelineBase(TrainingPipelineBase):
    def perform_single_train_step(self, env: Env, environment_info: EnvironmentInfo, policy: nn.Module) -> Tuple[ModelOutput, Dict[str, Any]]:
        raise NotImplementedError
