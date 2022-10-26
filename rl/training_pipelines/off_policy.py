from typing import Any, Dict, Tuple, Union

import torch.nn as nn

from gym import Env

from rl.infrastructure import ModelOutput, to_numpy
from rl.training_pipelines.base import TrainingPipelineBase


__all__ = ["OffPolicyTrainingPipelineBase"]


class OffPolicyTrainingPipelineBase(TrainingPipelineBase):
    TRAIN_BATCH_SIZE: Union[None, int] = None  # Number of trajectories worth of steps

    def perform_single_train_step(self, env: Env, policy: nn.Module) -> Tuple[ModelOutput, Dict[str, Any]]:
        batch_size = self.TRAIN_BATCH_SIZE

        trajectories = self.record_trajectories(env, policy, batch_size)

        policy.train()
        model_output: ModelOutput = policy(trajectories)

        train_log = {
            "loss_train": to_numpy(model_output.loss),
        }

        return model_output, train_log
