from typing import Any, Dict, Tuple, Union

import torch.nn as nn

from gym import Env

from rl.infrastructure import EnvironmentInfo, ModelOutput, pytorch_utils
from rl.training_pipelines.base import TrainingPipelineBase


__all__ = ["OffPolicyTrainingPipelineBase"]


class OffPolicyTrainingPipelineBase(TrainingPipelineBase):
    TRAIN_BATCH_SIZE: Union[None, int] = None  # Number of trajectories worth of steps

    # This is an exact copy of the `OfflineTrainingPipelineBase.perform_single_train_step` unless specified otherwise
    def perform_single_train_step(self, env: Env, environment_info: EnvironmentInfo, policy: nn.Module) -> Tuple[ModelOutput, Dict[str, Any]]:
        batch_size = self.TRAIN_BATCH_SIZE

        ###########################
        # START sample `batch_size` trajectories and add them to replay buffer
        ###########################

        trajectories = self.record_trajectories(env, environment_info, policy, batch_size)

        ###########################
        # END sample `batch_size` trajectories and add them to replay buffer
        ###########################

        trajectories.to_device(pytorch_utils.TORCH_DEVICE)

        policy.train()
        model_output: ModelOutput = policy(trajectories)

        train_log = {
            "loss_train": pytorch_utils.to_numpy(model_output.loss),
        }

        return model_output, train_log
