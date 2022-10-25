from typing import Any, Dict, Tuple, Union

import torch.nn as nn

from gym import Env

from rl.infrastructure import EnvironmentInfo, ModelOutput, ReplayBuffer, pytorch_utils
from rl.training_pipelines.base import TrainingPipelineBase


__all__ = ["OffPolicyTrainingPipelineBase"]


class OffPolicyTrainingPipelineBase(TrainingPipelineBase):
    TRAIN_BATCH_SIZE: Union[None, int] = None

    # This is an exact copy of the `OfflineTrainingPipelineBase.perform_single_train_step` unless specified otherwise
    def perform_single_train_step(self, env: Env, environment_info: EnvironmentInfo, policy: nn.Module) -> Tuple[ModelOutput, Dict[str, Any]]:
        batch_size = self.TRAIN_BATCH_SIZE
        if not hasattr(self, "replay_buffer"):
            ###########################
            # START start with empty replay buffer
            ###########################

            # Original code:
            # self.replay_buffer = self.get_replay_buffer()

            self.replay_buffer = ReplayBuffer(environment_info)

            ###########################
            # END start with empty replay buffer
            ###########################

        ###########################
        # START sample `batch_size` trajectories and add them to replay buffer
        ###########################

        trajectories_to_add = [self.sample_single_trajectory(env, environment_info, policy) for _ in range(batch_size)]
        self.replay_buffer.add_trajectories_to_buffer(trajectories_to_add)

        ###########################
        # END sample `batch_size` trajectories and add them to replay buffer
        ###########################

        batch_trajectories = self.replay_buffer.sample(batch_size)
        batch_trajectories.to_device(pytorch_utils.TORCH_DEVICE)

        policy.train()
        model_output: ModelOutput = policy(batch_trajectories)

        train_log = {
            "train_loss": pytorch_utils.to_numpy(model_output.loss),
        }

        return model_output, train_log
