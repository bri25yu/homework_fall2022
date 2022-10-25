from typing import Any, Dict, Tuple, Union

from abc import abstractmethod

import torch.nn as nn

from gym import Env

from rl.infrastructure import EnvironmentInfo, ModelOutput, ReplayBuffer, pytorch_utils
from rl.training_pipelines.base import TrainingPipelineBase


__all__ = ["OfflineTrainingPipelineBase"]


class OfflineTrainingPipelineBase(TrainingPipelineBase):
    TRAIN_BATCH_SIZE: Union[None, int] = None

    @abstractmethod
    def get_replay_buffer(self, environment_info: EnvironmentInfo) -> ReplayBuffer:
        pass

    def perform_single_train_step(self, env: Env, environment_info: EnvironmentInfo, policy: nn.Module) -> Tuple[ModelOutput, Dict[str, Any]]:
        batch_size = self.TRAIN_BATCH_SIZE
        if not hasattr(self, "replay_buffer"):
            self.replay_buffer = self.get_replay_buffer(environment_info)

        batch_trajectories = self.replay_buffer.sample(batch_size)
        batch_trajectories.to_device(pytorch_utils.TORCH_DEVICE)

        policy.train()
        model_output: ModelOutput = policy(batch_trajectories)

        train_log = {
            "loss_train": pytorch_utils.to_numpy(model_output.loss),
        }

        return model_output, train_log
