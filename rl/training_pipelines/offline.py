from typing import Any, Dict, Tuple, Union

from abc import abstractmethod

import torch.nn as nn

from gym import Env

from rl.infrastructure import EnvironmentInfo, ModelOutput, ReplayBuffer, BatchTrajectoriesPyTorch
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
        batch_trajectories = BatchTrajectoriesPyTorch.from_batch_trajectories(batch_trajectories)

        model_output: ModelOutput = policy(batch_trajectories)

        train_log = {
            "train_loss": model_output.loss.item(),
        }

        return model_output, train_log