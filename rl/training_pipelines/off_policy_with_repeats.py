from typing import Any, Dict, Tuple, Union

import torch.nn as nn

from gym import Env

from rl.infrastructure import ModelOutput, to_numpy
from rl.training_pipelines.off_policy import OffPolicyTrainingPipelineBase


__all__ = ["OffPolicyWithRepeatsTrainingPipelineBase"]


class OffPolicyWithRepeatsTrainingPipelineBase(OffPolicyTrainingPipelineBase):
    # Number of training steps for each time the env is sampled and trajectories are recorded
    NUM_STEPS_PER_SAMPLE: Union[None, int] = None

    def perform_single_train_step(self, env: Env, policy: nn.Module) -> Tuple[ModelOutput, Dict[str, Any]]:
        batch_size = self.TRAIN_BATCH_SIZE
        num_steps_per_sample = self.NUM_STEPS_PER_SAMPLE

        if not hasattr(self, "step_number"):
            self.step_number = 0
            self.current_sample = None

        if self.step_number % num_steps_per_sample == 0:
            self.current_sample = self.record_trajectories(env, policy, batch_size)

        policy.train()
        model_output: ModelOutput = policy(self.current_sample)

        self.step_number += 1

        train_log = {
            "loss_train": to_numpy(model_output.loss),
        }

        return model_output, train_log
