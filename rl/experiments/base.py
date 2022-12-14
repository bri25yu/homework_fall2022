from typing import Union

from gym import Env, make

from rl.training_pipelines import OffPolicyTrainingPipelineBase


__all__ = ["ExperimentBase"]


class ExperimentBase(OffPolicyTrainingPipelineBase):
    TRAIN_STEPS = 100
    EVAL_STEPS = 1
    TRAIN_BATCH_SIZE = 1
    ENV_NAME: Union[None, str] = None

    def get_env(self) -> Env:
        return make(self.ENV_NAME)
