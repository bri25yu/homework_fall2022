from gym import Env, make

from rl.modeling.contrastive_v2 import ContrastiveV2Base
from rl.training_pipelines import OffPolicyWithRepeatsTrainingPipelineBase
from rl.infrastructure import PolicyBase


class ContrastiveV2WithRepeatsExperimentBase(OffPolicyWithRepeatsTrainingPipelineBase):
    NUM_STEPS_PER_SAMPLE = 2

    def get_policy(self, env: Env) -> PolicyBase:
        return ContrastiveV2Base(env=env, gamma=0.99)


class ContrastiveV2WithRepeatsCartPoleExperiment(ContrastiveV2WithRepeatsExperimentBase):
    TRAIN_STEPS = 100
    EVAL_STEPS = 1
    LEARNING_RATE = 5e-3
    TRAIN_BATCH_SIZE = 1

    def get_env(self) -> Env:
        return make("CartPole-v1")


class ContrastiveV2WithRepeatsInvertedPendulumExperiment(ContrastiveV2WithRepeatsExperimentBase):
    TRAIN_STEPS = 100
    EVAL_STEPS = 1
    LEARNING_RATE = 5e-3
    TRAIN_BATCH_SIZE = 1

    def get_env(self) -> Env:
        return make("InvertedPendulum-v4")
