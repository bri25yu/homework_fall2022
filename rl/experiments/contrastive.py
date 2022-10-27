from gym import Env, make

from rl.modeling.contrastive import ContrastiveBase
from rl.training_pipelines import OffPolicyTrainingPipelineBase
from rl.infrastructure import PolicyBase


class ContrastiveExperimentBase(OffPolicyTrainingPipelineBase):
    def get_policy(self, env: Env) -> PolicyBase:
        return ContrastiveBase(env=env, gamma=0.99)


class ContrastiveCartPoleExperiment(ContrastiveExperimentBase):
    TRAIN_STEPS = 100
    EVAL_STEPS = 1
    LEARNING_RATE = 5e-3
    TRAIN_BATCH_SIZE = 1

    def get_env(self) -> Env:
        return make("CartPole-v1")


class ContrastiveInvertedPendulumExperiment(ContrastiveExperimentBase):
    TRAIN_STEPS = 100
    EVAL_STEPS = 1
    LEARNING_RATE = 5e-3
    TRAIN_BATCH_SIZE = 1

    def get_env(self) -> Env:
        return make("InvertedPendulum-v4")
