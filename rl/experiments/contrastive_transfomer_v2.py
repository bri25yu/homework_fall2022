from gym import Env, make

from rl.modeling.contrastive_transformer_v2 import ContrastiveTransformerV2Base
from rl.training_pipelines import OffPolicyTrainingPipelineBase
from rl.infrastructure import PolicyBase


class ContrastiveTransformerV2ExperimentBase(OffPolicyTrainingPipelineBase):
    def get_policy(self, env: Env) -> PolicyBase:
        return ContrastiveTransformerV2Base(env=env, gamma=0.99)


class ContrastiveTransformerV2CartPoleExperiment(ContrastiveTransformerV2ExperimentBase):
    TRAIN_STEPS = 100
    EVAL_STEPS = 1
    LEARNING_RATE = 5e-3
    TRAIN_BATCH_SIZE = 1

    def get_env(self) -> Env:
        return make("CartPole-v1")


class ContrastiveTransformerV2InvertedPendulumExperiment(ContrastiveTransformerV2ExperimentBase):
    TRAIN_STEPS = 100
    EVAL_STEPS = 1
    LEARNING_RATE = 5e-3
    TRAIN_BATCH_SIZE = 1

    def get_env(self) -> Env:
        return make("InvertedPendulum-v4")
