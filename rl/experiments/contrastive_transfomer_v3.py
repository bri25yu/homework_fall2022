from gym import Env, make

from rl.modeling.contrastive_transformer_v3 import ContrastiveTransformerV3Base
from rl.training_pipelines import OffPolicyTrainingPipelineBase
from rl.infrastructure import PolicyBase


class ContrastiveTransformerV3ExperimentBase(OffPolicyTrainingPipelineBase):
    def get_policy(self, env: Env) -> PolicyBase:
        return ContrastiveTransformerV3Base(env=env, gamma=0.99)


class ContrastiveTransformerV3CartPoleExperiment(ContrastiveTransformerV3ExperimentBase):
    TRAIN_STEPS = 100
    EVAL_STEPS = 1
    LEARNING_RATE = 5e-3
    TRAIN_BATCH_SIZE = 1

    def get_env(self) -> Env:
        return make("CartPole-v1")


class ContrastiveTransformerV3InvertedPendulumExperiment(ContrastiveTransformerV3ExperimentBase):
    TRAIN_STEPS = 100
    EVAL_STEPS = 1
    LEARNING_RATE = 5e-3
    TRAIN_BATCH_SIZE = 1

    def get_env(self) -> Env:
        return make("InvertedPendulum-v4")
