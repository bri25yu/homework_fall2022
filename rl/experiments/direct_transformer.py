from gym import Env, make

from rl.modeling.direct_transformer import DirectTransformerBase
from rl.training_pipelines import OffPolicyTrainingPipelineBase
from rl.infrastructure import PolicyBase


class DirectTransformerExperimentBase(OffPolicyTrainingPipelineBase):
    def get_policy(self, env: Env) -> PolicyBase:
        return DirectTransformerBase(env=env, gamma=0.99)


class DirectTransformerCartPoleExperiment(DirectTransformerExperimentBase):
    TRAIN_STEPS = 100
    EVAL_STEPS = 1
    LEARNING_RATE = 5e-3
    TRAIN_BATCH_SIZE = 1

    def get_env(self) -> Env:
        return make("CartPole-v1")


class DirectTransformerInvertedPendulumExperiment(DirectTransformerExperimentBase):
    TRAIN_STEPS = 100
    EVAL_STEPS = 1
    LEARNING_RATE = 5e-3
    TRAIN_BATCH_SIZE = 1

    def get_env(self) -> Env:
        return make("InvertedPendulum-v4")
