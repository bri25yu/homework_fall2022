from gym import Env, make

from rl.modeling.simple_transformer import SimpleTransformerModel
from rl.training_pipelines import OffPolicyTrainingPipelineBase
from rl.infrastructure import PolicyBase


class SimpleTransformerExperimentBase(OffPolicyTrainingPipelineBase):
    def get_policy(self, env: Env) -> PolicyBase:
        return SimpleTransformerModel(env=env, gamma=0.99)


class SimpleTransformerInvertedPendulumExperiment(SimpleTransformerExperimentBase):
    TRAIN_STEPS = 100
    EVAL_STEPS = 1
    LEARNING_RATE = 1e-2
    TRAIN_BATCH_SIZE = 1

    def get_env(self) -> Env:
        return make("InvertedPendulum-v4")


class SimpleTransformerCartPoleExperiment(SimpleTransformerExperimentBase):
    TRAIN_STEPS = 100
    EVAL_STEPS = 1
    LEARNING_RATE = 1e-2
    TRAIN_BATCH_SIZE = 1

    def get_env(self) -> Env:
        return make("CartPole-v1")
