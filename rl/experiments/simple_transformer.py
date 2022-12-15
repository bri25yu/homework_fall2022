from gym import Env

from rl.modeling.simple_transformer import SimpleTransformerModel
from rl.infrastructure import PolicyBase
from rl.experiments.base import ExperimentBase


class SimpleTransformerExperimentBase(ExperimentBase):
    def get_policy(self, env: Env) -> PolicyBase:
        return SimpleTransformerModel(env=env, gamma=0.99)


class SimpleTransformerCartPoleExperiment(SimpleTransformerExperimentBase):
    TRAIN_STEPS = 100
    LEARNING_RATE = 1e-2
    ENV_NAME = "CartPole-v1"


class SimpleTransformerHalfCheetahExperiment(SimpleTransformerExperimentBase):
    TRAIN_STEPS = 1000
    SEEDS = [42]
    LEARNING_RATE = 1e-2
    ENV_NAME = "HalfCheetah-v4"
