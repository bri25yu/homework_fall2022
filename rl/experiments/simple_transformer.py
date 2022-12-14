from gym import Env

from rl.modeling.simple_transformer import SimpleTransformerModel
from rl.infrastructure import PolicyBase
from rl.experiments.base import ExperimentBase


class SimpleTransformerExperimentBase(ExperimentBase):
    def get_policy(self, env: Env) -> PolicyBase:
        return SimpleTransformerModel(env=env, gamma=0.99)


class SimpleTransformerInvertedPendulumExperiment(SimpleTransformerExperimentBase):
    LEARNING_RATE = 1e-2
    ENV_NAME = "InvertedPendulum-v4"


class SimpleTransformerCartPoleExperiment(SimpleTransformerExperimentBase):
    LEARNING_RATE = 1e-2
    ENV_NAME = "CartPole-v1"


class SimpleTransformerLunarLanderExperiment(SimpleTransformerExperimentBase):
    LEARNING_RATE = 1e-2
    ENV_NAME = "LunarLander-v2"


class SimpleTransformerHalfCheetahExperiment(SimpleTransformerExperimentBase):
    LEARNING_RATE = 1e-2
    ENV_NAME = "HalfCheetah-v4"
