from gym import Env

from rl.modeling.policy_gradient import PolicyGradientBase
from rl.infrastructure import PolicyBase
from rl.experiments.base import ExperimentBase


class PolicyGradientExperimentBase(ExperimentBase):
    def get_policy(self, env: Env) -> PolicyBase:
        return PolicyGradientBase(env=env, gamma=0.99)


class PolicyGradientInvertedPendulumExperiment(PolicyGradientExperimentBase):
    LEARNING_RATE = 5e-3
    ENV_NAME = "InvertedPendulum-v4"


class PolicyGradientCartPoleExperiment(PolicyGradientExperimentBase):
    LEARNING_RATE = 5e-3
    ENV_NAME = "CartPole-v1"


class PolicyGradientLunarLanderExperiment(PolicyGradientExperimentBase):
    LEARNING_RATE = 5e-3
    ENV_NAME = "LunarLander-v2"


class PolicyGradientHalfCheetahExperiment(PolicyGradientExperimentBase):
    LEARNING_RATE = 5e-3
    ENV_NAME = "HalfCheetah-v4"
