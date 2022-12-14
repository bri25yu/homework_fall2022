from gym import Env

from rl.modeling.policy_gradient import PolicyGradientBase
from rl.infrastructure import PolicyBase
from rl.experiments.base import ExperimentBase


class PolicyGradientExperimentBase(ExperimentBase):
    def get_policy(self, env: Env) -> PolicyBase:
        return PolicyGradientBase(env=env, gamma=0.99)


class PolicyGradientCartPoleExperiment(PolicyGradientExperimentBase):
    LEARNING_RATE = 5e-3
    ENV_NAME = "CartPole-v1"


class PolicyGradientHalfCheetahExperiment(PolicyGradientExperimentBase):
    LEARNING_RATE = 5e-3
    ENV_NAME = "HalfCheetah-v4"
