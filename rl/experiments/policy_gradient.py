from gym import Env

from rl.modeling.policy_gradient import PolicyGradientBase
from rl.infrastructure import PolicyBase
from rl.experiments.base import ExperimentBase


class PolicyGradientExperimentBase(ExperimentBase):
    def get_policy(self, env: Env) -> PolicyBase:
        return PolicyGradientBase(env=env, gamma=0.99)


class PolicyGradientCartPoleExperiment(PolicyGradientExperimentBase):
    TRAIN_STEPS = 100
    LEARNING_RATE = 5e-3
    ENV_NAME = "CartPole-v1"


class PolicyGradientHalfCheetahExperiment(PolicyGradientExperimentBase):
    TRAIN_STEPS = 1000
    SEEDS = [42]
    LEARNING_RATE = 1e-2
    ENV_NAME = "HalfCheetah-v4"
