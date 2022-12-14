from gym import Env

from rl.modeling.min_ensemble_baseline import MinEnsembleBaselineModel
from rl.infrastructure import PolicyBase
from rl.experiments.base import ExperimentBase


class MinEnsembleBaselineExperimentBase(ExperimentBase):
    def get_policy(self, env: Env) -> PolicyBase:
        return MinEnsembleBaselineModel(env=env, gamma=0.99)


class MinEnsembleBaselineCartPoleExperiment(MinEnsembleBaselineExperimentBase):
    LEARNING_RATE = 1e-2
    ENV_NAME = "CartPole-v1"


class MinEnsembleBaselineHalfCheetahExperiment(MinEnsembleBaselineExperimentBase):
    LEARNING_RATE = 1e-2
    ENV_NAME = "HalfCheetah-v4"
