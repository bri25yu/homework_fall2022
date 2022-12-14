from gym import Env, make

from rl.modeling.min_ensemble_baseline import MinEnsembleBaselineModel
from rl.infrastructure import PolicyBase
from rl.training_pipelines import OffPolicyWithOptimizerParamsTrainingPipelineBase


__all__ = ["OptimizationExperimentBase"]


class OptimizationExperimentBase(OffPolicyWithOptimizerParamsTrainingPipelineBase):
    TRAIN_STEPS = 100
    EVAL_STEPS = 1
    TRAIN_BATCH_SIZE = 1
    LEARNING_RATE = 1e-2

    def get_env(self) -> Env:
        return make("HalfCheetah-v4")

    def get_policy(self, env: Env) -> PolicyBase:
        return MinEnsembleBaselineModel(env=env, gamma=0.99)
