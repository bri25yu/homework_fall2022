from typing import Tuple

from gym import Env, make

from rl.modeling.policy_gradient import PolicyGradientBase
from rl.training_pipelines import OffPolicyTrainingPipelineBase
from rl.infrastructure import PolicyBase, EnvironmentInfo


class PolicyGradientExperimentBase(OffPolicyTrainingPipelineBase):
    def get_policy(self, environment_info: EnvironmentInfo) -> PolicyBase:
        return PolicyGradientBase(environment_info=environment_info, gamma=0.99)


class PolicyGradientInvertedPendulumExperimentBase(PolicyGradientExperimentBase):
    def get_env(self) -> Tuple[Env, EnvironmentInfo]:
        env = make("InvertedPendulum-v4")
        environment_info = EnvironmentInfo(
            max_trajectory_length=1000,
            observation_shape=(4,),
            action_shape=(1,),
        )

        return env, environment_info


class PolicyGradientInvertedPendulumExperiment(PolicyGradientInvertedPendulumExperimentBase):
    TRAIN_STEPS = 100
    EVAL_STEPS = 1
    LEARNING_RATE = 3e-4
    TRAIN_BATCH_SIZE = 1000


class PolicyGradientInvertedPendulumTestExperiment(PolicyGradientInvertedPendulumExperimentBase):
    TRAIN_STEPS = 100
    EVAL_STEPS = 1
    LEARNING_RATE = 3e-4
    TRAIN_BATCH_SIZE = 1
