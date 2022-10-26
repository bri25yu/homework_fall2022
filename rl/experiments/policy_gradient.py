from typing import Tuple

from gym import Env, make

from rl.modeling.policy_gradient import PolicyGradientBase
from rl.training_pipelines import OffPolicyTrainingPipelineBase
from rl.infrastructure import PolicyBase, EnvironmentInfo


class PolicyGradientExperimentBase(OffPolicyTrainingPipelineBase):
    def get_policy(self, environment_info: EnvironmentInfo) -> PolicyBase:
        return PolicyGradientBase(environment_info=environment_info, gamma=0.99)


class PolicyGradientInvertedPendulumExperiment(PolicyGradientExperimentBase):
    TRAIN_STEPS = 100
    EVAL_STEPS = 1
    LEARNING_RATE = 1e-4
    TRAIN_BATCH_SIZE = 1

    def get_env(self) -> Tuple[Env, EnvironmentInfo]:
        env = make("InvertedPendulum-v4")
        environment_info = EnvironmentInfo(
            observation_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            max_episode_steps=env.spec.max_episode_steps,
        )

        return env, environment_info
