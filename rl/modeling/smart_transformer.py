import torch
from torch import Tensor
from torch.nn import HuberLoss

from gym import Env

from rl.infrastructure import Trajectory, ModelOutput, PolicyBase, build_log_std, build_transformer, normalize
from rl.modeling.utils import assert_shape, calculate_log_probs, get_log_probs_logs


__all__ = ["SmartTransformerBase"]


class SmartTransformerOutput:
    def __init__(self, model_output: Tensor, obs_dim: int, acs_dim: int) -> None:
        self.next_obs: Tensor = model_output[:, : obs_dim]
        self.action_mean: Tensor = model_output[:, obs_dim: obs_dim + acs_dim]
        self.reward: Tensor = model_output[:, obs_dim + acs_dim: obs_dim + acs_dim + 1]
        self.q_values: Tensor = model_output[:, obs_dim + acs_dim + 1: obs_dim + acs_dim + 2]


class SmartTransformerBase(PolicyBase):
    """
    We have five pieces of information available to us from each trajectory
    - observations
    - actions that were taken
    - next_observations as a result of the action taken
    - rewards as a result of the action taken
    - terminals

    The total input into our model is
        observations + actions + rewards + terminals
    of dim
        obs_dim + ac_dim + 1 + 1 = obs_dim + ac_dim + 2

    We incur several losses:
    - model-based prediction error of the next observation from the current
        observation and action
    - reward prediction error from the current observation and action
    - q_values prediction error
    - improvement error using q_values

    So our model needs to predict
    - next_observation (ob_dim,)
    - action mean (ac_dim,)
    - reward (1,)
    - q_values of the current time step (1,)

    The total output dim for our transformer at each timestep should be
    ob_dim + ac_dim + 1 + 1 = ob_dim + ac_dim + 2

    """

    def __init__(self, env: Env, gamma: float) -> None:
        super().__init__(env)

        self.gamma = gamma
        self.loss_fn = HuberLoss(reduction="none")

        # Assumes 1d obs and acs
        self.obs_dim = env.observation_space.shape[0]
        if self.is_discrete:
            self.acs_dim = env.action_space.n
            self.log_std = None
        else:
            self.acs_dim = env.action_space.shape[0]
            self.log_std = build_log_std(env.action_space.shape)
        self.model_dim = self.obs_dim + self.acs_dim + 2

        self.model = build_transformer(model_dim=self.model_dim)

    def _get_model_outputs(
        self, observations: Tensor, actions: Tensor, rewards: Tensor, terminals: Tensor
    ) -> SmartTransformerOutput:
        L = observations.size()[0]
        model_dim = self.model_dim
        obs_dim = self.obs_dim
        acs_dim = self.acs_dim

        # Reshape actions for categorical features
        if self.is_discrete:
            size_to_fill = acs_dim - 1
            actions = torch.cat((
                actions.unsqueeze(1),
                torch.zeros((L, size_to_fill), dtype=actions.dtype, device=actions.device),
            ), dim=1)

        stacked_input = torch.cat((observations, actions, rewards, terminals), dim=1)
        assert_shape(stacked_input, (L, model_dim))

        # Unsqueeze for batch dimension and take first element of batch result
        model_output: Tensor = self.model(stacked_input[None])[0]

        return SmartTransformerOutput(model_output, obs_dim, acs_dim)

    def forward(self, traj: Trajectory) -> ModelOutput:
        loss_fn = self.loss_fn
        gamma = self.gamma

        preds = self._get_model_outputs(traj.observations, traj.actions, traj.rewards, traj.terminals)

        actions_dist = self.create_actions_distribution(preds.action_mean, self.log_std)

        if not self.training:
            actions: Tensor = actions_dist.sample()
            return ModelOutput(actions=actions, loss=None)

        mask = ~traj.terminals
        action_log_probs = calculate_log_probs(actions_dist, traj.actions)

        model_based_loss = (loss_fn(preds.next_obs, traj.next_observations) * mask).mean()
        reward_prediction_loss = loss_fn(preds.reward, traj.rewards).mean()

        # Q-value prediction error: Q_t = r_t + gamma * (1 - terminal_t) * Q_{t+1}
        next_q_values_prediction = preds.q_values.roll(-1)
        assert preds.q_values[1] == next_q_values_prediction[0]
        target_q_values = traj.rewards + gamma * mask * next_q_values_prediction
        q_value_prediction_loss = loss_fn(preds.q_values, target_q_values.detach()).mean()

        imagined_actions = actions_dist.rsample()
        imagined_q_values = self._get_model_outputs(traj.observations, imagined_actions, traj.rewards, traj.terminals).q_values
        improvement_loss = (-action_log_probs - imagined_q_values).mean()

        loss = model_based_loss + reward_prediction_loss + q_value_prediction_loss + improvement_loss

        logs = {
            **get_log_probs_logs(action_log_probs),
            "loss_model_based": model_based_loss,
            "loss_reward_prediction": reward_prediction_loss,
            "loss_q_value_prediction": q_value_prediction_loss,
            "loss_improvement": improvement_loss,
        }

        return ModelOutput(actions=None, loss=loss, logs=logs)
