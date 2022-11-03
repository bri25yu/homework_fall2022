import torch

from gym import Env

from rl.infrastructure import Trajectory, ModelOutput, normalize
from rl.modeling.direct_transformer import DirectTransformerBase


__all__ = ["ContrastiveTransformerBase"]


class ContrastiveTransformerBase(DirectTransformerBase):
    def __init__(self, env: Env, gamma: float) -> None:
        super().__init__(env, gamma)

        L = env.spec.max_episode_steps
        self.best_q_vals = torch.nn.Parameter(torch.zeros(L, 1), requires_grad=False)

    # This is an exact copy of `DirectTransformerBase.forward` unless specified otherwise
    def forward(self, trajectories: Trajectory) -> ModelOutput:
        L = trajectories.L
        model_dim = self.model_dim
        obs_dim = self.obs_dim
        acs_dim = self.acs_dim
        loss_fn = self.loss_fn
        gamma = self.gamma
        logs = dict()

        # Reshape actions for categorical features
        actions = trajectories.actions
        if self.is_discrete:
            size_to_fill = acs_dim - 1
            actions = torch.cat((
                actions.unsqueeze(1),
                torch.zeros((L, size_to_fill), dtype=actions.dtype, device=actions.device),
            ), dim=1)

        stacked_input = torch.cat((
            trajectories.observations,
            actions,
            trajectories.rewards,
            trajectories.terminals,
        ), dim=1)
        assert stacked_input.size() == (L, model_dim)

        model_output = self.model(stacked_input[None])[0]  # Unsqueeze for batch dimension and take first element of batch result
        assert model_output.size() == (L, model_dim)

        actions_mean: torch.Tensor = model_output[:, :acs_dim]
        actions_dist = self.create_actions_distribution(actions_mean, self.log_std)

        if not self.training:
            actions: torch.Tensor = actions_dist.sample()
            return ModelOutput(actions=actions, loss=None)

        mask = ~trajectories.terminals

        # Calculate log probs
        action_log_probs = actions_dist \
            .log_prob(trajectories.actions) \
            .view(L, -1) \
            .sum(dim=-1, keepdim=True)

        ###############################
        # START Calculate q_vals
        ###############################

        q_vals = trajectories.rewards.clone().detach()
        for i in torch.arange(L-2, -1, -1):
            q_vals[i] += gamma * mask[i] * q_vals[i+1]

        corresponding_best_q_vals = torch.empty_like(q_vals)
        new_best_q_vals = self.best_q_vals.data.clone().detach()
        corresponding_index = 0
        for i in torch.arange(L):
            corresponding_best_q_vals[i] = self.best_q_vals[corresponding_index]
            corresponding_index = (corresponding_index + 1) * mask[i]
            new_best_q_vals[corresponding_index] = max(new_best_q_vals[corresponding_index], q_vals[i])

        # Retrieve our q-value prediction
        q_value_prediction: torch.Tensor = model_output[:, acs_dim + obs_dim + 1: acs_dim + obs_dim + 2]
        assert q_value_prediction.size() == (L, 1)

        ###############################
        # END Calculate q_vals
        ###############################

        # Model-based prediction error of the next observation from the current observation and action
        next_observation_prediction: torch.Tensor = model_output[:, acs_dim: acs_dim + obs_dim]
        assert next_observation_prediction.size() == (L, obs_dim)
        model_based_loss_by_timestep = loss_fn(next_observation_prediction, trajectories.next_observations)
        model_based_loss = (model_based_loss_by_timestep * mask).mean()

        # Prediction error of the reward from the current observation and action
        reward_prediction: torch.Tensor = model_output[:, acs_dim + obs_dim: acs_dim + obs_dim + 1]
        assert reward_prediction.size() == (L, 1)
        reward_prediction_loss = loss_fn(reward_prediction, trajectories.rewards).mean()

        ###############################
        # START q-value prediction loss uses calculated q_values
        ###############################

        # Q-value prediction error
        q_value_prediction_loss = loss_fn(normalize(q_value_prediction), normalize(q_vals)).mean()

        ###############################
        # END q-value prediction loss uses calculated q_values
        ###############################

        ###############################
        # START improvement loss uses advantages
        ###############################

        # Improvement loss: -log_pi * A
        advantages = normalize(q_vals - corresponding_best_q_vals)
        improvement_loss = (-action_log_probs * advantages.detach()).sum()

        ###############################
        # END improvement loss uses advantages
        ###############################

        total_loss = model_based_loss + reward_prediction_loss + q_value_prediction_loss + improvement_loss

        logs.update({
            "loss_model_based": model_based_loss,
            "loss_reward_prediction": reward_prediction_loss,
            "loss_q_value_prediction": q_value_prediction_loss,
            "loss_improvement": improvement_loss,
            "value_log_probs_mean": -action_log_probs.detach().mean(),
        })

        return ModelOutput(actions=None, loss=total_loss, logs=logs)
