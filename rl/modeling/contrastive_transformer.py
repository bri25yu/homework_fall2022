import torch
from torch.nn import Sequential, GELU, Linear

from gym import Env

from rl.infrastructure import Trajectory, ModelOutput, PolicyBase, normalize, build_log_std, build_transformer
from rl.modeling.utils import assert_shape, calculate_log_probs, calculate_q_values, calculate_contrastive_q_values_update, get_log_probs_logs


__all__ = ["ContrastiveTransformerBase"]


class ContrastiveTransformerBase(PolicyBase):
    def __init__(self, env: Env, gamma: float) -> None:
        super().__init__(env)

        self.gamma = gamma

        # Assumes 1d obs and acs
        obs_dim = env.observation_space.shape[0]
        if self.is_discrete:
            self.acs_dim = env.action_space.n
            self.log_std = None
        else:
            self.acs_dim = env.action_space.shape[0]
            self.log_std = build_log_std(env.action_space.shape)

        self.model = Sequential(
            build_transformer(model_dim=obs_dim),
            GELU(),
            Linear(obs_dim, self.acs_dim),
        )

        L = env.spec.max_episode_steps
        self.best_q_vals = torch.nn.Parameter(torch.zeros(L, 1), requires_grad=False)

    def forward(self, trajectories: Trajectory) -> ModelOutput:
        L = trajectories.L
        acs_dim = self.acs_dim

        # Unsqueeze for batch dimension and take first element of batch result
        actions_mean: torch.Tensor = self.model(trajectories.observations[None])[0]
        assert_shape(actions_mean, (L, acs_dim))
        actions_dist = self.create_actions_distribution(actions_mean, self.log_std)

        if not self.training:
            actions: torch.Tensor = actions_dist.sample()
            return ModelOutput(actions=actions, loss=None)

        action_log_probs = calculate_log_probs(actions_dist, trajectories.actions)

        q_vals = calculate_q_values(trajectories.rewards, trajectories.terminals, self.gamma)
        corresponding_best_q_vals, self.best_q_vals.data = calculate_contrastive_q_values_update(
            q_vals, self.best_q_vals, trajectories.terminals
        )

        advantages = normalize(q_vals - corresponding_best_q_vals)
        loss = (-action_log_probs.mean()) + (-action_log_probs * advantages.detach()).sum()

        logs = {
            "value_best_q_val_0": self.best_q_vals.data[0].detach(),
            **get_log_probs_logs(action_log_probs),
        }

        return ModelOutput(actions=None, loss=loss, logs=logs)
