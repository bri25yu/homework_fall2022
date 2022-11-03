import torch
from torch.nn import Sequential, GELU, Linear

from gym import Env

from transformers import T5Config

from rl.infrastructure import Trajectory, ModelOutput, PolicyBase, normalize, build_log_std
from rl.modeling.direct_transformer import T5ForReinforcementLearning


__all__ = ["ContrastiveTransformerV2Base"]


class ContrastiveTransformerV2Base(PolicyBase):
    def __init__(self, env: Env, gamma: float) -> None:
        super().__init__(env)

        self.gamma = gamma

        # Assumes 1d obs and acs
        self.obs_dim = env.observation_space.shape[0]
        if self.is_discrete:
            self.acs_dim = env.action_space.n
            self.log_std = None
        else:
            self.acs_dim = env.action_space.shape[0]
            self.log_std = build_log_std(env.action_space.shape)

        # Create our model
        model_config = T5Config(
            d_model=self.obs_dim,
            d_kv=64,
            dff=64,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=4,
        )
        self.model = Sequential(
            T5ForReinforcementLearning(model_config),
            GELU(),
            Linear(self.obs_dim, self.acs_dim),
        )

        L = env.spec.max_episode_steps
        self.best_q_vals = torch.nn.Parameter(torch.zeros(L, 1), requires_grad=False)

    def forward(self, trajectories: Trajectory) -> ModelOutput:
        L = trajectories.L
        acs_dim = self.acs_dim
        gamma = self.gamma
        logs = dict()

        actions_mean: torch.Tensor = self.model(trajectories.observations[None])[0]  # Unsqueeze for batch dimension and take first element of batch result
        assert actions_mean.size() == (L, acs_dim)
        actions_dist = self.create_actions_distribution(actions_mean, self.log_std)

        if not self.training:
            actions: torch.Tensor = actions_dist.sample()
            return ModelOutput(actions=actions, loss=None)

        mask = ~trajectories.terminals

        # Calculate q_vals
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

        # Calculate log probs
        action_log_probs = actions_dist \
            .log_prob(trajectories.actions) \
            .view(L, -1) \
            .sum(dim=-1, keepdim=True)

        advantages = normalize(q_vals - corresponding_best_q_vals)
        loss = (-action_log_probs * advantages.detach()).sum()

        logs.update({
            "value_log_probs_mean": -action_log_probs.detach().mean(),
        })

        return ModelOutput(actions=None, loss=loss, logs=logs)
