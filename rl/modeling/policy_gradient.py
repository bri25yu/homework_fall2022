import torch

from gym import Env

from rl.infrastructure import (
    Trajectory, ModelOutput, PolicyBase, build_ffn, build_log_std, FFNConfig, to_numpy
)


__all__ = ["PolicyGradientBase"]


class PolicyGradientBase(PolicyBase):
    def __init__(self, env: Env, gamma: float) -> None:
        super().__init__(env)

        self.gamma = gamma

        if self.is_discrete:
            self.mean_net = build_ffn(FFNConfig(
                in_shape=env.observation_space.shape,
                out_shape=(env.action_space.n,),
            ))
            self.log_std = None
        else:
            self.mean_net = build_ffn(FFNConfig(
                in_shape=env.observation_space.shape,
                out_shape=env.action_space.shape,
            ))
            self.log_std = build_log_std(env.action_space.shape)

        self.baseline = build_ffn(FFNConfig(
            in_shape=env.observation_space.shape,
            out_shape=(1,),
        ))
        self.baseline_loss_fn = torch.nn.HuberLoss()

    def forward(self, trajectories: Trajectory) -> ModelOutput:
        L = trajectories.L
        gamma = self.gamma

        actions_mean: torch.Tensor = self.mean_net(trajectories.observations)
        actions_dist = self.create_actions_distribution(actions_mean, self.log_std)

        if not self.training:
            actions: torch.Tensor = actions_dist.sample()
            return ModelOutput(actions=actions, loss=None)

        action_log_probs = actions_dist \
            .log_prob(trajectories.actions) \
            .view(L, -1) \
            .sum(dim=-1, keepdim=True)

        mask = ~trajectories.terminals
        q_vals = trajectories.rewards.clone().detach()
        for i in torch.arange(mask.size()[0]-2, -1, -1):
            q_vals[i] += gamma * mask[i] * q_vals[i+1]

        q_values_normed = self._normalize(q_vals)

        values = self.baseline(trajectories.observations)
        values_to_q_statistics = values * q_vals.std() + q_vals.mean()

        advantages = self._normalize(q_vals - values_to_q_statistics)

        policy_loss_per_sample = -action_log_probs * (advantages.detach())
        policy_loss = policy_loss_per_sample.sum()
        baseline_loss = self.baseline_loss_fn(values, q_values_normed)

        total_loss = policy_loss + baseline_loss

        def check_shapes():
            assert action_log_probs.size() == (L, 1)

            assert q_vals.size() == (L, 1)
            assert values.size() == (L, 1)
            assert advantages.size() == (L, 1)

            assert policy_loss_per_sample.size() == (L, 1)

        check_shapes()

        logs = {
            "loss_policy": to_numpy(policy_loss),
            "loss_baseline": to_numpy(baseline_loss),
        }

        return ModelOutput(actions=None, loss=total_loss, logs=logs)

    def _normalize(self, t: torch.Tensor) -> torch.Tensor:
        return (t - t.mean()) / (t.std() + 1e-8)
