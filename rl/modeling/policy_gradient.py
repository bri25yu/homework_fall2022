import torch

from rl.infrastructure import Trajectory, EnvironmentInfo, ModelOutput, PolicyBase, pytorch_utils


__all__ = ["PolicyGradientBase"]


class PolicyGradientBase(PolicyBase):
    def __init__(self, environment_info: EnvironmentInfo, gamma: float) -> None:
        super().__init__(environment_info)

        self.gamma = gamma

        self.mean_net = pytorch_utils.build_ffn(pytorch_utils.FFNConfig(
            in_shape=environment_info.observation_shape,
            out_shape=environment_info.action_shape,
        ))
        if environment_info.is_discrete:
            self.log_std = None
        else:
            self.log_std = pytorch_utils.build_log_std(environment_info.action_shape)

        self.baseline = pytorch_utils.build_ffn(pytorch_utils.FFNConfig(
            in_shape=environment_info.observation_shape,
            out_shape=(1,),
        ))
        self.baseline_loss_fn = torch.nn.HuberLoss()

    def forward(self, trajectories: Trajectory) -> ModelOutput:
        L = trajectories.L
        action_shape = self.environment_info.action_shape

        actions_mean: torch.Tensor = self.mean_net(trajectories.observations)
        actions_dist = self.create_actions_distribution(actions_mean, self.log_std)

        if not self.training:
            actions: torch.Tensor = actions_dist.sample()
            assert actions.size() == (L, *action_shape)

            return ModelOutput(actions=actions, loss=None)

        action_log_probs = actions_dist \
            .log_prob(trajectories.actions) \
            .view(L, -1) \
            .sum(dim=-1, keepdim=True)

        q_vals = self._calculate_q_vals(trajectories.rewards, trajectories.terminals)
        q_values_normed = self._normalize(q_vals)

        values = self.baseline(trajectories.observations)
        values_to_q_statistics = values * q_vals.std() + q_vals.mean()

        advantages = self._normalize(q_vals - values_to_q_statistics)

        policy_loss_per_sample = -action_log_probs * (advantages.detach())
        policy_loss = policy_loss_per_sample.sum()
        baseline_loss = self.baseline_loss_fn(values, q_values_normed)

        total_loss = policy_loss + baseline_loss

        def check_shapes():
            assert actions_mean.size() == (L, *action_shape)
            assert action_log_probs.size() == (L, 1)

            assert q_vals.size() == (L, 1)
            assert values.size() == (L, 1)
            assert advantages.size() == (L, 1)

            assert policy_loss_per_sample.size() == (L, 1)

        check_shapes()

        logs = {
            "loss_policy": pytorch_utils.to_numpy(policy_loss),
            "loss_baseline": pytorch_utils.to_numpy(baseline_loss),
        }

        return ModelOutput(actions=None, loss=total_loss, logs=logs)

    def _calculate_q_vals(self, rewards: torch.Tensor, terminals: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma

        mask = ~terminals

        q_vals = rewards.clone().detach()
        for i in torch.arange(rewards.size()[0]-2, -1, -1):
            q_vals[i] += gamma * mask[i] * q_vals[i+1]

        return q_vals

    def _normalize(self, t: torch.Tensor) -> torch.Tensor:
        return (t - t.mean()) / (t.std() + 1e-8)
