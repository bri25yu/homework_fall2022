from dataclasses import dataclass

import torch
import torch.nn as nn

from gym import Env

from rl.infrastructure import (
    Trajectory, ModelOutput, PolicyBase, to_numpy
)
from rl.infrastructure.pytorch_utils import build_log_std
from rl.modeling.utils import assert_shape, calculate_log_probs, calculate_q_values, get_log_probs_logs

from rl.modeling.simple_transformer import (
    SimpleTransformerConfig,
    SimpleTransformer,
    Baseline,
)


__all__ = ["MinEnsembleBaselineModel"]


class MinEnsembleBaselineModel(PolicyBase):
    def __init__(self, env: Env, gamma: float) -> None:
        super().__init__(env)

        self.gamma = gamma

        if self.is_discrete:
            out_dim = env.action_space.n
            self.log_std = None
        else:
            out_dim = env.action_space.shape[0]
            self.log_std = build_log_std(env.action_space.shape)

        config = SimpleTransformerConfig(
            in_dim=env.observation_space.shape[0],
            out_dim=out_dim,
        )
        self.mean_net = SimpleTransformer(config)

        num_baselines = 3
        self.baselines = nn.ModuleList([Baseline(config) for _ in range(num_baselines)])
        self.baseline_loss_fn = torch.nn.HuberLoss()

    def forward(self, trajectories: Trajectory) -> ModelOutput:
        actions_mean: torch.Tensor = self.mean_net(trajectories.observations)
        actions_dist = self.create_actions_distribution(actions_mean, self.log_std)

        if not self.training:
            actions: torch.Tensor = actions_dist.sample()
            return ModelOutput(actions=actions, loss=None)

        action_log_probs = calculate_log_probs(actions_dist, trajectories.actions)

        q_vals = calculate_q_values(trajectories.rewards, trajectories.terminals, self.gamma)

        baseline_values = torch.cat([b(trajectories.observations) for b in self.baselines], dim=1)
        values = baseline_values.min(dim=1, keepdim=True)[0]
        advantages = q_vals - values

        L = trajectories.L
        assert_shape(values, (L, 1))
        assert_shape(advantages, (L, 1))

        policy_loss = (-action_log_probs * (advantages.detach())).sum()
        baseline_loss = self.baseline_loss_fn(values, q_vals)

        total_loss = policy_loss + baseline_loss

        logs = {
            "loss_policy": to_numpy(policy_loss),
            "loss_baseline": to_numpy(baseline_loss),
            **get_log_probs_logs(action_log_probs),
        }

        return ModelOutput(actions=None, loss=total_loss, logs=logs)
