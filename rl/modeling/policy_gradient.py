import time

import torch

from gym import Env

from rl.infrastructure import (
    Trajectory, ModelOutput, PolicyBase, build_ffn, FFNConfig, to_numpy, normalize
)


__all__ = ["PolicyGradientBase"]


class PolicyGradientBase(PolicyBase):
    def __init__(self, env: Env, gamma: float) -> None:
        super().__init__(env)

        self.gamma = gamma

        self.initialize_default_policy(env)

        self.baseline = build_ffn(FFNConfig(
            in_shape=env.observation_space.shape,
            out_shape=(1,),
        ))
        self.baseline_loss_fn = torch.nn.HuberLoss()

    def forward(self, trajectories: Trajectory) -> ModelOutput:
        L = trajectories.L
        gamma = self.gamma
        logs = dict()

        actions_mean: torch.Tensor = self.mean_net(trajectories.observations)
        actions_dist = self.create_actions_distribution(actions_mean, self.log_std)

        if not self.training:
            actions: torch.Tensor = actions_dist.sample()
            return ModelOutput(actions=actions, loss=None)

        action_log_probs = actions_dist \
            .log_prob(trajectories.actions) \
            .view(L, -1) \
            .sum(dim=-1, keepdim=True)

        logs["time_q_vals"] = -time.time()
        mask = ~trajectories.terminals
        q_vals = trajectories.rewards.clone().detach()
        for i in torch.arange(L-2, -1, -1):
            q_vals[i] += gamma * mask[i] * q_vals[i+1]
        logs["time_q_vals"] += time.time()

        values = self.baseline(trajectories.observations)
        advantages = normalize(q_vals - (values * q_vals.std() + q_vals.mean()))

        policy_loss = (-action_log_probs * (advantages.detach())).sum()
        baseline_loss = self.baseline_loss_fn(values, normalize(q_vals))

        total_loss = policy_loss + baseline_loss

        def check_shapes():
            assert action_log_probs.size() == (L, 1)

            assert q_vals.size() == (L, 1)
            assert values.size() == (L, 1)
            assert advantages.size() == (L, 1)

        check_shapes()

        logs.update({
            "loss_policy": to_numpy(policy_loss),
            "loss_baseline": to_numpy(baseline_loss),
        })

        return ModelOutput(actions=None, loss=total_loss, logs=logs)
