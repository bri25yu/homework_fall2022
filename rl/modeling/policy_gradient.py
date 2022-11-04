import torch

from gym import Env

from rl.infrastructure import (
    Trajectory, ModelOutput, PolicyBase, build_ffn, FFNConfig, to_numpy, normalize
)
from rl.modeling.utils import assert_shape, calculate_log_probs, calculate_q_values


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
        actions_mean: torch.Tensor = self.mean_net(trajectories.observations)
        actions_dist = self.create_actions_distribution(actions_mean, self.log_std)

        if not self.training:
            actions: torch.Tensor = actions_dist.sample()
            return ModelOutput(actions=actions, loss=None)

        action_log_probs = calculate_log_probs(actions_dist, trajectories.actions)

        q_vals = calculate_q_values(trajectories.rewards, trajectories.terminals, self.gamma)

        values = self.baseline(trajectories.observations)
        advantages = normalize(q_vals - (values * q_vals.std() + q_vals.mean()))

        L = trajectories.L
        assert_shape(values, (L, 1))
        assert_shape(advantages, (L, 1))

        policy_loss = (-action_log_probs * (advantages.detach())).sum()
        baseline_loss = self.baseline_loss_fn(values, normalize(q_vals))

        total_loss = policy_loss + baseline_loss

        logs = {
            "loss_policy": to_numpy(policy_loss),
            "loss_baseline": to_numpy(baseline_loss),
            "value_log_probs_mean": -action_log_probs.detach().mean(),
        }

        return ModelOutput(actions=None, loss=total_loss, logs=logs)
