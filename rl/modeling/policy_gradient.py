from email.mime import base
from typing import Any, Dict

import torch
import torch.nn as nn

from rl.infrastructure import BatchTrajectory, EnvironmentInfo, ModelOutput, PolicyBase, pytorch_utils


__all__ = ["PolicyGradientBase"]


class PolicyGradientBase(PolicyBase):
    def __init__(self, environment_info: EnvironmentInfo, gamma: float) -> None:
        super().__init__()

        self.mean_net = pytorch_utils.build_ffn(pytorch_utils.FFNConfig(
            in_shape=environment_info.observation_shape,
            out_shape=environment_info.action_shape,
        ))
        self.log_std = pytorch_utils.build_log_std(environment_info.action_shape)
        self.baseline = pytorch_utils.build_ffn(pytorch_utils.FFNConfig(
            in_shape=environment_info.observation_shape,
            out_shape=(1,),
        ))

        # Precompute gamma vector
        # gamma_vector_precomputed[i] = gamma ** i
        max_trajectory_length = environment_info.max_trajectory_length
        # gamma_vector_precomputed is of shape (max_trajectory_length, 1)
        self.gamma_vector_precomputed = nn.Parameter(
            (gamma ** torch.arange(max_trajectory_length)).unsqueeze(1),
            requires_grad=False,
        )

    def forward(self, trajectories: BatchTrajectory) -> ModelOutput:
        batch_size = trajectories.batch_size
        max_sequence_length = trajectories.environment_info.max_trajectory_length
        action_shape = trajectories.environment_info.action_shape
        action_dims = len(action_shape)

        actions_mean: torch.Tensor = self.mean_net(trajectories.observations)
        actions_std = self.log_std.exp().repeat(batch_size, max_sequence_length, *(1,) * action_dims)
        actions_dist = torch.distributions.Normal(actions_mean, actions_std, validate_args=False)

        if self.training:
            actions: torch.Tensor = actions_dist.sample()
            assert actions.size() == (batch_size, max_sequence_length, *action_shape)

            return ModelOutput(actions=actions, loss=None)

        q_vals = self._calculate_q_vals(trajectories.rewards)
        values = self.baseline(trajectories.observations)

        action_log_probs: torch.Tensor = actions_dist.log_prob(trajectories.actions)
        action_log_probs = action_log_probs.view(batch_size, max_sequence_length, -1).sum(dim=2, keepdim=True)

        advantages_unnormalized: torch.Tensor = q_vals - values
        advantages = (advantages_unnormalized - advantages_unnormalized.mean()) / (advantages_unnormalized.std() + 1e-8)

        policy_loss_per_sample = -action_log_probs * advantages.detach() * trajectories.mask
        policy_loss = policy_loss_per_sample.sum()
        baseline_loss_per_sample = (advantages_unnormalized ** 2) * trajectories.mask
        baseline_loss = baseline_loss_per_sample.sum()

        total_loss = policy_loss + baseline_loss

        def check_shapes():
            assert values.size() == (batch_size, max_sequence_length, 1)
            assert actions_mean.size() == (batch_size, max_sequence_length, *action_shape)
            assert actions_std.size() == (batch_size, max_sequence_length, *action_shape)
            assert action_log_probs.size() == (batch_size, max_sequence_length, 1)
            assert advantages.size() == (batch_size, max_sequence_length, 1)
            assert policy_loss_per_sample.size() == (batch_size, max_sequence_length, 1)
            assert baseline_loss_per_sample.size() == (batch_size, max_sequence_length, 1)

        check_shapes()

        logs = {
            "policy_loss": pytorch_utils.to_numpy(policy_loss),
            "baseline_loss": pytorch_utils.to_numpy(baseline_loss),
        }

        return ModelOutput(actions=None, loss=total_loss, logs=logs)

    def _calculate_q_vals(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        We use the following discounted rewards formulation:
            q_vals[t] = sum_{t'=t}^T gamma^(t'-t) * r_{t'}

        Translated to code:
            q_vals[i] = (gamma ** 0) * rewards[i] + ... + (gamma ** (T-1 - i)) * rewards[T-1]

        A popular alternative recursive formulation is below, but we stick to using tensor operations.
            q_vals[i] = gamma * q_vals[i+1] + rewards[i]

        q_vals is of shape (batch_size, max_trajectory_length, 1), the same as rewards

        Inspired by https://discuss.pytorch.org/t/cumulative-sum-with-decay-factor/69788/2

        """
        gamma = self.gamma_vector_precomputed

        """
        rewards_discounted[i] = (gamma ** i) * rewards[i]

        `rewards` is of shape (batch_size, max_trajectory_length, 1)
        `gamma` is of shape (max_trajectory_length, 1)
        """
        rewards_discounted = torch.mul(rewards, gamma)

        """
        A cumsum proceeds from front to back, but we need to go from back to front so we need to perform a reverse cumsum.
        See https://github.com/pytorch/pytorch/issues/33520#issuecomment-812907290

        torch.cumsum(...) performs
            y[i] = x[0] + ... + x[i]
        torch.cumsum(rewards_discounted, dim=1) performs
            a[i] = (gamma ** 0) * rewards[0] + ... + (gamma ** i) * rewards[i]
        torch.sum(rewards_discounted, dim=1, keepdims=True) performs
            b = (gamma ** 0) * rewards[0] + ... + (gamma ** (T-1)) * rewards[T-1]

        (gamma ** i) * rewards[i] + ... + (gamma ** (T-1)) * rewards[T-1]
             = (gamma ** i) * rewards[i] + ((gamma ** 0) * rewards[0] + ... + (gamma ** (T-1)) * rewards[T-1])
                - ((gamma ** 0) * rewards[0] + ... + (gamma ** i) * rewards[i])

        q_vals_unnormalized[i] = (gamma ** i) * rewards[i] + ... + (gamma ** (T-1)) * rewards[T-1]
        """
        q_vals_unnormalized = rewards_discounted + torch.sum(rewards_discounted, dim=1, keepdim=True) - torch.cumsum(rewards_discounted, dim=1)

        """
        q_vals[i] = (gamma ** 0) * rewards[i] + ... + (gamma ** (T-1 - i)) * rewards[T-1]
        """
        q_vals = torch.div(q_vals_unnormalized, gamma)

        def check_shapes():
            assert rewards_discounted.size() == rewards.size()
            assert q_vals_unnormalized.size() == rewards.size()
            assert q_vals.size() == rewards.size()

        check_shapes()

        return q_vals
