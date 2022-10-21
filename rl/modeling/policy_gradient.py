from typing import Union

import torch
import torch.nn as nn

from rl.infrastructure import BatchTrajectoriesPyTorch, EnvironmentInfo, ModelOutput, PolicyBase, pytorch_utils


__all__ = ["PolicyGradientBase"]


class PolicyGradientBase(PolicyBase):
    GAMMA: Union[None, float] = None

    def __init__(self, environment_info: EnvironmentInfo) -> None:
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
        gamma = self.GAMMA
        max_trajectory_length = environment_info.max_trajectory_length
        # gamma_vector_precomputed is of shape (max_trajectory_length, 1)
        self.gamma_vector_precomputed = nn.Parameter(
            (gamma ** torch.arange(max_trajectory_length)).unsqueeze(1),
            requires_grad=False,
        )

    def forward(self, trajectories: BatchTrajectoriesPyTorch) -> ModelOutput:
        batch_size = trajectories.batch_size
        max_sequence_length = trajectories.observations.size()[1]
        action_dims = len(trajectories.environment_info.action_shape)

        q_vals = self._calculate_q_vals(trajectories.rewards)

        values = self.baseline(trajectories.observations)
        assert values.size() == q_vals.size()

        actions_mean = self.mean_net(trajectories.observations)
        actions_std = self.log_std.repeat(batch_size, max_sequence_length, *(1,) * action_dims)
        assert actions_mean.size() == trajectories.actions.size()
        assert actions_std.size() == trajectories.actions.size()

        actions_dist = torch.distributions.Normal(actions_mean, actions_std)
        actions: torch.Tensor = actions_dist.rsample()
        assert actions.size() == trajectories.actions.size()

        action_log_probs: torch.Tensor = actions_dist.log_prob(actions)
        action_log_probs = action_log_probs.sum(dim=2, keepdim=True)

        advantages_unnormalized = q_vals - values
        advantages = nn.functional.layer_norm(advantages_unnormalized, (max_sequence_length, 1))

        policy_loss = -action_log_probs * advantages.detach()
        baseline_loss = nn.functional.mse_loss(values, q_vals)

        total_loss = policy_loss + baseline_loss

        return ModelOutput(actions=actions, loss=total_loss)

    def _calculate_q_vals(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        We use the following discounted rewards formulation:
            q_vals[t] = sum_{t'=t}^T gamma^(t'-t) * r_{t'}

        Translated to code:
            q_vals[i] = (gamma ** 0) * rewards[i] + ... + (gamma ** (T-1 - i)) * rewards[T-1]

        A popular alternative recursive formulation is below, but we stick to using tensor operations.
            q_vals[i] = gamma * q_vals[i-1] + rewards[i]

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
        assert rewards_discounted.size() == rewards.size()

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
        assert q_vals_unnormalized.size() == rewards.size()

        """
        q_vals[i] = (gamma ** 0) * rewards[i] + ... + (gamma ** (T-1 - i)) * rewards[T-1]
        """
        q_vals = torch.div(q_vals_unnormalized, gamma)
        assert q_vals.size() == rewards.size()

        return q_vals
