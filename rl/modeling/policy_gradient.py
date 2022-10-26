import torch

from rl.infrastructure import Trajectory, EnvironmentInfo, ModelOutput, PolicyBase, pytorch_utils


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

        self.gamma = gamma

    def forward(self, trajectories: Trajectory) -> ModelOutput:
        L = trajectories.L
        action_shape = trajectories.environment_info.action_shape

        actions_mean: torch.Tensor = self.mean_net(trajectories.observations)
        actions_std = self.log_std.exp().repeat(L, *(1,) * len(action_shape))
        actions_dist = torch.distributions.Normal(actions_mean, actions_std)

        if not self.training:
            actions: torch.Tensor = actions_dist.sample()
            assert actions.size() == (L, *action_shape)

            return ModelOutput(actions=actions, loss=None)

        action_log_probs: torch.Tensor = actions_dist.log_prob(trajectories.actions)
        action_log_probs = action_log_probs.view(L, -1).sum(dim=-1, keepdim=True)

        q_vals_batched = self._calculate_q_vals(trajectories.reshape_rewards_by_trajectory())
        q_vals = trajectories.flatten_tensor_by_trajectory(q_vals_batched)
        values = self.baseline(trajectories.observations)
        values = (values - values.mean()) / (values.std() + 1e-8)
        values = values * q_vals.std() + q_vals.mean()

        advantages_unnormalized: torch.Tensor = q_vals - values
        advantages = (advantages_unnormalized - advantages_unnormalized.mean()) / (advantages_unnormalized.std() + 1e-8)

        policy_loss_per_sample = -action_log_probs * advantages.detach()
        baseline_loss_per_sample = (advantages ** 2)
        policy_loss = policy_loss_per_sample.sum()
        baseline_loss = baseline_loss_per_sample.sum()

        total_loss = policy_loss + baseline_loss

        def check_shapes():
            assert q_vals.size() == (L, 1)
            assert values.size() == (L, 1)
            assert actions_mean.size() == (L, *action_shape)
            assert actions_std.size() == (L, *action_shape)
            assert action_log_probs.size() == (L, 1)
            assert advantages.size() == (L, 1)
            assert policy_loss_per_sample.size() == (L, 1)
            assert baseline_loss_per_sample.size() == (L, 1)

        check_shapes()

        logs = {
            "loss_policy": pytorch_utils.to_numpy(policy_loss),
            "loss_baseline": pytorch_utils.to_numpy(baseline_loss),
        }

        return ModelOutput(actions=None, loss=total_loss, logs=logs)

    def _calculate_q_vals(self, rewards_batched: torch.Tensor) -> torch.Tensor:
        """
        `rewards_batched` is shaped (n_trajectories, max_trajectory_length, 1)

        We use the following discounted rewards formulation:
            q_vals[t] = sum_{t'=t}^T gamma^(t'-t) * r_{t'}

        Translated to code:
            q_vals[i] = (gamma ** 0) * rewards[i] + ... + (gamma ** (T-1 - i)) * rewards[T-1]

        A popular alternative recursive formulation is below, but we stick to using tensor operations.
            q_vals[i] = gamma * q_vals[i+1] + rewards[i]

        q_vals is of shape (batch_size, max_trajectory_length, 1), the same as rewards

        Inspired by https://discuss.pytorch.org/t/cumulative-sum-with-decay-factor/69788/2

        """

        """
        Precompute gamma vector of shape (max_trajectory_length, 1)
        gamma_vector_precomputed[i] = gamma ** i
        """
        max_trajectory_length = rewards_batched.shape[1]
        gamma_range = torch.arange(max_trajectory_length, device=rewards_batched.device, dtype=rewards_batched.dtype).unsqueeze(1)
        gamma_vector = self.gamma ** gamma_range

        """
        rewards_discounted[i] = (gamma ** i) * rewards[i]

        `rewards_batched` is of shape (n_trajectories, max_trajectory_length, 1)
        `gamma_vector` is of shape (max_trajectory_length, 1)
        `rewards_discounted` is of shape (n_trajectories, max_trajectory_length, 1)
        """
        rewards_discounted = torch.mul(rewards_batched, gamma_vector)

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

        # q_vals[i] = (gamma ** 0) * rewards[i] + ... + (gamma ** (T-1 - i)) * rewards[T-1]
        q_vals = torch.div(q_vals_unnormalized, gamma_vector)

        def check_shapes():
            assert rewards_discounted.size() == rewards_batched.size()
            assert q_vals_unnormalized.size() == rewards_batched.size()
            assert q_vals.size() == rewards_batched.size()

        check_shapes()

        return q_vals
