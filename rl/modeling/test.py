from dataclasses import dataclass

import torch
import torch.nn as nn

from gym import Env

from rl.infrastructure import (
    Trajectory, ModelOutput, PolicyBase, build_ffn, FFNConfig, to_numpy, normalize
)
from rl.infrastructure.pytorch_utils import build_log_std
from rl.modeling.utils import assert_shape, calculate_log_probs, calculate_q_values, get_log_probs_logs


__all__ = ["TestModel"]


class ReducedLayerNorm(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones((hidden_dim,)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.weights * inputs


class Dense(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layernorm = ReducedLayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.nonlinearity = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, out_dim, bias=False)

        # Initialize weights using Gaussian distribution
        self.linear1.weight.data.normal_(mean=0.0, std=in_dim ** -0.5)
        self.linear2.weight.data.normal_(mean=0.0, std=hidden_dim ** -0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.layernorm(inputs)
        inputs = self.linear1(inputs)
        inputs = self.nonlinearity(inputs)
        inputs = self.linear2(inputs)

        return inputs


class ReducedAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.layernorm = ReducedLayerNorm(hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.layernorm(inputs)
        return torch.softmax(inputs, dim=1)


@dataclass
class SimpleTransformerConfig:
    in_dim: int
    out_dim: int
    num_layers: int = 2
    hidden_dim: int = 64


class SimpleTransformer(nn.Module):
    def __init__(self, config: SimpleTransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.dense = Dense(config.in_dim, config.hidden_dim, config.hidden_dim)
        self.attention = ReducedAttention(config.hidden_dim)
        self.output = Dense(config.hidden_dim, config.hidden_dim, config.out_dim)
        self.final_layer_norm = ReducedLayerNorm(config.out_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.dense(inputs)
        inputs = self.attention(inputs)
        inputs = self.output(inputs)
        inputs = self.final_layer_norm(inputs)

        return inputs


class TestModel(PolicyBase):
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
            in_dim=env.observation_space.shape,
            out_dim=out_dim,
        )
        self.mean_net = SimpleTransformer(config)

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
            **get_log_probs_logs(action_log_probs),
        }

        return ModelOutput(actions=None, loss=total_loss, logs=logs)
