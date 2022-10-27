from typing import Any, Dict, Union

from dataclasses import dataclass

import torch
from torch.nn import Module
from torch.distributions import Categorical, Normal

from gym import Env
from gym.spaces import Discrete

from rl.infrastructure.trajectory import Trajectory
from rl.infrastructure.pytorch_utils import build_ffn, build_log_std, FFNConfig


@dataclass
class ModelOutput:
    actions: Union[None, torch.Tensor] = None  # Of shape (L, *action_shape)
    loss: Union[None, torch.Tensor] = None
    logs: Union[None, Dict[str, Any]] = None


class PolicyBase(Module):
    LOG_SCALE_BOUNDS = [-200, 2]

    def __init__(self, env: Env) -> None:
        super().__init__()

        self.is_discrete = isinstance(env.action_space, Discrete)

    def initialize_default_policy(self, env: Env) -> None:
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

    def forward(self, trajectory: Trajectory) -> ModelOutput:
        raise NotImplementedError

    def create_actions_distribution(self, loc: torch.Tensor, log_scale: Union[None, torch.Tensor]) -> torch.distributions.Distribution:
        if self.is_discrete:
            return Categorical(logits=loc)
        else:
            log_scale_min, log_scale_max = self.LOG_SCALE_BOUNDS
            L = loc.size()[0]
            n_action_dims = len(loc.shape) - 1

            log_scale_clipped = log_scale.clamp(min=log_scale_min, max=log_scale_max)
            scale = log_scale_clipped.exp().repeat(L, *(1,) * n_action_dims)

            actions_dist = Normal(loc, scale)

            return actions_dist
