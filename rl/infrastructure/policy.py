from typing import Union

import torch
from torch.nn import Module, Parameter
from torch.distributions import AffineTransform, Categorical, Normal, TanhTransform, TransformedDistribution

from gym import Env
from gym.spaces import Discrete

from rl.infrastructure.model_output import ModelOutput
from rl.infrastructure.trajectory import Trajectory


class PolicyBase(Module):
    LOG_SCALE_BOUNDS = [-200, 2]

    def __init__(self, env: Env) -> None:
        super().__init__()

        self.is_discrete = isinstance(env.action_space, Discrete)

        if not self.is_discrete:
            action_range_low = env.action_space.low
            action_range_high = env.action_space.high
            action_range_loc = (action_range_high + action_range_low) / 2
            action_range_scale = (action_range_high - action_range_low) / 2

            torch_parameter_from_np = lambda v: Parameter(torch.from_numpy(v), requires_grad=False)
            self.action_range_loc = torch_parameter_from_np(action_range_loc)
            self.action_range_scale = torch_parameter_from_np(action_range_scale)

            action_range_transform = AffineTransform(loc=self.action_range_loc, scale=self.action_range_scale)

            self.action_transforms = [TanhTransform(), action_range_transform]

    def forward(self, trajectory: Trajectory) -> ModelOutput:
        raise NotImplementedError

    def create_actions_distribution(self, loc: torch.Tensor, log_scale: Union[None, torch.Tensor]) -> torch.distributions.Distribution:
        if self.is_discrete:
            return Categorical(logits=loc)

        # Otherwise we need to create a continous distribution

        log_scale_min, log_scale_max = self.LOG_SCALE_BOUNDS
        L = loc.size()[0]

        log_scale_clipped = log_scale.clamp(min=log_scale_min, max=log_scale_max)
        scale = log_scale_clipped.exp().repeat(L, *(1,) * (len(loc.shape) - 1))

        actions_dist_unscaled = Normal(loc, scale)

        action_dist = TransformedDistribution(actions_dist_unscaled, self.action_transforms)

        return action_dist
