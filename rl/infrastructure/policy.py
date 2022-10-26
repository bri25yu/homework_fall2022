import torch
from torch.nn import Module, Parameter
from torch.distributions import AffineTransform, Categorical, Normal, TanhTransform, TransformedDistribution

from rl.infrastructure.environment_info import EnvironmentInfo
from rl.infrastructure.model_output import ModelOutput
from rl.infrastructure.trajectory import Trajectory


class PolicyBase(Module):
    LOG_SCALE_BOUNDS = [-200, 2]

    def __init__(self, environment_info: EnvironmentInfo) -> None:
        super().__init__()

        self.environment_info = environment_info

        if not environment_info.is_discrete:
            action_range_low, action_range_high = self.environment_info.action_range
            action_range_loc = (action_range_high + action_range_low) / 2
            action_range_scale = (action_range_high - action_range_low) / 2

            torch_parameter_from_np = lambda v: Parameter(torch.from_numpy(v), requires_grad=False)
            self.action_range_loc = torch_parameter_from_np(action_range_loc)
            self.action_range_scale = torch_parameter_from_np(action_range_scale)

            action_range_transform = AffineTransform(loc=self.action_range_loc, scale=self.action_range_scale)

            self.action_transforms = [TanhTransform(), action_range_transform]

    def forward(self, trajectory: Trajectory) -> ModelOutput:
        raise NotImplementedError

    def create_actions_distribution(self, loc: torch.Tensor, log_scale: torch.Tensor) -> torch.distributions.Distribution:
        if self.environment_info.is_discrete:
            return Categorical(logits=loc)

        # Otherwise we need to create a continous distribution

        log_scale_min, log_scale_max = self.LOG_SCALE_BOUNDS
        L = loc.size()[0]
        action_shape = self.environment_info.action_shape

        log_scale_clipped = log_scale.clamp(min=log_scale_min, max=log_scale_max)
        scale = log_scale_clipped.exp().repeat(L, *(1,) * len(action_shape))
        assert scale.size() == (L, *action_shape)

        actions_dist_unscaled = Normal(loc, scale)

        action_dist = TransformedDistribution(actions_dist_unscaled, self.action_transforms)

        return action_dist
