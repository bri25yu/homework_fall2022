import torch

from gym import Env

from rl.infrastructure import Trajectory, ModelOutput, PolicyBase, normalize
from rl.modeling.utils import assert_shape, calculate_log_probs, calculate_q_values, calculate_contrastive_q_values_update, get_log_probs_logs


__all__ = ["ContrastiveBase"]


class ContrastiveBase(PolicyBase):
    def __init__(self, env: Env, gamma: float) -> None:
        super().__init__(env)

        self.gamma = gamma

        self.initialize_default_policy(env)

        L = env.spec.max_episode_steps
        self.best_q_vals = torch.nn.Parameter(torch.zeros(L, 1), requires_grad=False)

    def forward(self, trajectories: Trajectory) -> ModelOutput:
        actions_mean: torch.Tensor = self.mean_net(trajectories.observations)
        actions_dist = self.create_actions_distribution(actions_mean, self.log_std)

        if not self.training:
            actions: torch.Tensor = actions_dist.sample()
            return ModelOutput(actions=actions, loss=None)

        action_log_probs = calculate_log_probs(actions_dist, trajectories.actions)

        q_vals = calculate_q_values(trajectories.rewards, trajectories.terminals, self.gamma)
        corresponding_best_q_vals, self.best_q_vals.data = calculate_contrastive_q_values_update(
            q_vals, self.best_q_vals, trajectories.terminals
        )

        advantages = normalize(q_vals - corresponding_best_q_vals)
        L = trajectories.L
        assert_shape(advantages, (L, 1))

        loss = (-action_log_probs * advantages).sum()

        logs = {
            "value_best_q_val_0": self.best_q_vals.data[0].detach(),
            **get_log_probs_logs(action_log_probs),
        }

        return ModelOutput(actions=None, loss=loss, logs=logs)
