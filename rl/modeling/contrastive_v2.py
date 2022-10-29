import time

import torch

from rl.modeling.contrastive import ContrastiveBase
from rl.infrastructure import Trajectory, ModelOutput, normalize, to_numpy


__all__ = ["ContrastiveV2Base"]


class ContrastiveV2Base(ContrastiveBase):
    # This is an exact copy of `Contrastive.forward` unless specified otherwise
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

        corresponding_best_q_vals = torch.empty_like(q_vals)
        new_best_q_vals = self.best_q_vals.data.clone().detach()
        corresponding_index = 0
        for i in torch.arange(L):
            corresponding_best_q_vals[i] = self.best_q_vals[corresponding_index]
            corresponding_index = (corresponding_index + 1) * mask[i]
            new_best_q_vals[corresponding_index] = max(new_best_q_vals[corresponding_index], q_vals[i])

        logs["time_q_vals"] += time.time()

        advantages = normalize(q_vals - corresponding_best_q_vals)
        self.best_q_vals.data = new_best_q_vals

        regularization_loss = (1 / L) * sum(p.abs().sum() for p in self.parameters())
        improvement_loss = (-action_log_probs * advantages).sum()
        loss = regularization_loss + improvement_loss

        def check_shapes():
            assert action_log_probs.size() == (L, 1)

            assert q_vals.size() == (L, 1)
            assert corresponding_best_q_vals.size() == (L, 1)
            assert advantages.size() == (L, 1)

        check_shapes()

        logs.update({
            "loss_regularization": to_numpy(regularization_loss),
            "loss_improvement": to_numpy(improvement_loss),
        })

        return ModelOutput(actions=None, loss=loss, logs=logs)
