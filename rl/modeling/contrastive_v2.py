import torch

from rl.modeling.contrastive import ContrastiveBase
from rl.infrastructure import Trajectory, ModelOutput, normalize, to_numpy
from rl.modeling.utils import assert_shape, calculate_log_probs, calculate_q_values, calculate_contrastive_q_values_update


__all__ = ["ContrastiveV2Base"]


class ContrastiveV2Base(ContrastiveBase):
    # This is an exact copy of `Contrastive.forward` unless specified otherwise
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

        ###############################
        # START add log_prob proportional regularization
        ###############################

        # Original code:
        # loss = (-action_log_probs * advantages).sum()

        regularization_strength = 1 / (-action_log_probs.detach().mean())
        regularization_loss = regularization_strength * sum(p.abs().sum() for p in self.parameters())

        improvement_loss = (-action_log_probs * advantages).sum()
        loss = regularization_loss + improvement_loss

        ###############################
        # END add log_prob proportional regularization
        ###############################

        L = trajectories.L
        assert_shape(advantages, (L, 1))

        logs = {
            "loss_regularization": to_numpy(regularization_loss),
            "loss_improvement": to_numpy(improvement_loss),
            "value_regularization_strength": to_numpy(regularization_strength),
            "value_log_probs_mean": -action_log_probs.detach().mean(),
            "value_best_q_val_0": self.best_q_vals.data[0].detach(),
        }

        return ModelOutput(actions=None, loss=loss, logs=logs)
