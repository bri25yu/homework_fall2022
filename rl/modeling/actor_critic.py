import time

import torch

from gym import Env

from rl.infrastructure import (
    Trajectory, ModelOutput, PolicyBase, build_ffn, build_log_std, FFNConfig, to_numpy
)


__all__ = ["ActorCriticBase"]


class ActorCriticBase(PolicyBase):
    def __init__(self, env: Env, gamma: float) -> None:
        super().__init__(env)

        self.gamma = gamma

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

        self.critic = build_ffn(FFNConfig(
            in_shape=env.observation_space.shape,
            out_shape=(1,),
        ))
        self.critic_loss_fn = torch.nn.HuberLoss()

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

        V_s = self.critic(trajectories.observations)

        with torch.no_grad():
            rewards_normed = self._normalize(trajectories.rewards)
            V_s_prime = self.critic(trajectories.next_observations)
            Q_s_a = rewards_normed + gamma * V_s_prime * (~trajectories.terminals)
            advantages = self._normalize(Q_s_a - V_s)

        policy_loss = (-action_log_probs * advantages).sum()
        critic_loss = self.critic_loss_fn(V_s, advantages)

        total_loss = policy_loss + critic_loss

        def check_shapes():
            assert action_log_probs.size() == (L, 1)
            assert V_s.size() == (L, 1)
            assert Q_s_a.size() == (L, 1)
            assert advantages.size() == (L, 1)

        check_shapes()

        logs.update({
            "loss_policy": to_numpy(policy_loss),
            "loss_critic": to_numpy(critic_loss),
        })

        return ModelOutput(actions=None, loss=total_loss, logs=logs)

    def _normalize(self, t: torch.Tensor) -> torch.Tensor:
        return (t - t.mean()) / (t.std() + 1e-8)
