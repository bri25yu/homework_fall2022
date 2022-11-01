from collections import OrderedDict

import torch

import gym

import cs285.infrastructure.pytorch_util as ptu
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.infrastructure.sac_utils import soft_update_params

from cs285.agents.base_agent import BaseAgent
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic


class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # Prepare inputs
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        next_ob_no = ptu.from_numpy(next_ob_no)
        re_n = ptu.from_numpy(re_n).unsqueeze(1)
        terminal_n = ptu.from_numpy(terminal_n).unsqueeze(1)

        batch_size = ob_no.size()[0]

        with torch.no_grad():
            # Calculate next action and next log prob
            next_action, next_log_prob, _ = self.actor(next_ob_no)

            # Calculate target Q value
            target_Q1, target_Q2 = self.critic_target(next_ob_no, next_action)
            target_Q = torch.minimum(target_Q1, target_Q2)

            # Calculate target value
            target = re_n + self.gamma * (1 - terminal_n) * (target_Q - self.actor.alpha * next_log_prob)

        # Calculate current Q value
        Q1, Q2 = self.critic(ob_no, ac_na)

        # Check shapes
        def check_shapes():
            assert next_log_prob.size() == (batch_size, 1)
            assert target_Q.size() == (batch_size, 1)
            assert re_n.size() == (batch_size, 1)
            assert terminal_n.size() == (batch_size, 1)
            assert target.size() == (batch_size, 1)
            assert Q1.size() == (batch_size, 1)
            assert Q2.size() == (batch_size, 1)

        check_shapes()

        # Calculate critic loss and update the critic
        Q1_loss = self.critic.loss(Q1, target)
        Q2_loss = self.critic.loss(Q2, target)
        critic_loss = Q1_loss + Q2_loss

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Calculate action and log_prob
        action, log_prob, _ = self.actor(ob_no)

        # Calculate Q_values
        Q1, Q2 = self.critic(ob_no, action)
        Q = torch.minimum(Q1, Q2)

        # Check shapes
        def check_shapes():
            assert log_prob.size() == (batch_size, 1)
            assert Q.size() == (batch_size, 1)

        check_shapes()

        # Calculate actor loss and update the actor
        actor_loss = (self.actor.alpha.detach() * log_prob - Q).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Calculate alpha loss and update alpha
        alpha_loss = (- self.actor.alpha * (log_prob.detach() + self.actor.target_entropy)).mean()
        self.actor.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.actor.log_alpha_optimizer.step()

        if self.training_step % self.critic_target_update_frequency == 0:
            soft_update_params(self.critic, self.critic_target, self.critic_tau)

        # 4. gather losses for logging
        loss = OrderedDict()
        loss['Critic_Loss'] = ptu.to_numpy(critic_loss)
        loss['Actor_Loss'] = ptu.to_numpy(actor_loss)
        loss['Alpha_Loss'] = ptu.to_numpy(alpha_loss)
        loss['Temperature'] = ptu.to_numpy(self.actor.alpha)

        # Record that this action has been taken
        self.training_step += 1

        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)
