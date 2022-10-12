from collections import OrderedDict

import torch

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from functools import reduce

from cs285.infrastructure.sac_utils import soft_update_params
from .base_agent import BaseAgent
import gym
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic
import cs285.infrastructure.pytorch_util as ptu

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

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        # TODO: 
        # 1. Compute the target Q value. 
        # HINT: You need to use the entropy term (alpha)
        # 2. Get current Q estimates and calculate critic loss
        # 3. Optimize the critic  
        return critic_loss

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # Retrieve relevant objects from self
        num_critic_updates_per_agent_update = self.agent_params["num_critic_updates_per_agent_update"]
        num_actor_updates_per_agent_update = self.agent_params["num_actor_updates_per_agent_update"]
        actor_update_frequency = self.actor_update_frequency
        critic_target_update_frequency = self.critic_target_update_frequency
        critic = self.critic
        critic_target = self.critic_target
        critic_tau = self.critic_tau
        actor = self.actor
        current_training_step = self.training_step

        # Setup logging vars
        critic_loss = actor_loss = alpha_loss = alpha = None

        # 1. Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        for _ in range(num_critic_updates_per_agent_update):
            # update the critic
            critic_loss = self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)

        # 2. Softly update the target every critic_target_update_frequency (HINT: look at sac_utils)
        if current_training_step % critic_target_update_frequency == 0:
            soft_update_params(critic, critic_target, critic_tau)

        # 3. Implement following pseudocode:
        # If you need to update actor
        if current_training_step % actor_update_frequency == 0:
            # for agent_params['num_actor_updates_per_agent_update'] steps,
            for _ in range(num_actor_updates_per_agent_update):
                # update the actor
                actor_loss, alpha_loss, alpha = actor.update(ob_no, critic)

        # 4. gather losses for logging
        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_loss
        loss['Alpha_Loss'] = alpha_loss
        loss['Temperature'] = alpha

        # Record that this action has been taken
        self.training_step += 1

        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
