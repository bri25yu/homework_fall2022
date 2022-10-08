from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC

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

        # Retrieve relevant objects from self
        gamma = self.gamma
        alpha = self.actor.alpha
        critic = self.critic
        critic_target = self.critic_target
        critic_optimizer = critic.optimizer
        loss_fn = critic.loss

        # Convert inputs to usable form
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        next_ob_no = ptu.from_numpy(next_ob_no)
        re_n = ptu.from_numpy(re_n)

        # Reset optimizers
        critic_optimizer.zero_grad()

        # !TODO

        target_Q1, target_Q2 = critic_target(ob_no, ac_na)
        Q1, Q2 = critic(ob_no)
        calculate_single_target = lambda target_Q: re_n + gamma * alpha * None
        critic_loss = loss_fn(None, target_Q1)

        # Update parameters
        critic_loss.backward()
        critic_optimizer.step()

        return ptu.to_numpy(critic_loss)

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # Retrieve relevant objects from self
        num_critic_updates_per_agent_update = self.agent_params["num_critic_updates_per_agent_update"]
        actor_update_frequency = self.actor_update_frequency
        critic_target_update_frequency = self.critic_target_update_frequency
        critic = self.critic
        critic_target = self.critic_target
        critic_tau = self.critic_tau

        # TODO 
        # 1. Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic
        critic_loss = None
        for step in range(num_critic_updates_per_agent_update):
            if step % critic_target_update_frequency == 0:
                soft_update_params(critic, critic_target, critic_tau)

            critic_loss = self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)

        # 2. Softly update the target every critic_target_update_frequency (HINT: look at sac_utils)

        # 3. Implement following pseudocode:
        # If you need to update actor
        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor
        actor_loss, alpha_loss, alpha = None, None, None
        for _ in range(actor_update_frequency):
            actor_loss, alpha_loss, alpha = self.actor.update(ob_no, critic)

        # 4. gather losses for logging
        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_loss
        loss['Alpha_Loss'] = alpha_loss
        loss['Temperature'] = alpha

        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
