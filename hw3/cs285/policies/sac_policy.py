from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

from cs285.infrastructure.sac_utils import SquashedNormal

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        return torch.exp(self.log_alpha)

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution 
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # Convert our obs into a form usable by our model
        obs_pt = ptu.from_numpy(observation)

        # Run our model
        action_pt, _ = self(obs_pt)

        # Convert our output action into a form usable by downstream
        action = ptu.to_numpy(action_pt)

        # TODO return the action that the policy prescribes
        return action

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing
        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file 

        # Retrieve relevant objects from self
        log_std = self.logstd
        log_std_min, log_std_max = self.log_std_bounds
        action_min, action_max = self.action_range
        ac_dim = self.ac_dim

        # Clip log_std
        log_std = log_std.clip(log_std_min, log_std_max)

        # Calculate squashed action and log probs
        loc = self.mean_net(observation)
        std = torch.exp(log_std) * torch.ones_like(loc)
        squashed_action_distribution = SquashedNormal(loc, std)
        squashed_action: torch.Tensor = squashed_action_distribution.sample()
        assert loc.size()[0] == squashed_action.size()[0]
        assert squashed_action.size()[1] == ac_dim
        log_probs = squashed_action_distribution.log_prob(squashed_action)
        assert log_probs.size()[0] == 1

        # Calculate action bounds
        action_width = (action_max - action_min) / 2
        action_offset = (action_max + action_min) / 2

        # Rescale action to action range
        action = squashed_action * action_width + action_offset
        assert action.size() == squashed_action.size()

        return action, log_probs

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value

        obs = ptu.from_numpy(obs)

        # Retrieve relevant objects from self
        actor_optimizer = self.optimizer
        alpha_optimizer = self.log_alpha_optimizer
        alpha = self.alpha
        target_entropy = self.target_entropy

        # Setup our optimizers for this train step
        actor_optimizer.zero_grad()
        alpha_optimizer.zero_grad()

        # Get action and log probs
        action, action_log_probs = self(obs)

        # Calculate Q values using critic
        Q1, Q2 = critic(obs, action)
        Q_values = ((Q1 + Q2) / 2)  # Using mean to match the paper implementation

        # Calculate actor loss
        actor_loss_by_sample = alpha.detach() * action_log_probs - Q_values.detach()
        assert actor_loss_by_sample.size() == Q_values.size()
        actor_loss = actor_loss_by_sample.mean()

        # Calculate alpha loss
        alpha_loss_by_sample = - alpha * (action_log_probs + target_entropy).detach()
        assert alpha_loss_by_sample.size() == action_log_probs.size()
        alpha_loss = alpha_loss_by_sample.mean()

        # Update parameters
        actor_loss.backward()
        alpha_loss.backward()
        actor_optimizer.step()
        alpha_optimizer.step()

        return ptu.to_numpy(action.max()), ptu.to_numpy(action.min()), alpha
