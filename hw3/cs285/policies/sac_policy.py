from typing import Tuple

import numpy as np
import torch

from cs285.policies.MLP_policy import MLPPolicy
from cs285.infrastructure import pytorch_util as ptu
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
        
        # Prepare inputs
        obs = ptu.from_numpy(obs)

        # Run our policy
        action, _, action_mean = self(obs)

        # Get our action to return
        if sample:
            action_to_return = action
        else:
            action_to_return = action_mean

        action_to_return = ptu.to_numpy(action_to_return)

        return action_to_return

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing

        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file 

        # Retrieve relevant objects from self
        mean_net = self.mean_net
        logstd = self.logstd
        logstd_min, logstd_max = self.log_std_bounds
        action_range_min, action_range_max = self.action_range
        action_dim = self.ac_dim

        # Calculate relevant properties
        batch_size = observation.size()[0]

        # Calculate loc for our SquashedNormal action distribution
        loc = mean_net(observation)
        assert loc.size() == (batch_size, action_dim)

        # Calculate scale for our SquashedNormal action distribution
        logstd_clipped = logstd.clamp(min=logstd_min, max=logstd_max)
        scale_single = torch.exp(logstd_clipped)
        assert scale_single.size() == (action_dim,)
        scale = scale_single.repeat(batch_size, 1)
        assert scale.size() == (batch_size, action_dim)

        # Create our SquashedNormal action distribution
        squashed_action_distribution = SquashedNormal(loc, scale)
        squashed_action = squashed_action_distribution.sample()  # For the moment we use `sample` rather than `rsample`
        log_prob = squashed_action_distribution.log_prob(squashed_action)
        assert squashed_action.size() == (batch_size, action_dim)
        assert log_prob.size() == (batch_size, action_dim)

        # Calculate parameters of action range
        action_range_width = (action_range_max - action_range_min) / 2
        action_range_offset = (action_range_max + action_range_min) / 2

        # Shift and scale our squashed action to action range
        action = squashed_action * action_range_width + action_range_offset
        assert action.size() == (batch_size, action_dim)

        # Apply the same shifting and scaling for our action mean
        squashed_action_mean = squashed_action_distribution.mean
        action_mean = squashed_action_mean * action_range_width + action_range_offset

        return action, log_prob, action_mean

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value

        # Prepare inputs
        obs = ptu.from_numpy(obs)
        batch_size = obs.size()[0]

        # Retrieve relevant objects from self
        alpha = self.alpha
        target_entropy = self.target_entropy
        actor_optimizer = self.optimizer
        alpha_optimizer = self.log_alpha_optimizer

        # Setup optimizers for new update step
        actor_optimizer.zero_grad()
        alpha_optimizer.zero_grad()

        # Calculate action and log_prob
        action, log_prob, _ = self(obs)

        # Calculate Q_values
        Q1, Q2 = critic(obs, action)
        Q = torch.minimum(Q1, Q2)
        assert Q.size() == (batch_size, 1)

        # Calculate usable log_prob per sample
        log_prob_per_sample = log_prob.sum(dim=1)
        assert log_prob_per_sample.size() == (batch_size,)

        # Calculate actor loss
        Q_squeezed = Q.squeeze()
        assert Q_squeezed.size() == (batch_size,)
        actor_loss_per_sample = alpha.detach() * log_prob_per_sample - Q_squeezed.detach()
        assert actor_loss_per_sample.size() == (batch_size,)
        actor_loss = actor_loss_per_sample.mean()

        # Calculate alpha loss
        alpha_loss_per_sample = - alpha * (log_prob_per_sample.detach() + target_entropy)
        assert alpha_loss_per_sample.size() == (batch_size,)
        alpha_loss = alpha_loss_per_sample.mean()

        # Apply updates to parameters
        actor_loss.backward()
        alpha_loss.backward()
        actor_optimizer.step()
        alpha_optimizer.step()

        return ptu.to_numpy(actor_loss), ptu.to_numpy(alpha_loss), alpha
