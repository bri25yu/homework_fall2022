import copy

from gym import Env

import torch
from torch.nn import HuberLoss

from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Stack, T5PreTrainedModel

from rl.infrastructure import Trajectory, ModelOutput, PolicyBase, build_log_std, normalize


__all__ = ["DirectTransformerBase"]


class T5ForReinforcementLearning(T5PreTrainedModel):
    def __init__(self, config: T5Config) -> None:
        super().__init__(config)

        # This is an exact copy of `T5Model.__init__`
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        encoder_outputs = self.encoder(inputs_embeds=trajectory)
        last_hidden_state = encoder_outputs.last_hidden_state

        decoder_outputs = self.decoder(inputs_embeds=last_hidden_state)
        return decoder_outputs.last_hidden_state


class DirectTransformerBase(PolicyBase):
    """
    We have five pieces of information available to us from each trajectory
    - observations
    - actions that were taken
    - next_observations as a result of the action taken
    - rewards as a result of the action taken
    - terminals

    The total input dim into our model is
    obs_dim + ac_dim + 1 + 1 = obs_dim + ac_dim + 2

    We incur several losses:
    - model-based prediction error of the next observation from the current
        observation and action
    - prediction error of the reward from the current observation and action
    - q-value prediction error: Q_t = r_t + gamma * (1 - terminal_t) * Q_{t+1}
    - improvement error: log pi - Q

    So our model needs to predict
    - action mean (ac_dim,)
    - next_observation (ob_dim,)
    - reward (1,)
    - q_value at current timestep (1,)

    The total output dim for our transformer at each timestep should be
    ac_dim + ob_dim + 1 + 1 = ac_dim + ob_dim + 2

    We also need a log_std for our action distribution.

    """
    def __init__(self, env: Env, gamma: float) -> None:
        super().__init__(env)

        self.gamma = gamma
        self.loss_fn = HuberLoss(reduction="none")

        # Assumes 1d obs and acs
        self.obs_dim = env.observation_space.shape[0]
        if self.is_discrete:
            self.acs_dim = env.action_space.n
            self.log_std = None
        else:
            self.acs_dim = env.action_space.shape[0]
            self.log_std = build_log_std(env.action_space.shape)
        self.model_dim = self.obs_dim + self.acs_dim + 1 + 1

        # Create our model
        model_config = T5Config(
            d_model=self.model_dim,
            d_kv=64,
            dff=64,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=4,
        )
        self.model = T5ForReinforcementLearning(model_config)

    def forward(self, trajectories: Trajectory) -> ModelOutput:
        L = trajectories.L
        model_dim = self.model_dim
        obs_dim = self.obs_dim
        acs_dim = self.acs_dim
        loss_fn = self.loss_fn
        gamma = self.gamma
        logs = dict()

        # Reshape actions for categorical features
        actions = trajectories.actions
        if self.is_discrete:
            size_to_fill = acs_dim - 1
            actions = torch.cat((
                actions,
                torch.zeros((L, size_to_fill), dtype=actions.dtype, device=actions.device),
            ), dim=1)

        stacked_input = torch.cat((
            trajectories.observations,
            actions,
            trajectories.rewards,
            trajectories.terminals,
        ), dim=1)
        assert stacked_input.size() == (L, model_dim)

        model_output = self.model(stacked_input[None])[0]  # Unsqueeze for batch dimension and take first element of batch result
        assert model_output.size() == (L, model_dim)

        actions_mean: torch.Tensor = model_output[:, :acs_dim]
        actions_dist = self.create_actions_distribution(actions_mean, self.log_std)

        if not self.training:
            actions: torch.Tensor = actions_dist.sample()
            return ModelOutput(actions=actions, loss=None)

        mask = ~trajectories.terminals

        # Model-based prediction error of the next observation from the current observation and action
        next_observation_prediction: torch.Tensor = model_output[:, acs_dim: acs_dim + obs_dim]
        assert next_observation_prediction.size() == (L, obs_dim)
        model_based_loss_by_timestep = loss_fn(next_observation_prediction, trajectories.next_observations)
        model_based_loss = (model_based_loss_by_timestep * mask).mean()

        # Prediction error of the reward from the current observation and action
        reward_prediction: torch.Tensor = model_output[:, acs_dim + obs_dim: acs_dim + obs_dim + 1]
        assert reward_prediction.size() == (L, 1)
        reward_prediction_loss = loss_fn(reward_prediction, trajectories.rewards).mean()

        # Q-value prediction error: Q_t = r_t + gamma * (1 - terminal_t) * Q_{t+1}
        q_value_prediction: torch.Tensor = model_output[:, acs_dim + obs_dim + 1: acs_dim + obs_dim + 2]
        assert q_value_prediction.size() == (L, 1)
        next_q_value_prediction = q_value_prediction.roll(-1)
        assert q_value_prediction[1] == next_q_value_prediction[0]
        target_q_value = trajectories.rewards + gamma * mask * next_q_value_prediction
        assert target_q_value.size() == (L, 1)
        q_value_prediction_loss = loss_fn(next_q_value_prediction, target_q_value.detach()).mean()

        # Improvement loss: log pi - Q
        action_log_probs = actions_dist \
            .log_prob(trajectories.actions) \
            .view(L, -1) \
            .sum(dim=-1, keepdim=True)
        improvement_loss = (-action_log_probs * normalize(q_value_prediction.detach())).mean()

        total_loss = model_based_loss + reward_prediction_loss + q_value_prediction_loss + improvement_loss

        logs.update({
            "loss_model_based": model_based_loss,
            "loss_reward_prediction": reward_prediction_loss,
            "loss_q_value_prediction": q_value_prediction_loss,
            "loss_improvement": improvement_loss,
        })

        return ModelOutput(actions=None, loss=total_loss, logs=logs)
