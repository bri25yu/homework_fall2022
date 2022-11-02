import numpy as np

from .base_policy import BasePolicy

import cs285.infrastructure.pytorch_util as ptu


class MPCPolicy(BasePolicy):

    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 sample_strategy='random',
                 cem_iterations=4,
                 cem_num_elites=5,
                 cem_alpha=1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert sample_strategy in allowed_sampling, f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy
        self.cem_iterations = cem_iterations
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem':
            print(f"CEM params: alpha={self.cem_alpha}, "
                + f"num_elites={self.cem_num_elites}, iterations={self.cem_iterations}")

    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        def sample_uniformly_random():
            loc = (self.low + self.high) / 2
            scale = (self.high - self.low) / 2

            r = np.random.rand(num_sequences, horizon, self.ac_dim)
            return (2 * r - 1) * scale + loc

        if self.sample_strategy == 'random' \
            or (self.sample_strategy == 'cem' and obs is None):
            # TODO(Q1) uniformly sample trajectories and return an array of
            # dimensions (num_sequences, horizon, self.ac_dim) in the range
            # [self.low, self.high]
            random_action_sequences = sample_uniformly_random()
            return random_action_sequences
        elif self.sample_strategy == 'cem':
            # TODO(Q5): Implement action selection using CEM.
            # Begin with randomly selected actions, then refine the sampling distribution
            # iteratively as described in Section 3.3, "Iterative Random-Shooting with Refinement" of
            # https://arxiv.org/pdf/1909.11652.pdf 

            def get_elites_statistics(candidate_action_sequences):
                rewards = self.evaluate_candidate_sequences_vectorized(candidate_action_sequences, obs)
                best_k = np.argpartition(rewards, -self.cem_num_elites)[-self.cem_num_elites:]
                elites = candidate_action_sequences[best_k]
                return elites.mean(axis=0), elites.std(axis=0)

            candidate_action_sequences = sample_uniformly_random()  # (N, H, ac_dim)
            elites_mean, elites_std = get_elites_statistics(candidate_action_sequences)

            for _ in range(1, self.cem_iterations):
                # - Sample candidate sequences from a Gaussian with the current 
                #   elite mean and variance
                #     (Hint: remember that for the first iteration, we instead sample
                #      uniformly at random just like we do for random-shooting)
                # - Get the top `self.cem_num_elites` elites
                #     (Hint: what existing function can we use to compute rewards for
                #      our candidate sequences in order to rank them?)
                # - Update the elite mean and variance
                candidate_action_sequences = np.random.normal(elites_mean, elites_std, size=(num_sequences, *elites_mean.shape))
                candidate_elite_mean, candidate_elite_std = get_elites_statistics(candidate_action_sequences)
                elites_mean = self.cem_alpha * candidate_elite_mean + (1 - self.cem_alpha) * elites_mean
                elites_std = self.cem_alpha * candidate_elite_std + (1 - self.cem_alpha) * elites_std

            # TODO(Q5): Set `cem_action` to the appropriate action chosen by CEM
            cem_action = elites_mean

            return cem_action[None]
        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        # TODO(Q2): for each model in ensemble, compute the predicted sum of rewards
        # for each candidate action sequence.
        #
        # Then, return the mean predictions across all ensembles.
        # Hint: the return value should be an array of shape (N,)
        N = candidate_action_sequences.shape[0]
        rewards = np.zeros((N,))
        for model in self.dyn_models:
            rewards += self.calculate_sum_of_rewards(obs, candidate_action_sequences, model)

        return rewards / len(self.dyn_models)

    def evaluate_candidate_sequences_vectorized(self, candidate_action_sequences, obs_single):
        """
        `candidate_action_sequences` is (n_sequences, horizon, *ac_shape)
        `obs` is ob_shape

        Inspired by https://pytorch.org/functorch/stable/notebooks/ensembling.html

        For convenience, we break several abstraction barriers related to `FFModel`. It's possible
        to write this vectorized version without breaking them, but unfortunately it would require modifications
        to multiple files and it's honestly just easier to make the change in this manner.
        """

        # If functorch is not available, we use the regular non-vectorized version
        import torch
        try:
            from functorch import combine_state_for_ensemble, vmap
        except ImportError:
            return self.evaluate_candidate_sequences(candidate_action_sequences, obs_single)

        # Retrieve relevant objects from self
        ensemble = [m.delta_network for m in self.dyn_models]
        data_statistics = self.data_statistics
        env = self.env

        # Relevent parameters
        n_sequences, horizon = candidate_action_sequences.shape[:2]
        ac_shape = candidate_action_sequences.shape[2:]
        ob_shape = obs_single.shape
        ensemble_size = len(ensemble)

        # Get our data statistics as variables for convenience
        obs_mean = ptu.from_numpy(data_statistics["obs_mean"])
        obs_std = ptu.from_numpy(data_statistics["obs_std"])
        acs_mean = ptu.from_numpy(data_statistics["acs_mean"])
        acs_std = ptu.from_numpy(data_statistics["acs_std"])
        delta_mean = ptu.from_numpy(data_statistics["delta_mean"])
        delta_std = ptu.from_numpy(data_statistics["delta_std"])

        # Create our ensemble
        vectorized_model, params, buffers = combine_state_for_ensemble(ensemble)

        # Ensemble our initial obs
        obs_batched = torch.tile(ptu.from_numpy(obs_single), (ensemble_size, n_sequences, *(1,) * len(ob_shape)))

        # Create our pt buffers
        obs_buffer = torch.empty((horizon, ensemble_size, n_sequences, *ob_shape), dtype=torch.float32, device=ptu.device)
        acs_buffer = torch.tile(
            (ptu.from_numpy(candidate_action_sequences) - acs_mean) / (acs_std + 1e-8),
            (ensemble_size, *(1,) * len(candidate_action_sequences.shape))
        ).movedim(2, 0)
        assert acs_buffer.size() == (horizon, ensemble_size, n_sequences, *ac_shape)
        normalized_input = torch.empty((ensemble_size, n_sequences, ob_shape[0] + ac_shape[0]), dtype=torch.float32, device=ptu.device)

        for t in range(horizon):
            obs_buffer[t] = obs_batched  # Store our obs. This will fail if the shape of `obs_batched` is incorrect

            normalized_input[:, :, :ob_shape[0]] = (obs_batched - obs_mean) / (obs_std + 1e-8)
            normalized_input[:, :, ob_shape[0]:] = acs_buffer[t]

            with torch.no_grad():
                delta_predictions_normalized = vmap(vectorized_model)(params, buffers, normalized_input)

            obs_batched += delta_predictions_normalized * delta_std + delta_mean

        # Calculate rewards
        obs = ptu.to_numpy(obs_buffer).reshape(-1, *ob_shape)
        acs = ptu.to_numpy(acs_buffer).reshape(-1, *ac_shape)
        rewards_by_sequence = env.get_reward(obs, acs)[0] \
            .reshape(-1, n_sequences) \
            .sum(axis=0)

        return rewards_by_sequence

    def get_action(self, obs):
        if self.data_statistics is None:
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon x action_dim)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.N, horizon=self.horizon, obs=obs)

        if candidate_action_sequences.shape[0] == 1:
            # CEM: only a single action sequence to consider; return the first action
            return candidate_action_sequences[0][0][None]
        else:
            predicted_rewards = self.evaluate_candidate_sequences_vectorized(candidate_action_sequences, obs)

            # pick the action sequence and return the 1st element of that sequence
            best_action_sequence = candidate_action_sequences[predicted_rewards.argmax()]
            action_to_take = best_action_sequence[0]
            return action_to_take[None]  # Unsqueeze the first index

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        N = candidate_action_sequences.shape[0]
        obs_shape = obs.shape
        acs_shape = candidate_action_sequences.shape[2:]

        obs = np.tile(obs, (N, 1))

        sum_of_rewards = np.zeros((N,))
        for t in range(self.horizon):
            assert obs.shape == (N, *obs_shape)

            acs = candidate_action_sequences[:, t, :]
            assert acs.shape == (N, *acs_shape)

            rewards = self.env.get_reward(obs, acs)[0]
            assert rewards.shape == (N,)

            sum_of_rewards += rewards

            obs = model.get_prediction(obs, acs, self.data_statistics)

        # TODO (Q2)
        # For each candidate action sequence, predict a sequence of
        # states for each dynamics model in your ensemble.
        # Once you have a sequence of predicted states from each model in
        # your ensemble, calculate the sum of rewards for each sequence
        # using `self.env.get_reward(predicted_obs, action)`
        # You should sum across `self.horizon` time step.
        # Hint: you should use model.get_prediction and you shouldn't need
        #       to import pytorch in this file.
        # Hint: Remember that the model can process observations and actions
        #       in batch, which can be much faster than looping through each
        #       action sequence.
        return sum_of_rewards
