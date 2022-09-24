from gym.vector.sync_vector_env import SyncVectorEnv


class ThreadedVectorEnv(SyncVectorEnv):
    def step_wait(self):
        """Steps through each of the environments returning the batched results.
        Returns:
            The batched environment step results
        """
        observations, infos = [], {}
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):

            (
                observation,
                self._rewards[i],
                self._terminateds[i],
                self._truncateds[i],
                info,
            ) = env.step(action)

            if self._terminateds[i] or self._truncateds[i]:
                old_observation = observation
                observation, info = env.reset()
                info["final_observation"] = old_observation
            observations.append(observation)
            infos = self._add_info(infos, info, i)
        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._terminateds),
            np.copy(self._truncateds),
            infos,
        )
