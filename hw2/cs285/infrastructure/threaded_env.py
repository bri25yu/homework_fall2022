from copy import deepcopy

import os

import numpy as np

import concurrent.futures

from gym.utils.step_api_compatibility import step_api_compatibility

from gym.vector.utils import concatenate
from gym.vector.sync_vector_env import SyncVectorEnv


class ThreadedVectorEnv(SyncVectorEnv):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.envs))

    def step_wait(self):
        """Steps through each of the environments returning the batched results.

        Returns:
            The batched environment step results
        """
        executor = self.executor

        def step(env, action):
            return step_api_compatibility(env.step(action), True)

        results = list(executor.map(step, self.envs, self._actions))

        observations, infos = [], {}
        for i, (env, result) in enumerate(zip(self.envs, results)):
            (
                observation,
                self._rewards[i],
                self._terminateds[i],
                self._truncateds[i],
                info,
            ) = result
            if self._terminateds[i] or self._truncateds[i]:
                info["final_observation"] = observation
                observation = env.reset()
            observations.append(observation)
            infos = self._add_info(infos, info, i)
        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )

        return step_api_compatibility(
            (
                deepcopy(self.observations) if self.copy else self.observations,
                np.copy(self._rewards),
                np.copy(self._terminateds),
                np.copy(self._truncateds),
                infos,
            ),
            new_step_api=self.new_step_api,
            is_vector_env=True,
        )

    def close_extras(self, **kwargs):
        """Close the environments."""
        super().close_extras(**kwargs)

        self.executor.shutdown()
