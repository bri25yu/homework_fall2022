from typing import Any, Dict, Tuple, Union

from abc import ABC, abstractmethod

import os

import time

from tqdm.notebook import trange

import numpy as np

import torch
from torch.optim import AdamW, Optimizer

from tensorboardX import SummaryWriter

from gym import Env

from rl import OUTPUT_DIR
from rl.infrastructure import (
    EnvironmentInfo, ModelOutput, PolicyBase, pytorch_utils, BatchTrajectory
)
from rl.infrastructure.pytorch_utils import TORCH_FLOAT_DTYPE


class TrainingPipelineBase(ABC):
    TRAIN_STEPS: Union[None, int] = None
    EVAL_STEPS: Union[None, int] = None
    LEARNING_RATE: Union[None, float] = None

    EVAL_BATCH_SIZE = 1000

    @abstractmethod
    def perform_single_train_step(self, env: Env, environment_info: EnvironmentInfo, policy: PolicyBase) -> Tuple[ModelOutput, Dict[str, Any]]:
        """
        Perform a single train step. Returns the ModelOutput and the training logs
        """
        pass

    @abstractmethod
    def get_env(self) -> Tuple[Env, EnvironmentInfo]:
        pass

    @abstractmethod
    def get_policy(self, environment_info: EnvironmentInfo) -> PolicyBase:
        pass

    def run(self) -> None:
        train_steps = self.TRAIN_STEPS
        eval_steps = self.EVAL_STEPS

        # Setup our environment, model, and relevant training objects
        env, environment_info = self.get_env()
        policy = self.get_policy(environment_info)
        policy = policy.to(device=pytorch_utils.TORCH_DEVICE, dtype=TORCH_FLOAT_DTYPE)
        optimizer = self.setup_optimizer(policy)
        self.setup_logging()

        for step in trange(train_steps, desc="Training agent"):
            # Take a training step
            self.train_step_time -= time.time()
            model_output, train_logs = self.perform_single_train_step(env, environment_info, policy)
            self.train_step_time += time.time()

            # Update our model
            loss = model_output.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % eval_steps == 0:
                eval_logs = self.evaluate(env, environment_info, policy)
            else:
                eval_logs = dict()

            logs = {**train_logs, **eval_logs, **(model_output.logs if model_output.logs else {})}
            self.log_to_tensorboard(logs, step)

        # Perform one final eval if the last step wasn't an eval step
        if step % eval_steps != 0:
            eval_logs = self.evaluate(env, environment_info, policy)
            self.log_to_tensorboard(eval_logs, step+1)

    def setup_optimizer(self, policy: PolicyBase) -> Optimizer:
        learning_rate = self.LEARNING_RATE

        return AdamW(policy.parameters(), lr=learning_rate)

    def evaluate(self, env: Env, environment_info: EnvironmentInfo, policy: PolicyBase) -> Dict[str, Any]:
        eval_batch_size = self.EVAL_BATCH_SIZE

        returns = []
        for _ in trange(eval_batch_size, desc="Evaluating agent", leave=False):
            trajectory = self.sample_single_trajectory(env, environment_info, policy)
            returns.append(torch.sum(trajectory.rewards).item())

        return {
            "eval_average_total_return": np.mean(returns),
        }

    def sample_single_trajectory(self, env: Env, environment_info: EnvironmentInfo, policy: PolicyBase) -> BatchTrajectory:
        observation, _ = env.reset()

        trajectory = BatchTrajectory.create(
            environment_info=environment_info, batch_size=1, device=pytorch_utils.TORCH_DEVICE, initial_observation=observation,
        )
        current_step = 0
        while True:
            self.policy_forward_time -= time.time()
            model_output: ModelOutput = policy(trajectory)
            self.policy_forward_time += time.time()

            action = model_output.actions[0, -1].detach().cpu().numpy()
            assert action.shape == environment_info.action_shape

            self.env_step_time -= time.time()
            next_observation, reward, terminal, _, _ = env.step(action)
            self.env_step_time += time.time()

            current_step += 1
            terminal = terminal or (current_step >= environment_info.max_trajectory_length)

            trajectory.update_from_numpy(current_step, observation, action, next_observation, reward, terminal)
            observation = next_observation

            if terminal:
                break

        trajectory.to_device("cpu")

        return trajectory

    @property
    def experiment_name(self) -> str:
        return self.__class__.__name__

    @property
    def experiment_output_dir(self) -> str:
        return os.path.join(OUTPUT_DIR, self.experiment_name)

    def reset_timers(self) -> None:
        self.env_step_time = 0.0
        self.train_step_time = 0.0
        self.policy_forward_time = 0.0

    def setup_logging(self) -> None:
        self.logger = SummaryWriter(log_dir=self.experiment_output_dir)
        self.reset_timers()

    def log_to_tensorboard(self, log: Dict[str, Any], step: int) -> None:
        log.update({
            "env_step_time": self.env_step_time,
            "train_step_time": self.train_step_time,
            "policy_forward_time": self.policy_forward_time,
        })
        for key, value in log.items():
            self.logger.add_scalar(key, value, step)

        self.reset_timers()

        self.logger.flush()
