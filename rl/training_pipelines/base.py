from typing import Any, Dict, Tuple, Union

from abc import ABC, abstractmethod

import os

from tqdm import trange

import numpy as np

import torch
from torch.optim import AdamW, Optimizer

from tensorboardX import SummaryWriter

from gym import Env

from rl import OUTPUT_DIR
from rl.infrastructure import EnvironmentInfo, ModelOutput, Trajectory, BatchTrajectoriesPyTorch, PolicyBase


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
        policy = policy.to(device="cuda", dtype=torch.float16)
        optimizer = self.setup_optimizer(policy)

        for step in trange(train_steps, desc="Training agent"):
            # Take a training step
            model_output, train_logs = self.perform_single_train_step(env, environment_info, policy)

            # Update our model
            loss = model_output.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % eval_steps == 0:
                eval_logs = self.evaluate(env, environment_info, policy)
            else:
                eval_logs = dict()

            logs = {**train_logs, **eval_logs}
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
        for _ in trange(eval_batch_size, desc="Evaluating agent"):
            trajectory = self.sample_single_trajectory(env, environment_info, policy)
            returns.append(np.sum(trajectory.rewards))

        return {
            "eval_average_total_return": np.mean(returns),
        }

    def sample_single_trajectory(self, env: Env, environment_info: EnvironmentInfo, policy: PolicyBase):
        trajectory = Trajectory.create(environment_info=environment_info)

        observation = env.reset()
        current_step = 0
        while True:
            trajectory_pt = BatchTrajectoriesPyTorch.from_trajectory(trajectory, policy.device)
            model_output: ModelOutput = policy(trajectory_pt)
            action = model_output.actions[0, -1].detach().cpu().numpy()

            next_observation, reward, terminal, _ = env.step(action)

            current_step += 1
            terminal = terminal or (current_step >= environment_info.max_trajectory_length)

            trajectory.update(current_step, observation, action, next_observation, reward, terminal)
            observation = next_observation

            if terminal:
                break

        return trajectory

    @property
    def experiment_name(self) -> str:
        return self.__class__.__name__

    @property
    def experiment_output_dir(self) -> str:
        return os.path.join(OUTPUT_DIR, self.name)

    def log_to_tensorboard(self, log: Dict[str, Any], step: int) -> None:
        if not hasattr(self, "logger"):
            self.logger = SummaryWriter(log_dir=self.experiment_output_dir)

        for key, value in log:
            self.logger.add_scalar(key, value, step)

        self.logger.flush()
