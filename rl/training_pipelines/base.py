from typing import Any, Dict, List, Tuple, Union

from abc import ABC, abstractmethod

import os

import time

from tqdm.notebook import trange, tqdm

import torch
from torch.optim import AdamW, Optimizer

from tensorboardX import SummaryWriter

from gym import Env

from rl import OUTPUT_DIR, RESULTS_DIR
from rl.infrastructure import ModelOutput, PolicyBase, Trajectory
from rl.infrastructure.pytorch_utils import TORCH_DEVICE, TORCH_FLOAT_DTYPE, to_numpy
from rl.infrastructure.visualization_utils import create_graph


class TrainingPipelineBase(ABC):
    TRAIN_STEPS: Union[None, int] = None
    EVAL_STEPS: Union[None, int] = None
    LEARNING_RATE: Union[None, float] = None

    EVAL_BATCH_SIZE = 2  # Number of trajectories worth of steps

    @abstractmethod
    def perform_single_train_step(self, env: Env, policy: PolicyBase) -> Tuple[ModelOutput, Dict[str, Any]]:
        """
        Perform a single train step. Returns the ModelOutput and the training logs
        """
        pass

    @abstractmethod
    def get_env(self) -> Env:
        pass

    @abstractmethod
    def get_policy(self, env: Env) -> PolicyBase:
        pass

    def run(self, seed=42, leave_tqdm=True) -> None:
        torch.manual_seed(seed)

        train_steps = self.TRAIN_STEPS
        eval_steps = self.EVAL_STEPS

        # Setup our environment, model, and relevant training objects
        env = self.get_env()
        policy = self.get_policy(env)
        policy = policy.to(device=TORCH_DEVICE, dtype=TORCH_FLOAT_DTYPE)
        optimizer = self.setup_optimizer(policy)
        self.setup_logging()

        for step in trange(train_steps, desc="Training agent", leave=leave_tqdm):
            # Take a training step
            self.time_train_step -= time.time()
            model_output, train_logs = self.perform_single_train_step(env, policy)
            self.time_train_step += time.time()

            # Update our model
            loss = model_output.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % eval_steps == 0:
                eval_logs = self.evaluate(env, policy)
            else:
                eval_logs = dict()

            logs = {**train_logs, **eval_logs, **(model_output.logs if model_output.logs else {})}
            self.log_to_tensorboard(logs, step)

        # Perform one final eval if the last step wasn't an eval step
        if step % eval_steps != 0:
            eval_logs = self.evaluate(env, policy)
            self.log_to_tensorboard(eval_logs, step+1)

    def benchmark(self) -> None:
        seeds, log_dirs = self.setup_benchmarking()

        for seed, log_dir in tqdm(zip(seeds, log_dirs), desc="Benchmarking", total=len(seeds)):
            self.logger = None
            self.setup_logging(log_dir)

            self.run(seed=seed, leave_tqdm=False)

            self.logger.close()

        self.finalize_benchmarking()

    def setup_benchmarking(self) -> Tuple[List[float], List[str]]:
        """
        Returns the seeds and log_dirs for benchmarking.
        """
        seeds = [41, 42, 43]  # Averaging over 3 seeds is good enough
        seed_to_log_dir = lambda s: os.path.join(self.experiment_results_dir, str(s))
        log_dirs = list(map(seed_to_log_dir, seeds))

        return seeds, log_dirs

    def finalize_benchmarking(self) -> None:
        create_graph(self)

    def setup_optimizer(self, policy: PolicyBase) -> Optimizer:
        learning_rate = self.LEARNING_RATE

        return AdamW(policy.parameters(), lr=learning_rate)

    def evaluate(self, env: Env, policy: PolicyBase) -> Dict[str, Any]:
        eval_batch_size = self.EVAL_BATCH_SIZE

        trajectory = self.record_trajectories(env, policy, eval_batch_size)
        last_terminal_index = trajectory.terminals.nonzero(as_tuple=True)[0][-1]
        n_trajectories = trajectory.terminals.sum()
        rewards_clipped = trajectory.rewards[:last_terminal_index]
        average_total_return = rewards_clipped.sum() / n_trajectories

        return {
            "return_eval": to_numpy(average_total_return),
        }

    def record_trajectories(self, env: Env, policy: PolicyBase, batch_size: int) -> Trajectory:
        max_episode_steps = env.spec.max_episode_steps
        steps = batch_size * max_episode_steps
        policy.eval()

        trajectory = Trajectory(steps)
        terminal = True  # We reset our env on the first step

        for current_step in trange(steps, desc="Stepping", leave=False):
            if terminal:
                trajectory.update_observations_from_numpy(current_step, env.reset()[0])
                current_trajectory_step = 0

            self.time_policy_forward -= time.time()
            model_output: ModelOutput = policy(trajectory)
            self.time_policy_forward += time.time()

            action = to_numpy(model_output.actions[current_step])

            self.time_env_step -= time.time()
            next_observation, reward, terminal, _, _ = env.step(action)
            self.time_env_step += time.time()

            current_trajectory_step += 1
            terminal = terminal or (current_trajectory_step >= max_episode_steps)

            trajectory.update_consequences_from_numpy(current_step, action, next_observation, reward, terminal)

        return trajectory

    @property
    def experiment_name(self) -> str:
        return self.__class__.__name__

    @property
    def experiment_output_dir(self) -> str:
        return os.path.join(OUTPUT_DIR, self.experiment_name)

    @property
    def experiment_results_dir(self) -> str:
        return os.path.join(RESULTS_DIR, self.experiment_name)

    def reset_timers(self) -> None:
        self.time_env_step = 0.0
        self.time_train_step = 0.0
        self.time_policy_forward = 0.0

    def setup_logging(self, log_dir: Union[None, str]=None) -> None:
        if log_dir is None:
            log_dir = os.path.join(self.experiment_output_dir, f"run{time.time()}")

        if not hasattr(self, "logger") or (self.logger is None):
            self.logger = SummaryWriter(log_dir=log_dir)

        self.reset_timers()

    def log_to_tensorboard(self, log: Dict[str, Any], step: int) -> None:
        log.update({
            "time_env_step": self.time_env_step / (step+1),
            "time_train_step": self.time_train_step / (step+1),
            "time_policy_forward": self.time_policy_forward / (step+1),
        })
        for key, value in log.items():
            self.logger.add_scalar(key, value, step)

        self.reset_timers()

        self.logger.flush()
