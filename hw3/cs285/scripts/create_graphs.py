from typing import Any, Dict, List, Tuple

import os
from pathlib import Path

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.python.summary.summary_iterator import summary_iterator

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


run_logs_dir = os.path.join(*Path(__file__).parts[:-3], "run_logs")

def load_eventfile_by_folder_prefix(prefix: str) -> List:
    # Find the appropriate full file name
    is_prefix = lambda s: s.startswith(prefix)
    # We take the first element by default
    full_folder_name = list(filter(is_prefix, os.listdir(run_logs_dir)))[0]

    # Get the full path of the eventfile directory
    eventfile_dir = os.path.join(run_logs_dir, full_folder_name)

    # Get the eventfile_path
    eventfile_name = os.listdir(eventfile_dir)[0]
    eventfile_path = os.path.join(eventfile_dir, eventfile_name)

    return list(summary_iterator(eventfile_path))


def filter_summaries_by_tag(summaries: List, tag: str) -> List[Tuple]:
    """
    Filters summaries for all events 
    """
    value_is_tag = lambda v: v.tag == tag
    get_value_tag_from_event = lambda e: next(filter(value_is_tag, e.summary.value), None)

    filtered = []
    for event in summaries:
        value = get_value_tag_from_event(event)
        if value is None:
            continue

        filtered.append((event, value))

    return filtered


def get_first_simple_value(summaries: List[Tuple]) -> float:
    """
    Takes in the output of `filter_summaries_by_tag`
    """
    return next(iter(summaries))[1].simple_value


def get_first_tag_simple_value(summaries: List, tag: str) -> float:
    filtered = filter_summaries_by_tag(summaries, tag)
    return get_first_simple_value(filtered)


def get_property_and_steps(experiment_prefix: str, property_name: str) -> Tuple[List[float], List[float]]:
    """
    Returns a tuple of steps and property values.

    The arrays are sorted ascending in steps.
    """
    experiment_summary = load_eventfile_by_folder_prefix(experiment_prefix)

    train_returns = filter_summaries_by_tag(experiment_summary, property_name)
    steps = [r[0].step for r in train_returns]
    returns = [r[1].simple_value for r in train_returns]

    steps = np.array(steps)
    returns = np.array(returns)

    sorted_idxs = steps.argsort()

    steps = steps[sorted_idxs]
    returns = returns[sorted_idxs]

    return steps, returns


def get_train_averagereturns(experiment_prefix: str) -> Tuple[List[float], List[float]]:
    return get_property_and_steps(experiment_prefix, "Train_AverageReturn")


def get_eval_averagereturns(experiment_prefix: str) -> Tuple[List[float], List[float]]:
    return get_property_and_steps(experiment_prefix, "Eval_AverageReturn")


def get_train_bestreturns(experiment_prefix: str) -> Tuple[List[float], List[float]]:
    return get_property_and_steps(experiment_prefix, "Train_BestReturn")


def q1():
    rows, cols = 1, 1
    fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

    experiment_prefix = "q1_MsPacman"
    steps, average_returns = get_train_averagereturns(experiment_prefix)
    ax.plot(steps, average_returns, label="Average returns")

    steps, best_returns = get_train_bestreturns(experiment_prefix)
    ax.plot(steps, best_returns, label="Best returns")

    ax.set_title(f"Deep Q network performance on MsPacman-v0 environment")
    ax.set_xlabel("Train iterations")
    ax.set_ylabel("Train return")
    ax.legend()

    fig.tight_layout()
    fig.savefig("report_resources/q1.png")


def q2():
    configs = {
        "Deep Q network (DQN)": "q2_dqn_",
        "Double Deep Q network (DDQN)": "q2_doubledqn_",
    }
    seeds = [1, 2, 3]
    prefix_template = "{config}{seed}"

    rows, cols = 1, 1
    fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

    for config_name, config in configs.items():
        data = []
        for seed in seeds:
            experiment_prefix = prefix_template.format(
                config=config, seed=seed
            )
            steps, returns = get_train_averagereturns(experiment_prefix)

            data.append(returns)

        data = np.array(data)
        stds = data.std(axis=0)
        means = data.mean(axis=0)

        ax.plot(steps, means, label=config_name)
        ax.fill_between(steps, means-stds, means+stds, alpha=.1)

    ax.set_title(f"Comparison of DQN and DDQN averaged over 3 seeds")
    ax.set_xlabel("Train iterations")
    ax.set_ylabel("Train return")
    ax.legend()

    fig.tight_layout()
    fig.savefig("report_resources/q2.png")


def q3():
    configs = {
        "Baseline (Adam + no lr scheduling)": "q3_hparam0_",
        "Learning rate scheduling (warmup ratio=0.1)": "q3_hparam1_",
        "AdamW": "q3_hparam2_",
        "AdamW + LR scheduling (warmup ratio=0.1)": "q3_hparam3_",
    }
    seeds = [1, 2, 3]
    prefix_template = "{config}{seed}"

    rows, cols = 1, 1
    fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

    for config_name, config in configs.items():
        data = []
        for seed in seeds:
            experiment_prefix = prefix_template.format(
                config=config, seed=seed
            )
            steps, returns = get_train_averagereturns(experiment_prefix)

            data.append(returns)

        data = np.array(data)
        stds = data.std(axis=0)
        means = data.mean(axis=0)

        ax.plot(steps, means, label=config_name)
        ax.fill_between(steps, means-stds, means+stds, alpha=.1)

    ax.set_title(f"Effect of learning rate scheduling and optimizer choice averaged over 3 seeds")
    ax.set_xlabel("Train iterations")
    ax.set_ylabel("Train return")
    ax.legend()

    fig.tight_layout()
    fig.savefig("report_resources/q3.png")


def q4():
    configs = {
        "# target updates = 1, # grad steps per target update = 1": "q4_ac_1_1",
        "# target updates = 100, # grad steps per target update = 1": "q4_100_1",
        "# target updates = 1, # grad steps per target update = 100": "q4_1_100",
        "# target updates = 10, # grad steps per target update = 10": "q4_10_10",
    }

    rows, cols = 1, 1
    fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

    for config_name, config_prefix in configs.items():
        steps, returns = get_eval_averagereturns(config_prefix)

        ax.plot(steps, returns, label=config_name)

    ax.set_title(f"Comparison of number of target updates and grad steps per target update")
    ax.set_xlabel("Train iterations")
    ax.set_ylabel("Eval return")
    ax.legend()

    fig.tight_layout()
    fig.savefig("report_resources/q4.png")


def q5():
    configs = {
        "HalfCheetah environment": "q5_10_10_HalfCheetah",
        "InvertedPendulum environment": "q5_10_10_InvertedPendulum",
    }

    rows, cols = 1, 2
    fig, axs = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

    for ax, (config_name, config_prefix) in zip(axs, configs.items()):
        steps, returns = get_eval_averagereturns(config_prefix)

        ax.plot(steps, returns)

        ax.set_title(config_name)
        ax.set_xlabel("Train iterations")
        ax.set_ylabel("Eval return")

    fig.suptitle(f"Validation of actor critic on different environments")
    fig.tight_layout()
    fig.savefig("report_resources/q5.png")


if __name__ == "__main__":
    q5()
