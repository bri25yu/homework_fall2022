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


def q2():
    config_prefix = "hw4_q2_obstacles_singleiteration_obstacles-cs285-v0"

    rows, cols = 1, 1
    fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

    def scatter_and_label(steps, returns, label, expected_return):
        points = ax.scatter(steps, returns, label=label)
        for xy in zip(steps, returns):
            ax.annotate(f"({xy[0]}, {xy[1]:.1f})", xy=xy, textcoords="data")

        ax.hlines(y=expected_return, xmin=-0.3, xmax=0.3, label=f"Expected {label.lower()}", linestyles=["--"], color=points.get_facecolor())

    scatter_and_label(*get_eval_averagereturns(config_prefix), "Eval returns", -50)
    scatter_and_label(*get_train_averagereturns(config_prefix), "Train returns", -160)

    ax.set_xlabel("Train iterations")
    ax.set_ylabel("Return")
    ax.legend()

    fig.suptitle("Single iteration MPC policy performance on the obstacles environment")
    fig.tight_layout()
    fig.savefig("report_resources/q2.png")


def q3():
    configs = {
        ("Obstacles", "hw4_q3_obstacles_obstacles-cs285-v0", -20),
        ("Reacher", "hw4_q3_reacher_reacher-cs285-v0", -250),
        ("Cheetah", "hw4_q3_cheetah_cheetah-cs285-v0", 350),
    }

    rows, cols = 1, 3
    fig, axs = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

    for ax, (config_name, config_prefix, expected_return) in zip(axs, configs):
        steps, eval_returns = get_eval_averagereturns(config_prefix)

        ax.plot(steps, eval_returns)
        ax.hlines(y=expected_return, xmin=min(steps), xmax=max(steps), label="Expected eval return", color="red")
        ax.set_title(f"MBRL performance on {config_name} environment")
        ax.set_xlabel("Train iterations")
        ax.set_ylabel("Eval average return")
        ax.legend()

    fig.suptitle("Model based RL (MBRL) performance on various environments")
    fig.tight_layout()
    fig.savefig("report_resources/q3.png")


def q4():
    prefix_template = "hw4_q4_reacher_{key}{value}_reacher-cs285-v0"
    configs = [
        {
            "name": "Ensemble size",
            "key": "ensemble",
            "values": [1, 3, 5],
        },
        {
            "name": "Horizon",
            "key": "horizon",
            "values": [5, 15, 30],
        },
        {
            "name": "Num candidate sequences",
            "key": "numseq",
            "values": [100, 1000],
        },
    ]

    rows, cols = 1, 3
    fig, axs = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

    for ax, config in zip(axs, configs):
        name, key, values = config["name"], config["key"], config["values"]
        for value in values:
            config_prefix = prefix_template.format(key=key, value=value)
            steps, eval_returns = get_eval_averagereturns(config_prefix)

            ax.plot(steps, eval_returns, label=f"{name}={value}")

        ax.set_title(f"Ablation over {name.lower()}")
        ax.set_xlabel("Train iterations")
        ax.set_ylabel("Eval average return")
        ax.legend()

    fig.suptitle("Ablation of model-based RL (MBRL) performance on reacher environment")
    fig.tight_layout()
    fig.savefig("report_resources/q4.png")


def q5():
    configs = {
        "CEM 2 iterations": "hw4_q5_cheetah_cem_2_cheetah-cs285-v0",
        "CEM 4 iterations": "hw4_q5_cheetah_cem_4_cheetah-cs285-v0",
        "Random shooting": "hw4_q5_cheetah_random_cheetah-cs285-v0",
    }

    rows, cols = 1, 1
    fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

    for config_name, config_prefix in configs.items():
        steps, eval_returns = get_eval_averagereturns(config_prefix)

        ax.plot(steps, eval_returns, label=config_name)

    ax.set_xlabel("Train iterations")
    ax.set_ylabel("Eval average return")
    ax.legend()

    fig.suptitle("Comparison of sampling methods for Model based RL (MBRL) performance on cheetah environment")
    fig.tight_layout()
    fig.savefig("report_resources/q5.png")


def q6():
    configs = {
        "MBPO rollout length 0": "hw4_q6_cheetah_rlenl0_cheetah-cs285-v0",
        "MBPO rollout length 1": "hw4_q6_cheetah_rlen1_cheetah-cs285-v0",
        "MBPO rollout length 10": "hw4_q6_cheetah_rlen10_cheetah-cs285-v0",
    }

    rows, cols = 1, 1
    fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

    for config_name, config_prefix in configs.items():
        steps, eval_returns = get_eval_averagereturns(config_prefix)

        ax.plot(steps, eval_returns, label=config_name)

    ax.set_xlabel("Train iterations")
    ax.set_ylabel("Eval average return")
    ax.legend()

    fig.suptitle("Comparison of rollout lengths for model-based policy optimization (MBPO) performance on cheetah environment")
    fig.tight_layout()
    fig.savefig("report_resources/q6.png")


if __name__ == "__main__":
    q3()
