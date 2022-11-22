from typing import List, Tuple

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
    tf_eventfile_prefix = "events.out.tfevents"
    eventfile_name = [p for p in os.listdir(eventfile_dir) if p.startswith(tf_eventfile_prefix)][0]
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


def get_eval_averagereturns(experiment_prefix: str) -> Tuple[List[float], List[float]]:
    return get_property_and_steps(experiment_prefix, "Eval_AverageReturn")


def q1_1():
    configs = [
        ("Episilon-greedy on PointmassEasy", "hw5_expl_q1_env1_random_PointmassEasy"),
        ("RND on PointmassEasy", "hw5_expl_q1_env1_rnd_PointmassEasy"),
        ("Episilon-greedy on PointmassMedium", "hw5_expl_q1_env2_random_PointmassMedium"),
        ("RND on PointmassMedium", "hw5_expl_q1_env2_rnd_PointmassMedium"),
    ]

    rows, cols = 1, 1
    fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

    for config_name, config_prefix in configs:
        steps, eval_returns = get_eval_averagereturns(config_prefix)

        ax.plot(steps, eval_returns, label=config_name)

    ax.set_xlabel("Train iterations")
    ax.set_ylabel("Eval average return")
    ax.legend()

    fig.suptitle("Comparison of Epsilon-greedy and RND exploration on various environments")
    fig.tight_layout()
    fig.savefig("report_resources/q1_1.png")


if __name__ == "__main__":
    q1_1()
