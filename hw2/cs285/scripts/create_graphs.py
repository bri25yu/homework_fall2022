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


def get_eval_averagereturns(experiment_prefix: str) -> Tuple[List[float], List[float]]:
    """
    Returns a tuple of steps and eval average returns.

    The arrays are sorted ascending in steps.
    """
    experiment_summary = load_eventfile_by_folder_prefix(experiment_prefix)

    eval_returns = filter_summaries_by_tag(experiment_summary, "Eval_AverageReturn")
    steps = [r[0].step for r in eval_returns]
    returns = [r[1].simple_value for r in eval_returns]

    steps = np.array(steps)
    returns = np.array(returns)

    sorted_idxs = steps.argsort()

    steps = steps[sorted_idxs]
    returns = returns[sorted_idxs]

    return steps, returns


def q_5_1():
    print("Creating results for exercise 5.1")

    prefix_template = "q2_pg_{batch_prefix}_{config_prefix}"
    batch_prefixes = {
        "Small batch (1000 examples)": "q1_sb",
        "Large batch (5000 examples)": "q1_lb",
    }
    config_prefixes = {
        "No reward-to-go, don't standardize advantages": "no_rtg_dsa",
        "Reward-to-go, don't standardize advantages": "rtg_dsa",
        "Reward-to-go, standardize advantages": "rtg_na",
    }

    rows, cols = 1, 2
    fig, axs = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

    for ax, (batch_name, batch_prefix) in zip(axs, batch_prefixes.items()):
        for config_name, config_prefix in config_prefixes.items():
            experiment_prefix = prefix_template.format(
                batch_prefix=batch_prefix, config_prefix=config_prefix
            )

            steps, returns = get_eval_averagereturns(experiment_prefix)

            ax.plot(steps, returns, label=config_name)

        ax.set_title(f"{batch_name} reward-to-go and advantage standardization results")
        ax.set_xlabel("Train iterations")
        ax.set_ylabel("Eval return")
        ax.legend()

    fig.suptitle("Batch sizes, reward-to-go, and advantage standardization")
    fig.tight_layout()
    fig.savefig("report_resources/q5_1.jpg")


def q_5_2():
    print("Creating results for exercise 5.2")

    batch_sizes = [10000, 5000, 2500, 1000, 500, 250, 100]
    learning_rates = [
        "1e-4", "2e-4", "3e-4", "5e-4", "8e-4", "1e-3", "2e-3", "3e-3", "5e-3", "8e-3", "1e-2", "2e-2", "3e-2"
    ]
    prefix_template = "q2_pg_q2_b{batch_size}_r{learning_rate}"

    data: List[Dict[str, Any]] = []
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            experiment_prefix = prefix_template.format(
                batch_size=batch_size, learning_rate=learning_rate
            )
            _, returns = get_eval_averagereturns(experiment_prefix)

            data.append({
                "Batch size": batch_size,
                "Learning rate": learning_rate,
                "Eval Average Return": returns[-1],
            })

    df = pd.DataFrame(data)
    print(df)

    pivoted_df = df.pivot(index="Learning rate", columns="Batch size", values="Eval Average Return")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivoted_df, ax=ax, annot=True, fmt=".0f")

    ax.set_title("Eval average return of hyperparameter tuning over learning rate and loss function")
    fig.tight_layout()
    fig.savefig("report_resources/q1_3.jpg")


if __name__ == "__main__":
    q_5_1()