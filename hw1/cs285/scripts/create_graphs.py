from typing import Any, Dict, List, Tuple

import os
from pathlib import Path

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.python.summary.summary_iterator import summary_iterator

from itertools import chain

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


def _get_experiment_stats(experiment_prefix: str) -> Dict[str, float]:
    """
    Returns (Eval_AverageReturn, Eval_StdReturn, Initial_DataCollection_AverageReturn)
    """
    summaries = load_eventfile_by_folder_prefix(experiment_prefix)

    return {
        "Eval Average Return": get_first_tag_simple_value(summaries, "Eval_AverageReturn"),
        "Eval Return Stddev": get_first_tag_simple_value(summaries, "Eval_StdReturn"),
        "Expert Return": get_first_tag_simple_value(summaries, "Initial_DataCollection_AverageReturn")
    }


def q1_2():
    print("Results for question 1.2")

    experiments = {
        "Ant": "q1_bc_ant_Ant-v4",
        "HalfCheetah": "q1_bc_HalfCheetah_HalfCheetah-v4",
        "Hopper": "q1_bc_Hopper_Hopper-v4",
        "Walker2d": "q1_bc_Walker2d_Walker2d-v4",
    }

    data: List[Dict[str, float]] = []
    for experiment_name, experiment_prefix in experiments.items():
        stats = _get_experiment_stats(experiment_prefix)
        data.append({
            "Experiment name": experiment_name,
            **stats,
        })

    df = pd.DataFrame(data)
    print(df)


def q1_3():
    print("Results for question 1.3")

    losses = ["Huber", "L1", "MSE"]
    learning_rates = ["1e-3", "2e-3", "3e-3"]
    prefix_template = "q1_bc_Hopper_{loss}Loss_{learning_rate}_Hopper-v4"

    data: List[Dict[str, Any]] = []
    for loss in losses:
        for learning_rate in learning_rates:
            experiment_prefix = prefix_template.format(loss=loss, learning_rate=learning_rate)
            stats = _get_experiment_stats(experiment_prefix)
            data.append({
                "Loss": loss,
                "Learning rate": learning_rate,
                **stats,
            })

    df = pd.DataFrame(data)
    print(df)

    pivoted_df = df.pivot(index="Learning rate", columns="Loss", values="Eval Average Return")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivoted_df, ax=ax, annot=True, fmt=".0f")

    ax.set_title("Eval average return of hyperparameter tuning over learning rate and loss function")
    fig.tight_layout()
    fig.savefig("report_resources/q1_3.jpg")


def q2_2():
    prefixes = {
        "bc": "q1_bc",
        "dagger": "q2_dagger",
    }
    experiment_prefix_templates = {
        "Ant": "{prefix}_ant_Ant-v4",
        "HalfCheetah": "{prefix}_HalfCheetah_HalfCheetah-v4",
        "Hopper": "{prefix}_Hopper_Hopper-v4",
        "Walker2d": "{prefix}_Walker2d_Walker2d-v4",
    }

    rows, cols = 2, 2
    fig, axs = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))
    axs = iter(chain.from_iterable(axs))

    for experiment_name, experiment_prefix_template in experiment_prefix_templates.items():
        ax = next(axs)

        experiment_summaries = dict()
        for prefix_name, prefix in prefixes.items():
            experiment_prefix = experiment_prefix_template.format(prefix=prefix)
            experiment_summaries[prefix_name] = load_eventfile_by_folder_prefix(experiment_prefix)

        bc_performance = get_first_tag_simple_value(experiment_summaries["bc"], "Eval_AverageReturn")
        expert_performance = get_first_tag_simple_value(experiment_summaries["bc"], "Initial_DataCollection_AverageReturn")

        dagger_summaries = experiment_summaries["dagger"]
        dagger_eval_returns = filter_summaries_by_tag(dagger_summaries, "Eval_AverageReturn")

        dagger_steps = [r[0].step for r in dagger_eval_returns]
        dagger_returns = [r[1].simple_value for r in dagger_eval_returns]

        ax.plot(dagger_steps, dagger_returns, label="Dagger", color="blue")

        ax.axhline(bc_performance, label="BC", color="red")
        ax.axhline(expert_performance, label="Expert", color="black")

        ax.set_title(f"BC vs Dagger for the {experiment_name} environment")
        ax.set_xlabel("Dagger iterations")
        ax.set_ylabel("Eval return")
        ax.legend()

    fig.suptitle("Comparison between behavioral cloning (BC) and Dagger")
    fig.tight_layout()
    fig.savefig("report_resources/q2_2.jpg")


if __name__ == "__main__":
    # q1_2()
    # q1_3()
    q2_2()
