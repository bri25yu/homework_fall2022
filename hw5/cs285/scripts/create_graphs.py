from typing import List, Tuple

from itertools import product

import numpy as np
import pandas as pd

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


def q1_2():
    configs = [
        ("RND", "hw5_expl_q1_env2_rnd_PointmassMedium"),
        ("RND L1", "hw5_expl_q1_alg_med_PointmassMedium"),
    ]

    rows, cols = 1, 1
    fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

    for config_name, config_prefix in configs:
        steps, eval_returns = get_eval_averagereturns(config_prefix)

        ax.plot(steps, eval_returns, label=config_name)

    ax.set_xlabel("Train iterations")
    ax.set_ylabel("Eval average return")
    ax.legend()

    fig.suptitle("Comparison of RND and RND L1 exploration on PointmassMedium environment")
    fig.tight_layout()
    fig.savefig("report_resources/q1_2.png")


def q2_1():
    configs = [
        ("DQN", "hw5_expl_q2_dqn_PointmassMedium"),
        ("DQN shifted and scaled", "hw5_expl_q2_dqn_scaled_PointmassMedium"),
        ("CQL", "hw5_expl_q2_cql_PointmassMedium"),
    ]
    properties = ["Eval_AverageReturn", "Exploitation_Data_q-values", "Exploitation_OOD_q-values"]

    rows, cols = 1, len(properties)
    fig, axs = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

    for property_to_plot, ax in zip(properties, axs):
        for config_name, config_prefix in configs:
            steps, eval_returns = get_property_and_steps(config_prefix, property_to_plot)

            ax.plot(steps, eval_returns, label=config_name)

        ax.set_title(property_to_plot.replace("_", " "))
        ax.set_xlabel("Train iterations")
        ax.legend()

    fig.suptitle("Comparison of DQN and CQL offline learning on exploration data on PointmassMedium environment")
    fig.tight_layout()
    fig.savefig("report_resources/q2_1.png")


def q2_2():
    generic_prefix = "hw5_expl_q2"
    env_name = "PointmassMedium"
    configs = [
        ("CQL 5k exploration steps", "cql_numsteps_5000"),
        ("CQL 10k exploration steps", "cql"),
        ("CQL 15k exploration steps", "cql_numsteps_15000"),

        ("DQN 5k exploration steps", "dqn_numsteps_5000"),
        ("DQN 10k exploration steps", "dqn"),
        ("DQN 15k exploration steps", "dqn_numsteps_15000"),

        ("DQN shifted and scaled 5k exploration steps", "dqn_scaled_numsteps_5000"),
        ("DQN shifted and scaled 10k exploration steps", "dqn_scaled"),
        ("DQN shifted and scaled 15k exploration steps", "dqn_scaled_numsteps_15000"),
    ]
    properties = ["Eval_AverageReturn", "Exploitation_Data_q-values", "Exploitation_OOD_q-values"]

    rows, cols = 1, len(properties)
    fig, axs = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

    for property_to_plot, ax in zip(properties, axs):
        for config_name, config_prefix in configs:
            config_prefix = f"{generic_prefix}_{config_prefix}_{env_name}"
            steps, eval_returns = get_property_and_steps(config_prefix, property_to_plot)

            ax.plot(steps, eval_returns, label=config_name)

        ax.set_title(property_to_plot.replace("_", " "))
        ax.set_xlabel("Train iterations")
        ax.legend()

    fig.suptitle("Effect of num exploration steps on DQN and CQL")
    fig.tight_layout()
    fig.savefig("report_resources/q2_2.png")


def q2_3():
    configs = [
        ("alpha=0.02", "hw5_expl_q2_alpha0.02_PointmassMedium"),
        ("alpha=0.1", "hw5_expl_q2_cql_PointmassMedium"),
        ("alpha=0.5", "hw5_expl_q2_alpha0.5_PointmassMedium"),
    ]
    properties = ["Eval_AverageReturn", "Exploitation_Data_q-values", "Exploitation_OOD_q-values"]

    rows, cols = 1, len(properties)
    fig, axs = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

    for property_to_plot, ax in zip(properties, axs):
        for config_name, config_prefix in configs:
            steps, eval_returns = get_property_and_steps(config_prefix, property_to_plot)

            ax.plot(steps, eval_returns, label=config_name)

        ax.set_title(property_to_plot.replace("_", " "))
        ax.set_xlabel("Train iterations")
        ax.legend()

    fig.suptitle("Ablation of CQL over alpha")
    fig.tight_layout()
    fig.savefig("report_resources/q2_3.png")


def q4():
    envs = [("easy", "PointmassEasy-v0"), ("medium", "PointmassMedium-v0")]
    lams = ["0.1", "1", "2", "10", "20", "50"]
    supervision_types = ["supervised", "unsupervised"]
    prefix_template = "hw5_expl_q4_awac_{env0}_{supervision_type}_lam{lam}_{env1}"

    rows, cols = 3, len(envs)
    fig, axs = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

    for env_axs, (env0, env1) in zip(axs.T, envs):
        learning_curve_axs = env_axs[:-1]
        heatmap_ax = env_axs[-1]

        df = pd.DataFrame(columns=["AWAC lambda", "Supervision type", "score"])
        for lam, (learning_curve_ax, supervision_type) in product(lams, zip(learning_curve_axs, supervision_types)):
            prefix = prefix_template.format(env0=env0, env1=env1, lam=lam, supervision_type=supervision_type)
            steps, eval_returns = get_eval_averagereturns(prefix)

            score = np.mean(eval_returns[int(len(eval_returns) * 0.9):])
            df = df.append({"AWAC lambda": float(lam), "Supervision type": supervision_type, "score": score}, ignore_index=True)

            learning_curve_ax.plot(steps, eval_returns, label=f"lambda={lam}")

        df = df.pivot(index="AWAC lambda", columns="Supervision type", values="score")
        sns.heatmap(df, ax=heatmap_ax)

        for learning_curve_ax, supervision_type in zip(learning_curve_axs, supervision_types):
            learning_curve_ax.set_title(f"{env1} environment, {supervision_type}")
            learning_curve_ax.set_xlabel("Train iterations")
            learning_curve_ax.set_ylabel("Eval Average Return")
            learning_curve_ax.legend()

    fig.suptitle("Ablation of AWAC over lambda")
    fig.tight_layout()
    fig.savefig("report_resources/q4.png")


if __name__ == "__main__":
    q4()
