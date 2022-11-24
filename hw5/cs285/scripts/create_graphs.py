from typing import List, Tuple

from glob import glob
from os.path import join
from pathlib import Path

from collections import defaultdict

import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.python.summary.summary_iterator import summary_iterator

run_logs_dir = join(*Path(__file__).parts[:-3], "run_logs")
get_path_from_prefix = lambda p: glob(f"{p}*")[0]
tf_eventfile_prefix = "events.out.tfevents"


def get_logs(experiment_prefix: str) -> DataFrame:
    eventfile_dir = get_path_from_prefix(join(run_logs_dir, experiment_prefix))
    eventfile_path = get_path_from_prefix(join(eventfile_dir, tf_eventfile_prefix))

    data = defaultdict(dict)
    for event in summary_iterator(eventfile_path):
        for v in event.summary.value:
            if not v.simple_value:
                continue
            data[event.step][v.tag] = v.simple_value

    return DataFrame(data=data.values(), index=data.keys())


def get_logs_with_return(experiment_prefix: str):
    logs = get_logs(experiment_prefix)
    return logs[logs["Eval_AverageReturn"].notna() & logs["Train_AverageReturn"].notna()]


def q1_1():
    envs = [
        {"env_value": "env1", "env_name": "PointmassEasy"},
        {"env_value": "env2", "env_name": "PointmassMedium"},
    ]
    strategies = [
        {"strategy_name": "Episilon-greedy", "strategy_value": "random"},
        {"strategy_name": "RND", "strategy_value": "rnd"},
    ]
    prefix_template = "hw5_expl_q1_{env_value}_{strategy_value}_{env_name}"

    for env in envs:
        rows, cols = 1, 3
        fig, axs = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))
        learning_curve_ax = axs[2]

        for state_density_ax, strategy in zip(axs, strategies):
            strategy_name = strategy.pop("strategy_name")
            config_prefix = prefix_template.format(**env, **strategy)
            strategy["strategy_name"] = strategy_name

            eventfile_dir = get_path_from_prefix(join(run_logs_dir, config_prefix))
            state_density_path = get_path_from_prefix(join(eventfile_dir, "curr_state_density"))
            state_density_ax.imshow(plt.imread(state_density_path))
            state_density_ax.set_title(strategy_name)
            state_density_ax.set_axis_off()

            logs = get_logs_with_return(config_prefix)
            logs.plot(y="Eval_AverageReturn", ax=learning_curve_ax, label=strategy_name)

        learning_curve_ax.set_title("Learning curves")
        learning_curve_ax.set_xlabel("Train iterations")
        learning_curve_ax.set_ylabel("Eval average return")

        fig.suptitle(f"Comparison of Epsilon-greedy and RND exploration on {env['env_name']}")
        fig.tight_layout()
        fig.savefig(f"report_resources/q1_1_{env['env_name']}.png")


def q1_2():
    envs = [
        {"env_name": "PointmassMedium"},
    ]
    strategies = [
        {"strategy_name": "RND", "config_value": "env2_rnd"},
        {"strategy_name": "RND L1", "config_value": "alg_med"},
    ]
    prefix_template = "hw5_expl_q1_{config_value}_{env_name}"

    for env in envs:
        rows, cols = 2, 3
        fig, axs = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))
        learning_curve_ax = axs[0, 2]
        fig.delaxes(axs[1][2])

        for strategy_axs, strategy in zip(axs.T, strategies):
            strategy_name = strategy.pop("strategy_name")
            config_prefix = prefix_template.format(**env, **strategy)
            strategy["strategy_name"] = strategy_name

            state_density_ax, eval_trajectory_ax = strategy_axs[:2]

            eventfile_dir = get_path_from_prefix(join(run_logs_dir, config_prefix))

            state_density_path = get_path_from_prefix(join(eventfile_dir, "curr_state_density"))
            state_density_ax.imshow(plt.imread(state_density_path))
            state_density_ax.set_title(f"State densities for {strategy_name}")
            state_density_ax.set_axis_off()

            eval_trajectory_path = get_path_from_prefix(join(eventfile_dir, "expl_last_traj"))
            eval_trajectory_ax.imshow(plt.imread(eval_trajectory_path))
            eval_trajectory_ax.set_axis_off()
            eval_trajectory_ax.set_title(f"Last exploration trajectory for {strategy_name}")

            logs = get_logs_with_return(config_prefix)
            logs.plot(y="Eval_AverageReturn", ax=learning_curve_ax, label=strategy_name)

        learning_curve_ax.set_title("Learning curves")
        learning_curve_ax.set_xlabel("Train iterations")
        learning_curve_ax.set_ylabel("Eval average return")

        fig.suptitle(f"Comparison of RND and RND L1 exploration on {env['env_name']}")
        fig.tight_layout()
        fig.savefig(f"report_resources/q1_2_{env['env_name']}.png")


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


def q3():
    envs = [
        {"env_name": "PointmassMedium", "env_value": "medium"},
        {"env_name": "PointmassHard", "env_value": "hard"},
    ]
    configs = ["dqn", "cql", "dqn_scaled", "cql_scaled"]
    prefix_template = "hw5_expl_q3_{env_value}_{config}_{env_name}"

    for env in envs:
        rows, cols = 1, 1
        fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

        for config in configs:
            config_prefix = prefix_template.format(**env, **{"config": config})

            logs = get_logs_with_return(config_prefix)
            logs.plot(y="Eval_AverageReturn", ax=ax, label=config)

        ax.set_xlabel("Train iterations")
        ax.set_ylabel("Eval average return")

        fig.suptitle(f"Learning curves for DQN vs CQL supervised exploration on {env['env_name']} environment")
        fig.tight_layout()
        fig.savefig(f"report_resources/q3_{env['env_name']}.png")


def q4():
    envs = [("easy", "PointmassEasy-v0"), ("medium", "PointmassMedium-v0")]
    lams = ["0.1", "1", "2", "10", "20", "50"]
    supervision_types = [("supervised", "Supervised exploration"), ("unsupervised", "Unsupervised exploration")]
    prefix_template = "hw5_expl_q4_awac_{env_value}_{supervision_type}_lam{lam}_{env_name}"

    for env_value, env_name in envs:
        rows, cols = 2, len(supervision_types)
        fig, axs = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))
        supervision_axs = axs.T

        for supervision_ax, (supervision_type, supervision_name) in zip(supervision_axs, supervision_types):
            learning_curve_ax, lambda_ax = supervision_ax

            lambda_scores = []
            for lam in lams:
                prefix = prefix_template.format(env_value=env_value, env_name=env_name, lam=lam, supervision_type=supervision_type)
                steps, eval_returns = get_eval_averagereturns(prefix)

                score = np.mean(eval_returns[int(len(eval_returns) * 0.9):])
                lambda_scores.append(score)

                learning_curve_ax.plot(steps, eval_returns, label=f"lambda={lam}")

            lambda_ax.plot(lams, lambda_scores)

            learning_curve_ax.set_title(supervision_name)
            learning_curve_ax.set_xlabel("Train iterations")
            learning_curve_ax.set_ylabel("Eval Average Return")
            learning_curve_ax.legend()

            lambda_ax.set_xlabel("Lambda")
            lambda_ax.set_ylabel("Score")

        fig.suptitle(f"Ablation of AWAC over lambda for {env_name} environment")
        fig.tight_layout()
        fig.savefig(f"report_resources/q4_{env_value}.png")


def q5():
    envs = [
        {"env_name": "PointmassEasy", "env_value": "easy"},
        {"env_name": "PointmassMedium", "env_value": "medium"},
    ]
    supervision_types = [
        {"supervision_type": "supervised"},
        {"supervision_type": "unsupervised"},
    ]
    taus = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    prefix_template = "hw5_expl_q5_{env_value}_{supervision_type}_lam1.0_tau{tau}_{env_name}"

    for env in envs:
        rows, cols = 2, 2
        fig, axs = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

        for supervision_axs, supervision_type in zip(axs.T, supervision_types):
            learning_curve_ax, tau_ax = supervision_axs

            tau_scores = []
            for tau in taus:
                config_prefix = prefix_template.format(**env, **supervision_type, **{"tau": tau})

                logs = get_logs_with_return(config_prefix)
                logs.plot(y="Eval_AverageReturn", ax=learning_curve_ax, label=f"tau={tau}")

                tau_scores.append(logs["Eval_AverageReturn"][int(len(logs) * 0.9):].mean())

            tau_ax.plot(taus, tau_scores)
            tau_ax.set_title(f"Ablation over Tau for {supervision_type['supervision_type']} exploration")
            tau_ax.set_xlabel("Tau")
            tau_ax.set_ylabel("Score")

            learning_curve_ax.set_title(f"Learning curves for {supervision_type['supervision_type']} exploration")
            learning_curve_ax.set_xlabel("Train iterations")
            learning_curve_ax.set_ylabel("Eval average return")

        fig.suptitle(f"Ablation of IQL over tau on {env['env_name']}")
        fig.tight_layout()
        fig.savefig(f"report_resources/q5_{env['env_name']}.png")


if __name__ == "__main__":
    q3()
