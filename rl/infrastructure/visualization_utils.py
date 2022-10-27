from typing import Any, List, Tuple

import os

from dataclasses import dataclass

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.python.summary.summary_iterator import summary_iterator

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from rl import RESULTS_DIR


__all__ = ["create_graph", "create_comparative_graph"]


def load_eventfile(log_dir: str) -> List:
    # Get the eventfile_path
    eventfile_name = os.listdir(log_dir)[0]
    eventfile_path = os.path.join(log_dir, eventfile_name)

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


def get_property_and_steps(log_dir: str, property_name: str) -> Tuple[List[float], List[float]]:
    """
    Returns a tuple of steps and property values.

    The arrays are sorted ascending in steps.
    """
    experiment_summary = load_eventfile(log_dir)

    train_returns = filter_summaries_by_tag(experiment_summary, property_name)
    steps = [r[0].step for r in train_returns]
    returns = [r[1].simple_value for r in train_returns]

    steps = np.array(steps)
    returns = np.array(returns)

    sorted_idxs = steps.argsort()

    steps = steps[sorted_idxs]
    returns = returns[sorted_idxs]

    return steps, returns


def get_eval_returns(log_dir: str) -> Tuple[List[float], List[float]]:
    return get_property_and_steps(log_dir, "return_eval")


@dataclass
class BenchmarkVisualizationInfo:
    experiment: Any

    @property
    def display_name(self) -> str:
        """
        Try to parse out the algorithm name from the experiment name
        """
        experiment_name: str = self.experiment.experiment_name
        env_name = self.env_name
        remove_suffix = lambda s, suffix: s[:-len(suffix)] if s.endswith(suffix) else s

        experiment_name = remove_suffix(experiment_name, "Experiment")
        experiment_name = remove_suffix(experiment_name, env_name)
        experiment_name = remove_suffix(experiment_name, env_name.split("-")[0])
        experiment_name = remove_suffix(experiment_name, "".join(env_name.split("-")))

        return experiment_name

    @property
    def env_name(self) -> str:
        return self.experiment.get_env().spec.id

    @property
    def reward_threshold(self) -> str:
        return self.experiment.get_env().spec.reward_threshold

    @property
    def log_dirs(self) -> List[str]:
        return self.experiment.setup_benchmarking()[1]


def plot_single_benchmark(ax: Axes, benchmark: BenchmarkVisualizationInfo) -> None:
    data = []
    for log_dir in benchmark.log_dirs:
        steps, returns = get_eval_returns(log_dir)

        data.append(returns)

    data = np.array(data)
    stds = data.std(axis=0)
    means = data.mean(axis=0)

    ax.plot(steps, means, label=benchmark.display_name)
    ax.fill_between(steps, means-stds, means+stds, alpha=.1)


def setup_single_graph() -> Tuple[Figure, Axes]:
    rows, cols = 1, 1
    fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

    ax.set_xlabel("Train iterations")
    ax.set_ylabel("Eval return")

    return fig, ax


def finalize_graph(fig: Figure, ax: Axes, benchmark: BenchmarkVisualizationInfo) -> None:
    ax.axhline(benchmark.reward_threshold, label="Target reward threshold", color="red")

    ax.legend()

    fig.tight_layout()


def create_graph(experiment: Any) -> None:
    benchmark = BenchmarkVisualizationInfo(experiment)
    fig, ax = setup_single_graph()

    plot_single_benchmark(ax, benchmark)
    ax.set_title(f"Performance of {benchmark.display_name} on {benchmark.env_name} averaged over {len(benchmark.log_dirs)} seeds")

    finalize_graph(fig, ax, benchmark)
    fig.savefig(os.path.join(benchmark.experiment.experiment_results_dir, "benchmark.png"))


def create_comparative_graph(experiments: List[Any]) -> None:
    benchmarks = [BenchmarkVisualizationInfo(e) for e in experiments]
    fig, ax = setup_single_graph()

    for benchmark in benchmarks:
        plot_single_benchmark(ax, benchmark)

    experiment_names = [benchmark.display_name for benchmark in benchmarks]

    ax.set_title(f"Comparison of {','.join(experiment_names)} on {benchmark.env_name}")

    finalize_graph(fig, ax, benchmark)
    fig.savefig(os.path.join(RESULTS_DIR, f"{benchmark.env_name}_{'_'.join(experiment_names)}.png"))
