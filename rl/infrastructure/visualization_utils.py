from typing import Any, Dict, List, Tuple

from os.path import join

from dataclasses import dataclass

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from numpy import array, ndarray

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from rl import RESULTS_DIR


__all__ = ["create_graph", "create_comparative_graph"]


def get_summary(experiment_dir: str) -> Tuple[List[float], Dict[str, List[float]]]:
    event_accumulator = EventAccumulator(join(RESULTS_DIR, experiment_dir))
    event_accumulator.Reload()

    tags = event_accumulator.Tags()

    steps = None
    data = dict()
    for value_name in tags["scalars"]:
        if steps is None:
            scalars = event_accumulator.Scalars(value_name)
            steps = [s.step for s in scalars]

        scalars = event_accumulator.Scalars(value_name)
        data[value_name] = [s.value for s in scalars]

    return steps, data


def get_eval_returns(log_dir: str) -> Tuple[List[float], List[float]]:
    steps, summary = get_summary(log_dir)
    return steps, summary("return_eval")


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

    data = array(data)
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
    fig.savefig(join(benchmark.experiment.experiment_results_dir, "benchmark.png"))


def create_comparative_graph(experiments: List[Any]) -> None:
    benchmarks = [BenchmarkVisualizationInfo(e) for e in experiments]
    fig, ax = setup_single_graph()

    for benchmark in benchmarks:
        plot_single_benchmark(ax, benchmark)

    experiment_names = [benchmark.display_name for benchmark in benchmarks]

    ax.set_title(f"Comparison of {','.join(experiment_names)} on {benchmark.env_name}")

    finalize_graph(fig, ax, benchmark)
    fig.savefig(join(RESULTS_DIR, f"{benchmark.env_name}_{'_'.join(experiment_names)}.png"))
