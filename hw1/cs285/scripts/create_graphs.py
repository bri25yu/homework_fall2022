from typing import List, Tuple

import os
from pathlib import Path

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.python.summary.summary_iterator import summary_iterator


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


def q1_2():
    experiments = {
        "Ant": "q1_bc_ant_Ant-v4",
        "HalfCheetah": "q1_bc_HalfCheetah_HalfCheetah-v4",
        "Hopper": "q1_bc_Hopper_Hopper-v4",
        "Walker2d": "q1_bc_Walker2d_Walker2d-v4",
    }

    print("Results for question 1.2")
    for experiment_name, experiment_prefix in experiments.items():
        summaries = load_eventfile_by_folder_prefix(experiment_prefix)

        eval_average_return = get_first_tag_simple_value(summaries, "Eval_AverageReturn")
        eval_std_return = get_first_tag_simple_value(summaries, "Eval_StdReturn")
        train_average_return = get_first_tag_simple_value(summaries, "Train_AverageReturn")

        print(experiment_name)
        print(f"\tEval_AverageReturn: {eval_average_return:.3f}")
        print(f"\tEval_StdReturn: {eval_std_return:.3f}")
        print(f"\tTrain_AverageReturn: {train_average_return:.3f}")


def q1_3():
    pass


if __name__ == "__main__":
    q1_2()
