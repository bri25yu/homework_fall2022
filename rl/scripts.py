from argparse import ArgumentParser
from rl.infrastructure import create_comparative_graph
from rl.experiments import available_experiments


def parse_action() -> str:
    parser = ArgumentParser()
    parser.add_argument("--action", type=str, required=True)
    return parser.parse_known_args()[0].action


def compare_experiments():
    parser = ArgumentParser()
    parser.add_argument("--experiments", "-e", type=str, nargs="+", required=True)
    experiment_names = parser.parse_known_args()[0].experiments

    experiments = [available_experiments[e]() for e in experiment_names]
    create_comparative_graph(experiments)


STR_TO_ACTIONS = {
    "compare_experiments": compare_experiments,
}


if __name__ == "__main__":
    action = parse_action()

    if action not in STR_TO_ACTIONS:
        raise ValueError(f"Action {action} not recognized")

    STR_TO_ACTIONS[action]()
