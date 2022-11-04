"""
A suite of commonly used modeling utils.
"""
from typing import Any, Tuple

import inspect

from torch import Tensor, arange, empty_like
from torch.nn import Parameter
from torch.distributions import Distribution


__all__ = [
    "assert_shape",
    "calculate_log_probs",
    "calculate_q_values",
    "calculate_contrastive_q_values_update",
]


def _get_name_in_parent_frame(o: Any) -> str:
    # grand_parent of this frame, which is the parent of the frame that called this frame
    grand_parent = inspect.currentframe().f_back.f_back
    name_to_value = grand_parent.f_locals
    for name, value in name_to_value.items():
        if value is o:
            return name

    raise ValueError(f"Object {o} was not found in the grandparent frame")


def assert_shape(t: Tensor, expected_shape: Tuple) -> None:
    assert t.size() == expected_shape, f"Expected `{_get_name_in_parent_frame(t)}` of shape {expected_shape} but got {t.size()} instead"


def calculate_log_probs(action_distribution: Distribution, actions: Tensor) -> Tensor:
    """
    Parameters
    ----------
    action_distribution: Distribution
    actions: Tensor of shape (L, *)
        The dimensions of actions must match that of action_distribution.

    Returns
    -------
    log_probs: Tensor of shape (L, 1)

    """
    L = actions.size()[0]

    log_probs = action_distribution \
        .log_prob(actions) \
        .view(L, -1) \
        .sum(dim=-1, keepdim=True)

    assert_shape(log_probs, (L, 1))

    return log_probs


def calculate_q_values(rewards: Tensor, terminals: Tensor, gamma: float) -> Tensor:
    """
    This uses the discounted cumulative sum formulation.

    Parameters
    ----------
    rewards: Tensor of shape (L, 1)
    terminals: Tensor of shape (L, 1)
    gamma: float
        Discount factor

    Returns
    -------
    q_values: Tensor of shape (L, 1)

    """
    L = rewards.size()[0]

    assert_shape(rewards, (L, 1))
    assert_shape(terminals, (L, 1))

    mask = ~terminals

    q_values = rewards.clone().detach()
    for i in arange(L-2, -1, -1):
        q_values[i] += gamma * mask[i] * q_values[i+1]

    assert_shape(q_values, (L, 1))

    return q_values


def calculate_contrastive_q_values_update(q_values: Tensor, best_q_values: Parameter, terminals: Tensor):
    """
    Parameters
    ----------
    q_values: Tensor of shape (L, 1)
    best_q_values: Parameter of shape (max_L, 1)
    terminals: Tensor of shape (L, 1)

    Returns
    -------
    corresponding_best_q_values: Tensor of shape (L, 1)
        The best q_values corresponding to each time step in q_values
    new_best_q_values: Tensor of shape (max_L, 1)
        The updated best q_values

    """
    L = q_values.size()[0]
    max_L = best_q_values.size()[0]

    assert_shape(q_values, (L, 1))
    assert_shape(terminals, (L, 1))
    assert_shape(best_q_values, (max_L, 1))

    mask = ~terminals

    corresponding_best_q_values = empty_like(q_values)
    new_best_q_values = best_q_values.data.clone().detach()

    corresponding_index = 0
    for i in arange(L):
        corresponding_best_q_values[i] = best_q_values[corresponding_index]
        corresponding_index = (corresponding_index + 1) * mask[i]
        new_best_q_values[corresponding_index] = max(new_best_q_values[corresponding_index], q_values[i])

    assert_shape(corresponding_best_q_values, (L, 1))
    assert_shape(new_best_q_values, (max_L, 1))

    return corresponding_best_q_values, new_best_q_values
