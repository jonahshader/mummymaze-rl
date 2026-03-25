"""Stop-condition combinators for training loops.

Usage:
  from src.train.stopping import stop_after, any_of
  inner_stop = any_of(stop_after(10), stop_at_step(5000))
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from src.train.config import TrainState

StopFn = Callable[["TrainState", int], bool]


def stop_after(n: int) -> StopFn:
  """Stop after n iterations (epochs or rounds)."""
  return lambda _state, i: i >= n


def stop_at_step(max_steps: int) -> StopFn:
  """Stop when global_step reaches max_steps."""
  return lambda state, _i: state.global_step >= max_steps


def any_of(*fns: StopFn) -> StopFn:
  """Stop when any condition is met."""
  return lambda state, i: any(f(state, i) for f in fns)


def all_of(*fns: StopFn) -> StopFn:
  """Stop when all conditions are met."""
  return lambda state, i: all(f(state, i) for f in fns)
