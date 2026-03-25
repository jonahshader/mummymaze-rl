"""Unified training loop: handles both plain BC and adversarial training.

BC mode:   training_loop(session, inner_stop=stop_after(10))
Adversarial: training_loop(session, inner_stop=..., outer_stop=stop_after(3),
                           on_round_end=make_ga_round_end(...))
"""

import itertools
from collections.abc import Callable
from typing import Any

from src.train.augment import augment_dataset
from src.train.callbacks import CheckpointFn, make_checkpoint_fn
from src.train.config import StopFn, TrainState
from src.train.session import TrainingSession
from src.train.stopping import stop_after
from src.train.train_bc import train_epochs

# (state, datasets, sources, round_idx) -> list[Level] | None
RoundEndFn = Callable[[TrainState, dict, dict, int], list[Any] | None]


def training_loop(
  session: TrainingSession,
  *,
  inner_stop: StopFn | None = None,
  outer_stop: StopFn | None = None,
  on_round_end: RoundEndFn | None = None,
  on_event: Callable[[dict], None] | None = None,
  round_checkpoint_dir: Callable[[int], str | None] | None = None,
) -> TrainState:
  """Run training with optional outer (adversarial) loop.

  Args:
    session: Fully initialized TrainingSession from setup_training().
    inner_stop: When to stop each inner training round. Defaults to
      stop_after(config.epochs).
    outer_stop: When to stop the outer loop. None = single round (BC mode).
    on_round_end: Called after each round. Returns new levels to augment
      the dataset, or None. Typical use: GA level generation.
    on_event: Optional callback for structured events (round_start, etc.).
    round_checkpoint_dir: Given round index, returns checkpoint dir for that
      round (or None to skip). Overrides session.checkpoint_fn per-round.
  """
  if outer_stop is None:
    outer_stop = stop_after(1)
  if inner_stop is None:
    inner_stop = stop_after(session.config.epochs)

  state = session.state
  datasets = session.datasets
  sources = session.sources
  global_epoch = state.epoch_offset

  def _emit(event: dict) -> None:
    if on_event is not None:
      on_event(event)

  for round_idx in itertools.count():
    if outer_stop(state, round_idx):
      break

    _emit({"type": "round_start", "round": round_idx})

    # Per-round checkpoint callback
    checkpoint_fn: CheckpointFn | None = session.checkpoint_fn
    if round_checkpoint_dir is not None:
      ckpt_dir = round_checkpoint_dir(round_idx)
      checkpoint_fn = make_checkpoint_fn(ckpt_dir) if ckpt_dir else None

    # Inner training
    state.epoch_offset = global_epoch
    state = train_epochs(
      state,
      session.config,
      datasets,
      sources,
      session.reporter,
      log_fn=session.log_fn,
      checkpoint_fn=checkpoint_fn,
      components=session.components,
      inner_stop=inner_stop,
    )
    global_epoch += session.config.epochs

    # Round-end callback (GA phase for adversarial)
    if on_round_end is not None:
      new_levels = on_round_end(state, datasets, sources, round_idx)
      if new_levels:
        datasets = augment_dataset(datasets, new_levels)

    _emit({"type": "round_end", "round": round_idx})

  session.state = state
  session.datasets = datasets
  return state
