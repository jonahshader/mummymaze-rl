"""Training callbacks for logging and checkpointing.

These are simple callable types — construct them with lambdas or use the
provided factory functions for common patterns (wandb, directory checkpoints).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from src.train.checkpoint import save_checkpoint

if TYPE_CHECKING:
  from src.train.config import TrainConfig, TrainState

# step, metrics_dict
LogFn = Callable[[int, dict[str, float]], None]

# state, epoch, config
CheckpointFn = Callable[["TrainState", int, "TrainConfig"], None]


def make_wandb_log_fn() -> LogFn:
  """Create a log callback that forwards to wandb.log."""
  import wandb

  def log_fn(step: int, metrics: dict[str, float]) -> None:
    wandb.log(metrics, step=step)

  return log_fn


def make_checkpoint_fn(base_dir: Path) -> CheckpointFn:
  """Create a checkpoint callback that saves to base_dir/epoch{N:03d}/."""

  def checkpoint_fn(
    state: TrainState,
    epoch: int,
    config: TrainConfig,
  ) -> None:
    ckpt_path = base_dir / f"epoch{epoch:03d}"
    save_checkpoint(
      ckpt_path,
      state.model,
      state.opt_state,
      epoch=epoch,
      global_step=state.global_step,
      arch=config.arch,
      key=state.key,
      lr=config.lr,
      batch_size=config.batch_size,
      hparams=config.hparams,
    )

  return checkpoint_fn
