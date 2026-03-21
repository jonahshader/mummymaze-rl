"""Training configuration and state dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path

import equinox as eqx
import jax
import optax

from src.train.model import DEFAULT_ARCH


@dataclass(frozen=True)
class TrainConfig:
  """Immutable training hyperparameters and session settings."""

  epochs: int = 10
  batch_size: int = 1024
  lr: float = 3e-4
  arch: str = DEFAULT_ARCH
  hparams: dict[str, object] = field(default_factory=dict)
  level_metrics: bool = False
  max_steps: int | None = None
  run_id: str = "bc-cnn"
  maze_dir: Path = Path("mazes")


@dataclass
class TrainState:
  """Mutable training loop state."""

  model: eqx.Module
  opt_state: optax.OptState
  optimizer: optax.GradientTransformation
  key: jax.Array
  global_step: int = 0
  epoch_offset: int = 0
