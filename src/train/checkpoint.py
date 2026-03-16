"""Checkpoint save/load with full training state.

Format: a directory containing:
  model.eqx           — model weights (equinox serialization)
  opt_state.eqx        — optimizer state (Adam moments, gradient clip state)
  training_state.json  — epoch, global_step, arch, hyperparams, rng key
"""

import json
from pathlib import Path

import equinox as eqx
import jax
import optax

from src.train.model import DEFAULT_ARCH, make_model


def save_checkpoint(
  path: Path,
  model: eqx.Module,
  opt_state: optax.OptState,
  *,
  epoch: int,
  global_step: int,
  arch: str,
  key: jax.Array,
  lr: float,
  batch_size: int,
) -> None:
  """Save a full checkpoint directory."""
  path.mkdir(parents=True, exist_ok=True)
  eqx.tree_serialise_leaves(path / "model.eqx", model)
  eqx.tree_serialise_leaves(path / "opt_state.eqx", opt_state)

  state = {
    "epoch": epoch,
    "global_step": global_step,
    "arch": arch,
    "lr": lr,
    "batch_size": batch_size,
    "rng_key": jax.random.key_data(key).tolist(),
  }
  (path / "training_state.json").write_text(json.dumps(state, indent=2))


class CheckpointData:
  """Result of loading a checkpoint."""

  __slots__ = (
    "model",
    "opt_state",
    "epoch",
    "global_step",
    "arch",
    "key",
    "lr",
    "batch_size",
  )

  def __init__(
    self,
    model: eqx.Module,
    opt_state: optax.OptState | None,
    epoch: int,
    global_step: int,
    arch: str,
    key: jax.Array | None,
    lr: float,
    batch_size: int,
  ) -> None:
    self.model = model
    self.opt_state = opt_state
    self.epoch = epoch
    self.global_step = global_step
    self.arch = arch
    self.key = key
    self.lr = lr
    self.batch_size = batch_size

  @property
  def has_optimizer(self) -> bool:
    return self.opt_state is not None


def load_checkpoint(
  path: Path,
  arch: str | None = None,
  optimizer: optax.GradientTransformation | None = None,
) -> CheckpointData:
  """Load a checkpoint directory.

  Args:
    path: Path to checkpoint directory containing model.eqx and
      training_state.json.
    arch: Architecture override. If None, read from training_state.json.
    optimizer: If provided and the checkpoint has opt_state.eqx, the
      optimizer state is deserialized using this as the reference structure.
      If None, optimizer state is skipped.
  """
  state_path = path / "training_state.json"
  state = json.loads(state_path.read_text())

  resolved_arch = arch or state.get("arch", DEFAULT_ARCH)
  model = make_model(resolved_arch, jax.random.key(0))
  model = eqx.tree_deserialise_leaves(path / "model.eqx", model)

  opt_state = None
  opt_state_path = path / "opt_state.eqx"
  if optimizer is not None and opt_state_path.exists():
    ref_opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    opt_state = eqx.tree_deserialise_leaves(opt_state_path, ref_opt_state)

  key = None
  if "rng_key" in state:
    import numpy as np

    key_data = np.array(state["rng_key"], dtype=np.uint32)
    key = jax.random.wrap_key_data(key_data)

  return CheckpointData(
    model=model,
    opt_state=opt_state,
    epoch=state.get("epoch", 0),
    global_step=state.get("global_step", 0),
    arch=resolved_arch,
    key=key,
    lr=state.get("lr", 3e-4),
    batch_size=state.get("batch_size", 1024),
  )


def load_model_weights(path: Path, arch: str | None = None) -> eqx.Module:
  """Load just model weights from a checkpoint directory."""
  ckpt = load_checkpoint(path, arch=arch)
  return ckpt.model
