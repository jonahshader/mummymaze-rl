"""Checkpoint save/load with full training state.

New format: a directory containing:
  model.eqx           — model weights (equinox serialization)
  opt_state.eqx        — optimizer state (Adam moments, gradient clip state)
  training_state.json  — epoch, global_step, arch, hyperparams, rng key

Legacy format: a bare .eqx file (model weights only, no optimizer state).
load_checkpoint() auto-detects the format.
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
  """Load a checkpoint (directory or legacy .eqx file).

  Args:
    path: Path to checkpoint directory or bare .eqx file.
    arch: Architecture override. Required for legacy .eqx files, optional
      for directory checkpoints (where it's read from training_state.json).
    optimizer: If provided and the checkpoint has opt_state.eqx, the
      optimizer state is deserialized using this as the reference structure.
      If None, optimizer state is skipped.
  """
  if path.is_dir():
    return _load_dir_checkpoint(path, arch_override=arch, optimizer=optimizer)
  else:
    return _load_legacy_checkpoint(path, arch=arch or DEFAULT_ARCH)


def load_model_weights(path: Path, arch: str = DEFAULT_ARCH) -> eqx.Module:
  """Load just model weights from any checkpoint format. Convenience wrapper."""
  ckpt = load_checkpoint(path, arch=arch)
  return ckpt.model


def _load_dir_checkpoint(
  path: Path,
  arch_override: str | None,
  optimizer: optax.GradientTransformation | None,
) -> CheckpointData:
  """Load a directory-format checkpoint."""
  state_path = path / "training_state.json"
  state = json.loads(state_path.read_text())

  arch = arch_override or state["arch"]
  model = make_model(arch, jax.random.key(0))
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
    arch=arch,
    key=key,
    lr=state.get("lr", 3e-4),
    batch_size=state.get("batch_size", 1024),
  )


def _load_legacy_checkpoint(path: Path, arch: str) -> CheckpointData:
  """Load a bare .eqx file (weights only)."""
  model = make_model(arch, jax.random.key(0))
  model = eqx.tree_deserialise_leaves(path, model)
  return CheckpointData(
    model=model,
    opt_state=None,
    epoch=0,
    global_step=0,
    arch=arch,
    key=None,
    lr=3e-4,
    batch_size=1024,
  )
