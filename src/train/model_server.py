"""Unified model server: inference + training over a binary frame protocol.

Single long-lived process for both inference and training. Avoids JAX VRAM
competition and eliminates re-JIT compilation across GA phases and rounds.

Frame protocol (little-endian):
  [u32 length][u8 type][payload]  where length includes the type byte.

Request types (Rust -> Python):
  0x01  Evaluate       — binary level data + state tuples
  0x02  Train          — UTF-8 JSON config
  0x03  StopTrain      — empty
  0x04  ReloadCheckpoint — UTF-8 path to .eqx file
  0x05  Shutdown       — empty

Response types (Python -> Rust):
  0x81  EvaluateResult  — raw f32 action probabilities
  0x82  TrainingEvent   — UTF-8 JSON line
  0x83  Error           — UTF-8 error message

Usage:
  uv run python -m src.train.model_server --mazes mazes/ [--checkpoint path.eqx]
"""

import functools
import json
import sys
import threading
import time
from pathlib import Path
from typing import BinaryIO

import equinox as eqx
import jax
import jax.random as jr
import numpy as np
import optax
from scipy.special import softmax as scipy_softmax

from src.env.obs import observe
from src.env.types import EnvState, LevelData
from src.train.checkpoint import load_model_weights
from src.train.dataset import BCDataset, load_bc_dataset
from src.train.model import DEFAULT_ARCH, make_model
from src.train.reporter import (
  FRAME_TYPE_ERROR,
  FRAME_TYPE_TRAINING_EVENT,
  FrameReporter,
  write_frame,
)
from src.train.train_bc import train_epochs
from src.train.wire import (
  next_power_of_2,
  read_exact,
  read_level_data,
  read_u32,
  state_tuples_to_env_states,
)

# --- Frame type constants ---
REQ_EVALUATE = 0x01
REQ_TRAIN = 0x02
REQ_STOP_TRAIN = 0x03
REQ_RELOAD_CHECKPOINT = 0x04
REQ_SHUTDOWN = 0x05

RESP_EVALUATE_RESULT = 0x81

# Stderr shorthand
_stderr = sys.stderr


def _log(msg: str) -> None:
  print(f"model_server: {msg}", file=_stderr, flush=True)


def _read_frame(stream: BinaryIO) -> tuple[int, bytes]:
  """Read a length-prefixed frame. Returns (type, payload)."""
  import struct

  length = struct.unpack("<I", read_exact(stream, 4))[0]
  if length < 1:
    raise ValueError(f"Invalid frame length: {length}")
  data = read_exact(stream, length)
  return data[0], data[1:]


# --- Server ---


class ModelServer:
  """Unified model server: handles Evaluate and Train requests."""

  def __init__(
    self,
    maze_dir: Path,
    checkpoint: Path | None = None,
    arch: str = DEFAULT_ARCH,
  ) -> None:
    self.maze_dir = maze_dir
    self.arch = arch

    # Initialize model
    if checkpoint is not None and checkpoint.exists():
      _log(f"loading checkpoint: {checkpoint}")
      self.model = load_model_weights(checkpoint, arch=arch)
    else:
      _log("starting with random initialization")
      self.model = make_model(arch, jax.random.key(0))

    # JIT'd inference function
    @functools.partial(jax.jit, static_argnums=(0,))
    def _obs_and_forward(
      grid_size: int,
      level_data: LevelData,
      env_states: EnvState,
    ) -> jax.Array:
      obs = jax.vmap(lambda es: observe(grid_size, level_data, es))(env_states)
      return jax.vmap(self.model)(obs)

    self._obs_and_forward = _obs_and_forward
    self._jitted_sizes: set[int] = set()

    # Dataset cache (loaded on first Train request)
    self._datasets: dict[int, BCDataset] | None = None
    self._sources: dict[int, list[tuple[str, int]]] | None = None

    # Training stop event
    self._stop_event = threading.Event()

  def _rebind_forward(self) -> None:
    """Rebind the forward function after model weights change.

    JAX JIT cache keys on pytree structure + static args, not leaf values.
    After model weight updates (training or checkpoint reload), we need a
    new closure so JIT sees the new model leaves. The static_argnums
    (grid_size) traces are shared across closures, so grid-size JIT cache
    still persists.
    """
    model = self.model

    @functools.partial(jax.jit, static_argnums=(0,))
    def _obs_and_forward(
      grid_size: int,
      level_data: LevelData,
      env_states: EnvState,
    ) -> jax.Array:
      obs = jax.vmap(lambda es: observe(grid_size, level_data, es))(env_states)
      return jax.vmap(model)(obs)

    self._obs_and_forward = _obs_and_forward

  def _load_datasets(
    self,
  ) -> tuple[dict[int, BCDataset], dict[int, list[tuple[str, int]]]]:
    """Load or return cached datasets."""
    if self._datasets is None:
      _log("loading dataset...")
      t0 = time.time()
      self._datasets, self._sources = load_bc_dataset(self.maze_dir)
      _log(f"dataset loaded in {time.time() - t0:.1f}s")
      for gs, ds in sorted(self._datasets.items()):
        n_train = int(ds.train_mask.sum())
        n_val = int(ds.val_mask.sum())
        _log(f"  grid_size={gs}: {ds.n_states} states ({n_train} train, {n_val} val)")
    assert self._sources is not None
    return self._datasets, self._sources

  def handle_evaluate(self, stdin: BinaryIO, stdout: BinaryIO) -> None:
    """Process an Evaluate request: read level data, run inference, write results."""
    grid_size = read_u32(stdin)
    n_levels = read_u32(stdin)
    max_batch_size = read_u32(stdin)

    all_level_data: list[LevelData] = []
    all_state_tuples: list[np.ndarray] = []
    state_counts: list[int] = []

    for _ in range(n_levels):
      ld = read_level_data(stdin, grid_size)
      all_level_data.append(ld)

      n_states = read_u32(stdin)
      state_counts.append(n_states)

      if n_states > 0:
        raw = read_exact(stdin, n_states * 12 * 4)
        tuples = np.frombuffer(raw, dtype=np.int32).reshape(n_states, 12).copy()
      else:
        tuples = np.zeros((0, 12), dtype=np.int32)
      all_state_tuples.append(tuples)

    total_states = sum(state_counts)
    _log(f"evaluating {n_levels} levels ({total_states} total states)")

    # Accumulate all result bytes
    result_parts: list[bytes] = []
    states_done = 0

    for ld, st, n_st in zip(all_level_data, all_state_tuples, state_counts):
      if n_st == 0:
        continue

      chunk_size = max_batch_size if max_batch_size > 0 else n_st
      all_probs = []
      for chunk_start in range(0, n_st, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_st)
        chunk_n = chunk_end - chunk_start
        chunk_st = st[chunk_start:chunk_end]

        padded_size = next_power_of_2(chunk_n)
        if padded_size not in self._jitted_sizes:
          _log(f"JIT compiling for batch_size={padded_size}")
          self._jitted_sizes.add(padded_size)

        if padded_size > chunk_n:
          padding = np.zeros((padded_size - chunk_n, 12), dtype=np.int32)
          padded_st = np.concatenate([chunk_st, padding], axis=0)
        else:
          padded_st = chunk_st

        env_states = state_tuples_to_env_states(padded_st)
        logits = self._obs_and_forward(grid_size, ld, env_states)
        all_probs.append(np.array(logits[:chunk_n]))

      probs = scipy_softmax(
        np.concatenate(all_probs, axis=0) if len(all_probs) > 1 else all_probs[0],
        axis=-1,
      ).astype(np.float32)
      result_parts.append(probs.tobytes())

      states_done += n_st
      if states_done % 50000 < n_st:
        _log(f"{states_done}/{total_states} states")

    payload = b"".join(result_parts)
    write_frame(stdout, RESP_EVALUATE_RESULT, payload)
    _log("evaluate response sent")

  def handle_train(self, payload: bytes, stdout: BinaryIO) -> None:
    """Process a Train request: run training loop with FrameReporter."""
    config = json.loads(payload.decode("utf-8"))
    _log(f"train request: {config}")

    self._stop_event.clear()
    reporter = FrameReporter(stdout, self._stop_event)

    epochs = config.get("epochs", 5)
    batch_size = config.get("batch_size", 1024)
    lr = config.get("lr", 3e-4)
    seed = config.get("seed", 0)
    checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    epoch_offset = config.get("epoch_offset", 0)
    step_offset = config.get("step_offset", 0)
    augment_levels_path = config.get("augment_levels")
    run_id = f"bc-{self.arch}-{seed}"

    key = jr.key(seed)
    datasets, sources = self._load_datasets()

    # Augment dataset if requested
    if augment_levels_path:
      import mummymaze_rust

      from src.train.augment import augment_dataset

      aug_path = Path(augment_levels_path)
      if aug_path.exists():
        with open(aug_path) as f:
          level_dicts = json.load(f)
        _COORD_KEYS = {"player", "mummy1", "mummy2", "scorpion", "gate", "key"}
        for d in level_dicts:
          for k in _COORD_KEYS:
            v = d.get(k)
            if isinstance(v, list):
              d[k] = tuple(v)
          if isinstance(d.get("traps"), list):
            d["traps"] = [tuple(t) for t in d["traps"]]
        levels = [mummymaze_rust.Level.from_dict(d) for d in level_dicts]
        _log(f"augmenting dataset with {len(levels)} levels")
        datasets = augment_dataset(datasets, levels)

    # Optimizer
    steps_per_epoch = sum(
      int(ds.train_mask.sum()) // batch_size for ds in datasets.values()
    )
    total_steps = steps_per_epoch * epochs

    schedule = optax.warmup_cosine_decay_schedule(
      init_value=lr * 0.1,
      peak_value=lr,
      warmup_steps=min(500, total_steps // 10),
      decay_steps=total_steps,
      end_value=lr * 0.01,
    )
    optimizer = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adam(schedule),
    )
    opt_state = optimizer.init(eqx.filter(self.model, eqx.is_array))

    arrays = jax.tree.leaves(eqx.filter(self.model, eqx.is_array))
    n_params = sum(x.size for x in arrays)
    reporter.report_init(
      {
        "n_params": n_params,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "datasets": {
          str(gs): {"n_states": ds.n_states, "n_levels": ds.n_levels}
          for gs, ds in datasets.items()
        },
      }
    )

    self.model, _opt_state, _final_step, _key = train_epochs(
      self.model,
      opt_state,
      optimizer,
      datasets,
      sources,
      epochs=epochs,
      batch_size=batch_size,
      checkpoint_dir=checkpoint_dir,
      reporter=reporter,
      maze_dir=self.maze_dir,
      key=key,
      epoch_offset=epoch_offset,
      step_offset=step_offset,
      run_id=run_id,
      arch=self.arch,
      lr=lr,
    )

    reporter.report_done()
    # Rebind forward function with new model weights
    self._rebind_forward()
    _log("training complete, model weights updated in-place")

  def handle_reload_checkpoint(self, payload: bytes, stdout: BinaryIO) -> None:
    """Reload model weights from a checkpoint (directory or legacy .eqx)."""
    path = Path(payload.decode("utf-8").strip())
    _log(f"reloading checkpoint: {path}")
    if not path.exists():
      msg = f"Checkpoint not found: {path}".encode("utf-8")
      write_frame(stdout, FRAME_TYPE_ERROR, msg)
      return
    self.model = load_model_weights(path, arch=self.arch)
    self._rebind_forward()
    # Acknowledge with a training event
    ack = {"type": "status", "status": f"Reloaded checkpoint: {path}"}
    write_frame(
      stdout,
      FRAME_TYPE_TRAINING_EVENT,
      json.dumps(ack).encode("utf-8"),
    )
    _log("checkpoint reloaded")

  def handle_stop_train(self) -> None:
    """Signal the training loop to stop."""
    _log("stop_train received")
    self._stop_event.set()


def serve(
  maze_dir: Path,
  checkpoint: Path | None = None,
  arch: str = DEFAULT_ARCH,
) -> None:
  """Main serve loop: read framed requests from stdin, write responses to stdout."""
  server = ModelServer(maze_dir, checkpoint, arch=arch)

  stdin = sys.stdin.buffer
  stdout = sys.stdout.buffer

  _log("ready")

  while True:
    try:
      frame_type, payload = _read_frame(stdin)
    except EOFError:
      _log("stdin closed, shutting down")
      break

    try:
      if frame_type == REQ_EVALUATE:
        # Evaluate payload is the raw binary data after the frame header,
        # but we already consumed it. We need to re-feed from payload.
        # Actually, the evaluate request's binary data is in the payload.
        import io

        server.handle_evaluate(io.BytesIO(payload), stdout)

      elif frame_type == REQ_TRAIN:
        server.handle_train(payload, stdout)

      elif frame_type == REQ_STOP_TRAIN:
        server.handle_stop_train()

      elif frame_type == REQ_RELOAD_CHECKPOINT:
        server.handle_reload_checkpoint(payload, stdout)

      elif frame_type == REQ_SHUTDOWN:
        _log("shutdown requested")
        break

      else:
        _log(f"unknown request type: {frame_type:#x}")
        write_frame(
          stdout,
          FRAME_TYPE_ERROR,
          f"Unknown request type: {frame_type:#x}".encode("utf-8"),
        )

    except Exception as e:
      _log(f"error handling request {frame_type:#x}: {e}")
      import traceback

      traceback.print_exc(file=_stderr)
      write_frame(stdout, FRAME_TYPE_ERROR, str(e).encode("utf-8"))


def main() -> None:
  import argparse

  parser = argparse.ArgumentParser(description="Unified model server")
  parser.add_argument(
    "--mazes",
    type=Path,
    default=Path("mazes"),
    help="Directory containing B-*.dat files",
  )
  parser.add_argument(
    "--checkpoint",
    type=Path,
    default=None,
    help="Path to .eqx checkpoint (optional, uses random init if absent)",
  )
  parser.add_argument(
    "--arch",
    type=str,
    default=DEFAULT_ARCH,
    help=f"Model architecture (default: {DEFAULT_ARCH})",
  )
  args = parser.parse_args()

  serve(args.mazes, checkpoint=args.checkpoint, arch=args.arch)


if __name__ == "__main__":
  main()
