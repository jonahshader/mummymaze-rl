"""Model server: inference + training, decoupled from any transport.

Holds the JAX model in-process and exposes methods for evaluation,
training, checkpoint reload, etc. Transport (WebSocket, CLI) is handled
by callers.
"""

import functools
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

import equinox as eqx
import jax
import jax.random as jr
import mummymaze_rust
import numpy as np
from scipy.special import softmax as scipy_softmax

from src.env.obs import observe
from src.env.types import EnvState, LevelData
from src.train.callbacks import make_checkpoint_fn
from src.train.checkpoint import load_checkpoint, load_model_weights
from src.train.config import TrainConfig, TrainState
from src.train.dataset import BCDataset, load_bc_dataset
from src.train.ga import level_to_level_data
from src.train.model import DEFAULT_ARCH, make_model
from src.train.optim import count_params, make_optimizer
from src.train.reporter import MetricsReporter
from src.train.train_bc import train_epochs
from src.train.wire import next_power_of_2, state_tuples_to_env_states

log = logging.getLogger(__name__)


class ModelServer:
  """In-process model server for inference and training.

  Holds a JAX model, dataset cache, and provides transport-agnostic
  methods that the WebSocket server (or CLI) can call.
  """

  def __init__(
    self,
    maze_dir: Path,
    checkpoint: Path | None = None,
    arch: str = DEFAULT_ARCH,
    hparams: dict[str, object] | None = None,
  ) -> None:
    self.maze_dir = maze_dir
    self.arch = arch
    self.hparams: dict[str, object] = hparams or {}

    # Initialize model
    if checkpoint is not None and checkpoint.exists():
      log.info("loading checkpoint: %s", checkpoint)
      ckpt = load_checkpoint(checkpoint, arch=arch, hparams=hparams)
      self.model = ckpt.model
      self.arch = ckpt.arch
      self.hparams = ckpt.hparams
    else:
      log.info("starting with random initialization")
      self.model = make_model(arch, jax.random.key(0), **self.hparams)

    self._obs_and_forward = self._make_forward(self.model)

    # Dataset cache (loaded lazily)
    self._datasets: dict[int, BCDataset] | None = None
    self._sources: dict[int, list[tuple[str, int]]] | None = None
    # Reverse lookup: "B-5:23" → (grid_size, bank_idx)
    self._level_index: dict[str, tuple[int, int]] | None = None
    # Cached LevelData per (stem, sub) — avoids re-parsing .dat files
    self._level_data_cache: dict[tuple[str, int], tuple[int, LevelData]] = {}

    # Training stop event
    self._stop_event = threading.Event()

  @property
  def obs_and_forward(self) -> Callable:
    """Public access to the JIT'd inference closure for GA/adversarial loops."""
    return self._obs_and_forward

  @staticmethod
  def _make_forward(model: eqx.Module) -> Callable:
    """Create a JIT'd inference closure capturing the given model weights."""

    @functools.partial(jax.jit, static_argnums=(0,))
    def obs_and_forward(
      grid_size: int,
      level_data: LevelData,
      env_states: EnvState,
    ) -> jax.Array:
      obs = jax.vmap(lambda es: observe(grid_size, level_data, es))(env_states)
      return jax.vmap(model)(obs)

    return obs_and_forward

  def _rebind_forward(self) -> None:
    """Rebind the forward function after model weights change.

    JAX JIT cache keys on pytree structure + static args, not leaf values.
    After model weight updates, we need a new closure so JIT sees the new
    leaves. The XLA kernels (keyed on grid_size) persist across closures.
    """
    self._obs_and_forward = self._make_forward(self.model)

  def load_datasets(
    self,
  ) -> tuple[dict[int, BCDataset], dict[int, list[tuple[str, int]]]]:
    """Load or return cached datasets."""
    if self._datasets is None:
      log.info("loading dataset...")
      t0 = time.time()
      self._datasets, self._sources = load_bc_dataset(self.maze_dir)
      log.info("dataset loaded in %.1fs", time.time() - t0)
      for gs, ds in sorted(self._datasets.items()):
        n_train = int(ds.train_mask.sum())
        n_val = int(ds.val_mask.sum())
        log.info(
          "  grid_size=%d: %d states (%d train, %d val)",
          gs,
          ds.n_states,
          n_train,
          n_val,
        )
      # Build reverse index and cache numpy level_idx arrays
      self._level_index = {}
      self._level_idx_np: dict[int, np.ndarray] = {}
      for gs, src_list in self._sources.items():
        self._level_idx_np[gs] = np.array(self._datasets[gs].level_idx)
        for bank_idx, (stem, sub) in enumerate(src_list):
          self._level_index[f"{stem}:{sub}"] = (gs, bank_idx)
    assert self._sources is not None
    return self._datasets, self._sources

  def _get_level_data(self, stem: str, sub: int) -> tuple[int, LevelData]:
    """Get LevelData for a level, caching parsed results."""
    cache_key = (stem, sub)
    if cache_key not in self._level_data_cache:
      levels = mummymaze_rust.parse_file(str(self.maze_dir / f"{stem}.dat"))
      # Cache all sublevels from this file at once
      for i, lev in enumerate(levels):
        if (stem, i) not in self._level_data_cache:
          self._level_data_cache[(stem, i)] = level_to_level_data(lev)
    return self._level_data_cache[cache_key]

  def evaluate_level(
    self,
    level_key: str,
    *,
    max_batch_size: int = 4096,
  ) -> list[dict]:
    """Run inference on all states for a level, looked up by key.

    Args:
      level_key: e.g. "B-5:23"
      max_batch_size: Max states per inference chunk.

    Returns:
      List of {"state": [12 ints], "probs": [5 floats]} per state.
    """
    datasets, sources = self.load_datasets()
    assert self._level_index is not None

    if level_key not in self._level_index:
      raise KeyError(f"Unknown level: {level_key}")

    gs, bank_idx = self._level_index[level_key]
    ds = datasets[gs]

    # Find state indices belonging to this level
    state_indices = np.where(self._level_idx_np[gs] == bank_idx)[0]
    n_states = len(state_indices)

    if n_states == 0:
      return []

    state_tuples_np = np.asarray(ds.state_tuples[state_indices], dtype=np.int32)

    # Look up level data (cached)
    stem, sub = sources[gs][bank_idx]
    _, ld = self._get_level_data(stem, sub)

    # Run inference in chunks
    all_probs = []
    for chunk_start in range(0, n_states, max_batch_size):
      chunk_end = min(chunk_start + max_batch_size, n_states)
      chunk_n = chunk_end - chunk_start
      chunk_st = state_tuples_np[chunk_start:chunk_end]

      padded_size = next_power_of_2(chunk_n)
      if padded_size > chunk_n:
        padding = np.zeros((padded_size - chunk_n, 12), dtype=np.int32)
        padded_st = np.concatenate([chunk_st, padding], axis=0)
      else:
        padded_st = chunk_st

      env_states = state_tuples_to_env_states(padded_st)
      logits = self._obs_and_forward(gs, ld, env_states)
      all_probs.append(np.array(logits[:chunk_n]))

    probs = scipy_softmax(
      np.concatenate(all_probs, axis=0) if len(all_probs) > 1 else all_probs[0],
      axis=-1,
    ).astype(np.float32)

    return [
      {
        "state": state_tuples_np[i].tolist(),
        "probs": probs[i].tolist(),
      }
      for i in range(n_states)
    ]

  def train(
    self,
    reporter: MetricsReporter,
    *,
    epochs: int = 5,
    batch_size: int = 1024,
    lr: float = 3e-4,
    seed: int = 0,
    checkpoint_dir: Path = Path("checkpoints"),
    epoch_offset: int = 0,
    step_offset: int = 0,
    augment_levels_path: str | None = None,
  ) -> None:
    """Run training loop with the given reporter for progress events."""
    self._stop_event.clear()
    run_id = f"bc-{self.arch}-{seed}"
    key = jr.key(seed)
    datasets, sources = self.load_datasets()

    # Augment dataset if requested
    if augment_levels_path:
      from src.train.augment import augment_dataset, load_augment_levels

      aug_path = Path(augment_levels_path)
      if aug_path.exists():
        levels = load_augment_levels(aug_path)
        log.info("augmenting dataset with %d levels", len(levels))
        datasets = augment_dataset(datasets, levels)

    # Optimizer
    steps_per_epoch = sum(
      int(ds.train_mask.sum()) // batch_size for ds in datasets.values()
    )
    total_steps = steps_per_epoch * epochs

    optimizer = make_optimizer(lr, total_steps)
    opt_state = optimizer.init(eqx.filter(self.model, eqx.is_array))

    n_params = count_params(self.model)
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

    config = TrainConfig(
      epochs=epochs,
      batch_size=batch_size,
      lr=lr,
      arch=self.arch,
      hparams=self.hparams,
      level_metrics=True,
      run_id=run_id,
      maze_dir=self.maze_dir,
    )
    train_state = TrainState(
      model=self.model,
      opt_state=opt_state,
      optimizer=optimizer,
      key=key,
      global_step=step_offset,
      epoch_offset=epoch_offset,
    )
    train_state = train_epochs(
      train_state,
      config,
      datasets,
      sources,
      reporter,
      checkpoint_fn=make_checkpoint_fn(checkpoint_dir),
    )
    self.model = train_state.model

    reporter.report_done()
    self._rebind_forward()
    log.info("training complete, model weights updated")

  def reload_checkpoint(self, path: str | Path) -> None:
    """Reload model weights from a checkpoint directory."""
    path = Path(path)
    if not path.exists():
      raise FileNotFoundError(f"Checkpoint not found: {path}")
    self.model = load_model_weights(path, arch=self.arch)
    self._rebind_forward()
    log.info("checkpoint reloaded: %s", path)

  def list_checkpoints(self, checkpoint_dir: Path = Path("checkpoints")) -> list[str]:
    """List available checkpoint directories."""
    if not checkpoint_dir.exists():
      return []
    return sorted(
      str(p)
      for p in checkpoint_dir.iterdir()
      if p.is_dir() and (p / "model.eqx").exists()
    )

  def stop_train(self) -> None:
    """Signal the training loop to stop after the current batch."""
    log.info("stop_train requested")
    self._stop_event.set()
