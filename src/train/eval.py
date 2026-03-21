"""Evaluation utilities: per-level metrics and Markov win-probability computation."""

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import mummymaze_rust
import numpy as np
from tqdm import tqdm

from src.train.dataset import BCDataset, make_batch_obs


def compute_level_metrics(
  model: eqx.Module,
  ds: BCDataset,
  batch_size: int,
  pbar: tqdm | None = None,
  logits_buffer: np.ndarray | None = None,
  jit_make_obs_fn: Callable[..., Any] | None = None,
) -> dict[int, dict[str, object]]:
  """Compute per-level accuracy and loss for level_metrics.json.

  If logits_buffer is provided, train logits are taken from the buffer
  and only val states need fresh inference (~10x speedup).
  """
  n = ds.n_states

  if logits_buffer is not None:
    # Use accumulated train logits from buffer, only run inference on val states
    all_logits_arr = jnp.array(logits_buffer)

    # Run inference only on val states
    val_indices = jnp.where(ds.val_mask, size=int(ds.val_mask.sum()))[0]
    n_val = val_indices.shape[0]

    for start in range(0, n_val, batch_size):
      end = min(start + batch_size, n_val)
      batch_idx = val_indices[start:end]
      batch_tuples = ds.state_tuples[batch_idx]
      batch_level_idx = ds.level_idx[batch_idx]
      if jit_make_obs_fn is not None:
        obs = jit_make_obs_fn(batch_tuples, batch_level_idx)
      else:
        obs = make_batch_obs(ds.grid_size, ds.bank, batch_tuples, batch_level_idx)
      logits = jax.vmap(model)(obs)
      all_logits_arr = all_logits_arr.at[batch_idx].set(logits)
      if pbar is not None:
        pbar.update(end - start)
  else:
    # Full inference on all states (no buffer available)
    all_logits = []
    for start in range(0, n, batch_size):
      end = min(start + batch_size, n)
      batch_tuples = ds.state_tuples[start:end]
      batch_level_idx = ds.level_idx[start:end]
      if jit_make_obs_fn is not None:
        obs = jit_make_obs_fn(batch_tuples, batch_level_idx)
      else:
        obs = make_batch_obs(ds.grid_size, ds.bank, batch_tuples, batch_level_idx)
      logits = jax.vmap(model)(obs)
      all_logits.append(logits)
      if pbar is not None:
        pbar.update(end - start)
    all_logits_arr = jnp.concatenate(all_logits, axis=0)

  # Compute per-state correctness and loss
  preds = jnp.argmax(all_logits_arr, axis=-1)
  per_state_correct = (
    jnp.take_along_axis(ds.action_targets, preds[:, None], axis=1).squeeze(-1) > 0
  ).astype(jnp.float32)

  log_probs = jax.nn.log_softmax(all_logits_arr, axis=-1)
  per_state_loss = -jnp.sum(ds.action_targets * log_probs, axis=-1)

  # Aggregate by level using bincount
  level_idx_np = np.array(ds.level_idx)
  correct_np = np.array(per_state_correct)
  loss_np = np.array(per_state_loss)

  n_levels = ds.n_levels
  counts = np.bincount(level_idx_np, minlength=n_levels)
  correct_sums = np.bincount(level_idx_np, weights=correct_np, minlength=n_levels)
  loss_sums = np.bincount(level_idx_np, weights=loss_np, minlength=n_levels)

  metrics: dict[int, dict[str, object]] = {}
  for lvl_idx in range(n_levels):
    if counts[lvl_idx] == 0:
      continue
    metrics[lvl_idx] = {
      "n_states": int(counts[lvl_idx]),
      "accuracy": round(float(correct_sums[lvl_idx] / counts[lvl_idx]), 4),
      "mean_loss": round(float(loss_sums[lvl_idx] / counts[lvl_idx]), 4),
    }

  return metrics


def parse_rust_levels(
  maze_dir: Path,
  level_keys: list[tuple[str, int]],
) -> list["mummymaze_rust.Level"]:
  """Parse Level objects from .dat files for a list of (file_stem, sublevel) keys."""
  by_file: dict[str, list[tuple[int, int]]] = defaultdict(list)
  for i, (stem, sub) in enumerate(level_keys):
    by_file[stem].append((sub, i))

  result: list[mummymaze_rust.Level | None] = [None] * len(level_keys)
  for stem, entries in by_file.items():
    levels = mummymaze_rust.parse_file(str(maze_dir / f"{stem}.dat"))
    for sub, idx in entries:
      result[idx] = levels[sub]

  return result  # type: ignore[return-value]


def compute_markov_win_probs(
  model: eqx.Module,
  datasets: dict[int, BCDataset],
  sources: dict[int, list[tuple[str, int]]],
  maze_dir: Path,
  batch_size: int,
  jit_make_obs: dict[int, Callable[..., Any]],
  reporter_log: Callable[[str], None] | None = None,
  use_tqdm: bool = False,
) -> tuple[list[float], dict[int, dict[int, dict[str, object]]]]:
  """Compute agent win% via Markov solver on validation levels.

  Returns (all_win_probs, metrics_updates) where metrics_updates maps
  grid_size -> level_idx -> {"agent_win_prob": float}.
  """
  all_win_probs: list[float] = []
  metrics_updates: dict[int, dict[int, dict[str, object]]] = {}

  for gs, ds in datasets.items():
    level_idx_np = np.array(ds.level_idx)
    val_mask_np = np.array(ds.val_mask)

    val_level_set = set(int(x) for x in level_idx_np[val_mask_np])
    if not val_level_set:
      continue

    # Run fresh inference on val states for policy probs
    val_indices_arr = jnp.where(ds.val_mask, size=int(ds.val_mask.sum()))[0]
    val_logits_list = []
    for start in range(0, val_indices_arr.shape[0], batch_size):
      end = min(start + batch_size, val_indices_arr.shape[0])
      batch_idx = val_indices_arr[start:end]
      obs = jit_make_obs[gs](ds.state_tuples[batch_idx], ds.level_idx[batch_idx])
      val_logits_list.append(np.array(jax.vmap(model)(obs)))
    val_logits = np.concatenate(val_logits_list, axis=0)
    val_probs = np.exp(val_logits - val_logits.max(axis=-1, keepdims=True))
    val_probs /= val_probs.sum(axis=-1, keepdims=True)

    val_state_tuples = np.asarray(ds.state_tuples, dtype=np.int32)[val_mask_np]
    val_level_idx = level_idx_np[val_mask_np]

    # Remap level indices to dense range for the batch call
    val_levels_sorted = sorted(val_level_set)
    old_to_new = {old: new for new, old in enumerate(val_levels_sorted)}
    remapped_idx = np.array([old_to_new[int(x)] for x in val_level_idx])

    n_val_levels = len(val_levels_sorted)
    counts = np.bincount(remapped_idx, minlength=n_val_levels)
    offsets = np.zeros(n_val_levels + 1, dtype=np.intp)
    np.cumsum(counts, out=offsets[1:])

    # Sort states by remapped level index (required by batch solver)
    sort_order = np.argsort(remapped_idx, kind="stable")
    val_state_tuples = val_state_tuples[sort_order]
    val_probs = val_probs[sort_order]

    level_keys = list(sources[gs])
    val_keys = [level_keys[i] for i in val_levels_sorted]
    rust_levels = parse_rust_levels(maze_dir, val_keys)

    win_probs = mummymaze_rust.policy_win_prob_batch(
      rust_levels, val_state_tuples, val_probs, offsets.tolist()
    )

    gs_updates: dict[int, dict[str, object]] = {}
    failed_levels: list[str] = []
    for new_idx, wp in enumerate(win_probs):
      old_idx = val_levels_sorted[new_idx]
      if np.isnan(wp):
        stem, sub = level_keys[old_idx]
        failed_levels.append(f"{stem}:{sub}")
      else:
        all_win_probs.append(wp)
        gs_updates[old_idx] = {"agent_win_prob": round(wp, 6)}
    if failed_levels:
      msg = (
        f"WARNING: {len(failed_levels)} gs={gs} levels failed convergence: "
        f"{', '.join(failed_levels[:10])}"
        f"{'...' if len(failed_levels) > 10 else ''}"
      )
      if reporter_log is not None:
        reporter_log(msg)
      if use_tqdm:
        print(f"  {msg}")
    if gs_updates:
      metrics_updates[gs] = gs_updates

  return all_win_probs, metrics_updates
