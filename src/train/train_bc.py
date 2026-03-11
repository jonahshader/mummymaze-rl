"""Behavioral cloning training script for Mummy Maze."""

import argparse
import json
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import mummymaze_rust
import numpy as np
import optax
from collections.abc import Callable
from typing import Any

from tqdm import tqdm

from jaxtyping import Array, Float, Int

from src.train.checkpoint import load_checkpoint, save_checkpoint
from src.train.dataset import BCDataset, load_bc_dataset, make_batch_obs
from src.train.model import DEFAULT_ARCH, MODEL_REGISTRY, make_model
from src.train.reporter import FileReporter, StdioReporter


def cross_entropy_loss(
  logits: Float[Array, "B 5"],
  targets: Float[Array, "B 5"],
) -> Float[Array, ""]:
  """Cross-entropy against soft targets: -sum(targets * log_softmax(logits))."""
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  return -jnp.mean(jnp.sum(targets * log_probs, axis=-1))


def top1_accuracy(
  logits: Float[Array, "B 5"],
  targets: Float[Array, "B 5"],
) -> Float[Array, ""]:
  """Fraction where argmax(logits) is an optimal action."""
  preds = jnp.argmax(logits, axis=-1)
  # An action is correct if target > 0 for that action
  correct = jnp.take_along_axis(targets, preds[:, None], axis=1).squeeze(-1)
  return jnp.mean(correct > 0)


@eqx.filter_jit
def train_step(
  model: eqx.Module,
  opt_state: optax.OptState,
  optimizer: optax.GradientTransformation,
  obs: Float[Array, "B 10 H W"],
  targets: Float[Array, "B 5"],
) -> tuple[
  eqx.Module,
  optax.OptState,
  Float[Array, ""],
  Float[Array, ""],
  Float[Array, "B 5"],
]:
  """Single training step: forward, loss, backward, update. Returns logits too."""

  def loss_fn(m: eqx.Module) -> tuple[Float[Array, ""], Float[Array, "B 5"]]:
    logits = jax.vmap(m)(obs)
    loss = cross_entropy_loss(logits, targets)
    return loss, logits

  (loss, logits), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
  updates, new_opt_state = optimizer.update(grads, opt_state, model)  # type: ignore[arg-type]
  new_model = eqx.apply_updates(model, updates)
  acc = top1_accuracy(logits, targets)
  return new_model, new_opt_state, loss, acc, logits


@eqx.filter_jit
def eval_step(
  model: eqx.Module,
  obs: Float[Array, "B 10 H W"],
  targets: Float[Array, "B 5"],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
  """Evaluation step: forward + metrics, no gradients."""
  logits = jax.vmap(model)(obs)
  loss = cross_entropy_loss(logits, targets)
  acc = top1_accuracy(logits, targets)
  return loss, acc


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


def _parse_rust_levels(
  maze_dir: Path,
  level_keys: list[tuple[str, int]],
) -> list["mummymaze_rust.Level"]:
  """Parse Level objects from .dat files for a list of (file_stem, sublevel) keys."""
  # Group by file to avoid re-parsing the same .dat file
  from collections import defaultdict

  by_file: dict[str, list[tuple[int, int]]] = defaultdict(list)
  for i, (stem, sub) in enumerate(level_keys):
    by_file[stem].append((sub, i))

  result: list[mummymaze_rust.Level | None] = [None] * len(level_keys)
  for stem, entries in by_file.items():
    levels = mummymaze_rust.parse_file(str(maze_dir / f"{stem}.dat"))
    for sub, idx in entries:
      result[idx] = levels[sub]

  return result  # type: ignore[return-value]


def train_epochs(
  model: eqx.Module,
  opt_state: optax.OptState,
  optimizer: optax.GradientTransformation,
  datasets: dict[int, BCDataset],
  sources: dict[int, list[tuple[str, int]]],
  *,
  epochs: int,
  batch_size: int,
  checkpoint_dir: Path,
  reporter: FileReporter | StdioReporter,
  maze_dir: Path,
  key: jax.Array,
  epoch_offset: int = 0,
  step_offset: int = 0,
  wandb_project: str | None = None,
  run_id: str = "bc-cnn",
  arch: str = DEFAULT_ARCH,
  lr: float = 3e-4,
) -> tuple[eqx.Module, optax.OptState, int, jax.Array]:
  """Run training epochs. Returns (model, opt_state, final_step, key)."""
  use_tqdm = isinstance(reporter, FileReporter)

  # Pre-extract train/val indices per grid_size
  train_indices: dict[int, Int[Array, "n_train"]] = {}
  val_indices: dict[int, Int[Array, "n_val"]] = {}
  for gs, ds in datasets.items():
    train_indices[gs] = jnp.where(ds.train_mask, size=int(ds.train_mask.sum()))[0]
    val_indices[gs] = jnp.where(ds.val_mask, size=int(ds.val_mask.sum()))[0]

  # JIT make_batch_obs per grid_size (traces once)
  jit_make_obs: dict[int, Callable[..., Any]] = {}
  for gs, ds in datasets.items():
    jit_make_obs[gs] = jax.jit(
      lambda tuples, lidx, _gs=gs, _bank=ds.bank: make_batch_obs(
        _gs, _bank, tuples, lidx
      )
    )

  # Pre-allocate logits buffers for accumulation
  logits_buffers: dict[int, np.ndarray] = {}
  for gs, ds in datasets.items():
    logits_buffers[gs] = np.zeros((ds.n_states, 5), dtype=np.float32)

  jitted_grid_sizes: set[int] = set()

  global_step = step_offset
  stop_requested = False

  steps_per_epoch = sum(
    int(ds.train_mask.sum()) // batch_size for ds in datasets.values()
  )
  if use_tqdm:
    total_steps = steps_per_epoch * epochs
    print(f"\nTraining for {epochs} epochs ({total_steps} steps)...")

  for epoch_rel in range(epochs):
    if stop_requested:
      break

    epoch = epoch_offset + epoch_rel
    epoch_t0 = time.time()
    epoch_losses: list[float] = []
    epoch_accs: list[float] = []
    reporter.report_epoch_start(epoch + 1, epoch_offset + epochs, steps_per_epoch)

    # Build train batches across all grid sizes for a single progress bar
    # Each entry: (grid_size, jax_indices, numpy_indices_for_scatter)
    train_batches: list[tuple[int, Int[Array, "B"], np.ndarray]] = []
    for gs in sorted(datasets):
      ds = datasets[gs]
      ti = train_indices[gs]
      n_train = ti.shape[0]
      if n_train < batch_size:
        continue
      key, shuffle_key = jr.split(key)
      perm = jr.permutation(shuffle_key, n_train)
      shuffled = ti[perm]
      shuffled_np = np.array(shuffled)
      n_batches = n_train // batch_size
      for b in range(n_batches):
        slc = slice(b * batch_size, (b + 1) * batch_size)
        train_batches.append((gs, shuffled[slc], shuffled_np[slc]))

    if use_tqdm:
      total_epochs = epoch_offset + epochs
      pbar = tqdm(train_batches, desc=f"Epoch {epoch + 1}/{total_epochs}", unit="batch")
    else:
      pbar = train_batches
    epoch_step = 0
    for gs, batch_idx, batch_idx_np in pbar:
      # Check for stop command
      cmd = reporter.check_command()
      if cmd == "stop":
        stop_requested = True
        break

      ds = datasets[gs]
      batch_tuples = ds.state_tuples[batch_idx]
      batch_targets = ds.action_targets[batch_idx]
      batch_level_idx = ds.level_idx[batch_idx]

      if gs not in jitted_grid_sizes:
        reporter.report_status(f"Jitting grid_size={gs}...")
        jitted_grid_sizes.add(gs)

      obs = jit_make_obs[gs](batch_tuples, batch_level_idx)
      model, opt_state, loss, acc, logits = train_step(
        model, opt_state, optimizer, obs, batch_targets
      )
      # Scatter logits into buffer for level metrics
      logits_buffers[gs][batch_idx_np] = np.array(logits)
      global_step += 1
      epoch_step += 1

      loss_val = float(loss)
      acc_val = float(acc)
      epoch_losses.append(loss_val)
      epoch_accs.append(acc_val)

      if use_tqdm:
        pbar.set_postfix(loss=f"{loss_val:.3f}", acc=f"{acc_val:.3f}", gs=gs)  # type: ignore[union-attr]

      reporter.report_batch(global_step, epoch_step, loss_val, acc_val, gs)

      if wandb_project is not None:
        import wandb

        wandb.log(
          {
            "train/loss": loss_val,
            "train/accuracy": acc_val,
            "train/grid_size": gs,
          },
          step=global_step,
        )

    if stop_requested:
      break

    # Validation
    reporter.report_status("Validating...")
    val_batches: list[tuple[int, Int[Array, "B"]]] = []
    for gs in sorted(datasets):
      vi = val_indices[gs]
      n_val = vi.shape[0]
      if n_val == 0:
        continue
      for start in range(0, n_val, batch_size):
        end = min(start + batch_size, n_val)
        val_batches.append((gs, vi[start:end]))

    val_losses: list[float] = []
    val_accs: list[float] = []
    if use_tqdm:
      val_iter = tqdm(val_batches, desc="  Validating", unit="batch")
    else:
      val_iter = val_batches
    for gs, batch_idx in val_iter:
      ds = datasets[gs]
      batch_tuples = ds.state_tuples[batch_idx]
      batch_targets = ds.action_targets[batch_idx]
      batch_level_idx = ds.level_idx[batch_idx]

      obs = jit_make_obs[gs](batch_tuples, batch_level_idx)
      loss, acc = eval_step(model, obs, batch_targets)
      val_losses.append(float(loss))
      val_accs.append(float(acc))

    mean_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
    mean_train_acc = float(np.mean(epoch_accs)) if epoch_accs else 0.0
    mean_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
    mean_val_acc = float(np.mean(val_accs)) if val_accs else 0.0
    epoch_time = time.time() - epoch_t0

    if use_tqdm:
      print(
        f"  train loss={mean_train_loss:.4f} acc={mean_train_acc:.4f} — "
        f"val loss={mean_val_loss:.4f} acc={mean_val_acc:.4f} ({epoch_time:.1f}s)"
      )

    reporter.report_epoch_end(
      epoch + 1,
      mean_train_loss,
      mean_train_acc,
      mean_val_loss,
      mean_val_acc,
      epoch_time,
    )

    if wandb_project is not None:
      import wandb

      wandb.log(
        {
          "epoch": epoch + 1,
          "val/loss": mean_val_loss,
          "val/accuracy": mean_val_acc,
          "train/epoch_loss": mean_train_loss,
          "train/epoch_accuracy": mean_train_acc,
        },
        step=global_step,
      )

    # Checkpoint
    ckpt_path = checkpoint_dir / f"epoch{epoch + 1:03d}"
    save_checkpoint(
      ckpt_path,
      model,
      opt_state,
      epoch=epoch + 1,
      global_step=global_step,
      arch=arch,
      key=key,
      lr=lr,
      batch_size=batch_size,
    )

    # Compute level metrics
    reporter.report_status("Computing level metrics...")
    all_metrics: dict[int, dict[int, dict[str, object]]] = {}
    total_val_states = sum(int(ds.val_mask.sum()) for ds in datasets.values())
    if use_tqdm:
      metrics_pbar = tqdm(
        total=total_val_states,
        desc="  Level metrics",
        unit="state",
      )
    else:
      metrics_pbar = None
    for gs, ds in datasets.items():
      all_metrics[gs] = compute_level_metrics(
        model,
        ds,
        batch_size,
        metrics_pbar,
        logits_buffer=logits_buffers[gs],
        jit_make_obs_fn=jit_make_obs[gs],
      )
    if metrics_pbar is not None:
      metrics_pbar.close()

    # Compute agent win% via Markov solver
    reporter.report_status("Computing agent win%...")
    from scipy.special import softmax as scipy_softmax

    for gs, ds in datasets.items():
      probs = scipy_softmax(logits_buffers[gs], axis=-1)  # (n_states, 5) f32

      level_idx_np = np.array(ds.level_idx)
      counts = np.bincount(level_idx_np, minlength=ds.n_levels)
      offsets = np.zeros(ds.n_levels + 1, dtype=np.intp)
      np.cumsum(counts, out=offsets[1:])

      level_keys = list(sources[gs])
      state_tuples_np = np.asarray(ds.state_tuples, dtype=np.int32)

      # Parse Level objects for the Rust Markov solver
      rust_levels = _parse_rust_levels(maze_dir, level_keys)

      win_probs = mummymaze_rust.policy_win_prob_batch(
        rust_levels, state_tuples_np, probs, offsets.tolist()
      )

      failed_levels: list[str] = []
      for lvl_idx, wp in enumerate(win_probs):
        if lvl_idx in all_metrics[gs]:
          if np.isnan(wp):
            stem, sub = level_keys[lvl_idx]
            failed_levels.append(f"{stem}:{sub}")
          else:
            all_metrics[gs][lvl_idx]["agent_win_prob"] = round(wp, 6)
      if failed_levels:
        msg = (
          f"WARNING: {len(failed_levels)} gs={gs} levels failed convergence: "
          f"{', '.join(failed_levels[:10])}"
          f"{'...' if len(failed_levels) > 10 else ''}"
        )
        reporter.report_log(msg)
        if use_tqdm:
          print(f"  {msg}")

    reporter.report_level_metrics(global_step, run_id, all_metrics, sources)

  return model, opt_state, global_step, key


def train(
  maze_dir: Path,
  epochs: int = 10,
  batch_size: int = 1024,
  lr: float = 3e-4,
  seed: int = 0,
  wandb_project: str | None = None,
  metrics_path: Path = Path("level_metrics.json"),
  checkpoint_dir: Path = Path("checkpoints"),
  reporter: FileReporter | StdioReporter | None = None,
  checkpoint: Path | None = None,
  augment_levels: Path | None = None,
  epoch_offset: int = 0,
  step_offset: int = 0,
  arch: str = DEFAULT_ARCH,
) -> eqx.Module:
  """Main training loop."""
  if reporter is None:
    reporter = FileReporter(metrics_path)

  use_tqdm = isinstance(reporter, FileReporter)
  key = jr.key(seed)

  # Load dataset
  if use_tqdm:
    print("Loading dataset...")
  t0 = time.time()
  datasets, sources = load_bc_dataset(maze_dir)
  if use_tqdm:
    print(f"Dataset loaded in {time.time() - t0:.1f}s")
    for gs, ds in sorted(datasets.items()):
      n_train = int(ds.train_mask.sum())
      n_val = int(ds.val_mask.sum())
      print(f"  grid_size={gs}: {ds.n_states} states ({n_train} train, {n_val} val)")

  # Augment dataset with additional levels if provided
  if augment_levels is not None:
    from src.train.augment import augment_dataset

    with open(augment_levels) as f:
      level_dicts = json.load(f)
    # JSON arrays → Python lists, but PyO3 from_dict expects tuples for
    # coordinate fields. Convert coordinate lists to tuples.
    _COORD_KEYS = {"player", "mummy1", "mummy2", "scorpion", "gate", "key"}
    for d in level_dicts:
      for k in _COORD_KEYS:
        v = d.get(k)
        if isinstance(v, list):
          d[k] = tuple(v)
      if isinstance(d.get("traps"), list):
        d["traps"] = [tuple(t) for t in d["traps"]]
    levels = [mummymaze_rust.Level.from_dict(d) for d in level_dicts]
    if use_tqdm:
      print(f"Augmenting dataset with {len(levels)} levels...")
    datasets = augment_dataset(datasets, levels)
    if use_tqdm:
      for gs, ds in sorted(datasets.items()):
        n_train = int(ds.train_mask.sum())
        print(f"  grid_size={gs}: {ds.n_states} states ({n_train} train)")

  # Initialize model (or resume from checkpoint)
  if checkpoint is not None:
    if use_tqdm:
      print(f"Loading checkpoint: {checkpoint}")
    ckpt = load_checkpoint(checkpoint, arch=arch)
    model = ckpt.model
    arch = ckpt.arch
    # Use checkpoint's training state for resume if no explicit offsets given
    if epoch_offset == 0 and ckpt.epoch > 0:
      epoch_offset = ckpt.epoch
    if step_offset == 0 and ckpt.global_step > 0:
      step_offset = ckpt.global_step
    if ckpt.key is not None:
      key = ckpt.key
  else:
    key, model_key = jr.split(key)
    model = make_model(arch, model_key)

  n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
  if use_tqdm:
    print(f"Model: {arch} ({n_params:,} parameters)")

  # Optimizer
  total_train_states = sum(int(ds.train_mask.sum()) for ds in datasets.values())
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

  # Restore optimizer state from checkpoint if available
  opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
  if checkpoint is not None:
    opt_state_path = checkpoint / "opt_state.eqx"
    if opt_state_path.exists():
      opt_state = eqx.tree_deserialise_leaves(opt_state_path, opt_state)
      if use_tqdm:
        print("Restored optimizer state from checkpoint")

  # wandb init
  run_id = f"bc-{arch}-{seed}"
  if wandb_project is not None:
    import wandb

    wandb.init(
      project=wandb_project,
      name=run_id,
      config={
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "n_params": n_params,
        "total_train_states": total_train_states,
      },
    )

  # Report init
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

  model, _opt_state, _final_step, _key = train_epochs(
    model,
    opt_state,
    optimizer,
    datasets,
    sources,
    epochs=epochs,
    batch_size=batch_size,
    checkpoint_dir=checkpoint_dir,
    reporter=reporter,
    maze_dir=maze_dir,
    key=key,
    epoch_offset=epoch_offset,
    step_offset=step_offset,
    wandb_project=wandb_project,
    run_id=run_id,
    arch=arch,
    lr=lr,
  )

  reporter.report_done()

  if wandb_project is not None:
    import wandb

    wandb.finish()

  return model


def main() -> None:
  """CLI entry point."""
  parser = argparse.ArgumentParser(description="Behavioral cloning for Mummy Maze")
  parser.add_argument(
    "--mazes",
    type=Path,
    default=Path("mazes"),
    help="Directory containing B-*.dat files",
  )
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--batch-size", type=int, default=1024)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--wandb-project", type=str, default=None)
  parser.add_argument("--metrics-path", type=Path, default=Path("level_metrics.json"))
  parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
  parser.add_argument(
    "--mode",
    choices=["standalone", "subprocess"],
    default="standalone",
    help="standalone: tqdm + file output; subprocess: JSON lines to stdout",
  )
  parser.add_argument(
    "--checkpoint",
    type=Path,
    default=None,
    help="Resume from checkpoint (directory or legacy .eqx file)",
  )
  parser.add_argument(
    "--augment-levels",
    type=Path,
    default=None,
    help="JSON file of serialized levels to augment the dataset",
  )
  parser.add_argument("--epoch-offset", type=int, default=0)
  parser.add_argument("--step-offset", type=int, default=0)
  parser.add_argument(
    "--arch",
    type=str,
    default=DEFAULT_ARCH,
    choices=sorted(MODEL_REGISTRY),
    help=f"Model architecture (default: {DEFAULT_ARCH})",
  )
  args = parser.parse_args()

  # Create reporter based on mode
  if args.mode == "subprocess":
    reporter: FileReporter | StdioReporter = StdioReporter()
  else:
    reporter = FileReporter(args.metrics_path)

  try:
    train(
      maze_dir=args.mazes,
      epochs=args.epochs,
      batch_size=args.batch_size,
      lr=args.lr,
      seed=args.seed,
      wandb_project=args.wandb_project,
      metrics_path=args.metrics_path,
      checkpoint_dir=args.checkpoint_dir,
      reporter=reporter,
      checkpoint=args.checkpoint,
      augment_levels=args.augment_levels,
      epoch_offset=args.epoch_offset,
      step_offset=args.step_offset,
      arch=args.arch,
    )
  except Exception as e:
    if args.mode == "subprocess":
      sys.stdout.write(json.dumps({"type": "error", "message": str(e)}) + "\n")
      sys.stdout.flush()
    raise


if __name__ == "__main__":
  main()
