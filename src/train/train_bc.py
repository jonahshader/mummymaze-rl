"""Behavioral cloning training script for Mummy Maze."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from collections.abc import Callable
from typing import Any

from jaxtyping import Array, Float, Int

from src.train.dataset import BCDataset, load_bc_dataset, make_batch_obs
from src.train.model import MazeCNN


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
  model: MazeCNN,
  opt_state: optax.OptState,
  optimizer: optax.GradientTransformation,
  obs: Float[Array, "B 10 H W"],
  targets: Float[Array, "B 5"],
) -> tuple[MazeCNN, optax.OptState, Float[Array, ""], Float[Array, ""]]:
  """Single training step: forward, loss, backward, update."""

  def loss_fn(m: MazeCNN) -> tuple[Float[Array, ""], Float[Array, "B 5"]]:
    logits = jax.vmap(m)(obs)
    loss = cross_entropy_loss(logits, targets)
    return loss, logits

  (loss, logits), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
  updates, new_opt_state = optimizer.update(grads, opt_state, model)  # type: ignore[arg-type]
  new_model = eqx.apply_updates(model, updates)
  acc = top1_accuracy(logits, targets)
  return new_model, new_opt_state, loss, acc


@eqx.filter_jit
def eval_step(
  model: MazeCNN,
  obs: Float[Array, "B 10 H W"],
  targets: Float[Array, "B 5"],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
  """Evaluation step: forward + metrics, no gradients."""
  logits = jax.vmap(model)(obs)
  loss = cross_entropy_loss(logits, targets)
  acc = top1_accuracy(logits, targets)
  return loss, acc


def compute_level_metrics(
  model: MazeCNN,
  ds: BCDataset,
  batch_size: int,
) -> dict[int, dict[str, object]]:
  """Compute per-level accuracy and loss for level_metrics.json."""
  # Run inference on all states in chunks
  n = ds.n_states
  all_logits = []

  for start in range(0, n, batch_size):
    end = min(start + batch_size, n)
    batch_tuples = ds.state_tuples[start:end]
    batch_level_idx = ds.level_idx[start:end]
    obs = make_batch_obs(ds.grid_size, ds.bank, batch_tuples, batch_level_idx)
    logits = jax.vmap(model)(obs)
    all_logits.append(logits)

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


def write_level_metrics(
  all_metrics: dict[int, dict[int, dict[str, object]]],
  sources: dict[int, list[tuple[str, int]]],
  step: int,
  run_id: str,
  metrics_path: Path,
) -> None:
  """Write level_metrics.json with per-level stats."""
  levels: dict[str, object] = {}
  for gs, gs_metrics in all_metrics.items():
    src_list = sources.get(gs, [])
    for bank_idx, stats in gs_metrics.items():
      if bank_idx < len(src_list):
        file_stem, sublevel = src_list[bank_idx]
        key = f"{file_stem}:{sublevel}"
      else:
        key = f"gs{gs}:idx{bank_idx}"
      levels[key] = {"grid_size": gs, **stats}

  output = {
    "run_id": run_id,
    "step": step,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "levels": levels,
  }
  metrics_path.parent.mkdir(parents=True, exist_ok=True)
  metrics_path.write_text(json.dumps(output, indent=2))


def train(
  maze_dir: Path,
  epochs: int = 10,
  batch_size: int = 1024,
  lr: float = 3e-4,
  seed: int = 0,
  wandb_project: str | None = None,
  metrics_path: Path = Path("level_metrics.json"),
  checkpoint_dir: Path = Path("checkpoints"),
) -> MazeCNN:
  """Main training loop."""
  key = jr.key(seed)

  # Load dataset
  print("Loading dataset...")
  t0 = time.time()
  datasets, sources = load_bc_dataset(maze_dir)
  print(f"Dataset loaded in {time.time() - t0:.1f}s")
  for gs, ds in sorted(datasets.items()):
    n_train = int(ds.train_mask.sum())
    n_val = int(ds.val_mask.sum())
    print(f"  grid_size={gs}: {ds.n_states} states ({n_train} train, {n_val} val)")

  # Initialize model
  key, model_key = jr.split(key)
  model = MazeCNN(model_key)
  n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
  print(f"Model: {n_params:,} parameters")

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
  opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

  # wandb init
  run_id = f"bc-cnn-{seed}"
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

  global_step = 0
  print(f"\nTraining for {epochs} epochs ({total_steps} steps)...")

  for epoch in range(epochs):
    epoch_t0 = time.time()
    epoch_losses: list[float] = []
    epoch_accs: list[float] = []

    # Train on each grid_size
    for gs in sorted(datasets):
      ds = datasets[gs]
      ti = train_indices[gs]
      n_train = ti.shape[0]
      if n_train < batch_size:
        continue

      # Shuffle train indices
      key, shuffle_key = jr.split(key)
      perm = jr.permutation(shuffle_key, n_train)
      shuffled = ti[perm]

      # Mini-batch loop
      n_batches = n_train // batch_size
      for b in range(n_batches):
        batch_idx = shuffled[b * batch_size : (b + 1) * batch_size]
        batch_tuples = ds.state_tuples[batch_idx]
        batch_targets = ds.action_targets[batch_idx]
        batch_level_idx = ds.level_idx[batch_idx]

        obs = jit_make_obs[gs](batch_tuples, batch_level_idx)
        model, opt_state, loss, acc = train_step(
          model, opt_state, optimizer, obs, batch_targets
        )
        global_step += 1

        loss_val = float(loss)
        acc_val = float(acc)
        epoch_losses.append(loss_val)
        epoch_accs.append(acc_val)

        if wandb_project is not None:
          import wandb

          wandb.log(
            {
              "train/loss": loss_val,
              "train/accuracy": acc_val,
              "train/grid_size": gs,
              "lr": float(schedule(global_step)),  # type: ignore[arg-type]
            },
            step=global_step,
          )

    # Validation
    val_losses: list[float] = []
    val_accs: list[float] = []
    for gs in sorted(datasets):
      ds = datasets[gs]
      vi = val_indices[gs]
      n_val = vi.shape[0]
      if n_val == 0:
        continue

      # Evaluate in chunks
      for start in range(0, n_val, batch_size):
        end = min(start + batch_size, n_val)
        batch_idx = vi[start:end]
        batch_tuples = ds.state_tuples[batch_idx]
        batch_targets = ds.action_targets[batch_idx]
        batch_level_idx = ds.level_idx[batch_idx]

        obs = jit_make_obs[gs](batch_tuples, batch_level_idx)
        loss, acc = eval_step(model, obs, batch_targets)
        val_losses.append(float(loss))
        val_accs.append(float(acc))

    mean_train_loss = np.mean(epoch_losses) if epoch_losses else 0.0
    mean_train_acc = np.mean(epoch_accs) if epoch_accs else 0.0
    mean_val_loss = np.mean(val_losses) if val_losses else 0.0
    mean_val_acc = np.mean(val_accs) if val_accs else 0.0
    epoch_time = time.time() - epoch_t0

    print(
      f"Epoch {epoch + 1}/{epochs} ({epoch_time:.1f}s) — "
      f"train loss={mean_train_loss:.4f} acc={mean_train_acc:.4f} — "
      f"val loss={mean_val_loss:.4f} acc={mean_val_acc:.4f}"
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
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"model_epoch{epoch + 1:03d}.eqx"
    eqx.tree_serialise_leaves(ckpt_path, model)

    # Write level_metrics.json
    print("  Computing per-level metrics...")
    all_metrics: dict[int, dict[int, dict[str, object]]] = {}
    for gs, ds in datasets.items():
      all_metrics[gs] = compute_level_metrics(model, ds, batch_size)
    write_level_metrics(all_metrics, sources, global_step, run_id, metrics_path)
    print(f"  Wrote {metrics_path}")

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
  args = parser.parse_args()

  train(
    maze_dir=args.mazes,
    epochs=args.epochs,
    batch_size=args.batch_size,
    lr=args.lr,
    seed=args.seed,
    wandb_project=args.wandb_project,
    metrics_path=args.metrics_path,
    checkpoint_dir=args.checkpoint_dir,
  )


if __name__ == "__main__":
  main()
