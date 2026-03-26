"""BC vs direct V_win training comparison.

Measures wall time to reach a target mean log10(win_prob) on the gs=6
validation set. Both arms use the same model init, optimizer schedule,
and evaluation (exact Rust Markov solver). JIT warmup is timed separately.

Usage:
  uv run python -m experiments.bc_vs_direct
  uv run python -m experiments.bc_vs_direct --mode direct --target-log-wp -1.0
  uv run python -m experiments.bc_vs_direct --mode both --eval-every-secs 30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import mummymaze_rust
import numpy as np

from src.train.dataset import BCDataset, load_bc_dataset, make_batch_obs
from src.train.eval import parse_rust_levels
from src.train.loss import cross_entropy_loss, top1_accuracy
from src.train.markov_jax import (
  GraphSkeleton,
  _make_solve_single,
  build_skeleton,
  pad_and_stack,
)
from src.train.model import make_model
from src.train.optim import count_params, make_optimizer
from src.train.train_bc import make_train_step

GRID_SIZE = 6


# ---------------------------------------------------------------------------
# Evaluation — shared between both arms
# ---------------------------------------------------------------------------


def make_evaluator(
  ds: BCDataset,
  val_level_indices: list[int],
  val_rust_levels: list[mummymaze_rust.Level],
  jit_make_obs: object,
  batch_size: int,
) -> object:
  """Build a reusable evaluate(model) -> mean_log10_win_prob closure."""
  val_mask_np = np.array(ds.val_mask)
  val_indices = jnp.where(ds.val_mask, size=int(ds.val_mask.sum()))[0]
  val_state_tuples_np = np.asarray(ds.state_tuples, dtype=np.int32)[val_mask_np]
  val_level_idx_np = np.array(ds.level_idx)[val_mask_np]

  old_to_new = {old: new for new, old in enumerate(val_level_indices)}
  remapped = np.array([old_to_new[int(x)] for x in val_level_idx_np])
  n_val = len(val_level_indices)
  counts = np.bincount(remapped, minlength=n_val)
  offsets = np.zeros(n_val + 1, dtype=np.intp)
  np.cumsum(counts, out=offsets[1:])
  sort_order = np.argsort(remapped, kind="stable")
  sorted_state_tuples = val_state_tuples_np[sort_order]

  def evaluate(model: eqx.Module) -> float:
    """Compute mean log10(win_prob) on validation levels via Rust Markov solver."""
    logits_list = []
    for start in range(0, val_indices.shape[0], batch_size):
      end = min(start + batch_size, val_indices.shape[0])
      batch_idx = val_indices[start:end]
      obs = jit_make_obs(ds.state_tuples[batch_idx], ds.level_idx[batch_idx])
      logits_list.append(np.array(jax.vmap(model)(obs)))
    val_logits = np.concatenate(logits_list, axis=0)

    # Stable softmax → probs
    val_probs = np.exp(val_logits - val_logits.max(axis=-1, keepdims=True))
    val_probs /= val_probs.sum(axis=-1, keepdims=True)
    sorted_probs = val_probs[sort_order]

    win_probs = mummymaze_rust.policy_win_prob_batch(
      val_rust_levels, sorted_state_tuples, sorted_probs, offsets.tolist()
    )

    valid = [wp for wp in win_probs if not np.isnan(wp) and wp > 0]
    if not valid:
      return float("-inf")
    return float(np.mean([np.log10(wp) for wp in valid]))

  return evaluate


# ---------------------------------------------------------------------------
# BC arm
# ---------------------------------------------------------------------------


def run_bc_arm(
  args: argparse.Namespace,
  ds: BCDataset,
  jit_make_obs: object,
  evaluate: object,
) -> None:
  """Train via behavioral cloning (cross-entropy on expert actions)."""
  key = jr.key(args.seed)
  key, model_key = jr.split(key)
  model = make_model(args.arch, model_key)
  print(f"  Model: {args.arch} ({count_params(model):,} params)")

  n_train = int(ds.train_mask.sum())
  steps_per_epoch = n_train // args.batch_size
  total_steps = steps_per_epoch * 1000  # generous schedule
  optimizer = make_optimizer(args.lr, total_steps)
  opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

  _train_step = make_train_step(cross_entropy_loss, top1_accuracy)
  train_indices = jnp.where(ds.train_mask, size=n_train)[0]

  # --- JIT warmup ---
  print("  JIT warmup...")
  warmup_t0 = time.time()
  key, warmup_key = jr.split(key)
  perm = jr.permutation(warmup_key, n_train)
  batch_idx = train_indices[perm[: args.batch_size]]
  obs = jit_make_obs(ds.state_tuples[batch_idx], ds.level_idx[batch_idx])
  targets = ds.action_targets[batch_idx]
  result = _train_step(model, opt_state, optimizer, obs, targets)
  jax.block_until_ready(result)
  warmup_time = time.time() - warmup_t0
  print(f"  JIT warmup: {warmup_time:.1f}s")

  # --- Initial eval ---
  mean_log_wp = evaluate(model)
  print(f"  Initial mean_log_wp: {mean_log_wp:.4f} (target: {args.target_log_wp})")

  # --- Training loop ---
  global_step = 0
  wall_start = time.time()
  last_eval_time = 0.0

  for epoch in range(1000):
    key, shuffle_key = jr.split(key)
    perm = jr.permutation(shuffle_key, n_train)
    shuffled = train_indices[perm]

    for b in range(steps_per_epoch):
      slc = slice(b * args.batch_size, (b + 1) * args.batch_size)
      batch_idx = shuffled[slc]
      obs = jit_make_obs(ds.state_tuples[batch_idx], ds.level_idx[batch_idx])
      targets = ds.action_targets[batch_idx]
      model, opt_state, loss, acc, _logits = _train_step(
        model, opt_state, optimizer, obs, targets
      )
      global_step += 1

    # Evaluate periodically (wall-time based)
    jax.block_until_ready(model)
    wall_elapsed = time.time() - wall_start

    if wall_elapsed - last_eval_time >= args.eval_every_secs or epoch == 0:
      last_eval_time = wall_elapsed
      mean_log_wp = evaluate(model)
      print(
        f"  Epoch {epoch + 1} step={global_step}: "
        f"loss={float(loss):.4f} acc={float(acc):.4f} "
        f"mean_log_wp={mean_log_wp:.4f} wall={wall_elapsed:.1f}s"
      )

      if mean_log_wp >= args.target_log_wp:
        print(f"  TARGET REACHED at wall={wall_elapsed:.1f}s (epoch {epoch + 1})")
        return

    if wall_elapsed > args.max_wall_seconds:
      print(f"  Wall time budget exceeded ({args.max_wall_seconds}s)")
      return


# ---------------------------------------------------------------------------
# Direct V_win arm
# ---------------------------------------------------------------------------


def _run_bc_warmup(
  model: eqx.Module,
  key: jax.Array,
  ds: BCDataset,
  jit_make_obs: object,
  batch_size: int,
  lr: float,
  n_epochs: int,
  evaluate: object,
) -> tuple[eqx.Module, jax.Array]:
  """Run a few BC epochs to warmstart the model. Returns (model, key)."""

  n_train = int(ds.train_mask.sum())
  steps_per_epoch = n_train // batch_size
  total_steps = steps_per_epoch * n_epochs
  optimizer = make_optimizer(lr, total_steps)
  opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

  _train_step = make_train_step(cross_entropy_loss, top1_accuracy)
  train_indices = jnp.where(ds.train_mask, size=n_train)[0]

  for epoch in range(n_epochs):
    key, shuffle_key = jr.split(key)
    perm = jr.permutation(shuffle_key, n_train)
    shuffled = train_indices[perm]

    for b in range(steps_per_epoch):
      slc = slice(b * batch_size, (b + 1) * batch_size)
      batch_idx = shuffled[slc]
      obs = jit_make_obs(ds.state_tuples[batch_idx], ds.level_idx[batch_idx])
      targets = ds.action_targets[batch_idx]
      model, opt_state, loss, acc, _ = _train_step(
        model, opt_state, optimizer, obs, targets
      )

    mean_log_wp = evaluate(model)
    print(
      f"    Warmup epoch {epoch + 1}/{n_epochs}: "
      f"loss={float(loss):.4f} acc={float(acc):.4f} "
      f"mean_log_wp={mean_log_wp:.4f}"
    )

  return model, key


def run_direct_arm(
  args: argparse.Namespace,
  ds: BCDataset,
  skeletons: list[GraphSkeleton],
  jit_make_obs: object,
  evaluate: object,
) -> None:
  """Train by directly optimizing win probability via differentiable Markov solve."""
  key = jr.key(args.seed)
  key, model_key = jr.split(key)

  if args.checkpoint is not None:
    from src.train.checkpoint import load_checkpoint

    ckpt = load_checkpoint(args.checkpoint, arch=args.arch)
    model = ckpt.model
    print(f"  Warmstart from {args.checkpoint}")
  else:
    model = make_model(args.arch, model_key)
  print(f"  Model: {args.arch} ({count_params(model):,} params)")

  # --- Optional BC warmup phase ---
  if args.bc_warmup_epochs > 0 and args.checkpoint is None:
    print(f"\n  --- BC warmup ({args.bc_warmup_epochs} epochs) ---")
    model, key = _run_bc_warmup(
      model,
      key,
      ds,
      jit_make_obs,
      args.batch_size,
      args.lr,
      args.bc_warmup_epochs,
      evaluate,
    )

  direct_lr = args.direct_lr if args.direct_lr is not None else args.lr
  n_levels = len(skeletons)
  steps_per_epoch = max(1, n_levels // args.levels_per_step)
  total_steps = steps_per_epoch * 1000  # generous schedule
  optimizer = make_optimizer(direct_lr, total_steps)
  print(f"  LR: {direct_lr}, {n_levels} levels, {args.levels_per_step} per step")
  opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

  solve = _make_solve_single(args.vi_iters, args.vi_iters)

  @eqx.filter_jit
  def train_step(  # noqa: ANN202
    model,
    opt_state,
    optimizer,  # noqa: ANN001
    state_tuples,
    level_idx,
    action_mask,  # noqa: ANN001
    edge_src,
    edge_action,
    edge_dst,
    edge_mask,  # noqa: ANN001
    win_src,
    win_action,
    win_mask,
    start_idx,  # noqa: ANN001
  ):  # JIT'd closure — types are dynamic JAX arrays
    n_lvl, max_s = action_mask.shape[:2]

    def _loss(m):  # noqa: ANN001, ANN202
      obs = jit_make_obs(state_tuples.reshape(-1, 12), level_idx.reshape(-1))
      logits_flat = jax.vmap(m)(obs)
      logits = logits_flat.reshape(n_lvl, max_s, 5)
      v_starts = jax.vmap(solve)(
        logits,
        action_mask,
        edge_src,
        edge_action,
        edge_dst,
        edge_mask,
        win_src,
        win_action,
        win_mask,
        start_idx,
      )
      return -jnp.mean(v_starts)

    loss, grads = eqx.filter_value_and_grad(_loss)(model)
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss

  # --- JIT warmup ---
  print("  JIT warmup (compiling forward + adjoint solver)...")
  warmup_t0 = time.time()
  key, warmup_key = jr.split(key)
  warmup_idxs = jr.choice(
    warmup_key, n_levels, shape=(args.levels_per_step,), replace=False
  )
  warmup_batch = pad_and_stack([skeletons[int(i)] for i in warmup_idxs])

  result = train_step(
    model,
    opt_state,
    optimizer,
    warmup_batch.state_tuples,
    warmup_batch.level_idx,
    warmup_batch.action_mask,
    warmup_batch.edge_src,
    warmup_batch.edge_action,
    warmup_batch.edge_dst,
    warmup_batch.edge_mask,
    warmup_batch.win_src,
    warmup_batch.win_action,
    warmup_batch.win_mask,
    warmup_batch.start_idx,
  )
  jax.block_until_ready(result)
  warmup_time = time.time() - warmup_t0
  print(f"  JIT warmup: {warmup_time:.1f}s")

  # --- Initial eval ---
  mean_log_wp = evaluate(model)
  print(f"  Initial mean_log_wp: {mean_log_wp:.4f} (target: {args.target_log_wp})")

  # --- Training loop ---
  global_step = 0
  wall_start = time.time()
  last_eval_time = 0.0

  for step_idx in range(total_steps):
    key, sample_key = jr.split(key)
    level_indices = jr.choice(
      sample_key, n_levels, shape=(args.levels_per_step,), replace=False
    )
    batch = pad_and_stack([skeletons[int(i)] for i in level_indices])

    model, opt_state, loss = train_step(
      model,
      opt_state,
      optimizer,
      batch.state_tuples,
      batch.level_idx,
      batch.action_mask,
      batch.edge_src,
      batch.edge_action,
      batch.edge_dst,
      batch.edge_mask,
      batch.win_src,
      batch.win_action,
      batch.win_mask,
      batch.start_idx,
    )
    global_step += 1

    # Evaluate periodically (wall-time based)
    jax.block_until_ready(model)
    wall_elapsed = time.time() - wall_start

    if wall_elapsed - last_eval_time >= args.eval_every_secs or step_idx == 0:
      last_eval_time = wall_elapsed
      mean_log_wp = evaluate(model)
      print(
        f"  Step {global_step}: loss={float(loss):.4f} "
        f"mean_log_wp={mean_log_wp:.4f} wall={wall_elapsed:.1f}s"
      )

      if mean_log_wp >= args.target_log_wp:
        print(f"  TARGET REACHED at wall={wall_elapsed:.1f}s (step {global_step})")
        return

    if wall_elapsed > args.max_wall_seconds:
      print(f"  Wall time budget exceeded ({args.max_wall_seconds}s)")
      return


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
  parser = argparse.ArgumentParser(description="BC vs direct V_win training comparison")
  parser.add_argument("--maze-dir", type=Path, default=Path("mazes"))
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--arch", type=str, default="cnn")
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument(
    "--batch-size", type=int, default=1024, help="BC: states per batch"
  )
  parser.add_argument(
    "--levels-per-step", type=int, default=16, help="Direct: levels per step"
  )
  parser.add_argument(
    "--target-log-wp",
    type=float,
    default=-1.0,
    help="Target mean log10(win_prob) on val set",
  )
  parser.add_argument(
    "--max-wall-seconds",
    type=float,
    default=3600,
    help="Wall time budget per arm",
  )
  parser.add_argument(
    "--eval-every-secs",
    type=float,
    default=30,
    help="Evaluate every N seconds of wall time",
  )
  parser.add_argument(
    "--vi-iters",
    type=int,
    default=500,
    help="Value iteration steps (forward + adjoint)",
  )
  parser.add_argument(
    "--direct-lr",
    type=float,
    default=None,
    help="Learning rate for direct arm (defaults to --lr)",
  )
  parser.add_argument(
    "--checkpoint",
    type=Path,
    default=None,
    help="BC checkpoint to warmstart the direct arm from",
  )
  parser.add_argument(
    "--bc-warmup-epochs",
    type=int,
    default=0,
    help="Direct arm: run BC for N epochs first, then switch to direct",
  )
  parser.add_argument("--wandb-project", type=str, default=None)
  parser.add_argument(
    "--mode",
    type=str,
    default="both",
    choices=["bc", "direct", "both"],
  )
  args = parser.parse_args()

  # ===== Shared setup =====
  print("Loading dataset...")
  datasets, sources = load_bc_dataset(args.maze_dir)
  ds = datasets[GRID_SIZE]
  n_train = int(ds.train_mask.sum())
  n_val = int(ds.val_mask.sum())
  print(f"  gs={GRID_SIZE}: {ds.n_states} states ({n_train} train, {n_val} val)")

  # Validation levels (Rust Level objects for exact eval)
  level_idx_np = np.array(ds.level_idx)
  val_mask_np = np.array(ds.val_mask)
  val_level_indices = sorted(set(int(x) for x in level_idx_np[val_mask_np]))
  val_keys = [sources[GRID_SIZE][i] for i in val_level_indices]
  val_rust_levels = parse_rust_levels(args.maze_dir, val_keys)
  print(f"  {len(val_rust_levels)} validation levels")

  # JIT obs function
  jit_make_obs = jax.jit(
    lambda tuples, lidx: make_batch_obs(GRID_SIZE, ds.bank, tuples, lidx)
  )

  # Evaluator
  evaluate = make_evaluator(
    ds, val_level_indices, val_rust_levels, jit_make_obs, args.batch_size
  )

  # ===== Build graph skeletons (for direct arm) =====
  skeletons: list[GraphSkeleton] = []
  if args.mode in ("direct", "both"):
    print("\nBuilding graph skeletons for training levels...")
    t0 = time.time()
    train_level_indices = sorted(
      set(int(x) for x in level_idx_np[np.array(ds.train_mask)])
    )
    train_keys = [sources[GRID_SIZE][i] for i in train_level_indices]
    train_rust_levels = parse_rust_levels(args.maze_dir, train_keys)

    for lev, bank_idx in zip(train_rust_levels, train_level_indices):
      skeletons.append(build_skeleton(lev, bank_idx))

    total_skel_states = sum(s.n_states for s in skeletons)
    max_skel_states = max(s.n_states for s in skeletons)
    print(
      f"  {len(skeletons)} skeletons, {total_skel_states} total states, "
      f"max {max_skel_states} states/level ({time.time() - t0:.1f}s)"
    )

  # ===== Run arms =====
  if args.mode in ("bc", "both"):
    print(f"\n{'=' * 60}")
    print("BC ARM")
    print(f"{'=' * 60}")
    run_bc_arm(args, ds, jit_make_obs, evaluate)

  if args.mode in ("direct", "both"):
    print(f"\n{'=' * 60}")
    print("DIRECT V_WIN ARM")
    print(f"{'=' * 60}")
    run_direct_arm(args, ds, skeletons, jit_make_obs, evaluate)

  print("\nDone.")


if __name__ == "__main__":
  main()
