"""Behavioral cloning training script for Mummy Maze."""

import enum
import json
import sys
import time
from pathlib import Path
from typing import Annotated

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from collections.abc import Callable
from typing import Any

import click
import typer
from tqdm import tqdm

from jaxtyping import Array, Float, Int

from src.train.callbacks import (
  CheckpointFn,
  LogFn,
  make_checkpoint_fn,
  make_wandb_log_fn,
)
from src.train.checkpoint import load_checkpoint
from src.train.config import TrainConfig, TrainState
from src.train.dataset import BCDataset, load_bc_dataset, make_batch_obs
from src.train.eval import compute_level_metrics, compute_markov_win_probs
from src.train.loss import cross_entropy_loss, top1_accuracy
from src.train.model import DEFAULT_ARCH, MODEL_REGISTRY, make_model, parse_hparams
from src.train.optim import count_params, make_optimizer
from src.train.reporter import FileReporter, MetricsReporter, StdioReporter


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


def train_epochs(
  state: TrainState,
  config: TrainConfig,
  datasets: dict[int, BCDataset],
  sources: dict[int, list[tuple[str, int]]],
  reporter: MetricsReporter,
  *,
  log_fn: LogFn | None = None,
  checkpoint_fn: CheckpointFn | None = None,
) -> TrainState:
  """Run training epochs. Mutates and returns state.

  If config.max_steps is set, training stops once global_step reaches that
  limit, even if not all epochs have completed.
  """
  use_tqdm = isinstance(reporter, FileReporter)
  batch_size = config.batch_size

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

  stop_requested = False

  steps_per_epoch = sum(
    int(ds.train_mask.sum()) // batch_size for ds in datasets.values()
  )
  if use_tqdm:
    total_steps = steps_per_epoch * config.epochs
    print(f"\nTraining for {config.epochs} epochs ({total_steps} steps)...")

  for epoch_rel in range(config.epochs):
    if stop_requested:
      break
    if config.max_steps is not None and state.global_step >= config.max_steps:
      break

    epoch = state.epoch_offset + epoch_rel
    epoch_t0 = time.time()
    epoch_losses: list[float] = []
    epoch_accs: list[float] = []
    reporter.report_epoch_start(
      epoch + 1, state.epoch_offset + config.epochs, steps_per_epoch
    )

    # Build train batches across all grid sizes for a single progress bar
    # Each entry: (grid_size, jax_indices, numpy_indices_for_scatter)
    train_batches: list[tuple[int, Int[Array, "B"], np.ndarray]] = []
    for gs in sorted(datasets):
      ds = datasets[gs]
      ti = train_indices[gs]
      n_train = ti.shape[0]
      if n_train < batch_size:
        continue
      state.key, shuffle_key = jr.split(state.key)
      perm = jr.permutation(shuffle_key, n_train)
      shuffled = ti[perm]
      shuffled_np = np.array(shuffled)
      n_batches = n_train // batch_size
      for b in range(n_batches):
        slc = slice(b * batch_size, (b + 1) * batch_size)
        train_batches.append((gs, shuffled[slc], shuffled_np[slc]))

    if use_tqdm:
      total_epochs = state.epoch_offset + config.epochs
      pbar = tqdm(train_batches, desc=f"Epoch {epoch + 1}/{total_epochs}", unit="batch")
    else:
      pbar = train_batches
    epoch_step = 0
    for gs, batch_idx, batch_idx_np in pbar:
      # Check for stop command or step budget
      cmd = reporter.check_command()
      if cmd == "stop":
        stop_requested = True
        break
      if config.max_steps is not None and state.global_step >= config.max_steps:
        break

      ds = datasets[gs]
      batch_tuples = ds.state_tuples[batch_idx]
      batch_targets = ds.action_targets[batch_idx]
      batch_level_idx = ds.level_idx[batch_idx]

      if gs not in jitted_grid_sizes:
        reporter.report_status(f"Jitting grid_size={gs}...")
        jitted_grid_sizes.add(gs)

      obs = jit_make_obs[gs](batch_tuples, batch_level_idx)
      state.model, state.opt_state, loss, acc, logits = train_step(
        state.model, state.opt_state, state.optimizer, obs, batch_targets
      )
      # Scatter logits into buffer for level metrics
      logits_buffers[gs][batch_idx_np] = np.array(logits)
      state.global_step += 1
      epoch_step += 1

      loss_val = float(loss)
      acc_val = float(acc)
      epoch_losses.append(loss_val)
      epoch_accs.append(acc_val)

      if use_tqdm:
        pbar.set_postfix(loss=f"{loss_val:.3f}", acc=f"{acc_val:.3f}", gs=gs)  # type: ignore[union-attr]

      reporter.report_batch(state.global_step, epoch_step, loss_val, acc_val, gs)

      if log_fn is not None:
        log_fn(
          state.global_step,
          {
            "train/loss": loss_val,
            "train/accuracy": acc_val,
            "train/grid_size": float(gs),
          },
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
      loss, acc = eval_step(state.model, obs, batch_targets)
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

    if log_fn is not None:
      log_fn(
        state.global_step,
        {
          "epoch": float(epoch + 1),
          "val/loss": mean_val_loss,
          "val/accuracy": mean_val_acc,
          "train/epoch_loss": mean_train_loss,
          "train/epoch_accuracy": mean_train_acc,
        },
      )

    if checkpoint_fn is not None:
      checkpoint_fn(state, epoch + 1, config)

    # Per-level metrics (expensive — disabled by default, viewer enables it)
    all_metrics: dict[int, dict[int, dict[str, object]]] = {}
    if config.level_metrics:
      reporter.report_status("Computing level metrics...")
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
          state.model,
          ds,
          batch_size,
          metrics_pbar,
          logits_buffer=logits_buffers[gs],
          jit_make_obs_fn=jit_make_obs[gs],
        )
      if metrics_pbar is not None:
        metrics_pbar.close()

    # Compute agent win% via Markov solver (validation levels only)
    reporter.report_status("Computing agent win%...")
    all_win_probs, wp_updates = compute_markov_win_probs(
      state.model,
      datasets,
      sources,
      config.maze_dir,
      batch_size,
      jit_make_obs,
      reporter_log=reporter.report_log,
      use_tqdm=use_tqdm,
    )
    # Merge win prob updates into level metrics
    for gs, gs_updates in wp_updates.items():
      for lvl_idx, update in gs_updates.items():
        if lvl_idx in all_metrics.get(gs, {}):
          all_metrics[gs][lvl_idx].update(update)

    if config.level_metrics:
      reporter.report_level_metrics(
        state.global_step, config.run_id, all_metrics, sources
      )

    if log_fn is not None and all_win_probs:
      log_fn(
        state.global_step,
        {
          "eval/mean_win_prob": float(np.mean(all_win_probs)),
          "eval/median_win_prob": float(np.median(all_win_probs)),
          "eval/min_win_prob": float(np.min(all_win_probs)),
          "eval/max_win_prob": float(np.max(all_win_probs)),
          "eval/n_levels": float(len(all_win_probs)),
        },
      )

  return state


def train(
  maze_dir: Path,
  epochs: int = 10,
  batch_size: int = 1024,
  lr: float = 3e-4,
  seed: int = 0,
  wandb_project: str | None = None,
  metrics_path: Path = Path("level_metrics.json"),
  checkpoint_dir: Path | None = None,
  reporter: FileReporter | StdioReporter | None = None,
  checkpoint: Path | None = None,
  augment_levels: Path | None = None,
  epoch_offset: int = 0,
  step_offset: int = 0,
  arch: str = DEFAULT_ARCH,
  dihedral_augment: bool = False,
  hparams: dict[str, object] | None = None,
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
    from src.train.augment import augment_dataset, load_augment_levels

    levels = load_augment_levels(augment_levels)
    if use_tqdm:
      print(f"Augmenting dataset with {len(levels)} levels...")
    datasets = augment_dataset(datasets, levels)
    if use_tqdm:
      for gs, ds in sorted(datasets.items()):
        n_train = int(ds.train_mask.sum())
        print(f"  grid_size={gs}: {ds.n_states} states ({n_train} train)")

  # Dihedral augmentation: expand train levels with rotations/reflections
  if dihedral_augment:
    from src.train.augment import apply_dihedral_augmentation

    banks = {gs: ds.bank for gs, ds in datasets.items()}
    datasets = apply_dihedral_augmentation(
      datasets,
      sources,
      banks,
      maze_dir,
      verbose=use_tqdm,
    )

  # Initialize model (or resume from checkpoint)
  model_hps = hparams or {}
  if checkpoint is not None:
    if use_tqdm:
      print(f"Loading checkpoint: {checkpoint}")
    ckpt = load_checkpoint(checkpoint, arch=arch, hparams=model_hps or None)
    model = ckpt.model
    arch = ckpt.arch
    model_hps = ckpt.hparams
    if epoch_offset == 0 and ckpt.epoch > 0:
      epoch_offset = ckpt.epoch
    if step_offset == 0 and ckpt.global_step > 0:
      step_offset = ckpt.global_step
    if ckpt.key is not None:
      key = ckpt.key
  else:
    key, model_key = jr.split(key)
    model = make_model(arch, model_key, **model_hps)

  n_params = count_params(model)
  if use_tqdm:
    print(f"Model: {arch} ({n_params:,} parameters)")

  # Optimizer
  total_train_states = sum(int(ds.train_mask.sum()) for ds in datasets.values())
  steps_per_epoch = sum(
    int(ds.train_mask.sum()) // batch_size for ds in datasets.values()
  )
  total_steps = steps_per_epoch * epochs

  optimizer = make_optimizer(lr, total_steps)

  # Restore optimizer state from checkpoint if available
  opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
  if checkpoint is not None:
    opt_state_path = checkpoint / "opt_state.eqx"
    if opt_state_path.exists():
      opt_state = eqx.tree_deserialise_leaves(opt_state_path, opt_state)
      if use_tqdm:
        print("Restored optimizer state from checkpoint")

  run_id = f"bc-{arch}-{seed}"
  config = TrainConfig(
    epochs=epochs,
    batch_size=batch_size,
    lr=lr,
    arch=arch,
    hparams=model_hps,
    run_id=run_id,
    maze_dir=maze_dir,
  )
  state = TrainState(
    model=model,
    opt_state=opt_state,
    optimizer=optimizer,
    key=key,
    global_step=step_offset,
    epoch_offset=epoch_offset,
  )

  # Build callbacks
  log_fn: LogFn | None = None
  if wandb_project is not None:
    import wandb

    wandb.init(
      project=wandb_project,
      name=run_id,
      config={
        "arch": arch,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "n_params": n_params,
        "total_train_states": total_train_states,
      },
    )
    log_fn = make_wandb_log_fn()

  checkpoint_fn: CheckpointFn | None = None
  if checkpoint_dir is not None:
    checkpoint_fn = make_checkpoint_fn(checkpoint_dir)

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

  state = train_epochs(
    state,
    config,
    datasets,
    sources,
    reporter,
    log_fn=log_fn,
    checkpoint_fn=checkpoint_fn,
  )

  reporter.report_done()

  if wandb_project is not None:
    import wandb

    wandb.finish()

  return state.model


class Mode(str, enum.Enum):
  standalone = "standalone"
  subprocess = "subprocess"


def main(
  mazes: Annotated[
    Path, typer.Option(help="Directory containing B-*.dat files")
  ] = Path("mazes"),
  epochs: int = 10,
  batch_size: int = 1024,
  lr: float = 3e-4,
  seed: int = 0,
  wandb_project: Annotated[str | None, typer.Option(help="W&B project name")] = None,
  metrics_path: Path = Path("level_metrics.json"),
  checkpoint_dir: Annotated[
    Path | None, typer.Option(help="Save checkpoints to this directory")
  ] = None,
  mode: Annotated[
    Mode, typer.Option(help="standalone: tqdm; subprocess: JSON")
  ] = Mode.standalone,
  checkpoint: Annotated[
    Path | None, typer.Option(help="Resume from checkpoint directory")
  ] = None,
  augment_levels: Annotated[
    Path | None, typer.Option(help="JSON file of extra levels")
  ] = None,
  epoch_offset: int = 0,
  step_offset: int = 0,
  arch: Annotated[
    str,
    typer.Option(
      help="Model architecture",
      click_type=click.Choice(sorted(MODEL_REGISTRY)),
    ),
  ] = DEFAULT_ARCH,
  dihedral_augment: Annotated[
    bool, typer.Option(help="Expand training set with dihedral variants")
  ] = False,
  hparam: Annotated[
    list[str] | None,
    typer.Option(help="Model hparam as key=value (repeatable)"),
  ] = None,
) -> None:
  """Behavioral cloning for Mummy Maze."""
  if mode == Mode.subprocess:
    reporter: FileReporter | StdioReporter = StdioReporter()
  else:
    reporter = FileReporter(metrics_path)

  hparams = parse_hparams(arch, hparam or [])

  try:
    train(
      maze_dir=mazes,
      epochs=epochs,
      batch_size=batch_size,
      lr=lr,
      seed=seed,
      wandb_project=wandb_project,
      metrics_path=metrics_path,
      checkpoint_dir=checkpoint_dir,
      reporter=reporter,
      checkpoint=checkpoint,
      augment_levels=augment_levels,
      epoch_offset=epoch_offset,
      step_offset=step_offset,
      arch=arch,
      dihedral_augment=dihedral_augment,
      hparams=hparams,
    )
  except Exception as e:
    if mode == Mode.subprocess:
      sys.stdout.write(json.dumps({"type": "error", "message": str(e)}) + "\n")
      sys.stdout.flush()
    raise


if __name__ == "__main__":
  typer.run(main)
