"""Training session: shared setup factory for all training entry points."""

import time
from dataclasses import dataclass, field
from pathlib import Path

import equinox as eqx
import jax.random as jr

from src.train.callbacks import (
  CheckpointFn,
  LogFn,
  make_checkpoint_fn,
  make_wandb_log_fn,
)
from src.train.checkpoint import load_checkpoint
from src.train.config import TrainComponents, TrainConfig, TrainState
from src.train.dataset import BCDataset, load_bc_dataset
from src.train.model import DEFAULT_ARCH, make_model
from src.train.optim import count_params, make_optimizer
from src.train.reporter import FileReporter, MetricsReporter
from src.train.train_bc import train_epochs


@dataclass
class TrainingSession:
  """Result of setting up a training run. Holds everything train_epochs needs."""

  state: TrainState
  config: TrainConfig
  datasets: dict[int, BCDataset]
  sources: dict[int, list[tuple[str, int]]]
  reporter: MetricsReporter
  log_fn: LogFn | None
  checkpoint_fn: CheckpointFn | None
  components: TrainComponents | None = None
  _wandb_active: bool = field(default=False, repr=False)

  def run(self) -> TrainState:
    """Run train_epochs with all session fields."""
    self.state = train_epochs(
      self.state,
      self.config,
      self.datasets,
      self.sources,
      self.reporter,
      log_fn=self.log_fn,
      checkpoint_fn=self.checkpoint_fn,
      components=self.components,
    )
    return self.state

  def finish(self) -> None:
    """Call after training is complete. Reports done + finishes wandb."""
    self.reporter.report_done()
    if self._wandb_active:
      import wandb

      wandb.finish()


def setup_training(
  maze_dir: Path,
  *,
  # Model
  arch: str = DEFAULT_ARCH,
  hparams: dict[str, object] | None = None,
  checkpoint: Path | None = None,
  # Training
  epochs: int = 10,
  batch_size: int = 1024,
  lr: float = 3e-4,
  seed: int = 0,
  # Data augmentation
  augment_levels: Path | None = None,
  dihedral_augment: bool = False,
  # Output
  checkpoint_dir: Path | None = None,
  wandb_project: str | None = None,
  wandb_config: dict[str, object] | None = None,
  reporter: MetricsReporter | None = None,
  metrics_path: Path = Path("level_metrics.json"),
  run_id: str | None = None,
  level_metrics: bool = False,
  # Resume
  epoch_offset: int = 0,
  step_offset: int = 0,
  # Optimizer schedule override (e.g. adversarial: epochs_per_round * n_rounds)
  schedule_epochs: int | None = None,
  # Swappable components
  optimizer: object | None = None,  # optax.GradientTransformation; skip make_optimizer
  components: TrainComponents | None = None,
) -> TrainingSession:
  """Set up a training run: load data, init model, create optimizer, build callbacks.

  Returns a TrainingSession ready for session.run() or manual train_epochs() calls.
  """
  if reporter is None:
    reporter = FileReporter(metrics_path)

  verbose = isinstance(reporter, FileReporter)
  key = jr.key(seed)

  # --- Load dataset ---
  if verbose:
    print("Loading dataset...")
  t0 = time.time()
  datasets, sources = load_bc_dataset(maze_dir)
  if verbose:
    print(f"Dataset loaded in {time.time() - t0:.1f}s")
    for gs, ds in sorted(datasets.items()):
      n_train = int(ds.train_mask.sum())
      n_val = int(ds.val_mask.sum())
      print(f"  grid_size={gs}: {ds.n_states} states ({n_train} train, {n_val} val)")

  # --- Augment with extra levels ---
  if augment_levels is not None:
    from src.train.augment import augment_dataset, load_augment_levels

    levels = load_augment_levels(augment_levels)
    if verbose:
      print(f"Augmenting dataset with {len(levels)} levels...")
    datasets = augment_dataset(datasets, levels)
    if verbose:
      for gs, ds in sorted(datasets.items()):
        n_train = int(ds.train_mask.sum())
        print(f"  grid_size={gs}: {ds.n_states} states ({n_train} train)")

  # --- Dihedral augmentation ---
  if dihedral_augment:
    from src.train.augment import apply_dihedral_augmentation

    banks = {gs: ds.bank for gs, ds in datasets.items()}
    datasets = apply_dihedral_augmentation(
      datasets, sources, banks, maze_dir, verbose=verbose
    )

  # --- Model init or resume ---
  model_hps = hparams or {}
  if checkpoint is not None:
    if verbose:
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
  if verbose:
    print(f"Model: {arch} ({n_params:,} parameters)")

  # --- Optimizer ---
  total_train_states = sum(int(ds.train_mask.sum()) for ds in datasets.values())
  if optimizer is None:
    steps_per_epoch = sum(
      int(ds.train_mask.sum()) // batch_size for ds in datasets.values()
    )
    effective_epochs = schedule_epochs if schedule_epochs is not None else epochs
    total_steps = steps_per_epoch * effective_epochs
    optimizer = make_optimizer(lr, total_steps)

  # Restore optimizer state from checkpoint if available
  opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
  if checkpoint is not None:
    opt_state_path = checkpoint / "opt_state.eqx"
    if opt_state_path.exists():
      opt_state = eqx.tree_deserialise_leaves(opt_state_path, opt_state)
      if verbose:
        print("Restored optimizer state from checkpoint")

  # --- Config + State ---
  resolved_run_id = run_id or f"bc-{arch}-{seed}"
  config = TrainConfig(
    epochs=epochs,
    batch_size=batch_size,
    lr=lr,
    arch=arch,
    hparams=model_hps,
    run_id=resolved_run_id,
    maze_dir=maze_dir,
    level_metrics=level_metrics,
  )
  state = TrainState(
    model=model,
    opt_state=opt_state,
    optimizer=optimizer,
    key=key,
    global_step=step_offset,
    epoch_offset=epoch_offset,
  )

  # --- Callbacks ---
  log_fn: LogFn | None = None
  wandb_active = False
  if wandb_project is not None:
    import wandb

    init_config: dict[str, object] = {
      "arch": arch,
      "epochs": epochs,
      "batch_size": batch_size,
      "lr": lr,
      "seed": seed,
      "n_params": n_params,
      "total_train_states": total_train_states,
    }
    if wandb_config is not None:
      init_config.update(wandb_config)
    wandb.init(project=wandb_project, name=resolved_run_id, config=init_config)
    log_fn = make_wandb_log_fn()
    wandb_active = True

  checkpoint_fn: CheckpointFn | None = None
  if checkpoint_dir is not None:
    checkpoint_fn = make_checkpoint_fn(checkpoint_dir)

  # --- Report init ---
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

  return TrainingSession(
    state=state,
    config=config,
    datasets=datasets,
    sources=sources,
    reporter=reporter,
    log_fn=log_fn,
    checkpoint_fn=checkpoint_fn,
    components=components,
    _wandb_active=wandb_active,
  )
