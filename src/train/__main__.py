"""Unified training entry point with nanoGPT-style configuration.

Usage:
  uv run python -m src.train config/bc_default.py
  uv run python -m src.train config/bc_default.py --lr=1e-4 --seed=42
  uv run python -m src.train --epochs=5 --arch=cnn
  uv run python -m src.train config/adversarial_default.py --n_rounds=1
"""

import json
import sys
from pathlib import Path

from src.train.configurator import load_config, parse_argv

# Default config values — union of BC and adversarial parameters.
# Config files and CLI overrides add to or replace these.
DEFAULTS: dict[str, object] = {
  "mode": "bc",
  "maze_dir": "mazes",
  # Training
  "epochs": 10,
  "batch_size": 1024,
  "lr": 3e-4,
  "seed": 0,
  "arch": "cnn",
  "hparams": {},
  # Output
  "checkpoint_dir": None,
  "wandb_project": None,
  "metrics_path": "level_metrics.json",
  # Augmentation
  "dihedral_augment": False,
  "augment_levels": None,
  # Resume
  "checkpoint": None,
  "epoch_offset": 0,
  "step_offset": 0,
  # Subprocess mode
  "subprocess": False,
  # Adversarial-specific
  "n_rounds": 3,
  "epochs_per_round": 5,
  "ga_generations": 50,
  "ga_pop_size": 64,
  "target_log_policy_wp": -1.0,
  "ga_seeds_per_gs": 100,
  "archive_bfs_bins": 20,
  "archive_states_bins": 20,
  "grid_sizes": [6, 8, 10],
}

# Config keys that are callables (not coercible from CLI strings)
CALLABLE_KEYS = {
  "loss_fn",
  "metric_fn",
  "make_obs_fn",
  "optimizer",
  "inner_stop",
  "outer_stop",
  "on_round_end",
}


def main() -> None:
  config_file, overrides = parse_argv()
  config = load_config(config_file, overrides, DEFAULTS)

  mode = config.pop("mode")
  subprocess_mode = config.pop("subprocess")

  # Extract callables before primitive coercion
  callables = {k: config.pop(k) for k in CALLABLE_KEYS if k in config}

  # Path coercion
  maze_dir = Path(str(config.pop("maze_dir")))
  metrics_path = Path(str(config.pop("metrics_path")))
  checkpoint_dir_raw = config.pop("checkpoint_dir")
  checkpoint_dir = Path(str(checkpoint_dir_raw)) if checkpoint_dir_raw else None
  checkpoint_raw = config.pop("checkpoint")
  checkpoint = Path(str(checkpoint_raw)) if checkpoint_raw else None
  augment_raw = config.pop("augment_levels")
  augment_levels = Path(str(augment_raw)) if augment_raw else None

  # Reporter
  if subprocess_mode:
    from src.train.reporter import StdioReporter

    reporter = StdioReporter()
  else:
    from src.train.reporter import FileReporter

    reporter = FileReporter(metrics_path)

  # Build TrainComponents from config callables (with defaults)
  components = _build_components(callables)

  if mode == "bc":
    _run_bc(
      config,
      callables,
      components,
      maze_dir,
      checkpoint_dir,
      checkpoint,
      augment_levels,
      reporter,
    )
  elif mode == "adversarial":
    _run_adversarial(
      config,
      callables,
      components,
      maze_dir,
      checkpoint_dir,
      metrics_path,
      reporter,
    )
  else:
    print(f"Unknown mode: {mode!r}. Expected 'bc' or 'adversarial'.")
    sys.exit(1)


def _build_components(callables: dict) -> object | None:
  """Build TrainComponents from config callables, or None for defaults."""
  if not any(k in callables for k in ("loss_fn", "metric_fn", "make_obs_fn")):
    return None

  from src.train.config import TrainComponents
  from src.train.dataset import make_batch_obs
  from src.train.loss import cross_entropy_loss, top1_accuracy

  return TrainComponents(
    loss_fn=callables.get("loss_fn", cross_entropy_loss),
    metric_fn=callables.get("metric_fn", top1_accuracy),
    make_obs_fn=callables.get("make_obs_fn", make_batch_obs),
  )


def _run_bc(
  config: dict,
  callables: dict,
  components: object | None,
  maze_dir: Path,
  checkpoint_dir: Path | None,
  checkpoint: Path | None,
  augment_levels: Path | None,
  reporter: object,
) -> None:
  from src.train.loop import training_loop
  from src.train.session import setup_training

  session = setup_training(
    maze_dir=maze_dir,
    epochs=int(config["epochs"]),
    batch_size=int(config["batch_size"]),
    lr=float(config["lr"]),
    seed=int(config["seed"]),
    arch=str(config["arch"]),
    hparams=config["hparams"] if config["hparams"] else None,
    wandb_project=config.get("wandb_project"),
    checkpoint_dir=checkpoint_dir,
    checkpoint=checkpoint,
    augment_levels=augment_levels,
    epoch_offset=int(config.get("epoch_offset", 0)),
    step_offset=int(config.get("step_offset", 0)),
    dihedral_augment=bool(config.get("dihedral_augment", False)),
    reporter=reporter,
    optimizer=callables.get("optimizer"),
    components=components,
  )

  training_loop(
    session,
    inner_stop=callables.get("inner_stop"),
    outer_stop=callables.get("outer_stop"),
    on_round_end=callables.get("on_round_end"),
  )
  session.finish()


def _run_adversarial(
  config: dict,
  callables: dict,
  components: object | None,
  maze_dir: Path,
  checkpoint_dir: Path | None,
  metrics_path: Path,
  reporter: object,
) -> None:
  from src.train.adversarial_loop import adversarial_loop

  arch = str(config["arch"])
  hparams = config["hparams"]
  if isinstance(hparams, dict) and not hparams:
    hparams = None

  adversarial_loop(
    maze_dir=maze_dir,
    n_rounds=int(config["n_rounds"]),
    epochs_per_round=int(config["epochs_per_round"]),
    batch_size=int(config["batch_size"]),
    lr=float(config["lr"]),
    seed=int(config["seed"]),
    ga_generations=int(config["ga_generations"]),
    ga_pop_size=int(config["ga_pop_size"]),
    target_log_policy_wp=float(config["target_log_policy_wp"]),
    ga_seeds_per_gs=int(config["ga_seeds_per_gs"]),
    archive_bfs_bins=int(config["archive_bfs_bins"]),
    archive_states_bins=int(config["archive_states_bins"]),
    grid_sizes=config["grid_sizes"],
    checkpoint_dir=checkpoint_dir or Path("checkpoints/adversarial"),
    metrics_path=metrics_path,
    arch=arch,
    dihedral_augment=bool(config.get("dihedral_augment", False)),
    wandb_project=config.get("wandb_project"),
    hparams=hparams,
    reporter=reporter,
  )


if __name__ == "__main__":
  try:
    main()
  except Exception as e:
    # In subprocess mode, emit JSON error
    if "--subprocess=true" in sys.argv or "--subprocess=True" in sys.argv:
      sys.stdout.write(json.dumps({"type": "error", "message": str(e)}) + "\n")
      sys.stdout.flush()
    raise
