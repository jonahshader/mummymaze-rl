"""Unified training entry point with nanoGPT-style configuration.

All training modes flow through the same path:
  setup_training() → training_loop(inner_stop, outer_stop, on_round_end)

BC = single round, no on_round_end.
Adversarial = multiple rounds, GA callback as on_round_end.
Custom = config file provides its own lambdas.

Usage:
  uv run python -m src.train config/bc_default.py
  uv run python -m src.train config/bc_default.py --lr=1e-4 --seed=42
  uv run python -m src.train config/adversarial_default.py --n_rounds=1
"""

import json
import sys
from pathlib import Path

from src.train.configurator import load_config, parse_argv

# Default config values. Config files and CLI overrides add to or replace these.
DEFAULTS: dict[str, object] = {
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
  # GA params (used when on_round_end is auto-constructed)
  "ga_generations": 50,
  "ga_pop_size": 64,
  "target_log_policy_wp": -1.0,
  "ga_seeds_per_gs": 100,
  "archive_bfs_bins": 20,
  "archive_states_bins": 20,
  "grid_sizes": [6, 8, 10],
}

# Config keys that are callables (pass through without type coercion)
CALLABLE_KEYS = {
  "loss_fn",
  "metric_fn",
  "make_obs_fn",
  "optimizer",
  "inner_stop",
  "outer_stop",
  "on_round_end",
  "on_event",
}


def main() -> None:
  config_file, overrides = parse_argv()
  config = load_config(config_file, overrides, DEFAULTS)

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

  # Determine schedule_epochs for optimizer LR schedule
  outer_stop = callables.get("outer_stop")
  n_rounds = int(config.get("n_rounds", 1))
  epochs = int(config["epochs"])
  schedule_epochs = epochs * n_rounds if n_rounds > 1 else None

  from src.train.loop import training_loop
  from src.train.session import setup_training

  session = setup_training(
    maze_dir=maze_dir,
    epochs=epochs,
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
    metrics_path=metrics_path,
    optimizer=callables.get("optimizer"),
    components=components,
    schedule_epochs=schedule_epochs,
  )

  # Build on_round_end: use config callable, or auto-construct from GA params
  on_round_end = callables.get("on_round_end")
  if on_round_end is None and n_rounds > 1:
    on_round_end = _build_ga_round_end(config, maze_dir, checkpoint_dir, session)

  # Build stopping conditions
  inner_stop = callables.get("inner_stop")
  if outer_stop is None and n_rounds > 1:
    from src.train.stopping import stop_after

    outer_stop = stop_after(n_rounds)

  training_loop(
    session,
    inner_stop=inner_stop,
    outer_stop=outer_stop,
    on_round_end=on_round_end,
    on_event=callables.get("on_event"),
    round_checkpoint_dir=(
      (lambda r: str(checkpoint_dir / f"round{r:03d}"))
      if checkpoint_dir and n_rounds > 1
      else None
    ),
  )
  session.finish()


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


def _build_ga_round_end(
  config: dict,
  maze_dir: Path,
  checkpoint_dir: Path | None,
  session: object,
) -> object:
  """Auto-construct GA round-end callback from config params."""
  from src.train.adversarial_loop import make_ga_round_end

  return make_ga_round_end(
    maze_dir=maze_dir,
    grid_sizes=config.get("grid_sizes", [6, 8, 10]),
    ga_generations=int(config.get("ga_generations", 50)),
    ga_pop_size=int(config.get("ga_pop_size", 64)),
    target_log_policy_wp=float(config.get("target_log_policy_wp", -1.0)),
    ga_seeds_per_gs=int(config.get("ga_seeds_per_gs", 100)),
    archive_bfs_bins=int(config.get("archive_bfs_bins", 20)),
    archive_states_bins=int(config.get("archive_states_bins", 20)),
    checkpoint_dir=checkpoint_dir or Path("checkpoints/adversarial"),
    seed=int(config.get("seed", 0)),
    dihedral_augment=bool(config.get("dihedral_augment", False)),
    n_rounds=int(config.get("n_rounds", 3)),
    log_fn=session.log_fn,
  )


if __name__ == "__main__":
  try:
    main()
  except Exception as e:
    if "--subprocess=true" in sys.argv or "--subprocess=True" in sys.argv:
      sys.stdout.write(json.dumps({"type": "error", "message": str(e)}) + "\n")
      sys.stdout.flush()
    raise
