"""MAP-Elites adversarial training loop.

Train → generate hard levels via GA → retrain on augmented data → repeat.
The GA targets a Goldilocks zone of difficulty and MAP-Elites ensures diversity.

Usage:
  uv run python -m src.train.adversarial_loop --mazes mazes/ --n-rounds 3
"""

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.random as jr
import mummymaze_rust
import optax

from src.train.augment import augment_dataset
from src.train.dataset import load_bc_dataset
from src.train.model import MazeCNN
from src.train.reporter import FileReporter
from src.train.train_bc import train_epochs


def adversarial_loop(
  maze_dir: Path,
  *,
  n_rounds: int = 3,
  epochs_per_round: int = 5,
  batch_size: int = 1024,
  lr: float = 3e-4,
  seed: int = 0,
  ga_generations: int = 50,
  ga_pop_size: int = 64,
  target_log_policy_wp: float = -1.0,
  archive_bfs_bins: int = 20,
  archive_states_bins: int = 20,
  grid_sizes: list[int] | None = None,
  checkpoint_dir: Path = Path("checkpoints/adversarial"),
  metrics_path: Path = Path("level_metrics_adversarial.json"),
) -> None:
  """Run the adversarial training loop."""
  if grid_sizes is None:
    grid_sizes = [6, 8, 10]

  reporter = FileReporter(metrics_path)
  key = jr.key(seed)

  # Load base dataset
  print("Loading base dataset...")
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
  steps_per_epoch = sum(
    int(ds.train_mask.sum()) // batch_size for ds in datasets.values()
  )
  total_steps = steps_per_epoch * epochs_per_round * n_rounds

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

  reporter.report_init(
    {
      "n_params": n_params,
      "n_rounds": n_rounds,
      "epochs_per_round": epochs_per_round,
      "batch_size": batch_size,
      "lr": lr,
      "seed": seed,
      "ga_generations": ga_generations,
      "ga_pop_size": ga_pop_size,
      "target_log_policy_wp": target_log_policy_wp,
      "datasets": {
        str(gs): {"n_states": ds.n_states, "n_levels": ds.n_levels}
        for gs, ds in datasets.items()
      },
    }
  )

  global_step = 0
  global_epoch = 0
  total_ga_levels = 0

  for round_idx in range(n_rounds):
    round_t0 = time.time()
    print(f"\n{'=' * 60}")
    print(f"Round {round_idx}/{n_rounds - 1}")
    print(f"{'=' * 60}")

    # --- Train Phase ---
    print(f"\n--- Training ({epochs_per_round} epochs) ---")
    round_ckpt_dir = checkpoint_dir / f"round{round_idx:03d}"
    model, opt_state, global_step, key = train_epochs(
      model,
      opt_state,
      optimizer,
      datasets,
      sources,
      epochs=epochs_per_round,
      batch_size=batch_size,
      checkpoint_dir=round_ckpt_dir,
      reporter=reporter,
      maze_dir=maze_dir,
      key=key,
      epoch_offset=global_epoch,
      step_offset=global_step,
      run_id=f"adversarial-r{round_idx}",
    )
    global_epoch += epochs_per_round

    # Save latest checkpoint for GA to reference
    latest_ckpt = round_ckpt_dir / f"model_epoch{global_epoch:03d}.eqx"
    if not latest_ckpt.exists():
      # Find the most recent checkpoint in this round's dir
      ckpts = sorted(round_ckpt_dir.glob("model_epoch*.eqx"))
      if ckpts:
        latest_ckpt = ckpts[-1]
      else:
        print("WARNING: No checkpoint found, skipping GA phase")
        continue

    # Skip GA on the last round (no point generating levels we won't train on)
    if round_idx == n_rounds - 1:
      print("\nLast round — skipping GA phase.")
      continue

    # --- GA Phase ---
    print(f"\n--- GA Phase (grid_sizes={grid_sizes}) ---")

    # Collect seed levels for each grid_size from the maze directory
    ga_config = {
      "pop_size": ga_pop_size,
      "generations": ga_generations,
      "elite_frac": 0.1,
      "crossover_rate": 0.2,
      "extra_wall_prob": 0.3,
      "fitness_expr": f"-abs(log_policy_win_prob - ({target_log_policy_wp}))",
      "seed": seed + round_idx + 1,
    }

    round_ga_levels: list[mummymaze_rust.Level] = []

    for gs in grid_sizes:
      if gs not in datasets:
        continue

      print(f"\n  Grid size {gs}:")

      # Use a subset of the training levels as seeds (batch-parse per .dat file)
      gs_sources = sources[gs][:100]
      by_stem: dict[str, list[int]] = {}
      for stem, sub in gs_sources:
        by_stem.setdefault(stem, []).append(sub)
      seed_levels = []
      for stem, subs in by_stem.items():
        try:
          all_in_file = mummymaze_rust.parse_file(str(maze_dir / f"{stem}.dat"))
          for sub in subs:
            if sub < len(all_in_file):
              seed_levels.append(all_in_file[sub])
        except Exception:
          continue

      if not seed_levels:
        print(f"    No seed levels found for gs={gs}, skipping")
        continue

      print(f"    Seeds: {len(seed_levels)} levels")
      gs_config = dict(ga_config, grid_size=gs)

      ga_t0 = time.time()
      archive_results = mummymaze_rust.run_ga_round(
        seed_levels,
        gs_config,
        str(latest_ckpt),
        target_log_policy_wp,
        archive_bfs_range=(1, 50),
        archive_states_range=(1, 500),
        archive_bfs_bins=archive_bfs_bins,
        archive_states_bins=archive_states_bins,
      )
      ga_time = time.time() - ga_t0

      n_archive = len(archive_results)
      print(f"    Archive: {n_archive} levels ({ga_time:.1f}s)")

      for entry in archive_results:
        round_ga_levels.append(entry["level"])
        if n_archive <= 5:
          print(
            f"      bfs={entry['bfs_moves']} states={entry['n_states']} "
            f"log_pwp={entry['log_policy_win_prob']:.2f}"
          )

    if not round_ga_levels:
      print("\n  No GA levels generated, continuing without augmentation")
      continue

    total_ga_levels += len(round_ga_levels)

    # --- Augment Phase ---
    print(f"\n--- Augmenting dataset with {len(round_ga_levels)} GA levels ---")
    datasets = augment_dataset(datasets, round_ga_levels)
    for gs, ds in sorted(datasets.items()):
      n_train = int(ds.train_mask.sum())
      print(f"  grid_size={gs}: {ds.n_states} states ({n_train} train)")

    # Save archive state
    archive_path = round_ckpt_dir / "archive.json"
    archive_data = []
    for level in round_ga_levels:
      archive_data.append(level.to_dict())
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with open(archive_path, "w") as f:
      json.dump(archive_data, f)
    print(f"  Saved archive to {archive_path}")

    round_time = time.time() - round_t0
    print(f"\nRound {round_idx} complete ({round_time:.1f}s)")

  reporter.report_done()
  print(f"\nAdversarial loop complete. Total GA levels: {total_ga_levels}")


def main() -> None:
  parser = argparse.ArgumentParser(description="MAP-Elites adversarial training loop")
  parser.add_argument(
    "--mazes",
    type=Path,
    default=Path("mazes"),
    help="Directory containing B-*.dat files",
  )
  parser.add_argument("--n-rounds", type=int, default=3)
  parser.add_argument("--epochs-per-round", type=int, default=5)
  parser.add_argument("--batch-size", type=int, default=1024)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--ga-generations", type=int, default=50)
  parser.add_argument("--ga-pop-size", type=int, default=64)
  parser.add_argument("--target-log-wp", type=float, default=-1.0)
  parser.add_argument("--archive-bfs-bins", type=int, default=20)
  parser.add_argument("--archive-states-bins", type=int, default=20)
  parser.add_argument("--grid-sizes", type=int, nargs="+", default=[6, 8, 10])
  parser.add_argument(
    "--checkpoint-dir",
    type=Path,
    default=Path("checkpoints/adversarial"),
  )
  parser.add_argument(
    "--metrics-path",
    type=Path,
    default=Path("level_metrics_adversarial.json"),
  )
  args = parser.parse_args()

  adversarial_loop(
    maze_dir=args.mazes,
    n_rounds=args.n_rounds,
    epochs_per_round=args.epochs_per_round,
    batch_size=args.batch_size,
    lr=args.lr,
    seed=args.seed,
    ga_generations=args.ga_generations,
    ga_pop_size=args.ga_pop_size,
    target_log_policy_wp=args.target_log_wp,
    archive_bfs_bins=args.archive_bfs_bins,
    archive_states_bins=args.archive_states_bins,
    grid_sizes=args.grid_sizes,
    checkpoint_dir=args.checkpoint_dir,
    metrics_path=args.metrics_path,
  )


if __name__ == "__main__":
  main()
