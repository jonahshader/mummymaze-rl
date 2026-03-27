"""Curriculum training experiment — grid size 6.

Trains a policy from scratch on randomly generated solvable levels.
Each round:
  1. Train until mean win prob on current levels exceeds threshold
  2. Run GA to generate new levels targeting ~10% policy win prob
  3. Augment dataset with generated levels
  4. Repeat

Usage:
  uv run python experiments/curriculum_gs6.py
  uv run python experiments/curriculum_gs6.py --seed 42 --n-rounds 5
"""

import argparse
import time
from pathlib import Path

import mummymaze_rust
import numpy as np

from src.train.augment import augment_dataset
from src.train.ga import MapElitesArchive, run_ga
from src.train.inference import make_obs_and_forward
from src.train.ga import compute_policy_win_probs
from src.train.session import setup_training
from src.train.stopping import stop_after
from src.train.train_bc import train_epochs


def main() -> None:
  parser = argparse.ArgumentParser(description="Curriculum training on gs=6")
  parser.add_argument("--maze-dir", type=Path, default=Path("mazes"))
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--n-rounds", type=int, default=10)
  parser.add_argument("--epochs-per-eval", type=int, default=1)
  parser.add_argument("--win-prob-threshold", type=float, default=0.80)
  parser.add_argument("--max-epochs-per-round", type=int, default=20)
  parser.add_argument(
    "--target-log-wp",
    type=float,
    default=-1.0,
    help="Target log10(policy_win_prob) for GA fitness",
  )
  parser.add_argument("--n-random-levels", type=int, default=100)
  parser.add_argument("--ga-generations", type=int, default=50)
  parser.add_argument("--ga-pop-size", type=int, default=64)
  parser.add_argument("--batch-size", type=int, default=1024)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--arch", type=str, default="cnn")
  parser.add_argument("--hparams", type=str, default=None, help="JSON hparams dict")
  parser.add_argument(
    "--checkpoint", type=Path, default=None, help="Resume from checkpoint"
  )
  parser.add_argument("--checkpoint-dir", type=Path, default=None)
  parser.add_argument("--wandb-project", type=str, default=None)
  args = parser.parse_args()

  import json as _json

  grid_size = 6
  total_round_epochs = args.n_rounds * args.max_epochs_per_round
  hparams = _json.loads(args.hparams) if args.hparams else None

  # --- Setup ---
  print(f"Setting up curriculum training (gs={grid_size}, {args.n_rounds} rounds)")
  session = setup_training(
    maze_dir=args.maze_dir,
    arch=args.arch,
    hparams=hparams,
    checkpoint=args.checkpoint,
    epochs=args.epochs_per_eval,
    batch_size=args.batch_size,
    lr=args.lr,
    seed=args.seed,
    wandb_project=args.wandb_project,
    checkpoint_dir=args.checkpoint_dir,
    schedule_epochs=total_round_epochs,
  )

  state = session.state
  datasets = session.datasets
  sources = session.sources

  # Filter to grid_size 6 only for validation
  val_levels_gs6 = _get_val_levels(datasets, sources, grid_size, args.maze_dir)
  print(f"Validation set: {len(val_levels_gs6)} real gs={grid_size} levels")

  # --- Initial levels: GA-generated when resuming, random otherwise ---
  if args.checkpoint is not None:
    # Warm model — go straight to GA for initial level generation
    print("\nResuming from checkpoint — running GA for initial levels...")
    obs_fwd = make_obs_and_forward(state.model)
    fitness_expr = f"-abs(log_policy_win_prob - ({args.target_log_wp}))"

    seed_levels = mummymaze_rust.generate_random_solvable(
      args.n_random_levels,
      grid_size=grid_size,
      seed=args.seed,
    )
    archive = MapElitesArchive(
      bfs_range=(1, 50),
      states_range=(1, 500),
      bfs_bins=20,
      states_bins=20,
      target_log_wp=args.target_log_wp,
    )
    _population, archive = run_ga(
      seed_levels,
      obs_and_forward=obs_fwd,
      pop_size=args.ga_pop_size,
      generations=args.ga_generations,
      fitness_expr=fitness_expr,
      seed=args.seed,
      archive=archive,
    )
    assert archive is not None
    initial_levels = [ind.level for ind in archive.all_individuals()]
    occupied, total = archive.occupancy()
    print(f"  GA: {len(initial_levels)} levels, {occupied}/{total} cells")
  else:
    # Cold start — random levels
    n_rand = args.n_random_levels
    print(f"\nGenerating {n_rand} random solvable levels (gs={grid_size})...")
    t0 = time.time()
    initial_levels = mummymaze_rust.generate_random_solvable(
      args.n_random_levels,
      grid_size=grid_size,
      seed=args.seed,
    )
    print(f"  Generated in {time.time() - t0:.1f}s")

  datasets = augment_dataset(datasets, initial_levels)
  n_states = datasets[grid_size].n_states
  print(f"  Dataset after augment: {n_states} states (gs={grid_size})")

  # --- Curriculum loop ---
  for round_idx in range(args.n_rounds):
    round_t0 = time.time()
    print(f"\n{'=' * 60}")
    print(f"Round {round_idx}/{args.n_rounds - 1}")
    print(f"{'=' * 60}")

    # Train until win-prob threshold or max epochs
    for epoch_in_round in range(args.max_epochs_per_round):
      state = train_epochs(
        state,
        session.config,
        datasets,
        sources,
        session.reporter,
        log_fn=session.log_fn,
        checkpoint_fn=session.checkpoint_fn,
        inner_stop=stop_after(args.epochs_per_eval),
      )
      state.epoch_offset += args.epochs_per_eval

      # Evaluate win prob on real validation levels
      obs_fwd = make_obs_and_forward(state.model)
      log_wps = compute_policy_win_probs(val_levels_gs6, obs_fwd)
      valid_wps = [10**lwp for lwp in log_wps if lwp > float("-inf")]

      if valid_wps:
        mean_wp = float(np.mean(valid_wps))
        median_wp = float(np.median(valid_wps))
      else:
        mean_wp = 0.0
        median_wp = 0.0

      print(
        f"  Epoch {epoch_in_round + 1}: "
        f"mean_wp={mean_wp:.4f} median_wp={median_wp:.4f} "
        f"(threshold={args.win_prob_threshold})"
      )

      if session.log_fn is not None:
        session.log_fn(
          state.global_step,
          {
            "curriculum/mean_win_prob": mean_wp,
            "curriculum/median_win_prob": median_wp,
            "curriculum/round": float(round_idx),
            "curriculum/epoch_in_round": float(epoch_in_round),
          },
        )

      if mean_wp >= args.win_prob_threshold:
        print(f"  Threshold reached after {epoch_in_round + 1} epochs")
        break
    else:
      print(f"  Max epochs ({args.max_epochs_per_round}) reached without threshold")

    # Skip GA on last round
    if round_idx == args.n_rounds - 1:
      print("\nLast round — skipping level generation.")
      continue

    # --- GA phase: generate new challenging levels ---
    print(f"\n--- GA Phase (targeting log_wp={args.target_log_wp}) ---")
    obs_fwd = make_obs_and_forward(state.model)
    fitness_expr = f"-abs(log_policy_win_prob - ({args.target_log_wp}))"

    # Fresh random seeds for GA each round
    seed_levels = mummymaze_rust.generate_random_solvable(
      args.n_random_levels,
      grid_size=grid_size,
      seed=args.seed + round_idx + 1000,
    )

    archive = MapElitesArchive(
      bfs_range=(1, 50),
      states_range=(1, 500),
      bfs_bins=20,
      states_bins=20,
      target_log_wp=args.target_log_wp,
    )

    def _on_gen(r):  # noqa: ANN001
      print(
        f"  Gen {r.generation}: best={r.best.fitness:.3f} "
        f"avg={r.avg_fitness:.3f} pop={r.pop_size}"
      )

    ga_t0 = time.time()
    _population, archive = run_ga(
      seed_levels,
      obs_and_forward=obs_fwd,
      pop_size=args.ga_pop_size,
      generations=args.ga_generations,
      fitness_expr=fitness_expr,
      seed=args.seed + round_idx + 1,
      archive=archive,
      on_generation=_on_gen,
    )
    ga_time = time.time() - ga_t0

    assert archive is not None
    new_levels = [ind.level for ind in archive.all_individuals()]
    occupied, total = archive.occupancy()
    n_new = len(new_levels)
    print(f"  Archive: {n_new} levels, {occupied}/{total} cells ({ga_time:.1f}s)")

    if new_levels:
      datasets = augment_dataset(datasets, new_levels)
      n_train = int(datasets[grid_size].train_mask.sum())
      print(f"  Dataset after augment: {n_train} train states (gs={grid_size})")

    round_time = time.time() - round_t0
    print(f"\nRound {round_idx} complete ({round_time:.1f}s)")

  session.finish()
  print("\nCurriculum training complete.")


def _get_val_levels(
  datasets: dict,
  sources: dict,
  grid_size: int,
  maze_dir: Path,
) -> list[mummymaze_rust.Level]:
  """Extract real validation levels for a grid size as Rust Level objects."""
  if grid_size not in datasets or grid_size not in sources:
    return []

  ds = datasets[grid_size]
  bank = ds.bank
  val_set = set(int(x) for x in np.array(bank.val_indices))
  src_list = sources[grid_size]

  # Group by file to avoid re-parsing
  by_file: dict[str, list[tuple[int, int]]] = {}
  for idx in val_set:
    if idx < len(src_list):
      stem, sub = src_list[idx]
      by_file.setdefault(stem, []).append((sub, idx))

  levels = []
  for stem, entries in by_file.items():
    try:
      all_in_file = mummymaze_rust.parse_file(str(maze_dir / f"{stem}.dat"))
      for sub, _idx in entries:
        if sub < len(all_in_file):
          levels.append(all_in_file[sub])
    except Exception:
      continue

  return levels


if __name__ == "__main__":
  main()
