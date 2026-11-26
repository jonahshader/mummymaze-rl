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
from src.train.ga import (
  compute_policy_win_probs_batched,
  run_multi_ga,
)
from src.train.session import setup_training
from src.train.stopping import stop_after
from src.train.timer import SectionTimer
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
  parser.add_argument("--ga-pop-size", type=int, default=64, help="Per-run pop size")
  parser.add_argument("--n-ga-runs", type=int, default=32)
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
  # Constant LR for curriculum: dataset size changes each round so cosine
  # schedule (which needs total_steps upfront) doesn't apply.
  import optax

  constant_optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(args.lr),
  )

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
    optimizer=constant_optimizer,
  )

  state = session.state

  # Extract gs=6 validation levels from real data, then discard real training data.
  # Curriculum trains ONLY on GA-generated levels.
  val_levels_gs6 = _get_val_levels(
    session.datasets, session.sources, grid_size, args.maze_dir
  )
  print(f"Validation set: {len(val_levels_gs6)} real gs={grid_size} levels")

  datasets = _empty_curriculum_datasets(session.datasets, grid_size)
  sources = {grid_size: []}

  # --- Initial levels: GA-generated when resuming, random otherwise ---
  if args.checkpoint is not None:
    # Warm model — go straight to GA for initial level generation
    print("\nResuming from checkpoint — running multi-GA for initial levels...")
    fitness_expr = f"-abs(log_policy_win_prob - ({args.target_log_wp}))"

    seed_levels = mummymaze_rust.generate_random_solvable(
      args.n_random_levels,
      grid_size=grid_size,
      seed=args.seed,
    )
    best_per_run = run_multi_ga(
      args.n_ga_runs,
      seed_levels,
      model=state.model,
      grid_size=grid_size,
      pop_size=args.ga_pop_size,
      generations=args.ga_generations,
      fitness_expr=fitness_expr,
      seed=args.seed,
    )
    initial_levels = [ind.level for ind in best_per_run]
    print(f"  Multi-GA: {len(initial_levels)} levels")
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

  # Save initial levels
  levels_dir = Path(args.checkpoint_dir) / "levels" if args.checkpoint_dir else None
  if levels_dir:
    levels_dir.mkdir(parents=True, exist_ok=True)
    path = levels_dir / "round_initial.jsonl"
    _save_levels_jsonl(path, initial_levels)
    print(f"  Saved {len(initial_levels)} levels to {path}")

  # --- Curriculum loop ---
  # Track current round's levels for threshold evaluation
  round_levels: list[mummymaze_rust.Level] = list(initial_levels)
  timer = SectionTimer()

  for round_idx in range(args.n_rounds):
    print(f"\n{'=' * 60}")
    print(f"Round {round_idx}/{args.n_rounds - 1}")
    print(f"{'=' * 60}")

    # Train until model masters current round's levels or max epochs
    for epoch_in_round in range(args.max_epochs_per_round):
      with timer("train"):
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

      # Threshold check: win prob on THIS ROUND'S levels only
      with timer("eval_round"):
        round_log_wps = compute_policy_win_probs_batched(
          round_levels,
          state.model,
          grid_size,
        )
      round_wps = [10**lwp for lwp in round_log_wps if lwp > float("-inf")]
      round_mean = float(np.mean(round_wps)) if round_wps else 0.0

      # Monitoring: win prob on real validation levels (not used for stopping)
      with timer("eval_val"):
        val_log_wps = compute_policy_win_probs_batched(
          val_levels_gs6,
          state.model,
          grid_size,
        )
      val_wps = [10**lwp for lwp in val_log_wps if lwp > float("-inf")]
      val_mean = float(np.mean(val_wps)) if val_wps else 0.0

      print(
        f"  Epoch {epoch_in_round + 1}: "
        f"round_wp={round_mean:.4f} val_wp={val_mean:.4f} "
        f"(threshold={args.win_prob_threshold})"
      )

      if session.log_fn is not None:
        session.log_fn(
          state.global_step,
          {
            "curriculum/round_mean_win_prob": round_mean,
            "curriculum/val_mean_win_prob": val_mean,
            "curriculum/round": float(round_idx),
            "curriculum/epoch_in_round": float(epoch_in_round),
            "curriculum/n_round_levels": float(len(round_levels)),
            "curriculum/n_total_states": float(datasets[grid_size].n_states),
          },
        )

      if round_mean >= args.win_prob_threshold:
        print(f"  Threshold reached after {epoch_in_round + 1} epochs")
        break
    else:
      print(
        f"  Max epochs ({args.max_epochs_per_round}) reached "
        f"(round_wp={round_mean:.4f})"
      )

    # Skip GA on last round
    if round_idx == args.n_rounds - 1:
      print("\nLast round — skipping level generation.")
      continue

    # --- GA phase: generate new challenging levels ---
    print(
      f"\n--- Multi-GA Phase "
      f"({args.n_ga_runs} runs × {args.ga_pop_size}, "
      f"targeting log_wp={args.target_log_wp}) ---"
    )
    fitness_expr = f"-abs(log_policy_win_prob - ({args.target_log_wp}))"

    # Fresh random seeds for GA each round
    seed_levels = mummymaze_rust.generate_random_solvable(
      args.n_random_levels,
      grid_size=grid_size,
      seed=args.seed + round_idx + 1000,
    )

    def _on_gen(r):  # noqa: ANN001
      print(
        f"  Gen {r.generation}: best={r.best_overall.fitness:.3f} "
        f"avg={r.avg_fitness:.3f} pop={r.total_pop} "
        f"solvable={r.solvable_rate:.0%}"
      )

    with timer("ga"):
      best_per_run = run_multi_ga(
        args.n_ga_runs,
        seed_levels,
        model=state.model,
        grid_size=grid_size,
        pop_size=args.ga_pop_size,
        generations=args.ga_generations,
        fitness_expr=fitness_expr,
        seed=args.seed + round_idx + 1,
        on_generation=_on_gen,
      )

    round_levels = [ind.level for ind in best_per_run]
    n_new = len(round_levels)
    print(f"  Multi-GA: {n_new} levels (1 best per run)")

    if round_levels:
      with timer("augment"):
        datasets = augment_dataset(datasets, round_levels)
      n_train = int(datasets[grid_size].train_mask.sum())
      print(f"  Dataset after augment: {n_train} train states (gs={grid_size})")

      if levels_dir:
        path = levels_dir / f"round_{round_idx:03d}.jsonl"
        _save_levels_jsonl(path, round_levels)
        print(f"  Saved {len(round_levels)} levels to {path}")

    # Log per-round timing
    round_times = timer.round_times()
    timing_str = " | ".join(f"{k}={v:.1f}s" for k, v in round_times.items() if v > 0)
    print(f"  Timing: {timing_str}")

    if session.log_fn is not None:
      session.log_fn(
        state.global_step,
        {f"timing/{k}": v for k, v in round_times.items()},
      )

  print(f"\n{timer.summary()}")
  session.finish()
  print("\nCurriculum training complete.")


def _save_levels_jsonl(
  path: Path,
  levels: list[mummymaze_rust.Level],
) -> None:
  """Save levels as JSONL (one JSON-serialized level per line)."""
  with open(path, "w") as f:
    for level in levels:
      f.write(level.to_json() + "\n")


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


def _empty_curriculum_datasets(
  real_datasets: dict,
  grid_size: int,
) -> dict:
  """Build a gs=6-only dataset dict with zero training states.

  Keeps the LevelBank (needed for obs computation when augment_dataset
  appends GA levels) but clears all state arrays and masks. The first
  augment_dataset call populates it with GA-generated training data.
  """
  import jax.numpy as jnp

  from src.train.dataset import BCDataset

  real_ds = real_datasets[grid_size]
  bank = real_ds.bank

  empty = BCDataset(
    grid_size=grid_size,
    n_levels=bank.n_levels,
    n_states=0,
    bank=bank,
    state_tuples=jnp.zeros((0, 12), dtype=jnp.int32),
    action_targets=jnp.zeros((0, 5), dtype=jnp.float32),
    level_idx=jnp.zeros(0, dtype=jnp.int32),
    train_mask=jnp.zeros(0, dtype=jnp.bool_),
    val_mask=jnp.zeros(0, dtype=jnp.bool_),
  )
  return {grid_size: empty}


if __name__ == "__main__":
  main()
