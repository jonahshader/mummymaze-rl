"""MAP-Elites adversarial training loop.

Train → generate hard levels via GA → retrain on augmented data → repeat.
The GA targets a Goldilocks zone of difficulty and MAP-Elites ensures diversity.

Usage:
  uv run python -m src.train.adversarial_loop --mazes mazes/ --n-rounds 3
"""

import enum
import functools
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import equinox as eqx
import typer
import jax
import jax.random as jr
import mummymaze_rust
import numpy as np
import optax

from src.env.obs import observe
from src.env.types import EnvState, LevelData
from src.train.augment import (
  apply_dihedral_augmentation,
  augment_dataset,
)
from src.train.dataset import load_bc_dataset
from src.train.ga import GenerationResult, MapElitesArchive, run_ga
from src.train.model import DEFAULT_ARCH, make_model
from src.train.reporter import FileReporter, MetricsReporter
from src.train.train_bc import train_epochs


def _make_obs_and_forward(
  model: eqx.Module,
) -> Callable:
  """Build a JIT'd function: (grid_size, LevelData, EnvState) → logits."""

  @functools.partial(jax.jit, static_argnums=(0,))
  def obs_and_forward(
    grid_size: int,
    level_data: LevelData,
    env_states: EnvState,
  ) -> jax.Array:
    obs = jax.vmap(lambda es: observe(grid_size, level_data, es))(
      env_states,
    )
    return jax.vmap(model)(obs)

  return obs_and_forward


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
  ga_seeds_per_gs: int = 100,
  archive_bfs_bins: int = 20,
  archive_states_bins: int = 20,
  grid_sizes: list[int] | None = None,
  checkpoint_dir: Path = Path("checkpoints/adversarial"),
  metrics_path: Path = Path("level_metrics_adversarial.json"),
  arch: str = DEFAULT_ARCH,
  reporter: MetricsReporter | None = None,
  on_event: Callable[[dict], None] | None = None,
  dihedral_augment: bool = False,
  wandb_project: str | None = None,
) -> None:
  """Run the adversarial training loop.

  Args:
    reporter: Training metrics reporter. Defaults to FileReporter.
    on_event: Callback for adversarial-specific events (round_start,
      ga_generation, archive_update, round_end, done).
    wandb_project: If set, log training metrics to this W&B project.
  """
  if grid_sizes is None:
    grid_sizes = [6, 8, 10]

  if reporter is None:
    reporter = FileReporter(metrics_path)

  def _emit(event: dict) -> None:
    if on_event is not None:
      on_event(event)

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

  if dihedral_augment:
    banks = {gs: ds.bank for gs, ds in datasets.items()}
    datasets = apply_dihedral_augmentation(
      datasets,
      sources,
      banks,
      maze_dir,
    )

  # Initialize model
  key, model_key = jr.split(key)
  model = make_model(arch, model_key)
  n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
  print(f"Model: {n_params:,} parameters")

  # Optimizer — step budget is based on initial dataset size.
  # As the dataset grows from GA augmentation, later rounds will complete
  # fewer epochs but the total step count (and LR schedule) stays fixed.
  initial_steps_per_epoch = sum(
    int(ds.train_mask.sum()) // batch_size for ds in datasets.values()
  )
  steps_per_round = initial_steps_per_epoch * epochs_per_round
  total_steps = steps_per_round * n_rounds

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

  # wandb init
  run_id = f"adversarial-{arch}-{seed}"
  if wandb_project is not None:
    import wandb

    wandb.init(
      project=wandb_project,
      name=run_id,
      config={
        "mode": "adversarial",
        "arch": arch,
        "n_rounds": n_rounds,
        "epochs_per_round": epochs_per_round,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "n_params": n_params,
        "ga_generations": ga_generations,
        "ga_pop_size": ga_pop_size,
        "target_log_policy_wp": target_log_policy_wp,
        "dihedral_augment": dihedral_augment,
      },
    )

  global_step = 0
  global_epoch = 0
  total_ga_levels = 0

  for round_idx in range(n_rounds):
    round_t0 = time.time()
    print(f"\n{'=' * 60}")
    print(f"Round {round_idx}/{n_rounds - 1}")
    print(f"{'=' * 60}")
    _emit(
      {
        "type": "round_start",
        "round": round_idx,
        "n_rounds": n_rounds,
      }
    )

    # --- Train Phase ---
    print(f"\n--- Training ({epochs_per_round} epochs) ---")
    round_ckpt_dir = checkpoint_dir / f"round{round_idx:03d}"
    round_max_steps = global_step + steps_per_round
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
      wandb_project=wandb_project,
      run_id=run_id,
      arch=arch,
      lr=lr,
      max_steps=round_max_steps,
    )
    global_epoch += epochs_per_round

    # Skip GA on the last round (no point generating levels we won't train on)
    if round_idx == n_rounds - 1:
      print("\nLast round — skipping GA phase.")
      continue

    # --- GA Phase ---
    print(f"\n--- GA Phase (grid_sizes={grid_sizes}) ---")

    fitness_expr = f"-abs(log_policy_win_prob - ({target_log_policy_wp}))"
    obs_and_forward = _make_obs_and_forward(model)

    round_ga_levels: list[mummymaze_rust.Level] = []

    for gs in grid_sizes:
      if gs not in datasets:
        continue

      print(f"\n  Grid size {gs}:")

      # Sample 100 random train-only levels as GA seeds
      bank = datasets[gs].bank
      train_set = set(np.array(bank.train_indices).tolist())
      all_sources = sources[gs]
      train_sources = [
        all_sources[i] for i in range(len(all_sources)) if i in train_set
      ]
      key, seed_key = jr.split(key)
      n_seeds = min(ga_seeds_per_gs, len(train_sources))
      perm = jr.permutation(seed_key, len(train_sources))[:n_seeds]
      gs_sources = [train_sources[int(i)] for i in perm]
      by_stem: dict[str, list[int]] = {}
      for stem, sub in gs_sources:
        by_stem.setdefault(stem, []).append(sub)
      seed_levels: list[mummymaze_rust.Level] = []
      for stem, subs in by_stem.items():
        try:
          all_in_file = mummymaze_rust.parse_file(
            str(maze_dir / f"{stem}.dat"),
          )
          for sub in subs:
            if sub < len(all_in_file):
              seed_levels.append(all_in_file[sub])
        except Exception:
          continue

      if not seed_levels:
        print(f"    No seed levels found for gs={gs}, skipping")
        continue

      print(f"    Seeds: {len(seed_levels)} levels")

      archive = MapElitesArchive(
        bfs_range=(1, 50),
        states_range=(1, 500),
        bfs_bins=archive_bfs_bins,
        states_bins=archive_states_bins,
        target_log_wp=target_log_policy_wp,
      )

      def _on_generation(r: GenerationResult) -> None:
        print(
          f"    Gen {r.generation}: best={r.best.fitness:.3f} "
          f"avg={r.avg_fitness:.3f} pop={r.pop_size}"
        )
        _emit(
          {
            "type": "ga_generation",
            "round": round_idx,
            "grid_size": gs,
            "generation": r.generation,
            "best_fitness": r.best.fitness,
            "avg_fitness": r.avg_fitness,
            "solvable_rate": r.solvable_rate,
            "pop_size": r.pop_size,
          }
        )

      ga_t0 = time.time()
      _population, archive = run_ga(
        seed_levels,
        obs_and_forward=obs_and_forward,
        pop_size=ga_pop_size,
        generations=ga_generations,
        fitness_expr=fitness_expr,
        seed=seed + round_idx + 1,
        archive=archive,
        on_generation=_on_generation,
      )
      ga_time = time.time() - ga_t0

      assert archive is not None
      archive_individuals = archive.all_individuals()
      n_archive = len(archive_individuals)
      occupied, total = archive.occupancy()
      print(f"    Archive: {n_archive} levels ({ga_time:.1f}s)")

      # Build enriched cells array (flat row-major [bfs_bin * states_bins + states_bin])
      archive_cells: list[dict | None] = []
      for bi in range(archive.bfs_bins):
        for si in range(archive.states_bins):
          ind = archive.grid[bi][si]
          if ind is None:
            archive_cells.append(None)
          else:
            archive_cells.append(
              {
                "level_json": ind.level.to_json(),
                "bfs_moves": ind.bfs_moves,
                "n_states": ind.n_states,
                "log_policy_win_prob": ind.log_policy_win_prob,
                "fitness": ind.fitness,
              }
            )

      _emit(
        {
          "type": "archive_update",
          "round": round_idx,
          "grid_size": gs,
          "n_levels": n_archive,
          "occupancy": occupied,
          "total_cells": total,
          "time": ga_time,
          "bfs_bins": archive.bfs_bins,
          "states_bins": archive.states_bins,
          "bfs_range": list(archive.bfs_range),
          "states_range": list(archive.states_range),
          "target_log_wp": archive.target_log_wp,
          "cells": archive_cells,
        }
      )

      # Save enriched per-grid-size archive file
      gs_archive_path = round_ckpt_dir / f"archive_gs{gs}.json"
      gs_archive_path.parent.mkdir(parents=True, exist_ok=True)
      gs_archive_data = {
        "bfs_bins": archive.bfs_bins,
        "states_bins": archive.states_bins,
        "bfs_range": list(archive.bfs_range),
        "states_range": list(archive.states_range),
        "target_log_wp": archive.target_log_wp,
        "grid_size": gs,
        "cells": archive_cells,
      }
      with open(gs_archive_path, "w") as f:
        json.dump(gs_archive_data, f)
      print(f"    Saved enriched archive to {gs_archive_path}")

      for ind in archive_individuals:
        round_ga_levels.append(ind.level)
        if n_archive <= 5:
          print(
            f"      bfs={ind.bfs_moves} states={ind.n_states} "
            f"log_pwp={ind.log_policy_win_prob:.2f}"
          )

    if not round_ga_levels:
      print("\n  No GA levels generated, continuing without augmentation")
      continue

    total_ga_levels += len(round_ga_levels)

    # --- Augment Phase ---
    print(f"\n--- Augmenting dataset with {len(round_ga_levels)} GA levels ---")
    datasets = augment_dataset(
      datasets,
      round_ga_levels,
      dihedral_augment=dihedral_augment,
    )
    for gs, ds in sorted(datasets.items()):
      n_train = int(ds.train_mask.sum())
      print(f"  grid_size={gs}: {ds.n_states} states ({n_train} train)")

    # Log adversarial-specific metrics to wandb
    if wandb_project is not None:
      import wandb

      total_train_states = sum(int(ds.train_mask.sum()) for ds in datasets.values())
      total_train_levels = sum(len(ds.bank.train_indices) for ds in datasets.values())
      wandb.log(
        {
          "adversarial/round": round_idx,
          "adversarial/ga_levels_added": len(round_ga_levels),
          "adversarial/total_ga_levels": total_ga_levels,
          "adversarial/total_train_states": total_train_states,
          "adversarial/total_train_levels": total_train_levels,
        },
        step=global_step,
      )

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
    _emit(
      {
        "type": "round_end",
        "round": round_idx,
        "time": round_time,
        "ga_levels": len(round_ga_levels),
      }
    )

  reporter.report_done()
  print(f"\nAdversarial loop complete. Total GA levels: {total_ga_levels}")
  _emit(
    {
      "type": "done",
      "total_ga_levels": total_ga_levels,
    }
  )

  if wandb_project is not None:
    import wandb

    wandb.finish()


class Arch(str, enum.Enum):
  cnn = "cnn"
  resnet = "resnet"


def main(
  mazes: Annotated[
    Path, typer.Option(help="Directory containing B-*.dat files")
  ] = Path("mazes"),
  n_rounds: int = 3,
  epochs_per_round: int = 5,
  batch_size: int = 1024,
  lr: float = 3e-4,
  seed: int = 0,
  ga_generations: int = 50,
  ga_pop_size: int = 64,
  target_log_wp: float = -1.0,
  ga_seeds_per_gs: int = 100,
  archive_bfs_bins: int = 20,
  archive_states_bins: int = 20,
  grid_sizes: Annotated[list[int], typer.Option(help="Grid sizes to train on")] = [
    6,
    8,
    10,
  ],
  checkpoint_dir: Annotated[Path, typer.Option(help="Save checkpoints here")] = Path(
    "checkpoints/adversarial"
  ),
  metrics_path: Path = Path("level_metrics_adversarial.json"),
  arch: Annotated[Arch, typer.Option(help="Model architecture")] = Arch.cnn,
  dihedral_augment: Annotated[
    bool, typer.Option(help="Expand training set with dihedral variants")
  ] = False,
  wandb_project: Annotated[str | None, typer.Option(help="W&B project name")] = None,
) -> None:
  """MAP-Elites adversarial training loop."""
  adversarial_loop(
    maze_dir=mazes,
    n_rounds=n_rounds,
    epochs_per_round=epochs_per_round,
    batch_size=batch_size,
    lr=lr,
    seed=seed,
    ga_generations=ga_generations,
    ga_pop_size=ga_pop_size,
    target_log_policy_wp=target_log_wp,
    ga_seeds_per_gs=ga_seeds_per_gs,
    archive_bfs_bins=archive_bfs_bins,
    archive_states_bins=archive_states_bins,
    grid_sizes=grid_sizes,
    checkpoint_dir=checkpoint_dir,
    metrics_path=metrics_path,
    arch=arch.value,
    dihedral_augment=dihedral_augment,
    wandb_project=wandb_project,
  )


if __name__ == "__main__":
  typer.run(main)
