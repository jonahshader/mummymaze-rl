"""Random baseline agent: uniform-random valid actions, per-level win rates."""

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from src.env.env import EnvOut, MummyMazeEnv
from src.env.level_bank import LevelBank, get_level, load_all_levels
from src.env.mechanics import can_move
from src.env.types import EnvState, LevelData

NUM_ACTIONS = 5
HEAD_TAIL = 10

HIST_BINS = [0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.01]
HIST_LABELS = [
  "   0%",
  " <1%",
  "1-5%",
  "5-10%",
  "10-25%",
  "25-50%",
  "50-100%",
]


def valid_action_mask(
  grid_size: int,
  level: LevelData,
  state: EnvState,
) -> Bool[Array, "5"]:
  """Return a boolean mask of which actions are valid from the current state."""
  r, c = state.player[0], state.player[1]
  mask = jnp.array(
    [can_move(grid_size, level, state.gate_open, r, c, jnp.int32(a)) for a in range(4)]
    + [True]
  )  # wait is always valid
  return mask


def sample_valid_action(
  grid_size: int,
  level: LevelData,
  state: EnvState,
  key: PRNGKeyArray,
) -> Int[Array, ""]:
  """Sample uniformly from valid actions."""
  mask = valid_action_mask(grid_size, level, state)
  probs = mask / mask.sum()
  return jr.choice(key, NUM_ACTIONS, p=probs)


def run_episode(
  grid_size: int,
  level: LevelData,
  max_steps: int,
  key: PRNGKeyArray,
) -> Bool[Array, ""]:
  """Run one episode with random valid actions, return whether the agent won."""
  env = MummyMazeEnv(grid_size)
  state, _obs = env.reset(level)

  def scan_step(
    carry: tuple[EnvState, Bool[Array, ""]], step_key: PRNGKeyArray
  ) -> tuple[tuple[EnvState, Bool[Array, ""]], None]:
    state, won = carry
    action = sample_valid_action(grid_size, level, state, step_key)
    out: EnvOut = env.step(level, state, action)
    # Once done, keep the terminal state (steps become no-ops)
    new_state = jax.tree.map(
      lambda old, new: jnp.where(state.done, old, new), state, out.state
    )
    new_won = won | out.state.won
    return (new_state, new_won), None

  keys = jr.split(key, max_steps)
  (final_state, won), _ = jax.lax.scan(scan_step, (state, jnp.bool_(False)), keys)
  return won


def eval_level(
  grid_size: int,
  level: LevelData,
  episodes: int,
  max_steps: int,
  key: PRNGKeyArray,
) -> Float[Array, ""]:
  """Run many episodes on one level, return win rate."""
  keys = jr.split(key, episodes)
  episode_fn = jax.vmap(lambda k: run_episode(grid_size, level, max_steps, k))
  wins = episode_fn(keys)
  return wins.mean()


def print_histogram(rates: list[float]) -> None:
  """Print an ASCII histogram of win rates."""
  counts = [0] * len(HIST_LABELS)
  for r in rates:
    for i in range(len(HIST_BINS) - 1):
      if HIST_BINS[i] <= r < HIST_BINS[i + 1]:
        counts[i] += 1
        break
  max_count = max(counts) if counts else 1
  bar_width = 40
  for label, count in zip(HIST_LABELS, counts):
    bar = "#" * int(count / max_count * bar_width)
    print(f"  {label}  {bar} {count}")


def print_level_line(
  idx: int, wr: float, source: tuple[str, int] | None = None
) -> None:
  bar = "#" * int(wr * 30)
  src = f"  ({source[0]} sub {source[1]})" if source else ""
  print(f"  Level {idx:4d}: {wr:6.2%}  {bar}{src}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Random baseline for Mummy Maze")
  parser.add_argument("--mazes", type=Path, default=Path("mazes"))
  parser.add_argument("--episodes", type=int, default=1000)
  parser.add_argument("--max-steps", type=int, default=100)
  parser.add_argument("--out", type=Path, default=Path("random_baseline.csv"))
  args = parser.parse_args()

  print(f"Loading levels from {args.mazes} ...")
  banks, sources = load_all_levels(args.mazes)

  key = jr.key(0)
  all_rows: list[dict[str, object]] = []

  for gs in sorted(banks):
    bank: LevelBank = banks[gs]
    gs_sources = sources[gs]
    n = bank.n_levels
    print(f"\n{'=' * 60}")
    print(
      f"Grid size {gs}: {n} levels,"
      f" {args.episodes} episodes, {args.max_steps} max steps"
    )
    print(f"{'=' * 60}")

    # JIT the eval function for this grid size
    n_ep, ms = args.episodes, args.max_steps
    eval_fn = jax.jit(lambda level, k: eval_level(gs, level, n_ep, ms, k))

    # Warmup JIT on first level
    first_level = get_level(bank, jnp.int32(0))
    key, sub = jr.split(key)
    _ = eval_fn(first_level, sub).block_until_ready()
    print("JIT compiled, running evaluation...")

    win_rates: list[tuple[int, float]] = []
    for i in range(n):
      level = get_level(bank, jnp.int32(i))
      key, sub = jr.split(key)
      wr = float(eval_fn(level, sub))
      win_rates.append((i, wr))
      file_stem, sub_idx = gs_sources[i]
      all_rows.append(
        {
          "grid_size": gs,
          "level_idx": i,
          "file": file_stem,
          "sublevel": sub_idx,
          "win_rate": wr,
        }
      )

    # Sort by win rate (hardest first)
    win_rates.sort(key=lambda x: x[1])
    rates = [wr for _, wr in win_rates]

    # Histogram
    print("\n  Distribution:")
    print_histogram(rates)

    # Head (easiest) and tail (hardest)
    show = min(HEAD_TAIL, len(win_rates))
    print(f"\n  Easiest {show}:")
    for idx, wr in win_rates[-show:]:
      print_level_line(idx, wr, gs_sources[idx])
    print(f"\n  Hardest {show}:")
    for idx, wr in win_rates[:show]:
      print_level_line(idx, wr, gs_sources[idx])

    # Summary
    print(f"\n  Mean:   {sum(rates) / len(rates):.2%}")
    print(f"  Median: {sorted(rates)[len(rates) // 2]:.2%}")
    print(f"  Min:    {min(rates):.2%}")
    print(f"  Max:    {max(rates):.2%}")
    n_zero = sum(1 for r in rates if r == 0.0)
    print(f"  Unsolved (0%): {n_zero}/{n}")

  # Write CSV
  with open(args.out, "w", newline="") as f:
    writer = csv.DictWriter(
      f, fieldnames=["grid_size", "level_idx", "file", "sublevel", "win_rate"]
    )
    writer.writeheader()
    writer.writerows(all_rows)
  print(f"\nResults written to {args.out}")


if __name__ == "__main__":
  main()
