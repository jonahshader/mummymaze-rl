"""Replay Python solver solutions through JAX env and verify wins.

Loads cached BFS solutions from solve_all.py, replays each through the
JAX step function using scan+vmap for speed, and asserts every solvable
level is won at the correct step.

Usage:
    uv run python solve_all.py                      # populate cache (once)
    uv run pytest tests/test_jax_solutions.py -v    # run tests
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mummy_maze.parser import parse_file

from src.env.level_load import load_level as jax_load_level
from src.env.step import step as jax_step
from src.env.types import EnvState, LevelData

DAT_DIR = Path(__file__).resolve().parent.parent / "mazes"


def _load_solutions_by_grid() -> dict[
  int, tuple[list[LevelData], list[EnvState], list[list[int]], list[str]]
]:
  """Load cached solutions grouped by grid_size.

  Returns {grid_size: (levels, states, action_seqs, ids)}.
  """
  try:
    from solve_all import load_solutions
  except ImportError:
    pytest.skip("solve_all.py not found")

  try:
    solutions = load_solutions()
  except FileNotFoundError:
    pytest.skip("No solver cache. Run: uv run python solve_all.py")

  groups: dict[
    int, tuple[list[LevelData], list[EnvState], list[list[int]], list[str]]
  ] = {}

  for (file_stem, sub_idx), actions in sorted(solutions.items()):
    parsed = parse_file(DAT_DIR / f"{file_stem}.dat")
    assert parsed is not None
    grid_size, level, state = jax_load_level(parsed.sublevels[sub_idx], parsed.header)
    if grid_size not in groups:
      groups[grid_size] = ([], [], [], [])
    levels, states, action_seqs, ids = groups[grid_size]
    levels.append(level)
    states.append(state)
    action_seqs.append(actions)
    ids.append(f"{file_stem}-{sub_idx}")

  return groups


_GROUPS = _load_solutions_by_grid()


def _stack_pytree(items: list[LevelData] | list[EnvState]) -> LevelData | EnvState:
  """Stack a list of pytree leaves into a single batched pytree."""
  return jax.tree.map(lambda *xs: jnp.stack(xs), *items)


def _run_batch(
  grid_size: int,
  levels: list[LevelData],
  states: list[EnvState],
  action_seqs: list[list[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Replay all solutions for one grid_size via scan+vmap.

  Returns (won, died_early, died_step) arrays of shape (n_levels,).
  """
  n = len(levels)
  max_steps = max(len(a) for a in action_seqs)

  # Pad action sequences to uniform length (pad with WAIT=4)
  actions_padded = np.full((n, max_steps), 4, dtype=np.int32)
  sol_lengths = np.zeros(n, dtype=np.int32)
  for i, actions in enumerate(action_seqs):
    actions_padded[i, : len(actions)] = actions
    sol_lengths[i] = len(actions)
  actions_jax = jnp.array(actions_padded)  # (n, max_steps)

  # Stack levels and states into batched pytrees
  batch_level = _stack_pytree(levels)
  batch_state = _stack_pytree(states)

  # scan body: step one action across all levels
  def scan_body(carry: EnvState, action_col: jnp.ndarray) -> tuple[EnvState, EnvState]:
    new_state = jax.vmap(lambda lv, s, a: jax_step(grid_size, lv, s, a))(
      carry[0], carry[1], action_col
    )
    return (carry[0], new_state), new_state

  # JIT the scan over all timesteps
  @jax.jit
  def run_scan(level: LevelData, state: EnvState, actions: jnp.ndarray) -> EnvState:
    # actions: (n, max_steps) -> transpose to (max_steps, n) for scan
    actions_t = actions.T
    (_, final_state), all_states = jax.lax.scan(scan_body, (level, state), actions_t)
    return final_state, all_states

  final_state, all_states = run_scan(batch_level, batch_state, actions_jax)

  # Check results
  won = np.array(final_state.won)  # (n,)

  # Check for early death: done became True before the solution's last step
  # all_states.done has shape (max_steps, n)
  all_done = np.array(all_states.done)  # (max_steps, n)
  died_early = np.zeros(n, dtype=bool)
  died_step = np.zeros(n, dtype=np.int32)
  for i in range(n):
    length = sol_lengths[i]
    # Check if done at any step before the last
    for t in range(length - 1):
      if all_done[t, i]:
        died_early[i] = True
        died_step[i] = t + 1
        break

  return won, died_early, died_step


@pytest.mark.parametrize("grid_size", sorted(_GROUPS.keys()))
def test_jax_replay_batch(grid_size: int) -> None:
  """Replay all cached solutions for one grid_size through JAX env."""
  levels, states, action_seqs, ids = _GROUPS[grid_size]
  n = len(levels)

  won, died_early, died_step = _run_batch(grid_size, levels, states, action_seqs)

  # Collect failures for a clear report
  failures = []
  for i in range(n):
    if died_early[i]:
      failures.append(f"  {ids[i]}: died at step {died_step[i]}/{len(action_seqs[i])}")
    elif not won[i]:
      failures.append(f"  {ids[i]}: not won after {len(action_seqs[i])} moves")

  if failures:
    n_pass = n - len(failures)
    header = (
      f"grid_size={grid_size}: {len(failures)}/{n} levels failed ({n_pass}/{n} passed)"
    )
    # Show first 20 failures
    detail = "\n".join(failures[:20])
    if len(failures) > 20:
      detail += f"\n  ... and {len(failures) - 20} more"
    pytest.fail(f"{header}\n{detail}")
