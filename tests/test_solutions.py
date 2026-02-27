"""Validate game logic by matching walkthrough solutions to levels.

Parses solutions from solutions.txt, then brute-force matches each solution
against all sublevels to find which level it solves. Validates that:
1. The solution results in a win (player reaches exit alive)
2. Both Python and JAX engines agree on the outcome at every step
"""

import re
import time
from collections.abc import Callable
from pathlib import Path

import jax.numpy as jnp
import pytest
from mummy_maze.parser import Header, SubLevel, parse_file

from src.env.level_load import load_level as jax_load_level
from src.env.step import step as jax_step
from src.game import (
  ACTION_EAST,
  ACTION_NORTH,
  ACTION_SOUTH,
  ACTION_WAIT,
  ACTION_WEST,
  load_level as py_load_level,
  step as py_step,
)

DAT_DIR = Path(__file__).resolve().parent.parent / "mazes"
SOLUTIONS_FILE = Path(__file__).parent.parent / "solutions.txt"

MOVE_MAP = {
  "L": ACTION_WEST,
  "R": ACTION_EAST,
  "U": ACTION_NORTH,
  "D": ACTION_SOUTH,
  "W": ACTION_WAIT,
}


def _parse_solutions(path: Path) -> list[tuple[str, int, list[int]]]:
  """Parse solutions.txt into (pyramid_name, room_number, actions) tuples."""
  solutions = []
  current_pyramid = ""

  for line in path.read_text().splitlines():
    line = line.strip()
    if not line:
      continue

    # Detect pyramid headers (end with ...)
    if line.endswith("..."):
      current_pyramid = line.rstrip(".")
      continue

    # Detect solution lines: start with a number followed by period
    m = re.match(r"^(\d+)\.\s+(.+)$", line)
    if m:
      room_num = int(m.group(1))
      moves_str = m.group(2).replace(" ", "")
      actions = [MOVE_MAP[c] for c in moves_str if c in MOVE_MAP]
      if actions:
        solutions.append((current_pyramid, room_num, actions))

  return solutions


def _run_py(sublevel: SubLevel, header: Header, actions: list[int]) -> bool:
  """Run Python engine, return whether the solution wins."""
  state = py_load_level(sublevel, header)
  for action in actions:
    state = py_step(state, action)
    if not state.alive:
      return False
    if state.won:
      return True
  return False


_jit_cache: dict[int, Callable] = {}


def _get_jitted_step(grid_size: int) -> Callable:
  """Cache a JIT-compiled step per grid_size."""
  if grid_size not in _jit_cache:
    import jax

    _jit_cache[grid_size] = jax.jit(lambda lv, s, a: jax_step(grid_size, lv, s, a))
  return _jit_cache[grid_size]


def _compare_engines_on_match(
  sublevel: SubLevel, header: Header, actions: list[int]
) -> str | None:
  """Run both engines step-by-step, return error message or None if ok."""
  py_state = py_load_level(sublevel, header)
  grid_size, level, jax_state = jax_load_level(sublevel, header)
  jitted_step = _get_jitted_step(grid_size)

  for i, action in enumerate(actions):
    py_state = py_step(py_state, action)
    jax_state = jitted_step(level, jax_state, jnp.int32(action))

    py_done = not py_state.alive or py_state.won
    jax_done = bool(jax_state.done) | bool(jax_state.won)

    if py_done != jax_done:
      return f"step {i}: done mismatch py={py_done} jax={jax_done}"
    if py_state.won != bool(jax_state.won):
      return f"step {i}: won mismatch py={py_state.won} jax={bool(jax_state.won)}"

    # Check positions agree
    if not py_done:
      jpr, jpc = int(jax_state.player[0]), int(jax_state.player[1])
      if (jpr, jpc) != py_state.player:
        return f"step {i}: player pos mismatch py={py_state.player} jax=({jpr},{jpc})"

    if py_done:
      break

  return None


# Pre-load levels once as module-level cache
_ALL_LEVELS: list[tuple[str, int, SubLevel, Header]] | None = None


def _get_all_levels() -> list[tuple[str, int, SubLevel, Header]]:
  """Load all (dat_name, sublevel_idx, sublevel, header) tuples (cached)."""
  global _ALL_LEVELS  # noqa: PLW0603
  if _ALL_LEVELS is None:
    _ALL_LEVELS = []
    for dat_path in sorted(DAT_DIR.glob("B-*.dat")):
      parsed = parse_file(dat_path)
      if parsed is None:
        continue
      for idx, sublevel in enumerate(parsed.sublevels):
        _ALL_LEVELS.append((dat_path.stem, idx, sublevel, parsed.header))
  return _ALL_LEVELS


@pytest.mark.skipif(
  not DAT_DIR.exists() or not SOLUTIONS_FILE.exists(),
  reason="Need dat files and solutions.txt",
)
class TestSolutions:
  """Match walkthrough solutions to levels and validate game logic."""

  def test_match_and_validate(self) -> None:
    """Find levels matching each solution, verify both engines agree."""
    solutions = _parse_solutions(SOLUTIONS_FILE)
    all_levels = _get_all_levels()

    print(f"\nLoaded {len(solutions)} solutions, {len(all_levels)} levels")  # noqa: T201

    # Phase 1: Python-only brute force matching (fast)
    matches: list[tuple[str, SubLevel, Header, list[int]]] = []
    unmatched_names: list[str] = []

    t0 = time.monotonic()
    for sol_idx, (pyramid, room, actions) in enumerate(solutions):
      label = f"{pyramid} room {room}"
      found = False

      for dat_name, sub_idx, sublevel, header in all_levels:
        if _run_py(sublevel, header, actions):
          tag = f"{label} on {dat_name}[{sub_idx}]"
          matches.append((tag, sublevel, header, actions))
          found = True
          break

      if not found:
        unmatched_names.append(label)

      if (sol_idx + 1) % 50 == 0:
        elapsed = time.monotonic() - t0
        print(  # noqa: T201
          f"  [{sol_idx + 1}/{len(solutions)}] {len(matches)} matched, {elapsed:.1f}s"
        )

    t_match = time.monotonic() - t0
    print(  # noqa: T201
      f"Phase 1 (Python matching): {len(matches)}/{len(solutions)} in {t_match:.1f}s"
    )

    # Phase 2: JAX comparison on matches only (JIT compiles once per grid size)
    engine_errors: list[str] = []
    t1 = time.monotonic()
    for label, sublevel, header, actions in matches:
      err = _compare_engines_on_match(sublevel, header, actions)
      if err is not None:
        engine_errors.append(f"{label}: {err}")
    t_jax = time.monotonic() - t1
    print(f"Phase 2 (JAX verify): {len(matches)} matches in {t_jax:.1f}s")  # noqa: T201

    # Report
    print(f"\n{'=' * 60}")  # noqa: T201
    print(  # noqa: T201
      f"Matched: {len(matches)}/{len(solutions)} "
      f"({len(matches) * 100 // len(solutions)}%)"
    )
    if unmatched_names:
      print(f"Unmatched ({len(unmatched_names)}):")  # noqa: T201
      for name in unmatched_names[:20]:
        print(f"  {name}")  # noqa: T201
    if engine_errors:
      print(f"Engine mismatches ({len(engine_errors)}):")  # noqa: T201
      for err in engine_errors:
        print(f"  {err}")  # noqa: T201
    print(f"{'=' * 60}")  # noqa: T201

    assert len(engine_errors) == 0, f"{len(engine_errors)} engine mismatches"
    # Allow some unmatched (wrong mapping), but most should work
    assert len(matches) > len(solutions) * 0.3, (
      f"Only {len(matches)}/{len(solutions)} matched"
    )
