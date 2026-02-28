#!/usr/bin/env python3
"""Solve all levels using the Rust BFS solver.

Returns action sequences for all solvable levels. Fast — all 10,000 levels
in a few seconds via rayon parallelism.

Usage:
    uv run python solve_all.py
"""

from pathlib import Path

from mummymaze_rust import solve_all_actions

PROJECT_ROOT = Path(__file__).resolve().parent
DAT_DIR = PROJECT_ROOT / "mazes"


def solve_all() -> dict[tuple[str, int], list[int] | None]:
  """Solve all levels, returning {(file_stem, sub_idx): actions | None}."""
  results = solve_all_actions(str(DAT_DIR))

  out: dict[tuple[str, int], list[int] | None] = {}
  for r in results:
    out[(r["file"], r["sublevel"])] = r["actions"]

  solved = sum(1 for v in out.values() if v is not None)
  unsolvable = len(out) - solved
  print(f"Done: {solved} solved, {unsolvable} unsolvable, {len(out)} total")
  return out


def load_solutions() -> dict[tuple[str, int], list[int]]:
  """Load only the solvable solutions.

  Returns {(file_stem, sub_idx): actions} for levels with solutions.
  """
  results = solve_all_actions(str(DAT_DIR))
  return {
    (r["file"], r["sublevel"]): r["actions"]
    for r in results
    if r["actions"] is not None
  }


if __name__ == "__main__":
  solve_all()
