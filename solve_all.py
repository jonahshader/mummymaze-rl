#!/usr/bin/env python3
"""Solve all levels with joblib caching.

First run solves via BFS and caches full move sequences to disk.
Subsequent runs return cached results instantly.

Usage:
    uv run python solve_all.py                # solve & cache all
    uv run python solve_all.py --clear        # clear cache and re-solve
"""

import argparse
import sys
from pathlib import Path

from joblib import Memory, Parallel, delayed
from mummy_maze.parser import parse_file

from src.game import load_level
from src.solver import solve

PROJECT_ROOT = Path(__file__).resolve().parent
DAT_DIR = PROJECT_ROOT / "mazes"
CACHE_DIR = PROJECT_ROOT / ".cache" / "solver"

memory = Memory(location=str(CACHE_DIR), verbose=0)


@memory.cache
def _solve_one(file_stem: str, sub_idx: int) -> list[int] | None:
  """Solve a single sublevel via BFS. Returns action list or None."""
  dat_path = DAT_DIR / f"{file_stem}.dat"
  parsed = parse_file(dat_path)
  if parsed is None or sub_idx >= len(parsed.sublevels):
    return None
  state = load_level(parsed.sublevels[sub_idx], parsed.header)
  result = solve(state)
  if result is None:
    return None
  actions, _explored = result
  return actions


def solve_all() -> dict[tuple[str, int], list[int] | None]:
  """Solve all levels, returning {(file_stem, sub_idx): actions | None}."""
  dat_files = sorted(
    DAT_DIR.glob("B-*.dat"),
    key=lambda p: int(p.stem.split("-")[1]),
  )
  if not dat_files:
    print(f"No B-*.dat files found in {DAT_DIR}", file=sys.stderr)
    return {}

  work = []
  for dat_path in dat_files:
    parsed = parse_file(dat_path)
    if parsed is None:
      continue
    for sub_idx in range(len(parsed.sublevels)):
      work.append((dat_path.stem, sub_idx))

  print(f"Solving {len(work)} levels (cached results are instant)...")

  results = Parallel(n_jobs=-1, verbose=10)(
    delayed(_solve_one)(stem, idx) for stem, idx in work
  )

  out = {}
  for (stem, idx), actions in zip(work, results):
    out[(stem, idx)] = actions

  solved = sum(1 for v in out.values() if v is not None)
  unsolvable = len(out) - solved
  print(f"Done: {solved} solved, {unsolvable} unsolvable, {len(out)} total")
  return out


def load_solutions() -> dict[tuple[str, int], list[int]]:
  """Load only the solvable cached solutions (no re-solving).

  Returns {(file_stem, sub_idx): actions} for levels with solutions.
  Raises FileNotFoundError if cache is empty.
  """
  if not CACHE_DIR.exists():
    msg = f"No solver cache at {CACHE_DIR}. Run: uv run python solve_all.py"
    raise FileNotFoundError(msg)

  dat_files = sorted(
    DAT_DIR.glob("B-*.dat"),
    key=lambda p: int(p.stem.split("-")[1]),
  )

  out = {}
  for dat_path in dat_files:
    parsed = parse_file(dat_path)
    if parsed is None:
      continue
    for sub_idx in range(len(parsed.sublevels)):
      # Call the cached function — returns instantly if cached
      actions = _solve_one(file_stem=dat_path.stem, sub_idx=sub_idx)
      if actions is not None:
        out[(dat_path.stem, sub_idx)] = actions
  return out


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Solve all Mummy Maze levels")
  parser.add_argument("--clear", action="store_true", help="clear cache before solving")
  args = parser.parse_args()

  if args.clear:
    memory.clear()
    print("Cache cleared.")

  solve_all()
