"""BFS solver for Mummy Maze — finds optimal (shortest) solutions."""

import argparse
import copy
import csv
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from mummy_maze.parser import parse_file

from src.game import (
  ACTION_EAST,
  ACTION_NORTH,
  ACTION_SOUTH,
  ACTION_WAIT,
  ACTION_WEST,
  GameState,
  _can_move,
  load_level,
  step,
)

ACTIONS = [ACTION_NORTH, ACTION_SOUTH, ACTION_EAST, ACTION_WEST, ACTION_WAIT]
ACTION_NAMES = {
  ACTION_NORTH: "N",
  ACTION_SOUTH: "S",
  ACTION_EAST: "E",
  ACTION_WEST: "W",
  ACTION_WAIT: "wait",
}


def state_key(state: GameState) -> tuple[object, ...]:
  """Hashable snapshot of the dynamic parts of game state."""
  return (
    state.player,
    tuple(state.mummies),
    tuple(state.scorpions),
    state.gate_active,
  )


def solve(state: GameState) -> tuple[list[int], int] | None:
  """BFS for the shortest action sequence that wins.

  Returns (actions, states_explored) or None if unsolvable.
  """
  visited: set[tuple[object, ...]] = set()
  key = state_key(state)
  visited.add(key)

  queue: deque[tuple[GameState, list[int]]] = deque()
  queue.append((state, []))

  while queue:
    current, actions = queue.popleft()

    for action in ACTIONS:
      next_state = copy.deepcopy(current)
      step(next_state, action)

      if next_state.won:
        return actions + [action], len(visited)

      if not next_state.alive:
        continue

      key = state_key(next_state)
      if key in visited:
        continue

      visited.add(key)
      queue.append((next_state, actions + [action]))

  return None


# -- State graph (full reachable state space) --

_WIN: tuple[str] = ("WIN",)
_DEAD: tuple[str] = ("DEAD",)


@dataclass(frozen=True)
class StateGraph:
  """Complete state-action transition graph from BFS exploration."""

  start: tuple[object, ...]
  # transitions[state_key][action] -> next_key (WIN/DEAD sentinels for terminals)
  transitions: dict[tuple[object, ...], dict[int, tuple[object, ...]]]
  n_transient: int  # number of non-terminal reachable states


def build_graph(state: GameState) -> StateGraph:
  """BFS over all reachable states, recording transitions for every valid action.

  Unlike solve(), this does NOT short-circuit on first win — it explores
  the entire reachable state space. Blocked moves (wall/gate) are skipped
  since they are equivalent to WAIT.
  """
  start_key = state_key(state)
  transitions: dict[tuple[object, ...], dict[int, tuple[object, ...]]] = {}

  visited: set[tuple[object, ...]] = {start_key}
  queue: deque[tuple[GameState, tuple[object, ...]]] = deque()
  queue.append((state, start_key))

  while queue:
    current, cur_key = queue.popleft()
    action_map: dict[int, tuple[object, ...]] = {}

    # Determine which directional moves are actually available
    pr, pc = current.player
    valid_actions = [ACTION_WAIT]
    for a in (ACTION_NORTH, ACTION_SOUTH, ACTION_EAST, ACTION_WEST):
      if _can_move(current, pr, pc, a):
        valid_actions.append(a)

    for action in valid_actions:
      next_state = copy.deepcopy(current)
      step(next_state, action)

      if next_state.won:
        action_map[action] = _WIN
      elif not next_state.alive:
        action_map[action] = _DEAD
      else:
        nk = state_key(next_state)
        action_map[action] = nk
        if nk not in visited:
          visited.add(nk)
          queue.append((next_state, nk))

    transitions[cur_key] = action_map

  return StateGraph(
    start=start_key,
    transitions=transitions,
    n_transient=len(transitions),
  )


def _solve_one(args: tuple[Path, int]) -> tuple[str, int, int | None, int]:
  """Solve a single sublevel. Designed for use with ProcessPoolExecutor.

  Args:
    args: (dat_path, sub_idx)

  Returns:
    (file_stem, sub_idx, n_moves or None, states_explored)
  """
  dat_path, sub_idx = args
  parsed = parse_file(dat_path)
  if parsed is None or sub_idx >= len(parsed.sublevels):
    return (dat_path.stem, sub_idx, None, 0)
  state = load_level(parsed.sublevels[sub_idx], parsed.header)
  result = solve(state)
  if result is None:
    return (dat_path.stem, sub_idx, None, 0)
  actions, explored = result
  return (dat_path.stem, sub_idx, len(actions), explored)


def _solve_file(dat_dir: Path, file_stem: str, sub_idx: int) -> None:
  dat_path = dat_dir / f"{file_stem}.dat"
  parsed = parse_file(dat_path)
  if parsed is None:
    print(f"Could not parse {dat_path}")
    return
  if sub_idx >= len(parsed.sublevels):
    print(f"{file_stem} only has {len(parsed.sublevels)} sublevels")
    return

  state = load_level(parsed.sublevels[sub_idx], parsed.header)
  print(f"Solving {file_stem} sublevel {sub_idx} (grid {state.grid_size})...")

  result = solve(state)
  if result is None:
    print("  UNSOLVABLE")
  else:
    actions, explored = result
    names = " ".join(ACTION_NAMES[a] for a in actions)
    print(f"  Solved in {len(actions)} moves ({explored} states explored)")
    print(f"  Solution: {names}")


def _solve_all(dat_dir: Path, jobs: int, out: Path | None) -> None:
  dat_files = sorted(
    dat_dir.glob("B-*.dat"),
    key=lambda p: int(p.stem.split("-")[1]),
  )
  if not dat_files:
    print(f"No B-*.dat files found in {dat_dir}")
    return

  # Build work items
  work: list[tuple[Path, int]] = []
  for dat_path in dat_files:
    parsed = parse_file(dat_path)
    if parsed is None:
      continue
    for sub_idx in range(len(parsed.sublevels)):
      work.append((dat_path, sub_idx))

  print(f"Solving {len(work)} sublevels with {jobs} workers...")

  # Solve in parallel
  results: list[tuple[str, int, int | None, int]] = []
  done = 0
  with ProcessPoolExecutor(max_workers=jobs) as pool:
    for result in pool.map(_solve_one, work, chunksize=16):
      results.append(result)
      done += 1
      if done % 500 == 0:
        print(f"  {done}/{len(work)}...")

  # Summarize
  solved = 0
  unsolvable = 0
  max_moves = 0
  max_moves_level = ""

  for file_stem, sub_idx, n_moves, explored in results:
    label = f"{file_stem} sub {sub_idx}"
    if n_moves is None:
      unsolvable += 1
    else:
      solved += 1
      if n_moves > max_moves:
        max_moves = n_moves
        max_moves_level = label

  total = solved + unsolvable
  print(f"\nSummary: {solved} solved, {unsolvable} unsolvable, {total} total")
  if solved > 0:
    print(f"Hardest solved: {max_moves_level} ({max_moves} moves)")

  # Write CSV
  if out is not None:
    with open(out, "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(["file", "sublevel", "moves", "states_explored"])
      for file_stem, sub_idx, n_moves, explored in results:
        writer.writerow(
          [
            file_stem,
            sub_idx,
            n_moves if n_moves is not None else "",
            explored,
          ]
        )
    print(f"Results written to {out}")


def main() -> None:
  parser = argparse.ArgumentParser(description="BFS solver for Mummy Maze")
  parser.add_argument("dat_dir", type=Path, help="directory containing B-*.dat files")
  parser.add_argument("--file", default=None, help="dat file stem (e.g. B-68)")
  parser.add_argument("--sublevel", type=int, default=0, help="sublevel index")
  parser.add_argument("--all", action="store_true", help="solve all levels")
  parser.add_argument(
    "--jobs",
    "-j",
    type=int,
    default=os.cpu_count() or 1,
    help="parallel workers (default: all cores)",
  )
  parser.add_argument("--out", type=Path, default=None, help="CSV output path")
  args = parser.parse_args()

  dat_dir = args.dat_dir.resolve()

  if args.all:
    _solve_all(dat_dir, args.jobs, args.out)
  elif args.file:
    _solve_file(dat_dir, args.file, args.sublevel)
  else:
    parser.error("specify --file or --all")


if __name__ == "__main__":
  main()
