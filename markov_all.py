#!/usr/bin/env python3
"""Exact win-probability analysis for all levels via absorbing Markov chains.

Builds the full state graph for each level, then solves the Markov chain
for exact P(win) and E[steps] under a uniform-random valid-action policy.

Usage:
    uv run python markov_all.py                # analyze all, cache results
    uv run python markov_all.py --clear        # clear cache and re-analyze
"""

import argparse
import csv
import sys
from pathlib import Path

from joblib import Memory, Parallel, delayed
from mummy_maze.parser import parse_file

from src.game import load_level
from src.markov import analyze
from src.solver import build_graph

PROJECT_ROOT = Path(__file__).resolve().parent
DAT_DIR = PROJECT_ROOT / "mazes"
CACHE_DIR = PROJECT_ROOT / ".cache" / "markov"

memory = Memory(location=str(CACHE_DIR), verbose=0)

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


@memory.cache
def _analyze_one(file_stem: str, sub_idx: int) -> tuple[int, int, float, float] | None:
  """Build graph and analyze one sublevel.

  Returns (grid_size, n_states, win_prob, expected_steps).
  """
  dat_path = DAT_DIR / f"{file_stem}.dat"
  parsed = parse_file(dat_path)
  if parsed is None or sub_idx >= len(parsed.sublevels):
    return None
  state = load_level(parsed.sublevels[sub_idx], parsed.header)
  graph = build_graph(state)
  result = analyze(graph)
  return (state.grid_size, result.n_transient, result.win_prob, result.expected_steps)


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


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Markov chain analysis for all Mummy Maze levels"
  )
  parser.add_argument(
    "--clear", action="store_true", help="clear cache before analyzing"
  )
  parser.add_argument(
    "--out", type=Path, default=Path("markov_results.csv"), help="CSV output path"
  )
  parser.add_argument(
    "-j", "--jobs", type=int, default=-1, help="parallel workers (-1 = all cores)"
  )
  args = parser.parse_args()

  if args.clear:
    memory.clear()
    print("Cache cleared.")

  dat_files = sorted(
    DAT_DIR.glob("B-*.dat"),
    key=lambda p: int(p.stem.split("-")[1]),
  )
  if not dat_files:
    print(f"No B-*.dat files found in {DAT_DIR}", file=sys.stderr)
    return

  work: list[tuple[str, int]] = []
  for dat_path in dat_files:
    parsed = parse_file(dat_path)
    if parsed is None:
      continue
    for sub_idx in range(len(parsed.sublevels)):
      work.append((dat_path.stem, sub_idx))

  print(f"Analyzing {len(work)} levels (cached results are instant)...")

  results = Parallel(n_jobs=args.jobs, verbose=10)(
    delayed(_analyze_one)(stem, idx) for stem, idx in work
  )

  # Collect rows and stats
  rows: list[dict[str, str | int]] = []
  win_probs: list[float] = []
  n_solved = 0
  n_unsolvable = 0
  n_skipped = 0

  for (stem, idx), result in zip(work, results):
    if result is None:
      n_skipped += 1
      continue
    grid_size, n_states, win_prob, expected_steps = result
    rows.append(
      {
        "file": stem,
        "sublevel": idx,
        "grid_size": grid_size,
        "n_states": n_states,
        "win_prob": f"{win_prob:.6f}",
        "expected_steps": f"{expected_steps:.2f}",
      }
    )
    win_probs.append(win_prob)
    if win_prob > 0:
      n_solved += 1
    else:
      n_unsolvable += 1

  # Summary
  print(f"\nDone: {n_solved} solvable, {n_unsolvable} unsolvable, {len(rows)} total")
  if n_skipped:
    print(f"  ({n_skipped} skipped due to parse errors)")

  if win_probs:
    probs = sorted(win_probs)
    nonzero = [p for p in probs if p > 0]

    print("\nWin probability distribution (all levels):")
    print_histogram(probs)

    if nonzero:
      mean = sum(nonzero) / len(nonzero)
      median = nonzero[len(nonzero) // 2]
      print(f"\n  Solvable levels ({len(nonzero)}):")
      print(f"    Mean win prob:   {mean:.4%}")
      print(f"    Median win prob: {median:.4%}")
      print(f"    Min:             {min(nonzero):.4%}")
      print(f"    Max:             {max(nonzero):.4%}")

    # Hardest and easiest solvable
    solvable = [(r, float(r["win_prob"])) for r in rows if float(r["win_prob"]) > 0]
    solvable.sort(key=lambda x: x[1])
    show = min(10, len(solvable))
    if solvable:
      print(f"\n  Hardest {show} (lowest win prob):")
      for row, wp in solvable[:show]:
        print(
          f"    {row['file']} sub {row['sublevel']}:"
          f" {wp:.4%}  ({row['n_states']} states)"
        )
      print(f"\n  Easiest {show} (highest win prob):")
      for row, wp in solvable[-show:]:
        print(
          f"    {row['file']} sub {row['sublevel']}:"
          f" {wp:.4%}  ({row['n_states']} states)"
        )

  # Write CSV
  with open(args.out, "w", newline="") as f:
    writer = csv.DictWriter(
      f,
      fieldnames=[
        "file",
        "sublevel",
        "grid_size",
        "n_states",
        "win_prob",
        "expected_steps",
      ],
    )
    writer.writeheader()
    writer.writerows(rows)
  print(f"\nResults written to {args.out}")


if __name__ == "__main__":
  main()
