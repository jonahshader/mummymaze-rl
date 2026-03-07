"""Test that policy_win_prob_batch with uniform policy matches analyze_all win_prob."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

import mummymaze_rust

MAZE_DIR = "mazes"


def _parse_levels(level_keys: list[tuple[str, int]]) -> list[mummymaze_rust.Level]:
  """Parse Level objects from .dat files, grouping by file to avoid re-parsing."""
  by_file: dict[str, list[tuple[int, int]]] = defaultdict(list)
  for i, (stem, sub) in enumerate(level_keys):
    by_file[stem].append((sub, i))

  result: list[mummymaze_rust.Level | None] = [None] * len(level_keys)
  for stem, entries in by_file.items():
    levels = mummymaze_rust.parse_file(str(Path(MAZE_DIR) / f"{stem}.dat"))
    for sub, idx in entries:
      result[idx] = levels[sub]

  return result  # type: ignore[return-value]


def test_uniform_policy_matches_analyze_all():
  """Uniform 1/5 action probs should reproduce the same win_prob as analyze_all."""
  # Get reference win probs from analyze_all
  analyses = mummymaze_rust.analyze_all(MAZE_DIR)
  ref_by_key: dict[tuple[str, int], float] = {}
  for a in analyses:
    ref_by_key[(a["file"], a["sublevel"])] = a["win_prob"]

  # Get all states per level via best_actions_all (only winnable levels)
  all_best = mummymaze_rust.best_actions_all(MAZE_DIR)

  # Build flat arrays grouped for the batch call
  level_keys: list[tuple[str, int]] = []
  all_state_tuples: list[list[int]] = []
  all_action_probs: list[list[float]] = []
  offsets: list[int] = [0]

  for entry in all_best:
    file_stem = entry["file"]
    sublevel = entry["sublevel"]
    states = entry["states"]
    n = len(states)

    level_keys.append((file_stem, sublevel))
    for s in states:
      # Convert state tuple (with bools) to i32 list
      all_state_tuples.append([int(x) for x in s])
      all_action_probs.append([0.2, 0.2, 0.2, 0.2, 0.2])
    offsets.append(offsets[-1] + n)

  state_tuples_np = np.array(all_state_tuples, dtype=np.int32)
  action_probs_np = np.array(all_action_probs, dtype=np.float32)

  rust_levels = _parse_levels(level_keys)

  results = mummymaze_rust.policy_win_prob_batch(
    rust_levels, state_tuples_np, action_probs_np, offsets
  )

  assert len(results) == len(level_keys)

  mismatches = []
  for i, (file_stem, sublevel) in enumerate(level_keys):
    ref_wp = ref_by_key.get((file_stem, sublevel))
    if ref_wp is None:
      continue
    got = results[i]
    if abs(got - ref_wp) > 1e-6:
      mismatches.append((file_stem, sublevel, ref_wp, got))

  if mismatches:
    for file_stem, sublevel, ref_wp, got in mismatches[:10]:
      print(f"  {file_stem} sub {sublevel}: ref={ref_wp:.10f} got={got:.10f}")
    pytest.fail(f"{len(mismatches)} levels have win_prob mismatch > 1e-6")
