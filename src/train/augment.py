"""Dataset augmentation: merge GA-generated levels into BC datasets."""

import json
import time
from collections import defaultdict
from pathlib import Path

import jax.numpy as jnp
import mummymaze_rust
import numpy as np

from src.env.level_bank import LevelBank, dihedral_syms
from src.env.level_load import exit_cell
from src.env.types import MAX_MUMMIES, MAX_SCORPIONS, MAX_TRAPS
from src.train.dataset import BCDataset, decode_action_masks


_COORD_KEYS = {"player", "mummy1", "mummy2", "scorpion", "gate", "key"}


def load_augment_levels(path: Path) -> list[mummymaze_rust.Level]:
  """Load levels from a JSON file, converting coordinate lists to tuples."""
  with open(path) as f:
    level_dicts = json.load(f)
  for d in level_dicts:
    for k in _COORD_KEYS:
      v = d.get(k)
      if isinstance(v, list):
        d[k] = tuple(v)
    if isinstance(d.get("traps"), list):
      d["traps"] = [tuple(t) for t in d["traps"]]
  return [mummymaze_rust.Level.from_dict(d) for d in level_dicts]


def dihedral_variants(
  level: mummymaze_rust.Level,
) -> list[mummymaze_rust.Level]:
  """Return all non-identity dihedral variants of a level."""
  return [level.apply_dihedral(sym) for sym in dihedral_syms(level.has_gate)]


def solve_levels(
  levels: list[mummymaze_rust.Level],
) -> dict[int, list[dict]]:
  """BFS-solve GA levels via Rust, grouped by grid_size.

  Returns {grid_size: [entry_dicts]} where each entry has
  'level_idx', 'grid_size', 'states', 'action_masks'.
  """
  results = mummymaze_rust.best_actions_for_levels(levels)

  by_gs: dict[int, list[dict]] = {}
  for entry in results:
    gs = entry["grid_size"]
    if gs not in by_gs:
      by_gs[gs] = []
    by_gs[gs].append(entry)

  return by_gs


def _level_to_bank_arrays(level: mummymaze_rust.Level) -> dict[str, np.ndarray]:
  """Convert a Rust Level to numpy arrays matching LevelBank fields."""
  d = level.to_dict()
  n = d["grid_size"]

  h_walls = np.array(d["h_walls"], dtype=np.bool_).reshape(n + 1, n)
  v_walls = np.array(d["v_walls"], dtype=np.bool_).reshape(n, n + 1)

  is_red = d["flip"]
  has_gate = d["gate"] is not None and d["key"] is not None

  gate_r, gate_c = d["gate"] if has_gate else (0, 0)
  key_r, key_c = d["key"] if has_gate else (0, 0)

  traps = d["traps"]
  trap_arr = np.zeros((MAX_TRAPS, 2), dtype=np.int32)
  trap_active = np.zeros(MAX_TRAPS, dtype=np.bool_)
  for i, (tr, tc) in enumerate(traps[:MAX_TRAPS]):
    trap_arr[i] = [tr, tc]
    trap_active[i] = True

  player = np.array(d["player"], dtype=np.int32)
  m1 = np.array(d["mummy1"], dtype=np.int32)

  mummy_arr = np.zeros((MAX_MUMMIES, 2), dtype=np.int32)
  mummy_alive = np.zeros(MAX_MUMMIES, dtype=np.bool_)
  mummy_arr[0] = m1
  mummy_alive[0] = True
  if d["mummy2"] is not None:
    mummy_arr[1] = np.array(d["mummy2"], dtype=np.int32)
    mummy_alive[1] = True

  scorpion_arr = np.zeros((MAX_SCORPIONS, 2), dtype=np.int32)
  scorpion_alive = np.zeros(MAX_SCORPIONS, dtype=np.bool_)
  if d["scorpion"] is not None:
    scorpion_arr[0] = np.array(d["scorpion"], dtype=np.int32)
    scorpion_alive[0] = True

  ex_r, ex_c = exit_cell(d["exit_side"], d["exit_pos"], n)
  exit_arr = np.array([ex_r, ex_c], dtype=np.int32)

  return {
    "h_walls_base": h_walls,
    "v_walls_base": v_walls,
    "is_red": is_red,
    "has_key_gate": has_gate,
    "gate_row": gate_r,
    "gate_col": gate_c,
    "trap_pos": trap_arr,
    "trap_active": trap_active,
    "key_pos": np.array([key_r, key_c], dtype=np.int32),
    "exit_cell": exit_arr,
    "initial_player": player,
    "initial_mummy_pos": mummy_arr,
    "initial_mummy_alive": mummy_alive,
    "initial_scorpion_pos": scorpion_arr,
    "initial_scorpion_alive": scorpion_alive,
  }


def _extend_bank(bank: LevelBank, new_arrays: list[dict]) -> LevelBank:
  """Append new level arrays to an existing LevelBank (all new go to train)."""
  if not new_arrays:
    return bank

  n_new = len(new_arrays)
  n_old = bank.n_levels

  # Stack all array fields
  fields_nd = [
    "h_walls_base",
    "v_walls_base",
    "trap_pos",
    "trap_active",
    "key_pos",
    "exit_cell",
    "initial_player",
    "initial_mummy_pos",
    "initial_mummy_alive",
    "initial_scorpion_pos",
    "initial_scorpion_alive",
  ]
  fields_1d = ["is_red", "has_key_gate", "gate_row", "gate_col"]

  stacked = {}
  for f in fields_nd:
    old = np.asarray(getattr(bank, f))
    new = np.stack([a[f] for a in new_arrays])
    stacked[f] = jnp.array(np.concatenate([old, new], axis=0))

  for f in fields_1d:
    old = np.asarray(getattr(bank, f))
    new = np.array([a[f] for a in new_arrays])
    stacked[f] = jnp.array(np.concatenate([old, new], axis=0))

  # All new levels go to train
  new_train = jnp.concatenate(
    [
      bank.train_indices,
      jnp.arange(n_old, n_old + n_new, dtype=jnp.int32),
    ]
  )

  return LevelBank(
    grid_size=bank.grid_size,
    n_levels=n_old + n_new,
    **stacked,
    train_indices=new_train,
    val_indices=bank.val_indices,
  )


def augment_dataset(
  base: dict[int, BCDataset],
  new_levels: list[mummymaze_rust.Level],
  dihedral_augment: bool = False,
) -> dict[int, BCDataset]:
  """Append GA level data to existing datasets. New levels go to train set only.

  If dihedral_augment is True, each new level is expanded with all valid
  dihedral variants before solving and merging.
  """
  if not new_levels:
    return base

  if dihedral_augment:
    expanded = list(new_levels)
    for lev in new_levels:
      expanded.extend(dihedral_variants(lev))
    new_levels = expanded

  # Solve all new levels
  solved_by_gs = solve_levels(new_levels)

  result = dict(base)

  for gs, entries in solved_by_gs.items():
    if gs not in base:
      continue

    ds = base[gs]
    level_arrays: list[dict] = []
    all_tuples: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_level_idx: list[int] = []
    all_is_train: list[bool] = []

    bank_offset = ds.n_levels

    for entry_idx, entry in enumerate(entries):
      orig_level_idx = entry["level_idx"]
      level = new_levels[orig_level_idx]
      level_arrays.append(_level_to_bank_arrays(level))

      states = entry["states"]
      n = len(states)

      tuples_np = np.array(states, dtype=np.int32).reshape(n, 12)
      targets_np = decode_action_masks(entry["action_masks"])

      all_tuples.append(tuples_np)
      all_targets.append(targets_np)
      bank_idx = bank_offset + entry_idx
      all_level_idx.extend([bank_idx] * n)
      all_is_train.extend([True] * n)

    if not all_tuples:
      continue

    # Extend bank
    new_bank = _extend_bank(ds.bank, level_arrays)

    # Concatenate arrays
    new_tuples = np.concatenate(all_tuples, axis=0)
    new_targets = np.concatenate(all_targets, axis=0)
    new_level_idx = np.array(all_level_idx, dtype=np.int32)
    new_is_train = np.array(all_is_train, dtype=np.bool_)

    result[gs] = BCDataset(
      grid_size=gs,
      n_levels=new_bank.n_levels,
      n_states=ds.n_states + new_tuples.shape[0],
      bank=new_bank,
      state_tuples=jnp.concatenate([ds.state_tuples, jnp.array(new_tuples)]),
      action_targets=jnp.concatenate([ds.action_targets, jnp.array(new_targets)]),
      level_idx=jnp.concatenate([ds.level_idx, jnp.array(new_level_idx)]),
      train_mask=jnp.concatenate([ds.train_mask, jnp.array(new_is_train)]),
      val_mask=jnp.concatenate(
        [
          ds.val_mask,
          jnp.zeros(len(new_is_train), dtype=jnp.bool_),
        ]
      ),
    )

  return result


def expand_dihedral_train(
  maze_dir: Path,
  sources: dict[int, list[tuple[str, int]]],
  banks: dict[int, LevelBank],
) -> list[mummymaze_rust.Level]:
  """Generate dihedral variants of train-only canonical levels.

  Only expands levels in the train split to avoid leaking val data into
  training. Parses canonical levels from .dat files, applies each valid
  non-identity dihedral symmetry, and returns the expanded list.
  """
  # Build train-only source indices per grid_size
  train_indices: dict[int, set[int]] = {}
  for gs, bank in banks.items():
    train_indices[gs] = set(np.array(bank.train_indices).tolist())

  # Group source keys by file to avoid re-parsing the same .dat
  by_file: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
  for gs, src_list in sources.items():
    train_set = train_indices.get(gs, set())
    for idx, (stem, sub) in enumerate(src_list):
      if idx in train_set:
        by_file[stem].append((sub, gs, idx))

  variants: list[mummymaze_rust.Level] = []
  for stem, entries in by_file.items():
    dat_path = maze_dir / f"{stem}.dat"
    rust_levels = mummymaze_rust.parse_file(str(dat_path))
    for sub, _gs, _idx in entries:
      variants.extend(dihedral_variants(rust_levels[sub]))

  return variants


def apply_dihedral_augmentation(
  datasets: dict[int, BCDataset],
  sources: dict[int, list[tuple[str, int]]],
  banks: dict[int, LevelBank],
  maze_dir: Path,
  verbose: bool = True,
) -> dict[int, BCDataset]:
  """Expand training set with dihedral variants of train-only levels.

  Generates non-identity dihedral variants, BFS-solves them, and merges
  into the datasets. Val levels are not expanded.
  """
  if verbose:
    print("Generating dihedral augmentations...")
  t0 = time.time()
  variants = expand_dihedral_train(maze_dir, sources, banks)
  if verbose:
    print(f"  {len(variants)} variants generated, solving...")
  datasets = augment_dataset(datasets, variants)
  if verbose:
    print(f"  Dihedral augmentation done in {time.time() - t0:.1f}s")
    for gs, ds in sorted(datasets.items()):
      n_train = int(ds.train_mask.sum())
      n_val = int(ds.val_mask.sum())
      print(f"  grid_size={gs}: {ds.n_states} states ({n_train} train, {n_val} val)")
  return datasets
