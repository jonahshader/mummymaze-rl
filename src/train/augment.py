"""Dataset augmentation: merge GA-generated levels into BC datasets."""

import jax.numpy as jnp
import mummymaze_rust
import numpy as np

from src.env.level_bank import LevelBank
from src.env.level_load import exit_cell
from src.env.types import MAX_MUMMIES, MAX_SCORPIONS, MAX_TRAPS
from src.train.dataset import BCDataset, decode_action_masks


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
) -> dict[int, BCDataset]:
  """Append GA level data to existing datasets. New levels go to train set only."""
  if not new_levels:
    return base

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
