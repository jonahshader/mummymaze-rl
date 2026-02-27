"""Level bank: load all .dat files, deduplicate, bin by grid_size, train/val split.

Usage:
  banks = load_all_levels(Path("data/mazes"))
  # banks is a dict: {6: LevelBank, 8: LevelBank, 10: LevelBank}

  # Sample a batch of levels for training
  level_batch = sample_batch(banks[6], banks[6].train_indices, key, batch_size=64)
"""

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Bool, Int, PRNGKeyArray

from src.env.level_load import exit_cell
from src.env.types import MAX_MUMMIES, MAX_SCORPIONS, MAX_TRAPS, LevelData

try:
  from mummy_maze.parser import parse_file
except ImportError:
  parse_file = None  # type: ignore[assignment]


@dataclass(frozen=True)
class LevelBank:
  """All levels for one grid_size, stacked for batched indexing.

  Invariant: all levels share the same grid_size (determines array shapes).
  """

  grid_size: int
  n_levels: int

  # Stacked level data — index with [level_idx] to get one LevelData
  h_walls_base: Bool[Array, "n_levels Np1 N"]
  v_walls_base: Bool[Array, "n_levels N Np1"]
  is_red: Bool[Array, "n_levels"]
  has_key_gate: Bool[Array, "n_levels"]
  gate_wall_row: Int[Array, "n_levels"]
  gate_wall_col: Int[Array, "n_levels"]
  trap_pos: Int[Array, "n_levels 2 2"]
  trap_active: Bool[Array, "n_levels 2"]
  key_pos: Int[Array, "n_levels 2"]
  exit_cell: Int[Array, "n_levels 2"]
  initial_player: Int[Array, "n_levels 2"]
  initial_mummy_pos: Int[Array, "n_levels 2 2"]
  initial_mummy_alive: Bool[Array, "n_levels 2"]
  initial_scorpion_pos: Int[Array, "n_levels 1 2"]
  initial_scorpion_alive: Bool[Array, "n_levels 1"]

  # Train/val split indices
  train_indices: Int[Array, "n_train"]
  val_indices: Int[Array, "n_val"]


def get_level(bank: LevelBank, idx: Int[Array, ""]) -> LevelData:
  """Extract a single LevelData from the bank by index."""
  return LevelData(
    h_walls_base=bank.h_walls_base[idx],
    v_walls_base=bank.v_walls_base[idx],
    is_red=bank.is_red[idx],
    has_key_gate=bank.has_key_gate[idx],
    gate_wall_row=bank.gate_wall_row[idx],
    gate_wall_col=bank.gate_wall_col[idx],
    trap_pos=bank.trap_pos[idx],
    trap_active=bank.trap_active[idx],
    key_pos=bank.key_pos[idx],
    exit_cell=bank.exit_cell[idx],
    initial_player=bank.initial_player[idx],
    initial_mummy_pos=bank.initial_mummy_pos[idx],
    initial_mummy_alive=bank.initial_mummy_alive[idx],
    initial_scorpion_pos=bank.initial_scorpion_pos[idx],
    initial_scorpion_alive=bank.initial_scorpion_alive[idx],
  )


def sample_batch(
  bank: LevelBank,
  indices: Int[Array, "n"],
  key: PRNGKeyArray,
  batch_size: int,
) -> LevelData:
  """Sample a batch of LevelData from the bank using given index set.

  Args:
    bank: Level bank to sample from.
    indices: Subset of level indices to sample from (e.g. train_indices).
    key: PRNG key for random sampling.
    batch_size: Number of levels to sample.

  Returns:
    LevelData with leading batch dimension (vmappable).
  """
  chosen = jr.choice(key, indices, shape=(batch_size,), replace=True)
  return jax.vmap(lambda i: get_level(bank, i))(chosen)


def _load_level_np(sublevel: object, header: object) -> tuple[int, dict[str, object]]:
  """Convert parser SubLevel/Header to numpy arrays (no JAX overhead).

  Returns (grid_size, dict of numpy arrays matching LevelData fields).
  """
  from mummy_maze.parser import EntityType

  n = header.grid_size  # type: ignore[union-attr]

  player = (0, 0)
  mummies: list[tuple[int, int]] = []
  scorpions: list[tuple[int, int]] = []
  traps: list[tuple[int, int]] = []
  key_pos: tuple[int, int] | None = None
  gate_wall: tuple[int, int] | None = None

  for ent in sublevel.entities:  # type: ignore[union-attr]
    pos = (ent.row, ent.col)
    if ent.type == EntityType.PLAYER:
      player = pos
    elif ent.type == EntityType.MUMMY:
      mummies.append(pos)
    elif ent.type == EntityType.SCORPION:
      scorpions.append(pos)
    elif ent.type == EntityType.TRAP:
      traps.append(pos)
    elif ent.type == EntityType.KEY:
      key_pos = pos
    elif ent.type == EntityType.GATE:
      gate_wall = (ent.row + 1, ent.col)

  h_walls = np.array(sublevel.h_walls, dtype=np.bool_)  # type: ignore[union-attr]
  v_walls = np.array(sublevel.v_walls, dtype=np.bool_)  # type: ignore[union-attr]

  # .dat files store walls without the gate wall. Gate starts closed (wall present).
  if gate_wall is not None:
    h_walls[gate_wall[0], gate_wall[1]] = True

  mummy_arr = np.zeros((MAX_MUMMIES, 2), dtype=np.int32)
  mummy_alive = np.zeros(MAX_MUMMIES, dtype=np.bool_)
  for i, (mr, mc) in enumerate(mummies[:MAX_MUMMIES]):
    mummy_arr[i] = [mr, mc]
    mummy_alive[i] = True

  scorpion_arr = np.zeros((MAX_SCORPIONS, 2), dtype=np.int32)
  scorpion_alive = np.zeros(MAX_SCORPIONS, dtype=np.bool_)
  for i, (sr, sc) in enumerate(scorpions[:MAX_SCORPIONS]):
    scorpion_arr[i] = [sr, sc]
    scorpion_alive[i] = True

  trap_arr = np.zeros((MAX_TRAPS, 2), dtype=np.int32)
  trap_active = np.zeros(MAX_TRAPS, dtype=np.bool_)
  for i, (tr, tc) in enumerate(traps[:MAX_TRAPS]):
    trap_arr[i] = [tr, tc]
    trap_active[i] = True

  kp = np.array(key_pos if key_pos is not None else (0, 0), dtype=np.int32)
  gwr = np.int32(gate_wall[0] if gate_wall is not None else 0)
  gwc = np.int32(gate_wall[1] if gate_wall is not None else 0)
  ec = exit_cell(sublevel.exit_side, sublevel.exit_pos, n)  # type: ignore[union-attr]

  return n, {
    "h_walls_base": h_walls,
    "v_walls_base": v_walls,
    "is_red": np.bool_(header.flip),  # type: ignore[union-attr]
    "has_key_gate": np.bool_(bool(header.key_gate)),  # type: ignore[union-attr]
    "gate_wall_row": gwr,
    "gate_wall_col": gwc,
    "trap_pos": trap_arr,
    "trap_active": trap_active,
    "key_pos": kp,
    "exit_cell": np.array(ec, dtype=np.int32),
    "initial_player": np.array(player, dtype=np.int32),
    "initial_mummy_pos": mummy_arr,
    "initial_mummy_alive": mummy_alive,
    "initial_scorpion_pos": scorpion_arr,
    "initial_scorpion_alive": scorpion_alive,
  }


def _np_fingerprint(d: dict[str, object]) -> bytes:
  """Fast fingerprint from numpy arrays for deduplication."""
  parts = []
  for key in sorted(d):
    v = d[key]
    if isinstance(v, np.ndarray):
      parts.append(v.tobytes())
    else:
      parts.append(bytes(np.array(v)))
  return b"".join(parts)


def _level_fingerprint(level: LevelData) -> tuple[object, ...]:
  """Create a hashable fingerprint of a level for deduplication."""
  return (
    level.h_walls_base.tobytes(),
    level.v_walls_base.tobytes(),
    bool(level.is_red),
    bool(level.has_key_gate),
    int(level.gate_wall_row),
    int(level.gate_wall_col),
    level.trap_pos.tobytes(),
    level.trap_active.tobytes(),
    level.key_pos.tobytes(),
    level.exit_cell.tobytes(),
    level.initial_player.tobytes(),
    level.initial_mummy_pos.tobytes(),
    level.initial_mummy_alive.tobytes(),
    level.initial_scorpion_pos.tobytes(),
    level.initial_scorpion_alive.tobytes(),
  )


def _build_bank(
  grid_size: int,
  levels: list[dict[str, object]],
  val_fraction: float,
  rng: PRNGKeyArray,
) -> LevelBank:
  """Stack numpy level dicts into a LevelBank with train/val split."""
  n = len(levels)

  # Stack numpy arrays, then convert to JAX once
  def _stack(field: str) -> jax.Array:
    return jnp.array(np.stack([lv[field] for lv in levels]))  # type: ignore[arg-type]

  h_walls = _stack("h_walls_base")
  v_walls = _stack("v_walls_base")
  is_red = _stack("is_red")
  has_key_gate = _stack("has_key_gate")
  gate_wall_row = _stack("gate_wall_row")
  gate_wall_col = _stack("gate_wall_col")
  trap_pos = _stack("trap_pos")
  trap_active = _stack("trap_active")
  key_pos = _stack("key_pos")
  exit_cell = _stack("exit_cell")
  initial_player = _stack("initial_player")
  initial_mummy_pos = _stack("initial_mummy_pos")
  initial_mummy_alive = _stack("initial_mummy_alive")
  initial_scorpion_pos = _stack("initial_scorpion_pos")
  initial_scorpion_alive = _stack("initial_scorpion_alive")

  # Shuffle and split
  n_val = max(1, int(n * val_fraction))
  n_train = n - n_val
  perm = jr.permutation(rng, n)
  train_indices = perm[:n_train]
  val_indices = perm[n_train:]

  return LevelBank(
    grid_size=grid_size,
    n_levels=n,
    h_walls_base=h_walls,
    v_walls_base=v_walls,
    is_red=is_red,
    has_key_gate=has_key_gate,
    gate_wall_row=gate_wall_row,
    gate_wall_col=gate_wall_col,
    trap_pos=trap_pos,
    trap_active=trap_active,
    key_pos=key_pos,
    exit_cell=exit_cell,
    initial_player=initial_player,
    initial_mummy_pos=initial_mummy_pos,
    initial_mummy_alive=initial_mummy_alive,
    initial_scorpion_pos=initial_scorpion_pos,
    initial_scorpion_alive=initial_scorpion_alive,
    train_indices=train_indices,
    val_indices=val_indices,
  )


def load_all_levels(
  dat_dir: Path,
  val_fraction: float = 0.1,
  seed: int = 42,
) -> tuple[dict[int, LevelBank], dict[int, list[tuple[str, int]]]]:
  """Load all .dat files, deduplicate, bin by grid_size, train/val split.

  Args:
    dat_dir: Directory containing B-*.dat files.
    val_fraction: Fraction of levels to hold out for validation.
    seed: Random seed for reproducible train/val splits.

  Returns:
    Tuple of (banks, sources) where:
      banks: dict mapping grid_size -> LevelBank.
      sources: dict mapping grid_size -> list of (file_stem, sublevel_idx)
        aligned with bank indices (before train/val shuffle).
  """
  if parse_file is None:
    msg = "mummy-maze-parser is required. Install with: uv add mummy-maze-parser"
    raise ImportError(msg)

  # Collect all unique levels binned by grid_size (numpy-only, no JAX overhead)
  bins: dict[int, list[dict[str, object]]] = {}
  sources: dict[int, list[tuple[str, int]]] = {}
  seen: dict[int, set[bytes]] = {}

  dat_files = sorted(dat_dir.glob("B-*.dat"))
  for dat_path in dat_files:
    parsed = parse_file(dat_path)
    if parsed is None:
      continue
    for sub_idx, sublevel in enumerate(parsed.sublevels):
      gs, level_np = _load_level_np(sublevel, parsed.header)
      fp = _np_fingerprint(level_np)

      if gs not in bins:
        bins[gs] = []
        sources[gs] = []
        seen[gs] = set()

      if fp not in seen[gs]:
        seen[gs].add(fp)
        bins[gs].append(level_np)
        sources[gs].append((dat_path.stem, sub_idx))

  # Build banks
  rng = jax.random.key(seed)
  banks: dict[int, LevelBank] = {}
  for gs in sorted(bins):
    rng, sub_rng = jax.random.split(rng)
    banks[gs] = _build_bank(gs, bins[gs], val_fraction, sub_rng)

  return banks, sources
