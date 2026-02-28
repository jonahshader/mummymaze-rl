"""Wall checks, movement helpers, and gate logic for the JAX environment.

grid_size is the only trace-time constant (Python int).
is_red is a runtime bool — axis priority is selected via jnp.where.
"""

import jax.numpy as jnp
from jaxtyping import Array, Bool, Int

from src.env.types import LevelData

# Actions (same as game.py)
ACTION_NORTH = 0
ACTION_SOUTH = 1
ACTION_EAST = 2
ACTION_WEST = 3
ACTION_WAIT = 4

# Direction deltas: (dr, dc) indexed by action
_DR = jnp.array([-1, 1, 0, 0, 0], dtype=jnp.int32)
_DC = jnp.array([0, 0, 1, -1, 0], dtype=jnp.int32)


def gate_blocked(
  level: LevelData,
  gate_open: Bool[Array, ""],
  r: Int[Array, ""],
  c: Int[Array, ""],
  direction: Int[Array, ""],
) -> Bool[Array, ""]:
  """Check if the gate blocks movement from (r, c) in direction.

  Gate is a vertical barrier on the EAST edge of (gate_row, gate_col).
  It blocks east from gate_cell and west into gate_cell (from gate_col+1).
  Only active when has_key_gate & ~gate_open.
  """
  active = level.has_key_gate & ~gate_open
  gr = level.gate_row
  gc = level.gate_col

  # East from gate cell
  east_blocked = (direction == ACTION_EAST) & (r == gr) & (c == gc)
  # West into gate cell (from the cell to the east)
  west_blocked = (direction == ACTION_WEST) & (r == gr) & (c == gc + 1)

  return active & (east_blocked | west_blocked)


def wall_blocked(
  h_walls: Bool[Array, "Np1 N"],
  v_walls: Bool[Array, "N Np1"],
  r: Int[Array, ""],
  c: Int[Array, ""],
  direction: Int[Array, ""],
) -> Bool[Array, ""]:
  """Check if there's a wall blocking movement from (r, c) in direction.

  Computes all 4 wall lookups unconditionally and selects by direction.
  """
  north_blocked = h_walls[r, c]
  south_blocked = h_walls[r + 1, c]
  east_blocked = v_walls[r, c + 1]
  west_blocked = v_walls[r, c]

  blocked = jnp.array([north_blocked, south_blocked, east_blocked, west_blocked, True])
  return blocked[direction]


def can_move(
  grid_size: int,
  level: LevelData,
  gate_open: Bool[Array, ""],
  r: Int[Array, ""],
  c: Int[Array, ""],
  direction: Int[Array, ""],
) -> Bool[Array, ""]:
  """Check if movement from (r,c) in direction is valid (bounds+wall+gate)."""
  dr = _DR[direction]
  dc = _DC[direction]
  nr = r + dr
  nc = c + dc

  n = grid_size
  in_bounds = (nr >= 0) & (nr < n) & (nc >= 0) & (nc < n)
  not_wall = ~wall_blocked(level.h_walls_base, level.v_walls_base, r, c, direction)
  not_gate = ~gate_blocked(level, gate_open, r, c, direction)

  return in_bounds & not_wall & not_gate


def move_enemy_one_step(
  grid_size: int,
  level: LevelData,
  gate_open: Bool[Array, ""],
  is_red: Bool[Array, ""],
  er: Int[Array, ""],
  ec: Int[Array, ""],
  pr: Int[Array, ""],
  pc: Int[Array, ""],
) -> tuple[Int[Array, ""], Int[Array, ""]]:
  """Compute one step of enemy chase toward player.

  is_red is a runtime bool — axis priority selected via jnp.where.
  White (is_red=False): try horizontal first, then vertical.
  Red  (is_red=True):  try vertical first, then horizontal.
  """
  # Horizontal chase direction
  h_dir = jnp.where(ec > pc, jnp.int32(ACTION_WEST), jnp.int32(ACTION_EAST))
  h_aligned = ec == pc

  # Vertical chase direction
  v_dir = jnp.where(er > pr, jnp.int32(ACTION_NORTH), jnp.int32(ACTION_SOUTH))
  v_aligned = er == pr

  # Select primary/secondary based on is_red
  # Red (is_red=True): primary = horizontal, secondary = vertical
  # White (is_red=False): primary = vertical, secondary = horizontal
  primary_dir = jnp.where(is_red, h_dir, v_dir)
  primary_aligned = jnp.where(is_red, h_aligned, v_aligned)
  secondary_dir = jnp.where(is_red, v_dir, h_dir)
  secondary_aligned = jnp.where(is_red, v_aligned, h_aligned)

  primary_ok = ~primary_aligned & can_move(
    grid_size, level, gate_open, er, ec, primary_dir
  )
  secondary_ok = ~secondary_aligned & can_move(
    grid_size, level, gate_open, er, ec, secondary_dir
  )

  # Try primary, then secondary, then stay
  move_dir = jnp.where(
    primary_ok,
    primary_dir,
    jnp.where(secondary_ok, secondary_dir, jnp.int32(ACTION_WAIT)),
  )

  dr = _DR[move_dir]
  dc = _DC[move_dir]
  return er + dr, ec + dc
