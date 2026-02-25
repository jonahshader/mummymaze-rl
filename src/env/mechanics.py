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


def effective_h_walls(
  level: LevelData, gate_open: Bool[Array, ""]
) -> Bool[Array, "Np1 N"]:
  """Return h_walls with gate state applied.

  Always runs the gate logic but masks it with has_key_gate.
  When has_key_gate is False, should_flip is False, so XOR is a no-op.
  gate_wall_row/col must be valid indices even when no gate (use 0,0).
  """
  should_flip = level.has_key_gate & gate_open
  gr = level.gate_wall_row
  gc = level.gate_wall_col
  return level.h_walls_base.at[gr, gc].set(level.h_walls_base[gr, gc] ^ should_flip)


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
  h_walls: Bool[Array, "Np1 N"],
  v_walls: Bool[Array, "N Np1"],
  r: Int[Array, ""],
  c: Int[Array, ""],
  direction: Int[Array, ""],
) -> Bool[Array, ""]:
  """Check if movement from (r, c) in direction is valid (in bounds + no wall)."""
  dr = _DR[direction]
  dc = _DC[direction]
  nr = r + dr
  nc = c + dc

  n = grid_size
  in_bounds = (nr >= 0) & (nr < n) & (nc >= 0) & (nc < n)
  not_blocked = ~wall_blocked(h_walls, v_walls, r, c, direction)

  return in_bounds & not_blocked


def move_enemy_one_step(
  grid_size: int,
  h_walls: Bool[Array, "Np1 N"],
  v_walls: Bool[Array, "N Np1"],
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
  primary_dir = jnp.where(is_red, v_dir, h_dir)
  primary_aligned = jnp.where(is_red, v_aligned, h_aligned)
  secondary_dir = jnp.where(is_red, h_dir, v_dir)
  secondary_aligned = jnp.where(is_red, h_aligned, v_aligned)

  primary_ok = ~primary_aligned & can_move(
    grid_size, h_walls, v_walls, er, ec, primary_dir
  )
  secondary_ok = ~secondary_aligned & can_move(
    grid_size, h_walls, v_walls, er, ec, secondary_dir
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
