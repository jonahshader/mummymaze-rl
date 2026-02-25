"""CNN observation encoder for the JAX Mummy Maze environment."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from src.env.mechanics import effective_h_walls
from src.env.types import MAX_MUMMIES, MAX_SCORPIONS, MAX_TRAPS, EnvState, LevelData


def observe(
  grid_size: int, level: LevelData, state: EnvState
) -> Float[Array, "11 N N"]:
  """Build an 11-channel grid observation.

  Channels:
    0: North wall (h_walls[r][c] for each cell)
    1: South wall (h_walls[r+1][c] for each cell)
    2: West wall  (v_walls[r][c] for each cell)
    3: East wall  (v_walls[r][c+1] for each cell)
    4: Player position
    5: Alive mummies
    6: Alive scorpions
    7: Active traps
    8: Key (if has_key_gate)
    9: Exit cell
   10: Gate open (scalar broadcast to full grid)
  """
  n = grid_size
  h_walls = effective_h_walls(level, state.gate_open)
  v_walls = level.v_walls_base

  # Wall channels
  north_wall = h_walls[:n, :].astype(jnp.float32)
  south_wall = h_walls[1:, :].astype(jnp.float32)
  west_wall = v_walls[:, :n].astype(jnp.float32)
  east_wall = v_walls[:, 1:].astype(jnp.float32)

  # Entity channels
  player_ch = jnp.zeros((n, n), dtype=jnp.float32)
  player_ch = player_ch.at[state.player[0], state.player[1]].set(1.0)

  mummy_ch = jnp.zeros((n, n), dtype=jnp.float32)
  for i in range(MAX_MUMMIES):
    mummy_ch = mummy_ch.at[state.mummy_pos[i, 0], state.mummy_pos[i, 1]].add(
      state.mummy_alive[i].astype(jnp.float32)
    )

  scorpion_ch = jnp.zeros((n, n), dtype=jnp.float32)
  for i in range(MAX_SCORPIONS):
    scorpion_ch = scorpion_ch.at[
      state.scorpion_pos[i, 0], state.scorpion_pos[i, 1]
    ].add(state.scorpion_alive[i].astype(jnp.float32))

  trap_ch = jnp.zeros((n, n), dtype=jnp.float32)
  for i in range(MAX_TRAPS):
    trap_ch = trap_ch.at[level.trap_pos[i, 0], level.trap_pos[i, 1]].add(
      level.trap_active[i].astype(jnp.float32)
    )

  key_ch = jnp.zeros((n, n), dtype=jnp.float32)
  key_ch = key_ch.at[level.key_pos[0], level.key_pos[1]].add(
    level.has_key_gate.astype(jnp.float32)
  )

  exit_ch = jnp.zeros((n, n), dtype=jnp.float32)
  exit_ch = exit_ch.at[level.exit_cell[0], level.exit_cell[1]].set(1.0)

  gate_ch = jnp.full((n, n), state.gate_open.astype(jnp.float32))

  return jnp.stack(
    [
      north_wall,
      south_wall,
      west_wall,
      east_wall,
      player_ch,
      mummy_ch,
      scorpion_ch,
      trap_ch,
      key_ch,
      exit_ch,
      gate_ch,
    ]
  )
