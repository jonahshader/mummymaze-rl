"""CNN observation encoder for the JAX Mummy Maze environment."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from src.env.types import MAX_MUMMIES, MAX_SCORPIONS, MAX_TRAPS, EnvState, LevelData


def observe(
  grid_size: int, level: LevelData, state: EnvState
) -> Float[Array, "10 Np1 Np1"]:
  """Build a 10-channel grid observation.

  Wall channels use the raw (N+1)-sized arrays, padded to (N+1)×(N+1).
  Entity channels are N×N padded to (N+1)×(N+1) with zeros.

  Channels:
    0: h_walls — horizontal walls, (N+1, N) padded to (N+1, N+1)
    1: v_walls — vertical walls, (N, N+1) padded to (N+1, N+1)
    2: Player position
    3: Alive mummies (is_red broadcast distinguishes white/red)
    4: Alive scorpions
    5: Active traps
    6: Key position (if has_key_gate)
    7: Exit cell
    8: Gate: 1=open, -1=closed, 0=no gate
    9: Is red (scalar broadcast)
  """
  n = grid_size
  n1 = n + 1

  # Wall channels — pad to (N+1, N+1)
  h_walls = jnp.zeros((n1, n1), dtype=jnp.float32)
  h_walls = h_walls.at[:n1, :n].set(level.h_walls_base.astype(jnp.float32))

  v_walls = jnp.zeros((n1, n1), dtype=jnp.float32)
  v_walls = v_walls.at[:n, :n1].set(level.v_walls_base.astype(jnp.float32))

  # Entity channels — build at N×N then pad to (N+1, N+1)
  def pad(ch: Float[Array, "N N"]) -> Float[Array, "Np1 Np1"]:
    return jnp.pad(ch, ((0, 1), (0, 1)))

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

  # Gate channel: 1=open, -1=closed, 0=no gate
  gate_ch = jnp.zeros((n, n), dtype=jnp.float32)
  gate_val = jnp.where(state.gate_open, 1.0, -1.0)
  gate_val = gate_val * level.has_key_gate.astype(jnp.float32)
  gate_ch = gate_ch.at[level.gate_row, level.gate_col].set(gate_val)

  # Is red — scalar broadcast
  is_red_ch = jnp.full((n1, n1), level.is_red.astype(jnp.float32))

  return jnp.stack(
    [
      h_walls,
      v_walls,
      pad(player_ch),
      pad(mummy_ch),
      pad(scorpion_ch),
      pad(trap_ch),
      pad(key_ch),
      pad(exit_ch),
      pad(gate_ch),
      is_red_ch,
    ]
  )
