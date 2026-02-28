"""Convert parser SubLevel/Header to JAX LevelData + EnvState."""

import jax.numpy as jnp
from mummy_maze.parser import EntityType, Header, SubLevel

from src.env.types import MAX_TRAPS, EnvState, LevelData


def exit_cell(exit_side: str, exit_pos: int, grid_size: int) -> tuple[int, int]:
  """Compute the in-grid cell adjacent to the exit opening."""
  n = grid_size
  if exit_side == "N":
    return (0, exit_pos)
  if exit_side == "S":
    return (n - 1, exit_pos)
  if exit_side == "W":
    return (exit_pos, 0)
  if exit_side == "E":
    return (exit_pos, n - 1)
  msg = f"Invalid exit side: {exit_side}"
  raise ValueError(msg)


def load_level(sublevel: SubLevel, header: Header) -> tuple[int, LevelData, EnvState]:
  """Convert parser output to (grid_size, LevelData, initial EnvState)."""
  n = header.grid_size

  # Parse entities
  player = (0, 0)
  mummies: list[tuple[int, int]] = []
  scorpions: list[tuple[int, int]] = []
  traps: list[tuple[int, int]] = []
  key_pos: tuple[int, int] | None = None
  gate_pos: tuple[int, int] | None = None

  for ent in sublevel.entities:
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
      gate_pos = (ent.row, ent.col)

  # Convert walls to JAX arrays
  h_walls_base = jnp.array(sublevel.h_walls, dtype=jnp.bool_)
  v_walls_base = jnp.array(sublevel.v_walls, dtype=jnp.bool_)

  # Pad mummies to fixed size 2
  mummy_arr = jnp.zeros((2, 2), dtype=jnp.int32)
  mummy_alive = jnp.zeros(2, dtype=jnp.bool_)
  for i, (mr, mc) in enumerate(mummies[:2]):
    mummy_arr = mummy_arr.at[i].set(jnp.array([mr, mc], dtype=jnp.int32))
    mummy_alive = mummy_alive.at[i].set(True)

  # Pad scorpions to fixed size 1
  scorpion_arr = jnp.zeros((1, 2), dtype=jnp.int32)
  scorpion_alive = jnp.zeros(1, dtype=jnp.bool_)
  for i, (sr, sc) in enumerate(scorpions[:1]):
    scorpion_arr = scorpion_arr.at[i].set(jnp.array([sr, sc], dtype=jnp.int32))
    scorpion_alive = scorpion_alive.at[i].set(True)

  # Pad traps to fixed size MAX_TRAPS with active mask
  trap_arr = jnp.zeros((MAX_TRAPS, 2), dtype=jnp.int32)
  trap_active = jnp.zeros(MAX_TRAPS, dtype=jnp.bool_)
  for i, (tr, tc) in enumerate(traps[:MAX_TRAPS]):
    trap_arr = trap_arr.at[i].set(jnp.array([tr, tc], dtype=jnp.int32))
    trap_active = trap_active.at[i].set(True)

  # Key position (0,0 if no key — safe since has_key_gate masks usage)
  key_arr = jnp.array(key_pos if key_pos is not None else (0, 0), dtype=jnp.int32)

  # Gate cell position (0,0 if no gate — safe since has_key_gate masks usage)
  gate_row = jnp.int32(gate_pos[0] if gate_pos is not None else 0)
  gate_col = jnp.int32(gate_pos[1] if gate_pos is not None else 0)

  # Exit cell
  ec = exit_cell(sublevel.exit_side, sublevel.exit_pos, n)
  exit_arr = jnp.array(ec, dtype=jnp.int32)

  # Player
  player_arr = jnp.array(player, dtype=jnp.int32)

  level = LevelData(
    h_walls_base=h_walls_base,
    v_walls_base=v_walls_base,
    is_red=jnp.bool_(header.flip),
    has_key_gate=jnp.bool_(bool(header.key_gate)),
    gate_row=gate_row,
    gate_col=gate_col,
    trap_pos=trap_arr,
    trap_active=trap_active,
    key_pos=key_arr,
    exit_cell=exit_arr,
    initial_player=player_arr,
    initial_mummy_pos=mummy_arr,
    initial_mummy_alive=mummy_alive,
    initial_scorpion_pos=scorpion_arr,
    initial_scorpion_alive=scorpion_alive,
  )

  state = EnvState(
    player=player_arr,
    mummy_pos=mummy_arr,
    mummy_alive=mummy_alive,
    scorpion_pos=scorpion_arr,
    scorpion_alive=scorpion_alive,
    gate_open=jnp.bool_(False),
    done=jnp.bool_(False),
    won=jnp.bool_(False),
    turn=jnp.int32(0),
  )

  return n, level, state
