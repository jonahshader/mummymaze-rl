"""Full turn pipeline for the JAX Mummy Maze environment.

grid_size is the only trace-time constant. Everything else (is_red,
has_key_gate, entity counts) is handled via runtime masks.
Always loops over max entity counts (2 mummies, 1 scorpion, 2 traps).
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int

from src.env.mechanics import (
  ACTION_WAIT,
  _DC,
  _DR,
  can_move,
  effective_h_walls,
  move_enemy_one_step,
)
from src.env.types import MAX_MUMMIES, MAX_SCORPIONS, MAX_TRAPS, EnvState, LevelData


def step(
  grid_size: int,
  level: LevelData,
  state: EnvState,
  action: Int[Array, ""],
) -> EnvState:
  """Execute one full game turn. Returns new state."""
  return jax.lax.cond(
    state.done | state.won,
    lambda s, _a: s,
    lambda s, a: _live_step(grid_size, level, s, a),
    state,
    action,
  )


def _live_step(
  grid_size: int,
  level: LevelData,
  state: EnvState,
  action: Int[Array, ""],
) -> EnvState:
  """Execute a turn for a live (not done/won) game."""
  h_walls = effective_h_walls(level, state.gate_open)
  v_walls = level.v_walls_base

  # --- 1. Player movement ---
  pr, pc = state.player[0], state.player[1]
  can = can_move(grid_size, h_walls, v_walls, pr, pc, action)
  is_move = action != ACTION_WAIT
  do_move = is_move & can

  dr = _DR[action]
  dc = _DC[action]
  new_pr = jnp.where(do_move, pr + dr, pr)
  new_pc = jnp.where(do_move, pc + dc, pc)
  player = jnp.array([new_pr, new_pc])

  # --- 2. Trap check (always check MAX_TRAPS, masked by trap_active) ---
  on_trap = jnp.bool_(False)
  for i in range(MAX_TRAPS):
    trap_r, trap_c = level.trap_pos[i, 0], level.trap_pos[i, 1]
    on_trap = on_trap | (level.trap_active[i] & (new_pr == trap_r) & (new_pc == trap_c))

  dead = on_trap
  state_after_trap = EnvState(
    player=player,
    mummy_pos=state.mummy_pos,
    mummy_alive=state.mummy_alive,
    scorpion_pos=state.scorpion_pos,
    scorpion_alive=state.scorpion_alive,
    gate_open=state.gate_open,
    done=jnp.bool_(True),
    won=jnp.bool_(False),
    turn=state.turn,
  )
  return jax.lax.cond(
    dead,
    lambda: state_after_trap,
    lambda: _continue_after_player_move(
      grid_size, level, state, player, new_pr, new_pc, h_walls, v_walls
    ),
  )


def _continue_after_player_move(
  grid_size: int,
  level: LevelData,
  state: EnvState,
  player: Int[Array, "2"],
  pr: Int[Array, ""],
  pc: Int[Array, ""],
  h_walls: Bool[Array, "Np1 N"],
  v_walls: Bool[Array, "N Np1"],
) -> EnvState:
  """Continue turn after player has moved and survived traps."""

  # --- 3. Check if player stepped on enemy ---
  on_enemy = jnp.bool_(False)
  for i in range(MAX_MUMMIES):
    alive = state.mummy_alive[i]
    mr, mc = state.mummy_pos[i, 0], state.mummy_pos[i, 1]
    on_enemy = on_enemy | (alive & (pr == mr) & (pc == mc))
  for i in range(MAX_SCORPIONS):
    alive = state.scorpion_alive[i]
    sr, sc = state.scorpion_pos[i, 0], state.scorpion_pos[i, 1]
    on_enemy = on_enemy | (alive & (pr == sr) & (pc == sc))

  state_dead = EnvState(
    player=player,
    mummy_pos=state.mummy_pos,
    mummy_alive=state.mummy_alive,
    scorpion_pos=state.scorpion_pos,
    scorpion_alive=state.scorpion_alive,
    gate_open=state.gate_open,
    done=jnp.bool_(True),
    won=jnp.bool_(False),
    turn=state.turn,
  )

  return jax.lax.cond(
    on_enemy,
    lambda: state_dead,
    lambda: _continue_after_enemy_check(
      grid_size, level, state, player, pr, pc, h_walls, v_walls
    ),
  )


def _continue_after_enemy_check(
  grid_size: int,
  level: LevelData,
  state: EnvState,
  player: Int[Array, "2"],
  pr: Int[Array, ""],
  pc: Int[Array, ""],
  h_walls: Bool[Array, "Np1 N"],
  v_walls: Bool[Array, "N Np1"],
) -> EnvState:
  """Continue after player survived stepping on enemies."""

  # --- 4. Player key toggle (only on entry, not every turn) ---
  gate_open = state.gate_open
  prev_pr, prev_pc = state.player[0], state.player[1]
  was_on_key = (prev_pr == level.key_pos[0]) & (prev_pc == level.key_pos[1])
  now_on_key = (pr == level.key_pos[0]) & (pc == level.key_pos[1])
  entered_key = level.has_key_gate & now_on_key & ~was_on_key
  gate_open = jnp.where(entered_key, ~gate_open, gate_open)
  # Recompute h_walls with new gate state
  h_walls = effective_h_walls(level, gate_open)

  # --- 5. Enemy AI (always loop max, masked by alive) ---
  old_mummy_pos = state.mummy_pos
  old_scorpion_pos = state.scorpion_pos
  mummy_pos = state.mummy_pos
  scorpion_pos = state.scorpion_pos

  # Move mummies (2 steps each)
  for i in range(MAX_MUMMIES):
    alive_i = state.mummy_alive[i]
    mr, mc = mummy_pos[i, 0], mummy_pos[i, 1]
    # Step 1
    nr1, nc1 = move_enemy_one_step(
      grid_size, h_walls, v_walls, level.is_red, mr, mc, pr, pc
    )
    mr1 = jnp.where(alive_i, nr1, mr)
    mc1 = jnp.where(alive_i, nc1, mc)
    # Step 2
    nr2, nc2 = move_enemy_one_step(
      grid_size, h_walls, v_walls, level.is_red, mr1, mc1, pr, pc
    )
    mr2 = jnp.where(alive_i, nr2, mr1)
    mc2 = jnp.where(alive_i, nc2, mc1)
    mummy_pos = mummy_pos.at[i].set(jnp.array([mr2, mc2]))

  # Move scorpions (1 step each)
  for i in range(MAX_SCORPIONS):
    alive_i = state.scorpion_alive[i]
    sr, sc = scorpion_pos[i, 0], scorpion_pos[i, 1]
    nr, nc = move_enemy_one_step(
      grid_size, h_walls, v_walls, level.is_red, sr, sc, pr, pc
    )
    sr1 = jnp.where(alive_i, nr, sr)
    sc1 = jnp.where(alive_i, nc, sc)
    scorpion_pos = scorpion_pos.at[i].set(jnp.array([sr1, sc1]))

  # --- 6. Enemy key toggle (only on entry) ---
  toggle_count = jnp.int32(0)
  for i in range(MAX_MUMMIES):
    alive_i = state.mummy_alive[i]
    now_on = (mummy_pos[i, 0] == level.key_pos[0]) & (
      mummy_pos[i, 1] == level.key_pos[1]
    )
    was_on = (old_mummy_pos[i, 0] == level.key_pos[0]) & (
      old_mummy_pos[i, 1] == level.key_pos[1]
    )
    entered = alive_i & now_on & ~was_on
    toggle_count = toggle_count + jnp.where(entered, jnp.int32(1), jnp.int32(0))
  for i in range(MAX_SCORPIONS):
    alive_i = state.scorpion_alive[i]
    now_on = (scorpion_pos[i, 0] == level.key_pos[0]) & (
      scorpion_pos[i, 1] == level.key_pos[1]
    )
    was_on = (old_scorpion_pos[i, 0] == level.key_pos[0]) & (
      old_scorpion_pos[i, 1] == level.key_pos[1]
    )
    entered = alive_i & now_on & ~was_on
    toggle_count = toggle_count + jnp.where(entered, jnp.int32(1), jnp.int32(0))
  # Only toggle if level actually has key/gate
  should_toggle = level.has_key_gate & (toggle_count % 2 == 1)
  gate_open = jnp.where(should_toggle, ~gate_open, gate_open)

  # --- 7. Collision resolution ---
  mummy_alive = state.mummy_alive
  scorpion_alive = state.scorpion_alive

  # Mummy vs mummy (m0 vs m1) — lower index survives
  same_pos_mm = (
    mummy_alive[0]
    & mummy_alive[1]
    & (mummy_pos[0, 0] == mummy_pos[1, 0])
    & (mummy_pos[0, 1] == mummy_pos[1, 1])
  )
  mummy_alive = mummy_alive.at[1].set(mummy_alive[1] & ~same_pos_mm)

  # Mummy vs scorpion — mummy wins
  for mi in range(MAX_MUMMIES):
    for si in range(MAX_SCORPIONS):
      same_pos_ms = (
        mummy_alive[mi]
        & scorpion_alive[si]
        & (mummy_pos[mi, 0] == scorpion_pos[si, 0])
        & (mummy_pos[mi, 1] == scorpion_pos[si, 1])
      )
      scorpion_alive = scorpion_alive.at[si].set(scorpion_alive[si] & ~same_pos_ms)

  # --- 8. Death check — enemy on player after movement ---
  enemy_on_player = jnp.bool_(False)
  for i in range(MAX_MUMMIES):
    enemy_on_player = enemy_on_player | (
      mummy_alive[i] & (mummy_pos[i, 0] == pr) & (mummy_pos[i, 1] == pc)
    )
  for i in range(MAX_SCORPIONS):
    enemy_on_player = enemy_on_player | (
      scorpion_alive[i] & (scorpion_pos[i, 0] == pr) & (scorpion_pos[i, 1] == pc)
    )

  # --- 9. Win check ---
  on_exit = (pr == level.exit_cell[0]) & (pc == level.exit_cell[1])
  won = on_exit & ~enemy_on_player
  done = enemy_on_player | won

  return EnvState(
    player=player,
    mummy_pos=mummy_pos,
    mummy_alive=mummy_alive,
    scorpion_pos=scorpion_pos,
    scorpion_alive=scorpion_alive,
    gate_open=gate_open,
    done=done,
    won=won,
    turn=jnp.where(done, state.turn, state.turn + 1),
  )
