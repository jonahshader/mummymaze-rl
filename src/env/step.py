"""Full turn pipeline for the JAX Mummy Maze environment.

Matches the original Mummy Maze Deluxe binary (FUN_00405580) exactly:
1. Player moves
2. Player key toggle (absolute assignment, not relative)
3. Trap check -> early return
4. Scorpion moves 1 step
5. Scorpion-on-player death -> early return
6. Scorpion-mummy collision (scorpion dies)
7. Scorpion key toggle
8. Mummy loop (2 iterations), each containing:
   a. Each mummy moves 1 step
   b. Mummy-mummy collision
   c. Mummy-scorpion collision (scorpion dies, gate toggles if at key)
   d. Mummy-player death check
   e. Mummy key toggle (mutually exclusive, turn-start moved check)
9. Final enemy-on-player check
10. Win check

grid_size is the only trace-time constant. Everything else (is_red,
has_key_gate, entity counts) is handled via runtime masks.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int

from src.env.mechanics import (
  ACTION_WAIT,
  _DC,
  _DR,
  can_move,
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

  # --- 1. Player movement ---
  pr, pc = state.player[0], state.player[1]
  gate_open = state.gate_open
  can = can_move(grid_size, level, gate_open, pr, pc, action)
  is_move = action != ACTION_WAIT
  do_move = is_move & can

  dr = _DR[action]
  dc = _DC[action]
  new_pr = jnp.where(do_move, pr + dr, pr)
  new_pc = jnp.where(do_move, pc + dc, pc)
  player = jnp.array([new_pr, new_pc])

  # --- 2. Player key toggle (absolute: gate_open = ~original_gate) ---
  original_gate = state.gate_open
  was_on_key = (pr == level.key_pos[0]) & (pc == level.key_pos[1])
  now_on_key = (new_pr == level.key_pos[0]) & (new_pc == level.key_pos[1])
  player_entered_key = level.has_key_gate & now_on_key & ~was_on_key
  gate_open = jnp.where(player_entered_key, ~original_gate, gate_open)
  key_toggled = player_entered_key

  # --- 3. Trap check -> early return if dead ---
  on_trap = jnp.bool_(False)
  for i in range(MAX_TRAPS):
    trap_r, trap_c = level.trap_pos[i, 0], level.trap_pos[i, 1]
    on_trap = on_trap | (level.trap_active[i] & (new_pr == trap_r) & (new_pc == trap_c))

  state_dead_trap = EnvState(
    player=player,
    mummy_pos=state.mummy_pos,
    mummy_alive=state.mummy_alive,
    scorpion_pos=state.scorpion_pos,
    scorpion_alive=state.scorpion_alive,
    gate_open=gate_open,
    done=jnp.bool_(True),
    won=jnp.bool_(False),
    turn=state.turn,
  )
  return jax.lax.cond(
    on_trap,
    lambda: state_dead_trap,
    lambda: _after_player(
      grid_size,
      level,
      state,
      player,
      new_pr,
      new_pc,
      gate_open,
      original_gate,
      key_toggled,
    ),
  )


def _after_player(
  grid_size: int,
  level: LevelData,
  state: EnvState,
  player: Int[Array, "2"],
  pr: Int[Array, ""],
  pc: Int[Array, ""],
  gate_open: Bool[Array, ""],
  original_gate: Bool[Array, ""],
  key_toggled: Bool[Array, ""],
) -> EnvState:
  """Continue after player moved and survived traps."""

  # --- 4. Scorpion movement (1 step each) ---
  old_scorpion_pos = state.scorpion_pos
  scorpion_pos = state.scorpion_pos
  for i in range(MAX_SCORPIONS):
    alive_i = state.scorpion_alive[i]
    sr, sc = scorpion_pos[i, 0], scorpion_pos[i, 1]
    nr, nc = move_enemy_one_step(
      grid_size, level, gate_open, level.is_red, sr, sc, pr, pc
    )
    sr1 = jnp.where(alive_i, nr, sr)
    sc1 = jnp.where(alive_i, nc, sc)
    scorpion_pos = scorpion_pos.at[i].set(jnp.array([sr1, sc1]))

  # --- 5. Scorpion-on-player death check -> early return ---
  scorp_on_player = jnp.bool_(False)
  for i in range(MAX_SCORPIONS):
    scorp_on_player = scorp_on_player | (
      state.scorpion_alive[i] & (scorpion_pos[i, 0] == pr) & (scorpion_pos[i, 1] == pc)
    )

  state_dead_scorp = EnvState(
    player=player,
    mummy_pos=state.mummy_pos,
    mummy_alive=state.mummy_alive,
    scorpion_pos=scorpion_pos,
    scorpion_alive=state.scorpion_alive,
    gate_open=gate_open,
    done=jnp.bool_(True),
    won=jnp.bool_(False),
    turn=state.turn,
  )
  return jax.lax.cond(
    scorp_on_player,
    lambda: state_dead_scorp,
    lambda: _after_scorpion(
      grid_size,
      level,
      state,
      player,
      pr,
      pc,
      gate_open,
      original_gate,
      key_toggled,
      scorpion_pos,
      old_scorpion_pos,
    ),
  )


def _after_scorpion(
  grid_size: int,
  level: LevelData,
  state: EnvState,
  player: Int[Array, "2"],
  pr: Int[Array, ""],
  pc: Int[Array, ""],
  gate_open: Bool[Array, ""],
  original_gate: Bool[Array, ""],
  key_toggled: Bool[Array, ""],
  scorpion_pos: Int[Array, "1 2"],
  old_scorpion_pos: Int[Array, "1 2"],
) -> EnvState:
  """Continue after scorpion moved and survived."""

  # --- 6. Scorpion-mummy collision (scorpion dies stepping onto mummy) ---
  # Binary does NOT check scorpion_alive; dead scorpion positions persist.
  scorpion_alive = state.scorpion_alive
  for si in range(MAX_SCORPIONS):
    scorp_on_mummy = jnp.bool_(False)
    for mi in range(MAX_MUMMIES):
      scorp_on_mummy = scorp_on_mummy | (
        state.mummy_alive[mi]
        & (scorpion_pos[si, 0] == state.mummy_pos[mi, 0])
        & (scorpion_pos[si, 1] == state.mummy_pos[mi, 1])
      )
    scorpion_alive = scorpion_alive.at[si].set(scorpion_alive[si] & ~scorp_on_mummy)

  # --- 7. Scorpion key toggle ---
  # Uses original_gate for absolute assignment, guarded by key_toggled.
  for i in range(MAX_SCORPIONS):
    now_on = (scorpion_pos[i, 0] == level.key_pos[0]) & (
      scorpion_pos[i, 1] == level.key_pos[1]
    )
    was_on = (old_scorpion_pos[i, 0] == level.key_pos[0]) & (
      old_scorpion_pos[i, 1] == level.key_pos[1]
    )
    entered = scorpion_alive[i] & now_on & ~was_on & level.has_key_gate & ~key_toggled
    gate_open = jnp.where(entered, ~original_gate, gate_open)
    key_toggled = key_toggled | entered

  # --- 8. Mummy loop (2 iterations) ---
  # Save turn-start positions for "moved" check in key toggle.
  initial_mummy_pos = state.mummy_pos
  mummy_pos = state.mummy_pos
  mummy_alive = state.mummy_alive
  dead = jnp.bool_(False)

  for _mummy_step in range(2):
    # 8a. Each mummy moves 1 step (skip if player already dead)
    for i in range(MAX_MUMMIES):
      alive_i = mummy_alive[i] & ~dead
      mr, mc = mummy_pos[i, 0], mummy_pos[i, 1]
      nr, nc = move_enemy_one_step(
        grid_size, level, gate_open, level.is_red, mr, mc, pr, pc
      )
      mr1 = jnp.where(alive_i, nr, mr)
      mc1 = jnp.where(alive_i, nc, mc)
      mummy_pos = mummy_pos.at[i].set(jnp.array([mr1, mc1]))

    # 8b. Mummy-mummy collision: lower index survives
    same_pos_mm = (
      ~dead
      & mummy_alive[0]
      & mummy_alive[1]
      & (mummy_pos[0, 0] == mummy_pos[1, 0])
      & (mummy_pos[0, 1] == mummy_pos[1, 1])
    )
    mummy_alive = mummy_alive.at[1].set(mummy_alive[1] & ~same_pos_mm)

    # 8c. Mummy-scorpion collision: scorpion dies, gate toggles if at key.
    # Binary does NOT check scorpion_alive — dead scorpion position still triggers.
    for mi in range(MAX_MUMMIES):
      for si in range(MAX_SCORPIONS):
        same_pos_ms = (
          ~dead
          & mummy_alive[mi]
          & (
            (mummy_pos[mi, 0] == scorpion_pos[si, 0])
            & (mummy_pos[mi, 1] == scorpion_pos[si, 1])
          )
        )
        # Gate toggle on collision at key cell (relative: ~gate_open)
        at_key = (
          level.has_key_gate
          & (scorpion_pos[si, 0] == level.key_pos[0])
          & (scorpion_pos[si, 1] == level.key_pos[1])
        )
        gate_open = jnp.where(same_pos_ms & at_key, ~gate_open, gate_open)
        scorpion_alive = scorpion_alive.at[si].set(scorpion_alive[si] & ~same_pos_ms)

    # 8d. Mummy-player death check
    mummy_on_player = jnp.bool_(False)
    for i in range(MAX_MUMMIES):
      mummy_on_player = mummy_on_player | (
        mummy_alive[i] & (mummy_pos[i, 0] == pr) & (mummy_pos[i, 1] == pc)
      )
    dead = dead | mummy_on_player

    # 8e. Mummy key toggle — mutually exclusive, first matching mummy wins.
    # "Moved" check: compare against turn-start positions (initial_mummy_pos).
    # Absolute assignment: gate_open = ~original_gate.
    # Only mummy0 sets key_toggled; mummy1 toggles gate but doesn't set it.
    # Guarded by ~dead (game.py returns before this on death).
    mummy_entered_this_iter = jnp.bool_(False)
    for i in range(MAX_MUMMIES):
      now_on = (mummy_pos[i, 0] == level.key_pos[0]) & (
        mummy_pos[i, 1] == level.key_pos[1]
      )
      moved = (mummy_pos[i, 0] != initial_mummy_pos[i, 0]) | (
        mummy_pos[i, 1] != initial_mummy_pos[i, 1]
      )
      entered = (
        ~dead
        & mummy_alive[i]
        & now_on
        & moved
        & level.has_key_gate
        & ~key_toggled
        & ~mummy_entered_this_iter
      )
      gate_open = jnp.where(entered, ~original_gate, gate_open)
      # Only mummy0 sets key_toggled (binary: mummy1's goto skips it)
      key_toggled = jnp.where(entered & (i == 0), jnp.bool_(True), key_toggled)
      # Both mummies set per-iteration exclusion (emulates break)
      mummy_entered_this_iter = mummy_entered_this_iter | entered

  # --- 9. Final enemy-on-player check ---
  enemy_on_player = jnp.bool_(False)
  for i in range(MAX_MUMMIES):
    enemy_on_player = enemy_on_player | (
      mummy_alive[i] & (mummy_pos[i, 0] == pr) & (mummy_pos[i, 1] == pc)
    )
  for i in range(MAX_SCORPIONS):
    enemy_on_player = enemy_on_player | (
      scorpion_alive[i] & (scorpion_pos[i, 0] == pr) & (scorpion_pos[i, 1] == pc)
    )
  dead = dead | enemy_on_player

  # --- 10. Win check ---
  on_exit = (pr == level.exit_cell[0]) & (pc == level.exit_cell[1])
  won = on_exit & ~dead
  done = dead | won

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
