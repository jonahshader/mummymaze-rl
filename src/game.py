"""Mummy Maze game engine — pure Python reference implementation.

Implements the full game rules for manual testing and verification against
the original Mummy Maze Deluxe. This is NOT the JAX RL environment; it will
be reimplemented in JAX later.
"""

from dataclasses import dataclass, field
from mummy_maze.parser import EntityType, Header, SubLevel

# Actions
ACTION_NORTH = 0
ACTION_SOUTH = 1
ACTION_EAST = 2
ACTION_WEST = 3
ACTION_WAIT = 4


@dataclass
class GameState:
  h_walls: list[list[bool]]
  v_walls: list[list[bool]]
  grid_size: int
  player: tuple[int, int]  # (row, col)
  mummies: list[tuple[int, int]]
  scorpions: list[tuple[int, int]]
  traps: set[tuple[int, int]]
  key_pos: tuple[int, int] | None
  gate_cell: tuple[int, int] | None  # cell position of the gate
  gate_active: bool  # when True, gate blocks movement through gate_cell
  exit_side: str
  exit_pos: int
  exit_cell: tuple[int, int]
  is_red: bool
  alive: bool = True
  won: bool = False
  turn: int = 0
  # Track initial state for restart
  _initial_player: tuple[int, int] = (0, 0)
  _initial_mummies: list[tuple[int, int]] = field(default_factory=list)
  _initial_scorpions: list[tuple[int, int]] = field(default_factory=list)
  _initial_gate_active: bool = False


def _exit_cell(exit_side: str, exit_pos: int, grid_size: int) -> tuple[int, int]:
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


def load_level(sublevel: SubLevel, header: Header) -> GameState:
  """Construct a GameState from parser output."""
  player = (0, 0)
  mummies: list[tuple[int, int]] = []
  scorpions: list[tuple[int, int]] = []
  traps: set[tuple[int, int]] = set()
  key_pos: tuple[int, int] | None = None
  gate_cell: tuple[int, int] | None = None

  for ent in sublevel.entities:
    pos = (ent.row, ent.col)
    if ent.type == EntityType.PLAYER:
      player = pos
    elif ent.type == EntityType.MUMMY:
      mummies.append(pos)
    elif ent.type == EntityType.SCORPION:
      scorpions.append(pos)
    elif ent.type == EntityType.TRAP:
      traps.add(pos)
    elif ent.type == EntityType.KEY:
      key_pos = pos
    elif ent.type == EntityType.GATE:
      gate_cell = pos

  h_walls = [row[:] for row in sublevel.h_walls]
  v_walls = [row[:] for row in sublevel.v_walls]

  # Do NOT clear any wall at the gate position. The binary keeps the wall
  # and gate as independent blocking mechanisms — when the gate opens, any
  # underlying wall still blocks.

  ec = _exit_cell(sublevel.exit_side, sublevel.exit_pos, header.grid_size)

  return GameState(
    h_walls=h_walls,
    v_walls=v_walls,
    grid_size=header.grid_size,
    player=player,
    mummies=list(mummies),
    scorpions=list(scorpions),
    traps=traps,
    key_pos=key_pos,
    gate_cell=gate_cell,
    gate_active=gate_cell is not None,
    exit_side=sublevel.exit_side,
    exit_pos=sublevel.exit_pos,
    exit_cell=ec,
    is_red=header.flip,
    _initial_player=player,
    _initial_mummies=list(mummies),
    _initial_scorpions=list(scorpions),
    _initial_gate_active=gate_cell is not None,
  )


def restart(state: GameState) -> GameState:
  """Reset mutable game state to initial positions."""
  state.player = state._initial_player
  state.mummies = list(state._initial_mummies)
  state.scorpions = list(state._initial_scorpions)
  state.gate_active = state._initial_gate_active
  state.alive = True
  state.won = False
  state.turn = 0
  return state


# ---------------------------------------------------------------------------
# Movement helpers
# ---------------------------------------------------------------------------


def _can_move(state: GameState, r: int, c: int, direction: int) -> bool:
  """Check if movement from (r, c) in the given direction is unblocked."""
  dr, dc = _direction_delta(direction)
  nr, nc = r + dr, c + dc

  # Bounds check — can't leave the grid
  n = state.grid_size
  if nr < 0 or nr >= n or nc < 0 or nc >= n:
    return False

  # Wall check
  if not _wall_open(state, r, c, direction):
    return False

  # Gate check: gate is a vertical barrier on the EAST edge of gate_cell.
  # It blocks east/west movement, not north/south.
  # East from gate cell: blocked if gate_active
  # West into gate cell: blocked if gate_active (approaching from east side)
  if state.gate_active and state.gate_cell is not None:
    gr, gc = state.gate_cell
    if direction == ACTION_EAST and r == gr and c == gc:
      return False
    if direction == ACTION_WEST and r == gr and c == gc + 1:
      return False

  return True


def _wall_open(state: GameState, r: int, c: int, direction: int) -> bool:
  """Check if there is no wall between (r, c) and the adjacent cell."""
  if direction == ACTION_NORTH:
    return not state.h_walls[r][c]
  if direction == ACTION_SOUTH:
    return not state.h_walls[r + 1][c]
  if direction == ACTION_WEST:
    return not state.v_walls[r][c]
  if direction == ACTION_EAST:
    return not state.v_walls[r][c + 1]
  return False


def _direction_delta(direction: int) -> tuple[int, int]:
  if direction == ACTION_NORTH:
    return (-1, 0)
  if direction == ACTION_SOUTH:
    return (1, 0)
  if direction == ACTION_EAST:
    return (0, 1)
  if direction == ACTION_WEST:
    return (0, -1)
  return (0, 0)


# ---------------------------------------------------------------------------
# Enemy AI
# ---------------------------------------------------------------------------


def _move_enemy_one_step(
  state: GameState,
  er: int,
  ec: int,
  is_red: bool,
) -> tuple[int, int]:
  """Compute one step of enemy chase toward the player.

  White (is_red=False): try vertical first, then horizontal.
  Red (is_red=True): try horizontal first, then vertical.
  """
  pr, pc = state.player

  # Determine desired directions
  if is_red:
    # Primary: horizontal, secondary: vertical
    primary = ACTION_WEST if ec > pc else ACTION_EAST if ec < pc else None
    secondary = ACTION_NORTH if er > pr else ACTION_SOUTH if er < pr else None
  else:
    # Primary: vertical, secondary: horizontal
    primary = ACTION_NORTH if er > pr else ACTION_SOUTH if er < pr else None
    secondary = ACTION_WEST if ec > pc else ACTION_EAST if ec < pc else None

  # Try primary direction
  if primary is not None and _can_move(state, er, ec, primary):
    dr, dc = _direction_delta(primary)
    return (er + dr, ec + dc)

  # Try secondary direction
  if secondary is not None and _can_move(state, er, ec, secondary):
    dr, dc = _direction_delta(secondary)
    return (er + dr, ec + dc)

  # Both blocked — stay
  return (er, ec)


# ---------------------------------------------------------------------------
# Main step function
# ---------------------------------------------------------------------------


def _check_key_toggle(
  state: GameState,
  pos: tuple[int, int],
  old_pos: tuple[int, int],
  original_gate: bool,
  key_toggled: list[bool],
) -> None:
  """Toggle gate if entity entered the key cell (moved onto it).

  Binary behavior: at most one entity toggles via key entry per turn.
  Gate is set to !original_gate (absolute assignment, not relative toggle).
  """
  if key_toggled[0]:
    return
  if state.key_pos is not None and pos == state.key_pos and old_pos != state.key_pos:
    state.gate_active = not original_gate
    key_toggled[0] = True


def step(state: GameState, action: int) -> GameState:
  """Execute one full game turn. Mutates and returns state.

  Order matches the original Mummy Maze Deluxe binary (FUN_00405580):
  1. Player moves
  2. Key toggle (if player entered key cell)
  3. Trap check
  4. Scorpion moves 1 step, then resolve scorpion interactions
  5. Mummy step loop (2 iterations): each mummy moves 1 step,
     then resolve collisions/deaths/key toggles per iteration
  6. Win check
  """
  if not state.alive or state.won:
    return state

  # Save turn-start state for "moved" checks and gate toggle.
  # Binary compares against original state (param_1/puVar4), not per-iteration.
  original_gate = state.gate_active
  orig_mummy_pos = list(state.mummies)  # turn-start positions
  # key_toggled persists across ALL entities for the entire turn.
  # Binary: at most one entity toggles via key entry per turn.
  key_toggled: list[bool] = [False]

  # 1. Player movement
  prev_player = state.player
  if action != ACTION_WAIT:
    if _can_move(state, state.player[0], state.player[1], action):
      dr, dc = _direction_delta(action)
      state.player = (state.player[0] + dr, state.player[1] + dc)

  # 2. Key/gate toggle from player entering key cell (before trap check)
  _check_key_toggle(state, state.player, prev_player, original_gate, key_toggled)

  # 3. Trap check
  if state.player in state.traps:
    state.alive = False
    return state

  # 4. Scorpion movement (scorpions move BEFORE mummies)
  old_scorpions = list(state.scorpions)
  for i, (sr, sc) in enumerate(state.scorpions):
    state.scorpions[i] = _move_enemy_one_step(state, sr, sc, state.is_red)

  # Scorpion-on-player death
  for pos in state.scorpions:
    if pos == state.player:
      state.alive = False
      return state

  # Scorpion-mummy collisions from scorpion stepping onto mummy.
  # Binary does NOT check scorpion_alive; use a separate alive list so
  # dead scorpion positions are preserved (needed for mummy loop below).
  scorpion_alive = [True] * len(state.scorpions)
  for i, spos in enumerate(state.scorpions):
    for mpos in state.mummies:
      if spos == mpos:
        scorpion_alive[i] = False
        break

  # Scorpion key toggles — NO reset of key_toggled (binary carries it
  # from the player toggle). Uses original gate for assignment.
  for i, pos in enumerate(state.scorpions):
    if scorpion_alive[i]:
      _check_key_toggle(state, pos, old_scorpions[i], original_gate, key_toggled)

  # 5. Mummy movement — 2 iterations, interleaved with collision checks.
  # Binary uses alive flags with positions preserved; we do the same.
  mummy_alive = [pos != (-1, -1) for pos in state.mummies]
  for _mummy_step in range(2):
    for i, (mr, mc) in enumerate(state.mummies):
      if not mummy_alive[i]:
        continue
      state.mummies[i] = _move_enemy_one_step(state, mr, mc, state.is_red)

    # Mummy-mummy collisions: if two mummies overlap, later one dies
    for i in range(len(state.mummies)):
      if not mummy_alive[i]:
        continue
      for j in range(i + 1, len(state.mummies)):
        if mummy_alive[j] and state.mummies[i] == state.mummies[j]:
          mummy_alive[j] = False

    # Mummy-scorpion collisions: mummy kills scorpion.
    # Binary does NOT check scorpion_alive — dead scorpion position still
    # triggers the collision (and gate toggle). This matters in iteration 1
    # when the mummy is still on the dead scorpion's position.
    for i, (mr, mc) in enumerate(state.mummies):
      if not mummy_alive[i]:
        continue
      for j, spos in enumerate(state.scorpions):
        if spos == (mr, mc):
          # Gate toggle on mummy-scorpion collision at key cell
          if state.key_pos is not None and spos == state.key_pos:
            state.gate_active = not state.gate_active
          scorpion_alive[j] = False

    # Mummy-player death check
    for i, pos in enumerate(state.mummies):
      if mummy_alive[i] and pos == state.player:
        state.alive = False
        return state

    # Mummy key toggles — mutually exclusive (if/else).
    # "Moved" check compares against turn-start positions, not iteration-start.
    if not key_toggled[0]:
      for i, pos in enumerate(state.mummies):
        if not mummy_alive[i] or i >= len(orig_mummy_pos):
          continue
        if (
          state.key_pos is not None
          and pos == state.key_pos
          and pos != orig_mummy_pos[i]
        ):
          state.gate_active = not original_gate
          if i == 0:
            # Only mummy1 sets key_toggled (binary: mummy2 goto skips it)
            key_toggled[0] = True
          break  # mutually exclusive: first matching mummy wins

  # Clean up dead entities
  state.mummies = [m for i, m in enumerate(state.mummies) if mummy_alive[i]]
  state.scorpions = [s for i, s in enumerate(state.scorpions) if scorpion_alive[i]]

  # 6. Final enemy-on-player check
  if state.player in state.mummies + state.scorpions:
    state.alive = False
    return state

  # 7. Win check
  if state.player == state.exit_cell:
    state.won = True
    return state

  state.turn += 1
  return state


def _resolve_collisions(state: GameState) -> None:
  """Remove enemies that collide. Lower spawn index survives.

  Mummy beats scorpion. Between same type, lower index wins.
  """
  occupied: dict[tuple[int, int], list[tuple[str, int]]] = {}

  for i, pos in enumerate(state.mummies):
    occupied.setdefault(pos, []).append(("mummy", i))
  for i, pos in enumerate(state.scorpions):
    occupied.setdefault(pos, []).append(("scorpion", i))

  mummies_to_remove: set[int] = set()
  scorpions_to_remove: set[int] = set()

  for entities in occupied.values():
    if len(entities) <= 1:
      continue

    # Mummies beat scorpions. Among same type, lowest index survives.
    has_mummy = any(t == "mummy" for t, _ in entities)
    if has_mummy:
      # All scorpions at this cell die
      for t, idx in entities:
        if t == "scorpion":
          scorpions_to_remove.add(idx)
      # All but first mummy die
      mummy_indices = [idx for t, idx in entities if t == "mummy"]
      for idx in mummy_indices[1:]:
        mummies_to_remove.add(idx)
    else:
      # All scorpions, lowest index survives
      scorpion_indices = [idx for t, idx in entities if t == "scorpion"]
      for idx in scorpion_indices[1:]:
        scorpions_to_remove.add(idx)

  if mummies_to_remove:
    state.mummies = [
      m for i, m in enumerate(state.mummies) if i not in mummies_to_remove
    ]
  if scorpions_to_remove:
    state.scorpions = [
      s for i, s in enumerate(state.scorpions) if i not in scorpions_to_remove
    ]
