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
  gate_wall: tuple[int, int] | None  # h_walls index (row+1, col) for south edge
  gate_open: bool
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
  _initial_gate_open: bool = False


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
  gate_wall: tuple[int, int] | None = None

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
      # Gate is a wall on the south edge of the entity's cell.
      # In h_walls, that's h_walls[row + 1][col].
      gate_wall = (ent.row + 1, ent.col)

  # Deep copy walls so toggling the gate doesn't mutate the sublevel
  h_walls = [row[:] for row in sublevel.h_walls]
  v_walls = [row[:] for row in sublevel.v_walls]

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
    gate_wall=gate_wall,
    gate_open=False,
    exit_side=sublevel.exit_side,
    exit_pos=sublevel.exit_pos,
    exit_cell=ec,
    is_red=header.flip,
    _initial_player=player,
    _initial_mummies=list(mummies),
    _initial_scorpions=list(scorpions),
    _initial_gate_open=False,
  )


def restart(state: GameState) -> GameState:
  """Reset mutable game state to initial positions."""
  state.player = state._initial_player
  state.mummies = list(state._initial_mummies)
  state.scorpions = list(state._initial_scorpions)
  # Reset gate wall to closed
  if state.gate_wall is not None and state.gate_open:
    gr, gc = state.gate_wall
    state.h_walls[gr][gc] = not state.h_walls[gr][gc]
  state.gate_open = state._initial_gate_open
  state.alive = True
  state.won = False
  state.turn = 0
  return state


def _toggle_gate(state: GameState) -> None:
  """Toggle the gate wall in h_walls."""
  if state.gate_wall is not None:
    gr, gc = state.gate_wall
    state.h_walls[gr][gc] = not state.h_walls[gr][gc]
    state.gate_open = not state.gate_open


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

  # Wall check (gate wall is part of h_walls, so this covers it)
  return _wall_open(state, r, c, direction)


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

  White (is_red=False): try horizontal first, then vertical.
  Red (is_red=True): try vertical first, then horizontal.
  """
  pr, pc = state.player

  # Determine desired directions
  if is_red:
    # Primary: vertical, secondary: horizontal
    primary = ACTION_NORTH if er > pr else ACTION_SOUTH if er < pr else None
    secondary = ACTION_WEST if ec > pc else ACTION_EAST if ec < pc else None
  else:
    # Primary: horizontal, secondary: vertical
    primary = ACTION_WEST if ec > pc else ACTION_EAST if ec < pc else None
    secondary = ACTION_NORTH if er > pr else ACTION_SOUTH if er < pr else None

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


def step(state: GameState, action: int) -> GameState:
  """Execute one full game turn. Mutates and returns state."""
  if not state.alive or state.won:
    return state

  # 1. Player movement
  prev_player = state.player
  if action != ACTION_WAIT:
    if _can_move(state, state.player[0], state.player[1], action):
      dr, dc = _direction_delta(action)
      state.player = (state.player[0] + dr, state.player[1] + dc)

  # 2. Check if player stepped on trap
  if state.player in state.traps:
    state.alive = False
    return state

  # 3. Check if player stepped on enemy
  all_enemies = state.mummies + state.scorpions
  if state.player in all_enemies:
    state.alive = False
    return state

  # 4. Key/gate toggle from player entering key cell
  if state.key_pos is not None:
    if state.player == state.key_pos and prev_player != state.key_pos:
      _toggle_gate(state)

  # 5. Move enemies
  old_mummies = list(state.mummies)
  old_scorpions = list(state.scorpions)

  new_mummies: list[tuple[int, int]] = []
  for mr, mc in state.mummies:
    # Mummies get 2 steps
    r1, c1 = _move_enemy_one_step(state, mr, mc, state.is_red)
    r2, c2 = _move_enemy_one_step(state, r1, c1, state.is_red)
    new_mummies.append((r2, c2))

  new_scorpions: list[tuple[int, int]] = []
  for sr, sc in state.scorpions:
    # Scorpions get 1 step
    r1, c1 = _move_enemy_one_step(state, sr, sc, state.is_red)
    new_scorpions.append((r1, c1))

  state.mummies = new_mummies
  state.scorpions = new_scorpions

  # 5b. Key/gate toggle from enemies entering key cell (only on entry)
  if state.key_pos is not None:
    for i, pos in enumerate(state.mummies):
      if pos == state.key_pos and old_mummies[i] != state.key_pos:
        _toggle_gate(state)
    for i, pos in enumerate(state.scorpions):
      if pos == state.key_pos and old_scorpions[i] != state.key_pos:
        _toggle_gate(state)

  # 6. Resolve enemy-enemy collisions
  _resolve_collisions(state)

  # 7. Check if any enemy landed on player
  all_enemies = state.mummies + state.scorpions
  if state.player in all_enemies:
    state.alive = False
    return state

  # 8. Win check — player on exit cell and survived enemy movement
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
