"""Pytree definitions for the JAX Mummy Maze environment.

Design:
  - grid_size is the only trace-time constant (Python int). Different grid sizes
    (6/8/10) produce different compiled functions.
  - Everything else (is_red, has_key_gate, entity counts) is a runtime value
    with masks, enabling batching across heterogeneous levels.
  - LevelData holds per-level config (walls, positions, masks). Vmappable.
  - EnvState holds per-episode dynamic state. Vmappable.
  - Both are registered as JAX pytrees for jit/vmap compatibility.
"""

import jax
from jaxtyping import Array, Bool, Int

# Max entity counts — always allocate these, use alive/active masks
MAX_MUMMIES = 2
MAX_SCORPIONS = 1
MAX_TRAPS = 2


class LevelData:
  """Per-level configuration. Vmappable across a batch of different levels.

  All levels in a batch must share the same grid_size (determines array shapes),
  but can differ in everything else (walls, entity positions, is_red, etc.).
  """

  # Walls with gate closed
  h_walls_base: Bool[Array, "Np1 N"]
  v_walls_base: Bool[Array, "N Np1"]

  # Level flags (runtime bools)
  is_red: Bool[Array, ""]
  has_key_gate: Bool[Array, ""]

  # Gate cell position (valid index even if no gate — use (0,0))
  gate_row: Int[Array, ""]
  gate_col: Int[Array, ""]

  # Entity positions and masks
  trap_pos: Int[Array, "2 2"]
  trap_active: Bool[Array, "2"]
  key_pos: Int[Array, "2"]
  exit_cell: Int[Array, "2"]

  # Initial positions for reset
  initial_player: Int[Array, "2"]
  initial_mummy_pos: Int[Array, "2 2"]
  initial_mummy_alive: Bool[Array, "2"]
  initial_scorpion_pos: Int[Array, "1 2"]
  initial_scorpion_alive: Bool[Array, "1"]

  def __init__(
    self,
    h_walls_base: Bool[Array, "Np1 N"],
    v_walls_base: Bool[Array, "N Np1"],
    is_red: Bool[Array, ""],
    has_key_gate: Bool[Array, ""],
    gate_row: Int[Array, ""],
    gate_col: Int[Array, ""],
    trap_pos: Int[Array, "2 2"],
    trap_active: Bool[Array, "2"],
    key_pos: Int[Array, "2"],
    exit_cell: Int[Array, "2"],
    initial_player: Int[Array, "2"],
    initial_mummy_pos: Int[Array, "2 2"],
    initial_mummy_alive: Bool[Array, "2"],
    initial_scorpion_pos: Int[Array, "1 2"],
    initial_scorpion_alive: Bool[Array, "1"],
  ) -> None:
    self.h_walls_base = h_walls_base
    self.v_walls_base = v_walls_base
    self.is_red = is_red
    self.has_key_gate = has_key_gate
    self.gate_row = gate_row
    self.gate_col = gate_col
    self.trap_pos = trap_pos
    self.trap_active = trap_active
    self.key_pos = key_pos
    self.exit_cell = exit_cell
    self.initial_player = initial_player
    self.initial_mummy_pos = initial_mummy_pos
    self.initial_mummy_alive = initial_mummy_alive
    self.initial_scorpion_pos = initial_scorpion_pos
    self.initial_scorpion_alive = initial_scorpion_alive


class EnvState:
  """Dynamic game state. Vmapped across batches."""

  player: Int[Array, "2"]
  mummy_pos: Int[Array, "2 2"]
  mummy_alive: Bool[Array, "2"]
  scorpion_pos: Int[Array, "1 2"]
  scorpion_alive: Bool[Array, "1"]
  gate_open: Bool[Array, ""]
  done: Bool[Array, ""]
  won: Bool[Array, ""]
  turn: Int[Array, ""]

  def __init__(
    self,
    player: Int[Array, "2"],
    mummy_pos: Int[Array, "2 2"],
    mummy_alive: Bool[Array, "2"],
    scorpion_pos: Int[Array, "1 2"],
    scorpion_alive: Bool[Array, "1"],
    gate_open: Bool[Array, ""],
    done: Bool[Array, ""],
    won: Bool[Array, ""],
    turn: Int[Array, ""],
  ) -> None:
    self.player = player
    self.mummy_pos = mummy_pos
    self.mummy_alive = mummy_alive
    self.scorpion_pos = scorpion_pos
    self.scorpion_alive = scorpion_alive
    self.gate_open = gate_open
    self.done = done
    self.won = won
    self.turn = turn


# --- Pytree registration ---

_LEVEL_DATA_FIELDS = [
  "h_walls_base",
  "v_walls_base",
  "is_red",
  "has_key_gate",
  "gate_row",
  "gate_col",
  "trap_pos",
  "trap_active",
  "key_pos",
  "exit_cell",
  "initial_player",
  "initial_mummy_pos",
  "initial_mummy_alive",
  "initial_scorpion_pos",
  "initial_scorpion_alive",
]

_ENV_STATE_FIELDS = [
  "player",
  "mummy_pos",
  "mummy_alive",
  "scorpion_pos",
  "scorpion_alive",
  "gate_open",
  "done",
  "won",
  "turn",
]


def _register_pytrees() -> None:
  """Register LevelData and EnvState with JAX."""

  def _flatten_ld(ld: LevelData) -> tuple[list[Array], None]:
    return ([getattr(ld, f) for f in _LEVEL_DATA_FIELDS], None)

  def _unflatten_ld(_aux: None, children: list[Array]) -> LevelData:
    return LevelData(**dict(zip(_LEVEL_DATA_FIELDS, children, strict=True)))

  jax.tree_util.register_pytree_node(LevelData, _flatten_ld, _unflatten_ld)

  def _flatten_es(s: EnvState) -> tuple[list[Array], None]:
    return ([getattr(s, f) for f in _ENV_STATE_FIELDS], None)

  def _unflatten_es(_aux: None, children: list[Array]) -> EnvState:
    return EnvState(**dict(zip(_ENV_STATE_FIELDS, children, strict=True)))

  jax.tree_util.register_pytree_node(EnvState, _flatten_es, _unflatten_es)


_register_pytrees()


def replace_state(state: EnvState, **kwargs: Array) -> EnvState:
  """Create a new EnvState with specified fields replaced."""
  return EnvState(**{f: kwargs.get(f, getattr(state, f)) for f in _ENV_STATE_FIELDS})
