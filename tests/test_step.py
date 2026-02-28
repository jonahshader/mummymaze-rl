"""Compare JAX env step against game.py reference on identical action sequences."""

import os
from collections.abc import Callable
from pathlib import Path

import jax.numpy as jnp
import pytest
from mummy_maze.parser import Entity, EntityType, Header, SubLevel, parse_file

from src.env.level_load import load_level as jax_load_level
from src.env.step import step as jax_step
from src.env.types import EnvState, LevelData
from src.game import (
  ACTION_EAST,
  ACTION_NORTH,
  ACTION_SOUTH,
  ACTION_WAIT,
  ACTION_WEST,
  GameState,
  load_level as py_load_level,
  step as py_step,
)


def _make_simple_sublevel(
  grid_size: int = 6,
  flip: bool = False,
) -> tuple[SubLevel, Header]:
  """Create a minimal sublevel: player at (2,2), mummy at (0,0), exit S col 2."""
  n = grid_size
  h_walls = [[False] * n for _ in range(n + 1)]
  v_walls = [[False] * (n + 1) for _ in range(n)]

  # Set borders
  for i in range(n):
    h_walls[0][i] = True
    h_walls[n][i] = True
    v_walls[i][0] = True
    v_walls[i][n] = True

  # Open exit on south side at col 2
  h_walls[n][2] = False

  entities = [
    Entity(EntityType.PLAYER, col=2, row=2),
    Entity(EntityType.MUMMY, col=0, row=0),
  ]

  header = Header(
    grid_size=n,
    flip=flip,
    num_sublevels=1,
    mummy_count=1,
    key_gate=0,
    trap_count=0,
    scorpion=0,
    wall_bytes=n * (2 if n > 8 else 1) * 2,
    bytes_per_sub=0,
  )

  sublevel = SubLevel(
    h_walls=h_walls,
    v_walls=v_walls,
    exit_side="S",
    exit_pos=2,
    entities=entities,
    flip=flip,
  )

  return sublevel, header


def _compare_states(
  py_state: GameState,
  jax_state: EnvState,
  level: LevelData,
  turn_label: str,
) -> None:
  """Assert that Python and JAX states agree on all key fields."""
  # Player
  assert (
    int(jax_state.player[0]) == py_state.player[0]
    and int(jax_state.player[1]) == py_state.player[1]
  ), (
    f"{turn_label}: player mismatch: "
    f"py={py_state.player} "
    f"jax={tuple(int(x) for x in jax_state.player)}"
  )

  # Mummies (check alive ones match python list)
  py_mummy_idx = 0
  for i in range(2):
    jax_alive = bool(jax_state.mummy_alive[i])
    if jax_alive and py_mummy_idx < len(py_state.mummies):
      py_mr, py_mc = py_state.mummies[py_mummy_idx]
      jax_mr = int(jax_state.mummy_pos[i, 0])
      jax_mc = int(jax_state.mummy_pos[i, 1])
      assert jax_mr == py_mr and jax_mc == py_mc, (
        f"{turn_label}: mummy {i} mismatch: "
        f"py=({py_mr},{py_mc}) jax=({jax_mr},{jax_mc})"
      )
      py_mummy_idx += 1

  # Scorpions
  py_scorp_idx = 0
  for i in range(1):
    jax_alive = bool(jax_state.scorpion_alive[i])
    if jax_alive and py_scorp_idx < len(py_state.scorpions):
      py_sr, py_sc = py_state.scorpions[py_scorp_idx]
      jax_sr = int(jax_state.scorpion_pos[i, 0])
      jax_sc = int(jax_state.scorpion_pos[i, 1])
      assert jax_sr == py_sr and jax_sc == py_sc, (
        f"{turn_label}: scorpion {i} mismatch: "
        f"py=({py_sr},{py_sc}) jax=({jax_sr},{jax_sc})"
      )
      py_scorp_idx += 1

  # Done / alive
  jax_done = bool(jax_state.done)
  py_done = not py_state.alive or py_state.won
  assert jax_done == py_done, (
    f"{turn_label}: done mismatch: py_done={py_done} jax_done={jax_done}"
  )

  # Won
  jax_won = bool(jax_state.won)
  assert jax_won == py_state.won, (
    f"{turn_label}: won mismatch: py={py_state.won} jax={jax_won}"
  )

  # Gate open (JAX gate_open=True means open; game.py gate_active=True means closed)
  if bool(level.has_key_gate):
    py_gate_open = not py_state.gate_active
    assert bool(jax_state.gate_open) == py_gate_open, (
      f"{turn_label}: gate_open mismatch: py_gate_active={py_state.gate_active} "
      f"jax_gate_open={bool(jax_state.gate_open)}"
    )


_jit_cache: dict[int, Callable] = {}


def _get_jitted_step(grid_size: int) -> Callable:
  """Cache a JIT-compiled step per grid_size."""
  if grid_size not in _jit_cache:
    import jax

    _jit_cache[grid_size] = jax.jit(lambda lv, s, a: jax_step(grid_size, lv, s, a))
  return _jit_cache[grid_size]


def _run_action_sequence(
  sublevel: SubLevel,
  header: Header,
  actions: list[int],
) -> None:
  """Run actions through both engines and compare at every step."""
  py_state = py_load_level(sublevel, header)
  grid_size, level, jax_state = jax_load_level(sublevel, header)
  jitted_step = _get_jitted_step(grid_size)

  _compare_states(py_state, jax_state, level, "initial")

  for i, action in enumerate(actions):
    py_state = py_step(py_state, action)
    jax_state = jitted_step(level, jax_state, jnp.int32(action))
    _compare_states(py_state, jax_state, level, f"step {i} (action={action})")
    if bool(jax_state.done):
      break


class TestSimpleLevel:
  """Test with synthetic levels."""

  def test_wait_sequence(self) -> None:
    """Mummy chases player while player waits."""
    sublevel, header = _make_simple_sublevel()
    _run_action_sequence(sublevel, header, [ACTION_WAIT] * 5)

  def test_player_moves_south(self) -> None:
    """Player moves south toward exit."""
    sublevel, header = _make_simple_sublevel()
    _run_action_sequence(sublevel, header, [ACTION_SOUTH, ACTION_SOUTH, ACTION_SOUTH])

  def test_player_moves_all_directions(self) -> None:
    """Player exercises all movement directions."""
    sublevel, header = _make_simple_sublevel()
    actions = [ACTION_EAST, ACTION_SOUTH, ACTION_WEST, ACTION_NORTH, ACTION_WAIT]
    _run_action_sequence(sublevel, header, actions)

  def test_player_into_wall(self) -> None:
    """Player tries to walk into border wall — should stay put."""
    sublevel, header = _make_simple_sublevel()
    actions = [ACTION_NORTH, ACTION_NORTH, ACTION_NORTH]
    _run_action_sequence(sublevel, header, actions)

  def test_red_mummy(self) -> None:
    """Red mummy chase prioritizes vertical."""
    sublevel, header = _make_simple_sublevel(flip=True)
    _run_action_sequence(sublevel, header, [ACTION_WAIT] * 5)


class TestWithWalls:
  """Test with internal walls."""

  def test_wall_blocks_mummy(self) -> None:
    """Internal wall forces mummy to take alternate route."""
    n = 6
    h_walls = [[False] * n for _ in range(n + 1)]
    v_walls = [[False] * (n + 1) for _ in range(n)]

    for i in range(n):
      h_walls[0][i] = True
      h_walls[n][i] = True
      v_walls[i][0] = True
      v_walls[i][n] = True

    h_walls[n][5] = False

    for c in range(4):
      h_walls[2][c] = True

    entities = [
      Entity(EntityType.PLAYER, col=3, row=3),
      Entity(EntityType.MUMMY, col=1, row=0),
    ]

    header = Header(
      grid_size=n,
      flip=False,
      num_sublevels=1,
      mummy_count=1,
      key_gate=0,
      trap_count=0,
      scorpion=0,
      wall_bytes=n * 2,
      bytes_per_sub=0,
    )
    sublevel = SubLevel(
      h_walls=h_walls,
      v_walls=v_walls,
      exit_side="S",
      exit_pos=5,
      entities=entities,
      flip=False,
    )

    _run_action_sequence(sublevel, header, [ACTION_WAIT] * 10)


class TestTraps:
  """Test trap death."""

  def test_player_steps_on_trap(self) -> None:
    """Player walks into a trap and dies."""
    n = 6
    h_walls = [[False] * n for _ in range(n + 1)]
    v_walls = [[False] * (n + 1) for _ in range(n)]
    for i in range(n):
      h_walls[0][i] = True
      h_walls[n][i] = True
      v_walls[i][0] = True
      v_walls[i][n] = True
    h_walls[n][5] = False

    entities = [
      Entity(EntityType.PLAYER, col=2, row=2),
      Entity(EntityType.MUMMY, col=5, row=5),
      Entity(EntityType.TRAP, col=2, row=3),
    ]

    header = Header(
      grid_size=n,
      flip=False,
      num_sublevels=1,
      mummy_count=1,
      key_gate=0,
      trap_count=1,
      scorpion=0,
      wall_bytes=n * 2,
      bytes_per_sub=0,
    )
    sublevel = SubLevel(
      h_walls=h_walls,
      v_walls=v_walls,
      exit_side="S",
      exit_pos=5,
      entities=entities,
      flip=False,
    )

    _run_action_sequence(sublevel, header, [ACTION_SOUTH])


class TestKeyGate:
  """Test key/gate mechanics."""

  def test_key_toggles_gate(self) -> None:
    """Player steps on key, gate toggles."""
    n = 6
    h_walls = [[False] * n for _ in range(n + 1)]
    v_walls = [[False] * (n + 1) for _ in range(n)]
    for i in range(n):
      h_walls[0][i] = True
      h_walls[n][i] = True
      v_walls[i][0] = True
      v_walls[i][n] = True
    h_walls[n][5] = False

    h_walls[3][3] = True

    entities = [
      Entity(EntityType.PLAYER, col=3, row=1),
      Entity(EntityType.MUMMY, col=0, row=0),
      Entity(EntityType.KEY, col=3, row=2),
      Entity(EntityType.GATE, col=3, row=2),
    ]

    header = Header(
      grid_size=n,
      flip=False,
      num_sublevels=1,
      mummy_count=1,
      key_gate=1,
      trap_count=0,
      scorpion=0,
      wall_bytes=n * 2,
      bytes_per_sub=0,
    )
    sublevel = SubLevel(
      h_walls=h_walls,
      v_walls=v_walls,
      exit_side="S",
      exit_pos=5,
      entities=entities,
      flip=False,
    )

    _run_action_sequence(
      sublevel,
      header,
      [ACTION_SOUTH, ACTION_SOUTH, ACTION_SOUTH],
    )


class TestScorpion:
  """Test scorpion mechanics."""

  def test_scorpion_one_step(self) -> None:
    """Scorpion moves 1 step toward player."""
    n = 6
    h_walls = [[False] * n for _ in range(n + 1)]
    v_walls = [[False] * (n + 1) for _ in range(n)]
    for i in range(n):
      h_walls[0][i] = True
      h_walls[n][i] = True
      v_walls[i][0] = True
      v_walls[i][n] = True
    h_walls[n][5] = False

    entities = [
      Entity(EntityType.PLAYER, col=2, row=2),
      Entity(EntityType.MUMMY, col=5, row=5),
      Entity(EntityType.SCORPION, col=0, row=0),
    ]

    header = Header(
      grid_size=n,
      flip=False,
      num_sublevels=1,
      mummy_count=1,
      key_gate=0,
      trap_count=0,
      scorpion=1,
      wall_bytes=n * 2,
      bytes_per_sub=0,
    )
    sublevel = SubLevel(
      h_walls=h_walls,
      v_walls=v_walls,
      exit_side="S",
      exit_pos=5,
      entities=entities,
      flip=False,
    )

    _run_action_sequence(sublevel, header, [ACTION_WAIT] * 8)


# --- Tests using .dat files (skipped if no data directory) ---

DAT_DIR = os.environ.get("MUMMY_MAZE_DAT_DIR", "")


@pytest.mark.skipif(
  not DAT_DIR or not Path(DAT_DIR).exists(),
  reason="Set MUMMY_MAZE_DAT_DIR to run .dat file tests",
)
class TestDatFiles:
  """Run reference comparison on real .dat file levels."""

  def _get_dat_files(self) -> list[Path]:
    return sorted(Path(DAT_DIR).glob("B-*.dat"))

  def test_first_sublevel_wait_sequence(self) -> None:
    """First sublevel of each .dat file, 20 wait actions."""
    for dat_path in self._get_dat_files()[:5]:
      parsed = parse_file(dat_path)
      if parsed is None or not parsed.sublevels:
        continue
      sublevel = parsed.sublevels[0]
      _run_action_sequence(
        sublevel,
        parsed.header,
        [ACTION_WAIT] * 20,
      )

  def test_first_sublevel_mixed_actions(self) -> None:
    """First sublevel, mixed action sequence."""
    actions = [
      ACTION_SOUTH,
      ACTION_EAST,
      ACTION_NORTH,
      ACTION_WEST,
      ACTION_WAIT,
      ACTION_SOUTH,
      ACTION_SOUTH,
      ACTION_EAST,
      ACTION_EAST,
      ACTION_NORTH,
    ] * 3

    for dat_path in self._get_dat_files()[:5]:
      parsed = parse_file(dat_path)
      if parsed is None or not parsed.sublevels:
        continue
      sublevel = parsed.sublevels[0]
      _run_action_sequence(sublevel, parsed.header, actions)

  def test_multiple_sublevels(self) -> None:
    """First 10 sublevels of first .dat file."""
    dat_files = self._get_dat_files()
    if not dat_files:
      pytest.skip("No .dat files found")
    parsed = parse_file(dat_files[0])
    if parsed is None:
      pytest.skip("Failed to parse")

    actions = [
      ACTION_WAIT,
      ACTION_SOUTH,
      ACTION_EAST,
      ACTION_NORTH,
      ACTION_WEST,
    ] * 4
    for sublevel in parsed.sublevels[:10]:
      _run_action_sequence(sublevel, parsed.header, actions)
