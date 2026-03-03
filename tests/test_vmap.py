"""JIT/vmap smoke tests for the JAX Mummy Maze environment."""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from mummy_maze.parser import Entity, EntityType, Header, SubLevel

from src.env.env import MummyMazeEnv
from src.env.level_bank import LevelBank, load_all_levels, sample_batch
from src.env.level_load import load_level
from src.env.obs import observe
from src.env.step import step
from src.env.types import LevelData


def _make_simple_level() -> tuple[MummyMazeEnv, int, LevelData, SubLevel, Header]:
  """Create a simple 6x6 level for smoke testing."""
  n = 6
  h_walls = [[False] * n for _ in range(n + 1)]
  v_walls = [[False] * (n + 1) for _ in range(n)]
  for i in range(n):
    h_walls[0][i] = True
    h_walls[n][i] = True
    v_walls[i][0] = True
    v_walls[i][n] = True
  h_walls[n][2] = False

  entities = [
    Entity(EntityType.PLAYER, col=2, row=2),
    Entity(EntityType.MUMMY, col=0, row=0),
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
    exit_pos=2,
    entities=entities,
    flip=False,
  )

  grid_size, level, _ = load_level(sublevel, header)
  env = MummyMazeEnv(grid_size=grid_size)
  return env, grid_size, level, sublevel, header


class TestJit:
  """Test that step and observe compile with jax.jit."""

  def test_jit_step(self) -> None:
    """Step function compiles and runs under jit."""
    env, gs, level, _, _ = _make_simple_level()
    state, _ = env.reset(level)

    jitted_step = jax.jit(lambda lv, s, a: step(gs, lv, s, a))
    action = jnp.int32(4)  # WAIT

    new_state = jitted_step(level, state, action)
    assert new_state.player.shape == (2,)
    assert new_state.mummy_pos.shape == (2, 2)

  def test_jit_observe(self) -> None:
    """Observe function compiles under jit."""
    env, gs, level, _, _ = _make_simple_level()
    state, _ = env.reset(level)

    jitted_obs = jax.jit(lambda lv, s: observe(gs, lv, s))
    obs = jitted_obs(level, state)
    assert obs.shape == (10, 7, 7)

  def test_jit_env_step(self) -> None:
    """Full env.step compiles under jit."""
    env, _, level, _, _ = _make_simple_level()
    state, _ = env.reset(level)

    jitted = jax.jit(env.step)
    out = jitted(level, state, jnp.int32(0))
    assert out.obs.shape == (10, 7, 7)
    assert out.reward.shape == ()
    assert out.done.shape == ()


class TestVmap:
  """Test batched execution with jax.vmap."""

  def test_vmap_step_batch(self) -> None:
    """Vmap step across a batch of states with same level."""
    env, gs, level, _, _ = _make_simple_level()
    state, _ = env.reset(level)

    batch_size = 512

    batch_states = jax.tree.map(
      lambda x: jnp.broadcast_to(x, (batch_size, *x.shape)), state
    )
    batch_actions = jnp.full(batch_size, 4, dtype=jnp.int32)

    vmapped_step = jax.vmap(lambda s, a: step(gs, level, s, a))
    batch_new = vmapped_step(batch_states, batch_actions)

    assert batch_new.player.shape == (batch_size, 2)
    assert batch_new.mummy_pos.shape == (batch_size, 2, 2)
    assert batch_new.done.shape == (batch_size,)

  def test_vmap_different_levels(self) -> None:
    """Vmap step across different levels (the key new feature)."""
    env, gs, level, _, _ = _make_simple_level()
    state, _ = env.reset(level)

    batch_size = 32
    # Create batch of levels and states (same level repeated, but structure ok)
    batch_levels = jax.tree.map(
      lambda x: jnp.broadcast_to(x, (batch_size, *x.shape)), level
    )
    batch_states = jax.tree.map(
      lambda x: jnp.broadcast_to(x, (batch_size, *x.shape)), state
    )
    batch_actions = jnp.full(batch_size, 1, dtype=jnp.int32)

    vmapped = jax.vmap(lambda lv, s, a: step(gs, lv, s, a))
    batch_new = vmapped(batch_levels, batch_states, batch_actions)

    assert batch_new.player.shape == (batch_size, 2)

  def test_vmap_observe_batch(self) -> None:
    """Vmap observe across a batch."""
    env, gs, level, _, _ = _make_simple_level()
    state, _ = env.reset(level)

    batch_size = 64
    batch_states = jax.tree.map(
      lambda x: jnp.broadcast_to(x, (batch_size, *x.shape)), state
    )

    vmapped_obs = jax.vmap(lambda s: observe(gs, level, s))
    batch_obs = vmapped_obs(batch_states)
    assert batch_obs.shape == (batch_size, 10, 7, 7)

  def test_vmap_env_step(self) -> None:
    """Vmap full env.step with level batching."""
    env, gs, level, _, _ = _make_simple_level()
    state, _ = env.reset(level)

    batch_size = 128
    batch_levels = jax.tree.map(
      lambda x: jnp.broadcast_to(x, (batch_size, *x.shape)), level
    )
    batch_states = jax.tree.map(
      lambda x: jnp.broadcast_to(x, (batch_size, *x.shape)), state
    )
    batch_actions = jnp.full(batch_size, 1, dtype=jnp.int32)

    vmapped = jax.vmap(env.step)
    out = vmapped(batch_levels, batch_states, batch_actions)

    assert out.obs.shape == (batch_size, 10, 7, 7)
    assert out.reward.shape == (batch_size,)
    assert out.done.shape == (batch_size,)

  def test_jit_vmap_compose(self) -> None:
    """jit(vmap(step)) composes correctly."""
    env, gs, level, _, _ = _make_simple_level()
    state, _ = env.reset(level)

    batch_size = 256
    batch_states = jax.tree.map(
      lambda x: jnp.broadcast_to(x, (batch_size, *x.shape)), state
    )
    batch_actions = jnp.full(batch_size, 2, dtype=jnp.int32)

    jit_vmapped = jax.jit(jax.vmap(lambda s, a: step(gs, level, s, a)))
    batch_new = jit_vmapped(batch_states, batch_actions)

    assert batch_new.player.shape == (batch_size, 2)
    assert batch_new.done.shape == (batch_size,)

  def test_different_actions_per_env(self) -> None:
    """Each env in the batch takes a different action."""
    env, gs, level, _, _ = _make_simple_level()
    state, _ = env.reset(level)

    batch_size = 5
    batch_states = jax.tree.map(
      lambda x: jnp.broadcast_to(x, (batch_size, *x.shape)), state
    )
    batch_actions = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)

    vmapped = jax.vmap(lambda s, a: step(gs, level, s, a))
    batch_new = vmapped(batch_states, batch_actions)

    players = batch_new.player
    assert players.shape == (5, 2)
    expected = jnp.array([[1, 2], [3, 2], [2, 3], [2, 1], [2, 2]])
    assert jnp.array_equal(players, expected), f"Expected {expected}, got {players}"


# --- Level bank tests ---

DAT_DIR = os.environ.get("MUMMY_MAZE_DAT_DIR", "")

_banks_cache: dict[int, LevelBank] | None = None


def _get_banks() -> dict[int, LevelBank]:
  """Load level banks once across all tests."""
  global _banks_cache  # noqa: PLW0603
  if _banks_cache is None:
    _banks_cache, _ = load_all_levels(Path(DAT_DIR))
  return _banks_cache


@pytest.mark.skipif(
  not DAT_DIR or not Path(DAT_DIR).exists(),
  reason="Set MUMMY_MAZE_DAT_DIR to run level bank tests",
)
class TestLevelBank:
  """Test level bank loading, dedup, and batched sampling."""

  def test_load_all_levels(self) -> None:
    """Load all levels and verify bank structure."""
    banks = _get_banks()
    assert len(banks) > 0

    for gs, bank in banks.items():
      assert gs in (6, 8, 10)
      assert bank.grid_size == gs
      assert bank.n_levels > 0
      assert bank.h_walls_base.shape == (
        bank.n_levels,
        gs + 1,
        gs,
      )
      assert bank.v_walls_base.shape == (
        bank.n_levels,
        gs,
        gs + 1,
      )
      assert len(bank.train_indices) + len(bank.val_indices) == bank.n_levels
      assert len(bank.val_indices) >= 1

  def test_deduplication(self) -> None:
    """Verify that duplicate levels are removed."""
    banks = _get_banks()
    # Total unique levels should be less than 101 * n_dat_files
    total = sum(b.n_levels for b in banks.values())
    n_dat_files = len(list(Path(DAT_DIR).glob("B-*.dat")))
    # 100 sublevels per file, but with overlap + empty files
    assert total < 100 * n_dat_files
    assert total > 0

  def test_sample_batch(self) -> None:
    """Sample a batch from the bank and verify shapes."""
    banks = _get_banks()
    for gs, bank in banks.items():
      key = jax.random.key(0)
      batch = sample_batch(bank, bank.train_indices, key, batch_size=32)
      assert batch.h_walls_base.shape == (32, gs + 1, gs)
      assert batch.is_red.shape == (32,)
      assert batch.initial_player.shape == (32, 2)
      break  # just test one size

  def test_vmap_step_with_bank(self) -> None:
    """Full pipeline: sample from bank, reset, step with vmap."""
    banks = _get_banks()
    gs = min(banks)  # smallest grid for speed
    bank = banks[gs]
    env = MummyMazeEnv(grid_size=gs)

    key = jax.random.key(42)
    batch_size = 64
    batch_levels = sample_batch(bank, bank.train_indices, key, batch_size=batch_size)

    # Reset all envs
    batch_states, batch_obs = jax.vmap(env.reset)(batch_levels)
    assert batch_obs.shape == (batch_size, 10, gs + 1, gs + 1)

    # Step all envs
    batch_actions = jnp.full(batch_size, 4, dtype=jnp.int32)
    out = jax.vmap(env.step)(batch_levels, batch_states, batch_actions)
    assert out.obs.shape == (batch_size, 10, gs + 1, gs + 1)
    assert out.reward.shape == (batch_size,)
