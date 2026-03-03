"""Tests for the behavioral cloning dataset and training pipeline.

Test 1: Optimal action consistency — for one level per grid size, verify
that stepping with each optimal action in the JAX env produces a successor
with dist_to_win - 1 (or wins for dist=1 states).

Test 2: Overfit sanity check — train on a single level for 200 steps and
verify near-100% accuracy.
"""

from collections import deque
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest
from mummy_maze.parser import parse_file

import mummymaze_rust
from src.env.level_load import load_level as jax_load_level
from src.env.obs import observe
from src.env.step import step as jax_step
from src.env.types import EnvState
from src.train.model import MazeCNN
from src.train.train_bc import cross_entropy_loss, top1_accuracy

DAT_DIR = Path(__file__).resolve().parent.parent / "mazes"

# One representative level per grid size (file, sublevel)
SAMPLE_LEVELS = [
  ("B-0", 0, 6),  # grid_size 6
  ("B-34", 0, 8),  # grid_size 8
  ("B-68", 0, 10),  # grid_size 10
]


def _build_dist_to_win(
  graph_data: dict,
) -> tuple[dict[tuple, int], list[tuple]]:
  """Backward BFS from WIN using a Rust build_graph result.

  Returns (dist_map, states) where dist_map maps state_tuple -> distance.
  """
  states = graph_data["states"]
  edges = graph_data["edges"]

  reverse: dict[int, list[int]] = {}
  win_preds: list[int] = []
  for edge in edges:
    src_idx, _, dst = edge
    if dst == "WIN":
      win_preds.append(src_idx)
    elif isinstance(dst, int):
      reverse.setdefault(dst, []).append(src_idx)

  dist: dict[int, int] = {}
  queue: deque[int] = deque()
  for s in win_preds:
    if s not in dist:
      dist[s] = 1
      queue.append(s)

  while queue:
    cur = queue.popleft()
    d = dist[cur]
    for pred in reverse.get(cur, []):
      if pred not in dist:
        dist[pred] = d + 1
        queue.append(pred)

  return {states[idx]: d for idx, d in dist.items()}, states


def _env_state_to_rust_tuple(state: EnvState) -> tuple:
  """Convert a single (unbatched) JAX EnvState back to a Rust state tuple."""
  pr = int(state.player[0])
  pc = int(state.player[1])
  m1r = int(state.mummy_pos[0, 0])
  m1c = int(state.mummy_pos[0, 1])
  m1_alive = bool(state.mummy_alive[0])
  m2r = int(state.mummy_pos[1, 0])
  m2c = int(state.mummy_pos[1, 1])
  m2_alive = bool(state.mummy_alive[1])
  sr = int(state.scorpion_pos[0, 0])
  sc = int(state.scorpion_pos[0, 1])
  s_alive = bool(state.scorpion_alive[0])
  gate_open = not bool(state.gate_open)  # invert: JAX open -> Rust blocking

  if not m1_alive:
    m1r, m1c = 99, 99
  if not m2_alive:
    m2r, m2c = 99, 99
  if not s_alive:
    sr, sc = 99, 99

  return (
    pr,
    pc,
    m1r,
    m1c,
    m1_alive,
    m2r,
    m2c,
    m2_alive,
    sr,
    sc,
    s_alive,
    gate_open,
  )


def _rust_tuple_to_env_state(
  st: tuple,
) -> EnvState:
  """Convert a Rust state tuple to a single JAX EnvState."""
  pr, pc = st[0], st[1]
  m1r, m1c, m1_alive = st[2], st[3], st[4]
  m2r, m2c, m2_alive = st[5], st[6], st[7]
  sr, sc, s_alive = st[8], st[9], st[10]
  gate_open_rust = st[11]

  # Clamp dead entities to (0,0)
  if not m1_alive:
    m1r, m1c = 0, 0
  if not m2_alive:
    m2r, m2c = 0, 0
  if not s_alive:
    sr, sc = 0, 0

  return EnvState(
    player=jnp.array([pr, pc], dtype=jnp.int32),
    mummy_pos=jnp.array([[m1r, m1c], [m2r, m2c]], dtype=jnp.int32),
    mummy_alive=jnp.array([m1_alive, m2_alive]),
    scorpion_pos=jnp.array([[sr, sc]], dtype=jnp.int32),
    scorpion_alive=jnp.array([s_alive]),
    gate_open=jnp.bool_(not gate_open_rust),  # invert polarity
    done=jnp.bool_(False),
    won=jnp.bool_(False),
    turn=jnp.int32(0),
  )


class TestOptimalActionConsistency:
  """Verify optimal actions from Rust are consistent with JAX env stepping."""

  @pytest.mark.parametrize("file_stem,sublevel,grid_size", SAMPLE_LEVELS)
  def test_optimal_actions_decrease_dist(
    self,
    file_stem: str,
    sublevel: int,
    grid_size: int,
  ) -> None:
    """Stepping with an optimal action decreases dist_to_win by exactly 1."""
    # Load level in JAX
    parsed = parse_file(DAT_DIR / f"{file_stem}.dat")
    assert parsed is not None
    gs, level_data, _ = jax_load_level(parsed.sublevels[sublevel], parsed.header)
    assert gs == grid_size

    # Get Rust graph + dist_to_win
    graph = mummymaze_rust.build_graph(str(DAT_DIR / f"{file_stem}.dat"), sublevel)
    dist_map, states = _build_dist_to_win(graph)
    assert len(dist_map) > 0, "No winnable states"

    # Build action map from edges
    edges = graph["edges"]
    action_map: dict[int, list[tuple[int, object]]] = {}
    action_names = {"N": 0, "S": 1, "E": 2, "W": 3, "wait": 4}
    for edge in edges:
      src_idx, act_name, dst = edge
      action_map.setdefault(src_idx, []).append((action_names[act_name], dst))

    # Collect all (state, action, expected_dist) triples into arrays
    batch_states: list[EnvState] = []
    batch_actions: list[int] = []
    expect_win: list[bool] = []
    expect_dist: list[int] = []

    for state_idx, st in enumerate(states):
      if st not in dist_map:
        continue
      cur_dist = dist_map[st]

      optimal = []
      for act_idx, dst in action_map.get(state_idx, []):
        if dst == "WIN" and cur_dist == 1:
          optimal.append(act_idx)
        elif isinstance(dst, int) and states[dst] in dist_map:
          if dist_map[states[dst]] == cur_dist - 1:
            optimal.append(act_idx)

      if not optimal:
        continue

      env_state = _rust_tuple_to_env_state(st)
      for action in optimal:
        batch_states.append(env_state)
        batch_actions.append(action)
        expect_win.append(cur_dist == 1)
        expect_dist.append(cur_dist - 1)

    n = len(batch_states)
    assert n > 0

    # Stack into batched arrays and vmap the step
    batched_state = jax.tree.map(lambda *xs: jnp.stack(xs), *batch_states)
    batched_actions = jnp.array(batch_actions, dtype=jnp.int32)

    batched_step = jax.jit(jax.vmap(lambda s, a: jax_step(grid_size, level_data, s, a)))
    new_states = batched_step(batched_state, batched_actions)

    # Check results
    won_np = np.array(new_states.won)
    for i in range(n):
      if expect_win[i]:
        assert won_np[i], f"pair {i}: dist=1, should win"
      else:
        ns = _env_state_to_rust_tuple(jax.tree.map(lambda x: x[i], new_states))
        assert ns in dist_map, f"pair {i}: led to non-winnable state"
        assert dist_map[ns] == expect_dist[i], (
          f"pair {i}: expected dist {expect_dist[i]}, got {dist_map[ns]}"
        )

    print(
      f"\n  {file_stem}:{sublevel} (gs={grid_size}): "
      f"checked {n} state-action pairs "
      f"across {len(dist_map)} winnable states"
    )


class TestOverfitSingleLevel:
  """Train on a single level and verify near-100% accuracy."""

  def test_overfit_one_level(self) -> None:
    """Training on one level for many steps should reach >95% accuracy."""
    file_stem, sublevel, grid_size = "B-0", 0, 6

    # Load level
    parsed = parse_file(DAT_DIR / f"{file_stem}.dat")
    assert parsed is not None
    gs, level_data, _ = jax_load_level(parsed.sublevels[sublevel], parsed.header)
    assert gs == grid_size

    # Get optimal actions from Rust
    graph = mummymaze_rust.build_graph(str(DAT_DIR / f"{file_stem}.dat"), sublevel)
    dist_map, states = _build_dist_to_win(graph)

    # Build action bitmasks
    edges = graph["edges"]
    action_names = {"N": 0, "S": 1, "E": 2, "W": 3, "wait": 4}
    action_map: dict[int, list[tuple[int, object]]] = {}
    for edge in edges:
      src_idx, act_name, dst = edge
      action_map.setdefault(src_idx, []).append((action_names[act_name], dst))

    # Build observations and targets for all winnable states
    obs_list = []
    target_list = []
    for state_idx, st in enumerate(states):
      if st not in dist_map:
        continue
      cur_dist = dist_map[st]

      # Compute optimal action mask
      mask = np.zeros(5, dtype=np.float32)
      for act_idx, dst in action_map.get(state_idx, []):
        if dst == "WIN" and cur_dist == 1:
          mask[act_idx] = 1.0
        elif isinstance(dst, int) and states[dst] in dist_map:
          if dist_map[states[dst]] == cur_dist - 1:
            mask[act_idx] = 1.0

      k = mask.sum()
      if k == 0:
        continue
      mask /= k  # soft labels

      env_state = _rust_tuple_to_env_state(st)
      obs = observe(grid_size, level_data, env_state)
      obs_list.append(obs)
      target_list.append(mask)

    n_states = len(obs_list)
    assert n_states > 5, f"Too few states: {n_states}"

    obs_batch = jnp.stack(obs_list)
    targets = jnp.array(np.stack(target_list))

    # Train
    key = jr.key(123)
    model = MazeCNN(key)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def _step(
      model: MazeCNN,
      opt_state: optax.OptState,
      obs: jax.Array,
      targets: jax.Array,
    ) -> tuple[MazeCNN, optax.OptState, jax.Array, jax.Array]:
      def _loss_fn(
        m: MazeCNN,
      ) -> tuple[jax.Array, jax.Array]:
        logits = jax.vmap(m)(obs)
        return cross_entropy_loss(logits, targets), logits

      (loss, logits), grads = eqx.filter_value_and_grad(_loss_fn, has_aux=True)(model)
      updates, new_opt_state = optimizer.update(
        grads,
        opt_state,
        model,  # type: ignore[arg-type]
      )
      new_model = eqx.apply_updates(model, updates)
      acc = top1_accuracy(logits, targets)
      return new_model, new_opt_state, loss, acc

    loss = jnp.zeros(())
    acc = jnp.zeros(())
    for _i in range(200):
      model, opt_state, loss, acc = _step(model, opt_state, obs_batch, targets)

    final_loss = float(loss)
    final_acc = float(acc)
    print(
      f"\n  Overfit {file_stem}:{sublevel}: "
      f"{n_states} states, loss={final_loss:.4f}, acc={final_acc:.4f}"
    )

    assert final_acc > 0.95, f"Expected >95% accuracy, got {final_acc:.2%}"
    # With soft labels (1/k), minimum loss is the label entropy, not 0
    assert final_loss < 0.35, f"Expected loss <0.35, got {final_loss:.4f}"
