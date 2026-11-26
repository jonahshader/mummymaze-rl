"""The surrogate Markov solver against an exact dense solve.

The reference solves (I - P) V = b and its adjoint in float64 with numpy and
applies the softmax Jacobian by hand, so it shares no code with the solver.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mummymaze_rust
from src.train.markov_jax import GraphSkeleton, build_skeleton, make_solve_surrogate

MAZES = Path(__file__).resolve().parent.parent / "mazes"

# (file, sublevel): small grid-size-6 levels with different state counts.
LEVELS = [("B-0", 84), ("B-2", 56), ("B-5", 14), ("B-5", 77)]


def _skeleton(file: str, sublevel: int) -> GraphSkeleton:
  level = mummymaze_rust.Level.from_file(str(MAZES / f"{file}.dat"), sublevel)
  return build_skeleton(level, 0)


def _near_expert_logits(sk: GraphSkeleton, seed: int) -> np.ndarray:
  """Unit noise everywhere, +4 on the BFS-best actions."""
  rng = np.random.default_rng(seed)
  logits = rng.normal(0.0, 1.0, (sk.n_states, 5)).astype(np.float32)
  best = mummymaze_rust.best_actions_for_levels([sk.rust_level])[0]
  index = {sk.state_tuples[i].tobytes(): i for i in range(sk.n_states)}
  masks = np.frombuffer(best["action_masks"], dtype=np.uint8)
  for state, mask in zip(best["states"], masks):
    i = index[np.array([int(v) for v in state], dtype=np.int32).tobytes()]
    logits[i] += 4.0 * ((int(mask) >> np.arange(5)) & 1)
  return logits


def _reference(sk: GraphSkeleton, logits: np.ndarray) -> tuple[float, np.ndarray]:
  """Exact V[start] and its gradient w.r.t. the logits, in float64."""
  n = sk.n_states
  masked = np.where(sk.action_mask, logits.astype(np.float64), -np.inf)
  e = np.exp(masked - masked.max(axis=1, keepdims=True))
  probs = e / e.sum(axis=1, keepdims=True)

  p = np.zeros((n, n))
  np.add.at(p, (sk.edge_src, sk.edge_dst), probs[sk.edge_src, sk.edge_action])
  b = np.zeros(n)
  np.add.at(b, sk.win_src, probs[sk.win_src, sk.win_action])
  a = np.eye(n) - p
  v = np.linalg.solve(a, b)
  e_start = np.zeros(n)
  e_start[sk.start_idx] = 1.0
  lam = np.linalg.solve(a.T, e_start)

  # dV_start / d pi(a|s) = lam[s] * V[next(s, a)], with V = 1 at a win.
  grad_probs = np.zeros((n, 5))
  np.add.at(
    grad_probs, (sk.edge_src, sk.edge_action), lam[sk.edge_src] * v[sk.edge_dst]
  )
  np.add.at(grad_probs, (sk.win_src, sk.win_action), lam[sk.win_src])
  grad_logits = probs * (grad_probs - (probs * grad_probs).sum(axis=1, keepdims=True))
  return float(v[sk.start_idx]), grad_logits


def _make(n_iters: int):  # noqa: ANN202
  solve = make_solve_surrogate(n_iters, n_iters)

  def value_and_cache(logits, *args):  # noqa: ANN001, ANN002, ANN202
    v_start, v, lam = solve(logits, *args)
    return v_start, (v, lam)

  return jax.jit(jax.value_and_grad(value_and_cache, has_aux=True))


def _graph_args(sk: GraphSkeleton) -> tuple:
  return (
    jnp.asarray(sk.action_mask),
    jnp.asarray(sk.edge_src),
    jnp.asarray(sk.edge_action),
    jnp.asarray(sk.edge_dst),
    jnp.ones(sk.n_edges, dtype=bool),
    jnp.asarray(sk.win_src),
    jnp.asarray(sk.win_action),
    jnp.ones(sk.n_win_edges, dtype=bool),
    jnp.asarray(sk.start_idx, dtype=jnp.int32),
  )


def _grad_error(grad: jax.Array, grad_ref: np.ndarray) -> float:
  diff = np.linalg.norm(np.asarray(grad, dtype=np.float64) - grad_ref)
  return float(diff / np.linalg.norm(grad_ref))


@pytest.mark.parametrize(("file", "sublevel"), LEVELS)
def test_reference_matches_rust(file: str, sublevel: int) -> None:
  """The oracle itself agrees with the Rust Markov solver."""
  sk = _skeleton(file, sublevel)
  logits = _near_expert_logits(sk, seed=0)
  v_ref, _ = _reference(sk, logits)
  masked = np.where(sk.action_mask, logits, -np.inf)
  e = np.exp(masked - masked.max(axis=1, keepdims=True))
  probs = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
  rust = mummymaze_rust.policy_win_prob(sk.rust_level, sk.state_tuples, probs)
  assert rust == pytest.approx(v_ref, abs=1e-6)


@pytest.mark.parametrize(("file", "sublevel"), LEVELS)
def test_converged_solve_is_exact(file: str, sublevel: int) -> None:
  """From a cold start with enough iterations, value and gradient are exact."""
  sk = _skeleton(file, sublevel)
  logits = _near_expert_logits(sk, seed=0)
  v_ref, grad_ref = _reference(sk, logits)
  zeros = jnp.zeros(sk.n_states, dtype=jnp.float32)

  (v, _), grad = _make(500)(jnp.asarray(logits), *_graph_args(sk), zeros, zeros)

  assert float(v) == pytest.approx(v_ref, rel=1e-4)
  if np.linalg.norm(grad_ref) > 1e-6:  # a certain win has no gradient to compare
    assert _grad_error(grad, grad_ref) < 1e-3


@pytest.mark.parametrize(("file", "sublevel"), LEVELS)
def test_warm_start_tracks_a_drifting_policy(file: str, sublevel: int) -> None:
  """20 iterations a step are enough when the cache is carried forward."""
  sk = _skeleton(file, sublevel)
  logits = _near_expert_logits(sk, seed=0)
  args = _graph_args(sk)
  zeros = jnp.zeros(sk.n_states, dtype=jnp.float32)

  (_, (v_cache, lam_cache)), _ = _make(500)(jnp.asarray(logits), *args, zeros, zeros)

  step = _make(20)
  rng = np.random.default_rng(1)
  v, grad = v_cache[sk.start_idx], None
  for _ in range(20):
    logits = logits + rng.normal(0.0, 0.01, logits.shape).astype(np.float32)
    (v, (v_cache, lam_cache)), grad = step(
      jnp.asarray(logits), *args, v_cache, lam_cache
    )

  v_ref, grad_ref = _reference(sk, logits)
  assert float(v) == pytest.approx(v_ref, rel=0.05)
  if np.linalg.norm(grad_ref) > 1e-6:
    assert grad is not None
    assert _grad_error(grad, grad_ref) < 0.15


def test_cold_start_with_too_few_iterations_sees_nothing() -> None:
  """Why the warm start matters: 20 iterations from zero cannot reach a win."""
  sk = _skeleton("B-5", 77)
  logits = _near_expert_logits(sk, seed=0)
  v_ref, _ = _reference(sk, logits)
  zeros = jnp.zeros(sk.n_states, dtype=jnp.float32)

  (v, _), _ = _make(20)(jnp.asarray(logits), *_graph_args(sk), zeros, zeros)

  assert v_ref > 0.1
  assert float(v) < 0.01 * v_ref
