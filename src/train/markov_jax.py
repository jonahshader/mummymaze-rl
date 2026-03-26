"""Differentiable Markov win-probability via JAX.

Two solvers:
  - value_iteration: unrolled scan, simple but slow convergence (Jacobi).
    Gradients via BPTT through the scan — memory O(n_iters).
  - solve_win_prob: implicit differentiation via adjoint linear system.
    Forward/backward each do independent value iteration.
    Exact gradients with O(1) memory w.r.t. iteration count.

Also provides:
  - GraphSkeleton: pre-computed graph structure for one level (numpy arrays)
  - build_skeleton / build_skeletons: convert Rust build_graph output
  - pad_and_stack: batch variable-size skeletons into padded JAX arrays
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import mummymaze_rust
import numpy as np
from jaxtyping import Array, Bool, Float, Int

_ACTION_MAP = {"N": 0, "S": 1, "E": 2, "W": 3, "wait": 4}


# ---------------------------------------------------------------------------
# Graph skeleton — pre-computed per level (numpy, not JAX)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphSkeleton:
  """Pre-computed state graph structure for a single level.

  All arrays are numpy (host-side). Convert to JAX via pad_and_stack.
  """

  state_tuples: np.ndarray  # (N, 12) int32 — for NN observation
  action_mask: np.ndarray  # (N, 5) bool — valid (non-self-loop) actions
  edge_src: np.ndarray  # (E,) int32 — transient->transient source indices
  edge_action: np.ndarray  # (E,) int32 — action index for each edge
  edge_dst: np.ndarray  # (E,) int32 — transient->transient dest indices
  win_src: np.ndarray  # (W,) int32 — states with a WIN transition
  win_action: np.ndarray  # (W,) int32 — action index for each WIN edge
  start_idx: int
  n_states: int
  n_edges: int
  n_win_edges: int
  bank_idx: int  # index into LevelBank (for computing observations)
  rust_level: object  # mummymaze_rust.Level — for reference eval via Rust


def build_skeleton(level: mummymaze_rust.Level, bank_idx: int) -> GraphSkeleton:
  """Build a GraphSkeleton from a Rust Level via build_graph."""
  graph = mummymaze_rust.build_graph(level)
  states = graph["states"]
  edges_raw = graph["edges"]
  start_idx: int = graph["start_idx"]

  n = len(states)

  # Convert state tuples (mixed int/bool) to int32
  state_tuples = np.array([[int(x) for x in s] for s in states], dtype=np.int32)

  action_mask = np.zeros((n, 5), dtype=bool)
  t_src: list[int] = []
  t_act: list[int] = []
  t_dst: list[int] = []
  w_src: list[int] = []
  w_act: list[int] = []

  for src_idx, action_str, dst in edges_raw:
    a = _ACTION_MAP[action_str]
    action_mask[src_idx, a] = True

    if isinstance(dst, int):
      t_src.append(src_idx)
      t_act.append(a)
      t_dst.append(dst)
    elif dst == "WIN":
      w_src.append(src_idx)
      w_act.append(a)
    # DEAD: only marked in action_mask (contributes probability mass to
    # the implicit dead-absorption, which we don't need for the solve)

  return GraphSkeleton(
    state_tuples=state_tuples,
    action_mask=action_mask,
    edge_src=np.array(t_src, dtype=np.int32),
    edge_action=np.array(t_act, dtype=np.int32),
    edge_dst=np.array(t_dst, dtype=np.int32),
    win_src=np.array(w_src, dtype=np.int32),
    win_action=np.array(w_act, dtype=np.int32),
    start_idx=start_idx,
    n_states=n,
    n_edges=len(t_src),
    n_win_edges=len(w_src),
    bank_idx=bank_idx,
    rust_level=level,
  )


def build_skeletons(
  levels: list[mummymaze_rust.Level],
  bank_indices: list[int],
) -> list[GraphSkeleton]:
  """Build skeletons for a list of levels."""
  return [build_skeleton(lev, bi) for lev, bi in zip(levels, bank_indices)]


# ---------------------------------------------------------------------------
# Batching — pad variable-size skeletons into uniform JAX arrays
# ---------------------------------------------------------------------------


def _next_pow2(n: int) -> int:
  if n <= 1:
    return 1
  return 1 << (n - 1).bit_length()


@dataclass(frozen=True)
class PaddedSkeletonBatch:
  """A batch of graph skeletons padded to common sizes for vmap.

  Shapes use L = n_levels, S = max_states, E = max_edges, W = max_win_edges.
  """

  n_levels: int
  max_states: int
  max_edges: int
  max_win_edges: int

  # Per-level, per-state
  state_tuples: Int[Array, "L S 12"]
  action_mask: Bool[Array, "L S 5"]
  level_idx: Int[Array, "L S"]  # bank index broadcast to every state

  # Per-level edge COO
  edge_src: Int[Array, "L E"]
  edge_action: Int[Array, "L E"]
  edge_dst: Int[Array, "L E"]
  edge_mask: Bool[Array, "L E"]

  # Per-level WIN edges
  win_src: Int[Array, "L W"]
  win_action: Int[Array, "L W"]
  win_mask: Bool[Array, "L W"]

  start_idx: Int[Array, "L"]


def pad_and_stack(
  skeletons: list[GraphSkeleton],
  *,
  round_to_pow2: bool = True,
  fixed_max_states: int | None = None,
  fixed_max_edges: int | None = None,
  fixed_max_win_edges: int | None = None,
) -> PaddedSkeletonBatch:
  """Pad a list of skeletons to common sizes and stack as JAX arrays.

  If fixed_max_* are provided, use those sizes (avoids JIT recompilation
  when different batches have different max sizes).
  """
  L = len(skeletons)
  raw_max_s = max(s.n_states for s in skeletons)
  raw_max_e = max(s.n_edges for s in skeletons)
  raw_max_w = max(s.n_win_edges for s in skeletons)

  if fixed_max_states is not None:
    S = fixed_max_states
    E = fixed_max_edges or _next_pow2(raw_max_e)
    W = fixed_max_win_edges or _next_pow2(max(raw_max_w, 1))
  elif round_to_pow2:
    S = _next_pow2(raw_max_s)
    E = _next_pow2(raw_max_e)
    W = _next_pow2(max(raw_max_w, 1))
  else:
    S = raw_max_s
    E = raw_max_e
    W = max(raw_max_w, 1)

  # Pre-allocate numpy arrays
  st = np.zeros((L, S, 12), dtype=np.int32)
  am = np.zeros((L, S, 5), dtype=bool)
  li = np.zeros((L, S), dtype=np.int32)

  es = np.zeros((L, E), dtype=np.int32)
  ea = np.zeros((L, E), dtype=np.int32)
  ed = np.zeros((L, E), dtype=np.int32)
  em = np.zeros((L, E), dtype=bool)

  ws = np.zeros((L, W), dtype=np.int32)
  wa = np.zeros((L, W), dtype=np.int32)
  wm = np.zeros((L, W), dtype=bool)

  si = np.zeros(L, dtype=np.int32)

  for i, sk in enumerate(skeletons):
    n, ne, nw = sk.n_states, sk.n_edges, sk.n_win_edges
    st[i, :n] = sk.state_tuples
    am[i, :n] = sk.action_mask
    li[i, :] = sk.bank_idx  # broadcast to all slots

    es[i, :ne] = sk.edge_src
    ea[i, :ne] = sk.edge_action
    ed[i, :ne] = sk.edge_dst
    em[i, :ne] = True

    ws[i, :nw] = sk.win_src
    wa[i, :nw] = sk.win_action
    wm[i, :nw] = True

    si[i] = sk.start_idx

  return PaddedSkeletonBatch(
    n_levels=L,
    max_states=S,
    max_edges=E,
    max_win_edges=W,
    state_tuples=jnp.array(st),
    action_mask=jnp.array(am),
    level_idx=jnp.array(li),
    edge_src=jnp.array(es),
    edge_action=jnp.array(ea),
    edge_dst=jnp.array(ed),
    edge_mask=jnp.array(em),
    win_src=jnp.array(ws),
    win_action=jnp.array(wa),
    win_mask=jnp.array(wm),
    start_idx=jnp.array(si),
  )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _policy_from_logits(
  logits: Float[Array, "N 5"],
  action_mask: Bool[Array, "N 5"],
) -> Float[Array, "N 5"]:
  """Masked softmax: invalid actions get ~0 probability."""
  masked = jnp.where(action_mask, logits, jnp.float32(-1e30))
  return jax.nn.softmax(masked, axis=-1)


def _build_markov(
  probs: Float[Array, "N 5"],
  edge_src: Int[Array, "E"],
  edge_action: Int[Array, "E"],
  edge_dst: Int[Array, "E"],
  edge_mask: Bool[Array, "E"],
  win_src: Int[Array, "W"],
  win_action: Int[Array, "W"],
  win_mask: Bool[Array, "W"],
) -> tuple[Float[Array, "E"], Float[Array, "N"]]:
  """Extract transition probs and win-absorption vector from policy probs."""
  N = probs.shape[0]
  p_values = probs[edge_src, edge_action] * edge_mask
  win_p = probs[win_src, win_action] * win_mask
  win_absorb = jnp.zeros(N, dtype=jnp.float32).at[win_src].add(win_p)
  return p_values, win_absorb


def _forward_vi(
  p_values: Float[Array, "E"],
  win_absorb: Float[Array, "N"],
  edge_src: Int[Array, "E"],
  edge_dst: Int[Array, "E"],
  n_iters: int,
  V_init: Float[Array, "N"] | None = None,
) -> Float[Array, "N"]:
  """Value iteration: V <- win_absorb + P^pi @ V."""
  N = win_absorb.shape[0]
  V0 = V_init if V_init is not None else jnp.zeros(N, dtype=jnp.float32)

  def _step(V: Float[Array, "N"], _: None) -> tuple[Float[Array, "N"], None]:
    PV = jnp.zeros(N, dtype=jnp.float32).at[edge_src].add(p_values * V[edge_dst])
    return win_absorb + PV, None

  V, _ = jax.lax.scan(_step, V0, None, length=n_iters)
  return V


def _adjoint_vi(
  p_values: Float[Array, "E"],
  edge_src: Int[Array, "E"],
  edge_dst: Int[Array, "E"],
  start_idx: Int[Array, ""],
  N: int,
  n_iters: int,
  lam_init: Float[Array, "N"] | None = None,
) -> Float[Array, "N"]:
  """Adjoint solve: lambda <- e_start + P^T @ lambda."""
  e_start = jnp.zeros(N, dtype=jnp.float32).at[start_idx].set(1.0)
  lam0 = lam_init if lam_init is not None else jnp.zeros(N, dtype=jnp.float32)

  def _step(lam: Float[Array, "N"], _: None) -> tuple[Float[Array, "N"], None]:
    PT_lam = jnp.zeros(N, dtype=jnp.float32).at[edge_dst].add(p_values * lam[edge_src])
    return e_start + PT_lam, None

  lam, _ = jax.lax.scan(_step, lam0, None, length=n_iters)
  return lam


# ---------------------------------------------------------------------------
# Simple scan-based solver (BPTT gradients)
# ---------------------------------------------------------------------------


def value_iteration(
  logits: Float[Array, "N 5"],
  action_mask: Bool[Array, "N 5"],
  edge_src: Int[Array, "E"],
  edge_action: Int[Array, "E"],
  edge_dst: Int[Array, "E"],
  edge_mask: Bool[Array, "E"],
  win_src: Int[Array, "W"],
  win_action: Int[Array, "W"],
  win_mask: Bool[Array, "W"],
  start_idx: Int[Array, ""],
  n_iters: int = 50,
) -> Float[Array, ""]:
  """Differentiable win probability via unrolled value iteration (BPTT).

  Simple but: (1) Jacobi convergence can be slow, (2) backward memory is O(n_iters).
  Prefer solve_win_prob for training. This is useful for quick tests.
  """
  probs = _policy_from_logits(logits, action_mask)
  p_values, win_absorb = _build_markov(
    probs, edge_src, edge_action, edge_dst, edge_mask, win_src, win_action, win_mask
  )
  V = _forward_vi(p_values, win_absorb, edge_src, edge_dst, n_iters)
  return V[start_idx]


# ---------------------------------------------------------------------------
# Implicit-diff solver (adjoint method, O(1) memory)
# ---------------------------------------------------------------------------


def _make_solve_single(n_fwd_iters: int, n_bwd_iters: int) -> object:
  """Build a single-level differentiable solver with implicit differentiation.

  The forward solves (I - P^pi) V = win_absorb via value iteration.
  The backward solves the adjoint (I - P^pi^T) lambda = e_start,
  then chains through the softmax Jacobian. Memory is O(N) regardless
  of iteration count.
  """

  @jax.custom_vjp
  def _solve(
    logits: Float[Array, "N 5"],
    action_mask: Bool[Array, "N 5"],
    edge_src: Int[Array, "E"],
    edge_action: Int[Array, "E"],
    edge_dst: Int[Array, "E"],
    edge_mask: Bool[Array, "E"],
    win_src: Int[Array, "W"],
    win_action: Int[Array, "W"],
    win_mask: Bool[Array, "W"],
    start_idx: Int[Array, ""],
  ) -> Float[Array, ""]:
    probs = _policy_from_logits(logits, action_mask)
    p_values, win_absorb = _build_markov(
      probs,
      edge_src,
      edge_action,
      edge_dst,
      edge_mask,
      win_src,
      win_action,
      win_mask,
    )
    V = _forward_vi(p_values, win_absorb, edge_src, edge_dst, n_fwd_iters)
    return V[start_idx]

  def _solve_fwd(  # noqa: ANN202
    logits,  # noqa: ANN001
    action_mask,  # noqa: ANN001
    edge_src,  # noqa: ANN001
    edge_action,  # noqa: ANN001
    edge_dst,  # noqa: ANN001
    edge_mask,  # noqa: ANN001
    win_src,  # noqa: ANN001
    win_action,  # noqa: ANN001
    win_mask,  # noqa: ANN001
    start_idx,  # noqa: ANN001
  ):  # custom_vjp fwd — signature mirrors _solve
    probs = _policy_from_logits(logits, action_mask)
    p_values, win_absorb = _build_markov(
      probs,
      edge_src,
      edge_action,
      edge_dst,
      edge_mask,
      win_src,
      win_action,
      win_mask,
    )
    V = _forward_vi(p_values, win_absorb, edge_src, edge_dst, n_fwd_iters)
    result = V[start_idx]
    # Residuals: everything needed for the backward pass.
    residuals = (
      V,
      probs,
      action_mask,
      edge_src,
      edge_action,
      edge_dst,
      edge_mask,
      win_src,
      win_action,
      win_mask,
      start_idx,
    )
    return result, residuals

  def _solve_bwd(residuals, g):  # noqa: ANN001, ANN202
    (
      V,
      probs,
      action_mask,
      edge_src,
      edge_action,
      edge_dst,
      edge_mask,
      win_src,
      win_action,
      win_mask,
      start_idx,
    ) = residuals

    N = V.shape[0]
    p_values = probs[edge_src, edge_action] * edge_mask

    # --- Adjoint solve: (I - P^T) lambda = e_start ---
    # Iteration: lambda <- e_start + P^T @ lambda
    # P^T @ lambda: for edge (src, dst, p), scatter p * lambda[src] into [dst]
    e_start = jnp.zeros(N, dtype=jnp.float32).at[start_idx].set(1.0)

    def _adj_step(lam, _):  # noqa: ANN001, ANN202
      PT_lam = (
        jnp.zeros(N, dtype=jnp.float32).at[edge_dst].add(p_values * lam[edge_src])
      )
      return e_start + PT_lam, None

    lam, _ = jax.lax.scan(
      _adj_step, jnp.zeros(N, dtype=jnp.float32), None, length=n_bwd_iters
    )

    # --- Gradient w.r.t. action probabilities ---
    # dV_start/d pi(a|s) = lambda[s] * V_next(s,a)
    # where V_next = V[dst] for transient edges, 1.0 for WIN edges.
    grad_probs = jnp.zeros_like(probs)
    grad_probs = grad_probs.at[edge_src, edge_action].add(
      lam[edge_src] * V[edge_dst] * edge_mask
    )
    grad_probs = grad_probs.at[win_src, win_action].add(lam[win_src] * win_mask)

    # --- Chain through softmax Jacobian ---
    # d/d logits[s,a] = pi(a|s) * (grad_probs[s,a] - <grad_probs[s,:], pi(s,:)>)
    dot = jnp.sum(probs * grad_probs, axis=-1, keepdims=True)
    grad_logits = probs * (grad_probs - dot)
    grad_logits = jnp.where(action_mask, grad_logits, 0.0)
    grad_logits = g * grad_logits

    # Non-differentiable inputs get zero cotangents of matching dtype.
    z_am = jnp.zeros_like(action_mask, dtype=jnp.float32)
    z_e = jnp.zeros_like(edge_src, dtype=jnp.float32)
    z_w = jnp.zeros_like(win_src, dtype=jnp.float32)
    z_si = jnp.zeros_like(start_idx, dtype=jnp.float32)
    return (grad_logits, z_am, z_e, z_e, z_e, z_e, z_w, z_w, z_w, z_si)

  _solve.defvjp(_solve_fwd, _solve_bwd)
  return _solve


# Default solver: 500 forward + 500 adjoint iterations.
# Converges to ~1e-6 for typical gs=6 levels.
solve_win_prob = _make_solve_single(n_fwd_iters=500, n_bwd_iters=500)


# ---------------------------------------------------------------------------
# Surrogate solver — supports warm-start caching, no custom_vjp needed
# ---------------------------------------------------------------------------


def make_solve_surrogate(
  n_fwd_iters: int = 20,
  n_bwd_iters: int = 20,
) -> object:
  """Build a differentiable solver using the surrogate gradient trick.

  Both forward (V) and adjoint (lambda) solves run under stop_gradient.
  A surrogate expression then reconstructs the correct gradient through
  live policy probs. Supports warm-starting from cached V/lambda.

  Returns a function:
    (logits, action_mask, ..., V_init, lam_init) -> (V_start, V, lam)
  where V_start has the correct gradient, and V/lam are for caching.
  """

  def _solve(
    logits: Float[Array, "N 5"],
    action_mask: Bool[Array, "N 5"],
    edge_src: Int[Array, "E"],
    edge_action: Int[Array, "E"],
    edge_dst: Int[Array, "E"],
    edge_mask: Bool[Array, "E"],
    win_src: Int[Array, "W"],
    win_action: Int[Array, "W"],
    win_mask: Bool[Array, "W"],
    start_idx: Int[Array, ""],
    V_init: Float[Array, "N"],
    lam_init: Float[Array, "N"],
  ) -> tuple[Float[Array, ""], Float[Array, "N"], Float[Array, "N"]]:
    N = logits.shape[0]

    # Policy: live (gradient flows through here via surrogate)
    probs = _policy_from_logits(logits, action_mask)

    # Stopped policy for the VI solves
    probs_sg = jax.lax.stop_gradient(probs)
    p_values_sg, win_absorb_sg = _build_markov(
      probs_sg,
      edge_src,
      edge_action,
      edge_dst,
      edge_mask,
      win_src,
      win_action,
      win_mask,
    )

    # Forward solve (no grad): V = (I - P^pi)^{-1} b
    V = _forward_vi(p_values_sg, win_absorb_sg, edge_src, edge_dst, n_fwd_iters, V_init)
    V = jax.lax.stop_gradient(V)

    # Adjoint solve (no grad): (I - P^T) lam = e_start
    lam = _adjoint_vi(
      p_values_sg, edge_src, edge_dst, start_idx, N, n_bwd_iters, lam_init
    )
    lam = jax.lax.stop_gradient(lam)

    # Surrogate: correct gradient through live probs.
    # dV_start/d pi(a|s) = lam[s] * V_next(s,a)
    edge_w = lam[edge_src] * V[edge_dst] * edge_mask
    win_w = lam[win_src] * win_mask
    surrogate = jnp.sum(probs[edge_src, edge_action] * edge_w) + jnp.sum(
      probs[win_src, win_action] * win_w
    )

    # Straight-through: value = V[start], gradient = d(surrogate)/d(theta)
    V_start = V[start_idx]
    result = V_start + (surrogate - jax.lax.stop_gradient(surrogate))

    return result, V, lam

  return _solve


# ---------------------------------------------------------------------------
# Observation precomputation
# ---------------------------------------------------------------------------


def precompute_obs(
  skeletons: list[GraphSkeleton],
  grid_size: int,
  bank: object,
  make_obs_fn: object,
  batch_size: int = 4096,
) -> tuple[jax.Array, list[int]]:
  """Pre-compute observations for all states across all skeletons.

  Returns (obs_bank, offsets) where:
    obs_bank: (total_states, 10, H, W) — all observations concatenated
    offsets: list of per-skeleton start indices into obs_bank
  """
  jit_obs = jax.jit(lambda t, li: make_obs_fn(grid_size, bank, t, li))

  # Concatenate all state tuples and level indices, track offsets
  all_tuples: list[np.ndarray] = []
  all_level_idx: list[np.ndarray] = []
  offsets: list[int] = []
  total = 0
  for skel in skeletons:
    offsets.append(total)
    all_tuples.append(skel.state_tuples)
    all_level_idx.append(np.full(skel.n_states, skel.bank_idx, dtype=np.int32))
    total += skel.n_states

  flat_tuples = jnp.array(np.concatenate(all_tuples, axis=0))
  flat_level_idx = jnp.array(np.concatenate(all_level_idx, axis=0))

  # Process in batches to avoid OOM
  obs_parts: list[np.ndarray] = []
  for start in range(0, total, batch_size):
    end = min(start + batch_size, total)
    obs = jit_obs(flat_tuples[start:end], flat_level_idx[start:end])
    obs_parts.append(np.array(obs))

  obs_bank = jnp.array(np.concatenate(obs_parts, axis=0))
  return obs_bank, offsets


def build_obs_indices(
  skeletons: list[GraphSkeleton],
  offsets: list[int],
  skeleton_list_indices: list[int],
  max_states: int,
) -> np.ndarray:
  """Build padded obs indices for a batch of skeletons.

  Returns (L, max_states) int32 array indexing into the obs bank.
  Padded slots get index 0 (safe — masked out in the solve).
  """
  L = len(skeleton_list_indices)
  idx = np.zeros((L, max_states), dtype=np.int32)
  for i, si in enumerate(skeleton_list_indices):
    n = skeletons[si].n_states
    idx[i, :n] = np.arange(offsets[si], offsets[si] + n)
  return idx
