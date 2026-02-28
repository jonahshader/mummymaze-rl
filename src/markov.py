"""Absorbing Markov chain analysis for Mummy Maze state graphs.

Given a StateGraph (from solver.build_graph), computes exact win probability
and expected steps to termination under a uniform-random valid-action policy.
"""

from dataclasses import dataclass

import numpy as np

from src.solver import StateGraph, _DEAD, _WIN


@dataclass(frozen=True)
class MarkovResult:
  win_prob: float  # P(win | start, uniform random policy)
  expected_steps: float  # E[steps to absorption | start]
  n_transient: int


def analyze(graph: StateGraph) -> MarkovResult:
  """Solve the absorbing Markov chain defined by the state graph.

  Assigns uniform probability 1/k over the k valid actions at each state.
  Returns exact win probability and expected steps from the start state.
  """
  transitions = graph.transitions
  n = graph.n_transient
  if n == 0:
    return MarkovResult(win_prob=0.0, expected_steps=0.0, n_transient=0)

  # Map transient states to integer indices
  state_to_idx: dict[tuple[object, ...], int] = {}
  idx_states: list[tuple[object, ...]] = []
  for s in transitions:
    state_to_idx[s] = len(idx_states)
    idx_states.append(s)

  start_idx = state_to_idx[graph.start]

  # Build Q (transient -> transient) and win absorption vector
  q = np.zeros((n, n), dtype=np.float64)
  win_absorb = np.zeros(n, dtype=np.float64)

  for s, s_idx in state_to_idx.items():
    action_map = transitions[s]
    k = len(action_map)
    if k == 0:
      continue
    prob = 1.0 / k
    for next_key in action_map.values():
      if next_key == _WIN:
        win_absorb[s_idx] += prob
      elif next_key == _DEAD:
        pass  # absorbed into dead state, contributes nothing
      else:
        j = state_to_idx[next_key]
        q[s_idx, j] += prob

  # Solve (I - Q) x = win_absorb for absorption probabilities
  a = np.eye(n) - q
  x = np.linalg.solve(a, win_absorb)

  # Solve (I - Q) t = 1 for expected steps to absorption
  t = np.linalg.solve(a, np.ones(n))

  return MarkovResult(
    win_prob=float(x[start_idx]),
    expected_steps=float(t[start_idx]),
    n_transient=n,
  )
