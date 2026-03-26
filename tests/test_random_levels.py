"""Tests for random solvable level generation."""

import mummymaze_rust


def test_generate_random_solvable_basic():
  """Generate levels and verify they are all BFS-solvable."""
  levels = mummymaze_rust.generate_random_solvable(20, grid_size=6, seed=123)
  assert len(levels) == 20
  for lev in levels:
    sol = mummymaze_rust.solve(lev)
    assert sol is not None, "Generated level should be solvable"
    assert sol > 0, "Solution should require at least 1 move"


def test_generate_random_solvable_grid_sizes():
  """Verify generation works for all three grid sizes."""
  for gs in [6, 8, 10]:
    levels = mummymaze_rust.generate_random_solvable(5, grid_size=gs, seed=42)
    assert len(levels) == 5
    for lev in levels:
      d = lev.to_dict()
      assert d["grid_size"] == gs
      assert mummymaze_rust.solve(lev) is not None


def test_generate_random_solvable_deterministic():
  """Same seed should produce identical levels."""
  a = mummymaze_rust.generate_random_solvable(5, grid_size=6, seed=99)
  b = mummymaze_rust.generate_random_solvable(5, grid_size=6, seed=99)
  for la, lb in zip(a, b):
    assert la.fingerprint() == lb.fingerprint()


def test_generate_random_solvable_entity_probs():
  """With high entity probs, most levels should have extras."""
  levels = mummymaze_rust.generate_random_solvable(
    50,
    grid_size=8,
    seed=0,
    mummy2_prob=0.9,
    scorpion_prob=0.9,
    trap1_prob=0.9,
    gate_prob=0.9,
  )
  assert len(levels) == 50

  n_mummy2 = sum(1 for l in levels if l.to_dict()["mummy2"] is not None)
  n_scorpion = sum(1 for l in levels if l.to_dict()["scorpion"] is not None)
  n_gate = sum(1 for l in levels if l.to_dict()["gate"] is not None)
  n_traps = sum(1 for l in levels if len(l.to_dict()["traps"]) > 0)

  # With 90% probs, at least some should have each entity type
  assert n_mummy2 > 10, f"Expected many mummy2s, got {n_mummy2}/50"
  assert n_scorpion > 10, f"Expected many scorpions, got {n_scorpion}/50"
  assert n_gate > 5, f"Expected some gates, got {n_gate}/50"
  assert n_traps > 10, f"Expected many traps, got {n_traps}/50"


def test_generate_random_solvable_no_extras():
  """With zero entity probs, levels should only have player + mummy1."""
  levels = mummymaze_rust.generate_random_solvable(
    20,
    grid_size=6,
    seed=0,
    mummy2_prob=0.0,
    scorpion_prob=0.0,
    trap1_prob=0.0,
    gate_prob=0.0,
  )
  assert len(levels) == 20

  for lev in levels:
    d = lev.to_dict()
    assert d["mummy2"] is None
    assert d["scorpion"] is None
    assert d["gate"] is None
    assert len(d["traps"]) == 0
