"""Python-orchestrated GA loop using Rust primitives + in-process JAX inference.

Replaces the monolithic `mummymaze_rust.run_ga_round()` which spawned a
ModelServer subprocess for policy evaluation. Here the model lives in the
Python process, eliminating subprocess overhead and re-JIT costs.

Rust handles the hot paths (BFS, Markov, mutation/crossover) via PyO3.
Python handles orchestration, selection, archive, and neural net inference.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax.numpy as jnp
import mummymaze_rust
import numpy as np
from scipy.special import softmax as scipy_softmax

from src.env.level_load import exit_cell
from src.env.types import LevelData
from src.train.wire import next_power_of_2, state_tuples_to_env_states

if TYPE_CHECKING:
  from collections.abc import Callable


# ---------------------------------------------------------------------------
# Level conversion: mummymaze_rust.Level → JAX LevelData
# ---------------------------------------------------------------------------


def level_to_level_data(level: mummymaze_rust.Level) -> tuple[int, LevelData]:
  """Convert a Rust Level to a JAX LevelData for observation building."""
  d = level.to_dict()
  gs = d["grid_size"]
  n = gs

  h_walls = np.array(d["h_walls"], dtype=np.bool_).reshape(n + 1, n)
  v_walls = np.array(d["v_walls"], dtype=np.bool_).reshape(n, n + 1)

  traps = d.get("traps", [])
  trap_pos = np.zeros((2, 2), dtype=np.int32)
  trap_active = np.zeros(2, dtype=np.bool_)
  for i, (tr, tc) in enumerate(traps[:2]):
    trap_pos[i] = [tr, tc]
    trap_active[i] = True

  gate = d.get("gate")
  key = d.get("key")
  has_gate = gate is not None

  ec = exit_cell(d["exit_side"], d["exit_pos"], gs)

  return gs, LevelData(
    h_walls_base=jnp.array(h_walls),
    v_walls_base=jnp.array(v_walls),
    is_red=jnp.bool_(d["flip"]),
    has_key_gate=jnp.bool_(has_gate),
    gate_row=jnp.int32(gate[0] if gate else 0),
    gate_col=jnp.int32(gate[1] if gate else 0),
    trap_pos=jnp.array(trap_pos),
    trap_active=jnp.array(trap_active),
    key_pos=jnp.array(key if key else [0, 0], dtype=jnp.int32),
    exit_cell=jnp.array(ec, dtype=jnp.int32),
    initial_player=jnp.zeros(2, dtype=jnp.int32),
    initial_mummy_pos=jnp.zeros((2, 2), dtype=jnp.int32),
    initial_mummy_alive=jnp.zeros(2, dtype=jnp.bool_),
    initial_scorpion_pos=jnp.zeros((1, 2), dtype=jnp.int32),
    initial_scorpion_alive=jnp.zeros(1, dtype=jnp.bool_),
  )


# ---------------------------------------------------------------------------
# MAP-Elites Archive
# ---------------------------------------------------------------------------


@dataclass
class Individual:
  level: mummymaze_rust.Level
  bfs_moves: int
  n_states: int
  win_prob: float
  log_win_prob: float
  log_policy_win_prob: float
  fitness: float


@dataclass
class MapElitesArchive:
  """2D grid indexed by (bfs_moves, n_states) bins."""

  bfs_range: tuple[int, int]
  states_range: tuple[int, int]
  bfs_bins: int
  states_bins: int
  target_log_wp: float
  grid: list[list[Individual | None]] = field(init=False)

  def __post_init__(self) -> None:
    self.grid = [[None] * self.states_bins for _ in range(self.bfs_bins)]

  def _bin_idx(self, value: int, range_: tuple[int, int], n_bins: int) -> int:
    if value <= range_[0]:
      return 0
    if value >= range_[1]:
      return n_bins - 1
    frac = (value - range_[0]) / (range_[1] - range_[0])
    return min(int(frac * n_bins), n_bins - 1)

  def try_insert(self, ind: Individual) -> bool:
    bi = self._bin_idx(ind.bfs_moves, self.bfs_range, self.bfs_bins)
    si = self._bin_idx(ind.n_states, self.states_range, self.states_bins)
    existing = self.grid[bi][si]
    if existing is None:
      self.grid[bi][si] = ind
      return True
    new_dist = abs(ind.log_policy_win_prob - self.target_log_wp)
    existing_dist = abs(existing.log_policy_win_prob - self.target_log_wp)
    if new_dist < existing_dist:
      self.grid[bi][si] = ind
      return True
    return False

  def occupancy(self) -> tuple[int, int]:
    total = self.bfs_bins * self.states_bins
    occupied = sum(1 for row in self.grid for cell in row if cell is not None)
    return occupied, total

  def all_individuals(self) -> list[Individual]:
    return [cell for row in self.grid for cell in row if cell is not None]

  def all_levels(self) -> list[mummymaze_rust.Level]:
    return [cell.level for row in self.grid for cell in row if cell is not None]


# ---------------------------------------------------------------------------
# Policy evaluation (in-process JAX inference)
# ---------------------------------------------------------------------------


def compute_policy_win_probs(
  levels: list[mummymaze_rust.Level],
  obs_and_forward: Callable,
  *,
  max_batch_size: int = 4096,
) -> list[float]:
  """Compute policy win probability for each level using in-process JAX model.

  Args:
    levels: Rust Level objects (must be solvable).
    obs_and_forward: JIT'd closure (grid_size, LevelData, EnvState) → logits.
      Implicitly captures a trained JAX model.
    max_batch_size: Max states per inference batch.

  Returns:
    List of log10(policy_win_prob) per level.
  """
  # Build graphs and state arrays for all levels
  graphs = [mummymaze_rust.build_graph(lev) for lev in levels]

  all_states_np = []
  all_probs = []
  offsets = [0]

  for lev, graph in zip(levels, graphs):
    states_tuples = graph["states"]
    n_states = len(states_tuples)

    if n_states == 0:
      offsets.append(offsets[-1])
      continue

    # Convert states to numpy
    states_np = np.array(states_tuples, dtype=np.int32)

    # Convert level to JAX LevelData
    gs, level_data = level_to_level_data(lev)

    # Run inference in chunks (power-of-2 padding for JIT cache)
    chunk_probs = []
    for chunk_start in range(0, n_states, max_batch_size):
      chunk_end = min(chunk_start + max_batch_size, n_states)
      chunk_n = chunk_end - chunk_start
      chunk_st = states_np[chunk_start:chunk_end]

      padded_size = next_power_of_2(chunk_n)
      if padded_size > chunk_n:
        padding = np.zeros((padded_size - chunk_n, 12), dtype=np.int32)
        padded_st = np.concatenate([chunk_st, padding], axis=0)
      else:
        padded_st = chunk_st

      env_states = state_tuples_to_env_states(padded_st)
      logits = obs_and_forward(gs, level_data, env_states)
      chunk_probs.append(np.array(logits[:chunk_n]))

    probs = scipy_softmax(
      np.concatenate(chunk_probs, axis=0) if len(chunk_probs) > 1 else chunk_probs[0],
      axis=-1,
    ).astype(np.float32)

    all_states_np.append(states_np)
    all_probs.append(probs)
    offsets.append(offsets[-1] + n_states)

  if not all_states_np:
    return []

  # Batch Markov solve under policy
  st_concat = np.concatenate(all_states_np)
  pr_concat = np.concatenate(all_probs)
  win_probs = mummymaze_rust.policy_win_prob_batch(
    levels,
    st_concat,
    pr_concat,
    offsets,
  )

  return [math.log10(wp) if wp > 0 else float("-inf") for wp in win_probs]


# ---------------------------------------------------------------------------
# GA generation result
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
  generation: int
  best: Individual
  avg_fitness: float
  solvable_rate: float
  pop_size: int


def _individual_from_eval(e: dict) -> Individual:
  """Build an Individual from a ga_evaluate_batch result dict."""
  return Individual(
    level=e["level"],
    bfs_moves=e["bfs_moves"],
    n_states=e["n_states"],
    win_prob=e["win_prob"],
    log_win_prob=e["log_win_prob"],
    log_policy_win_prob=float("-inf"),
    fitness=e["fitness"],
  )


def _recompute_fitness(
  ind: Individual,
  lwp: float,
  fitness_expr: str,
) -> None:
  """Update Individual's policy win prob and recompute fitness."""
  ind.log_policy_win_prob = lwp
  metrics = {
    "win_prob": ind.win_prob,
    "log_win_prob": ind.log_win_prob,
    "bfs_moves": ind.bfs_moves,
    "n_states": ind.n_states,
    "policy_win_prob": 10.0**lwp if lwp > float("-inf") else 0.0,
    "log_policy_win_prob": lwp,
  }
  ind.fitness = mummymaze_rust.eval_fitness(fitness_expr, metrics)


# ---------------------------------------------------------------------------
# GA loop
# ---------------------------------------------------------------------------


def run_ga(
  seed_levels: list[mummymaze_rust.Level],
  *,
  obs_and_forward: Callable,
  pop_size: int = 64,
  generations: int = 50,
  elite_frac: float = 0.1,
  crossover_rate: float = 0.2,
  crossover_mode: str = "swap_entities",
  seed: int = 42,
  fitness_expr: str = "-log_policy_win_prob",
  max_batch_size: int = 4096,
  w_wall: float = 5.0,
  w_move_entity: float = 3.0,
  w_move_player: float = 2.0,
  w_add_entity: float = 1.0,
  w_remove_entity: float = 1.0,
  w_move_exit: float = 1.0,
  extra_wall_prob: float = 0.3,
  archive: MapElitesArchive | None = None,
  on_generation: Callable[[GenerationResult], None] | None = None,
  on_status: Callable[[str], None] | None = None,
) -> tuple[list[Individual], MapElitesArchive | None]:
  """Run a GA round with in-process policy evaluation.

  Args:
    seed_levels: Initial population of levels.
    obs_and_forward: JIT'd closure (grid_size, LevelData, EnvState) → logits.
      Implicitly captures a trained JAX model.
    archive: Optional MAP-Elites archive to populate.
    on_generation: Callback for per-generation results.
    on_status: Callback for status messages.

  Returns:
    (final_population, archive)
  """
  rng = random.Random(seed)
  mutation_kwargs = dict(
    w_wall=w_wall,
    w_move_entity=w_move_entity,
    w_move_player=w_move_player,
    w_add_entity=w_add_entity,
    w_remove_entity=w_remove_entity,
    w_move_exit=w_move_exit,
    extra_wall_prob=extra_wall_prob,
  )

  needs_policy = (
    "policy_win_prob" in fitness_expr or "log_policy_win_prob" in fitness_expr
  )

  def _status(msg: str) -> None:
    if on_status:
      on_status(msg)

  # --- Evaluate seed levels ---
  _status(f"Evaluating {len(seed_levels)} seed levels...")
  evals = mummymaze_rust.ga_evaluate_batch(seed_levels, fitness_expr=fitness_expr)

  # Build population from solvable seeds
  population: list[Individual] = []
  eval_levels: list[mummymaze_rust.Level] = []
  for e in evals:
    ind = _individual_from_eval(e)
    population.append(ind)
    eval_levels.append(e["level"])

  _status(f"Seeds: {len(population)}/{len(seed_levels)} solvable")

  if not population:
    return [], archive

  # Policy evaluation for seeds
  if needs_policy and eval_levels:
    _status(f"Policy inference: {len(eval_levels)} levels...")
    log_pwps = compute_policy_win_probs(
      eval_levels,
      obs_and_forward,
      max_batch_size=max_batch_size,
    )
    for ind, lwp in zip(population, log_pwps):
      _recompute_fitness(ind, lwp, fitness_expr)

  # Insert seeds into archive
  if archive:
    for ind in population:
      archive.try_insert(ind)

  # Sort and truncate
  population.sort(key=lambda x: x.fitness, reverse=True)
  population = population[:pop_size]

  best_ever = population[0]

  # Report generation 0
  avg_fitness = sum(i.fitness for i in population) / len(population)
  gen_result = GenerationResult(
    generation=0,
    best=best_ever,
    avg_fitness=avg_fitness,
    solvable_rate=1.0,
    pop_size=len(population),
  )
  if on_generation:
    on_generation(gen_result)

  # --- Generation loop ---
  for generation in range(1, generations + 1):
    n_elite = max(1, int(pop_size * elite_frac + 0.5))
    population.sort(key=lambda x: x.fitness, reverse=True)

    # Generate offspring levels
    offspring_levels: list[mummymaze_rust.Level] = []
    # Elites pass through
    for i in range(min(n_elite, len(population))):
      offspring_levels.append(population[i].level)

    # Fill rest with mutation/crossover
    while len(offspring_levels) < pop_size:
      if rng.random() < crossover_rate and len(population) >= 2:
        p1 = _tournament_select(population, rng, k=3)
        p2 = _tournament_select(population, rng, k=3)
        child = mummymaze_rust.ga_crossover(
          p1.level,
          p2.level,
          mode=crossover_mode,
          seed=rng.randint(0, 2**63),
        )
      else:
        parent = _tournament_select(population, rng, k=3)
        child = mummymaze_rust.mutate(
          parent.level,
          seed=rng.randint(0, 2**63),
          **mutation_kwargs,
        )
      offspring_levels.append(child)

    # Evaluate non-elite offspring
    non_elite = offspring_levels[n_elite:]
    n_evaluated = len(non_elite)
    evals = mummymaze_rust.ga_evaluate_batch(non_elite, fitness_expr=fitness_expr)

    new_individuals: list[Individual] = []
    new_levels: list[mummymaze_rust.Level] = []
    for e in evals:
      new_individuals.append(_individual_from_eval(e))
      new_levels.append(e["level"])

    # Policy evaluation for new offspring
    if needs_policy and new_levels:
      log_pwps = compute_policy_win_probs(
        new_levels,
        obs_and_forward,
        max_batch_size=max_batch_size,
      )
      for ind, lwp in zip(new_individuals, log_pwps):
        _recompute_fitness(ind, lwp, fitness_expr)

    # Merge elites + new offspring
    new_pop = population[: min(n_elite, len(population))] + new_individuals
    if not new_pop:
      new_pop = [best_ever]

    # Insert new offspring into archive
    if archive:
      for ind in new_individuals:
        archive.try_insert(ind)

    population = new_pop
    population.sort(key=lambda x: x.fitness, reverse=True)

    if population[0].fitness > best_ever.fitness:
      best_ever = population[0]

    avg = sum(i.fitness for i in population) / len(population)
    solvable_rate = len(evals) / n_evaluated if n_evaluated > 0 else 1.0

    gen_result = GenerationResult(
      generation=generation,
      best=best_ever,
      avg_fitness=avg,
      solvable_rate=solvable_rate,
      pop_size=len(population),
    )
    if on_generation:
      on_generation(gen_result)

  return population, archive


def _tournament_select(
  population: list[Individual],
  rng: random.Random,
  k: int = 3,
) -> Individual:
  """Pick k random individuals, return the fittest."""
  candidates = rng.sample(population, min(k, len(population)))
  return max(candidates, key=lambda x: x.fitness)
