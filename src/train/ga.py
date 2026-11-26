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
import threading
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

  from src.env.level_bank import LevelBank


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
  """Compute policy win probability for each level (sequential per-level).

  For batch inference across levels, use compute_policy_win_probs_batched.
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


def compute_policy_win_probs_batched(
  levels: list[mummymaze_rust.Level],
  model: object,
  grid_size: int,
  *,
  batch_size: int = 4096,
) -> list[float]:
  """Compute policy win probability with batched inference across all levels.

  Builds a temporary LevelBank, concatenates all state tuples, then runs
  obs + NN forward in fixed-size chunks (no per-level calls). Rust Markov
  solve runs in parallel across levels.
  """
  import jax

  from src.train.dataset import make_batch_obs

  if not levels:
    return []

  # Build all graphs and collect state arrays
  graphs = [mummymaze_rust.build_graph(lev) for lev in levels]

  all_states: list[np.ndarray] = []
  all_level_idx: list[np.ndarray] = []
  offsets = [0]

  for li, graph in enumerate(graphs):
    state_tuples = graph["states"]
    n = len(state_tuples)
    if n == 0:
      offsets.append(offsets[-1])
      continue
    states_np = np.array([[int(x) for x in s] for s in state_tuples], dtype=np.int32)
    all_states.append(states_np)
    all_level_idx.append(np.full(n, li, dtype=np.int32))
    offsets.append(offsets[-1] + n)

  if not all_states:
    return []

  flat_states = np.concatenate(all_states, axis=0)
  flat_level_idx = jnp.array(np.concatenate(all_level_idx, axis=0))
  flat_tuples = jnp.array(flat_states)
  total = flat_states.shape[0]

  # Build temporary LevelBank from GA levels
  bank = _build_temp_bank(levels, grid_size)

  # Obs + NN forward in chunks (obs computed on-the-fly, not materialized)
  jit_obs_fwd = jax.jit(lambda t, li: make_batch_obs(grid_size, bank, t, li))
  logits_parts: list[np.ndarray] = []
  for start in range(0, total, batch_size):
    end = min(start + batch_size, total)
    obs = jit_obs_fwd(flat_tuples[start:end], flat_level_idx[start:end])
    logits_parts.append(np.array(jax.vmap(model)(obs)))

  flat_logits = np.concatenate(logits_parts, axis=0)
  flat_probs = scipy_softmax(flat_logits, axis=-1).astype(np.float32)

  # Batch Markov solve (rayon-parallel in Rust)
  win_probs = mummymaze_rust.policy_win_prob_batch(
    levels, flat_states, flat_probs, offsets
  )

  return [math.log10(wp) if wp > 0 else float("-inf") for wp in win_probs]


def _build_temp_bank(
  levels: list[mummymaze_rust.Level],
  grid_size: int,
) -> LevelBank:
  """Build a temporary LevelBank from Rust levels (for obs computation)."""
  from src.env.level_bank import LevelBank
  from src.train.augment import _level_to_bank_arrays

  arrays = [_level_to_bank_arrays(lev) for lev in levels]
  n = len(arrays)

  fields_nd = [
    "h_walls_base",
    "v_walls_base",
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
  fields_1d = ["is_red", "has_key_gate", "gate_row", "gate_col"]

  stacked = {}
  for f in fields_nd:
    stacked[f] = jnp.array(np.stack([a[f] for a in arrays]))
  for f in fields_1d:
    stacked[f] = jnp.array(np.array([a[f] for a in arrays]))

  return LevelBank(
    grid_size=grid_size,
    n_levels=n,
    **stacked,
    train_indices=jnp.arange(n, dtype=jnp.int32),
    val_indices=jnp.zeros(0, dtype=jnp.int32),
  )


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


def _individual_from_eval(e: mummymaze_rust.GaEval) -> Individual:
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
  obs_and_forward: Callable | None = None,
  model: object | None = None,
  grid_size: int = 6,
  pop_size: int = 64,
  generations: int = 50,
  min_fitness: float | None = None,
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
  stop_flag: threading.Event | None = None,
) -> tuple[list[Individual], MapElitesArchive | None]:
  """Run a GA round with in-process policy evaluation.

  Args:
    seed_levels: Initial population of levels.
    obs_and_forward: JIT'd closure for per-level inference (legacy).
    model: Equinox model for batched inference (preferred over obs_and_forward).
    grid_size: Grid size for batched inference (only used with model).
    min_fitness: Stop early when best fitness >= this value.
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

  def _eval_policy(lvls: list[mummymaze_rust.Level]) -> list[float]:
    """Policy win prob evaluation — batched if model provided."""
    if model is not None:
      return compute_policy_win_probs_batched(
        lvls, model, grid_size, batch_size=max_batch_size
      )
    assert obs_and_forward is not None
    return compute_policy_win_probs(
      lvls, obs_and_forward, max_batch_size=max_batch_size
    )

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
    log_pwps = _eval_policy(eval_levels)
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
    if stop_flag is not None and stop_flag.is_set():
      break
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
      log_pwps = _eval_policy(new_levels)
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

    # Early stopping: best individual meets fitness threshold
    if min_fitness is not None and best_ever.fitness >= min_fitness:
      break

  return population, archive


def _tournament_select(
  population: list[Individual],
  rng: random.Random,
  k: int = 3,
) -> Individual:
  """Pick k random individuals, return the fittest."""
  candidates = rng.sample(population, min(k, len(population)))
  return max(candidates, key=lambda x: x.fitness)


# ---------------------------------------------------------------------------
# Multi-GA with batched inference
# ---------------------------------------------------------------------------


@dataclass
class MultiGAGenerationResult:
  generation: int
  best_per_run: list[Individual]
  best_overall: Individual
  avg_fitness: float
  solvable_rate: float
  total_pop: int


@dataclass
class _RunState:
  population: list[Individual]
  rng: random.Random
  best_ever: Individual | None


def run_multi_ga(
  n_runs: int,
  seed_levels: list[mummymaze_rust.Level],
  *,
  model: object,
  grid_size: int,
  pop_size: int = 16,
  generations: int = 50,
  fitness_expr: str = "-log_policy_win_prob",
  seed: int = 42,
  batch_size: int = 4096,
  elite_frac: float = 0.1,
  crossover_rate: float = 0.2,
  crossover_mode: str = "swap_entities",
  w_wall: float = 5.0,
  w_move_entity: float = 3.0,
  w_move_player: float = 2.0,
  w_add_entity: float = 1.0,
  w_remove_entity: float = 1.0,
  w_move_exit: float = 1.0,
  extra_wall_prob: float = 0.3,
  on_generation: Callable[[MultiGAGenerationResult], None] | None = None,
  stop_flag: threading.Event | None = None,
) -> list[Individual]:
  """Run N independent GAs with batched inference across all populations.

  Each generation, all populations produce offspring independently, then
  all offspring are evaluated in a single batched BFS + GPU inference pass.
  Returns the best individual from each run for maximum diversity.
  """
  import bisect

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

  # --- Distribute seeds across runs (round-robin) ---
  per_run_seeds: list[list[mummymaze_rust.Level]] = [[] for _ in range(n_runs)]
  for i, lev in enumerate(seed_levels):
    per_run_seeds[i % n_runs].append(lev)

  # Pad each run to pop_size by mutating its own seeds
  for run_idx, seeds in enumerate(per_run_seeds):
    if not seeds:
      continue
    run_rng = random.Random(seed + run_idx)
    while len(seeds) < pop_size:
      parent = seeds[run_rng.randint(0, len(seeds) - 1)]
      seeds.append(
        mummymaze_rust.mutate(parent, seed=run_rng.randint(0, 2**63), **mutation_kwargs)
      )

  # --- Initial evaluation (batched across all runs) ---
  all_initial: list[mummymaze_rust.Level] = []
  init_offsets = [0]
  for seeds in per_run_seeds:
    all_initial.extend(seeds)
    init_offsets.append(len(all_initial))

  evals = mummymaze_rust.ga_evaluate_batch(all_initial, fitness_expr=fitness_expr)

  # Scatter eval results to per-run populations
  runs: list[_RunState] = [
    _RunState(population=[], rng=random.Random(seed + i), best_ever=None)
    for i in range(n_runs)
  ]
  for e in evals:
    run_idx = bisect.bisect_right(init_offsets, e["level_idx"]) - 1
    runs[run_idx].population.append(_individual_from_eval(e))

  # Batched policy inference for all initial solvable levels
  if needs_policy:
    all_init_levels: list[mummymaze_rust.Level] = []
    init_map: list[tuple[int, int]] = []
    for ri, run in enumerate(runs):
      for li, ind in enumerate(run.population):
        all_init_levels.append(ind.level)
        init_map.append((ri, li))

    if all_init_levels:
      log_pwps = compute_policy_win_probs_batched(
        all_init_levels, model, grid_size, batch_size=batch_size
      )
      for (ri, li), lwp in zip(init_map, log_pwps):
        _recompute_fitness(runs[ri].population[li], lwp, fitness_expr)

  # Sort, truncate, set best_ever
  for run in runs:
    run.population.sort(key=lambda x: x.fitness, reverse=True)
    run.population = run.population[:pop_size]
    if run.population:
      run.best_ever = run.population[0]

  # Report generation 0
  _report_multi_gen(0, runs, 1.0, on_generation)

  # --- Generation loop ---
  for generation in range(1, generations + 1):
    if stop_flag is not None and stop_flag.is_set():
      break

    n_elite = max(1, int(pop_size * elite_frac + 0.5))

    # Phase 1: Each run produces offspring independently (CPU only)
    all_offspring: list[mummymaze_rust.Level] = []
    offspring_offsets = [0]
    per_run_n_elite: list[int] = []

    for run in runs:
      if not run.population:
        per_run_n_elite.append(0)
        offspring_offsets.append(len(all_offspring))
        continue

      run.population.sort(key=lambda x: x.fitness, reverse=True)
      actual_elite = min(n_elite, len(run.population))
      per_run_n_elite.append(actual_elite)

      n_needed = pop_size - actual_elite
      for _ in range(n_needed):
        if run.rng.random() < crossover_rate and len(run.population) >= 2:
          p1 = _tournament_select(run.population, run.rng, k=3)
          p2 = _tournament_select(run.population, run.rng, k=3)
          child = mummymaze_rust.ga_crossover(
            p1.level,
            p2.level,
            mode=crossover_mode,
            seed=run.rng.randint(0, 2**63),
          )
        else:
          parent = _tournament_select(run.population, run.rng, k=3)
          child = mummymaze_rust.mutate(
            parent.level,
            seed=run.rng.randint(0, 2**63),
            **mutation_kwargs,
          )
        all_offspring.append(child)

      offspring_offsets.append(len(all_offspring))

    # Phase 2: Batched BFS + Markov eval (one Rust call)
    total_offspring = len(all_offspring)
    evals = mummymaze_rust.ga_evaluate_batch(all_offspring, fitness_expr=fitness_expr)

    # Phase 3: Scatter eval results back to per-run groups
    per_run_new: list[list[Individual]] = [[] for _ in range(n_runs)]
    all_solvable: list[mummymaze_rust.Level] = []
    solvable_map: list[tuple[int, int]] = []

    for e in evals:
      run_idx = bisect.bisect_right(offspring_offsets, e["level_idx"]) - 1
      ind = _individual_from_eval(e)
      li = len(per_run_new[run_idx])
      per_run_new[run_idx].append(ind)
      all_solvable.append(e["level"])
      solvable_map.append((run_idx, li))

    # Phase 4: Batched GPU inference (one call, chunked internally)
    if needs_policy and all_solvable:
      log_pwps = compute_policy_win_probs_batched(
        all_solvable, model, grid_size, batch_size=batch_size
      )
      for (ri, li), lwp in zip(solvable_map, log_pwps):
        _recompute_fitness(per_run_new[ri][li], lwp, fitness_expr)

    # Phase 5: Per-run merge and selection
    total_solvable = len(all_solvable)
    for ri, run in enumerate(runs):
      actual_elite = per_run_n_elite[ri]
      elites = run.population[:actual_elite]
      new_pop = elites + per_run_new[ri]

      if not new_pop and run.best_ever is not None:
        new_pop = [run.best_ever]

      new_pop.sort(key=lambda x: x.fitness, reverse=True)
      run.population = new_pop

      if run.population and (
        run.best_ever is None or run.population[0].fitness > run.best_ever.fitness
      ):
        run.best_ever = run.population[0]

    solvable_rate = total_solvable / total_offspring if total_offspring > 0 else 1.0
    _report_multi_gen(generation, runs, solvable_rate, on_generation)

  return [run.best_ever for run in runs if run.best_ever is not None]


def _report_multi_gen(
  generation: int,
  runs: list[_RunState],
  solvable_rate: float,
  on_generation: Callable[[MultiGAGenerationResult], None] | None,
) -> None:
  """Build and emit a MultiGAGenerationResult."""
  bests = [run.best_ever for run in runs if run.best_ever is not None]
  if not bests:
    return
  best_overall = max(bests, key=lambda x: x.fitness)
  all_fitnesses = [ind.fitness for run in runs for ind in run.population]
  avg = sum(all_fitnesses) / len(all_fitnesses) if all_fitnesses else 0.0
  total_pop = sum(len(run.population) for run in runs)

  result = MultiGAGenerationResult(
    generation=generation,
    best_per_run=bests,
    best_overall=best_overall,
    avg_fitness=avg,
    solvable_rate=solvable_rate,
    total_pop=total_pop,
  )
  if on_generation:
    on_generation(result)
