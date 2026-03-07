"""Adversarial level generation via genetic algorithm.

Mutates existing Mummy Maze levels to find configurations that are solvable
(BFS finds a solution) but difficult (low win probability under uniform-random
policy, long BFS solutions, many states).

Usage:
  uv run python -m src.train.adversarial mazes/ \
    [--generations 50] [--pop-size 64] [--grid-size 6]
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import mummymaze_rust
from mummy_maze.parser import (
  Entity,
  EntityType,
  SubLevel,
  parse_file,
  render_maze,
)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class Individual:
  """A candidate level in the GA population."""

  sublevel: SubLevel
  grid_size: int
  is_red: bool
  origin: str  # e.g. "B-5:42" or "mut(B-5:42)"
  # Fitness (filled in by evaluate)
  bfs_moves: int | None = None
  n_states: int | None = None
  win_prob: float | None = None
  fitness: float | None = None


# ---------------------------------------------------------------------------
# SubLevel <-> Rust Level conversion
# ---------------------------------------------------------------------------


def _to_rust_level(ind: Individual) -> mummymaze_rust.Level:
  """Convert an Individual's SubLevel to a Rust Level for evaluation."""
  sub = ind.sublevel
  n = ind.grid_size

  h_flat = [sub.h_walls[r][c] for r in range(n + 1) for c in range(n)]
  v_flat = [sub.v_walls[r][c] for r in range(n) for c in range(n + 1)]

  player = (0, 0)
  mummy1 = (0, 0)
  mummy2 = None
  scorpion = None
  traps: list[tuple[int, int]] = []
  gate = None
  key = None
  mummy_count = 0

  for e in sub.entities:
    if e.type == EntityType.PLAYER:
      player = (e.row, e.col)
    elif e.type == EntityType.MUMMY:
      if mummy_count == 0:
        mummy1 = (e.row, e.col)
      else:
        mummy2 = (e.row, e.col)
      mummy_count += 1
    elif e.type == EntityType.SCORPION:
      scorpion = (e.row, e.col)
    elif e.type == EntityType.TRAP:
      traps.append((e.row, e.col))
    elif e.type == EntityType.GATE:
      gate = (e.row, e.col)
    elif e.type == EntityType.KEY:
      key = (e.row, e.col)

  return mummymaze_rust.Level.from_edges(
    grid_size=n,
    flip=ind.is_red,
    h_walls=h_flat,
    v_walls=v_flat,
    exit_side=sub.exit_side,
    exit_pos=sub.exit_pos,
    player=player,
    mummy1=mummy1,
    mummy2=mummy2,
    scorpion=scorpion,
    traps=traps,
    gate=gate,
    key=key,
  )


# ---------------------------------------------------------------------------
# Level loading
# ---------------------------------------------------------------------------


def load_seed_population(maze_dir: Path, grid_size: int) -> list[Individual]:
  """Load all levels of a given grid size as seed individuals."""
  seeds: list[Individual] = []
  for dat_path in sorted(maze_dir.glob("B-*.dat")):
    parsed = parse_file(dat_path)
    if parsed is None:
      continue
    if parsed.header.grid_size != grid_size:
      continue
    for i, sub in enumerate(parsed.sublevels):
      seeds.append(
        Individual(
          sublevel=sub,
          grid_size=grid_size,
          is_red=parsed.header.flip,
          origin=f"{dat_path.stem}:{i}",
        )
      )
  return seeds


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------


def _deep_copy_sublevel(sub: SubLevel) -> SubLevel:
  return SubLevel(
    h_walls=[row[:] for row in sub.h_walls],
    v_walls=[row[:] for row in sub.v_walls],
    exit_side=sub.exit_side,
    exit_pos=sub.exit_pos,
    entities=[Entity(e.type, e.col, e.row) for e in sub.entities],
    flip=sub.flip,
  )


def mutate_wall(ind: Individual) -> Individual:
  """Flip a random interior wall segment."""
  sub = _deep_copy_sublevel(ind.sublevel)
  n = ind.grid_size

  if random.random() < 0.5:
    # Horizontal wall: rows 1..n-1 (skip borders), cols 0..n-1
    r = random.randint(1, n - 1)
    c = random.randint(0, n - 1)
    sub.h_walls[r][c] = not sub.h_walls[r][c]
  else:
    # Vertical wall: rows 0..n-1, cols 1..n-1 (skip borders)
    r = random.randint(0, n - 1)
    c = random.randint(1, n - 1)
    sub.v_walls[r][c] = not sub.v_walls[r][c]

  return Individual(
    sublevel=sub,
    grid_size=ind.grid_size,
    is_red=ind.is_red,
    origin=f"mut({ind.origin})",
  )


def mutate_entity(ind: Individual) -> Individual:
  """Move a random non-player entity to a new position."""
  sub = _deep_copy_sublevel(ind.sublevel)
  n = ind.grid_size

  movable = [
    i
    for i, e in enumerate(sub.entities)
    if e.type in (EntityType.MUMMY, EntityType.SCORPION, EntityType.TRAP)
  ]
  if not movable:
    return Individual(sublevel=sub, grid_size=n, is_red=ind.is_red, origin=ind.origin)

  idx = random.choice(movable)
  ent = sub.entities[idx]

  occupied = {(e.row, e.col) for e in sub.entities}
  occupied.discard((ent.row, ent.col))

  for _ in range(20):
    nr, nc = random.randint(0, n - 1), random.randint(0, n - 1)
    if (nr, nc) not in occupied:
      sub.entities[idx] = Entity(ent.type, nc, nr)
      break

  return Individual(
    sublevel=sub,
    grid_size=n,
    is_red=ind.is_red,
    origin=f"mut({ind.origin})",
  )


def mutate_player(ind: Individual) -> Individual:
  """Move the player to a new position."""
  sub = _deep_copy_sublevel(ind.sublevel)
  n = ind.grid_size

  occupied = {(e.row, e.col) for e in sub.entities}
  player_idx = next(
    i for i, e in enumerate(sub.entities) if e.type == EntityType.PLAYER
  )
  old = sub.entities[player_idx]
  occupied.discard((old.row, old.col))

  for _ in range(20):
    nr, nc = random.randint(0, n - 1), random.randint(0, n - 1)
    if (nr, nc) not in occupied:
      sub.entities[player_idx] = Entity(EntityType.PLAYER, nc, nr)
      break

  return Individual(
    sublevel=sub,
    grid_size=n,
    is_red=ind.is_red,
    origin=f"mut({ind.origin})",
  )


def mutate(ind: Individual) -> Individual:
  """Apply a random mutation operator."""
  r = random.random()
  if r < 0.5:
    result = mutate_wall(ind)
  elif r < 0.8:
    result = mutate_entity(ind)
  else:
    result = mutate_player(ind)

  # Sometimes apply multiple mutations
  if random.random() < 0.3:
    result = mutate_wall(result)

  return result


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------


def crossover(a: Individual, b: Individual) -> Individual:
  """Combine walls from one parent with entities from another."""
  wall_parent = random.choice([a, b])
  entity_parent = b if wall_parent is a else a

  sub = SubLevel(
    h_walls=[row[:] for row in wall_parent.sublevel.h_walls],
    v_walls=[row[:] for row in wall_parent.sublevel.v_walls],
    exit_side=wall_parent.sublevel.exit_side,
    exit_pos=wall_parent.sublevel.exit_pos,
    entities=[Entity(e.type, e.col, e.row) for e in entity_parent.sublevel.entities],
    flip=wall_parent.sublevel.flip,
  )

  return Individual(
    sublevel=sub,
    grid_size=a.grid_size,
    is_red=a.is_red,
    origin=f"xo({a.origin},{b.origin})",
  )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(pop: list[Individual]) -> list[Individual]:
  """Evaluate fitness for all individuals. Filters out unsolvable levels."""
  evaluated: list[Individual] = []

  for ind in pop:
    lev = _to_rust_level(ind)
    moves = mummymaze_rust.solve(lev)
    if moves is None:
      continue

    analysis = mummymaze_rust.analyze(lev)
    ind.bfs_moves = moves
    ind.n_states = analysis["n_states"]
    ind.win_prob = analysis["win_prob"]

    # Fitness: prioritize low uniform-random win probability (harder),
    # break ties with longer BFS solutions
    ind.fitness = -ind.win_prob + moves / 1000.0
    evaluated.append(ind)

  return evaluated


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def tournament_select(pop: list[Individual], k: int = 3) -> Individual:
  """Tournament selection: pick k random individuals, return the fittest."""
  contestants = random.sample(pop, min(k, len(pop)))
  return max(contestants, key=lambda x: x.fitness or 0.0)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def display_best(ind: Individual, generation: int) -> None:
  """Print ASCII rendering of the best level with stats."""
  print(f"\n{'=' * 50}")
  print(f"Generation {generation} — Best level")
  print(f"Origin: {ind.origin}")
  print(f"Grid: {ind.grid_size}  Red: {ind.is_red}")
  print(f"BFS moves: {ind.bfs_moves}  States: {ind.n_states}  Win%: {ind.win_prob:.6f}")
  print(f"Fitness: {ind.fitness:.4f}")
  print(f"{'=' * 50}")
  print(render_maze(ind.sublevel, ind.grid_size))
  print()


# ---------------------------------------------------------------------------
# GA main loop
# ---------------------------------------------------------------------------


def run_ga(
  maze_dir: Path,
  *,
  grid_size: int = 6,
  pop_size: int = 64,
  generations: int = 50,
  elite_frac: float = 0.1,
  crossover_rate: float = 0.2,
  seed: int | None = None,
) -> Individual:
  if seed is not None:
    random.seed(seed)

  print(f"Loading seed levels (grid_size={grid_size})...")
  all_seeds = load_seed_population(maze_dir, grid_size)
  print(f"  {len(all_seeds)} levels loaded")

  print("Evaluating seed population...")
  evaluated_seeds = evaluate(all_seeds)
  print(f"  {len(evaluated_seeds)} solvable")

  if not evaluated_seeds:
    msg = f"No solvable levels found for grid_size={grid_size}"
    raise ValueError(msg)

  evaluated_seeds.sort(key=lambda x: x.fitness or 0.0, reverse=True)
  population = evaluated_seeds[:pop_size]

  best_ever = population[0]
  display_best(best_ever, 0)

  for gen in range(1, generations + 1):
    offspring: list[Individual] = []

    n_elite = max(1, int(pop_size * elite_frac))
    population.sort(key=lambda x: x.fitness or 0.0, reverse=True)
    offspring.extend(population[:n_elite])

    while len(offspring) < pop_size:
      if random.random() < crossover_rate and len(population) >= 2:
        p1 = tournament_select(population)
        p2 = tournament_select(population)
        child = crossover(p1, p2)
      else:
        parent = tournament_select(population)
        child = mutate(parent)

      offspring.append(child)

    to_eval = [ind for ind in offspring[n_elite:] if ind.fitness is None]
    evaluated_new = evaluate(to_eval)

    population = offspring[:n_elite] + evaluated_new

    if not population:
      print(f"  Gen {gen}: all offspring unsolvable, reseeding")
      population = [best_ever]
      continue

    population.sort(key=lambda x: x.fitness or 0.0, reverse=True)
    gen_best = population[0]

    fitnesses = [x.fitness for x in population if x.fitness is not None]
    solvable_rate = len(evaluated_new) / max(1, len(to_eval))
    print(
      f"  Gen {gen}: pop={len(population)} "
      f"best={gen_best.fitness:.4f} "
      f"avg={sum(fitnesses) / len(fitnesses):.4f} "
      f"solvable={solvable_rate:.0%}"
    )

    if gen_best.fitness is not None and (
      best_ever.fitness is None or gen_best.fitness > best_ever.fitness
    ):
      best_ever = gen_best
      display_best(best_ever, gen)

  print(f"\nFinal best: {best_ever.bfs_moves} BFS moves")
  display_best(best_ever, generations)
  return best_ever


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Adversarial level generation via genetic algorithm"
  )
  parser.add_argument("maze_dir", type=Path, help="directory containing B-*.dat files")
  parser.add_argument("--grid-size", type=int, default=6, choices=[6, 8, 10])
  parser.add_argument("--pop-size", type=int, default=64)
  parser.add_argument("--generations", type=int, default=50)
  parser.add_argument("--seed", type=int, default=None)
  args = parser.parse_args()

  run_ga(
    args.maze_dir.resolve(),
    grid_size=args.grid_size,
    pop_size=args.pop_size,
    generations=args.generations,
    seed=args.seed,
  )


if __name__ == "__main__":
  main()
