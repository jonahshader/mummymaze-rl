//! Genetic algorithm for adversarial level generation.
//!
//! Mutates existing levels to find configurations that are solvable (BFS finds a
//! solution) but difficult (low win probability under uniform-random policy).

use crate::graph::build_graph;
use crate::markov::MarkovChain;
use crate::parse::{Level, WALL_E, WALL_N, WALL_S, WALL_W};
use crate::solver;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Individual {
    pub level: Level,
    pub bfs_moves: u32,
    pub n_states: usize,
    pub win_prob: f64,
    pub fitness: f64,
}

#[derive(Debug, Clone)]
pub struct GaConfig {
    pub grid_size: i32,
    pub pop_size: usize,
    pub generations: usize,
    pub elite_frac: f64,
    pub crossover_rate: f64,
    pub seed: u64,
}

impl Default for GaConfig {
    fn default() -> Self {
        GaConfig {
            grid_size: 6,
            pop_size: 64,
            generations: 50,
            elite_frac: 0.1,
            crossover_rate: 0.2,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub generation: usize,
    pub best: Individual,
    pub avg_fitness: f64,
    pub solvable_rate: f64,
    pub pop_size: usize,
}

#[derive(Debug, Clone)]
pub enum GaMessage {
    SeedsDone {
        n_seeds: usize,
        n_solvable: usize,
    },
    Generation(GenerationResult),
    Done,
    Error(String),
}

/// Collect all occupied entity positions from a level.
fn occupied_cells(level: &Level, include_player: bool) -> std::collections::HashSet<(i32, i32)> {
    let mut occupied = std::collections::HashSet::new();
    if include_player {
        occupied.insert((level.player_row, level.player_col));
    }
    occupied.insert((level.mummy1_row, level.mummy1_col));
    if level.has_mummy2 {
        occupied.insert((level.mummy2_row, level.mummy2_col));
    }
    if level.has_scorpion {
        occupied.insert((level.scorpion_row, level.scorpion_col));
    }
    if level.trap_count >= 1 {
        occupied.insert((level.trap1_row, level.trap1_col));
    }
    if level.trap_count >= 2 {
        occupied.insert((level.trap2_row, level.trap2_col));
    }
    occupied
}

/// Which entity field to mutate.
#[derive(Clone, Copy)]
enum EntityKind {
    Mummy1,
    Mummy2,
    Scorpion,
    Trap1,
    Trap2,
}

/// Toggle a random interior wall bit on both adjacent cells.
pub fn mutate_wall(level: &Level, rng: &mut impl Rng) -> Level {
    let mut out = level.clone();
    let n = level.grid_size;

    if rng.random_bool(0.5) {
        // Horizontal wall between row r-1 and row r (N/S pair)
        let r = rng.random_range(1..n);
        let c = rng.random_range(0..n);
        let upper = (c + (r - 1) * 10) as usize;
        let lower = (c + r * 10) as usize;
        out.walls[upper] ^= WALL_S;
        out.walls[lower] ^= WALL_N;
    } else {
        // Vertical wall between col c-1 and col c (E/W pair)
        let r = rng.random_range(0..n);
        let c = rng.random_range(1..n);
        let left = ((c - 1) + r * 10) as usize;
        let right = (c + r * 10) as usize;
        out.walls[left] ^= WALL_E;
        out.walls[right] ^= WALL_W;
    }

    out
}

/// Move a random mummy/scorpion/trap to an unoccupied cell.
pub fn mutate_entity(level: &Level, rng: &mut impl Rng) -> Level {
    let mut out = level.clone();
    let n = level.grid_size;

    // Collect movable entities
    let mut kinds: Vec<EntityKind> = vec![EntityKind::Mummy1];
    if out.has_mummy2 {
        kinds.push(EntityKind::Mummy2);
    }
    if out.has_scorpion {
        kinds.push(EntityKind::Scorpion);
    }
    if out.trap_count >= 1 {
        kinds.push(EntityKind::Trap1);
    }
    if out.trap_count >= 2 {
        kinds.push(EntityKind::Trap2);
    }

    let kind = kinds[rng.random_range(0..kinds.len())];

    let mut occupied = occupied_cells(&out, true);
    // Remove the entity we're moving so it can land anywhere else
    let old_pos = match kind {
        EntityKind::Mummy1 => (out.mummy1_row, out.mummy1_col),
        EntityKind::Mummy2 => (out.mummy2_row, out.mummy2_col),
        EntityKind::Scorpion => (out.scorpion_row, out.scorpion_col),
        EntityKind::Trap1 => (out.trap1_row, out.trap1_col),
        EntityKind::Trap2 => (out.trap2_row, out.trap2_col),
    };
    occupied.remove(&old_pos);

    for _ in 0..20 {
        let nr = rng.random_range(0..n);
        let nc = rng.random_range(0..n);
        if !occupied.contains(&(nr, nc)) {
            match kind {
                EntityKind::Mummy1 => {
                    out.mummy1_row = nr;
                    out.mummy1_col = nc;
                }
                EntityKind::Mummy2 => {
                    out.mummy2_row = nr;
                    out.mummy2_col = nc;
                }
                EntityKind::Scorpion => {
                    out.scorpion_row = nr;
                    out.scorpion_col = nc;
                }
                EntityKind::Trap1 => {
                    out.trap1_row = nr;
                    out.trap1_col = nc;
                }
                EntityKind::Trap2 => {
                    out.trap2_row = nr;
                    out.trap2_col = nc;
                }
            }
            break;
        }
    }

    out
}

/// Move the player to an unoccupied cell.
pub fn mutate_player(level: &Level, rng: &mut impl Rng) -> Level {
    let mut out = level.clone();
    let n = level.grid_size;
    let occupied = occupied_cells(&out, false);

    for _ in 0..20 {
        let nr = rng.random_range(0..n);
        let nc = rng.random_range(0..n);
        if !occupied.contains(&(nr, nc)) {
            out.player_row = nr;
            out.player_col = nc;
            break;
        }
    }

    out
}

/// Apply a random mutation operator.
pub fn mutate(level: &Level, rng: &mut impl Rng) -> Level {
    let r: f64 = rng.random();
    let mut result = if r < 0.5 {
        mutate_wall(level, rng)
    } else if r < 0.8 {
        mutate_entity(level, rng)
    } else {
        mutate_player(level, rng)
    };

    // Sometimes apply an extra wall mutation
    if rng.random_bool(0.3) {
        result = mutate_wall(&result, rng);
    }

    result
}

/// Crossover: walls from one parent, entities from another.
pub fn crossover(a: &Level, b: &Level, rng: &mut impl Rng) -> Level {
    let (wall_parent, entity_parent) = if rng.random_bool(0.5) {
        (a, b)
    } else {
        (b, a)
    };

    let mut out = wall_parent.clone();
    out.player_row = entity_parent.player_row;
    out.player_col = entity_parent.player_col;
    out.mummy1_row = entity_parent.mummy1_row;
    out.mummy1_col = entity_parent.mummy1_col;
    out.mummy2_row = entity_parent.mummy2_row;
    out.mummy2_col = entity_parent.mummy2_col;
    out.scorpion_row = entity_parent.scorpion_row;
    out.scorpion_col = entity_parent.scorpion_col;
    out.trap1_row = entity_parent.trap1_row;
    out.trap1_col = entity_parent.trap1_col;
    out.trap2_row = entity_parent.trap2_row;
    out.trap2_col = entity_parent.trap2_col;

    out
}

/// Evaluate a level: solve + Markov analysis. Returns None if unsolvable.
pub fn evaluate(level: &Level) -> Option<Individual> {
    let result = solver::solve(level);
    let moves = result.moves?;

    let graph = build_graph(level);
    let chain = MarkovChain::from_graph(&graph);
    let win_prob = chain
        .solve_win_probs()
        .ok()
        .map(|wp| wp[chain.start_idx])
        .unwrap_or(0.0);
    let n_states = graph.n_transient;

    // Fitness: prioritize low win probability (harder), break ties with longer BFS
    let fitness = -win_prob + moves as f64 / 1000.0;

    Some(Individual {
        level: level.clone(),
        bfs_moves: moves,
        n_states,
        win_prob,
        fitness,
    })
}

/// Tournament selection: pick k random individuals, return the fittest.
fn tournament_select<'a>(pop: &'a [Individual], rng: &mut impl Rng, k: usize) -> &'a Individual {
    let mut best: Option<&Individual> = None;
    for _ in 0..k.min(pop.len()) {
        let idx = rng.random_range(0..pop.len());
        let candidate = &pop[idx];
        if best.is_none() || candidate.fitness > best.unwrap().fitness {
            best = Some(candidate);
        }
    }
    best.unwrap()
}

/// Run the GA. Sends progress via `tx`. Checks `stop_flag` each generation.
pub fn run_ga(
    config: &GaConfig,
    seed_levels: Vec<Level>,
    tx: Sender<GaMessage>,
    stop_flag: Arc<AtomicBool>,
) {
    let mut rng = StdRng::seed_from_u64(config.seed);

    // Evaluate seed levels
    let mut population: Vec<Individual> = seed_levels
        .iter()
        .filter_map(|lev| evaluate(lev))
        .collect();

    let n_solvable = population.len();
    if tx
        .send(GaMessage::SeedsDone {
            n_seeds: seed_levels.len(),
            n_solvable,
        })
        .is_err()
    {
        return;
    }

    if population.is_empty() {
        let _ = tx.send(GaMessage::Error(
            "No solvable seed levels found".to_string(),
        ));
        return;
    }

    // Sort by fitness descending, take top pop_size
    population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
    population.truncate(config.pop_size);

    let mut best_ever = population[0].clone();

    // Send generation 0 result
    let avg_fitness =
        population.iter().map(|i| i.fitness).sum::<f64>() / population.len() as f64;
    let _ = tx.send(GaMessage::Generation(GenerationResult {
        generation: 0,
        best: best_ever.clone(),
        avg_fitness,
        solvable_rate: 1.0,
        pop_size: population.len(),
    }));

    for generation in 1..=config.generations {
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }

        let n_elite = (config.pop_size as f64 * config.elite_frac).ceil() as usize;
        let n_elite = n_elite.max(1);
        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        let mut offspring_levels: Vec<Level> = Vec::with_capacity(config.pop_size);
        // Elites pass through
        for i in 0..n_elite.min(population.len()) {
            offspring_levels.push(population[i].level.clone());
        }

        // Generate new offspring
        while offspring_levels.len() < config.pop_size {
            let child_level = if rng.random_bool(config.crossover_rate) && population.len() >= 2 {
                let p1 = tournament_select(&population, &mut rng, 3);
                let p2 = tournament_select(&population, &mut rng, 3);
                crossover(&p1.level, &p2.level, &mut rng)
            } else {
                let parent = tournament_select(&population, &mut rng, 3);
                mutate(&parent.level, &mut rng)
            };
            offspring_levels.push(child_level);
        }

        // Evaluate non-elite offspring in parallel
        let non_elite = &offspring_levels[n_elite..];
        let n_evaluated = non_elite.len();
        let evaluated: Vec<Individual> = non_elite
            .par_iter()
            .filter_map(|level| evaluate(level))
            .collect();
        let n_solvable_gen = evaluated.len();

        let mut new_pop: Vec<Individual> = Vec::with_capacity(config.pop_size);
        // Keep elites
        for i in 0..n_elite.min(population.len()) {
            new_pop.push(population[i].clone());
        }
        new_pop.extend(evaluated);

        if new_pop.is_empty() {
            // All offspring unsolvable, reseed with best ever
            new_pop.push(best_ever.clone());
        }

        population = new_pop;
        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        let gen_best = &population[0];
        if gen_best.fitness > best_ever.fitness {
            best_ever = gen_best.clone();
        }

        let avg = population.iter().map(|i| i.fitness).sum::<f64>() / population.len() as f64;
        let solvable_rate = if n_evaluated > 0 {
            n_solvable_gen as f64 / n_evaluated as f64
        } else {
            1.0
        };

        if tx
            .send(GaMessage::Generation(GenerationResult {
                generation,
                best: best_ever.clone(),
                avg_fitness: avg,
                solvable_rate,
                pop_size: population.len(),
            }))
            .is_err()
        {
            return;
        }
    }

    let _ = tx.send(GaMessage::Done);
}
