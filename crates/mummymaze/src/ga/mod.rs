//! Genetic algorithm for adversarial level generation.
//!
//! Mutates existing levels to find configurations that are solvable (BFS finds a
//! solution) but difficult (low win probability under uniform-random policy).

pub mod crossover;
pub mod fitness;
pub mod mutation;

use crate::graph::build_graph;
use crate::markov::MarkovChain;
use crate::parse::Level;
use crate::solver;
use fitness::FitnessExpr;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;

pub use crossover::crossover;
pub use mutation::mutate_with_config;

#[derive(Debug, Clone)]
pub struct Individual {
    pub level: Level,
    pub bfs_moves: u32,
    pub n_states: usize,
    pub win_prob: f64,
    pub fitness: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossoverMode {
    /// Walls from one parent, entity positions from the other.
    SwapEntities,
    /// Split grid at a random row or column; take everything from each parent in their half.
    Region,
    /// Copy a random rectangular patch of walls from one parent onto the other.
    WallPatch,
    /// Take entity composition (types + flip) from one parent, re-place them randomly
    /// in the other parent's wall layout.
    FeatureLevel,
}

impl CrossoverMode {
    pub const ALL: [CrossoverMode; 4] = [
        CrossoverMode::SwapEntities,
        CrossoverMode::Region,
        CrossoverMode::WallPatch,
        CrossoverMode::FeatureLevel,
    ];

    pub fn label(self) -> &'static str {
        match self {
            CrossoverMode::SwapEntities => "Swap entities",
            CrossoverMode::Region => "Region split",
            CrossoverMode::WallPatch => "Wall patch",
            CrossoverMode::FeatureLevel => "Feature-level",
        }
    }
}

#[derive(Debug, Clone)]
pub struct GaConfig {
    pub grid_size: i32,
    pub pop_size: usize,
    pub generations: usize,
    pub elite_frac: f64,
    pub crossover_rate: f64,
    pub crossover_mode: CrossoverMode,
    pub seed: u64,
    /// Relative mutation weights (normalized at runtime).
    pub w_wall: f64,
    pub w_move_entity: f64,
    pub w_move_player: f64,
    pub w_add_entity: f64,
    pub w_remove_entity: f64,
    pub w_move_exit: f64,
    /// Probability of an extra wall mutation after the primary mutation.
    pub extra_wall_prob: f64,
    /// Fitness expression (math formula over metric variables).
    pub fitness_expr: String,
}

impl Default for GaConfig {
    fn default() -> Self {
        GaConfig {
            grid_size: 6,
            pop_size: 64,
            generations: 50,
            elite_frac: 0.1,
            crossover_rate: 0.2,
            crossover_mode: CrossoverMode::SwapEntities,
            seed: 42,
            w_wall: 5.0,
            w_move_entity: 3.0,
            w_move_player: 2.0,
            w_add_entity: 1.0,
            w_remove_entity: 1.0,
            w_move_exit: 1.0,
            extra_wall_prob: 0.3,
            fitness_expr: fitness::PRESET_DEFAULT.to_string(),
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

/// Evaluate a level: solve + Markov analysis + fitness expression. Returns None if unsolvable.
pub fn evaluate(level: &Level, fitness_expr: &FitnessExpr) -> Option<Individual> {
    let solve = solver::solve(level);
    let moves = solve.moves?;

    let graph = build_graph(level);
    let chain = MarkovChain::from_graph(&graph);
    let start_idx = chain.start_idx.expect("BFS-solvable level must have winnable start");

    // Use log-space solver to get log10(win_prob) without underflow,
    // then recover win_prob (may be 0.0 for extreme levels, but log is always finite).
    let log_win_prob = match chain.solve_log_win_probs() {
        Ok(lp) => lp[start_idx],
        Err(_) => return None,
    };
    let win_prob = 10.0f64.powf(log_win_prob).max(0.0);
    let n_states = graph.n_transient;

    let vars = fitness_expr.compute_vars(&graph, level, &solve, win_prob, log_win_prob);
    let fitness = fitness_expr.eval(&vars);

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
    let fitness_expr = match FitnessExpr::parse(&config.fitness_expr) {
        Ok(f) => f,
        Err(e) => {
            let _ = tx.send(GaMessage::Error(format!("Bad fitness expression: {e}")));
            return;
        }
    };

    let mut rng = StdRng::seed_from_u64(config.seed);

    // Evaluate seed levels
    let mut population: Vec<Individual> = seed_levels
        .iter()
        .filter_map(|lev| evaluate(lev, &fitness_expr))
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
                crossover(&p1.level, &p2.level, &mut rng, config.crossover_mode)
            } else {
                let parent = tournament_select(&population, &mut rng, 3);
                mutate_with_config(&parent.level, &mut rng, config)
            };
            offspring_levels.push(child_level);
        }

        // Evaluate non-elite offspring in parallel
        let non_elite = &offspring_levels[n_elite..];
        let n_evaluated = non_elite.len();
        let evaluated: Vec<Individual> = non_elite
            .par_iter()
            .filter_map(|level| evaluate(level, &fitness_expr))
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
