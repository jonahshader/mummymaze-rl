//! Random solvable level generator.
//!
//! Generates levels with random wall layouts and entity placements,
//! retrying until BFS-solvable. Entity composition is configurable
//! via spawn probabilities.

use crate::parse::{Level, MAX_GRID, EXIT_E, EXIT_N, EXIT_S, EXIT_W, WALL_E, WALL_N, WALL_S, WALL_W};
use crate::solver;
use rand::Rng;
use std::collections::HashSet;

use super::mutation::{find_unoccupied_cell, repair_gate};

/// Configuration for random level generation.
#[derive(Debug, Clone)]
pub struct RandomLevelConfig {
    pub grid_size: i32,
    /// Probability that each interior wall exists.
    pub wall_density: f64,
    /// Probability of a second mummy.
    pub mummy2_prob: f64,
    /// Probability of a scorpion.
    pub scorpion_prob: f64,
    /// Probability of 1 trap.
    pub trap1_prob: f64,
    /// Probability of 2 traps (only if trap1 also spawned).
    pub trap2_prob: f64,
    /// Probability of a gate+key pair.
    pub gate_prob: f64,
    /// Probability of flip (red mummies = horizontal-first).
    pub flip_prob: f64,
    /// Maximum BFS solve attempts before giving up on one candidate.
    pub max_attempts: usize,
}

impl Default for RandomLevelConfig {
    fn default() -> Self {
        Self {
            grid_size: 6,
            wall_density: 0.3,
            mummy2_prob: 0.3,
            scorpion_prob: 0.2,
            trap1_prob: 0.15,
            trap2_prob: 0.3,
            gate_prob: 0.15,
            flip_prob: 0.5,
            max_attempts: 1000,
        }
    }
}

/// Generate a single random solvable level.
///
/// Retries up to `config.max_attempts` times until BFS finds a solution.
/// Returns `None` if no solvable level is found within the attempt limit.
pub fn generate_random_solvable(
    rng: &mut impl Rng,
    config: &RandomLevelConfig,
) -> Option<Level> {
    for _ in 0..config.max_attempts {
        let level = generate_random_level(rng, config);
        let result = solver::solve(&level);
        if result.moves.is_some() {
            return Some(level);
        }
    }
    None
}

/// Generate a single random level (may or may not be solvable).
fn generate_random_level(rng: &mut impl Rng, config: &RandomLevelConfig) -> Level {
    let n = config.grid_size;
    let flip = rng.random_bool(config.flip_prob);

    // --- Walls ---
    let mut walls = [0u32; MAX_GRID * MAX_GRID];

    // Boundary walls (always present)
    for i in 0..n {
        // Top row: WALL_N
        walls[i as usize] |= WALL_N;
        // Bottom row: WALL_S
        walls[(i + (n - 1) * 10) as usize] |= WALL_S;
        // Left column: WALL_W
        walls[(i * 10) as usize] |= WALL_W;
        // Right column: WALL_E
        walls[((n - 1) + i * 10) as usize] |= WALL_E;
    }

    // Random interior horizontal walls
    for r in 1..n {
        for c in 0..n {
            if rng.random_bool(config.wall_density) {
                let upper = (c + (r - 1) * 10) as usize;
                let lower = (c + r * 10) as usize;
                walls[upper] |= WALL_S;
                walls[lower] |= WALL_N;
            }
        }
    }

    // Random interior vertical walls
    for r in 0..n {
        for c in 1..n {
            if rng.random_bool(config.wall_density) {
                let left = ((c - 1) + r * 10) as usize;
                let right = (c + r * 10) as usize;
                walls[left] |= WALL_E;
                walls[right] |= WALL_W;
            }
        }
    }

    // --- Exit ---
    let side = rng.random_range(0..4);
    let exit_pos = rng.random_range(0..n);
    let (exit_row, exit_col, exit_mask) = match side {
        0 => {
            // North
            let idx = exit_pos as usize;
            walls[idx] &= !WALL_N;
            walls[idx] |= EXIT_N;
            (0, exit_pos, EXIT_N)
        }
        1 => {
            // South
            let idx = (exit_pos + (n - 1) * 10) as usize;
            walls[idx] &= !WALL_S;
            walls[idx] |= EXIT_S;
            (n - 1, exit_pos, EXIT_S)
        }
        2 => {
            // West
            let idx = (exit_pos * 10) as usize;
            walls[idx] &= !WALL_W;
            walls[idx] |= EXIT_W;
            (exit_pos, 0, EXIT_W)
        }
        _ => {
            // East
            let idx = ((n - 1) + exit_pos * 10) as usize;
            walls[idx] &= !WALL_E;
            walls[idx] |= EXIT_E;
            (exit_pos, n - 1, EXIT_E)
        }
    };

    // --- Entity placement ---
    let mut occupied = HashSet::new();
    // Don't place entities on the exit cell
    occupied.insert((exit_row, exit_col));

    let player = find_unoccupied_cell(n, &occupied, rng).unwrap_or((0, 0));
    occupied.insert(player);

    let mummy1 = find_unoccupied_cell(n, &occupied, rng).unwrap_or((n - 1, n - 1));
    occupied.insert(mummy1);

    let (m2r, m2c, has_mummy2) = if rng.random_bool(config.mummy2_prob) {
        if let Some(pos) = find_unoccupied_cell(n, &occupied, rng) {
            occupied.insert(pos);
            (pos.0, pos.1, true)
        } else {
            (99, 99, false)
        }
    } else {
        (99, 99, false)
    };

    let (sr, sc, has_scorpion) = if rng.random_bool(config.scorpion_prob) {
        if let Some(pos) = find_unoccupied_cell(n, &occupied, rng) {
            occupied.insert(pos);
            (pos.0, pos.1, true)
        } else {
            (99, 99, false)
        }
    } else {
        (99, 99, false)
    };

    let mut trap_count = 0;
    let (t1r, t1c) = if rng.random_bool(config.trap1_prob) {
        if let Some(pos) = find_unoccupied_cell(n, &occupied, rng) {
            occupied.insert(pos);
            trap_count = 1;
            (pos.0, pos.1)
        } else {
            (99, 99)
        }
    } else {
        (99, 99)
    };

    let (t2r, t2c) = if trap_count == 1 && rng.random_bool(config.trap2_prob) {
        if let Some(pos) = find_unoccupied_cell(n, &occupied, rng) {
            occupied.insert(pos);
            trap_count = 2;
            (pos.0, pos.1)
        } else {
            (99, 99)
        }
    } else {
        (99, 99)
    };

    let (gr, gc, kr, kc, has_gate) = if rng.random_bool(config.gate_prob) {
        // Gate must not be on rightmost column (east edge would point off-grid)
        let gate_pos = find_unoccupied_cell(n - 1, &occupied, rng);
        if let Some(gp) = gate_pos {
            if let Some(kp) = find_unoccupied_cell(n, &occupied, rng) {
                occupied.insert(gp);
                occupied.insert(kp);
                (gp.0, gp.1, kp.0, kp.1, true)
            } else {
                (99, 99, 99, 99, false)
            }
        } else {
            (99, 99, 99, 99, false)
        }
    } else {
        (99, 99, 99, 99, false)
    };

    let mut level = Level {
        grid_size: n,
        flip,
        walls,
        player_row: player.0,
        player_col: player.1,
        mummy1_row: mummy1.0,
        mummy1_col: mummy1.1,
        mummy2_row: m2r,
        mummy2_col: m2c,
        has_mummy2,
        scorpion_row: sr,
        scorpion_col: sc,
        has_scorpion,
        trap1_row: t1r,
        trap1_col: t1c,
        trap2_row: t2r,
        trap2_col: t2c,
        trap_count,
        gate_row: gr,
        gate_col: gc,
        has_gate,
        key_row: kr,
        key_col: kc,
        exit_row,
        exit_col,
        exit_mask,
    };

    if has_gate {
        repair_gate(&mut level);
    }

    level
}
