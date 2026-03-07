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
    /// Probability of an extra wall mutation after the primary mutation.
    pub extra_wall_prob: f64,
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
            extra_wall_prob: 0.3,
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

/// Try up to 20 random cells to find one not in `occupied`.
fn find_unoccupied_cell(
    n: i32,
    occupied: &std::collections::HashSet<(i32, i32)>,
    rng: &mut impl Rng,
) -> Option<(i32, i32)> {
    for _ in 0..20 {
        let r = rng.random_range(0..n);
        let c = rng.random_range(0..n);
        if !occupied.contains(&(r, c)) {
            return Some((r, c));
        }
    }
    None
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

    if let Some((nr, nc)) = find_unoccupied_cell(n, &occupied, rng) {
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
    }

    out
}

/// Move the player to an unoccupied cell.
pub fn mutate_player(level: &Level, rng: &mut impl Rng) -> Level {
    let mut out = level.clone();
    let n = level.grid_size;
    let occupied = occupied_cells(&out, false);

    if let Some((nr, nc)) = find_unoccupied_cell(n, &occupied, rng) {
        out.player_row = nr;
        out.player_col = nc;
    }

    out
}

/// Add an entity (mummy2, scorpion, or trap) that doesn't exist yet.
/// Returns None if all entity slots are already occupied.
pub fn mutate_add_entity(level: &Level, rng: &mut impl Rng) -> Option<Level> {
    // Collect addable entity kinds (check before cloning)
    let mut addable: Vec<EntityKind> = Vec::new();
    if !level.has_mummy2 {
        addable.push(EntityKind::Mummy2);
    }
    if !level.has_scorpion {
        addable.push(EntityKind::Scorpion);
    }
    if level.trap_count == 0 {
        addable.push(EntityKind::Trap1);
    } else if level.trap_count == 1 {
        addable.push(EntityKind::Trap2);
    }

    if addable.is_empty() {
        return None;
    }

    let kind = addable[rng.random_range(0..addable.len())];
    let occupied = occupied_cells(level, true);
    let (nr, nc) = find_unoccupied_cell(level.grid_size, &occupied, rng)?;

    let mut out = level.clone();
    match kind {
        EntityKind::Mummy2 => {
            out.has_mummy2 = true;
            out.mummy2_row = nr;
            out.mummy2_col = nc;
        }
        EntityKind::Scorpion => {
            out.has_scorpion = true;
            out.scorpion_row = nr;
            out.scorpion_col = nc;
        }
        EntityKind::Trap1 => {
            out.trap_count = 1;
            out.trap1_row = nr;
            out.trap1_col = nc;
        }
        EntityKind::Trap2 => {
            out.trap_count = 2;
            out.trap2_row = nr;
            out.trap2_col = nc;
        }
        EntityKind::Mummy1 => unreachable!(),
    }
    Some(out)
}

/// Remove a random optional entity (mummy2, scorpion, or a trap).
/// Returns None if there are no removable entities (only mummy1 + player).
pub fn mutate_remove_entity(level: &Level, rng: &mut impl Rng) -> Option<Level> {
    let mut removable: Vec<EntityKind> = Vec::new();
    if level.has_mummy2 {
        removable.push(EntityKind::Mummy2);
    }
    if level.has_scorpion {
        removable.push(EntityKind::Scorpion);
    }
    if level.trap_count >= 2 {
        removable.push(EntityKind::Trap2);
    }
    if level.trap_count >= 1 {
        removable.push(EntityKind::Trap1);
    }

    if removable.is_empty() {
        return None;
    }

    let mut out = level.clone();
    let kind = removable[rng.random_range(0..removable.len())];
    match kind {
        EntityKind::Mummy2 => {
            out.has_mummy2 = false;
            out.mummy2_row = 99;
            out.mummy2_col = 99;
        }
        EntityKind::Scorpion => {
            out.has_scorpion = false;
            out.scorpion_row = 99;
            out.scorpion_col = 99;
        }
        EntityKind::Trap1 => {
            // If trap2 exists, move it to trap1 slot
            if out.trap_count == 2 {
                out.trap1_row = out.trap2_row;
                out.trap1_col = out.trap2_col;
                out.trap2_row = 99;
                out.trap2_col = 99;
                out.trap_count = 1;
            } else {
                out.trap_count = 0;
                out.trap1_row = 99;
                out.trap1_col = 99;
            }
        }
        EntityKind::Trap2 => {
            out.trap_count = 1;
            out.trap2_row = 99;
            out.trap2_col = 99;
        }
        EntityKind::Mummy1 => unreachable!(),
    }

    Some(out)
}

/// Apply a random mutation operator using weighted probabilities from config.
pub fn mutate_with_config(level: &Level, rng: &mut impl Rng, config: &GaConfig) -> Level {
    let total = config.w_wall + config.w_move_entity + config.w_move_player
        + config.w_add_entity + config.w_remove_entity;
    let r: f64 = rng.random::<f64>() * total;

    let mut cumulative = config.w_wall;
    let mut result = if r < cumulative {
        mutate_wall(level, rng)
    } else {
        cumulative += config.w_move_entity;
        if r < cumulative {
            mutate_entity(level, rng)
        } else {
            cumulative += config.w_move_player;
            if r < cumulative {
                mutate_player(level, rng)
            } else {
                cumulative += config.w_add_entity;
                if r < cumulative {
                    mutate_add_entity(level, rng).unwrap_or_else(|| level.clone())
                } else {
                    mutate_remove_entity(level, rng).unwrap_or_else(|| level.clone())
                }
            }
        }
    };

    if rng.random_bool(config.extra_wall_prob) {
        result = mutate_wall(&result, rng);
    }

    result
}

/// Dispatch crossover to the selected mode.
pub fn crossover(a: &Level, b: &Level, rng: &mut impl Rng, mode: CrossoverMode) -> Level {
    match mode {
        CrossoverMode::SwapEntities => crossover_swap_entities(a, b, rng),
        CrossoverMode::Region => crossover_region(a, b, rng),
        CrossoverMode::WallPatch => crossover_wall_patch(a, b, rng),
        CrossoverMode::FeatureLevel => crossover_feature_level(a, b, rng),
    }
}

/// Pick flip randomly from either parent.
fn random_flip(a: &Level, b: &Level, rng: &mut impl Rng) -> bool {
    if rng.random_bool(0.5) { a.flip } else { b.flip }
}

/// Walls from one parent, entity positions from the other.
fn crossover_swap_entities(a: &Level, b: &Level, rng: &mut impl Rng) -> Level {
    let (wall_parent, entity_parent) = if rng.random_bool(0.5) {
        (a, b)
    } else {
        (b, a)
    };

    let mut out = wall_parent.clone();
    out.flip = random_flip(a, b, rng);
    out.player_row = entity_parent.player_row;
    out.player_col = entity_parent.player_col;
    out.mummy1_row = entity_parent.mummy1_row;
    out.mummy1_col = entity_parent.mummy1_col;
    out.has_mummy2 = entity_parent.has_mummy2;
    out.mummy2_row = entity_parent.mummy2_row;
    out.mummy2_col = entity_parent.mummy2_col;
    out.has_scorpion = entity_parent.has_scorpion;
    out.scorpion_row = entity_parent.scorpion_row;
    out.scorpion_col = entity_parent.scorpion_col;
    out.trap_count = entity_parent.trap_count;
    out.trap1_row = entity_parent.trap1_row;
    out.trap1_col = entity_parent.trap1_col;
    out.trap2_row = entity_parent.trap2_row;
    out.trap2_col = entity_parent.trap2_col;

    out
}

/// Split grid at a random row or column; take walls + entities from each parent in their half.
fn crossover_region(a: &Level, b: &Level, rng: &mut impl Rng) -> Level {
    let (pa, pb) = if rng.random_bool(0.5) { (a, b) } else { (b, a) };
    let n = pa.grid_size;
    let mut out = pa.clone();
    out.flip = random_flip(a, b, rng);

    // Pick a split: horizontal (split_row) or vertical (split_col)
    let horizontal = rng.random_bool(0.5);
    let split = rng.random_range(1..n); // 1..n-1 ensures both halves non-empty

    // Copy walls from pb for cells in the second half
    for r in 0..n {
        for c in 0..n {
            let in_b_half = if horizontal { r >= split } else { c >= split };
            if in_b_half {
                let idx = (c + r * 10) as usize;
                out.walls[idx] = pb.walls[idx];
            }
        }
    }

    // Fix wall consistency along the split boundary (both sides must agree)
    if horizontal {
        let r = split;
        for c in 0..n {
            let above = (c + (r - 1) * 10) as usize;
            let below = (c + r * 10) as usize;
            // below cell comes from pb, above from pa — sync the shared wall
            if out.walls[below] & WALL_N != 0 {
                out.walls[above] |= WALL_S;
            } else {
                out.walls[above] &= !WALL_S;
            }
        }
    } else {
        let c = split;
        for r in 0..n {
            let left = ((c - 1) + r * 10) as usize;
            let right = (c + r * 10) as usize;
            if out.walls[right] & WALL_W != 0 {
                out.walls[left] |= WALL_E;
            } else {
                out.walls[left] &= !WALL_E;
            }
        }
    }

    // Place entities: use whichever parent's position falls in its own half
    fn in_half(row: i32, col: i32, horizontal: bool, split: i32) -> bool {
        if horizontal { row < split } else { col < split }
    }

    // Player
    if in_half(pa.player_row, pa.player_col, horizontal, split) {
        out.player_row = pa.player_row;
        out.player_col = pa.player_col;
    } else {
        out.player_row = pb.player_row;
        out.player_col = pb.player_col;
    }

    // Mummy1 — always present
    if in_half(pa.mummy1_row, pa.mummy1_col, horizontal, split) {
        out.mummy1_row = pa.mummy1_row;
        out.mummy1_col = pa.mummy1_col;
    } else {
        out.mummy1_row = pb.mummy1_row;
        out.mummy1_col = pb.mummy1_col;
    }

    // Optional entities: take from whichever parent has one in its half, prefer pa
    // Mummy2
    let pa_m2 = pa.has_mummy2 && in_half(pa.mummy2_row, pa.mummy2_col, horizontal, split);
    let pb_m2 = pb.has_mummy2 && !in_half(pb.mummy2_row, pb.mummy2_col, horizontal, split);
    if pa_m2 || pb_m2 {
        out.has_mummy2 = true;
        if pa_m2 {
            out.mummy2_row = pa.mummy2_row;
            out.mummy2_col = pa.mummy2_col;
        } else {
            out.mummy2_row = pb.mummy2_row;
            out.mummy2_col = pb.mummy2_col;
        }
    } else {
        out.has_mummy2 = pa.has_mummy2 || pb.has_mummy2;
        if pa.has_mummy2 {
            out.mummy2_row = pa.mummy2_row;
            out.mummy2_col = pa.mummy2_col;
        } else if pb.has_mummy2 {
            out.mummy2_row = pb.mummy2_row;
            out.mummy2_col = pb.mummy2_col;
        }
    }

    // Scorpion
    let pa_sc = pa.has_scorpion && in_half(pa.scorpion_row, pa.scorpion_col, horizontal, split);
    let pb_sc = pb.has_scorpion && !in_half(pb.scorpion_row, pb.scorpion_col, horizontal, split);
    if pa_sc || pb_sc {
        out.has_scorpion = true;
        if pa_sc {
            out.scorpion_row = pa.scorpion_row;
            out.scorpion_col = pa.scorpion_col;
        } else {
            out.scorpion_row = pb.scorpion_row;
            out.scorpion_col = pb.scorpion_col;
        }
    } else {
        out.has_scorpion = pa.has_scorpion || pb.has_scorpion;
        if pa.has_scorpion {
            out.scorpion_row = pa.scorpion_row;
            out.scorpion_col = pa.scorpion_col;
        } else if pb.has_scorpion {
            out.scorpion_row = pb.scorpion_row;
            out.scorpion_col = pb.scorpion_col;
        }
    }

    // Traps: collect from both parents, keep those in their respective halves
    let mut traps: Vec<(i32, i32)> = Vec::new();
    if pa.trap_count >= 1 && in_half(pa.trap1_row, pa.trap1_col, horizontal, split) {
        traps.push((pa.trap1_row, pa.trap1_col));
    }
    if pa.trap_count >= 2 && in_half(pa.trap2_row, pa.trap2_col, horizontal, split) {
        traps.push((pa.trap2_row, pa.trap2_col));
    }
    if pb.trap_count >= 1 && !in_half(pb.trap1_row, pb.trap1_col, horizontal, split) {
        traps.push((pb.trap1_row, pb.trap1_col));
    }
    if pb.trap_count >= 2 && !in_half(pb.trap2_row, pb.trap2_col, horizontal, split) {
        traps.push((pb.trap2_row, pb.trap2_col));
    }
    traps.truncate(2);
    out.trap_count = traps.len() as i32;
    if traps.len() >= 1 {
        out.trap1_row = traps[0].0;
        out.trap1_col = traps[0].1;
    }
    if traps.len() >= 2 {
        out.trap2_row = traps[1].0;
        out.trap2_col = traps[1].1;
    }

    // Keep exit from pa (walls determine valid exits)
    out
}

/// Start from one parent intact. Copy a random rectangular wall patch from the other.
fn crossover_wall_patch(a: &Level, b: &Level, rng: &mut impl Rng) -> Level {
    let (base, donor) = if rng.random_bool(0.5) { (a, b) } else { (b, a) };
    let n = base.grid_size;
    let mut out = base.clone();
    out.flip = random_flip(a, b, rng);

    // Random rectangle: r0..r1, c0..c1 (inclusive)
    let r0 = rng.random_range(0..n);
    let r1 = rng.random_range(r0..n);
    let c0 = rng.random_range(0..n);
    let c1 = rng.random_range(c0..n);

    // Copy walls within the patch from donor
    for r in r0..=r1 {
        for c in c0..=c1 {
            let idx = (c + r * 10) as usize;
            out.walls[idx] = donor.walls[idx];
        }
    }

    // Fix boundary consistency: walls shared between patch interior and exterior
    // Top boundary
    if r0 > 0 {
        for c in c0..=c1 {
            let inside = (c + r0 * 10) as usize;
            let outside = (c + (r0 - 1) * 10) as usize;
            if out.walls[inside] & WALL_N != 0 {
                out.walls[outside] |= WALL_S;
            } else {
                out.walls[outside] &= !WALL_S;
            }
        }
    }
    // Bottom boundary
    if r1 < n - 1 {
        for c in c0..=c1 {
            let inside = (c + r1 * 10) as usize;
            let outside = (c + (r1 + 1) * 10) as usize;
            if out.walls[inside] & WALL_S != 0 {
                out.walls[outside] |= WALL_N;
            } else {
                out.walls[outside] &= !WALL_N;
            }
        }
    }
    // Left boundary
    if c0 > 0 {
        for r in r0..=r1 {
            let inside = (c0 + r * 10) as usize;
            let outside = ((c0 - 1) + r * 10) as usize;
            if out.walls[inside] & WALL_W != 0 {
                out.walls[outside] |= WALL_E;
            } else {
                out.walls[outside] &= !WALL_E;
            }
        }
    }
    // Right boundary
    if c1 < n - 1 {
        for r in r0..=r1 {
            let inside = (c1 + r * 10) as usize;
            let outside = ((c1 + 1) + r * 10) as usize;
            if out.walls[inside] & WALL_E != 0 {
                out.walls[outside] |= WALL_W;
            } else {
                out.walls[outside] &= !WALL_W;
            }
        }
    }

    out
}

/// Take entity composition (types + flip) from one parent, re-place them randomly
/// in the other parent's wall layout.
fn crossover_feature_level(a: &Level, b: &Level, rng: &mut impl Rng) -> Level {
    let (wall_parent, feature_parent) = if rng.random_bool(0.5) {
        (a, b)
    } else {
        (b, a)
    };

    let mut out = wall_parent.clone();
    out.flip = random_flip(a, b, rng);
    out.has_mummy2 = feature_parent.has_mummy2;
    out.has_scorpion = feature_parent.has_scorpion;
    out.trap_count = feature_parent.trap_count;

    // Re-place all entities at random unoccupied positions
    let n = out.grid_size;
    let mut occupied = std::collections::HashSet::new();

    // Player
    if let Some((r, c)) = find_unoccupied_cell(n, &occupied, rng) {
        out.player_row = r;
        out.player_col = c;
        occupied.insert((r, c));
    }

    // Mummy1
    if let Some((r, c)) = find_unoccupied_cell(n, &occupied, rng) {
        out.mummy1_row = r;
        out.mummy1_col = c;
        occupied.insert((r, c));
    }

    // Mummy2
    if out.has_mummy2 {
        if let Some((r, c)) = find_unoccupied_cell(n, &occupied, rng) {
            out.mummy2_row = r;
            out.mummy2_col = c;
            occupied.insert((r, c));
        }
    } else {
        out.mummy2_row = 99;
        out.mummy2_col = 99;
    }

    // Scorpion
    if out.has_scorpion {
        if let Some((r, c)) = find_unoccupied_cell(n, &occupied, rng) {
            out.scorpion_row = r;
            out.scorpion_col = c;
            occupied.insert((r, c));
        }
    } else {
        out.scorpion_row = 99;
        out.scorpion_col = 99;
    }

    // Traps
    if out.trap_count >= 1 {
        if let Some((r, c)) = find_unoccupied_cell(n, &occupied, rng) {
            out.trap1_row = r;
            out.trap1_col = c;
            occupied.insert((r, c));
        }
    } else {
        out.trap1_row = 99;
        out.trap1_col = 99;
    }
    if out.trap_count >= 2 {
        if let Some((r, c)) = find_unoccupied_cell(n, &occupied, rng) {
            out.trap2_row = r;
            out.trap2_col = c;
            occupied.insert((r, c));
        }
    } else {
        out.trap2_row = 99;
        out.trap2_col = 99;
    }

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
