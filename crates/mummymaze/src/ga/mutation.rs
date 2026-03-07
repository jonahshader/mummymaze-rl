//! Mutation operators for adversarial level generation.

use crate::parse::{Level, WALL_E, WALL_N, WALL_S, WALL_W};
use rand::Rng;
use std::collections::HashSet;

use super::GaConfig;

/// Collect all occupied entity positions from a level.
pub(crate) fn occupied_cells(level: &Level, include_player: bool) -> HashSet<(i32, i32)> {
    let mut occupied = HashSet::new();
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
pub(crate) fn find_unoccupied_cell(
    n: i32,
    occupied: &HashSet<(i32, i32)>,
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
