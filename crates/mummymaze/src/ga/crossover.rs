//! Crossover operators for adversarial level generation.

use crate::parse::{Level, WALL_E, WALL_N, WALL_S, WALL_W};
use rand::Rng;

use super::CrossoverMode;
use super::mutation::{find_unoccupied_cell, repair_gate};

/// Dispatch crossover to the selected mode.
pub fn crossover(a: &Level, b: &Level, rng: &mut impl Rng, mode: CrossoverMode) -> Level {
    let mut result = match mode {
        CrossoverMode::SwapEntities => crossover_swap_entities(a, b, rng),
        CrossoverMode::Region => crossover_region(a, b, rng),
        CrossoverMode::WallPatch => crossover_wall_patch(a, b, rng),
        CrossoverMode::FeatureLevel => crossover_feature_level(a, b, rng),
    };
    repair_gate(&mut result);
    result
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

    // Optional entities: take from whichever parent has one in its half;
    // if neither is in its half, fall back to whichever parent has one.
    // Mummy2
    let pa_m2 = pa.has_mummy2 && in_half(pa.mummy2_row, pa.mummy2_col, horizontal, split);
    let pb_m2 = pb.has_mummy2 && !in_half(pb.mummy2_row, pb.mummy2_col, horizontal, split);
    if pa_m2 {
        out.has_mummy2 = true;
        out.mummy2_row = pa.mummy2_row;
        out.mummy2_col = pa.mummy2_col;
    } else if pb_m2 {
        out.has_mummy2 = true;
        out.mummy2_row = pb.mummy2_row;
        out.mummy2_col = pb.mummy2_col;
    } else if pa.has_mummy2 {
        out.has_mummy2 = true;
        out.mummy2_row = pa.mummy2_row;
        out.mummy2_col = pa.mummy2_col;
    } else if pb.has_mummy2 {
        out.has_mummy2 = true;
        out.mummy2_row = pb.mummy2_row;
        out.mummy2_col = pb.mummy2_col;
    } else {
        out.has_mummy2 = false;
        out.mummy2_row = 99;
        out.mummy2_col = 99;
    }

    // Scorpion
    let pa_sc = pa.has_scorpion && in_half(pa.scorpion_row, pa.scorpion_col, horizontal, split);
    let pb_sc = pb.has_scorpion && !in_half(pb.scorpion_row, pb.scorpion_col, horizontal, split);
    if pa_sc {
        out.has_scorpion = true;
        out.scorpion_row = pa.scorpion_row;
        out.scorpion_col = pa.scorpion_col;
    } else if pb_sc {
        out.has_scorpion = true;
        out.scorpion_row = pb.scorpion_row;
        out.scorpion_col = pb.scorpion_col;
    } else if pa.has_scorpion {
        out.has_scorpion = true;
        out.scorpion_row = pa.scorpion_row;
        out.scorpion_col = pa.scorpion_col;
    } else if pb.has_scorpion {
        out.has_scorpion = true;
        out.scorpion_row = pb.scorpion_row;
        out.scorpion_col = pb.scorpion_col;
    } else {
        out.has_scorpion = false;
        out.scorpion_row = 99;
        out.scorpion_col = 99;
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
    match traps.len() {
        0 => {
            out.trap1_row = 99; out.trap1_col = 99;
            out.trap2_row = 99; out.trap2_col = 99;
        }
        1 => {
            out.trap1_row = traps[0].0; out.trap1_col = traps[0].1;
            out.trap2_row = 99; out.trap2_col = 99;
        }
        _ => {
            out.trap1_row = traps[0].0; out.trap1_col = traps[0].1;
            out.trap2_row = traps[1].0; out.trap2_col = traps[1].1;
        }
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
