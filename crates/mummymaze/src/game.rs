//! Game engine — direct port of csolver/src/game.c
//!
//! State struct, step(), move_enemy(), can_move_player(), gate/key logic.
//! Same variable names and control flow as the C, only the outer API uses Rust idioms.

use crate::parse::{Level, WALL_E, WALL_N, WALL_S, WALL_W};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    North,
    South,
    East,
    West,
    Wait,
}

impl Action {
    pub const ALL: [Action; 5] = [
        Action::North,
        Action::South,
        Action::East,
        Action::West,
        Action::Wait,
    ];

    pub fn delta(self) -> (i32, i32) {
        match self {
            Action::North => (-1, 0),
            Action::South => (1, 0),
            Action::East => (0, 1),
            Action::West => (0, -1),
            Action::Wait => (0, 0),
        }
    }

    /// Integer index matching JAX env convention: N=0, S=1, E=2, W=3, Wait=4
    pub fn to_index(self) -> u8 {
        match self {
            Action::North => 0,
            Action::South => 1,
            Action::East => 2,
            Action::West => 3,
            Action::Wait => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepResult {
    Ok,
    Dead,
    Win,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct State {
    pub player_row: i32,
    pub player_col: i32,
    pub mummy1_row: i32,
    pub mummy1_col: i32,
    pub mummy1_alive: bool,
    pub mummy2_row: i32,
    pub mummy2_col: i32,
    pub mummy2_alive: bool,
    pub scorpion_row: i32,
    pub scorpion_col: i32,
    pub scorpion_alive: bool,
    pub gate_open: bool, // true = blocking (closed), false = open — matches C's gate_open=1
}

impl State {
    pub fn from_level(lev: &Level) -> State {
        State {
            player_row: lev.player_row,
            player_col: lev.player_col,
            mummy1_row: lev.mummy1_row,
            mummy1_col: lev.mummy1_col,
            mummy1_alive: true,
            mummy2_row: lev.mummy2_row,
            mummy2_col: lev.mummy2_col,
            mummy2_alive: lev.has_mummy2,
            scorpion_row: lev.scorpion_row,
            scorpion_col: lev.scorpion_col,
            scorpion_alive: lev.has_scorpion,
            gate_open: true, // starts closed/blocking
        }
    }

    /// Normalize dead entities to sentinel positions so that states with
    /// dead entities at different positions hash/compare as equal.
    /// The binary doesn't reset positions on death, but for state graph
    /// building we need canonical keys.
    pub fn normalize(&mut self) {
        if !self.mummy1_alive {
            self.mummy1_row = 99;
            self.mummy1_col = 99;
        }
        if !self.mummy2_alive {
            self.mummy2_row = 99;
            self.mummy2_col = 99;
        }
        if !self.scorpion_alive {
            self.scorpion_row = 99;
            self.scorpion_col = 99;
        }
    }
}

/// Move enemy one step toward the player, matching the binary's logic.
/// Returns the new (row, col) for the enemy.
///
/// flip=false (white mummies): try vertical first, then horizontal.
/// flip=true (red mummies): try horizontal first, then vertical.
fn move_enemy(
    lev: &Level,
    pr: i32,
    pc: i32,
    gate_open: bool,
    r: i32,
    c: i32,
) -> (i32, i32) {
    let n = lev.grid_size;

    // Bounds check
    if r < 0 || r >= n || c < 0 || c >= n {
        return (r, c);
    }

    let w = lev.walls[(c + r * 10) as usize];

    if !lev.flip {
        // flip=0: try vertical first (row), then horizontal (col)

        // Try vertical (south/north)
        if r < pr && (w & WALL_S) == 0 {
            return (r + 1, c);
        }
        if pr < r && (w & WALL_N) == 0 {
            return (r - 1, c);
        }

        // Try horizontal (east/west) with gate check
        if c < pc && (w & WALL_E) == 0 {
            // Gate check for flip=0 east: blocked if gate closed AND at gate cell
            if !gate_open || r != lev.gate_row || c != lev.gate_col {
                return (r, c + 1);
            }
        }
        if pc < c && (w & WALL_W) == 0 {
            // Gate check for flip=0 west: blocked if gate closed AND dest is gate cell
            if !gate_open || r != lev.gate_row || (c - 1) != lev.gate_col {
                return (r, c - 1);
            }
        }
    } else {
        // flip=1: try horizontal first (col), then vertical (row)

        // Try horizontal (east/west) with gate check
        if c < pc
            && ((w & WALL_E) == 0
                || (!gate_open && r == lev.gate_row && c == lev.gate_col))
        {
            return (r, c + 1);
        }
        if pc < c
            && ((w & WALL_W) == 0
                || (!gate_open && r == lev.gate_row && (c - 1) == lev.gate_col))
        {
            return (r, c - 1);
        }

        // Try vertical (south/north)
        if r < pr && (w & WALL_S) == 0 {
            return (r + 1, c);
        }
        if pr < r && (w & WALL_N) == 0 {
            return (r - 1, c);
        }
    }

    (r, c) // no move possible
}

/// Check and toggle gate when an entity enters the key cell.
fn check_key_toggle(
    lev: &Level,
    gate_open: &mut bool,
    new_r: i32,
    new_c: i32,
    old_r: i32,
    old_c: i32,
    original_gate: bool,
    key_toggled: &mut bool,
) {
    if !lev.has_gate {
        return;
    }
    if *key_toggled {
        return;
    }
    if new_r == lev.key_row && new_c == lev.key_col && (new_r != old_r || new_c != old_c) {
        *gate_open = !original_gate;
        *key_toggled = true;
    }
}

/// Can the player move from (src_r, src_c) to (dst_r, dst_c)?
/// Direct port of FUN_004079c0 from the binary.
pub fn can_move_player(lev: &Level, gate_open: bool, src_r: i32, src_c: i32, dst_r: i32, dst_c: i32) -> bool {
    let n = lev.grid_size;

    // Bounds check
    if dst_r < 0 || dst_r >= n || dst_c < 0 || dst_c >= n {
        return false;
    }

    // No movement = always valid
    if dst_r == src_r && dst_c == src_c {
        return true;
    }

    let w = lev.walls[(src_c + src_r * 10) as usize];

    // Vertical movement
    if dst_c == src_c {
        if dst_r == src_r - 1 {
            return (w & WALL_N) == 0;
        }
        if dst_r == src_r + 1 {
            return (w & WALL_S) == 0;
        }
        return false;
    }

    // Horizontal movement
    if dst_r == src_r {
        if dst_c == src_c - 1 {
            // West
            if (w & WALL_W) != 0 {
                return false;
            }
            if gate_open && src_r == lev.gate_row && (src_c - 1) == lev.gate_col {
                return false;
            }
            return true;
        }
        if dst_c == src_c + 1 {
            // East
            if (w & WALL_E) != 0 {
                return false;
            }
            if gate_open && src_r == lev.gate_row && src_c == lev.gate_col {
                return false;
            }
            return true;
        }
    }
    false
}

/// Check if the player can perform the given action. Used by graph builder.
pub fn can_do_action(lev: &Level, s: &State, action: Action) -> bool {
    let (dr, dc) = action.delta();
    if dr == 0 && dc == 0 {
        return true; // Wait is always valid
    }
    can_move_player(lev, s.gate_open, s.player_row, s.player_col, s.player_row + dr, s.player_col + dc)
}

/// Execute one full game turn. Direct port of game.c::step().
pub fn step(lev: &Level, s: &mut State, action: Action) -> StepResult {
    let (dr, dc) = action.delta();
    let n = lev.grid_size;

    // Save original turn-start state
    let orig_gate = s.gate_open;
    let orig_m1r = s.mummy1_row;
    let orig_m1c = s.mummy1_col;
    let orig_m2r = s.mummy2_row;
    let orig_m2c = s.mummy2_col;

    // 1. Player movement
    let old_pr = s.player_row;
    let old_pc = s.player_col;
    let new_pr = old_pr + dr;
    let new_pc = old_pc + dc;

    if dr == 0 && dc == 0 {
        // Wait action
    } else if can_move_player(lev, s.gate_open, old_pr, old_pc, new_pr, new_pc) {
        s.player_row = new_pr;
        s.player_col = new_pc;
    }

    // 2. Key/gate toggle from player entering key cell
    let mut key_toggled = false;
    {
        let pr = s.player_row;
        let pc = s.player_col;
        check_key_toggle(
            lev,
            &mut s.gate_open,
            pr,
            pc,
            old_pr,
            old_pc,
            orig_gate,
            &mut key_toggled,
        );
    }

    // 3. Trap check
    if (s.player_row == lev.trap1_row && s.player_col == lev.trap1_col)
        || (s.player_row == lev.trap2_row && s.player_col == lev.trap2_col)
    {
        return StepResult::Dead;
    }

    // 4. Scorpion movement (1 step)
    let old_scorp_r = s.scorpion_row;
    let old_scorp_c = s.scorpion_col;
    if s.scorpion_alive {
        let sr = s.scorpion_row;
        let sc = s.scorpion_col;
        if sr < 0 || sr >= n || sc < 0 || sc >= n {
            s.scorpion_alive = false;
        } else {
            let (nr, nc) = move_enemy(lev, s.player_row, s.player_col, s.gate_open, sr, sc);
            s.scorpion_row = nr;
            s.scorpion_col = nc;
        }
    }

    // Scorpion-player death check
    if s.scorpion_alive
        && s.player_row == s.scorpion_row
        && s.player_col == s.scorpion_col
    {
        return StepResult::Dead;
    }

    // Scorpion-mummy collision: scorpion landing on mummy kills scorpion
    if s.scorpion_alive
        && s.mummy1_alive
        && s.scorpion_row == s.mummy1_row
        && s.scorpion_col == s.mummy1_col
    {
        s.scorpion_alive = false;
    }
    if s.scorpion_alive
        && s.mummy2_alive
        && s.scorpion_row == s.mummy2_row
        && s.scorpion_col == s.mummy2_col
    {
        s.scorpion_alive = false;
    }

    // Scorpion key toggle
    if s.scorpion_alive {
        let sr = s.scorpion_row;
        let sc = s.scorpion_col;
        check_key_toggle(
            lev,
            &mut s.gate_open,
            sr,
            sc,
            old_scorp_r,
            old_scorp_c,
            orig_gate,
            &mut key_toggled,
        );
    }

    // 5. Mummy movement — 2 iterations
    for _mstep in 0..2 {
        if s.mummy1_alive {
            let (nr, nc) = move_enemy(
                lev, s.player_row, s.player_col, s.gate_open,
                s.mummy1_row, s.mummy1_col,
            );
            s.mummy1_row = nr;
            s.mummy1_col = nc;
        }
        if s.mummy2_alive {
            let (nr, nc) = move_enemy(
                lev, s.player_row, s.player_col, s.gate_open,
                s.mummy2_row, s.mummy2_col,
            );
            s.mummy2_row = nr;
            s.mummy2_col = nc;
        }

        // Mummy-mummy collision: if two mummies on same cell, mummy2 dies
        if s.mummy1_alive
            && s.mummy2_alive
            && s.mummy1_row == s.mummy2_row
            && s.mummy1_col == s.mummy2_col
        {
            s.mummy2_alive = false;
        }

        // Mummy-scorpion collision: mummy kills scorpion.
        // Binary: does NOT check scorpion_alive here. Also toggles gate
        // (using !current_gate, not key_toggled guard) if kill happens on key cell.
        if s.mummy1_alive
            && s.mummy1_row == s.scorpion_row
            && s.mummy1_col == s.scorpion_col
        {
            if lev.has_gate
                && s.scorpion_row == lev.key_row
                && s.scorpion_col == lev.key_col
            {
                s.gate_open = !s.gate_open;
            }
            s.scorpion_alive = false;
        }
        if s.mummy2_alive
            && s.mummy2_row == s.scorpion_row
            && s.mummy2_col == s.scorpion_col
        {
            if lev.has_gate
                && s.scorpion_row == lev.key_row
                && s.scorpion_col == lev.key_col
            {
                s.gate_open = !s.gate_open;
            }
            s.scorpion_alive = false;
        }

        // Mummy-player death check
        if s.mummy1_alive
            && s.player_row == s.mummy1_row
            && s.player_col == s.mummy1_col
        {
            return StepResult::Dead;
        }
        if s.mummy2_alive
            && s.player_row == s.mummy2_row
            && s.player_col == s.mummy2_col
        {
            return StepResult::Dead;
        }

        // Mummy key toggles — mutually exclusive (if/else).
        // key_toggled guard: only one entity toggles via key entry per turn.
        // "Moved" check compares against turn-start positions (orig_m*).
        if !key_toggled {
            if s.mummy1_alive
                && s.mummy1_row == lev.key_row
                && s.mummy1_col == lev.key_col
                && (s.mummy1_row != orig_m1r || s.mummy1_col != orig_m1c)
            {
                s.gate_open = !orig_gate;
                key_toggled = true;
            } else if s.mummy2_alive
                && s.mummy2_row == lev.key_row
                && s.mummy2_col == lev.key_col
                && (s.mummy2_row != orig_m2r || s.mummy2_col != orig_m2c)
            {
                s.gate_open = !orig_gate;
                // Binary does NOT set key_toggled here (goto skips it)
            }
        }
    }

    // 6. Win check — player reached exit cell and exit flag is set
    if s.player_row == lev.exit_row && s.player_col == lev.exit_col {
        let w = lev.walls[(lev.exit_col + lev.exit_row * 10) as usize];
        if (w & lev.exit_mask) != 0 {
            return StepResult::Win;
        }
    }

    // Normalize dead entities to sentinel positions so state hashing is canonical
    s.normalize();

    StepResult::Ok
}
