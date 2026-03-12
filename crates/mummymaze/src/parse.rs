//! .dat file parser — direct port of csolver/src/parse.c
//!
//! Wall encoding: `walls[col + row * 10]` with bit flags:
//!   bit 0 (1) = WALL_W, bit 1 (2) = WALL_E,
//!   bit 2 (4) = WALL_S, bit 3 (8) = WALL_N
//! Exit flags in upper nibble: EXIT_W=0x10, EXIT_E=0x20, EXIT_S=0x40, EXIT_N=0x80
//!
//! ## Why cell bitmasks instead of h_walls/v_walls edge arrays
//!
//! The Python side (parser, game.py, JAX env) uses non-redundant edge arrays
//! (h_walls, v_walls) where each wall is stored once. This crate uses redundant
//! per-cell bitmasks (a wall between two cells sets flags on both). We considered
//! switching to edge arrays for consistency, but kept bitmasks because:
//!
//! - The game engine is a verified port of the original binary (9,814 solution
//!   agreement). Rewriting all wall checks risks regressions for no functional gain.
//! - At max grid size 10×10, the entire walls array (400 bytes) fits in L1 cache,
//!   so the layout choice has no measurable performance impact.
//! - The bitmask format is internal — `Level::from_edges()` provides a clean
//!   constructor from edge arrays, hiding the redundancy from callers.

use crate::error::{MummyMazeError, Result};
use std::path::Path;

pub const MAX_GRID: usize = 10;

pub const WALL_W: u32 = 1;
pub const WALL_E: u32 = 2;
pub const WALL_S: u32 = 4;
pub const WALL_N: u32 = 8;
pub const EXIT_W: u32 = 0x10;
pub const EXIT_E: u32 = 0x20;
pub const EXIT_S: u32 = 0x40;
pub const EXIT_N: u32 = 0x80;

#[derive(Debug, Clone)]
pub struct Header {
    pub grid_size: i32,
    pub flip: bool,
    pub num_sublevels: i32,
    pub mummy_count: i32,
    pub key_gate: i32,
    pub trap_count: i32,
    pub scorpion: i32,
    pub wall_bytes: i32,
    pub bytes_per_sub: i32,
}

/// Serde helper for `[u32; 100]` — serde only supports arrays up to 32 natively.
mod walls_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(walls: &[u32; 100], ser: S) -> Result<S::Ok, S::Error> {
        walls.as_slice().serialize(ser)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<[u32; 100], D::Error> {
        let v: Vec<u32> = Vec::deserialize(de)?;
        v.try_into()
            .map_err(|v: Vec<u32>| serde::de::Error::custom(format!("expected 100 elements, got {}", v.len())))
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Level {
    pub grid_size: i32,
    pub flip: bool,
    #[serde(with = "walls_serde")]
    pub walls: [u32; MAX_GRID * MAX_GRID],

    pub player_row: i32,
    pub player_col: i32,
    pub mummy1_row: i32,
    pub mummy1_col: i32,
    pub mummy2_row: i32,
    pub mummy2_col: i32,
    pub scorpion_row: i32,
    pub scorpion_col: i32,
    pub trap1_row: i32,
    pub trap1_col: i32,
    pub trap2_row: i32,
    pub trap2_col: i32,
    pub trap_count: i32,
    pub has_mummy2: bool,
    pub has_scorpion: bool,

    pub gate_row: i32,
    pub gate_col: i32,
    pub has_gate: bool,
    pub key_row: i32,
    pub key_col: i32,

    pub exit_row: i32,
    pub exit_col: i32,
    pub exit_mask: u32,
}

pub fn parse_header(data: &[u8]) -> Result<Header> {
    if data.len() < 6 {
        return Err(MummyMazeError::Parse("header too short".into()));
    }
    let grid_size = (data[0] & 0x0F) as i32;
    let flip = (data[0] & 0xF0) != 0;
    let num_sublevels = data[1] as i32;
    let mummy_count = data[2] as i32;
    let key_gate = data[3] as i32;
    let trap_count = data[4] as i32;
    let scorpion = data[5] as i32;

    let n = grid_size;
    let wall_bytes = n * if n > 8 { 2 } else { 1 } * 2;
    let bytes_per_sub =
        wall_bytes + 3 + (mummy_count - 1) + 2 * key_gate + scorpion + trap_count;

    Ok(Header {
        grid_size,
        flip,
        num_sublevels,
        mummy_count,
        key_gate,
        trap_count,
        scorpion,
        wall_bytes,
        bytes_per_sub,
    })
}

fn read_wall_bits(data: &[u8], pos: &mut usize, n: i32) -> u16 {
    let bits = data[*pos] as u16;
    *pos += 1;
    if n > 8 {
        let bits = bits | ((data[*pos] as u16) << 8);
        *pos += 1;
        bits
    } else {
        bits
    }
}

fn load_walls_flip0(walls: &mut [u32; MAX_GRID * MAX_GRID], data: &[u8], pos: &mut usize, n: i32) {
    // First wall loop: N/S walls
    for col in 0..n {
        let bits = read_wall_bits(data, pos, n);
        for row in 0..n {
            if bits & (1 << row) != 0 {
                walls[(col + row * 10) as usize] |= WALL_N;
                if row > 0 {
                    walls[(col + (row - 1) * 10) as usize] |= WALL_S;
                }
            }
        }
    }

    // Second wall loop: W/E walls
    for col in 0..n {
        let bits = read_wall_bits(data, pos, n);
        for row in 0..n {
            if bits & (1 << row) != 0 {
                walls[(col + row * 10) as usize] |= WALL_W;
                if col > 0 {
                    walls[(col - 1 + row * 10) as usize] |= WALL_E;
                }
            }
        }
    }
}

fn load_walls_flip1(walls: &mut [u32; MAX_GRID * MAX_GRID], data: &[u8], pos: &mut usize, n: i32) {
    // First wall loop: W/E walls
    for i_var10 in 0..n {
        let bits = read_wall_bits(data, pos, n);
        for i_var8 in 0..n {
            if bits & (1 << i_var8) != 0 {
                let idx = (i_var8 + (n - i_var10) * 10) as usize;
                walls[idx - 10] |= WALL_W;
                if i_var8 > 0 {
                    walls[idx - 11] |= WALL_E;
                }
            }
        }
    }

    // Second wall loop: S/N walls
    for i_var10 in 0..n {
        let bits = read_wall_bits(data, pos, n);
        for i_var8 in 0..n {
            if bits & (1 << i_var8) != 0 {
                let idx = (i_var8 + (n - i_var10) * 10) as usize;
                walls[idx - 10] |= WALL_S;
                if i_var10 > 0 {
                    walls[idx] |= WALL_N;
                }
            }
        }
    }
}

/// Returns (exit_row, exit_col, exit_mask)
fn load_exit_flip0(walls: &mut [u32; MAX_GRID * MAX_GRID], data: &[u8], pos: &mut usize, n: i32) -> (i32, i32, u32) {
    let eb = data[*pos];
    *pos += 1;
    let side = (eb & 0x0F) as i32;
    let p = ((eb >> 4) & 0x0F) as i32;

    match side {
        0 => {
            // West exit
            let idx = (0 + p * 10) as usize;
            walls[idx] |= EXIT_W;
            walls[idx] ^= WALL_W;
            (p, 0, EXIT_W)
        }
        1 => {
            // North exit
            let idx = p as usize;
            walls[idx] |= EXIT_N;
            walls[idx] ^= WALL_N;
            (0, p, EXIT_N)
        }
        2 => {
            // South exit
            let idx = (p + (n - 1) * 10) as usize;
            walls[idx] |= EXIT_S;
            walls[idx] ^= WALL_S;
            (n - 1, p, EXIT_S)
        }
        3 => {
            // East exit
            let idx = ((n - 1) + p * 10) as usize;
            walls[idx] |= EXIT_E;
            walls[idx] ^= WALL_E;
            (p, n - 1, EXIT_E)
        }
        _ => (-1, -1, 0),
    }
}

/// Returns (exit_row, exit_col, exit_mask)
fn load_exit_flip1(walls: &mut [u32; MAX_GRID * MAX_GRID], data: &[u8], pos: &mut usize, n: i32) -> (i32, i32, u32) {
    let eb = data[*pos];
    *pos += 1;
    let side = (eb & 0x0F) as i32;
    let p = ((eb >> 4) & 0x0F) as i32;

    match side {
        0 => {
            // South exit
            let idx = (p + (n - 1) * 10) as usize;
            walls[idx] |= EXIT_S;
            walls[idx] ^= WALL_S;
            (n - 1, p, EXIT_S)
        }
        1 => {
            // West exit
            let idx = (0 + (n - p - 1) * 10) as usize;
            walls[idx] |= EXIT_W;
            walls[idx] ^= WALL_W;
            (n - p - 1, 0, EXIT_W)
        }
        2 => {
            // East exit
            let idx = ((n - 1) + (n - 1 - p) * 10) as usize;
            walls[idx] |= EXIT_E;
            walls[idx] ^= WALL_E;
            (n - 1 - p, n - 1, EXIT_E)
        }
        3 => {
            // North exit
            let idx = p as usize;
            walls[idx] |= EXIT_N;
            walls[idx] ^= WALL_N;
            (0, p, EXIT_N)
        }
        _ => (-1, -1, 0),
    }
}

/// Decode an entity position byte. Returns (row, col) in binary's coordinate system.
fn decode_entity(byte: u8, n: i32, flip: bool) -> (i32, i32) {
    let col_raw = (byte & 0x0F) as i32;
    let row_raw = ((byte >> 4) & 0x0F) as i32;
    if flip {
        (n - row_raw - 1, col_raw)
    } else {
        (col_raw, row_raw)
    }
}

pub fn parse_sublevel(data: &[u8], offset: usize, hdr: &Header) -> Result<Level> {
    let n = hdr.grid_size;
    let end = offset + hdr.bytes_per_sub as usize;
    if end > data.len() {
        return Err(MummyMazeError::Parse(format!(
            "sublevel data too short: need {} bytes at offset {}, have {}",
            hdr.bytes_per_sub, offset, data.len()
        )));
    }

    let mut out = Level {
        grid_size: n,
        flip: hdr.flip,
        walls: [0u32; MAX_GRID * MAX_GRID],
        player_row: 0,
        player_col: 0,
        mummy1_row: 0,
        mummy1_col: 0,
        mummy2_row: 99,
        mummy2_col: 99,
        scorpion_row: 99,
        scorpion_col: 99,
        trap1_row: 99,
        trap1_col: 99,
        trap2_row: 99,
        trap2_col: 99,
        trap_count: hdr.trap_count,
        has_mummy2: hdr.mummy_count >= 2,
        has_scorpion: hdr.scorpion > 0,
        gate_row: 99,
        gate_col: 99,
        has_gate: hdr.key_gate > 0,
        key_row: 99,
        key_col: 99,
        exit_row: -1,
        exit_col: -1,
        exit_mask: 0,
    };

    // Set border walls
    for col in 0..n {
        for row in 0..n {
            let w = &mut out.walls[(col + row * 10) as usize];
            if row == 0 {
                *w |= WALL_N;
            }
            if row == n - 1 {
                *w |= WALL_S;
            }
            if col == 0 {
                *w |= WALL_W;
            }
            if col == n - 1 {
                *w |= WALL_E;
            }
        }
    }

    let mut pos = offset;

    // Load walls
    if hdr.flip {
        load_walls_flip1(&mut out.walls, data, &mut pos, n);
    } else {
        load_walls_flip0(&mut out.walls, data, &mut pos, n);
    }

    // Load exit
    let (exit_row, exit_col, exit_mask) = if hdr.flip {
        load_exit_flip1(&mut out.walls, data, &mut pos, n)
    } else {
        load_exit_flip0(&mut out.walls, data, &mut pos, n)
    };
    out.exit_row = exit_row;
    out.exit_col = exit_col;
    out.exit_mask = exit_mask;

    // Player
    let (pr, pc) = decode_entity(data[pos], n, hdr.flip);
    pos += 1;
    out.player_row = pr;
    out.player_col = pc;

    // Mummy 1
    let (mr, mc) = decode_entity(data[pos], n, hdr.flip);
    pos += 1;
    out.mummy1_row = mr;
    out.mummy1_col = mc;

    // Mummy 2 (if present)
    if hdr.mummy_count >= 2 {
        let (mr, mc) = decode_entity(data[pos], n, hdr.flip);
        pos += 1;
        out.mummy2_row = mr;
        out.mummy2_col = mc;
    }

    // Scorpion (if present) — read BEFORE traps, matching binary byte order
    if hdr.scorpion > 0 {
        let (sr, sc) = decode_entity(data[pos], n, hdr.flip);
        pos += 1;
        out.scorpion_row = sr;
        out.scorpion_col = sc;
    }

    // Traps
    if hdr.trap_count >= 1 {
        let (tr, tc) = decode_entity(data[pos], n, hdr.flip);
        pos += 1;
        out.trap1_row = tr;
        out.trap1_col = tc;
    }
    if hdr.trap_count >= 2 {
        let (tr, tc) = decode_entity(data[pos], n, hdr.flip);
        pos += 1;
        out.trap2_row = tr;
        out.trap2_col = tc;
    }

    // Gate + key (if present) — comes LAST after scorpion and traps
    if hdr.key_gate > 0 {
        let (gr, gc) = decode_entity(data[pos], n, hdr.flip);
        pos += 1;
        out.gate_row = gr;
        out.gate_col = gc;

        let (kr, kc) = decode_entity(data[pos], n, hdr.flip);
        pos += 1;
        out.key_row = kr;
        out.key_col = kc;
    }

    // Sanity check: we should have consumed exactly bytes_per_sub bytes
    let consumed = pos - offset;
    if consumed != hdr.bytes_per_sub as usize {
        return Err(MummyMazeError::Parse(format!(
            "consumed {} bytes but expected {}",
            consumed, hdr.bytes_per_sub
        )));
    }

    Ok(out)
}

/// Remap wall/exit bits according to a mapping table.
/// Each entry `(from, to)` means: if bit `from` is set in `w`, set bit `to` in output.
/// Bits not mentioned in any `from` are preserved as-is.
fn remap_bits(w: u32, mapping: &[(u32, u32)]) -> u32 {
    let mut mentioned = 0u32;
    for &(from, _) in mapping {
        mentioned |= from;
    }
    let mut out = w & !mentioned; // preserve unmapped bits
    for &(from, to) in mapping {
        if w & from != 0 {
            out |= to;
        }
    }
    out
}

impl Level {
    /// Construct a `Level` from edge-array walls and entity positions.
    ///
    /// This accepts the same representation the Python parser produces
    /// (h_walls/v_walls edge arrays), converting to internal cell bitmasks.
    ///
    /// # Arguments
    /// * `h_walls` — `(n+1) * n` bools, row-major. `h_walls[r * n + c]` = wall
    ///   on top edge of cell `(r, c)`.
    /// * `v_walls` — `n * (n+1)` bools, row-major. `v_walls[r * (n+1) + c]` =
    ///   wall on left edge of cell `(r, c)`.
    /// * `exit_side` — `"N"`, `"S"`, `"E"`, or `"W"`.
    /// * Entity positions are `(row, col)`. Use `None` for absent entities.
    /// * `traps` — up to 2 trap positions.
    pub fn from_edges(
        grid_size: i32,
        flip: bool,
        h_walls: &[bool],
        v_walls: &[bool],
        exit_side: &str,
        exit_pos: i32,
        player: (i32, i32),
        mummy1: (i32, i32),
        mummy2: Option<(i32, i32)>,
        scorpion: Option<(i32, i32)>,
        traps: &[(i32, i32)],
        gate: Option<(i32, i32)>,
        key: Option<(i32, i32)>,
    ) -> Level {
        let n = grid_size;
        let mut walls = [0u32; MAX_GRID * MAX_GRID];

        // h_walls[r][c] → WALL_N on (r, c) and WALL_S on (r-1, c)
        for r in 0..=n {
            for c in 0..n {
                if h_walls[(r * n + c) as usize] {
                    if r < n {
                        walls[(c + r * 10) as usize] |= WALL_N;
                    }
                    if r > 0 {
                        walls[(c + (r - 1) * 10) as usize] |= WALL_S;
                    }
                }
            }
        }

        // v_walls[r][c] → WALL_W on (r, c) and WALL_E on (r, c-1)
        for r in 0..n {
            for c in 0..=n {
                if v_walls[(r * (n + 1) + c) as usize] {
                    if c < n {
                        walls[(c + r * 10) as usize] |= WALL_W;
                    }
                    if c > 0 {
                        walls[((c - 1) + r * 10) as usize] |= WALL_E;
                    }
                }
            }
        }

        // Exit: clear border wall, set exit flag
        let (exit_row, exit_col, exit_mask) = match exit_side {
            "N" => {
                let idx = exit_pos as usize;
                walls[idx] &= !WALL_N;
                walls[idx] |= EXIT_N;
                (0, exit_pos, EXIT_N)
            }
            "S" => {
                let idx = (exit_pos + (n - 1) * 10) as usize;
                walls[idx] &= !WALL_S;
                walls[idx] |= EXIT_S;
                (n - 1, exit_pos, EXIT_S)
            }
            "W" => {
                let idx = (exit_pos * 10) as usize;
                walls[idx] &= !WALL_W;
                walls[idx] |= EXIT_W;
                (exit_pos, 0, EXIT_W)
            }
            "E" => {
                let idx = ((n - 1) + exit_pos * 10) as usize;
                walls[idx] &= !WALL_E;
                walls[idx] |= EXIT_E;
                (exit_pos, n - 1, EXIT_E)
            }
            _ => (0, 0, 0),
        };

        let (m2r, m2c, has_mummy2) = match mummy2 {
            Some((r, c)) => (r, c, true),
            None => (99, 99, false),
        };
        let (sr, sc, has_scorpion) = match scorpion {
            Some((r, c)) => (r, c, true),
            None => (99, 99, false),
        };
        let (t1r, t1c) = traps.first().copied().unwrap_or((99, 99));
        let (t2r, t2c) = traps.get(1).copied().unwrap_or((99, 99));
        let (gr, gc, kr, kc, has_gate) = match (gate, key) {
            (Some((gr, gc)), Some((kr, kc))) => (gr, gc, kr, kc, true),
            _ => (99, 99, 99, 99, false),
        };

        Level {
            grid_size,
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
            trap_count: traps.len() as i32,
            gate_row: gr,
            gate_col: gc,
            has_gate,
            key_row: kr,
            key_col: kc,
            exit_row,
            exit_col,
            exit_mask,
        }
    }

    /// Convert internal wall bitmasks to edge arrays.
    ///
    /// Returns `(h_walls, v_walls)`:
    /// * `h_walls` — `(n+1) * n` bools, `h_walls[r * n + c]` = wall on top edge of `(r, c)`.
    /// * `v_walls` — `n * (n+1)` bools, `v_walls[r * (n+1) + c]` = wall on left edge of `(r, c)`.
    pub fn to_edges(&self) -> (Vec<bool>, Vec<bool>) {
        let n = self.grid_size;
        let mut h_walls = vec![false; ((n + 1) * n) as usize];
        let mut v_walls = vec![false; (n * (n + 1)) as usize];

        for r in 0..n {
            for c in 0..n {
                let w = self.walls[(c + r * 10) as usize];
                if w & WALL_N != 0 {
                    h_walls[(r * n + c) as usize] = true;
                }
                if w & WALL_S != 0 {
                    h_walls[((r + 1) * n + c) as usize] = true;
                }
                if w & WALL_W != 0 {
                    v_walls[(r * (n + 1) + c) as usize] = true;
                }
                if w & WALL_E != 0 {
                    v_walls[(r * (n + 1) + (c + 1)) as usize] = true;
                }
            }
        }
        (h_walls, v_walls)
    }

    /// Exit side as a string: `"N"`, `"S"`, `"E"`, or `"W"`.
    pub fn exit_side_str(&self) -> &'static str {
        if self.exit_mask & EXIT_N != 0 { "N" }
        else if self.exit_mask & EXIT_S != 0 { "S" }
        else if self.exit_mask & EXIT_W != 0 { "W" }
        else { "E" }
    }

    /// Position along the exit side (column for N/S, row for E/W).
    pub fn exit_pos(&self) -> i32 {
        match self.exit_side_str() {
            "N" | "S" => self.exit_col,
            _ => self.exit_row,
        }
    }

    /// Hash all gameplay-relevant fields to a u64 fingerprint.
    pub fn fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::hash::DefaultHasher::new();
        self.grid_size.hash(&mut h);
        self.walls.hash(&mut h);
        self.player_row.hash(&mut h);
        self.player_col.hash(&mut h);
        self.mummy1_row.hash(&mut h);
        self.mummy1_col.hash(&mut h);
        self.mummy2_row.hash(&mut h);
        self.mummy2_col.hash(&mut h);
        self.has_mummy2.hash(&mut h);
        self.scorpion_row.hash(&mut h);
        self.scorpion_col.hash(&mut h);
        self.has_scorpion.hash(&mut h);
        self.trap1_row.hash(&mut h);
        self.trap1_col.hash(&mut h);
        self.trap2_row.hash(&mut h);
        self.trap2_col.hash(&mut h);
        self.trap_count.hash(&mut h);
        self.gate_row.hash(&mut h);
        self.gate_col.hash(&mut h);
        self.has_gate.hash(&mut h);
        self.key_row.hash(&mut h);
        self.key_col.hash(&mut h);
        self.exit_row.hash(&mut h);
        self.exit_col.hash(&mut h);
        self.exit_mask.hash(&mut h);
        self.flip.hash(&mut h);
        h.finish()
    }

    /// Canonical fingerprint under dihedral symmetry (rotations + reflections).
    ///
    /// Gate-free levels use all 8 D4 symmetries. Gate levels use the 4 that
    /// preserve the gate's vertical (E/W) orientation: identity, rot180,
    /// h_mirror, v_mirror.
    pub fn canonical_fingerprint(&self) -> u64 {
        let syms: &[u8] = if self.has_gate {
            // rot180 and v_mirror also preserve vertical gate orientation
            // (with a cell shift handled in apply_dihedral).
            &[0, 2, 4, 5]
        } else {
            &[0, 1, 2, 3, 4, 5, 6, 7]
        };
        syms.iter()
            .map(|&s| self.apply_dihedral(s).fingerprint())
            .min()
            .unwrap()
    }

    /// Apply a dihedral symmetry transform.
    pub fn apply_dihedral(&self, sym: u8) -> Level {
        let n = self.grid_size;

        // Each transform: (r,c) → (r',c'), wall bits remapped
        //   rot90cw:  N→E, E→S, S→W, W→N  coord: (r,c)→(c, n-1-r)
        //   rot180:   N→S, S→N, E→W, W→E  coord: (r,c)→(n-1-r, n-1-c)
        //   rot270cw: N→W, W→S, S→E, E→N  coord: (r,c)→(n-1-c, r)
        //   h_mirror: N↔S                  coord: (r,c)→(n-1-r, c)
        //   v_mirror: W↔E                  coord: (r,c)→(r, n-1-c)
        //   transpose: N↔W, S↔E            coord: (r,c)→(c, r)
        //   anti_transpose: N↔E, S↔W       coord: (r,c)→(n-1-c, n-1-r)

        let coord = |r: i32, c: i32| -> (i32, i32) {
            match sym {
                0 => (r, c),
                1 => (c, n - 1 - r),
                2 => (n - 1 - r, n - 1 - c),
                3 => (n - 1 - c, r),
                4 => (n - 1 - r, c),
                5 => (r, n - 1 - c),
                6 => (c, r),
                7 => (n - 1 - c, n - 1 - r),
                _ => unreachable!(),
            }
        };

        let remap_walls = |w: u32| -> u32 {
            match sym {
                0 => w,
                1 => remap_bits(w, &[
                    (WALL_N, WALL_E), (WALL_E, WALL_S), (WALL_S, WALL_W), (WALL_W, WALL_N),
                    (EXIT_N, EXIT_E), (EXIT_E, EXIT_S), (EXIT_S, EXIT_W), (EXIT_W, EXIT_N),
                ]),
                2 => remap_bits(w, &[
                    (WALL_N, WALL_S), (WALL_S, WALL_N), (WALL_E, WALL_W), (WALL_W, WALL_E),
                    (EXIT_N, EXIT_S), (EXIT_S, EXIT_N), (EXIT_E, EXIT_W), (EXIT_W, EXIT_E),
                ]),
                3 => remap_bits(w, &[
                    (WALL_N, WALL_W), (WALL_W, WALL_S), (WALL_S, WALL_E), (WALL_E, WALL_N),
                    (EXIT_N, EXIT_W), (EXIT_W, EXIT_S), (EXIT_S, EXIT_E), (EXIT_E, EXIT_N),
                ]),
                4 => remap_bits(w, &[
                    (WALL_N, WALL_S), (WALL_S, WALL_N),
                    (EXIT_N, EXIT_S), (EXIT_S, EXIT_N),
                ]),
                5 => remap_bits(w, &[
                    (WALL_W, WALL_E), (WALL_E, WALL_W),
                    (EXIT_W, EXIT_E), (EXIT_E, EXIT_W),
                ]),
                6 => remap_bits(w, &[
                    (WALL_N, WALL_W), (WALL_W, WALL_N), (WALL_S, WALL_E), (WALL_E, WALL_S),
                    (EXIT_N, EXIT_W), (EXIT_W, EXIT_N), (EXIT_S, EXIT_E), (EXIT_E, EXIT_S),
                ]),
                7 => remap_bits(w, &[
                    (WALL_N, WALL_E), (WALL_E, WALL_N), (WALL_S, WALL_W), (WALL_W, WALL_S),
                    (EXIT_N, EXIT_E), (EXIT_E, EXIT_N), (EXIT_S, EXIT_W), (EXIT_W, EXIT_S),
                ]),
                _ => unreachable!(),
            }
        };

        let flip_toggle = matches!(sym, 1 | 3 | 6 | 7);

        let mut walls = [0u32; MAX_GRID * MAX_GRID];
        for r in 0..n {
            for c in 0..n {
                let (nr, nc) = coord(r, c);
                walls[(nc + nr * 10) as usize] = remap_walls(self.walls[(c + r * 10) as usize]);
            }
        }

        let remap_exit = |m: u32| -> u32 { remap_walls(m) };

        let xf = |r: i32, c: i32, exists: bool| -> (i32, i32) {
            if !exists || r >= 90 { return (r, c); }
            coord(r, c)
        };

        let (pr, pc) = xf(self.player_row, self.player_col, true);
        let (m1r, m1c) = xf(self.mummy1_row, self.mummy1_col, true);
        let (m2r, m2c) = xf(self.mummy2_row, self.mummy2_col, self.has_mummy2);
        let (sr, sc) = xf(self.scorpion_row, self.scorpion_col, self.has_scorpion);
        let (t1r, t1c) = xf(self.trap1_row, self.trap1_col, self.trap_count >= 1);
        let (t2r, t2c) = xf(self.trap2_row, self.trap2_col, self.trap_count >= 2);
        let (gr, mut gc) = xf(self.gate_row, self.gate_col, self.has_gate);
        // The gate is a vertical barrier on the east edge of its cell.
        // Transforms that flip E↔W (rot180, v_mirror) move the barrier to
        // the west edge; shift one cell left to keep it as an east edge.
        if self.has_gate && matches!(sym, 2 | 5) {
            gc -= 1;
        }
        let (kr, kc) = xf(self.key_row, self.key_col, self.has_gate);
        let (er, ec) = coord(self.exit_row, self.exit_col);

        Level {
            grid_size: n,
            flip: self.flip ^ flip_toggle,
            walls,
            player_row: pr, player_col: pc,
            mummy1_row: m1r, mummy1_col: m1c,
            mummy2_row: m2r, mummy2_col: m2c,
            has_mummy2: self.has_mummy2,
            scorpion_row: sr, scorpion_col: sc,
            has_scorpion: self.has_scorpion,
            trap1_row: t1r, trap1_col: t1c,
            trap2_row: t2r, trap2_col: t2c,
            trap_count: self.trap_count,
            gate_row: gr, gate_col: gc,
            has_gate: self.has_gate,
            key_row: kr, key_col: kc,
            exit_row: er, exit_col: ec,
            exit_mask: remap_exit(self.exit_mask),
        }
    }
}

/// Parse an entire .dat file, returning header and all sublevels.
pub fn parse_file(path: &Path) -> Result<(Header, Vec<Level>)> {
    let data = std::fs::read(path)?;
    if data.len() < 6 {
        return Err(MummyMazeError::Parse("file too short for header".into()));
    }

    let hdr = parse_header(&data[..6])?;

    let mut levels = Vec::with_capacity(hdr.num_sublevels as usize);
    let mut offset = 6usize;
    for _ in 0..hdr.num_sublevels {
        let level = parse_sublevel(&data, offset, &hdr)?;
        offset += hdr.bytes_per_sub as usize;
        levels.push(level);
    }

    Ok((hdr, levels))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_safety_invariant_under_rotation() {
        use crate::{graph, metrics, solver};

        let mazes = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap().parent().unwrap().join("mazes");

        let (_, levels_a) = parse_file(&mazes.join("B-53.dat")).unwrap();
        let (_, levels_b) = parse_file(&mazes.join("B-55.dat")).unwrap();
        let a = &levels_a[85];
        let b = &levels_b[85];

        let graph_a = graph::build_graph(a);
        let solve_a = solver::solve(a);
        let metrics_a = metrics::compute(&graph_a, a, &solve_a);

        let graph_b = graph::build_graph(b);
        let solve_b = solver::solve(b);
        let metrics_b = metrics::compute(&graph_b, b, &solve_b);

        assert_eq!(solve_a.moves, solve_b.moves);
        assert_eq!(metrics_a.n_optimal_solutions, metrics_b.n_optimal_solutions);
        assert_eq!(metrics_a.dead_end_ratio, metrics_b.dead_end_ratio);

        let safety_a = metrics_a.path_safety.unwrap();
        let safety_b = metrics_b.path_safety.unwrap();
        assert!(
            (safety_a - safety_b).abs() < 1e-10,
            "path_safety should be identical for rotations: {safety_a} vs {safety_b}"
        );
    }

    #[test]
    fn dihedral_rot90_matches_known_pair() {
        let mazes = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap().parent().unwrap().join("mazes");

        let (_, levels_a) = parse_file(&mazes.join("B-53.dat")).unwrap();
        let (_, levels_b) = parse_file(&mazes.join("B-55.dat")).unwrap();
        let a = &levels_a[85];
        let b = &levels_b[85];

        // B-55 sub 85 should be a 90° CW rotation (sym=1) of B-53 sub 85
        let rotated = a.apply_dihedral(1);
        assert_eq!(rotated.walls, b.walls);
        assert_eq!(rotated.flip, b.flip);
        assert_eq!((rotated.player_row, rotated.player_col), (b.player_row, b.player_col));
        assert_eq!((rotated.mummy1_row, rotated.mummy1_col), (b.mummy1_row, b.mummy1_col));
        assert_eq!((rotated.exit_row, rotated.exit_col, rotated.exit_mask),
                   (b.exit_row, b.exit_col, b.exit_mask));
    }

    #[test]
    fn gate_transforms_preserve_solvability() {
        // All 4 valid gate symmetries (0, 2, 4, 5) must produce levels
        // that solve with the same optimal move count.
        use crate::solver;

        let mazes = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap().parent().unwrap().join("mazes");

        let mut tested = 0;
        for entry in std::fs::read_dir(&mazes).unwrap() {
            let path = entry.unwrap().path();
            if !path.extension().is_some_and(|e| e == "dat") { continue; }
            let Ok((_, levels)) = parse_file(&path) else { continue };
            for (sub, lev) in levels.iter().enumerate() {
                if !lev.has_gate { continue; }
                let base_sol = solver::solve(lev).moves;
                if base_sol.is_none() { continue; }
                for &sym in &[2, 4, 5] {
                    let transformed = lev.apply_dihedral(sym);
                    let t_sol = solver::solve(&transformed).moves;
                    assert_eq!(
                        base_sol, t_sol,
                        "sym={sym} changed solution for {}:{sub} (base={base_sol:?}, got={t_sol:?})",
                        path.file_stem().unwrap().to_str().unwrap(),
                    );
                }
                tested += 1;
            }
        }
        assert!(tested > 100, "expected to test >100 gate levels, got {tested}");
    }

    #[test]
    fn gate_double_transform_is_identity() {
        // v_mirror and rot180 applied twice should return to identity for gate levels.
        let mazes = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap().parent().unwrap().join("mazes");

        let mut tested = 0;
        for entry in std::fs::read_dir(&mazes).unwrap() {
            let path = entry.unwrap().path();
            if !path.extension().is_some_and(|e| e == "dat") { continue; }
            let Ok((_, levels)) = parse_file(&path) else { continue };
            for lev in &levels {
                if !lev.has_gate { continue; }
                // v_mirror twice = identity
                let vv = lev.apply_dihedral(5).apply_dihedral(5);
                assert_eq!(lev.gate_row, vv.gate_row);
                assert_eq!(lev.gate_col, vv.gate_col);
                assert_eq!(lev.walls, vv.walls);
                assert_eq!(lev.player_row, vv.player_row);
                assert_eq!(lev.player_col, vv.player_col);
                assert_eq!(lev.key_row, vv.key_row);
                assert_eq!(lev.key_col, vv.key_col);

                // rot180 twice = identity
                let rr = lev.apply_dihedral(2).apply_dihedral(2);
                assert_eq!(lev.gate_row, rr.gate_row);
                assert_eq!(lev.gate_col, rr.gate_col);
                assert_eq!(lev.walls, rr.walls);
                assert_eq!(lev.player_row, rr.player_row);
                assert_eq!(lev.player_col, rr.player_col);

                // h_mirror twice = identity (was already working)
                let hh = lev.apply_dihedral(4).apply_dihedral(4);
                assert_eq!(lev.gate_row, hh.gate_row);
                assert_eq!(lev.gate_col, hh.gate_col);
                assert_eq!(lev.walls, hh.walls);

                tested += 1;
            }
        }
        assert!(tested > 100, "expected to test >100 gate levels, got {tested}");
    }

    #[test]
    fn dihedral_double_application_is_identity() {
        let mazes = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap().parent().unwrap().join("mazes");
        let (_, levels) = parse_file(&mazes.join("B-53.dat")).unwrap();
        let lev = &levels[85];

        // rot90 applied 4 times = identity
        let mut l = lev.clone();
        for _ in 0..4 { l = l.apply_dihedral(1); }
        assert_eq!(l.walls, lev.walls);
        assert_eq!(l.flip, lev.flip);
        assert_eq!((l.player_row, l.player_col), (lev.player_row, lev.player_col));

        // h_mirror applied twice = identity
        let hh = lev.apply_dihedral(4).apply_dihedral(4);
        assert_eq!(hh.walls, lev.walls);
        assert_eq!((hh.player_row, hh.player_col), (lev.player_row, lev.player_col));

        // transpose applied twice = identity
        let tt = lev.apply_dihedral(6).apply_dihedral(6);
        assert_eq!(tt.walls, lev.walls);
        assert_eq!((tt.player_row, tt.player_col), (lev.player_row, lev.player_col));
    }

}
