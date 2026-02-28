//! .dat file parser — direct port of csolver/src/parse.c
//!
//! Wall encoding: `walls[col + row * 10]` with bit flags:
//!   bit 0 (1) = WALL_W, bit 1 (2) = WALL_E,
//!   bit 2 (4) = WALL_S, bit 3 (8) = WALL_N
//! Exit flags in upper nibble: EXIT_W=0x10, EXIT_E=0x20, EXIT_S=0x40, EXIT_N=0x80

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

#[derive(Debug, Clone)]
pub struct Level {
    pub grid_size: i32,
    pub flip: bool,
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
