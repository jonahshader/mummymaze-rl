use eframe::egui;
use egui::{Color32, FontId, Painter, Pos2, Rect, Stroke, Vec2};
use mummymaze::game::State;
use mummymaze::parse::{Level, EXIT_E, EXIT_N, EXIT_S, EXIT_W, WALL_E, WALL_N, WALL_S, WALL_W};

const BG_COLOR: Color32 = Color32::from_rgb(40, 40, 50);
const CELL_COLOR: Color32 = Color32::from_rgb(60, 60, 75);
const WALL_COLOR: Color32 = Color32::from_rgb(200, 200, 210);
const GATE_CLOSED_COLOR: Color32 = Color32::from_rgb(200, 50, 50);
const GATE_OPEN_COLOR: Color32 = Color32::from_rgb(50, 200, 50);
const KEY_COLOR: Color32 = Color32::from_rgb(230, 200, 50);
const TRAP_COLOR: Color32 = Color32::from_rgb(220, 120, 40);
const EXIT_COLOR: Color32 = Color32::from_rgb(50, 220, 50);
const PLAYER_COLOR: Color32 = Color32::from_rgb(80, 140, 255);
const MUMMY_COLOR: Color32 = Color32::from_rgb(220, 220, 220);
const SCORPION_COLOR: Color32 = Color32::from_rgb(230, 160, 40);

const WALL_THICKNESS: f32 = 3.0;
const BORDER_THICKNESS: f32 = 4.0;

/// Compute cell_size and top-left origin for a grid drawn within the given rect.
fn maze_geometry(rect: Rect, grid_size: i32) -> (f32, Pos2) {
    let gsf = grid_size as f32;
    let cell_size = (rect.width().min(rect.height())) / gsf;
    let origin = Pos2::new(
        rect.left() + (rect.width() - cell_size * gsf) / 2.0,
        rect.top() + (rect.height() - cell_size * gsf) / 2.0,
    );
    (cell_size, origin)
}

/// Draw a maze level with the given state into a painter within the specified rect.
pub fn draw_maze_state(painter: &Painter, rect: Rect, lev: &Level, state: &State) {
    let gs = lev.grid_size as usize;
    let gsf = gs as f32;
    let (cell_size, origin) = maze_geometry(rect, lev.grid_size);

    let maze_rect = Rect::from_min_size(origin, Vec2::new(cell_size * gsf, cell_size * gsf));
    painter.rect_filled(maze_rect, 0.0, BG_COLOR);

    // Cell backgrounds
    for row in 0..gs {
        for col in 0..gs {
            let x = origin.x + col as f32 * cell_size;
            let y = origin.y + row as f32 * cell_size;
            let cell_rect = Rect::from_min_size(
                Pos2::new(x + 1.0, y + 1.0),
                Vec2::new(cell_size - 2.0, cell_size - 2.0),
            );
            painter.rect_filled(cell_rect, 2.0, CELL_COLOR);
        }
    }

    // Traps
    if lev.trap_count >= 1 {
        draw_trap(painter, origin, cell_size, lev.trap1_row, lev.trap1_col);
    }
    if lev.trap_count >= 2 {
        draw_trap(painter, origin, cell_size, lev.trap2_row, lev.trap2_col);
    }

    // Key
    if lev.has_gate {
        let cx = origin.x + lev.key_col as f32 * cell_size + cell_size / 2.0;
        let cy = origin.y + lev.key_row as f32 * cell_size + cell_size / 2.0;
        painter.circle_filled(Pos2::new(cx, cy), cell_size * 0.15, KEY_COLOR);
    }

    draw_walls(painter, origin, cell_size, lev, gs);

    // Border
    painter.rect_stroke(
        maze_rect,
        0.0,
        Stroke::new(BORDER_THICKNESS, WALL_COLOR),
        egui::StrokeKind::Outside,
    );

    draw_exit(painter, origin, cell_size, lev, gs);

    if lev.has_gate {
        draw_gate(painter, origin, cell_size, lev, state);
    }

    draw_entities(painter, origin, cell_size, lev, state);
}

fn draw_walls(painter: &Painter, origin: Pos2, cell_size: f32, lev: &Level, gs: usize) {
    let stroke = Stroke::new(WALL_THICKNESS, WALL_COLOR);
    // (bit_mask, boundary: row/col must satisfy, is_horizontal, offset from cell origin)
    // For horizontal walls: (dx_start, dy, dx_end, dy)
    // For vertical walls:   (dx, dy_start, dx, dy_end)
    let wall_defs: [(u32, bool, [f32; 4]); 4] = [
        // North: horizontal at top edge, skip border (row > 0)
        (WALL_N, true, [0.0, 0.0, cell_size, 0.0]),
        // South: horizontal at bottom edge, skip border (row < gs-1)
        (WALL_S, false, [0.0, cell_size, cell_size, cell_size]),
        // West: vertical at left edge, skip border (col > 0)
        (WALL_W, true, [0.0, 0.0, 0.0, cell_size]),
        // East: vertical at right edge, skip border (col < gs-1)
        (WALL_E, false, [cell_size, 0.0, cell_size, cell_size]),
    ];

    for row in 0..gs {
        for col in 0..gs {
            let w = lev.walls[col + row * 10];
            let x = origin.x + col as f32 * cell_size;
            let y = origin.y + row as f32 * cell_size;

            for &(bit, check_low, offsets) in &wall_defs {
                if w & bit == 0 {
                    continue;
                }
                // Skip border walls: N/W need idx > 0, S/E need idx < gs-1
                let skip = match bit {
                    WALL_N => row == 0,
                    WALL_S => row == gs - 1,
                    WALL_W => col == 0,
                    WALL_E => col == gs - 1,
                    _ => false,
                };
                if skip {
                    continue;
                }
                let _ = check_low; // used only to distinguish in the tuple
                painter.line_segment(
                    [
                        Pos2::new(x + offsets[0], y + offsets[1]),
                        Pos2::new(x + offsets[2], y + offsets[3]),
                    ],
                    stroke,
                );
            }
        }
    }
}

fn draw_exit(painter: &Painter, origin: Pos2, cell_size: f32, lev: &Level, gs: usize) {
    let er = lev.exit_row as f32;
    let ec = lev.exit_col as f32;
    let mask = lev.exit_mask;
    let gap = cell_size * 0.6;
    let stroke = Stroke::new(BORDER_THICKNESS + 1.0, EXIT_COLOR);
    let gsf = gs as f32;

    // (bit, is_horizontal, center_along_perpendicular, fixed_edge_coord)
    let exits: [(u32, bool, f32, f32); 4] = [
        (EXIT_N, true, origin.x + ec * cell_size + cell_size / 2.0, origin.y),
        (EXIT_S, true, origin.x + ec * cell_size + cell_size / 2.0, origin.y + gsf * cell_size),
        (EXIT_W, false, origin.y + er * cell_size + cell_size / 2.0, origin.x),
        (EXIT_E, false, origin.y + er * cell_size + cell_size / 2.0, origin.x + gsf * cell_size),
    ];

    for &(bit, is_horizontal, center, edge) in &exits {
        if mask & bit == 0 {
            continue;
        }
        let (p1, p2) = if is_horizontal {
            (Pos2::new(center - gap / 2.0, edge), Pos2::new(center + gap / 2.0, edge))
        } else {
            (Pos2::new(edge, center - gap / 2.0), Pos2::new(edge, center + gap / 2.0))
        };
        painter.line_segment([p1, p2], stroke);
    }
}

fn draw_gate(painter: &Painter, origin: Pos2, cell_size: f32, lev: &Level, state: &State) {
    let gr = lev.gate_row as f32;
    let gc = lev.gate_col as f32;
    let x = origin.x + (gc + 1.0) * cell_size;
    let y_top = origin.y + gr * cell_size + 2.0;
    let y_bot = origin.y + (gr + 1.0) * cell_size - 2.0;

    let color = if state.gate_open {
        GATE_CLOSED_COLOR // gate_open=true means blocking
    } else {
        GATE_OPEN_COLOR
    };
    painter.line_segment(
        [Pos2::new(x, y_top), Pos2::new(x, y_bot)],
        Stroke::new(4.0, color),
    );
}

fn draw_trap(painter: &Painter, origin: Pos2, cell_size: f32, tr: i32, tc: i32) {
    let x = origin.x + tc as f32 * cell_size;
    let y = origin.y + tr as f32 * cell_size;
    let inset = cell_size * 0.15;
    let trap_rect = Rect::from_min_size(
        Pos2::new(x + inset, y + inset),
        Vec2::new(cell_size - 2.0 * inset, cell_size - 2.0 * inset),
    );
    painter.rect_filled(trap_rect, 2.0, TRAP_COLOR.linear_multiply(0.3));
    painter.rect_stroke(
        trap_rect,
        2.0,
        Stroke::new(1.5, TRAP_COLOR),
        egui::StrokeKind::Outside,
    );
}

fn draw_entities(painter: &Painter, origin: Pos2, cell_size: f32, lev: &Level, state: &State) {
    let radius = cell_size * 0.3;
    let font = FontId::proportional(cell_size * 0.35);

    // Draw order: scorpion, mummy2, mummy1, player (player on top)
    let entities: [(bool, bool, i32, i32, Color32, &str); 4] = [
        (lev.has_scorpion, state.scorpion_alive, state.scorpion_row, state.scorpion_col, SCORPION_COLOR, "S"),
        (lev.has_mummy2, state.mummy2_alive, state.mummy2_row, state.mummy2_col, MUMMY_COLOR, "M"),
        (true, state.mummy1_alive, state.mummy1_row, state.mummy1_col, MUMMY_COLOR, "M"),
        (true, true, state.player_row, state.player_col, PLAYER_COLOR, "P"),
    ];

    for &(exists, alive, row, col, color, label) in &entities {
        if !exists || !alive || row >= 90 {
            continue;
        }
        let cx = origin.x + col as f32 * cell_size + cell_size / 2.0;
        let cy = origin.y + row as f32 * cell_size + cell_size / 2.0;
        painter.circle_filled(Pos2::new(cx, cy), radius, color);
        painter.text(
            Pos2::new(cx, cy),
            egui::Align2::CENTER_CENTER,
            label,
            font.clone(),
            Color32::BLACK,
        );
    }
}

/// Draw action probability bar charts on the player's cell and its neighbors.
///
/// Each action maps to a destination cell (N/S/E/W neighbor, or player cell for Wait).
/// On each cell we draw up to two bars side-by-side: left = BFS optimal (green),
/// right = agent probability (blue). Bar height ∝ value, anchored to cell bottom.
///
/// `agent_probs`: 5 floats (N=0, S=1, E=2, W=3, Wait=4), or None.
/// `bfs_mask`: bitmask of optimal actions (bit i = action i), or None.
pub fn draw_action_bars(
    painter: &Painter,
    rect: Rect,
    grid_size: i32,
    player_row: i32,
    player_col: i32,
    agent_probs: Option<&[f32; 5]>,
    bfs_mask: Option<u8>,
) {
    if agent_probs.is_none() && bfs_mask.is_none() {
        return;
    }

    let (cell_size, origin) = maze_geometry(rect, grid_size);

    // BFS: uniform probability over optimal actions
    let bfs_probs: [f32; 5] = if let Some(mask) = bfs_mask {
        let count = mask.count_ones() as f32;
        if count > 0.0 {
            std::array::from_fn(|i| if mask & (1 << i) != 0 { 1.0 / count } else { 0.0 })
        } else {
            [0.0; 5]
        }
    } else {
        [0.0; 5]
    };

    // Action → destination cell offset (row, col)
    let offsets: [(i32, i32); 5] = [
        (-1, 0), // N
        (1, 0),  // S
        (0, 1),  // E
        (0, -1), // W
        (0, 0),  // Wait
    ];

    let bar_color_bfs = Color32::from_rgba_unmultiplied(80, 220, 80, 130);
    let bar_color_agent = Color32::from_rgba_unmultiplied(80, 140, 255, 130);
    let max_height = cell_size * 0.85;
    let inset = cell_size * 0.15;
    let bar_width = (cell_size - 2.0 * inset) / 2.0 - 1.0; // two bars with 2px gap

    let has_bfs = bfs_mask.is_some();
    let has_agent = agent_probs.is_some();
    let agent = agent_probs.unwrap_or(&[0.0; 5]);

    for i in 0..5 {
        let bfs_p = bfs_probs[i];
        let agent_p = agent[i];
        if bfs_p < 0.01 && agent_p < 0.01 {
            continue;
        }

        let dr = offsets[i].0;
        let dc = offsets[i].1;
        let r = player_row + dr;
        let c = player_col + dc;

        // Skip out-of-bounds cells
        if r < 0 || r >= grid_size || c < 0 || c >= grid_size {
            continue;
        }

        let cell_x = origin.x + c as f32 * cell_size;
        let cell_y = origin.y + r as f32 * cell_size;
        let cell_bottom = cell_y + cell_size - inset;

        if has_bfs && has_agent {
            // Two bars side-by-side
            let left_x = cell_x + inset;
            let right_x = left_x + bar_width + 2.0;

            if bfs_p >= 0.01 {
                let h = max_height * bfs_p;
                let bar = Rect::from_min_max(
                    Pos2::new(left_x, cell_bottom - h),
                    Pos2::new(left_x + bar_width, cell_bottom),
                );
                painter.rect_filled(bar, 2.0, bar_color_bfs);
            }
            if agent_p >= 0.01 {
                let h = max_height * agent_p;
                let bar = Rect::from_min_max(
                    Pos2::new(right_x, cell_bottom - h),
                    Pos2::new(right_x + bar_width, cell_bottom),
                );
                painter.rect_filled(bar, 2.0, bar_color_agent);
            }
        } else {
            // Single centered bar
            let color = if has_bfs {
                bar_color_bfs
            } else {
                bar_color_agent
            };
            let p = if has_bfs { bfs_p } else { agent_p };
            if p >= 0.01 {
                let w = bar_width * 1.5;
                let x = cell_x + (cell_size - w) / 2.0;
                let h = max_height * p;
                let bar = Rect::from_min_max(
                    Pos2::new(x, cell_bottom - h),
                    Pos2::new(x + w, cell_bottom),
                );
                painter.rect_filled(bar, 2.0, color);
            }
        }
    }
}

/// Compute the ideal square size for a maze preview.
pub fn maze_preferred_size(available: Vec2) -> Vec2 {
    let side = available.x.min(available.y);
    Vec2::new(side, side)
}
