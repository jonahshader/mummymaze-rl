use crate::data::DataStore;
use eframe::egui;
use egui::Ui;
use egui_plot::{Line, Plot, PlotPoints, Points};

/// Precomputed point data for the scatter plot.
struct ScatterPoint {
    row_idx: usize,
    win_prob: f64,
    accuracy: f64,
    grid_size: i32,
}

/// Draw the training tab content. Returns Some(row_idx) if a point was clicked.
pub fn draw_training_panel(ui: &mut Ui, store: &DataStore) -> Option<usize> {
    let has_training = store
        .training_metrics
        .as_ref()
        .is_some_and(|tm| !tm.levels.is_empty());

    if !has_training {
        ui.centered_and_justified(|ui: &mut Ui| {
            ui.heading("No training data — write level_metrics.json to see results");
        });
        return None;
    }

    let tm = store.training_metrics.as_ref().unwrap();

    // Header
    ui.horizontal(|ui: &mut Ui| {
        ui.label(format!("Run: {}", tm.run_id));
        ui.separator();
        ui.label(format!("Step: {}", tm.step));
        ui.separator();
        ui.label(format!("{} levels", tm.levels.len()));
    });
    ui.separator();

    // Build scatter data
    let mut points: Vec<ScatterPoint> = Vec::new();
    for (i, row) in store.rows.iter().enumerate() {
        let Some(analysis) = &row.analysis else {
            continue;
        };
        let Some(metric) = tm.get(&row.file_stem, row.sublevel) else {
            continue;
        };
        points.push(ScatterPoint {
            row_idx: i,
            win_prob: analysis.win_prob,
            accuracy: metric.accuracy,
            grid_size: row.level.grid_size,
        });
    }

    let mut clicked_row = None;

    // Scatter plot
    let plot = Plot::new("training_scatter")
        .x_axis_label("Random Win%")
        .y_axis_label("Agent Accuracy")
        .data_aspect(1.0)
        .allow_boxed_zoom(true)
        .allow_drag(true)
        .allow_scroll(true);

    let plot_response = plot.show(ui, |plot_ui| {
        // Diagonal reference line y = x
        let line_points: PlotPoints = (0..=100)
            .map(|i| {
                let v = i as f64 / 100.0;
                [v, v]
            })
            .collect();
        plot_ui.line(
            Line::new(line_points)
                .color(egui::Color32::from_gray(100))
                .width(1.0)
                .name("y = x"),
        );

        // Separate points by grid size
        let colors = [
            (6, egui::Color32::from_rgb(70, 130, 230)),  // blue
            (8, egui::Color32::from_rgb(50, 180, 80)),    // green
            (10, egui::Color32::from_rgb(220, 70, 70)),   // red
        ];

        for (gs, color) in colors {
            let pts: Vec<[f64; 2]> = points
                .iter()
                .filter(|p| p.grid_size == gs)
                .map(|p| [p.win_prob, p.accuracy])
                .collect();

            if !pts.is_empty() {
                plot_ui.points(
                    Points::new(pts)
                        .color(color)
                        .radius(3.0)
                        .name(format!("Grid {gs}")),
                );
            }
        }

        // Highlight selected level
        if let Some(sel) = store.selected
            && let Some(p) = points.iter().find(|p| p.row_idx == sel)
        {
            plot_ui.points(
                Points::new(vec![[p.win_prob, p.accuracy]])
                    .color(egui::Color32::YELLOW)
                    .radius(6.0)
                    .name("Selected"),
            );
        }
    });

    // Handle click-to-select
    if plot_response.response.clicked()
        && let Some(hover_pos) = plot_response.response.hover_pos()
    {
        let plot_pos = plot_response.transform.value_from_position(hover_pos);
        // Find nearest point
        let mut best_dist = f64::MAX;
        let mut best_idx = None;
        for p in &points {
            let dx = p.win_prob - plot_pos.x;
            let dy = p.accuracy - plot_pos.y;
            let dist = dx * dx + dy * dy;
            if dist < best_dist {
                best_dist = dist;
                best_idx = Some(p.row_idx);
            }
        }
        // Only select if reasonably close (within 5% of axis range)
        if best_dist < 0.05 * 0.05 {
            clicked_row = best_idx;
        }
    }

    // Selected level stats panel
    ui.separator();
    if let Some(sel) = store.selected {
        let row = &store.rows[sel];
        if let Some(metric) = tm.get(&row.file_stem, row.sublevel) {
            ui.horizontal(|ui: &mut Ui| {
                ui.label(format!("Agent Accuracy: {:.1}%", metric.accuracy * 100.0));
                ui.separator();
                ui.label(format!("Mean Loss: {:.3}", metric.mean_loss));
            });
            ui.horizontal(|ui: &mut Ui| {
                if let Some(a) = &row.analysis {
                    ui.label(format!("States: {}", a.n_states));
                    ui.separator();
                    ui.label(format!("BFS Moves: {}", row.bfs_moves.map(|m| m.to_string()).unwrap_or("-".into())));
                    ui.separator();
                    ui.label(format!("Win% (random): {:.1}%", a.win_prob * 100.0));
                    ui.separator();
                    ui.label(format!("Dead-end%: {:.1}%", a.difficulty.dead_end_ratio * 100.0));
                }
            });
        } else {
            ui.label("No training data for selected level");
        }
    } else {
        ui.label("Select a level to see stats");
    }

    clicked_row
}
