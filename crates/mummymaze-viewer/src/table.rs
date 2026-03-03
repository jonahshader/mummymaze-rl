use crate::data::{DataStore, SortColumn, SortDir};
use eframe::egui;
use egui::{ScrollArea, Ui};
use egui_extras::{Column, TableBuilder};

/// Draw the filter bar (text filter, grid size dropdown, solvable-only checkbox).
pub fn draw_filters(ui: &mut Ui, store: &mut DataStore) {
    let mut changed = false;

    ui.horizontal(|ui: &mut Ui| {
        ui.label("Filter:");
        if ui
            .text_edit_singleline(&mut store.filter.text)
            .changed()
        {
            changed = true;
        }
    });

    ui.horizontal(|ui: &mut Ui| {
        ui.label("Grid:");
        let current = match store.filter.grid_size {
            None => "All",
            Some(6) => "6",
            Some(8) => "8",
            Some(10) => "10",
            _ => "All",
        };
        egui::ComboBox::from_id_salt("grid_filter")
            .selected_text(current)
            .show_ui(ui, |ui: &mut Ui| {
                if ui
                    .selectable_label(store.filter.grid_size.is_none(), "All")
                    .clicked()
                {
                    store.filter.grid_size = None;
                    changed = true;
                }
                for gs in [6, 8, 10] {
                    if ui
                        .selectable_label(store.filter.grid_size == Some(gs), gs.to_string())
                        .clicked()
                    {
                        store.filter.grid_size = Some(gs);
                        changed = true;
                    }
                }
            });

        if ui
            .checkbox(&mut store.filter.solvable_only, "Solvable")
            .changed()
        {
            changed = true;
        }
        if ui
            .checkbox(&mut store.filter.show_duplicates, "Duplicates")
            .changed()
        {
            changed = true;
        }
    });

    if changed {
        store.refresh_sort_filter();
    }
}

/// Draw the progress bar for background analysis.
pub fn draw_progress(ui: &mut Ui, store: &DataStore) {
    if let Some((done, total)) = store.analysis_progress {
        let frac = if total > 0 {
            done as f32 / total as f32
        } else {
            0.0
        };
        ui.add(
            egui::ProgressBar::new(frac)
                .text(format!("Analysis: {done}/{total}"))
                .desired_width(ui.available_width()),
        );
    }
}

fn sort_header(
    ui: &mut Ui,
    label: &str,
    col: SortColumn,
    store: &mut DataStore,
    tooltip: Option<&str>,
) {
    let arrow = if store.sort_col == col {
        match store.sort_dir {
            SortDir::Asc => " ^",
            SortDir::Desc => " v",
        }
    } else {
        ""
    };
    let text = format!("{label}{arrow}");
    let mut response = ui.button(text);
    if let Some(tip) = tooltip {
        response = response.on_hover_text(tip);
    }
    if response.clicked() {
        store.toggle_sort(col);
    }
}

/// Draw the sortable, filterable, virtual-scrolling level table.
/// Returns Some(row_index) if a row was clicked.
pub fn draw_table(ui: &mut Ui, store: &mut DataStore) -> Option<usize> {
    let mut clicked_row = None;

    let available = ui.available_size();
    let row_height = 20.0;

    ScrollArea::horizontal().show(ui, |ui: &mut Ui| {
    TableBuilder::new(ui)
        .id_salt("level_table")
        .striped(true)
        .sense(egui::Sense::click())
        .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
        .min_scrolled_height(available.y - 10.0)
        .max_scroll_height(available.y - 10.0)
        .column(Column::auto().at_least(50.0)) // File
        .column(Column::auto().at_least(35.0)) // Sub
        .column(Column::auto().at_least(35.0)) // Grid
        .column(Column::auto().at_least(35.0)) // Moves
        .column(Column::auto().at_least(50.0)) // States
        .column(Column::auto().at_least(50.0)) // Win%
        .column(Column::auto().at_least(50.0)) // Dead%
        .column(Column::auto().at_least(50.0)) // Safety
        .column(Column::auto().at_least(45.0)) // Acc%
        .column(Column::remainder().at_least(45.0)) // Loss
        .header(row_height, |mut header: egui_extras::TableRow<'_, '_>| {
            header.col(|ui: &mut Ui| {
                sort_header(ui, "File", SortColumn::File, store, Some("Pyramid .dat file (B-0 through B-100)"));
            });
            header.col(|ui: &mut Ui| {
                sort_header(ui, "Sub", SortColumn::Sub, store, Some("Sublevel index within the pyramid (0-99)"));
            });
            header.col(|ui: &mut Ui| {
                sort_header(ui, "Grid", SortColumn::Grid, store, Some("Grid size (6, 8, or 10)"));
            });
            header.col(|ui: &mut Ui| {
                sort_header(ui, "Moves", SortColumn::Bfs, store, Some("Minimum moves to win (BFS optimal solution)"));
            });
            header.col(|ui: &mut Ui| {
                sort_header(ui, "States", SortColumn::States, store, Some("Total reachable game states"));
            });
            header.col(|ui: &mut Ui| {
                sort_header(ui, "Win%", SortColumn::WinProb, store, Some("Probability of winning under uniformly random play"));
            });
            header.col(|ui: &mut Ui| {
                sort_header(ui, "Dead%", SortColumn::DeadEnd, store, Some("Fraction of reachable states from which winning is impossible"));
            });
            header.col(|ui: &mut Ui| {
                sort_header(ui, "Safety", SortColumn::Safety, store, Some("Average fraction of actions leading to winnable states along the optimal path"));
            });
            header.col(|ui: &mut Ui| {
                sort_header(ui, "Acc%", SortColumn::AgentAcc, store, Some("Agent accuracy on this level (from training metrics)"));
            });
            header.col(|ui: &mut Ui| {
                sort_header(ui, "Loss", SortColumn::AgentLoss, store, Some("Agent mean loss on this level (from training metrics)"));
            });
        })
        .body(|body: egui_extras::TableBody<'_>| {
            // Split borrows: take a snapshot of indices to avoid cloning
            let indices = &store.sorted_indices;
            let rows = &store.rows;
            let selected = store.selected;
            let tm = store.training_metrics.as_ref();
            let num_rows = indices.len();

            body.rows(
                row_height,
                num_rows,
                |mut row: egui_extras::TableRow<'_, '_>| {
                    let display_idx = row.index();
                    let real_idx = indices[display_idx];
                    let level_row = &rows[real_idx];

                    row.set_selected(selected == Some(real_idx));

                    row.col(|ui: &mut Ui| {
                        ui.label(&level_row.file_stem);
                    });
                    row.col(|ui: &mut Ui| {
                        ui.label(level_row.sublevel.to_string());
                    });
                    row.col(|ui: &mut Ui| {
                        ui.label(level_row.level.grid_size.to_string());
                    });
                    row.col(|ui: &mut Ui| {
                        ui.label(match level_row.bfs_moves {
                            Some(m) => m.to_string(),
                            None => "-".to_string(),
                        });
                    });

                    let a = level_row.analysis.as_ref();
                    row.col(|ui: &mut Ui| {
                        ui.label(match a.map(|a| a.n_states) {
                            Some(n) => n.to_string(),
                            None => "...".to_string(),
                        });
                    });
                    row.col(|ui: &mut Ui| {
                        ui.label(match a.map(|a| a.win_prob) {
                            Some(p) => format!("{:.1}%", p * 100.0),
                            None => "...".to_string(),
                        });
                    });
                    row.col(|ui: &mut Ui| {
                        ui.label(match a.map(|a| a.difficulty.dead_end_ratio) {
                            Some(d) => format!("{:.1}%", d * 100.0),
                            None => "...".to_string(),
                        });
                    });
                    row.col(|ui: &mut Ui| {
                        ui.label(match a.and_then(|a| a.difficulty.path_safety) {
                            Some(s) => format!("{:.2}", s),
                            None => "...".to_string(),
                        });
                    });

                    let metric = tm.and_then(|t| t.get(&level_row.file_stem, level_row.sublevel));
                    row.col(|ui: &mut Ui| {
                        ui.label(match metric {
                            Some(m) => format!("{:.1}%", m.accuracy * 100.0),
                            None => "-".to_string(),
                        });
                    });
                    row.col(|ui: &mut Ui| {
                        ui.label(match metric {
                            Some(m) => format!("{:.2}", m.mean_loss),
                            None => "-".to_string(),
                        });
                    });

                    if row.response().clicked() {
                        clicked_row = Some(real_idx);
                    }
                },
            );
        });
    });

    clicked_row
}
