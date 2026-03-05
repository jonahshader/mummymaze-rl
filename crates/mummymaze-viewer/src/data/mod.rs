mod training;
mod types;

pub use training::{EpochRecord, TrainingConfig, TrainingPhase, TrainingStatus};
pub use types::{FilterState, LevelRow, SortColumn, SortDir};

use crate::training_metrics::TrainingMetrics;
use crate::training_process::TrainingProcess;
use mummymaze::batch::{self, LevelAnalysis};
use mummymaze::parse::Level;
use mummymaze::solver;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::mpsc::{self, Receiver};

/// Hash a Level by its gameplay-relevant fields for dedup.
fn level_fingerprint(lev: &Level) -> u64 {
    let mut h = std::hash::DefaultHasher::new();
    lev.grid_size.hash(&mut h);
    lev.walls.hash(&mut h);
    lev.player_row.hash(&mut h);
    lev.player_col.hash(&mut h);
    lev.mummy1_row.hash(&mut h);
    lev.mummy1_col.hash(&mut h);
    lev.mummy2_row.hash(&mut h);
    lev.mummy2_col.hash(&mut h);
    lev.has_mummy2.hash(&mut h);
    lev.scorpion_row.hash(&mut h);
    lev.scorpion_col.hash(&mut h);
    lev.has_scorpion.hash(&mut h);
    lev.trap1_row.hash(&mut h);
    lev.trap1_col.hash(&mut h);
    lev.trap2_row.hash(&mut h);
    lev.trap2_col.hash(&mut h);
    lev.trap_count.hash(&mut h);
    lev.gate_row.hash(&mut h);
    lev.gate_col.hash(&mut h);
    lev.has_gate.hash(&mut h);
    lev.key_row.hash(&mut h);
    lev.key_col.hash(&mut h);
    lev.exit_row.hash(&mut h);
    lev.exit_col.hash(&mut h);
    lev.exit_mask.hash(&mut h);
    lev.flip.hash(&mut h);
    h.finish()
}

pub struct DataStore {
    pub rows: Vec<LevelRow>,
    pub sorted_indices: Vec<usize>,
    pub sort_col: SortColumn,
    pub sort_dir: SortDir,
    pub filter: FilterState,
    pub analysis_progress: Option<(usize, usize)>,
    analysis_rx: Option<Receiver<LevelAnalysis>>,
    row_index: HashMap<(String, usize), usize>,
    pub selected: Option<usize>,
    analysis_received: usize,
    pub training_metrics: Option<TrainingMetrics>,
    pub training_process: Option<TrainingProcess>,
    pub training_status: TrainingStatus,
    pub training_config: TrainingConfig,
    pub show_training_config: bool,
    pub epoch_history: Vec<EpochRecord>,
    pub batch_loss_history: Vec<[f64; 2]>,
    pub curve_plot_height: f32,
}

impl DataStore {
    /// Tier 1: parse all .dat files + BFS solve each level.
    pub fn load_levels(maze_dir: &Path) -> Self {
        let all_levels = batch::collect_levels(maze_dir).unwrap_or_default();
        let total = all_levels.len();

        let mut seen = HashSet::new();
        let rows: Vec<LevelRow> = all_levels
            .into_iter()
            .map(|(stem, sub, lev)| {
                let bfs = solver::solve(&lev).moves;
                let fp = level_fingerprint(&lev);
                let is_duplicate = !seen.insert(fp);
                LevelRow {
                    file_stem: stem,
                    sublevel: sub,
                    level: lev,
                    bfs_moves: bfs,
                    analysis: None,
                    is_duplicate,
                }
            })
            .collect();

        let row_index: HashMap<(String, usize), usize> = rows
            .iter()
            .enumerate()
            .map(|(i, r)| ((r.file_stem.clone(), r.sublevel), i))
            .collect();

        let sorted_indices: Vec<usize> = (0..rows.len()).collect();

        // Look for level_metrics.json next to the maze directory or in CWD
        let training_metrics = {
            let candidates = [
                maze_dir.join("level_metrics.json"),
                std::path::PathBuf::from("level_metrics.json"),
            ];
            candidates
                .into_iter()
                .find(|p| p.exists())
                .map(TrainingMetrics::new)
        };

        let mut store = DataStore {
            rows,
            sorted_indices,
            sort_col: SortColumn::File,
            sort_dir: SortDir::Asc,
            filter: FilterState::default(),
            analysis_progress: Some((0, total)),
            analysis_rx: None,
            row_index,
            selected: None,
            analysis_received: 0,
            training_metrics,
            training_process: None,
            training_status: TrainingStatus::default(),
            training_config: TrainingConfig::default(),
            show_training_config: false,
            epoch_history: Vec::new(),
            batch_loss_history: Vec::new(),
            curve_plot_height: 150.0,
        };
        store.refresh_sort_filter();
        store
    }

    /// Tier 2: spawn background thread for full analysis (graph + Markov + metrics).
    /// Passes already-parsed levels to avoid re-reading .dat files.
    pub fn start_analysis(&mut self) {
        let (tx, rx) = mpsc::channel();
        let total = self.rows.len();
        self.analysis_progress = Some((0, total));
        self.analysis_rx = Some(rx);

        let levels: Vec<(String, usize, Level)> = self
            .rows
            .iter()
            .map(|r| (r.file_stem.clone(), r.sublevel, r.level.clone()))
            .collect();

        std::thread::spawn(move || {
            use rayon::prelude::*;

            levels
                .par_iter()
                .for_each(|(stem, sub_idx, lev): &(String, usize, Level)| {
                    if let Ok(analysis) = batch::analyze_level(stem, *sub_idx, lev) {
                        let _ = tx.send(analysis);
                    }
                });
        });
    }

    /// Poll for analysis results from background thread. Returns true if new data arrived.
    /// Drains up to 200 results per call to stay responsive.
    pub fn poll_analysis(&mut self) -> bool {
        let mut received_any = false;

        // Poll training metrics file
        if let Some(ref mut tm) = self.training_metrics
            && tm.poll()
        {
            received_any = true;
            if self.sort_col.is_tier2() {
                self.refresh_sort_filter();
            }
        }

        let rx = match &self.analysis_rx {
            Some(rx) => rx,
            None => return received_any,
        };

        let mut analysis_received = false;
        for _ in 0..200 {
            match rx.try_recv() {
                Ok(analysis) => {
                    let key = (analysis.file_stem.clone(), analysis.sublevel);
                    if let Some(&idx) = self.row_index.get(&key) {
                        self.rows[idx].analysis = Some(analysis);
                    }
                    self.analysis_received += 1;
                    analysis_received = true;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.analysis_rx = None;
                    analysis_received = true;
                    break;
                }
            }
        }

        if analysis_received {
            received_any = true;
            let total = self.rows.len();
            if self.analysis_received >= total || self.analysis_rx.is_none() {
                self.analysis_progress = None;
                self.analysis_rx = None;
            } else {
                self.analysis_progress = Some((self.analysis_received, total));
            }
            // Only re-sort if user is sorting by a Tier 2 column
            if self.sort_col.is_tier2() {
                self.refresh_sort_filter();
            }
        }

        received_any
    }

    /// Rebuild sorted_indices based on current sort + filter.
    pub fn refresh_sort_filter(&mut self) {
        let filter = &self.filter;
        let needle = filter.text.to_lowercase();
        self.sorted_indices = (0..self.rows.len())
            .filter(|&i| {
                let row = &self.rows[i];
                if !needle.is_empty() {
                    let haystack = format!("{} {}", row.file_stem, row.sublevel);
                    if !haystack.to_lowercase().contains(&needle) {
                        return false;
                    }
                }
                if let Some(gs) = filter.grid_size {
                    if row.level.grid_size != gs {
                        return false;
                    }
                }
                if filter.solvable_only && row.bfs_moves.is_none() {
                    return false;
                }
                if !filter.show_duplicates && row.is_duplicate {
                    return false;
                }
                true
            })
            .collect();

        let sort_col = self.sort_col;
        let sort_dir = self.sort_dir;
        let rows = &self.rows;
        let tm = self.training_metrics.as_ref();

        self.sorted_indices.sort_by(|&a, &b| {
            let ra = &rows[a];
            let rb = &rows[b];
            let ord = match sort_col {
                SortColumn::File => ra
                    .file_stem
                    .cmp(&rb.file_stem)
                    .then(ra.sublevel.cmp(&rb.sublevel)),
                SortColumn::Sub => ra.sublevel.cmp(&rb.sublevel),
                SortColumn::Grid => ra.level.grid_size.cmp(&rb.level.grid_size),
                SortColumn::Bfs => ra.bfs_moves.cmp(&rb.bfs_moves),
                SortColumn::States => {
                    let sa = ra.analysis.as_ref().map(|a| a.n_states);
                    let sb = rb.analysis.as_ref().map(|a| a.n_states);
                    sa.cmp(&sb)
                }
                SortColumn::WinProb => {
                    let sa = ra.analysis.as_ref().map(|a| a.win_prob);
                    let sb = rb.analysis.as_ref().map(|a| a.win_prob);
                    sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                }
                SortColumn::DeadEnd => {
                    let sa = ra.analysis.as_ref().map(|a| a.difficulty.dead_end_ratio);
                    let sb = rb.analysis.as_ref().map(|a| a.difficulty.dead_end_ratio);
                    sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                }
                SortColumn::Safety => {
                    let sa = ra
                        .analysis
                        .as_ref()
                        .and_then(|a| a.difficulty.path_safety);
                    let sb = rb
                        .analysis
                        .as_ref()
                        .and_then(|a| a.difficulty.path_safety);
                    sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                }
                SortColumn::AgentAcc => {
                    let sa = tm.and_then(|t| t.get(&ra.file_stem, ra.sublevel).map(|m| m.accuracy));
                    let sb = tm.and_then(|t| t.get(&rb.file_stem, rb.sublevel).map(|m| m.accuracy));
                    sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                }
                SortColumn::AgentLoss => {
                    let sa = tm.and_then(|t| t.get(&ra.file_stem, ra.sublevel).map(|m| m.mean_loss));
                    let sb = tm.and_then(|t| t.get(&rb.file_stem, rb.sublevel).map(|m| m.mean_loss));
                    sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                }
            };
            match sort_dir {
                SortDir::Asc => ord,
                SortDir::Desc => ord.reverse(),
            }
        });
    }

    pub fn is_analyzing(&self) -> bool {
        self.analysis_rx.is_some()
    }

    pub fn toggle_sort(&mut self, col: SortColumn) {
        if self.sort_col == col {
            self.sort_dir.toggle();
        } else {
            self.sort_col = col;
            self.sort_dir = SortDir::Asc;
        }
        self.refresh_sort_filter();
    }
}
