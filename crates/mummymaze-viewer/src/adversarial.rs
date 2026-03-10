//! Adversarial training loop state machine.
//!
//! Orchestrates: Training(round N) → GA(gs=6) → GA(gs=8) → GA(gs=10) → Training(round N+1) → ...

use crate::level_gen_tab::latest_checkpoint;
use crate::data::{EpochRecord, TrainingConfig, TrainingPhase};
use crate::training_process::{TrainingEvent, TrainingProcess, TrainingSpawnArgs};
use mummymaze::ga::archive::ArchiveSnapshot;
use mummymaze::ga::{self, GaConfig, GaMessage};
use mummymaze::parse::Level;
use mummymaze::policy_client::PolicyClient;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;

/// Configuration for the adversarial training loop.
#[derive(Clone)]
pub struct AdversarialConfig {
    pub n_rounds: usize,
    pub epochs_per_round: u32,
    pub batch_size: u32,
    pub lr: f64,
    pub seed: u32,
    pub ga_config: GaConfig,
    pub target_log_wp: f64,
    pub archive_bfs_bins: usize,
    pub archive_states_bins: usize,
    pub grid_sizes: Vec<i32>,
}

impl AdversarialConfig {
    /// Build the fitness expression from the target log win probability.
    pub fn fitness_expr(&self) -> String {
        format!("-abs(log_policy_win_prob - ({}))", self.target_log_wp)
    }
}

impl Default for AdversarialConfig {
    fn default() -> Self {
        let target_log_wp = -1.0;
        AdversarialConfig {
            n_rounds: 3,
            epochs_per_round: 5,
            batch_size: 1024,
            lr: 3e-4,
            seed: 0,
            ga_config: GaConfig {
                fitness_expr: format!("-abs(log_policy_win_prob - ({target_log_wp}))"),
                ..GaConfig::default()
            },
            target_log_wp,
            archive_bfs_bins: 20,
            archive_states_bins: 20,
            grid_sizes: vec![6, 8, 10],
        }
    }
}

#[derive(Clone)]
pub enum AdversarialPhase {
    Training,
    GA { grid_size: i32, gs_idx: usize },
}

#[derive(Default)]
pub enum AdversarialStatus {
    #[default]
    Idle,
    Running {
        round: usize,
        phase: AdversarialPhase,
        // Training sub-state
        training_epoch: u32,
        training_total_epochs: u32,
        training_step: u32,
        training_steps_in_epoch: u32,
        training_loss: f64,
        training_acc: f64,
        training_gs: i32,
        training_phase: TrainingPhase,
        // GA sub-state
        ga_generation: usize,
        ga_total_generations: usize,
    },
    Done,
    Error(String),
}

pub struct AdversarialState {
    pub config: AdversarialConfig,
    pub show_config: bool,
    pub status: AdversarialStatus,
    // Training data (cumulative across rounds)
    pub epoch_history: Vec<EpochRecord>,
    pub batch_loss_history: Vec<[f64; 2]>,
    // GA data (current round, resets each GA phase)
    pub ga_history: Vec<(usize, f64, f64)>,
    // Archive
    pub archive_snapshot: Option<ArchiveSnapshot>,
    pub archive_occupancy_history: Vec<(usize, usize)>,
    // Round boundaries for vertical lines on training curves
    pub round_boundaries: Vec<usize>,
    pub curve_plot_height: f32,
    // Internal
    training_process: Option<TrainingProcess>,
    ga_rx: Option<Receiver<GaMessage>>,
    ga_stop_flag: Option<Arc<AtomicBool>>,
    round_archive_levels: Vec<Level>,
    global_epoch: usize,
    global_step: usize,
    latest_checkpoint: Option<PathBuf>,
    checkpoint_dir: PathBuf,
    pub log_messages: Vec<String>,
}

impl AdversarialState {
    pub fn new() -> Self {
        AdversarialState {
            config: AdversarialConfig::default(),
            show_config: false,
            status: AdversarialStatus::Idle,
            epoch_history: Vec::new(),
            batch_loss_history: Vec::new(),
            ga_history: Vec::new(),
            archive_snapshot: None,
            archive_occupancy_history: Vec::new(),
            round_boundaries: Vec::new(),
            curve_plot_height: 150.0,
            training_process: None,
            ga_rx: None,
            ga_stop_flag: None,
            round_archive_levels: Vec::new(),
            global_epoch: 0,
            global_step: 0,
            latest_checkpoint: None,
            checkpoint_dir: PathBuf::from("checkpoints/adversarial"),
            log_messages: Vec::new(),
        }
    }

    /// Begin the adversarial loop from round 0.
    pub fn start(&mut self, maze_dir: &Path, rows: &[crate::data::LevelRow]) {
        self.epoch_history.clear();
        self.batch_loss_history.clear();
        self.ga_history.clear();
        self.archive_snapshot = None;
        self.archive_occupancy_history.clear();
        self.round_boundaries.clear();
        self.round_archive_levels.clear();
        self.global_epoch = 0;
        self.global_step = 0;
        self.latest_checkpoint = latest_checkpoint().map(PathBuf::from);
        self.log_messages.clear();

        self.checkpoint_dir = PathBuf::from("checkpoints/adversarial");
        self.start_training_phase(0, maze_dir, rows);
    }

    /// Start a training subprocess for the given round.
    fn start_training_phase(
        &mut self,
        round: usize,
        maze_dir: &Path,
        _rows: &[crate::data::LevelRow],
    ) {
        self.round_boundaries.push(self.global_epoch);

        let round_ckpt_dir = self.checkpoint_dir.join(format!("round{round:03}"));
        let training_config = TrainingConfig {
            epochs: self.config.epochs_per_round,
            batch_size: self.config.batch_size,
            lr: self.config.lr,
            seed: self.config.seed,
            wandb: false,
            wandb_project: String::new(),
        };

        // Build extra args for checkpoint continuation + augmentation
        let archive_path = if round > 0 {
            let prev_dir = self.checkpoint_dir.join(format!("round{:03}", round - 1));
            let path = prev_dir.join("archive.json");
            if path.exists() { Some(path) } else { None }
        } else {
            None
        };

        let extra = TrainingSpawnArgs {
            checkpoint: self.latest_checkpoint.clone(),
            augment_levels: archive_path,
            checkpoint_dir: Some(round_ckpt_dir),
            epoch_offset: self.global_epoch as u32,
            step_offset: self.global_step as u32,
        };

        match TrainingProcess::spawn_with_args(maze_dir, &training_config, Some(&extra)) {
            Ok(proc) => {
                self.training_process = Some(proc);
                self.status = AdversarialStatus::Running {
                    round,
                    phase: AdversarialPhase::Training,
                    training_epoch: 0,
                    training_total_epochs: self.config.epochs_per_round,
                    training_step: 0,
                    training_steps_in_epoch: 0,
                    training_loss: 0.0,
                    training_acc: 0.0,
                    training_gs: 0,
                    training_phase: TrainingPhase::Status("Loading dataset...".into()),
                    ga_generation: 0,
                    ga_total_generations: self.config.ga_config.generations,
                };
            }
            Err(e) => {
                self.status = AdversarialStatus::Error(format!("Failed to start training: {e}"));
            }
        }
    }

    /// Start GA phase for a specific grid_size within a round.
    fn start_ga_phase(
        &mut self,
        round: usize,
        gs_idx: usize,
        rows: &[crate::data::LevelRow],
    ) {
        let grid_size = self.config.grid_sizes[gs_idx];
        let (tx, rx) = mpsc::channel();
        let stop_flag = Arc::new(AtomicBool::new(false));

        let mut config = self.config.ga_config.clone();
        config.grid_size = grid_size;
        config.fitness_expr = self.config.fitness_expr();
        config.seed = self.config.seed as u64 + round as u64 + gs_idx as u64 + 1;
        let total_generations = config.generations;

        let seed_levels: Vec<Level> = rows
            .iter()
            .filter(|r| r.level.grid_size == grid_size)
            .map(|r| r.level.clone())
            .collect();

        let flag = stop_flag.clone();
        let checkpoint_path = self.latest_checkpoint.clone();
        let target = self.config.target_log_wp;
        let bfs_bins = self.config.archive_bfs_bins;
        let states_bins = self.config.archive_states_bins;
        let max_batch_size = self.config.batch_size;

        std::thread::spawn(move || {
            if let Some(ref ckpt) = checkpoint_path {
                let _ = tx.send(GaMessage::Status("Starting policy server...".to_string()));
                match PolicyClient::spawn_with_max_batch(ckpt, max_batch_size) {
                    Ok(policy_client) => {
                        let archive = ga::archive::MapElitesArchive::new(
                            (1, 50),
                            (1, 500),
                            bfs_bins,
                            states_bins,
                            target,
                        );
                        ga::run_ga_with_archive(
                            &config,
                            seed_levels,
                            tx,
                            flag,
                            policy_client,
                            archive,
                        );
                    }
                    Err(e) => {
                        let _ = tx.send(GaMessage::Error(format!(
                            "Failed to start policy server: {e}"
                        )));
                    }
                }
            } else {
                let _ = tx.send(GaMessage::Error(
                    "No checkpoint available for policy evaluation".to_string(),
                ));
            }
        });

        self.ga_rx = Some(rx);
        self.ga_stop_flag = Some(stop_flag);
        self.ga_history.clear();
        self.status = AdversarialStatus::Running {
            round,
            phase: AdversarialPhase::GA {
                grid_size,
                gs_idx,
            },
            training_epoch: 0,
            training_total_epochs: 0,
            training_step: 0,
            training_steps_in_epoch: 0,
            training_loss: 0.0,
            training_acc: 0.0,
            training_gs: 0,
            training_phase: TrainingPhase::default(),
            ga_generation: 0,
            ga_total_generations: total_generations,
        };
    }

    /// Poll for events and handle phase transitions. Returns true if anything changed.
    pub fn poll(&mut self, maze_dir: &Path, rows: &[crate::data::LevelRow]) -> bool {
        match &self.status {
            AdversarialStatus::Idle | AdversarialStatus::Done | AdversarialStatus::Error(_) => return false,
            _ => {}
        }

        // Extract round and phase before mutable borrows
        let (round, phase) = match &self.status {
            AdversarialStatus::Running { round, phase, .. } => (*round, phase.clone()),
            _ => return false,
        };

        match phase {
            AdversarialPhase::Training => self.poll_training(round, maze_dir, rows),
            AdversarialPhase::GA { gs_idx, .. } => self.poll_ga(round, gs_idx, maze_dir, rows),
        }
    }

    /// Poll training subprocess events.
    fn poll_training(
        &mut self,
        round: usize,
        maze_dir: &Path,
        rows: &[crate::data::LevelRow],
    ) -> bool {
        let Some(ref mut proc) = self.training_process else {
            return false;
        };

        let events = proc.poll();
        if events.is_empty() {
            if !proc.is_running() {
                if matches!(self.status, AdversarialStatus::Running { .. }) {
                    self.status = AdversarialStatus::Error(
                        "Training process exited unexpectedly".into(),
                    );
                }
                self.training_process = None;
            }
            return false;
        }

        let mut training_done = false;

        for event in events {
            match event {
                TrainingEvent::Init { .. } => {}
                TrainingEvent::EpochStart {
                    epoch,
                    total_epochs,
                    steps_in_epoch,
                } => {
                    if let AdversarialStatus::Running {
                        training_epoch: ref mut e,
                        training_total_epochs: ref mut te,
                        training_steps_in_epoch: ref mut sie,
                        training_step: ref mut es,
                        ..
                    } = self.status
                    {
                        *e = epoch;
                        *te = total_epochs;
                        *sie = steps_in_epoch;
                        *es = 0;
                    }
                }
                TrainingEvent::Batch {
                    step,
                    epoch_step,
                    loss,
                    acc,
                    gs,
                } => {
                    if let AdversarialStatus::Running {
                        training_step: ref mut es,
                        training_loss: ref mut l,
                        training_acc: ref mut a,
                        training_gs: ref mut g,
                        training_phase: ref mut ph,
                        ..
                    } = self.status
                    {
                        *es = epoch_step;
                        *l = loss;
                        *a = acc;
                        *g = gs;
                        *ph = TrainingPhase::Training;
                    }
                    self.batch_loss_history
                        .push([self.batch_loss_history.len() as f64, loss]);
                    self.global_step = step as usize;
                }
                TrainingEvent::Status(text) => {
                    if let AdversarialStatus::Running {
                        training_phase: ref mut ph,
                        ..
                    } = self.status
                    {
                        *ph = TrainingPhase::Status(text);
                    }
                }
                TrainingEvent::EpochEnd {
                    epoch,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                    ..
                } => {
                    self.epoch_history.push(EpochRecord {
                        epoch,
                        train_loss,
                        train_acc,
                        val_loss,
                        val_acc,
                    });
                    self.global_epoch = epoch as usize;
                }
                TrainingEvent::LevelMetrics { .. } => {
                    // Could update store.training_metrics here if desired
                }
                TrainingEvent::Log(msg) => {
                    self.log_messages.push(msg);
                }
                TrainingEvent::Done => {
                    training_done = true;
                }
                TrainingEvent::Error(msg) => {
                    self.status = AdversarialStatus::Error(msg);
                    self.training_process = None;
                    return true;
                }
            }
        }

        if training_done {
            self.training_process = None;
            // Find latest checkpoint
            self.latest_checkpoint = find_latest_checkpoint_in(
                self.checkpoint_dir.to_str().unwrap_or("checkpoints/adversarial"),
            );
            self.log_messages.push(format!(
                "Round {round} training complete, checkpoint: {:?}",
                self.latest_checkpoint
            ));

            // Decide: GA or Done
            let is_last_round = round >= self.config.n_rounds - 1;
            if is_last_round {
                self.status = AdversarialStatus::Done;
            } else if !self.config.grid_sizes.is_empty() {
                // Start GA for first grid_size
                self.start_ga_phase(round, 0, rows);
            } else {
                // No grid sizes configured, skip GA
                self.start_training_phase(round + 1, maze_dir, rows);
            }
        }

        true
    }

    /// Poll GA channel events.
    fn poll_ga(
        &mut self,
        round: usize,
        gs_idx: usize,
        maze_dir: &Path,
        rows: &[crate::data::LevelRow],
    ) -> bool {
        let Some(ref rx) = self.ga_rx else {
            return false;
        };

        let mut changed = false;
        let mut ga_done = false;
        let mut ga_error = None;

        #[allow(unused_assignments)]
        while let Ok(msg) = rx.try_recv() {
            changed = true;
            match msg {
                GaMessage::Status(s) => {
                    if let AdversarialStatus::Running {
                        training_phase: ref mut ph,
                        ..
                    } = self.status
                    {
                        *ph = TrainingPhase::Status(s);
                    }
                }
                GaMessage::SeedsDone { .. } => {}
                GaMessage::Generation(result) => {
                    self.ga_history
                        .push((result.generation, result.best.fitness, result.avg_fitness));
                    if let AdversarialStatus::Running {
                        ga_generation: ref mut g,
                        ..
                    } = self.status
                    {
                        *g = result.generation;
                    }
                }
                GaMessage::ArchiveUpdate {
                    occupancy,
                    total_cells: _,
                    grid,
                } => {
                    self.archive_snapshot = Some(grid);
                    self.archive_occupancy_history.push((round, occupancy));
                }
                GaMessage::ArchiveLevels(levels) => {
                    self.round_archive_levels.extend(levels);
                }
                GaMessage::Done => {
                    ga_done = true;
                }
                GaMessage::Error(e) => {
                    ga_error = Some(e);
                }
            }
        }

        if let Some(e) = ga_error {
            self.log_messages.push(format!("GA error: {e}"));
            // Treat GA error as non-fatal — skip to next grid_size or next round
            ga_done = true;
        }

        if ga_done {
            self.ga_rx = None;
            self.ga_stop_flag = None;

            let next_gs_idx = gs_idx + 1;
            if next_gs_idx < self.config.grid_sizes.len() {
                // More grid sizes to run GA on
                self.start_ga_phase(round, next_gs_idx, rows);
            } else {
                // All grid sizes done for this round, start next training round
                // Write archive.json for augmentation
                let round_dir = self.checkpoint_dir.join(format!("round{round:03}"));
                if let Err(e) = self.write_archive_json(&round_dir) {
                    self.log_messages
                        .push(format!("Failed to write archive.json: {e}"));
                }
                self.round_archive_levels.clear();
                self.start_training_phase(round + 1, maze_dir, rows);
            }
        }

        changed
    }

    /// Write collected archive levels to JSON for the next training round.
    fn write_archive_json(&self, round_dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(round_dir)?;
        let path = round_dir.join("archive.json");
        // Serialize levels using Level::to_edges() etc.
        let mut entries = Vec::new();
        for level in &self.round_archive_levels {
            entries.push(level_to_json(level));
        }
        let json = serde_json::to_string(&entries)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Stop the running loop.
    pub fn stop(&mut self) {
        if let Some(ref mut proc) = self.training_process {
            proc.send_stop();
        }
        if let Some(ref flag) = self.ga_stop_flag {
            flag.store(true, Ordering::Relaxed);
        }
    }

    pub fn is_running(&self) -> bool {
        matches!(self.status, AdversarialStatus::Running { .. })
    }
}

/// Find the newest .eqx checkpoint in a directory.
fn find_latest_checkpoint_in(dir: &str) -> Option<PathBuf> {
    let dir = PathBuf::from(dir);
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;

    fn scan_dir(dir: &Path, best: &mut Option<(std::time::SystemTime, PathBuf)>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                scan_dir(&path, best);
            } else if path.extension().map_or(false, |e| e == "eqx") {
                if let Ok(meta) = entry.metadata() {
                    if let Ok(mtime) = meta.modified() {
                        if best.as_ref().map_or(true, |(t, _)| mtime > *t) {
                            *best = Some((mtime, path));
                        }
                    }
                }
            }
        }
    }

    scan_dir(&dir, &mut best);
    best.map(|(_, p)| p)
}

/// Serialize a Level to a serde_json::Value matching Python's Level.to_dict() format.
fn level_to_json(level: &Level) -> serde_json::Value {
    let (h_walls, v_walls) = level.to_edges();

    let mut obj = serde_json::Map::new();
    obj.insert("grid_size".into(), level.grid_size.into());
    obj.insert("flip".into(), level.flip.into());
    obj.insert("h_walls".into(), h_walls.into());
    obj.insert("v_walls".into(), v_walls.into());
    obj.insert("exit_side".into(), level.exit_side_str().into());
    obj.insert("exit_pos".into(), level.exit_pos().into());
    obj.insert(
        "player".into(),
        vec![level.player_row, level.player_col].into(),
    );
    obj.insert(
        "mummy1".into(),
        vec![level.mummy1_row, level.mummy1_col].into(),
    );
    obj.insert(
        "mummy2".into(),
        if level.has_mummy2 {
            vec![level.mummy2_row, level.mummy2_col].into()
        } else {
            serde_json::Value::Null
        },
    );
    obj.insert(
        "scorpion".into(),
        if level.has_scorpion {
            vec![level.scorpion_row, level.scorpion_col].into()
        } else {
            serde_json::Value::Null
        },
    );
    let mut traps = Vec::new();
    if level.trap_count >= 1 {
        traps.push(vec![level.trap1_row, level.trap1_col]);
    }
    if level.trap_count >= 2 {
        traps.push(vec![level.trap2_row, level.trap2_col]);
    }
    obj.insert("traps".into(), traps.into());
    obj.insert(
        "gate".into(),
        if level.has_gate {
            vec![level.gate_row, level.gate_col].into()
        } else {
            serde_json::Value::Null
        },
    );
    obj.insert(
        "key".into(),
        if level.has_gate {
            vec![level.key_row, level.key_col].into()
        } else {
            serde_json::Value::Null
        },
    );
    serde_json::Value::Object(obj)
}
