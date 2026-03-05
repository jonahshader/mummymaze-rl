use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

#[derive(Deserialize)]
struct MetricsFile {
    run_id: String,
    step: u64,
    levels: HashMap<String, LevelMetric>,
}

#[derive(Debug, Deserialize)]
pub struct LevelMetric {
    #[allow(dead_code)]
    pub grid_size: i32,
    #[allow(dead_code)]
    pub n_states: usize,
    pub accuracy: f64,
    pub mean_loss: f64,
}

pub struct TrainingMetrics {
    pub run_id: String,
    pub step: u64,
    pub levels: HashMap<String, LevelMetric>,
    path: PathBuf,
    last_mtime: Option<SystemTime>,
}

impl TrainingMetrics {
    pub fn new(path: PathBuf) -> Self {
        TrainingMetrics {
            run_id: String::new(),
            step: 0,
            levels: HashMap::new(),
            path,
            last_mtime: None,
        }
    }

    /// Check file mtime and re-read if changed. Returns true on update.
    pub fn poll(&mut self) -> bool {
        let mtime = match fs::metadata(&self.path).and_then(|m| m.modified()) {
            Ok(t) => t,
            Err(_) => return false,
        };

        if self.last_mtime == Some(mtime) {
            return false;
        }

        let contents = match fs::read_to_string(&self.path) {
            Ok(c) => c,
            Err(_) => return false,
        };

        let parsed: MetricsFile = match serde_json::from_str(&contents) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to parse level_metrics.json: {e}");
                return false;
            }
        };

        self.run_id = parsed.run_id;
        self.step = parsed.step;
        self.levels = parsed.levels;
        self.last_mtime = Some(mtime);
        true
    }

    /// Update directly from parsed event data (subprocess mode).
    pub fn update_from_event(
        &mut self,
        run_id: String,
        step: u64,
        levels: HashMap<String, LevelMetric>,
    ) {
        self.run_id = run_id;
        self.step = step;
        self.levels = levels;
    }

    /// Create an empty TrainingMetrics (no file backing).
    pub fn empty() -> Self {
        TrainingMetrics {
            run_id: String::new(),
            step: 0,
            levels: HashMap::new(),
            path: PathBuf::new(),
            last_mtime: None,
        }
    }

    pub fn get(&self, file_stem: &str, sublevel: usize) -> Option<&LevelMetric> {
        let key = format!("{file_stem}:{sublevel}");
        self.levels.get(&key)
    }
}
