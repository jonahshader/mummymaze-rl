use std::collections::HashMap;

pub use mummymaze::event_types::LevelMetric;

pub struct TrainingMetrics {
    pub run_id: String,
    pub step: u64,
    pub levels: HashMap<String, LevelMetric>,
}

impl TrainingMetrics {
    pub fn new() -> Self {
        TrainingMetrics {
            run_id: String::new(),
            step: 0,
            levels: HashMap::new(),
        }
    }

    /// Update directly from parsed event data (via WebSocket).
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

    pub fn get(&self, file_stem: &str, sublevel: usize) -> Option<&LevelMetric> {
        let key = format!("{file_stem}:{sublevel}");
        self.levels.get(&key)
    }
}
