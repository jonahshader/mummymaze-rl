use mummymaze::game::{Action, State, StepResult, step};
use mummymaze::parse::Level;

pub struct GameplayState {
    pub level: Level,
    pub current_state: State,
    pub initial_state: State,
    pub turn: u32,
    pub result: Option<StepResult>,
    pub history: Vec<(Action, State)>, // (action taken, state BEFORE action)
}

impl GameplayState {
    pub fn new(level: Level) -> Self {
        let state = State::from_level(&level);
        GameplayState {
            level,
            current_state: state,
            initial_state: state,
            turn: 0,
            result: None,
            history: Vec::new(),
        }
    }

    pub fn apply_action(&mut self, action: Action) {
        if self.result.is_some() {
            return; // Game already over
        }

        let prev_state = self.current_state;
        let res = step(&self.level, &mut self.current_state, action);

        self.history.push((action, prev_state));
        self.turn += 1;

        match res {
            StepResult::Ok => {}
            StepResult::Dead | StepResult::Win => {
                self.result = Some(res);
            }
        }
    }

    pub fn undo(&mut self) {
        if let Some((_action, prev_state)) = self.history.pop() {
            self.current_state = prev_state;
            self.turn -= 1;
            self.result = None;
        }
    }

    pub fn reset(&mut self) {
        self.current_state = self.initial_state;
        self.turn = 0;
        self.result = None;
        self.history.clear();
    }

    pub fn status_text(&self) -> String {
        match self.result {
            None => format!("Turn {}", self.turn),
            Some(StepResult::Win) => format!("WIN in {} turns!", self.turn),
            Some(StepResult::Dead) => format!("DEAD at turn {}", self.turn),
            Some(StepResult::Ok) => format!("Turn {}", self.turn),
        }
    }

    pub fn is_over(&self) -> bool {
        self.result.is_some()
    }
}
