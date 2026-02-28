//! Full state graph builder — port of src/solver.py::build_graph()
//!
//! BFS over all reachable states, recording transitions for every valid action.
//! Does NOT short-circuit on first win — explores entire reachable state space.

use crate::game::{Action, State, StepResult, can_do_action, step};
use crate::parse::Level;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// Destination of a transition: either a transient state, WIN, or DEAD.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateKey {
    Transient(State),
    Win,
    Dead,
}

/// Complete state-action transition graph.
#[derive(Debug)]
pub struct StateGraph {
    pub start: State,
    /// For each transient state: the list of (action, destination) transitions.
    pub transitions: FxHashMap<State, Vec<(Action, StateKey)>>,
    pub n_transient: usize,
}

/// BFS over all reachable states, recording transitions for every valid action.
/// Blocked moves (wall/gate) are skipped since they are equivalent to WAIT.
pub fn build_graph(lev: &Level) -> StateGraph {
    let start = State::from_level(lev);
    let mut transitions: FxHashMap<State, Vec<(Action, StateKey)>> = FxHashMap::default();
    let mut visited = rustc_hash::FxHashSet::default();
    visited.insert(start);

    let mut queue: VecDeque<State> = VecDeque::new();
    queue.push_back(start);

    while let Some(cur) = queue.pop_front() {
        let mut action_map: Vec<(Action, StateKey)> = Vec::new();

        for &action in &Action::ALL {
            // Skip blocked directional moves (equivalent to WAIT)
            if action != Action::Wait && !can_do_action(lev, &cur, action) {
                continue;
            }

            let mut next = cur;
            let result = step(lev, &mut next, action);

            let key = match result {
                StepResult::Win => StateKey::Win,
                StepResult::Dead => StateKey::Dead,
                StepResult::Ok => {
                    if visited.insert(next) {
                        queue.push_back(next);
                    }
                    StateKey::Transient(next)
                }
            };

            action_map.push((action, key));
        }

        transitions.insert(cur, action_map);
    }

    let n = transitions.len();
    StateGraph {
        start,
        transitions,
        n_transient: n,
    }
}
