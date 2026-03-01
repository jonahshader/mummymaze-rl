//! Full state graph builder — port of src/solver.py::build_graph()
//!
//! BFS over all reachable states, recording transitions for every valid action.
//! Does NOT short-circuit on first win — explores entire reachable state space.

use crate::game::{Action, State, StepResult, can_do_action, step};
use crate::parse::Level;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// Bidirectional state-to-index mapping for a `StateGraph`.
pub struct StateIndices {
    pub state_to_idx: FxHashMap<State, usize>,
    pub idx_to_state: Vec<State>,
}

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

impl StateGraph {
    /// Build a bidirectional state-to-index mapping for all transient states.
    pub fn state_indices(&self) -> StateIndices {
        let mut state_to_idx = FxHashMap::default();
        let mut idx_to_state = Vec::with_capacity(self.n_transient);
        for s in self.transitions.keys() {
            state_to_idx.insert(*s, idx_to_state.len());
            idx_to_state.push(*s);
        }
        StateIndices {
            state_to_idx,
            idx_to_state,
        }
    }

    /// BFS depth from the start state for each transient state.
    pub fn bfs_depths(&self) -> FxHashMap<State, u32> {
        let mut depths = FxHashMap::default();
        depths.insert(self.start, 0);
        let mut queue = VecDeque::new();
        queue.push_back(self.start);

        while let Some(cur) = queue.pop_front() {
            let d = depths[&cur];
            if let Some(transitions) = self.transitions.get(&cur) {
                for &(_action, dest) in transitions {
                    if let StateKey::Transient(ns) = dest {
                        if !depths.contains_key(&ns) {
                            depths.insert(ns, d + 1);
                            queue.push_back(ns);
                        }
                    }
                }
            }
        }
        depths
    }
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
