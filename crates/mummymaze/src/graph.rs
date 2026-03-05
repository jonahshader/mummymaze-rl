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

    /// Backward BFS from WIN: distance to win for each winnable transient state.
    /// States with a direct WIN transition get distance 1; predecessors propagate outward.
    pub fn dist_to_win(&self) -> FxHashMap<State, u32> {
        // Build reverse adjacency: for each transient destination, record sources.
        let mut reverse: FxHashMap<State, Vec<State>> = FxHashMap::default();
        let mut win_predecessors: Vec<State> = Vec::new();

        for (src, transitions) in &self.transitions {
            for &(_action, dest) in transitions {
                match dest {
                    StateKey::Win => {
                        win_predecessors.push(*src);
                    }
                    StateKey::Transient(dst) => {
                        reverse.entry(dst).or_default().push(*src);
                    }
                    StateKey::Dead => {}
                }
            }
        }

        // BFS backward from WIN predecessors
        let mut dist: FxHashMap<State, u32> = FxHashMap::default();
        let mut queue = VecDeque::new();
        for s in win_predecessors {
            if !dist.contains_key(&s) {
                dist.insert(s, 1);
                queue.push_back(s);
            }
        }

        while let Some(cur) = queue.pop_front() {
            let d = dist[&cur];
            if let Some(preds) = reverse.get(&cur) {
                for &p in preds {
                    if !dist.contains_key(&p) {
                        dist.insert(p, d + 1);
                        queue.push_back(p);
                    }
                }
            }
        }

        dist
    }

    /// For every winnable state, compute a 5-bit bitmask of distance-reducing actions.
    /// Bit i is set if action i (N=0,S=1,E=2,W=3,Wait=4) leads to a successor
    /// with dist_to_win = current_dist - 1 (or directly to WIN for dist=1).
    ///
    /// Returns ALL states with win_prob > 0 (not just those on a single shortest path).
    /// Every winnable state has at least one distance-reducing action, so the mask
    /// is always non-zero. Dead-end states (win_prob = 0) are excluded.
    pub fn best_actions_per_state(&self) -> Vec<(State, u8)> {
        let dist = self.dist_to_win();
        let mut result = Vec::with_capacity(dist.len());

        for (state, &d) in &dist {
            if let Some(transitions) = self.transitions.get(state) {
                let mut mask: u8 = 0;
                for &(action, dest) in transitions {
                    let is_optimal = match dest {
                        StateKey::Win => d == 1,
                        StateKey::Transient(ns) => {
                            dist.get(&ns).is_some_and(|&nd| nd == d - 1)
                        }
                        StateKey::Dead => false,
                    };
                    if is_optimal {
                        mask |= 1 << action.to_index();
                    }
                }
                if mask != 0 {
                    result.push((*state, mask));
                }
            }
        }

        result
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
                    // Skip self-loops: they don't affect win probability,
                    // only waste turns and cause Markov convergence issues.
                    if next == cur {
                        continue;
                    }
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
