//! BFS shortest-path solver — port of csolver/src/solver.c

use crate::game::{Action, State, StepResult, step};
use crate::parse::Level;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct SolveResult {
    /// Solution length, or None if unsolvable
    pub moves: Option<u32>,
    /// Full action sequence, or None if unsolvable
    pub actions: Option<Vec<Action>>,
    /// Total states visited during BFS
    pub states_explored: usize,
}

/// BFS solver. Returns optimal (shortest) solution with full action path.
pub fn solve(lev: &Level) -> SolveResult {
    let init = State::from_level(lev);

    let mut visited = FxHashSet::default();
    visited.insert(init);

    // parent map: state -> (parent_state, action_taken)
    let mut parent: FxHashMap<State, (State, Action)> = FxHashMap::default();

    let mut queue: VecDeque<(State, u32)> = VecDeque::new();
    queue.push_back((init, 0));

    while let Some((cur, depth)) = queue.pop_front() {
        for &action in &Action::ALL {
            let mut next = cur;
            let result = step(lev, &mut next, action);

            if result == StepResult::Win {
                // Reconstruct path
                let mut actions = vec![action];
                let mut trace = cur;
                while let Some(&(prev, act)) = parent.get(&trace) {
                    actions.push(act);
                    trace = prev;
                }
                actions.reverse();
                return SolveResult {
                    moves: Some(depth + 1),
                    actions: Some(actions),
                    states_explored: visited.len(),
                };
            }

            if result == StepResult::Dead {
                continue;
            }

            if visited.insert(next) {
                parent.insert(next, (cur, action));
                queue.push_back((next, depth + 1));
            }
        }
    }

    SolveResult {
        moves: None,
        actions: None,
        states_explored: visited.len(),
    }
}
