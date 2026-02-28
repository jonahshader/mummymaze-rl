//! BFS shortest-path solver — port of csolver/src/solver.c

use crate::game::{Action, State, StepResult, step};
use crate::parse::Level;
use rustc_hash::FxHashSet;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct SolveResult {
    /// Solution length, or None if unsolvable
    pub moves: Option<u32>,
    /// Total states visited during BFS
    pub states_explored: usize,
}

/// BFS solver. Returns optimal (shortest) solution length.
pub fn solve(lev: &Level) -> SolveResult {
    let init = State::from_level(lev);

    let mut visited = FxHashSet::default();
    visited.insert(init);

    let mut queue: VecDeque<(State, u32)> = VecDeque::new();
    queue.push_back((init, 0));

    while let Some((cur, depth)) = queue.pop_front() {
        for &action in &Action::ALL {
            let mut next = cur;
            let result = step(lev, &mut next, action);

            if result == StepResult::Win {
                return SolveResult {
                    moves: Some(depth + 1),
                    states_explored: visited.len(),
                };
            }

            if result == StepResult::Dead {
                continue;
            }

            if visited.insert(next) {
                queue.push_back((next, depth + 1));
            }
        }
    }

    SolveResult {
        moves: None,
        states_explored: visited.len(),
    }
}
