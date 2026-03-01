# Graph View Simplify Findings

## Quick fixes — DONE

- [x] Remove dead `max_velocity` field
- [x] Remove unnecessary `sqrt` in hover hit test
- [x] Single compute pass with multiple dispatches
- [x] Extract depth-grouping helper in layout.rs
- [x] Parameterize `clip_to_world` with zoom
- [x] Unify virtual node placement
- [x] Share one bind group layout for node/edge render
- [x] Remove `SolveResult.n` field
- [x] Remove unused `FullMarkovResult.win_prob`/`expected_steps`
- [x] Revert `winning_set` to private
- [x] Upload `SimParams` once at load time
- [x] Shared WGSL struct preamble + flag constants
- [x] Use `Arc<Level>` to avoid clone on hover
- [ ] ~~Avoid `response.clone()` in tooltip~~ — required by egui API

## Medium effort — DONE

- [x] Skip BFS depths for force-directed mode
- [x] `state_to_idx`/`idx_to_state` helper on `StateGraph`
- [x] `bfs_depths` reuse from mummymaze crate

## Larger tasks (separate PRs)

- [ ] **Async graph build + Markov solve** — `main.rs:115-122`. Blocks UI thread. Needs background task + loading state. Could reuse rayon pattern from `DataStore`.
- [ ] **Sync CPU positions from GPU during force sim** — `mod.rs:329`. Positions go stale immediately. Needs GPU readback on convergence (or periodic). Hover and fit_camera use wrong data.
- [ ] **Cache `StateGraph` / per-state win probs in `LevelRow`** — Redundant `build_graph` + `analyze_full` between background analysis and graph view. Needs `DataStore` schema change.
- [ ] **Win-probs-only Markov solver** — `markov.rs`. `analyze_full` computes `expected_steps` that the graph view discards. Skip the second Gauss-Seidel loop when not needed.
