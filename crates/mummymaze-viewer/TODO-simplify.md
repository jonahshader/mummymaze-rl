# Graph View Simplify Findings

## Quick fixes (< 5 min each)

- [x] **Remove dead `max_velocity` field** — `mod.rs:62,92,334`. Set but never read.
- [x] **Remove unnecessary `sqrt` in hover hit test** — `mod.rs:531`. Compare `dist_sq < hit_r_sq` instead.
- [x] **Single compute pass with multiple dispatches** — `gpu.rs:423-431`. Move `set_pipeline`/`set_bind_group` outside loop, one `begin_compute_pass`.
- [x] **Extract depth-grouping helper in layout.rs** — `layout.rs:53-62,88-97`. Identical 15-line block duplicated.
- [x] **Parameterize `clip_to_world` with zoom** — `mod.rs:479-484,511-513`. Inline formula duplicates the method with `old_zoom`.
- [x] **Unify virtual node placement** — `mod.rs:207-232`. Both BfsLayers and RadialTree do the same `max_depth` + win/dead placement.
- [x] **Share one bind group layout for node/edge render** — `gpu.rs:66-76`. `node_render_bgl` and `edge_render_bgl` are identical.
- [x] **Remove `SolveResult.n` field** — `markov.rs:38`. Derivable from `idx_to_state.len()`.
- [x] **Remove unused `FullMarkovResult.win_prob`/`expected_steps`** — `markov.rs:25-30`. Caller only uses `state_win_probs`.
- [x] **Revert `winning_set` to private** — `metrics.rs:50`. No external callers.
- [x] **Upload `SimParams` once at load time** — `mod.rs:386-393`. Constants rebuilt and uploaded every frame.
- [x] **Shared WGSL struct preamble** — `shaders.rs`. `CameraUniform` in 3 shaders, `NodeGpu` in 3, `EdgeGpu` in 2. Extract common block.
- [x] **Replace flag magic number `0x9u` in shader** — `shaders.rs:60`. Derive from named constants in preamble.
- [x] **Use `Arc<Level>` to avoid clone on hover** — `mod.rs:568`. Cloned every hovered frame.
- [ ] ~~**Avoid `response.clone()` in tooltip**~~ — `mod.rs:559,570`. Required by `on_hover_ui` taking `self` by value. Not avoidable.

## Medium effort (15-30 min)

- [ ] **Skip BFS depths for force-directed mode** — `mod.rs:202`. Also `NodeInfo.bfs_depth` is never read by any shader.
- [ ] **`state_to_idx`/`idx_to_state` helper on `StateGraph`** — Repeated in `markov.rs`, `python.rs`, `mod.rs`. Viewer version has extra complexity (start-first, virtual nodes).
- [ ] **`bfs_depths` reuse from mummymaze crate** — `layout.rs:9-29` duplicates forward BFS in `metrics.rs:111-163`. Could be a shared utility.

## Larger tasks (separate PRs)

- [ ] **Async graph build + Markov solve** — `main.rs:115-122`. Blocks UI thread. Needs background task + loading state. Could reuse rayon pattern from `DataStore`.
- [ ] **Sync CPU positions from GPU during force sim** — `mod.rs:329`. Positions go stale immediately. Needs GPU readback on convergence (or periodic). Hover and fit_camera use wrong data.
- [ ] **Cache `StateGraph` / per-state win probs in `LevelRow`** — Redundant `build_graph` + `analyze_full` between background analysis and graph view. Needs `DataStore` schema change.
- [ ] **Win-probs-only Markov solver** — `markov.rs`. `analyze_full` computes `expected_steps` that the graph view discards. Skip the second Gauss-Seidel loop when not needed.
