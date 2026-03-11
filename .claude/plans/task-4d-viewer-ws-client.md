# Task 4d — Viewer as WebSocket Client

Replace subprocess-based ModelServer with WebSocket client in the Rust viewer.

## Design Decisions

- **`tungstenite`** (sync WS library) — matches egui's sync polling model
- **`WsClient` lives in viewer crate** — avoids adding tungstenite to the PyO3 library crate
- **Shared types** (`TrainingEvent`, `LevelMetric`) extracted from `model_server.rs` before deletion
- **Reader thread** pattern: same as current model_server.rs reader loop
- **Level gen `use_policy`** removed — adversarial tab is the proper tool for policy-guided GA

## Sub-commits

### 4d-1: Extract shared types from model_server.rs
- Create `crates/mummymaze/src/event_types.rs` with `TrainingEvent`, `LevelMetric`, `DatasetInfo`
- Update `model_server.rs` to re-export from `event_types.rs`
- Add `pub mod event_types;` to `lib.rs`

### 4d-2: Add WsClient to viewer crate
- Add `tungstenite` to viewer Cargo.toml
- Create `crates/mummymaze-viewer/src/ws_client.rs`
  - `WsClient::connect(url)` — connects, spawns reader thread
  - `evaluate(level_key)` — blocking, oneshot response
  - `send_train(config)` / `send_stop_train()`
  - `send_adversarial(config)` / `send_stop_adversarial()`
  - `send_reload_checkpoint(path)` / `send_shutdown()`
  - `poll_events()` — non-blocking drain of `ServerEvent` channel
- New types: `ServerEvent`, `AdversarialEvent`

### 4d-3: Refactor training path
- `data/training.rs`: `ModelServer` → `WsClient` in start/stop/poll
- `training_tab.rs`: update ModelServer references
- `data/mod.rs`: remove `TrainingMetrics` field (metrics come via WS)
- Delete `training_metrics.rs`

### 4d-4: Replace AgentProbs with on-demand evaluate
- Delete `agent_probs.rs`
- `main.rs`: replace mmap reader with async evaluate + cache
  - Background thread calls `ws_client.evaluate(key)`
  - Poll for result, cache in `FxHashMap<State, [f32; 5]>`
- Remove `memmap2` from viewer Cargo.toml

### 4d-5: Refactor adversarial tab
- Rewrite `adversarial.rs` as thin WS message forwarder
  - `start()` → sends `{"type":"adversarial","config":{...}}`
  - `stop()` → sends `{"type":"stop_adversarial"}`
  - `poll()` → drains adversarial events, updates UI state
  - Keep chart/heatmap data structures, populate from events
- `adversarial_tab.rs`: update references

### 4d-6: Refactor level gen tab + delete dead code
- Remove `use_policy` from level gen tab (adversarial tab is the right tool)
- Delete `crates/mummymaze/src/model_server.rs`
- Remove `pub mod model_server;` from lib.rs
- Delete `run_ga_with_model_server*` and `PolicyQuery` from `ga/mod.rs`
- Clean up Cargo.toml deps

## Connection Model
- Server must be started externally (`uv run python -m src.train.ws_server`)
- Viewer connects on startup (or via "Connect" button)
- Graceful handling of connection failure
- Single connection, all message types multiplexed
