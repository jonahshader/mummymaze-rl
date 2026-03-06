# CLAUDE.md

Training an RL agent to play Mummy Maze Deluxe using JAX. See `docs/IDEAS.md` for goals and research directions.

## Project Structure

- `src/` — Python/JAX: game engine, RL environment, training pipeline
- `crates/mummymaze/` — Rust: engine, BFS solver, state graph, Markov analysis (PyO3 bindings)
- `crates/mummymaze-viewer/` — Rust/egui: playable maze, state graph, training metrics
- `mazes/` — `.dat` level files (101 pyramids, 10,100 sublevels)
- `docs/` — game rules, research ideas, binary analysis notes

## Key Design Constraints

- `grid_size` is the only trace-time constant (Python int). Different grid sizes (6/8/10) produce different compiled functions.
- Everything else (is_red, has_key_gate, entity counts) is a runtime value with masks, enabling vmap across heterogeneous levels.
- Both engines (Python/JAX and Rust) must agree exactly. The JAX env is verified against 9,814 BFS solutions via `tests/test_jax_solutions.py`.
- Game rules are subtle — see `docs/GAME_RULES.md` for the full verified spec. Do not change game logic without re-running the solution replay test.

## Documentation Hygiene

Keep docs accurate — when changing code, update any docs that reference it. Prefer pointing to source of truth (code, tests) over duplicating details that go stale. If a document has served its purpose (e.g., a planning doc for completed work), suggest deleting it.
