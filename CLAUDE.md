# CLAUDE.md

Training an RL agent to play Mummy Maze Deluxe using JAX. See `docs/IDEAS.md` for goals and research directions.

This file holds what nearly every session needs. Rules for one area go in that directory's `CLAUDE.md` (`src/`, `crates/`), and `docs/CLAUDE.md` indexes the docs.

## Project Structure

- `src/` — Python/JAX: game engine, RL environment, training pipeline
- `crates/mummymaze/` — Rust: engine, BFS solver, state graph, Markov analysis (PyO3 bindings)
- `crates/mummymaze-viewer/` — Rust/egui: playable maze, state graph, training metrics
- `config/` — training configs for `python -m src.train`
- `experiments/` — one-off experiment scripts (curriculum, BC vs direct)
- `tests/` — pytest suite; `notebooks/` — analysis notebooks
- `mazes/` — `.dat` level files (101 pyramids, 10,100 sublevels)
- `docs/` — game rules, research ideas, binary analysis notes, `TODO.md`

## Key Design Constraints

- `grid_size` is the only trace-time constant (Python int). Different grid sizes (6/8/10) produce different compiled functions.
- Everything else (is_red, has_key_gate, entity counts) is a runtime value with masks, enabling vmap across heterogeneous levels.
- Both engines (Python/JAX and Rust) must agree exactly. The JAX env is verified against 9,814 BFS solutions via `tests/test_jax_solutions.py`.
- Game rules are subtle — see `docs/GAME_RULES.md` for the full verified spec. Do not change game logic without re-running the solution replay test.

## Code Style Preferences

- Use callbacks/lambdas for side effects (logging, checkpointing, evaluation) rather than threading config values through function signatures. Let the caller construct the callable with whatever it needs captured in the closure.

## Version Control

Commit directly to `main` and push eagerly, as soon as a commit is checked. This is a solo repo with no pull requests, so this overrides the harness defaults of branching first and of pushing only when asked. Keep commits atomic, one logical change each.

## Documentation Hygiene

Keep docs accurate — when changing code, update any docs that reference it, in the same commit. Prefer pointing to source of truth (code, tests) over duplicating details that go stale. If a document has served its purpose (e.g., a planning doc for completed work), delete it and move anything still open to `docs/TODO.md`. The rules and the index are in `docs/CLAUDE.md`.

## Working with Claude

Auto-memory is only for facts about this machine, such as driver quirks and local paths. This overrides the harness default. Anything about the project goes in the tracked docs, in the same commit as the work.

In design discussions, work through one fork at a time and give a short recommendation.

## Roles

A session has no role unless its kickoff names one. No role docs exist yet; a research role is planned (`docs/TODO.md`, *Research track*).
