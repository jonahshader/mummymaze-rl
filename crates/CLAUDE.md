# crates/ — Rust

## mummymaze (core library + PyO3)

Game engine, BFS solver, state graph builder, Markov chain solver. Direct port of the original binary's logic.

CLI: `src/bin/cli.rs` — batch solver and Markov analyzer.

PyO3 bindings (`import mummymaze_rust`): see `src/python.rs` for the full API.

Action indices match JAX env: N=0, S=1, E=2, W=3, Wait=4.

### Building

`uv sync` builds via maturin automatically. After Rust changes:
```
uv pip install -e crates/mummymaze/
```

CLI only:
```
cargo build --manifest-path crates/mummymaze/Cargo.toml --no-default-features --bin mummymaze-cli --release
```

## mummymaze-viewer (egui GUI)

Playable maze with undo/redo, filterable level table, 3D force-directed state graph, training metrics scatter plot, agent action probability overlay.

```
cargo run --manifest-path crates/mummymaze-viewer/Cargo.toml --release -- mazes/
```
