# TODO

Outstanding work. When an item ships, delete it; `git log` records what shipped.

## Research track

- **Re-establish the state of the curriculum track.** Nobody knows it: the work stopped on 2026-03-28 and neither Jonah nor the repo holds the state of the runs. `CURRICULUM_TRAINING.md` gives one result (BC plateaus at about 65% mean validation win probability) with no recipe, seed, checkpoint or run directory behind it. It also says the curriculum starts from a BC-pretrained `resnet-attn` checkpoint, while `experiments/curriculum_gs6.py` defaults to `--arch cnn` and its docstring says it trains from scratch. Find out what runs and checkpoints exist on this machine and in wandb, and treat every number in that doc as unverified until it is reproduced.
- **Write `docs/roles/research.md`**: what a claim needs (held-out levels, given that GA levels are mutated from real ones; training seeds), where a run lives and what it keeps, and where results are recorded. The evidence rules are Jonah's calls.
- **Write a handoff brief (`prompt_training.md`)** once the state above is known.

## Docs

- **Split `CURRICULUM_TRAINING.md`** into design (current truth) and findings (results with their recipe and run).
- **Trim `IDEAS.md`** to what is still open. Its "Next Up" lists behavioral cloning and its backlog lists adversarial GA, both built; it says the observation has 11 channels where `src/env/obs.py` has 10; and its *Backlog / Nice-to-have* section belongs in this file.

## Code

- **Redundant graph rebuilds in policy evaluation.** `compute_policy_win_probs()` and `compute_policy_win_probs_batched()` in `src/train/ga.py` call `build_graph()` per level to get state tuples, then `policy_win_prob_batch()` rebuilds each graph inside Rust. A `policy_win_prob_from_graph()` variant, or returning graphs from `ga_evaluate_batch()`, would avoid the double build.
- **`reporter` and `log_fn` are two sinks for the same metrics.** `MetricsReporter` has `report_batch`, `report_epoch_end` and so on, and the `log_fn` callback runs beside it with similar data. Wandb could be a reporter implementation, leaving one sink. Open fork, not settled.
- **`uv run vulture` reports 30 unused items**, all at 60% confidence. Some are false positives because it scans only `src/` and misses uses in `experiments/` (`run_multi_ga`, `SectionTimer`); others look real (`_resolve_collisions` in `src/game.py`, `sample_batch`, `replace_state`). Sort them, delete the dead ones, and either widen `[tool.vulture] paths` or whitelist the rest.
- **Seven tests skip unless `MUMMY_MAZE_DAT_DIR=mazes` is set** (`tests/test_vmap.py`, `tests/test_step.py`), although `mazes/` is in the repo and the other tests find it by path. They pass when it is set. Default them to the repo's `mazes/`.
