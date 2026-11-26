# docs/

Every file here has a job. Keep each one current as you change the code it describes, in the same commit. Docs state what is true and load-bearing now; `git log` is the chronology. A planning doc for finished work is deleted, with anything still open moved to `TODO.md`.

## Index

- `TODO.md` holds outstanding work, and is the only backlog. Delete an item when it ships.
- `GAME_RULES.md` is the verified game rules spec, checked against the original binary. Update it only with a rule change that passes `tests/test_jax_solutions.py`.
- `BINARY_ANALYSIS.md` holds the Ghidra notes on `WinMM.exe`: function addresses and what each does. Update it when more of the binary is reverse-engineered.
- `solutions.txt` is a human walkthrough from GameFAQs. It is test data for `tests/test_solutions.py`, not a doc to edit.
- `IDEAS.md` holds the project goal and research directions. Parts of it describe a past state (`TODO.md`, *Docs*).
- `CURRICULUM_TRAINING.md` covers the curriculum track: why GA levels under a learned policy, and the loop in `experiments/curriculum_gs6.py`. The track's current state is unknown and its numbers are unverified (`TODO.md`, *Research track*).
