# Mummy Maze Deluxe — Game Rules

Verified against the original binary (`WinMM.exe`, FUN_00405580) via Ghidra
reverse engineering and a Rust engine port (`crates/mummymaze/`). Both
`src/game.py` (Python) and `src/env/step.py` (JAX) implement these rules
with 100% solver agreement across all 9,814 solvable levels.

## Grid

- Square grid of size 6x6, 8x8, or 10x10 (from the `.dat` header).
- Walls are stored as two edge arrays (no redundancy):
  - `h_walls[r][c]` — horizontal wall on top edge of cell `(r, c)`. Shape: `(N+1) x N`.
  - `v_walls[r][c]` — vertical wall on left edge of cell `(r, c)`. Shape: `N x (N+1)`.
- Movement checks: north = `not h_walls[r][c]`, south = `not h_walls[r+1][c]`,
  west = `not v_walls[r][c]`, east = `not v_walls[r][c+1]`.
- The exit is on the boundary (side + position from the sublevel data). The
  player wins by standing on the in-grid cell adjacent to the exit opening.

## Turn Structure

The game is turn-based. Each turn follows this exact order (matching the
binary's `FUN_00405580`):

1. **Player moves** — 1 step (N/S/E/W) or wait. Blocked by walls, gate, and
   grid bounds.
2. **Player key toggle** — if the player *entered* the key cell this turn
   (wasn't already on it), the gate toggles.
3. **Trap check** — if the player is on a trap, the player dies. Turn ends.
4. **Scorpion movement** — each scorpion moves 1 step toward the player.
5. **Scorpion-on-player check** — if any scorpion is on the player, the player
   dies. Turn ends.
6. **Scorpion-mummy collision** — if a scorpion landed on a mummy, the scorpion
   dies. (Positions preserved for later collision checks.)
7. **Scorpion key toggle** — if a *living* scorpion entered the key cell.
8. **Mummy loop** (2 iterations):
   - a. Each *living* mummy moves 1 step toward the player.
   - b. Mummy-mummy collision — if two mummies overlap, the higher-index one dies.
   - c. Mummy-scorpion collision — if a mummy is on a scorpion (dead or alive),
     the scorpion dies. If the collision is at the key cell, the gate toggles.
   - d. Mummy-on-player check — if any mummy is on the player, the player dies.
     Turn ends (no further processing).
   - e. Mummy key toggle — mutually exclusive (first matching mummy wins per
     iteration).
9. **Final enemy-on-player check** — redundant safety check after the loop.
10. **Win check** — if the player is on the exit cell and not dead, the level
    is won.

## Entities

### Player

- Moves exactly 1 step per turn (or waits).
- Cannot move through walls or the gate (when closed).
- Dies on contact with any enemy or by stepping on a trap.

### White Mummy

- Moves up to 2 steps per turn (one per mummy loop iteration).
- Each step follows this priority:
  1. Try to close **vertical** distance first (move N/S toward player).
  2. If blocked, try to close **horizontal** distance (move E/W toward player).
  3. If both are blocked, the mummy skips that step.
- If already on the same row as the player, tries horizontal only.
- If already on the same column, tries vertical only.

### Red Mummy

- Identical to white mummy but with **swapped priority**:
  1. Try to close **horizontal** distance first.
  2. If blocked, try **vertical**.
  3. If both blocked, skip.
- The `is_red` flag comes from the `.dat` header's `flip` field. An entire
  `.dat` file is either all-white or all-red.

### Scorpion

- Same movement logic as the corresponding mummy color (white or red).
- Moves only **1 step** per turn instead of 2.
- Moves **before** mummies each turn.

## Collisions

### Mummy-Mummy

When two mummies overlap after a movement step, the higher-index mummy dies.
The lower-index mummy survives. Checked per-iteration inside the mummy loop.

### Mummy-Scorpion

Mummy kills scorpion. Checked per-iteration inside the mummy loop. The binary
does **not** check `scorpion_alive` — a dead scorpion's position still triggers
the collision. This matters when a mummy sits on a dead scorpion's position
across iterations (double-toggle of the gate cancels out).

If the collision occurs at the key cell, the gate toggles (relative:
`gate_open = ~gate_open`).

### Scorpion-Mummy (pre-mummy-loop)

If a scorpion steps onto a mummy during scorpion movement (step 6), the
scorpion dies. This is checked before the mummy loop begins.

## Traps

- Fixed cells on the grid.
- The player dies immediately when stepping onto a trap (checked before enemy
  movement, so enemies don't move that turn).
- Enemies can cross traps without harm.
- Traps remain active permanently.

## Key & Gate

### Gate Geometry

The gate is a **vertical barrier** on the **east edge** of its cell position
`(gate_row, gate_col)`. It blocks:
- East movement from `(gate_row, gate_col)`
- West movement from `(gate_row, gate_col + 1)`

The gate is **independent** of the wall arrays. Any real wall at the gate
position still blocks even when the gate is open. The gate and walls are
separate blocking mechanisms.

### Gate State

- `gate_active = True` (Python) / `gate_open = False` (JAX) means the gate is
  **closed** (blocking).
- The gate starts closed when a level has a key/gate.

### Key Toggle Rules

When an entity *enters* the key cell (moves onto it from a different cell),
the gate toggles. The rules are nuanced:

1. **Entry only** — standing on the key cell doesn't toggle. Only the
   transition from "not on key" to "on key" triggers a toggle.

2. **At most one key-entry toggle per turn** — a `key_toggled` flag persists
   across all entities for the entire turn. Once any entity toggles via key
   entry, no other entity can toggle via key entry that turn.

3. **Absolute assignment** — the toggle is `gate_active = not original_gate`
   (where `original_gate` is the state at the start of the turn), not a
   relative flip. This means entering the key twice in one turn would set the
   same value, not flip twice. (Moot since at-most-one fires.)

4. **Priority order** — player toggles first (step 2), then scorpions (step 7),
   then mummies (step 8e).

5. **Mummy mutual exclusion** — within each mummy loop iteration, only the
   first matching mummy toggles (break after first match). Only mummy index 0
   sets the `key_toggled` flag; mummy index 1 can toggle the gate but doesn't
   block future toggles.

6. **Turn-start "moved" check** — mummy key toggle compares the mummy's
   current position against its **turn-start** position (`orig_mummy_pos`),
   not the per-iteration position. This means a mummy that was already on the
   key at turn start won't trigger a toggle even if it moved away and back.

### Mummy-Scorpion Kill at Key Cell

When a mummy kills a scorpion at the key cell (step 8c), the gate toggles
via a **relative** flip: `gate_active = not gate_active`. This is different
from key-entry toggles which use absolute assignment. This toggle does **not**
set `key_toggled` and does **not** check `key_toggled`.

## Death Conditions

The player loses if:
- Any enemy is on the player's cell after enemy movement.
- The player steps on a trap (checked before enemy movement).

Death checks happen at multiple points (steps 5, 8d, 9) and cause immediate
turn termination — no further movement or collisions occur after death.

## Win Condition

The player wins by standing on the exit cell after all movement and collision
resolution, provided the player is not dead.

## Dark Pyramid Levels (Partial Observability)

Some levels use fog of war:
- The player can only see their own cell and the 8 surrounding cells
  (Chebyshev distance <= 1).
- All other cells are hidden.
- Enemy positions and walls outside the visible region are unknown.
