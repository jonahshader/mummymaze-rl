# Mummy Maze Deluxe — Game Rules

Reference for implementing the JAX RL environment. Sourced from the
[SourceForge tutorial](https://maze.sourceforge.net/tutorial.html) and other
community resources, plus educated guesses marked with **(guess)**.

## Turn Structure

The game is turn-based. Each turn:

1. **Player acts** — move one square (N/S/E/W) or wait (no-op). 5 actions total.
2. **Enemies act** — simultaneous. Enemy-enemy collisions are explained later.
3. **Collision check** — if any enemy occupies the player's cell, the player dies.
4. **Win check** — if the player is on the exit cell, the level is won.

## Grid

- Square grid of size 6×6, 8×8, or 10×10 (from the `.dat` header).
- Walls are stored as two edge arrays (no redundancy):
  - `h_walls[r][c]` — horizontal wall on top edge of cell `(r, c)`. Shape: `(N+1) × N`.
  - `v_walls[r][c]` — vertical wall on left edge of cell `(r, c)`. Shape: `N × (N+1)`.
- Movement checks: north = `not h_walls[r][c]`, south = `not h_walls[r+1][c]`,
  west = `not v_walls[r][c]`, east = `not v_walls[r][c+1]`.
- The exit is on the boundary of the grid (a side + position from the sublevel data).

## Entities

### Player

- Moves exactly **1 step** per turn (or waits).
- Cannot move through walls.
- Dies on contact with any enemy (mummy or scorpion).
- Dies when stepping on a trap.

### White Mummy

- Moves up to **2 steps** per turn.
- Each step follows this priority:
  1. Try to close **horizontal** distance first (move E/W toward player).
  2. If horizontal move is blocked by a wall, try to close **vertical** distance (move N/S toward player).
  3. If both are blocked, the mummy **skips** that step.
- If already on the same column as the player, moves vertically toward them.
- If already on the same row, moves horizontally toward them.

### Red Mummy

- Identical to white mummy but with **swapped priority**:
  1. Try to close **vertical** distance first.
  2. If blocked, try **horizontal**.
  3. If both blocked, skip.

### Scorpion (White / Red)

- Same movement logic as the corresponding mummy color.
- Moves only **1 step** per turn instead of 2.

### Mummy–Mummy Collision

- When two mummies end up on the same cell, one is destroyed.
- This is a core puzzle mechanic — luring enemies into each other.

### Mummy-Scorpion Collision

- When a mummy and a scorpion end up on the same cell, the scorpion is destroyed but the mummy survives.

## Traps

- Traps are fixed cells on the grid.
- The player dies when stepping onto a trap.
- Enemies (mummies and scorpions) can cross traps without harm.
- Traps remain active after being crossed by an enemy.

## Key & Gate

- A level may contain one key and one gate.
- The gate is a toggleable wall on the **south edge** of its stored cell coordinate.
  In the wall arrays, it corresponds to `h_walls[row + 1][col]`.
- When the **player** enters the key cell, the gate toggles (wall appears/disappears).
- Enemies entering the key cell also toggle the gate.
- Exiting or remaining on the key cell does nothing.
- **Assumption:** Multiple toggles can happen per turn — each entity entering the
  key cell toggles independently. If player and mummy both enter, gate toggles
  twice (net no change). Needs verification.

## Dark Pyramid Levels (Partial Observability)

- The player can only see their own cell and the 8 surrounding cells.
- All other cells are hidden (fog of war).
- Enemy positions are unknown unless within the visible region.
- Wall layout outside the visible region is unknown.
- **Assumption:** Chebyshev distance ≤ 1 (the 8 surrounding cells). Needs verification.

## Death Conditions

The player loses if:
- An enemy moves onto the player's cell (after enemy movement phase).
- The player moves onto an enemy's cell.
- The player steps on a trap.
- **Future:** The original game has a gem that turns red when the maze becomes
  unsolvable. Could be useful for early termination in the RL env.

## Win Condition

The player wins by reaching the exit cell. The exit is checked after the
player's move and after enemy movement.

## TUI Implementation Assumptions

These assumptions are used in `src/game.py` and need verification against the
original game:

1. **Enemy movement is simultaneous** — all enemies compute new positions based
   on the player's post-move position, then all positions update at once.
2. **Collision survivor** — when two enemies land on the same cell, the one with
   the lower spawn index survives. Mummies always beat scorpions.
3. **Gate as wall** — the gate is a wall on the south edge of its stored cell,
   toggled via `h_walls[row+1][col]`. Standard wall movement checks handle blocking.
4. **Exit mechanic** — the player wins by being on the exit cell and surviving
   the subsequent enemy movement phase. No extra step out of bounds required.

