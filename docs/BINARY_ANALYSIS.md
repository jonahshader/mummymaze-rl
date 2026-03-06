# Binary Analysis — WinMM.exe (Mummy Maze Deluxe)

Reverse-engineered via Ghidra. PE32 executable, 632KB, image base `0x400000`.

Ghidra project binary name: `WinMM.exe-283e26`

## Key Functions

| Address | Purpose |
|---|---|
| `FUN_0040e1d0` | Level loader (parses .dat files) |
| `FUN_00405580` | Step / turn execution |
| `FUN_004079c0` | `can_move` — wall/boundary checks |
| `FUN_00404f60` | State comparison (used by solver) |

## State Struct Layout

The game state is a `0x3c`-byte struct:

| Offset | Field |
|---|---|
| `0x00` | player_row |
| `0x01` | player_col |
| `0x03` | mummy1_row |
| `0x04` | mummy1_col |
| `0x06` | mummy1_alive |
| `0x07` | mummy2_row |
| `0x08` | mummy2_col |
| `0x0a` | mummy2_alive |
| `0x0b` | scorpion_row |
| `0x0c` | scorpion_col |
| `0x0e` | scorpion_alive |
| `0x39` | gate_open |

## Object/Level Memory Layout

| Offset | Field |
|---|---|
| `+0x284`, `+0x288` | Trap 1 (row, col) |
| `+0x28c`, `+0x290` | Trap 2 (row, col) |
| `+0x2a0`, `+0x2a4` | Gate (row, col) |
| `+0x2a8`, `+0x2ac` | Key (row, col) |
| `+0x300` | Wall array — indexed as `walls[col + row*10]` |
| `+0x4b0` | Flip flag |

## Wall Bitfield

Each cell's wall byte at `walls[col + row*10]`:

| Bit | Meaning |
|---|---|
| `0x01` | West wall |
| `0x02` | East wall |
| `0x04` | South wall |
| `0x08` | North wall |
| `0x10` | Exit West |
| `0x20` | Exit East |
| `0x40` | Exit South |
| `0x80` | Exit North |

## Entity Byte Order in .dat Files

The binary's loader reads entities in this order:
1. Player
2. Mummies
3. Scorpion
4. Traps
5. Gate + Key

This differs from the Python parser in `mummy-maze-parser`, which reads gate+key before scorpion+traps. The mismatch causes position swaps on levels that have (scorpion OR traps) AND a gate.

## Flip Flag

- The binary applies coordinate transforms at **load time**; the engine itself is flip-agnostic.
- Engine always indexes walls as `walls[col + row*10]` regardless of flip.
- Flip only affects: movement priority order and gate polarity.
