#!/usr/bin/env python3
"""
Compare Python parser output vs C parser output across all 10,100 levels.

Prints average similarity score (0.0–100.0%) weighted equally across:
  - wall bits (4 bits per cell, N*N cells)
  - entity positions (each entity = 1 if match, 0 if not)
  - exit (position + side)

Usage:
    uv run python compare_parsers.py            # → "42.3%"
    uv run python compare_parsers.py B-5 0      # single sublevel detail
"""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CSIM = PROJECT_ROOT / "csolver" / "build" / "csim"
DAT_DIR = Path(__file__).resolve().parent / "mazes"

sys.path.insert(0, str(PROJECT_ROOT))
from mummy_maze.parser import EntityType, parse_file

WALL_W, WALL_E, WALL_S, WALL_N = 1, 2, 4, 8
ENTITY_FIELDS = ["player", "mummy1", "mummy2", "scorpion", "gate", "key", "trap1", "trap2"]


def py_walls_to_bitmask(h_walls: list[list[bool]], v_walls: list[list[bool]], n: int) -> list[int]:
    """Convert Python h_walls/v_walls edge arrays to C-style per-cell bitmask."""
    walls = [0] * (n * n)
    for r in range(n):
        for c in range(n):
            idx = r * n + c
            if h_walls[r][c]:
                walls[idx] |= WALL_N
            if h_walls[r + 1][c]:
                walls[idx] |= WALL_S
            if v_walls[r][c]:
                walls[idx] |= WALL_W
            if v_walls[r][c + 1]:
                walls[idx] |= WALL_E
    return walls


def get_c_level(file_stem: str, sub_idx: int) -> dict | None:
    result = subprocess.run(
        [str(CSIM), str(DAT_DIR), file_stem, str(sub_idx), "."],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)["level"]


def get_py_level(file_stem: str, sub_idx: int, parsed: object) -> dict | None:
    if sub_idx >= len(parsed.sublevels):
        return None

    sub = parsed.sublevels[sub_idx]
    hdr = parsed.header
    n = hdr.grid_size

    player = None
    mummies: list[tuple[int, int]] = []
    scorpions: list[tuple[int, int]] = []
    traps: list[tuple[int, int]] = []
    gate = None
    key = None
    for ent in sub.entities:
        pos = (ent.row, ent.col)
        if ent.type == EntityType.PLAYER:
            player = pos
        elif ent.type == EntityType.MUMMY:
            mummies.append(pos)
        elif ent.type == EntityType.SCORPION:
            scorpions.append(pos)
        elif ent.type == EntityType.TRAP:
            traps.append(pos)
        elif ent.type == EntityType.GATE:
            gate = pos
        elif ent.type == EntityType.KEY:
            key = pos

    walls = py_walls_to_bitmask(sub.h_walls, sub.v_walls, n)

    exit_mask_map = {"N": 0x80, "S": 0x40, "W": 0x10, "E": 0x20}
    exit_cell = {"N": (0, sub.exit_pos), "S": (n - 1, sub.exit_pos),
                 "W": (sub.exit_pos, 0), "E": (sub.exit_pos, n - 1)}.get(sub.exit_side)

    return {
        "player": list(player) if player else None,
        "mummy1": list(mummies[0]) if len(mummies) >= 1 else None,
        "mummy2": list(mummies[1]) if len(mummies) >= 2 else None,
        "scorpion": list(scorpions[0]) if scorpions else None,
        "gate": list(gate) if gate else None,
        "key": list(key) if key else None,
        "trap1": list(traps[0]) if len(traps) >= 1 else None,
        "trap2": list(traps[1]) if len(traps) >= 2 else None,
        "exit": list(exit_cell) if exit_cell else None,
        "exit_mask": exit_mask_map.get(sub.exit_side, 0),
        "walls": walls,
    }


def level_similarity(c: dict, p: dict) -> float:
    """Return similarity score between 0.0 and 1.0 for one level.

    Scores three components equally (each 0–1):
      walls:    fraction of wall bits that match (4 bits * N*N cells)
      entities: fraction of present entities with matching positions
      exit:     1 if both position and side match, 0 otherwise
    """
    n = c["grid_size"]

    # --- Walls: compare individual bits ---
    c_walls = [w & 0x0F for w in c["walls"]]
    total_bits = 0
    matching_bits = 0
    for i in range(n * n):
        xor = c_walls[i] ^ p["walls"][i]
        total_bits += 4
        matching_bits += 4 - bin(xor).count("1")
    wall_score = matching_bits / total_bits

    # --- Entities: fraction that match ---
    entity_checks = 0
    entity_matches = 0
    for field in ENTITY_FIELDS:
        c_val = c.get(field)
        if c_val == [99, 99]:
            c_val = None
        p_val = p.get(field)
        if c_val is None and p_val is None:
            continue  # both absent, skip
        entity_checks += 1
        if c_val == p_val:
            entity_matches += 1
    entity_score = entity_matches / entity_checks if entity_checks else 1.0

    # --- Exit ---
    exit_score = 1.0 if (c["exit"] == p["exit"] and c["exit_mask"] == p["exit_mask"]) else 0.0

    return (wall_score + entity_score + exit_score) / 3.0


def run_single(file_stem: str, sub_idx: int) -> None:
    parsed = parse_file(DAT_DIR / f"{file_stem}.dat")
    if not parsed:
        print("parse failed")
        return
    c = get_c_level(file_stem, sub_idx)
    p = get_py_level(file_stem, sub_idx, parsed)
    if c is None or p is None:
        print("parse failed")
        return

    sim = level_similarity(c, p)
    print(f"{sim * 100:.1f}%")

    # Also print per-field detail
    n = c["grid_size"]
    for field in ENTITY_FIELDS:
        c_val = c.get(field)
        if c_val == [99, 99]:
            c_val = None
        p_val = p.get(field)
        if c_val is None and p_val is None:
            continue
        status = "ok" if c_val == p_val else f"C={c_val} Py={p_val}"
        print(f"  {field}: {status}")
    c_walls = [w & 0x0F for w in c["walls"]]
    wall_diffs = sum(1 for i in range(n * n) if c_walls[i] != p["walls"][i])
    print(f"  walls: {n*n - wall_diffs}/{n*n} cells match")
    exit_ok = c["exit"] == p["exit"] and c["exit_mask"] == p["exit_mask"]
    if exit_ok:
        print("  exit: ok")
    else:
        print(f"  exit: C={c['exit']} mask={c['exit_mask']} Py={p['exit']} mask={p['exit_mask']}")


def run_all() -> None:
    stems = sorted(p.stem for p in DAT_DIR.glob("B-*.dat"))
    total = 0
    total_sim = 0.0
    for stem in stems:
        parsed = parse_file(DAT_DIR / f"{stem}.dat")
        if not parsed:
            continue
        for si in range(len(parsed.sublevels)):
            c = get_c_level(stem, si)
            p = get_py_level(stem, si, parsed)
            if c is None or p is None:
                continue
            total += 1
            total_sim += level_similarity(c, p)
    print(f"{total_sim / total * 100:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        run_single(sys.argv[1], int(sys.argv[2]))
    else:
        run_all()
