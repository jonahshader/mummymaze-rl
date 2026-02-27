#!/usr/bin/env python3
"""
Compare Python solver vs C solver across all 10,000 levels.

Both solvers use BFS so optimal move counts should match exactly
when the game engines agree.  Prints agreement rate.

For batch mode, runs both solvers in parallel (C solver via its batch
binary, Python solver via src/solver.py --all), then diffs the CSVs.

Usage:
    uv run python compare_solvers.py            # → "9500 / 10000 (95.0%)"
    uv run python compare_solvers.py B-5 0      # single sublevel detail
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CSIM = PROJECT_ROOT / "csolver" / "build" / "csim"
CSOLVER = PROJECT_ROOT / "csolver" / "build" / "csolver"
DAT_DIR = Path(__file__).resolve().parent / "mazes"

sys.path.insert(0, str(PROJECT_ROOT))
from mummy_maze.parser import parse_file
from src.game import load_level
from src.solver import solve


def run_single(file_stem: str, sub_idx: int) -> None:
    parsed = parse_file(DAT_DIR / f"{file_stem}.dat")
    if not parsed:
        print("parse failed")
        return

    # C solver
    result = subprocess.run(
        [str(CSIM), str(DAT_DIR), file_stem, str(sub_idx)],
        capture_output=True, text=True,
    )
    c = None
    if result.returncode == 0:
        data = json.loads(result.stdout)
        c = data["n_moves"] if data["solved"] else None

    # Python solver
    gs = load_level(parsed.sublevels[sub_idx], parsed.header)
    py_result = solve(gs)
    p = len(py_result[0]) if py_result else None

    c_str = str(c) if c is not None else "unsolvable"
    p_str = str(p) if p is not None else "unsolvable"
    if c == p:
        print(f"MATCH ({c_str})")
    else:
        print(f"DIFFER — C: {c_str}, Py: {p_str}")


def load_csv(path: Path) -> dict[tuple[str, int], int | None]:
    """Load solver CSV into {(file, sublevel): moves_or_None}."""
    results = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["file"], int(row["sublevel"]))
            moves = row["moves"].strip()
            results[key] = int(moves) if moves else None
    return results


def run_all() -> None:
    jobs = os.cpu_count() or 1

    with tempfile.TemporaryDirectory() as tmp:
        c_csv = Path(tmp) / "c.csv"
        py_csv = Path(tmp) / "py.csv"

        # Run both solvers in parallel
        c_proc = subprocess.Popen(
            [str(CSOLVER), str(DAT_DIR), "--all", "--out", str(c_csv)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        py_proc = subprocess.Popen(
            [sys.executable, "-m", "src.solver", str(DAT_DIR),
             "--all", "--jobs", str(jobs), "--out", str(py_csv)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            cwd=str(PROJECT_ROOT),
        )

        c_proc.wait()
        py_proc.wait()

        # Compare CSVs
        c_results = load_csv(c_csv)
        py_results = load_csv(py_csv)

    keys = sorted(set(c_results) & set(py_results))
    agree = sum(1 for k in keys if c_results[k] == py_results[k])
    total = len(keys)
    print(f"{agree} / {total} ({agree / total * 100:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        run_single(sys.argv[1], int(sys.argv[2]))
    else:
        run_all()
