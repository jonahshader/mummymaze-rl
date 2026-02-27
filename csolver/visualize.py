#!/usr/bin/env python3
"""
Side-by-side maze solver comparison visualizer.

Runs both the C solver and Python solver on a given level, then generates
an HTML file that lets you step through both solutions simultaneously.

Usage:
    uv run python csolver/visualize.py B-11 0
    uv run python csolver/visualize.py B-5 1
    uv run python csolver/visualize.py --all-diffs  # list all disagreements
"""

import argparse
import copy
import html
import json
import subprocess
import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSIM = PROJECT_ROOT / "csolver" / "build" / "csim"
DAT_DIR = Path("/home/jonah/repos/mummy-maze-parser/Mummy Maze Deluxe/data/mazes")

# Ensure the Python engine is importable
sys.path.insert(0, str(PROJECT_ROOT))

from mummy_maze.parser import parse_file, EntityType
from src.game import (
    ACTION_EAST,
    ACTION_NORTH,
    ACTION_SOUTH,
    ACTION_WAIT,
    ACTION_WEST,
    load_level,
    step,
)
from src.solver import solve, state_key

ACTIONS = [ACTION_NORTH, ACTION_SOUTH, ACTION_EAST, ACTION_WEST, ACTION_WAIT]
ACTION_NAMES = {
    ACTION_NORTH: "N",
    ACTION_SOUTH: "S",
    ACTION_EAST: "E",
    ACTION_WEST: "W",
    ACTION_WAIT: ".",
}
# For converting C solver action strings back to Python action ints
ACTION_FROM_NAME = {v: k for k, v in ACTION_NAMES.items()}


def run_python_solver(file_stem: str, sub_idx: int):
    """Run the Python BFS solver, return (level_info, actions, states_per_step)."""
    dat_path = DAT_DIR / f"{file_stem}.dat"
    parsed = parse_file(dat_path)
    if not parsed or sub_idx >= len(parsed.sublevels):
        return None

    sublevel = parsed.sublevels[sub_idx]
    hdr = parsed.header
    gs = load_level(sublevel, hdr)

    # Solve
    result = solve(gs)
    if result is None:
        actions = []
    else:
        actions, _ = result

    # Replay to get states at each step
    gs = load_level(sublevel, hdr)
    states = [snapshot_py(gs)]
    for act in actions:
        step(gs, act)
        states.append(snapshot_py(gs))
        if not gs.alive or gs.won:
            break

    # Build level info in Python's coordinate system
    level_info = {
        "grid_size": hdr.grid_size,
        "flip": int(hdr.flip),
        "h_walls": gs.h_walls,
        "v_walls": gs.v_walls,
        "gate_cell": list(gs.gate_cell) if gs.gate_cell else None,
        "key_pos": list(gs.key_pos) if gs.key_pos else None,
        "traps": [list(t) for t in gs.traps],
        "exit_side": gs.exit_side,
        "exit_pos": gs.exit_pos,
        "exit_cell": list(gs.exit_cell),
    }

    return {
        "actions": [ACTION_NAMES[a] for a in actions],
        "states": states,
        "level": level_info,
        "solved": len(actions) > 0,
        "n_moves": len(actions),
    }


def snapshot_py(gs):
    """Snapshot the Python GameState into a dict matching the visualizer format."""
    alive = gs.alive
    won = gs.won
    result = 0 if alive and not won else (2 if won else 1)
    return {
        "player": list(gs.player),
        "mummies": [list(m) for m in gs.mummies],
        "scorpions": [list(s) for s in gs.scorpions],
        "gate_active": gs.gate_active,
        "result": result,
    }


def replay_on_python(file_stem: str, sub_idx: int, action_names: list[str]):
    """Replay a given sequence of moves on the Python engine."""
    dat_path = DAT_DIR / f"{file_stem}.dat"
    parsed = parse_file(dat_path)
    if not parsed or sub_idx >= len(parsed.sublevels):
        return None

    sublevel = parsed.sublevels[sub_idx]
    hdr = parsed.header
    gs = load_level(sublevel, hdr)

    actions = [ACTION_FROM_NAME[a] for a in action_names]

    states = [snapshot_py(gs)]
    for act in actions:
        step(gs, act)
        states.append(snapshot_py(gs))
        if not gs.alive or gs.won:
            break

    level_info = {
        "grid_size": hdr.grid_size,
        "flip": int(hdr.flip),
        "h_walls": gs.h_walls,
        "v_walls": gs.v_walls,
        "gate_cell": list(gs.gate_cell) if gs.gate_cell else None,
        "key_pos": list(gs.key_pos) if gs.key_pos else None,
        "traps": [list(t) for t in gs.traps],
        "exit_side": gs.exit_side,
        "exit_pos": gs.exit_pos,
        "exit_cell": list(gs.exit_cell),
    }

    last = states[-1]
    solved = last["result"] == 2
    return {
        "actions": list(action_names[: len(states) - 1]),
        "states": states,
        "level": level_info,
        "solved": solved,
        "n_moves": len(states) - 1,
    }


def run_c_solver(file_stem: str, sub_idx: int):
    """Run the C solver via csim, return parsed JSON."""
    result = subprocess.run(
        [str(CSIM), str(DAT_DIR), file_stem, str(sub_idx)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"csim error: {result.stderr}", file=sys.stderr)
        return None
    return json.loads(result.stdout)


def generate_html(file_stem: str, sub_idx: int, py_data, c_data, cross_data=None):
    """Generate a self-contained HTML file with side-by-side visualization.

    If cross_data is provided, a third panel shows the C solver's moves
    replayed on the Python engine.
    """
    py_json = json.dumps(py_data)
    c_json = json.dumps(c_data)
    cross_json = json.dumps(cross_data) if cross_data else "null"

    title = f"{file_stem} sub {sub_idx}"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)} — Solver Comparison</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'SF Mono', 'Menlo', 'Consolas', monospace; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
h1 {{ text-align: center; margin-bottom: 10px; font-size: 18px; color: #7fdbca; }}
.info {{ text-align: center; margin-bottom: 15px; font-size: 13px; color: #888; }}
.container {{ display: flex; gap: 30px; justify-content: center; align-items: flex-start; }}
.panel {{ background: #16213e; border-radius: 8px; padding: 15px; }}
.panel h2 {{ text-align: center; margin-bottom: 8px; font-size: 15px; }}
.py-panel h2 {{ color: #82aaff; }}
.c-panel h2 {{ color: #c792ea; }}
.cross-panel h2 {{ color: #f78c6c; }}
canvas {{ display: block; margin: 0 auto; }}
.controls {{ text-align: center; margin-top: 15px; }}
.controls button {{
    background: #0f3460; border: 1px solid #3a506b; color: #e0e0e0;
    padding: 6px 16px; margin: 0 4px; cursor: pointer; border-radius: 4px;
    font-family: inherit; font-size: 13px;
}}
.controls button:hover {{ background: #1a5276; }}
.controls button:disabled {{ opacity: 0.3; cursor: default; }}
.step-info {{ text-align: center; margin-top: 8px; font-size: 13px; color: #aaa; }}
.status {{ text-align: center; margin-top: 4px; font-size: 13px; }}
.status.ok {{ color: #addb67; }}
.status.dead {{ color: #ff5370; }}
.status.win {{ color: #ffd700; }}
.status.unsolvable {{ color: #ff5370; }}
.diff {{ text-align: center; margin-top: 10px; padding: 8px; background: #2a1a3e;
         border-radius: 4px; font-size: 13px; color: #f78c6c; }}
.legend {{ text-align: center; margin-top: 15px; font-size: 12px; color: #666; }}
.legend span {{ margin: 0 8px; }}
.action-bar {{ display: flex; justify-content: center; gap: 2px; margin-top: 8px;
               flex-wrap: wrap; max-width: 600px; margin-left: auto; margin-right: auto; }}
.action-bar .act {{ width: 20px; height: 20px; line-height: 20px; text-align: center;
                    font-size: 10px; border-radius: 2px; background: #0f3460; color: #888; }}
.action-bar .act.current {{ background: #1a5276; color: #fff; font-weight: bold; }}
.action-bar .act.past {{ background: #1f4068; color: #aaa; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="info" id="levelinfo"></div>
<div class="controls">
    <button id="btn-start" onclick="goStart()">|&lt;</button>
    <button id="btn-prev" onclick="goPrev()">&lt; Prev</button>
    <button id="btn-next" onclick="goNext()">Next &gt;</button>
    <button id="btn-end" onclick="goEnd()">&gt;|</button>
    <button id="btn-play" onclick="togglePlay()">Play</button>
</div>
<div class="step-info" id="stepinfo"></div>
<div id="action-bars"></div>
<div class="container">
    <div class="panel py-panel">
        <h2>Python Solver</h2>
        <canvas id="py-canvas"></canvas>
        <div class="status" id="py-status"></div>
    </div>
    <div class="panel c-panel">
        <h2>C Solver</h2>
        <canvas id="c-canvas"></canvas>
        <div class="status" id="c-status"></div>
    </div>
    <div class="panel cross-panel" id="cross-panel-container" style="display:none">
        <h2 style="color:#f78c6c">C Moves on Py Engine</h2>
        <canvas id="cross-canvas"></canvas>
        <div class="status" id="cross-status"></div>
    </div>
</div>
<div class="diff" id="diff"></div>
<div class="legend">
    <span>&#x1F534; Player</span>
    <span>&#x1F7E2; Mummy</span>
    <span>&#x1F7E1; Scorpion</span>
    <span>&#x26A0;&#xFE0F; Trap</span>
    <span>&#x1F511; Key</span>
    <span>&#x1F6A7; Gate</span>
    <span>&#x1F6AA; Exit</span>
</div>

<script>
const pyData = {py_json};
const cData = {c_json};
const crossData = {cross_json};

let currentStep = 0;
let maxStep = 0;
let playing = false;
let playTimer = null;

const CELL = 48;
const PAD = 20;
const WALL_W = 2;

function init() {{
    const pyN = pyData.level.grid_size;
    const cN = cData.level.grid_size;
    const panels = [['py-canvas', pyN], ['c-canvas', cN]];

    if (crossData) {{
        panels.push(['cross-canvas', crossData.level.grid_size]);
        document.getElementById('cross-panel-container').style.display = '';
        maxStep = Math.max(pyData.states.length, cData.states.length, crossData.states.length) - 1;
    }} else {{
        maxStep = Math.max(pyData.states.length, cData.states.length) - 1;
    }}

    // Set canvas sizes
    for (const [id, n] of panels) {{
        const c = document.getElementById(id);
        c.width = n * CELL + 2 * PAD;
        c.height = n * CELL + 2 * PAD;
    }}

    // Level info
    const info = document.getElementById('levelinfo');
    let infoText = `Grid: ${{cN}} | Flip: ${{cData.level.flip}} | ` +
        `Gate: ${{cData.level.has_gate ? 'yes' : 'no'}} | ` +
        `Python: ${{pyData.solved ? pyData.n_moves + ' moves' : 'UNSOLVABLE'}} | ` +
        `C: ${{cData.solved ? cData.n_moves + ' moves' : 'UNSOLVABLE'}}`;
    if (crossData) {{
        const lastCross = crossData.states[crossData.states.length - 1];
        const crossResult = lastCross.result === 2 ? 'WIN' : lastCross.result === 1 ? 'DEAD' : 'stuck';
        infoText += ` | Cross: ${{crossResult}} at step ${{crossData.states.length - 1}}`;
    }}
    info.textContent = infoText;

    // Action bars
    buildActionBars();
    draw();
}}

function buildActionBars() {{
    const container = document.getElementById('action-bars');
    container.innerHTML = '';
    const sources = [['Py', pyData, 'py'], ['C', cData, 'c']];
    if (crossData) sources.push(['X', crossData, 'cross']);
    for (const [label, data, cls] of sources) {{
        if (!data.actions.length) continue;
        const bar = document.createElement('div');
        bar.className = 'action-bar';
        bar.id = `bar-${{cls}}`;
        const lbl = document.createElement('span');
        lbl.style.cssText = 'font-size:11px;color:#888;margin-right:4px;line-height:20px;';
        lbl.textContent = label + ':';
        bar.appendChild(lbl);
        for (let i = 0; i < data.actions.length; i++) {{
            const d = document.createElement('div');
            d.className = 'act';
            d.textContent = data.actions[i];
            d.dataset.idx = i;
            d.onclick = () => {{ currentStep = i + 1; draw(); }};
            bar.appendChild(d);
        }}
        container.appendChild(bar);
    }}
}}

function updateActionBars() {{
    for (const cls of ['py', 'c', 'cross']) {{
        const bar = document.getElementById(`bar-${{cls}}`);
        if (!bar) continue;
        const acts = bar.querySelectorAll('.act');
        acts.forEach((el, i) => {{
            el.className = 'act';
            if (i + 1 === currentStep) el.classList.add('current');
            else if (i + 1 < currentStep) el.classList.add('past');
        }});
    }}
}}

function drawMaze(canvasId, levelData, stateData, isPython) {{
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const N = levelData.grid_size;
    const W = canvas.width, H = canvas.height;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0d1b2a';
    ctx.fillRect(0, 0, W, H);

    const ox = PAD, oy = PAD;

    // Draw cells
    ctx.fillStyle = '#1b2838';
    for (let r = 0; r < N; r++)
        for (let c = 0; c < N; c++)
            ctx.fillRect(ox + c * CELL + 1, oy + r * CELL + 1, CELL - 2, CELL - 2);

    if (isPython) {{
        drawMazePython(ctx, levelData, stateData, N, ox, oy);
    }} else {{
        drawMazeC(ctx, levelData, stateData, N, ox, oy);
    }}
}}

function drawMazePython(ctx, level, state, N, ox, oy) {{
    const hw = level.h_walls;
    const vw = level.v_walls;

    // Draw walls
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = WALL_W;

    // Horizontal walls (top/bottom edges)
    for (let r = 0; r <= N; r++) {{
        for (let c = 0; c < N; c++) {{
            if (hw[r][c]) {{
                const x1 = ox + c * CELL, y = oy + r * CELL;
                ctx.beginPath();
                ctx.moveTo(x1, y);
                ctx.lineTo(x1 + CELL, y);
                ctx.stroke();
            }}
        }}
    }}

    // Vertical walls (left/right edges)
    for (let r = 0; r < N; r++) {{
        for (let c = 0; c <= N; c++) {{
            if (vw[r][c]) {{
                const x = ox + c * CELL, y1 = oy + r * CELL;
                ctx.beginPath();
                ctx.moveTo(x, y1);
                ctx.lineTo(x, y1 + CELL);
                ctx.stroke();
            }}
        }}
    }}

    // Gate wall (if active) — gate is on the east edge of gate_cell
    if (level.gate_cell && state.gate_active) {{
        const [gr, gc] = level.gate_cell;
        ctx.strokeStyle = '#ff5370';
        ctx.lineWidth = 3;
        const x = ox + (gc + 1) * CELL, y1 = oy + gr * CELL;
        ctx.beginPath();
        ctx.moveTo(x, y1);
        ctx.lineTo(x, y1 + CELL);
        ctx.stroke();
        ctx.lineWidth = WALL_W;
        ctx.strokeStyle = '#e0e0e0';
    }}

    // Exit marker
    drawExitPython(ctx, level, N, ox, oy);

    // Static entities
    if (level.key_pos) drawCircle(ctx, ox, oy, level.key_pos[0], level.key_pos[1], '#ffd700', 'K');
    for (const t of level.traps) drawCircle(ctx, ox, oy, t[0], t[1], '#ff9800', 'T');

    // Dynamic entities from state
    for (const m of state.mummies) {{
        if (m[0] >= 0 && m[0] < N) drawCircle(ctx, ox, oy, m[0], m[1], '#4caf50', 'M');
    }}
    for (const s of state.scorpions) {{
        if (s[0] >= 0 && s[0] < N) drawCircle(ctx, ox, oy, s[0], s[1], '#ffeb3b', 'S');
    }}
    drawCircle(ctx, ox, oy, state.player[0], state.player[1], '#f44336', 'P');
}}

function drawMazeC(ctx, level, state, N, ox, oy) {{
    const walls = level.walls; // flat array, row-major: walls[row*N + col]

    // Draw walls from the C wall bits
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = WALL_W;

    for (let r = 0; r < N; r++) {{
        for (let c = 0; c < N; c++) {{
            const w = walls[r * N + c];
            const x = ox + c * CELL, y = oy + r * CELL;
            // N wall (bit 3 = 8): top edge
            if (w & 8) {{ ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + CELL, y); ctx.stroke(); }}
            // S wall (bit 2 = 4): bottom edge
            if (w & 4) {{ ctx.beginPath(); ctx.moveTo(x, y + CELL); ctx.lineTo(x + CELL, y + CELL); ctx.stroke(); }}
            // W wall (bit 0 = 1): left edge
            if (w & 1) {{ ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y + CELL); ctx.stroke(); }}
            // E wall (bit 1 = 2): right edge
            if (w & 2) {{ ctx.beginPath(); ctx.moveTo(x + CELL, y); ctx.lineTo(x + CELL, y + CELL); ctx.stroke(); }}
        }}
    }}

    // Gate (virtual wall, drawn when gate_open=1 ie closed)
    if (level.has_gate && state.gate) {{
        // Gate blocks E wall of gate cell in C coords
        const gr = level.gate[0], gc = level.gate[1];
        if (gr < N && gc < N) {{
            ctx.strokeStyle = '#ff5370';
            ctx.lineWidth = 3;
            // E wall of gate cell
            const x = ox + (gc + 1) * CELL, y1 = oy + gr * CELL;
            ctx.beginPath(); ctx.moveTo(x, y1); ctx.lineTo(x, y1 + CELL); ctx.stroke();
            ctx.lineWidth = WALL_W;
            ctx.strokeStyle = '#e0e0e0';
        }}
    }}

    // Exit marker
    drawExitC(ctx, level, N, ox, oy);

    // Static entities
    if (level.has_gate) drawCircle(ctx, ox, oy, level.key[0], level.key[1], '#ffd700', 'K');
    if (level.trap_count >= 1) drawCircle(ctx, ox, oy, level.trap1[0], level.trap1[1], '#ff9800', 'T');
    if (level.trap_count >= 2) drawCircle(ctx, ox, oy, level.trap2[0], level.trap2[1], '#ff9800', 'T');

    // Dynamic entities
    if (state.m1a) drawCircle(ctx, ox, oy, state.m1r, state.m1c, '#4caf50', 'M');
    if (state.m2a) drawCircle(ctx, ox, oy, state.m2r, state.m2c, '#4caf50', 'M');
    if (state.sa) drawCircle(ctx, ox, oy, state.sr, state.sc, '#ffeb3b', 'S');
    drawCircle(ctx, ox, oy, state.pr, state.pc, '#f44336', 'P');
}}

function drawCircle(ctx, ox, oy, row, col, color, label) {{
    const cx = ox + col * CELL + CELL / 2;
    const cy = oy + row * CELL + CELL / 2;
    const r = CELL * 0.32;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#000';
    ctx.font = 'bold 14px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, cx, cy);
}}

function drawExitPython(ctx, level, N, ox, oy) {{
    ctx.fillStyle = '#7fdbca';
    const ep = level.exit_pos;
    const s = level.exit_side;
    let x, y, w, h;
    if (s === 'N') {{ x = ox + ep * CELL + 4; y = oy - 6; w = CELL - 8; h = 8; }}
    else if (s === 'S') {{ x = ox + ep * CELL + 4; y = oy + N * CELL - 2; w = CELL - 8; h = 8; }}
    else if (s === 'W') {{ x = ox - 6; y = oy + ep * CELL + 4; w = 8; h = CELL - 8; }}
    else {{ x = ox + N * CELL - 2; y = oy + ep * CELL + 4; w = 8; h = CELL - 8; }}
    ctx.fillRect(x, y, w, h);
}}

function drawExitC(ctx, level, N, ox, oy) {{
    ctx.fillStyle = '#7fdbca';
    const [er, ec] = level.exit;
    const mask = level.exit_mask;
    let x, y, w, h;
    if (mask === 0x80) {{ /* N */ x = ox + ec * CELL + 4; y = oy - 6; w = CELL - 8; h = 8; }}
    else if (mask === 0x40) {{ /* S */ x = ox + ec * CELL + 4; y = oy + N * CELL - 2; w = CELL - 8; h = 8; }}
    else if (mask === 0x10) {{ /* W */ x = ox - 6; y = oy + er * CELL + 4; w = 8; h = CELL - 8; }}
    else {{ /* E */ x = ox + N * CELL - 2; y = oy + er * CELL + 4; w = 8; h = CELL - 8; }}
    ctx.fillRect(x, y, w, h);
}}

function draw() {{
    const pyState = pyData.states[Math.min(currentStep, pyData.states.length - 1)];
    const cState = cData.states[Math.min(currentStep, cData.states.length - 1)];

    drawMaze('py-canvas', pyData.level, pyState, true);
    drawMaze('c-canvas', cData.level, cState, false);

    if (crossData) {{
        const crossState = crossData.states[Math.min(currentStep, crossData.states.length - 1)];
        drawMaze('cross-canvas', crossData.level, crossState, true);
        setStatus('cross-status', crossState.result, currentStep > crossData.states.length - 1);
    }}

    // Step info
    const stepInfo = document.getElementById('stepinfo');
    if (currentStep === 0) {{
        stepInfo.textContent = 'Initial state';
    }} else {{
        const pyAct = currentStep <= pyData.actions.length ? pyData.actions[currentStep - 1] : '-';
        const cAct = currentStep <= cData.actions.length ? cData.actions[currentStep - 1] : '-';
        let txt = `Step ${{currentStep}} / ${{maxStep}} — Py: ${{pyAct}} | C: ${{cAct}}`;
        if (crossData) {{
            const xAct = currentStep <= crossData.actions.length ? crossData.actions[currentStep - 1] : '-';
            txt += ` | X: ${{xAct}}`;
        }}
        stepInfo.textContent = txt;
    }}

    // Status
    setStatus('py-status', pyState.result, currentStep > pyData.states.length - 1);
    setStatus('c-status', cState.result, currentStep > cData.states.length - 1);

    // Diff detection
    detectDiff(pyState, cState);

    // Update action bars
    updateActionBars();

    // Button states
    document.getElementById('btn-prev').disabled = currentStep <= 0;
    document.getElementById('btn-next').disabled = currentStep >= maxStep;
    document.getElementById('btn-start').disabled = currentStep <= 0;
    document.getElementById('btn-end').disabled = currentStep >= maxStep;
}}

function setStatus(id, result, pastEnd) {{
    const el = document.getElementById(id);
    if (pastEnd) {{ el.textContent = '(no more moves)'; el.className = 'status unsolvable'; }}
    else if (result === 0) {{ el.textContent = 'OK'; el.className = 'status ok'; }}
    else if (result === 1) {{ el.textContent = 'DEAD'; el.className = 'status dead'; }}
    else if (result === 2) {{ el.textContent = 'WIN!'; el.className = 'status win'; }}
}}

function detectDiff(pyState, cState) {{
    const el = document.getElementById('diff');
    // Compare positions (accounting for coordinate transpose for flip=0)
    // Just show raw positions and let the human compare visually
    const diffs = [];
    // Check if player positions differ (accounting for the visual rendering)
    // Since each panel renders in its own coordinate system, visual comparison is key
    el.textContent = diffs.length ? diffs.join(' | ') : '';
    el.style.display = diffs.length ? 'block' : 'none';
}}

function goNext() {{ if (currentStep < maxStep) {{ currentStep++; draw(); }} }}
function goPrev() {{ if (currentStep > 0) {{ currentStep--; draw(); }} }}
function goStart() {{ currentStep = 0; draw(); }}
function goEnd() {{ currentStep = maxStep; draw(); }}

function togglePlay() {{
    playing = !playing;
    document.getElementById('btn-play').textContent = playing ? 'Pause' : 'Play';
    if (playing) {{
        playTimer = setInterval(() => {{
            if (currentStep >= maxStep) {{ togglePlay(); return; }}
            currentStep++;
            draw();
        }}, 400);
    }} else {{
        clearInterval(playTimer);
    }}
}}

document.addEventListener('keydown', (e) => {{
    if (e.key === 'ArrowRight' || e.key === ' ') {{ e.preventDefault(); goNext(); }}
    else if (e.key === 'ArrowLeft') {{ e.preventDefault(); goPrev(); }}
    else if (e.key === 'Home') {{ e.preventDefault(); goStart(); }}
    else if (e.key === 'End') {{ e.preventDefault(); goEnd(); }}
    else if (e.key === 'p') {{ togglePlay(); }}
}});

init();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Solver comparison visualizer")
    parser.add_argument("file", nargs="?", help="dat file stem (e.g. B-5)")
    parser.add_argument("sub", nargs="?", type=int, default=0, help="sublevel index")
    parser.add_argument(
        "--out", type=Path, default=None, help="output HTML path (default: auto)"
    )
    parser.add_argument(
        "--all-diffs", action="store_true", help="list all disagreements"
    )
    parser.add_argument(
        "--cross",
        action="store_true",
        help="add a third panel: C solver's moves replayed on Python engine",
    )
    args = parser.parse_args()

    if args.all_diffs:
        list_diffs()
        return

    if not args.file:
        parser.error("specify a file stem (e.g. B-5) or --all-diffs")

    file_stem = args.file
    sub_idx = args.sub

    print(f"Running Python solver on {file_stem} sub {sub_idx}...")
    py_data = run_python_solver(file_stem, sub_idx)
    if py_data is None:
        print("Python solver failed to parse level")
        return

    print(f"Running C solver on {file_stem} sub {sub_idx}...")
    c_data = run_c_solver(file_stem, sub_idx)
    if c_data is None:
        print("C solver failed")
        return

    print(
        f"Python: {'solved in ' + str(py_data['n_moves']) + ' moves' if py_data['solved'] else 'UNSOLVABLE'}"
    )
    print(
        f"C:      {'solved in ' + str(c_data['n_moves']) + ' moves' if c_data['solved'] else 'UNSOLVABLE'}"
    )

    cross_data = None
    if args.cross and c_data["solved"]:
        print("Replaying C moves on Python engine...")
        cross_data = replay_on_python(file_stem, sub_idx, c_data["actions"])
        if cross_data:
            last = cross_data["states"][-1]
            result = "WIN" if last["result"] == 2 else "DEAD" if last["result"] == 1 else "stuck"
            print(f"Cross:  {result} at step {len(cross_data['states']) - 1}")

    html_content = generate_html(file_stem, sub_idx, py_data, c_data, cross_data)

    out_path = args.out or Path(f"/tmp/maze_{file_stem}_{sub_idx}.html")
    out_path.write_text(html_content)
    print(f"Wrote {out_path}")
    print(f"Open: xdg-open {out_path}")


def list_diffs():
    """List all levels where C and Python disagree."""
    import csv

    c_csv = PROJECT_ROOT / "csolver" / "c_results.csv"
    py_csv = PROJECT_ROOT / "solver_results_v4.csv"

    if not c_csv.exists() or not py_csv.exists():
        print("Run both solvers first (--all) to generate CSV files")
        return

    c_results = {}
    with open(c_csv) as f:
        for row in csv.DictReader(f):
            key = (row["file"], int(row["sublevel"]))
            c_results[key] = int(row["moves"]) if row["moves"] else None

    py_results = {}
    with open(py_csv) as f:
        for row in csv.DictReader(f):
            key = (row["file"], int(row["sublevel"]))
            py_results[key] = int(row["moves"]) if row["moves"] else None

    print(f"{'Level':<16} {'Python':>8} {'C':>8} {'Diff':>8}")
    print("-" * 44)
    count = 0
    for key in sorted(set(c_results) & set(py_results)):
        cm, pm = c_results[key], py_results[key]
        if cm != pm:
            p_str = str(pm) if pm is not None else "UNSOL"
            c_str = str(cm) if cm is not None else "UNSOL"
            d_str = ""
            if pm is not None and cm is not None:
                d_str = f"{cm - pm:+d}"
            print(f"{key[0]:>5} sub {key[1]:<5} {p_str:>8} {c_str:>8} {d_str:>8}")
            count += 1
    print(f"\nTotal disagreements: {count}")


if __name__ == "__main__":
    main()
