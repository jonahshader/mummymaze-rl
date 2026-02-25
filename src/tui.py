"""Curses-based TUI for playing Mummy Maze levels.

Usage:
  uv run python src/tui.py <dat_dir> [--file B-0] [--sublevel 0]

Controls:
  Arrow keys / WASD  Move player
  Space              Wait
  R                  Restart level
  N / P              Next / Previous sublevel
  Q / Esc            Quit
"""

import argparse
import curses
from pathlib import Path

from mummy_maze.parser import parse_file

from src.game import (
  ACTION_EAST,
  ACTION_NORTH,
  ACTION_SOUTH,
  ACTION_WAIT,
  ACTION_WEST,
  GameState,
  load_level,
  restart,
  step,
)

# Curses color pair IDs
_COL_WALL = 1
_COL_PLAYER = 2
_COL_MUMMY = 3
_COL_SCORPION = 4
_COL_TRAP = 5
_COL_KEY = 6
_COL_GATE_CLOSED = 7
_COL_GATE_OPEN = 8
_COL_EXIT = 9
_COL_STATUS = 10


def _init_colors() -> None:
  curses.start_color()
  curses.use_default_colors()
  curses.init_pair(_COL_WALL, curses.COLOR_WHITE, -1)
  curses.init_pair(_COL_PLAYER, curses.COLOR_GREEN, -1)
  curses.init_pair(_COL_MUMMY, curses.COLOR_RED, -1)
  curses.init_pair(_COL_SCORPION, curses.COLOR_YELLOW, -1)
  curses.init_pair(_COL_TRAP, curses.COLOR_MAGENTA, -1)
  curses.init_pair(_COL_KEY, curses.COLOR_CYAN, -1)
  curses.init_pair(_COL_GATE_CLOSED, curses.COLOR_RED, -1)
  curses.init_pair(_COL_GATE_OPEN, curses.COLOR_GREEN, -1)
  curses.init_pair(_COL_EXIT, curses.COLOR_GREEN, -1)
  curses.init_pair(_COL_STATUS, curses.COLOR_WHITE, -1)


def _render(stdscr: curses.window, state: GameState) -> None:
  """Render the maze to the curses window."""
  stdscr.erase()
  n = state.grid_size
  h = 2 * n + 1
  w = 2 * n + 1

  # Build character grid
  grid: list[list[str]] = [[" "] * w for _ in range(h)]
  colors: list[list[int]] = [[0] * w for _ in range(h)]

  # Corners
  for r in range(n + 1):
    for c in range(n + 1):
      grid[r * 2][c * 2] = "+"
      colors[r * 2][c * 2] = _COL_WALL

  # Horizontal walls
  for r in range(n + 1):
    for c in range(n):
      if state.h_walls[r][c]:
        grid[r * 2][c * 2 + 1] = "-"
        colors[r * 2][c * 2 + 1] = _COL_WALL

  # Vertical walls
  for r in range(n):
    for c in range(n + 1):
      if state.v_walls[r][c]:
        grid[r * 2 + 1][c * 2] = "|"
        colors[r * 2 + 1][c * 2] = _COL_WALL

  # Exit marker
  if state.exit_side == "N":
    grid[0][state.exit_pos * 2 + 1] = " "
  elif state.exit_side == "S":
    grid[n * 2][state.exit_pos * 2 + 1] = " "
  elif state.exit_side == "W":
    grid[state.exit_pos * 2 + 1][0] = " "
  elif state.exit_side == "E":
    grid[state.exit_pos * 2 + 1][n * 2] = " "

  # Mark exit cell
  er, ec = state.exit_cell
  if grid[er * 2 + 1][ec * 2 + 1] == " ":
    grid[er * 2 + 1][ec * 2 + 1] = "X"
    colors[er * 2 + 1][ec * 2 + 1] = _COL_EXIT

  # Traps
  for tr, tc in state.traps:
    grid[tr * 2 + 1][tc * 2 + 1] = "T"
    colors[tr * 2 + 1][tc * 2 + 1] = _COL_TRAP

  # Key
  if state.key_pos is not None:
    kr, kc = state.key_pos
    grid[kr * 2 + 1][kc * 2 + 1] = "K"
    colors[kr * 2 + 1][kc * 2 + 1] = _COL_KEY

  # Gate (rendered on south wall edge of the gate's cell)
  if state.gate_wall is not None:
    gr, gc = state.gate_wall  # h_walls index: (row+1, col)
    wall_present = state.h_walls[gr][gc]
    if wall_present:
      grid[gr * 2][gc * 2 + 1] = "G"
      colors[gr * 2][gc * 2 + 1] = _COL_GATE_CLOSED
    else:
      grid[gr * 2][gc * 2 + 1] = "g"
      colors[gr * 2][gc * 2 + 1] = _COL_GATE_OPEN

  # Scorpions
  for sr, sc in state.scorpions:
    grid[sr * 2 + 1][sc * 2 + 1] = "S"
    colors[sr * 2 + 1][sc * 2 + 1] = _COL_SCORPION

  # Mummies
  for mr, mc in state.mummies:
    grid[mr * 2 + 1][mc * 2 + 1] = "M"
    colors[mr * 2 + 1][mc * 2 + 1] = _COL_MUMMY

  # Player (drawn last so it's on top)
  pr, pc = state.player
  if 0 <= pr < n and 0 <= pc < n:
    grid[pr * 2 + 1][pc * 2 + 1] = "P"
    colors[pr * 2 + 1][pc * 2 + 1] = _COL_PLAYER

  # Draw to screen
  max_y, max_x = stdscr.getmaxyx()
  for r in range(min(h, max_y - 3)):
    for c in range(min(w, max_x - 1)):
      try:
        stdscr.addch(r + 1, c + 1, grid[r][c], curses.color_pair(colors[r][c]))
      except curses.error:
        pass

  # Legend (right of maze)
  legend_x = w + 3
  legend: list[tuple[str, int]] = [
    ("P Player", _COL_PLAYER),
    ("M Mummy", _COL_MUMMY),
    ("S Scorpion", _COL_SCORPION),
    ("T Trap", _COL_TRAP),
    ("X Exit", _COL_EXIT),
    ("K Key", _COL_KEY),
    ("G Gate (closed)", _COL_GATE_CLOSED),
    ("g Gate (open)", _COL_GATE_OPEN),
  ]
  for i, (text, col) in enumerate(legend):
    y = i + 1
    if y < max_y - 2 and legend_x + len(text) < max_x:
      try:
        stdscr.addch(y, legend_x, text[0], curses.color_pair(col))
        stdscr.addstr(y, legend_x + 1, text[1:], curses.color_pair(_COL_STATUS))
      except curses.error:
        pass

  # Status line
  status_y = min(h + 2, max_y - 1)
  if state.won:
    status = "  YOU WIN!  Press R to restart, N/P for next/prev, Q to quit"
  elif not state.alive:
    status = "  YOU DIED!  Press R to restart, N/P for next/prev, Q to quit"
  else:
    mummy_type = "red" if state.is_red else "white"
    status = (
      f"  Turn: {state.turn}  |  Mummies: {mummy_type}"
      "  |  Arrows/WASD: move  Space: wait  R: restart  Q: quit"
    )

  try:
    stdscr.addnstr(status_y, 0, status, max_x - 1, curses.color_pair(_COL_STATUS))
  except curses.error:
    pass

  stdscr.refresh()


def _main_loop(
  stdscr: curses.window,
  dat_files: list[Path],
  file_idx: int,
  sub_idx: int,
) -> None:
  _init_colors()
  curses.curs_set(0)
  stdscr.keypad(True)

  def load_current() -> GameState | None:
    parsed = parse_file(dat_files[file_idx])
    if parsed is None or sub_idx >= len(parsed.sublevels):
      return None
    return load_level(parsed.sublevels[sub_idx], parsed.header)

  state = load_current()
  if state is None:
    return

  _render(stdscr, state)

  while True:
    key = stdscr.getch()

    action: int | None = None

    # Arrow keys
    if key == curses.KEY_UP or key == ord("w"):
      action = ACTION_NORTH
    elif key == curses.KEY_DOWN or key == ord("s"):
      action = ACTION_SOUTH
    elif key == curses.KEY_RIGHT or key == ord("d"):
      action = ACTION_EAST
    elif key == curses.KEY_LEFT or key == ord("a"):
      action = ACTION_WEST
    elif key == ord(" "):
      action = ACTION_WAIT
    elif key == ord("r"):
      state = restart(state)
    elif key == ord("n"):
      sub_idx = min(sub_idx + 1, 99)
      new_state = load_current()
      if new_state is not None:
        state = new_state
    elif key == ord("p"):
      sub_idx = max(sub_idx - 1, 0)
      new_state = load_current()
      if new_state is not None:
        state = new_state
    elif key == ord("q") or key == 27:  # q or Esc
      break

    if action is not None:
      step(state, action)

    _render(stdscr, state)


def main() -> None:
  parser = argparse.ArgumentParser(description="Play Mummy Maze in the terminal")
  parser.add_argument("dat_dir", type=Path, help="directory containing B-*.dat files")
  parser.add_argument("--file", default="B-0", help="dat file stem (default: B-0)")
  parser.add_argument(
    "--sublevel",
    type=int,
    default=0,
    help="sublevel index (default: 0)",
  )
  args = parser.parse_args()

  dat_dir = args.dat_dir.resolve()
  dat_files = sorted(
    dat_dir.glob("B-*.dat"),
    key=lambda p: int(p.stem.split("-")[1]),
  )
  if not dat_files:
    print(f"No B-*.dat files found in {dat_dir}")  # noqa: T201
    return

  # Find the requested file
  file_idx = 0
  for i, f in enumerate(dat_files):
    if f.stem == args.file:
      file_idx = i
      break

  curses.wrapper(lambda stdscr: _main_loop(stdscr, dat_files, file_idx, args.sublevel))


if __name__ == "__main__":
  main()
