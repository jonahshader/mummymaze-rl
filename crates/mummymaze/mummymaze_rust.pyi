"""Type stubs for the PyO3 module built from `src/python.rs`.

Hand-written: update it in the same commit as any change to a signature or
a returned dict in `python.rs`. `tests/test_rust_stubs.py` fails when the
names here and in the built module differ. maturin installs this file as
the package's `__init__.pyi` and adds `py.typed`.
"""

from collections.abc import Mapping, Sequence
from typing import Literal, NotRequired, TypedDict

import numpy as np
import numpy.typing as npt

type Cell = tuple[int, int]
type ExitSide = Literal["N", "S", "E", "W"]
type CrossoverMode = Literal["swap_entities", "region", "wall_patch", "feature_level"]

# (player_row, player_col, mummy1_row, mummy1_col, mummy1_alive,
#  mummy2_row, mummy2_col, mummy2_alive,
#  scorpion_row, scorpion_col, scorpion_alive, gate_open)
type StateTuple = tuple[int, int, int, int, bool, int, int, bool, int, int, bool, bool]

class LevelDict(TypedDict):
  grid_size: int
  flip: bool
  h_walls: list[bool]
  v_walls: list[bool]
  exit_side: ExitSide
  exit_pos: int
  player: Cell
  mummy1: Cell
  mummy2: Cell | None
  scorpion: Cell | None
  traps: list[Cell]
  gate: Cell | None
  key: Cell | None

class Analysis(TypedDict):
  file: str
  sublevel: int
  grid_size: int
  n_states: int
  win_prob: float
  expected_steps: float
  bfs_moves: int | None
  dead_end_ratio: float
  avg_branching_factor: float
  n_optimal_solutions: int
  greedy_deviation_count: int
  path_safety: float

class Graph(TypedDict):
  states: list[StateTuple]
  # (src_idx, action, dst): action is "N", "S", "E", "W" or "wait";
  # dst is a state index, "WIN" or "DEAD".
  edges: list[tuple[int, str, int | str]]
  start_idx: int

class SolvedActions(TypedDict):
  file: str
  sublevel: int
  actions: list[int] | None

class BestActions(TypedDict):
  """`action_masks` is bytes: one bitmask over the 5 actions per state."""

  file: str
  sublevel: int
  grid_size: int
  states: list[StateTuple]
  action_masks: bytes

class LevelBestActions(TypedDict):
  level_idx: int
  grid_size: int
  states: list[StateTuple]
  action_masks: bytes

class GaEval(TypedDict):
  level_idx: int
  level: Level
  bfs_moves: int
  n_states: int
  win_prob: float
  log_win_prob: float
  fitness: float
  # Not set by Rust; the Python GA adds them after policy evaluation.
  policy_win_prob: NotRequired[float]
  log_policy_win_prob: NotRequired[float]

class ReplayFrame(TypedDict):
  player: Cell
  mummy1: tuple[int, int, bool]
  mummy2: tuple[int, int, bool]
  scorpion: tuple[int, int, bool]
  gate_open: bool
  result: Literal["ok", "win", "dead"]

class Level:
  @staticmethod
  def from_file(dat_path: str, sublevel: int) -> Level: ...
  @staticmethod
  def from_edges(
    grid_size: int,
    flip: bool,
    h_walls: Sequence[bool],
    v_walls: Sequence[bool],
    exit_side: str,
    exit_pos: int,
    player: Cell,
    mummy1: Cell,
    mummy2: Cell | None = None,
    scorpion: Cell | None = None,
    traps: Sequence[Cell] = ...,
    gate: Cell | None = None,
    key: Cell | None = None,
  ) -> Level: ...
  @staticmethod
  def from_dict(d: LevelDict) -> Level: ...
  @property
  def grid_size(self) -> int: ...
  @property
  def flip(self) -> bool: ...
  @property
  def has_gate(self) -> bool: ...
  def fingerprint(self) -> int: ...
  def canonical_fingerprint(self) -> int: ...
  def apply_dihedral(self, sym: int) -> Level: ...
  def to_dict(self) -> LevelDict: ...
  def to_json(self) -> str: ...

def parse_file(dat_path: str) -> list[Level]: ...
def solve(level: Level) -> int | None: ...
def solve_actions(level: Level) -> list[int] | None: ...
def analyze(level: Level, label: str = "", sublevel: int = 0) -> Analysis: ...
def build_graph(level: Level) -> Graph: ...
def policy_win_prob(
  level: Level,
  state_tuples: npt.NDArray[np.int32],
  action_probs: npt.NDArray[np.float32],
) -> float: ...
def policy_win_prob_batch(
  levels: Sequence[Level],
  state_tuples: npt.NDArray[np.int32],
  action_probs: npt.NDArray[np.float32],
  offsets: Sequence[int],
) -> list[float]: ...
def solve_v_and_adjoint_batch(
  levels: Sequence[Level],
  state_tuples: npt.NDArray[np.int32],
  action_probs: npt.NDArray[np.float32],
  offsets: Sequence[int],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: ...
def analyze_all(maze_dir: str, jobs: int = 0) -> list[Analysis]: ...
def solve_all_actions(maze_dir: str, jobs: int = 0) -> list[SolvedActions]: ...
def best_actions_all(maze_dir: str, jobs: int = 0) -> list[BestActions]: ...
def best_actions_for_levels(levels: Sequence[Level]) -> list[LevelBestActions]: ...
def eval_fitness(expr: str, metrics: Mapping[str, float]) -> float: ...
def fitness_variables() -> list[tuple[str, str]]: ...
def fitness_presets() -> list[tuple[str, str]]: ...
def mutate(
  level: Level,
  seed: int,
  w_wall: float = 5.0,
  w_move_entity: float = 3.0,
  w_move_player: float = 2.0,
  w_add_entity: float = 1.0,
  w_remove_entity: float = 1.0,
  w_move_exit: float = 1.0,
  w_move_gate: float = 1.0,
  extra_wall_prob: float = 0.3,
) -> Level: ...
def mutate_batch(
  levels: Sequence[Level],
  base_seed: int,
  w_wall: float = 5.0,
  w_move_entity: float = 3.0,
  w_move_player: float = 2.0,
  w_add_entity: float = 1.0,
  w_remove_entity: float = 1.0,
  w_move_exit: float = 1.0,
  w_move_gate: float = 1.0,
  extra_wall_prob: float = 0.3,
) -> list[Level]: ...
def ga_crossover(
  a: Level, b: Level, mode: str = "swap_entities", seed: int = 0
) -> Level: ...
def ga_evaluate_batch(
  levels: Sequence[Level], fitness_expr: str | None = None
) -> list[GaEval]: ...
def replay_actions(level: Level, actions: Sequence[int]) -> list[ReplayFrame]: ...
def generate_random_solvable(
  count: int,
  grid_size: int = 6,
  seed: int = 0,
  wall_density: float = 0.3,
  mummy2_prob: float = 0.3,
  scorpion_prob: float = 0.2,
  trap1_prob: float = 0.15,
  trap2_prob: float = 0.3,
  gate_prob: float = 0.15,
  flip_prob: float = 0.5,
) -> list[Level]: ...
