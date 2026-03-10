"""Shared binary I/O helpers for policy_server and model_server.

Both servers use the same level observation wire format. This module
provides the common parsing functions.
"""

import struct
from typing import BinaryIO

import jax.numpy as jnp
import numpy as np

from src.env.types import EnvState, LevelData


def read_exact(stream: BinaryIO, n: int) -> bytes:
  """Read exactly n bytes from stream."""
  chunks: list[bytes] = []
  remaining = n
  while remaining > 0:
    chunk = stream.read(remaining)
    if not chunk:
      raise EOFError
    chunks.append(chunk)
    remaining -= len(chunk)
  return b"".join(chunks)


def read_u32(stream: BinaryIO) -> int:
  return struct.unpack("<I", read_exact(stream, 4))[0]


def read_i32(stream: BinaryIO) -> int:
  return struct.unpack("<i", read_exact(stream, 4))[0]


def read_level_data(stream: BinaryIO, gs: int) -> LevelData:
  """Read level observation data from binary stream."""
  n = gs
  n1 = n + 1

  h_bytes = read_exact(stream, n1 * n)
  h_walls = np.frombuffer(h_bytes, dtype=np.uint8).reshape(n1, n).astype(np.bool_)

  v_bytes = read_exact(stream, n * n1)
  v_walls = np.frombuffer(v_bytes, dtype=np.uint8).reshape(n, n1).astype(np.bool_)

  is_red = bool(read_exact(stream, 1)[0])
  has_key_gate = bool(read_exact(stream, 1)[0])
  gate_row = read_i32(stream)
  gate_col = read_i32(stream)

  td = struct.unpack("<4i", read_exact(stream, 16))
  trap_pos = np.array([[td[0], td[1]], [td[2], td[3]]], dtype=np.int32)
  ta_bytes = read_exact(stream, 2)
  trap_active = np.array([bool(ta_bytes[0]), bool(ta_bytes[1])], dtype=np.bool_)

  kd = struct.unpack("<2i", read_exact(stream, 8))
  key_pos = np.array(kd, dtype=np.int32)

  ed = struct.unpack("<2i", read_exact(stream, 8))
  exit_cell = np.array(ed, dtype=np.int32)

  return LevelData(
    h_walls_base=jnp.array(h_walls),
    v_walls_base=jnp.array(v_walls),
    is_red=jnp.bool_(is_red),
    has_key_gate=jnp.bool_(has_key_gate),
    gate_row=jnp.int32(gate_row),
    gate_col=jnp.int32(gate_col),
    trap_pos=jnp.array(trap_pos),
    trap_active=jnp.array(trap_active),
    key_pos=jnp.array(key_pos),
    exit_cell=jnp.array(exit_cell),
    initial_player=jnp.zeros(2, dtype=jnp.int32),
    initial_mummy_pos=jnp.zeros((2, 2), dtype=jnp.int32),
    initial_mummy_alive=jnp.zeros(2, dtype=jnp.bool_),
    initial_scorpion_pos=jnp.zeros((1, 2), dtype=jnp.int32),
    initial_scorpion_alive=jnp.zeros(1, dtype=jnp.bool_),
  )


def state_tuples_to_env_states(tuples: np.ndarray) -> EnvState:
  """Convert (N, 12) i32 state tuples to batched EnvState.

  Gate polarity inverted: Rust gate_open=True means blocking,
  JAX gate_open=True means open.
  Dead entities (99,99) clamped to (0,0).
  """
  t = jnp.array(tuples)
  pr, pc = t[:, 0], t[:, 1]
  m1r, m1c = t[:, 2], t[:, 3]
  m1_alive = t[:, 4].astype(jnp.bool_)
  m2r, m2c = t[:, 5], t[:, 6]
  m2_alive = t[:, 7].astype(jnp.bool_)
  sr, sc = t[:, 8], t[:, 9]
  s_alive = t[:, 10].astype(jnp.bool_)
  gate_open_rust = t[:, 11].astype(jnp.bool_)

  m1r = jnp.where(m1_alive, m1r, 0)
  m1c = jnp.where(m1_alive, m1c, 0)
  m2r = jnp.where(m2_alive, m2r, 0)
  m2c = jnp.where(m2_alive, m2c, 0)
  sr = jnp.where(s_alive, sr, 0)
  sc = jnp.where(s_alive, sc, 0)

  b = t.shape[0]
  return EnvState(
    player=jnp.stack([pr, pc], axis=-1),
    mummy_pos=jnp.stack(
      [
        jnp.stack([m1r, m1c], axis=-1),
        jnp.stack([m2r, m2c], axis=-1),
      ],
      axis=1,
    ),
    mummy_alive=jnp.stack([m1_alive, m2_alive], axis=-1),
    scorpion_pos=jnp.stack([sr, sc], axis=-1)[:, None, :],
    scorpion_alive=s_alive[:, None],
    gate_open=~gate_open_rust,
    done=jnp.zeros(b, dtype=jnp.bool_),
    won=jnp.zeros(b, dtype=jnp.bool_),
    turn=jnp.zeros(b, dtype=jnp.int32),
  )


def next_power_of_2(n: int) -> int:
  """Round up to next power of 2 (minimum 1)."""
  if n <= 1:
    return 1
  return 1 << (n - 1).bit_length()
