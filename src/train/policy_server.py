"""Policy server for GA adversarial training.

Reads state tuples from stdin, runs model inference, writes
action probabilities to stdout via binary protocol.

Binary protocol (little-endian):

Each message starts with a u8 message type:
  0 = Evaluate (followed by payload below, expects response)
  1 = Shutdown (server exits cleanly)

Evaluate payload:
  u32: grid_size
  u32: n_levels
  Per level:
    Level observation data:
      u8[(gs+1)*gs]: h_walls flattened row-major
      u8[gs*(gs+1)]: v_walls flattened row-major
      u8: is_red
      u8: has_key_gate
      i32: gate_row, gate_col
      i32[4]: trap_pos (r0,c0,r1,c1)
      u8[2]: trap_active
      i32[2]: key_pos (r,c)
      i32[2]: exit_cell (r,c)
    u32: n_states
    i32[n_states*12]: state_tuples

Response:
  Per level:
    f32[n_states*5]: action_probs (softmax of logits)

Shutdown: grid_size=0 signals exit.

Usage:
  uv run python -m src.train.policy_server \\
    --checkpoint path/to/model.eqx
"""

import functools
import struct
import sys
from typing import BinaryIO
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import softmax as scipy_softmax

from src.env.obs import observe
from src.env.types import EnvState, LevelData
from src.train.model import MazeCNN


def _read_exact(stream: BinaryIO, n: int) -> bytes:
  """Read exactly n bytes from stream."""
  data = b""
  while len(data) < n:
    chunk = stream.read(n - len(data))
    if not chunk:
      raise EOFError
    data += chunk
  return data


def _read_u32(stream: BinaryIO) -> int:
  return struct.unpack("<I", _read_exact(stream, 4))[0]


def _read_i32(stream: BinaryIO) -> int:
  return struct.unpack("<i", _read_exact(stream, 4))[0]


def _read_level_data(
  stream: BinaryIO,
  gs: int,
) -> LevelData:
  """Read level observation data from binary stream."""
  n = gs
  n1 = n + 1

  h_bytes = _read_exact(stream, n1 * n)
  h_walls = np.frombuffer(h_bytes, dtype=np.uint8)
  h_walls = h_walls.reshape(n1, n).astype(np.bool_)

  v_bytes = _read_exact(stream, n * n1)
  v_walls = np.frombuffer(v_bytes, dtype=np.uint8)
  v_walls = v_walls.reshape(n, n1).astype(np.bool_)

  is_red = bool(_read_exact(stream, 1)[0])
  has_key_gate = bool(_read_exact(stream, 1)[0])
  gate_row = _read_i32(stream)
  gate_col = _read_i32(stream)

  td = struct.unpack("<4i", _read_exact(stream, 16))
  trap_pos = np.array(
    [[td[0], td[1]], [td[2], td[3]]],
    dtype=np.int32,
  )
  ta_bytes = _read_exact(stream, 2)
  trap_active = np.array(
    [bool(ta_bytes[0]), bool(ta_bytes[1])],
    dtype=np.bool_,
  )

  kd = struct.unpack("<2i", _read_exact(stream, 8))
  key_pos = np.array(kd, dtype=np.int32)

  ed = struct.unpack("<2i", _read_exact(stream, 8))
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


def _state_tuples_to_env_states(
  tuples: np.ndarray,
) -> EnvState:
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


def _next_power_of_2(n: int) -> int:
  """Round up to next power of 2 (minimum 1)."""
  if n <= 1:
    return 1
  return 1 << (n - 1).bit_length()


def serve(checkpoint_path: Path, max_batch_size: int = 0) -> None:
  """Main loop: read requests from stdin, write to stdout.

  Args:
    max_batch_size: If >0, cap the padded batch size to this power-of-2 value
      and process large levels in chunks. Prevents GPU OOM on huge state graphs.
  """
  model = MazeCNN(jax.random.key(0))
  model = eqx.tree_deserialise_leaves(checkpoint_path, model)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _obs_and_forward(
    grid_size: int,
    level_data: LevelData,
    env_states: EnvState,
  ) -> jax.Array:
    """JIT'd obs building + forward pass for one level (padded batch)."""
    obs = jax.vmap(lambda es: observe(grid_size, level_data, es))(env_states)
    return jax.vmap(model)(obs)

  jitted_sizes: set[int] = set()

  stderr = sys.stderr
  stdin = sys.stdin.buffer
  stdout = sys.stdout.buffer

  MSG_EVALUATE = 0
  MSG_SHUTDOWN = 1

  print("policy_server: ready", file=stderr, flush=True)

  while True:
    try:
      msg_type = _read_exact(stdin, 1)[0]
    except EOFError:
      break

    if msg_type == MSG_SHUTDOWN:
      break

    if msg_type != MSG_EVALUATE:
      print(
        f"policy_server: unknown message type {msg_type}",
        file=stderr,
        flush=True,
      )
      break

    grid_size = _read_u32(stdin)
    n_levels = _read_u32(stdin)

    all_level_data: list[LevelData] = []
    all_state_tuples: list[np.ndarray] = []
    state_counts: list[int] = []

    for _ in range(n_levels):
      ld = _read_level_data(stdin, grid_size)
      all_level_data.append(ld)

      n_states = _read_u32(stdin)
      state_counts.append(n_states)

      if n_states > 0:
        raw = _read_exact(stdin, n_states * 12 * 4)
        tuples = np.frombuffer(raw, dtype=np.int32)
        tuples = tuples.reshape(n_states, 12).copy()
      else:
        tuples = np.zeros((0, 12), dtype=np.int32)
      all_state_tuples.append(tuples)

    # Per-level JIT'd obs+forward with power-of-2 padding.
    # Only ~11 unique padded sizes, so JIT traces are bounded.
    total_states = sum(state_counts)
    print(
      f"policy_server: processing {n_levels} levels ({total_states} total states)",
      file=stderr,
      flush=True,
    )
    states_done = 0
    for ld, st, n_st in zip(
      all_level_data,
      all_state_tuples,
      state_counts,
    ):
      if n_st == 0:
        continue

      # Chunk large levels to stay within GPU memory.
      chunk_size = max_batch_size if max_batch_size > 0 else n_st
      all_probs = []
      for chunk_start in range(0, n_st, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_st)
        chunk_n = chunk_end - chunk_start
        chunk_st = st[chunk_start:chunk_end]

        padded_size = _next_power_of_2(chunk_n)
        if padded_size not in jitted_sizes:
          print(
            f"policy_server: JIT compiling for batch_size={padded_size}",
            file=stderr,
            flush=True,
          )
          jitted_sizes.add(padded_size)

        # Pad state tuples to power-of-2 size
        if padded_size > chunk_n:
          padding = np.zeros((padded_size - chunk_n, 12), dtype=np.int32)
          padded_st = np.concatenate([chunk_st, padding], axis=0)
        else:
          padded_st = chunk_st

        env_states = _state_tuples_to_env_states(padded_st)
        logits = _obs_and_forward(grid_size, ld, env_states)
        all_probs.append(np.array(logits[:chunk_n]))

      probs = scipy_softmax(
        np.concatenate(all_probs, axis=0) if len(all_probs) > 1 else all_probs[0],
        axis=-1,
      ).astype(np.float32)
      stdout.write(probs.tobytes())

      states_done += n_st
      if states_done % 50000 < n_st:
        print(
          f"policy_server: {states_done}/{total_states} states",
          file=stderr,
          flush=True,
        )

    stdout.flush()
    print("policy_server: response sent", file=stderr, flush=True)


def main() -> None:
  import argparse

  parser = argparse.ArgumentParser(
    description="Policy inference server",
  )
  parser.add_argument(
    "--checkpoint",
    type=Path,
    required=True,
    help="Path to .eqx checkpoint",
  )
  parser.add_argument(
    "--max-batch-size",
    type=int,
    default=0,
    help="Cap padded batch size (power of 2). Levels with more states are "
    "processed in chunks. 0 = no cap.",
  )
  args = parser.parse_args()

  if not args.checkpoint.exists():
    print(
      f"Checkpoint not found: {args.checkpoint}",
      file=sys.stderr,
    )
    sys.exit(1)

  serve(args.checkpoint, max_batch_size=args.max_batch_size)


if __name__ == "__main__":
  main()
