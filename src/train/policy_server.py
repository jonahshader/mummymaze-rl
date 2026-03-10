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
import sys
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
from scipy.special import softmax as scipy_softmax

from src.env.obs import observe
from src.env.types import EnvState, LevelData
from src.train.model import DEFAULT_ARCH, make_model
from src.train.wire import (
  next_power_of_2,
  read_exact,
  read_level_data,
  read_u32,
  state_tuples_to_env_states,
)


def serve(
  checkpoint_path: Path, max_batch_size: int = 0, arch: str = DEFAULT_ARCH
) -> None:
  """Main loop: read requests from stdin, write to stdout.

  Args:
    max_batch_size: If >0, cap the padded batch size to this power-of-2 value
      and process large levels in chunks. Prevents GPU OOM on huge state graphs.
    arch: Model architecture name from MODEL_REGISTRY.
  """
  model = make_model(arch, jax.random.key(0))
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
      msg_type = read_exact(stdin, 1)[0]
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

    grid_size = read_u32(stdin)
    n_levels = read_u32(stdin)

    all_level_data: list[LevelData] = []
    all_state_tuples: list[np.ndarray] = []
    state_counts: list[int] = []

    for _ in range(n_levels):
      ld = read_level_data(stdin, grid_size)
      all_level_data.append(ld)

      n_states = read_u32(stdin)
      state_counts.append(n_states)

      if n_states > 0:
        raw = read_exact(stdin, n_states * 12 * 4)
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

        padded_size = next_power_of_2(chunk_n)
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

        env_states = state_tuples_to_env_states(padded_st)
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
  parser.add_argument(
    "--arch",
    type=str,
    default=DEFAULT_ARCH,
    help=f"Model architecture (default: {DEFAULT_ARCH})",
  )
  args = parser.parse_args()

  if not args.checkpoint.exists():
    print(
      f"Checkpoint not found: {args.checkpoint}",
      file=sys.stderr,
    )
    sys.exit(1)

  serve(args.checkpoint, max_batch_size=args.max_batch_size, arch=args.arch)


if __name__ == "__main__":
  main()
