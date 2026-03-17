"""WebSocket server for model inference, training, and adversarial loops.

Replaces the binary frame protocol with a JSON-over-WebSocket API.
Any client (Rust viewer, web frontend, CLI) can connect.

Usage:
  uv run python -m src.train.ws_server --mazes mazes/ [--port 8765]
"""

import argparse
import asyncio
import json
import logging
import queue
import threading
from pathlib import Path

import mummymaze_rust
import websockets
from websockets.asyncio.server import ServerConnection

from src.train.model import DEFAULT_ARCH, parse_hparams
from src.train.model_server import ModelServer
from src.train.reporter import WebSocketReporter

log = logging.getLogger(__name__)


async def _drain_queue(
  ws: ServerConnection,
  q: queue.Queue[dict],
  done_event: asyncio.Event,
) -> None:
  """Drain a sync queue to the WebSocket until done_event is set.

  Bridges the sync training/GA thread to the async WebSocket loop.
  """
  loop = asyncio.get_running_loop()
  while not done_event.is_set():
    try:
      msg = await loop.run_in_executor(None, q.get, True, 0.5)
      await ws.send(json.dumps(msg))
    except queue.Empty:
      continue
  # Drain any remaining messages
  while True:
    try:
      msg = q.get_nowait()
      await ws.send(json.dumps(msg))
    except queue.Empty:
      break


class WsHandler:
  """Handles a single WebSocket connection dispatching to ModelServer."""

  def __init__(self, server: ModelServer) -> None:
    self.server = server
    self._train_stop = threading.Event()
    self._adversarial_stop = threading.Event()
    self._ga_stop = threading.Event()
    # Background task for long-running operations (train/adversarial/GA).
    # Only one can run at a time; the message loop stays responsive for stop commands.
    self._bg_task: asyncio.Task | None = None  # type: ignore[type-arg]

  async def handle(self, ws: ServerConnection) -> None:
    log.info("client connected: %s", ws.remote_address)
    try:
      async for raw in ws:
        try:
          msg = json.loads(raw)
        except json.JSONDecodeError:
          await self._send_error(ws, None, "Invalid JSON")
          continue

        msg_type = msg.get("type")
        request_id = msg.get("request_id")
        try:
          await self._dispatch(ws, msg_type, msg, request_id)
        except Exception as e:
          log.exception("Error handling %s", msg_type)
          await self._send_error(ws, request_id, str(e))
    except websockets.ConnectionClosed:
      log.info("client disconnected")
    finally:
      if self._bg_task and not self._bg_task.done():
        self._bg_task.cancel()

  async def _dispatch(
    self,
    ws: ServerConnection,
    msg_type: str | None,
    msg: dict,
    request_id: str | None,
  ) -> None:
    if msg_type == "evaluate":
      await self._handle_evaluate(ws, msg, request_id)
    elif msg_type == "train":
      self._launch_bg(self._handle_train(ws, msg, request_id))
    elif msg_type == "stop_train":
      self._train_stop.set()
    elif msg_type == "adversarial":
      self._launch_bg(self._handle_adversarial(ws, msg, request_id))
    elif msg_type == "stop_adversarial":
      self._adversarial_stop.set()
    elif msg_type == "ga":
      self._launch_bg(self._handle_ga(ws, msg, request_id))
    elif msg_type == "stop_ga":
      self._ga_stop.set()
    elif msg_type == "reload_checkpoint":
      await self._handle_reload(ws, msg, request_id)
    elif msg_type == "list_checkpoints":
      await self._handle_list_checkpoints(ws, request_id)
    elif msg_type == "shutdown":
      log.info("shutdown requested")
      raise SystemExit
    else:
      await self._send_error(ws, request_id, f"Unknown message type: {msg_type}")

  def _launch_bg(self, coro: object) -> None:
    """Launch a long-running handler as a background task.

    The message loop stays responsive so stop commands can be processed.
    Only one background task runs at a time.
    """
    if self._bg_task and not self._bg_task.done():
      log.warning("background task already running, ignoring new request")
      return
    self._bg_task = asyncio.create_task(coro)  # type: ignore[arg-type]

  async def _handle_evaluate(
    self,
    ws: ServerConnection,
    msg: dict,
    request_id: str | None,
  ) -> None:
    level_key = msg.get("level_key", "")
    states = await asyncio.to_thread(self.server.evaluate_level, level_key)
    await ws.send(
      json.dumps(
        {
          "type": "evaluate_result",
          "request_id": request_id,
          "states": states,
        }
      )
    )

  async def _handle_train(
    self,
    ws: ServerConnection,
    msg: dict,
    request_id: str | None,
  ) -> None:
    config = msg.get("config", {})
    self._train_stop.clear()

    event_queue: queue.Queue[dict] = queue.Queue()
    reporter = WebSocketReporter(
      event_queue, self._train_stop, event_type="training_event"
    )
    done = asyncio.Event()

    async def run() -> None:
      drain_task = asyncio.create_task(_drain_queue(ws, event_queue, done))
      try:
        await asyncio.to_thread(self.server.train, reporter, **config)
      finally:
        done.set()
        await drain_task

    await run()

  async def _handle_adversarial(
    self,
    ws: ServerConnection,
    msg: dict,
    request_id: str | None,
  ) -> None:
    from src.train.adversarial_loop import adversarial_loop

    config = msg.get("config", {})
    self._adversarial_stop.clear()

    # Training events go through a WebSocketReporter
    train_queue: queue.Queue[dict] = queue.Queue()
    train_reporter = WebSocketReporter(
      train_queue, self._adversarial_stop, event_type="training_event"
    )

    # Adversarial events go through the on_event callback
    adv_queue: queue.Queue[dict] = queue.Queue()

    def on_event(event: dict) -> None:
      adv_queue.put({"type": "adversarial_event", "event": event})

    done = asyncio.Event()

    async def run() -> None:
      # Drain both queues concurrently
      drain_train = asyncio.create_task(_drain_queue(ws, train_queue, done))
      drain_adv = asyncio.create_task(_drain_queue(ws, adv_queue, done))
      try:
        await asyncio.to_thread(
          adversarial_loop,
          self.server.maze_dir,
          reporter=train_reporter,
          on_event=on_event,
          **config,
        )
      finally:
        done.set()
        await asyncio.gather(drain_train, drain_adv)

    await run()

  async def _handle_ga(
    self,
    ws: ServerConnection,
    msg: dict,
    request_id: str | None,
  ) -> None:
    from src.train.ga import GenerationResult, run_ga

    config = msg.get("config", {})
    self._ga_stop.clear()

    # Extract server-side keys before forwarding remainder to run_ga()
    seed_keys: list[str] = config.pop("seed_keys", [])
    config.pop("grid_size", None)  # consumed by viewer, not needed server-side

    ga_queue: queue.Queue[dict] = queue.Queue()

    def load_and_run() -> None:
      try:
        # Load seed levels from keys
        by_stem: dict[str, list[int]] = {}
        for key in seed_keys:
          stem, sub_s = key.split(":")
          by_stem.setdefault(stem, []).append(int(sub_s))

        seed_levels = []
        for stem, subs in by_stem.items():
          try:
            all_in_file = mummymaze_rust.parse_file(
              str(self.server.maze_dir / f"{stem}.dat"),
            )
            for sub in subs:
              if sub < len(all_in_file):
                seed_levels.append(all_in_file[sub])
          except Exception:
            pass

        if not seed_levels:
          ga_queue.put(
            {
              "type": "ga_event",
              "event": {"type": "error", "message": "No valid seed levels found"},
            }
          )
          return

        prev_best_fitness = float("-inf")

        def on_generation(gen_result: GenerationResult) -> None:
          nonlocal prev_best_fitness
          best = gen_result.best
          # Only serialize level when fitness improves (avoids per-gen overhead)
          best_dict: dict = {
            "bfs_moves": best.bfs_moves,
            "n_states": best.n_states,
            "win_prob": best.win_prob,
            "fitness": best.fitness,
          }
          if best.fitness > prev_best_fitness:
            best_dict["level"] = best.level.to_json()
            prev_best_fitness = best.fitness
          ga_queue.put(
            {
              "type": "ga_event",
              "event": {
                "type": "generation",
                "generation": gen_result.generation,
                "best_fitness": best.fitness,
                "avg_fitness": gen_result.avg_fitness,
                "solvable_rate": gen_result.solvable_rate,
                "pop_size": gen_result.pop_size,
                "best": best_dict,
              },
            }
          )

        def on_status(message: str) -> None:
          ga_queue.put(
            {
              "type": "ga_event",
              "event": {"type": "status", "message": message},
            }
          )

        run_ga(
          seed_levels,
          obs_and_forward=self.server.obs_and_forward,
          on_generation=on_generation,
          on_status=on_status,
          stop_flag=self._ga_stop,
          **config,
        )

        ga_queue.put(
          {
            "type": "ga_event",
            "event": {"type": "done"},
          }
        )
      except Exception as e:
        ga_queue.put(
          {
            "type": "ga_event",
            "event": {"type": "error", "message": str(e)},
          }
        )

    done = asyncio.Event()

    async def run() -> None:
      drain_task = asyncio.create_task(_drain_queue(ws, ga_queue, done))
      try:
        await asyncio.to_thread(load_and_run)
      finally:
        done.set()
        await drain_task

    await run()

  async def _handle_reload(
    self,
    ws: ServerConnection,
    msg: dict,
    request_id: str | None,
  ) -> None:
    path = msg.get("path", "")
    await asyncio.to_thread(self.server.reload_checkpoint, path)
    await ws.send(
      json.dumps(
        {
          "type": "reload_result",
          "request_id": request_id,
          "status": "ok",
        }
      )
    )

  async def _handle_list_checkpoints(
    self,
    ws: ServerConnection,
    request_id: str | None,
  ) -> None:
    checkpoints = await asyncio.to_thread(self.server.list_checkpoints)
    await ws.send(
      json.dumps(
        {
          "type": "checkpoints_list",
          "request_id": request_id,
          "checkpoints": checkpoints,
        }
      )
    )

  @staticmethod
  async def _send_error(
    ws: ServerConnection,
    request_id: str | None,
    message: str,
  ) -> None:
    await ws.send(
      json.dumps(
        {
          "type": "error",
          "request_id": request_id,
          "message": message,
        }
      )
    )


async def serve(
  maze_dir: Path,
  *,
  port: int = 8765,
  host: str = "localhost",
  checkpoint: Path | None = None,
  arch: str = DEFAULT_ARCH,
  hparams: dict[str, object] | None = None,
) -> None:
  """Start the WebSocket server."""
  server = ModelServer(maze_dir, checkpoint=checkpoint, arch=arch, hparams=hparams)
  handler = WsHandler(server)

  log.info("starting WebSocket server on %s:%d", host, port)

  # Pre-load datasets so first request is fast
  server.load_datasets()

  async with websockets.serve(handler.handle, host, port):
    log.info("ready — ws://%s:%d", host, port)
    await asyncio.Future()  # run forever


def main() -> None:
  parser = argparse.ArgumentParser(
    description="WebSocket server for model inference and training"
  )
  parser.add_argument(
    "--mazes",
    type=Path,
    default=Path("mazes"),
    help="Directory containing B-*.dat files",
  )
  parser.add_argument("--port", type=int, default=8765)
  parser.add_argument("--host", type=str, default="localhost")
  parser.add_argument("--checkpoint", type=Path, default=None)
  parser.add_argument("--arch", type=str, default=DEFAULT_ARCH)
  parser.add_argument(
    "--hparam",
    action="append",
    default=[],
    help="Model hparam as key=value (repeatable)",
  )
  parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="Enable debug logging",
  )
  args = parser.parse_args()

  logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
  )

  hparams = parse_hparams(args.arch, args.hparam)

  asyncio.run(
    serve(
      args.mazes,
      port=args.port,
      host=args.host,
      checkpoint=args.checkpoint,
      arch=args.arch,
      hparams=hparams,
    )
  )


if __name__ == "__main__":
  main()
