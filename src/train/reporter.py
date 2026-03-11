"""Metrics reporter protocol and implementations for training."""

import json
import select
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


def build_levels_dict(
  metrics: dict[int, dict[int, dict[str, object]]],
  sources: dict[int, list[tuple[str, int]]],
) -> dict[str, object]:
  """Build keyed levels dict from per-grid-size metrics and sources."""
  levels: dict[str, object] = {}
  for gs, gs_metrics in metrics.items():
    src_list = sources.get(gs, [])
    for bank_idx, stats in gs_metrics.items():
      if bank_idx < len(src_list):
        file_stem, sublevel = src_list[bank_idx]
        key = f"{file_stem}:{sublevel}"
      else:
        key = f"gs{gs}:idx{bank_idx}"
      levels[key] = {"grid_size": gs, **stats}
  return levels


def write_level_metrics(
  all_metrics: dict[int, dict[int, dict[str, object]]],
  sources: dict[int, list[tuple[str, int]]],
  step: int,
  run_id: str,
  metrics_path: Path,
) -> None:
  """Write level_metrics.json with per-level stats."""
  output = {
    "run_id": run_id,
    "step": step,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "levels": build_levels_dict(all_metrics, sources),
  }
  metrics_path.parent.mkdir(parents=True, exist_ok=True)
  metrics_path.write_text(json.dumps(output, indent=2))


class MetricsReporter(Protocol):
  """Protocol for reporting training progress and metrics."""

  def report_init(self, config: dict) -> None: ...
  def report_epoch_start(
    self, epoch: int, total_epochs: int, steps_in_epoch: int
  ) -> None: ...
  def report_batch(
    self, step: int, epoch_step: int, loss: float, acc: float, gs: int
  ) -> None: ...
  def report_epoch_end(
    self,
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    epoch_time: float,
  ) -> None: ...
  def report_level_metrics(
    self,
    step: int,
    run_id: str,
    metrics: dict,
    sources: dict,
  ) -> None: ...
  def report_status(self, status: str) -> None: ...
  def report_log(self, message: str) -> None: ...
  def report_done(self) -> None: ...
  def check_command(self) -> str | None: ...


class FileReporter:
  """File-based reporter. Writes level_metrics.json; tqdm handles display."""

  def __init__(self, metrics_path: Path) -> None:
    self.metrics_path = metrics_path

  def report_init(self, config: dict) -> None:
    pass

  def report_epoch_start(
    self, epoch: int, total_epochs: int, steps_in_epoch: int
  ) -> None:
    pass

  def report_batch(
    self, step: int, epoch_step: int, loss: float, acc: float, gs: int
  ) -> None:
    pass

  def report_epoch_end(
    self,
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    epoch_time: float,
  ) -> None:
    pass

  def report_level_metrics(
    self,
    step: int,
    run_id: str,
    metrics: dict,
    sources: dict,
  ) -> None:
    write_level_metrics(metrics, sources, step, run_id, self.metrics_path)

  def report_status(self, status: str) -> None:
    pass

  def report_log(self, message: str) -> None:
    pass

  def report_done(self) -> None:
    pass

  def check_command(self) -> str | None:
    return None


_BATCH_THROTTLE_INTERVAL = 0.1  # seconds


class _BaseStreamReporter:
  """Shared logic for reporters that emit JSON event dicts.

  Subclasses must implement `_emit(msg)` and `check_command()`.
  """

  def __init__(self) -> None:
    self._last_batch_time: float = 0.0
    self._pending_batch: dict | None = None

  def _emit(self, msg: dict) -> None:
    raise NotImplementedError

  def report_init(self, config: dict) -> None:
    self._emit({"type": "init", **config})

  def report_epoch_start(
    self, epoch: int, total_epochs: int, steps_in_epoch: int
  ) -> None:
    self._flush_batch()
    self._emit(
      {
        "type": "epoch_start",
        "epoch": epoch,
        "total_epochs": total_epochs,
        "steps_in_epoch": steps_in_epoch,
      }
    )

  def report_batch(
    self, step: int, epoch_step: int, loss: float, acc: float, gs: int
  ) -> None:
    now = time.monotonic()
    msg = {
      "type": "batch",
      "step": step,
      "epoch_step": epoch_step,
      "loss": loss,
      "acc": acc,
      "gs": gs,
    }
    if now - self._last_batch_time >= _BATCH_THROTTLE_INTERVAL:
      self._emit(msg)
      self._last_batch_time = now
      self._pending_batch = None
    else:
      self._pending_batch = msg

  def _flush_batch(self) -> None:
    if self._pending_batch is not None:
      self._emit(self._pending_batch)
      self._pending_batch = None

  def report_epoch_end(
    self,
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    epoch_time: float,
  ) -> None:
    self._flush_batch()
    self._emit(
      {
        "type": "epoch_end",
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "time": epoch_time,
      }
    )

  def report_level_metrics(
    self,
    step: int,
    run_id: str,
    metrics: dict,
    sources: dict,
  ) -> None:
    self._emit(
      {
        "type": "level_metrics",
        "step": step,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "levels": build_levels_dict(metrics, sources),
      }
    )

  def report_status(self, status: str) -> None:
    self._emit({"type": "status", "status": status})

  def report_log(self, message: str) -> None:
    self._emit({"type": "log", "message": message})

  def report_done(self) -> None:
    self._flush_batch()
    self._emit({"type": "done"})

  def check_command(self) -> str | None:
    raise NotImplementedError


class StdioReporter(_BaseStreamReporter):
  """Reporter that writes JSON lines to stdout for subprocess mode."""

  def __init__(self) -> None:
    super().__init__()
    self._last_cmd_check: float = 0.0

  def _emit(self, msg: dict) -> None:
    sys.stdout.write(json.dumps(msg, separators=(",", ":")) + "\n")
    sys.stdout.flush()

  def check_command(self) -> str | None:
    """Non-blocking stdin read, throttled to avoid per-step syscalls."""
    now = time.monotonic()
    if now - self._last_cmd_check < _BATCH_THROTTLE_INTERVAL:
      return None
    self._last_cmd_check = now
    if select.select([sys.stdin], [], [], 0)[0]:
      line = sys.stdin.readline().strip()
      if line:
        try:
          msg = json.loads(line)
          return msg.get("cmd")
        except json.JSONDecodeError:
          return None
    return None
