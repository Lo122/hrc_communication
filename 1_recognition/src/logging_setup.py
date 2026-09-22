"""Logging for the recognition layer.

Everything in `1_recognition/` logs through a single `"recognition"` parent
logger, configured here. Two reasons it is a parent logger rather than the
root one:

  - **Isolation.** The root logger is shared with every library in the
    process. Configuring it would route `ultralytics`, `torch` and
    `matplotlib` -- all chatty at INFO/DEBUG -- into our handlers, plus
    `4_execution/ros_communication.py`. Attaching to `"recognition"` with
    `propagate = False` seals this layer's output off from all of that, and
    keeps our records out of anyone else's handlers.
  - **It is NOT the event log.** `0_core/logger.py`'s `EventLogger` is a
    separate, hand-rolled JSON-lines writer that appends the communication
    dialog to `hrc_communication_events.log`. It never touches `logging`, so
    nothing configured here can affect it, and nothing here is written there.

Module-level usage, in place of `logging.getLogger(__name__)`:

    from logging_setup import get_logger
    logger = get_logger(__name__)          # -> "recognition.frame_source"

Entry-point usage, once, at the top of main():

    from logging_setup import configure_logging
    configure_logging("calibrate_camera")

The console shows INFO and above (what `print` used to show); the file gets
DEBUG and above, which is the point of the exercise -- per-frame detail that
is too noisy to watch live but is exactly what you want when reading back a
run afterwards.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

#: Every logger in this layer hangs off this one, which is where handlers attach.
ROOT_LOGGER_NAME = "recognition"

#: Default home for log files when a caller doesn't name one. Already gitignored.
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"

# The console stands in for print(), so it stays terse -- level and message, no
# timestamp or logger name. The file is the forensic record, so it carries
# everything needed to reconstruct the order of events months later.
CONSOLE_FORMAT = "%(levelname)-7s %(message)s"
FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Logger for a module in this layer, namespaced under ROOT_LOGGER_NAME.

    Pass `__name__`. Modules here are imported flat (via sys.path) rather than
    as a package, so `__name__` is a bare module name like "frame_source" with
    no common prefix -- prefixing is what gives the layer one configurable
    subtree.
    """
    name = name.split(".")[-1]
    if name in ("__main__", ROOT_LOGGER_NAME):
        return logging.getLogger(ROOT_LOGGER_NAME)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def configure_logging(
    tool: str,
    *,
    log_file: str | Path | None = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_dir: str | Path | None = None,
) -> Path | None:
    """Attach console + file handlers to the `"recognition"` logger.

    tool: short name of the entry point, used for the default filename.
    log_file: exact path to write to. `run_recognition.py` passes the run
        directory's own run.log, so a run's diagnostics sit beside the
        frames.csv/events.csv/run.json they describe. When omitted, defaults
        to <log_dir>/<tool>_<YYYYmmdd_HHMMSS>.log.
    log_dir: overrides DEFAULT_LOG_DIR for that default filename.

    Returns the log file path, or None if no file could be opened (a bad path
    is reported on the console and downgraded to console-only logging -- a
    tool that cannot write its log should still run).

    Idempotent: existing handlers are removed first, so the setup tools that
    chain into each other (calibrate_camera.py's `full` runs intrinsic then
    extrinsic) don't stack duplicates and print every line twice.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    # The logger itself must pass the most permissive of the two levels; each
    # handler then filters down to its own.
    logger.setLevel(min(console_level, file_level))
    # Without this, records also travel to the root logger, where anyone else's
    # basicConfig() would print them a second time.
    logger.propagate = False

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    logger.addHandler(console)

    path = Path(log_file) if log_file is not None else _default_log_path(tool, log_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not open log file %s (%s) -- logging to console only.",
                       path, exc)
        return None

    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
    logger.addHandler(file_handler)
    logger.debug("Logging to %s (console=%s, file=%s)", path,
                 logging.getLevelName(console_level), logging.getLevelName(file_level))
    return path


def _default_log_path(tool: str, log_dir: str | Path | None) -> Path:
    directory = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
    return directory / f"{tool}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
