"""Logging configuration shared by the CLI and the service.

setup_logging(cfg, verbose) installs two handlers on the root logger:
  * a rich console handler (level INFO, or DEBUG with --verbose)
  * a RotatingFileHandler at <data_dir>/<log.file> (5 MB x 3 backups, UTF-8)

Calling it twice (e.g. in tests) replaces the handlers it installed earlier instead of
stacking them. Noisy third-party loggers (httpx/openai/urllib3) are kept at WARNING
unless verbose is on.
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from .config import Config

_MARK = "_audio_note_handler"
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3
NOISY = ("httpx", "httpcore", "httpx2", "httpcore2", "openai", "urllib3", "PIL", "numba", "jieba")


def _remove_own_handlers(root: logging.Logger) -> None:
    for h in list(root.handlers):
        if getattr(h, _MARK, False):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:  # noqa: BLE001
                pass


def setup_logging(cfg: Config, verbose: bool = False) -> Path | None:
    """Install console + rotating file handlers. Returns the log file path (None if it could not be opened)."""
    level_name = "DEBUG" if verbose else str(cfg.log.level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    _remove_own_handlers(root)
    root.setLevel(min(level, logging.DEBUG) if verbose else level)

    # console
    try:
        from rich.logging import RichHandler

        console = RichHandler(level=level, rich_tracebacks=verbose, show_path=verbose, markup=False,
                              log_time_format="%H:%M:%S")
        console.setFormatter(logging.Formatter("%(message)s"))
    except Exception:  # noqa: BLE001  (rich missing or console unusable)
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    setattr(console, _MARK, True)
    root.addHandler(console)

    # file
    log_path: Path | None = None
    try:
        data_dir = cfg.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        log_path = data_dir / str(cfg.log.file or "audio_note.log")
        fh = logging.handlers.RotatingFileHandler(log_path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8")
        fh.setLevel(logging.DEBUG if verbose else logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(threadName)s %(name)s: %(message)s"))
        setattr(fh, _MARK, True)
        root.addHandler(fh)
    except Exception as e:  # noqa: BLE001
        root.warning("cannot open log file (%s); console logging only", e)
        log_path = None

    for name in NOISY:
        # even in verbose mode, HTTP wire-level debug lines drown everything else
        logging.getLogger(name).setLevel(logging.INFO if verbose else logging.WARNING)
    return log_path
