"""Self-playback guard: never record what audio_note's own page is playing.

Listening to old sentences in the page (to check who spoke, label them, judge a voice group) plays them through
the speakers or headphones. The system-audio channel (WASAPI loopback) and, with speakers, the microphone would
otherwise record them again as new speech of today, at the wrong time and with other people's voices.

So the page's player reports what it does and the recorder leaves those moments out:

* the player sends POST /api/playback {tab, playing: true} on "play" and every HEARTBEAT_S while it plays, and
  {playing: false} on pause / end / close. A tab that stops reporting (closed, crashed, asleep) counts as stopped
  TTL_S after its last report;
* the guard keeps the windows in wall-clock time (time.time()) for HISTORY_S. The recorder asks about each capture
  block by the block's own callback time, so a queue backlog cannot let a played block slip through. LEAD_S before
  a start and TAIL_S after a stop are covered too (output latency, the loopback backlog, the room echo dying away);
* audio.playback_guard picks what is left out:
  - "all" (default): microphone and system audio;
  - "system": only the system audio (with headphones you can keep talking while a clip plays);
  - "off": nothing.

A read-only page (the recorder runs in another process on the same data directory) mirrors its windows to
<data_dir>/playback.json (write_file), and the recording process reads that file (FileWindows).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

log = logging.getLogger(__name__)

HEARTBEAT_S = 2.0      # the page reports this often while it plays (ui/static/js/ui/player.js)
TTL_S = 6.0            # a playing tab that has not reported for this long is taken as stopped
LEAD_S = 0.3           # also leave out this much before a reported start
TAIL_S = 1.5           # ... and this much after a reported stop
HISTORY_S = 120.0      # closed windows are kept this long (the recorder may still be behind)
MAX_TABS = 32
MAX_WINDOWS = 256
FILE_NAME = "playback.json"
FILE_POLL_S = 0.2      # the recording process looks at playback.json at most this often

MODES = ("all", "system", "off")
DEFAULT_MODE = "all"

Window = tuple[float, float]


def guard_mode(cfg: Any) -> str:
    """audio.playback_guard as one of MODES. YAML true / false mean "all" / "off"; anything unknown is the default."""
    try:
        v = cfg.audio.get("playback_guard", DEFAULT_MODE)
    except Exception:  # noqa: BLE001 - a config without an audio section
        return DEFAULT_MODE
    if v is None:
        return DEFAULT_MODE
    if isinstance(v, bool):
        return "all" if v else "off"
    s = str(v).strip().lower()
    if s in ("false", "no", "0", "none"):
        return "off"
    if s in ("true", "yes", "1"):
        return "all"
    return s if s in MODES else DEFAULT_MODE


def covered(windows: Iterable[Window], t: float) -> bool:
    """Is wall-clock time `t` inside one of the windows (widened by LEAD_S before and TAIL_S after)?"""
    for s, e in windows:
        if s - LEAD_S <= t <= e + TAIL_S:
            return True
    return False


class PlaybackGuard:
    """What the page's players are doing, as wall-clock windows. Thread-safe; cheap enough to ask per block."""

    def __init__(self, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._lock = threading.Lock()
        self._open: dict[str, list[float]] = {}     # tab -> [start, last report] while it plays
        self._closed: list[Window] = []             # finished windows, oldest first
        self._seq: dict[str, int] = {}              # tab -> newest report number seen
        self.reports = 0
        self.stale = 0

    def report(self, tab: Any, playing: bool, now: float | None = None, seq: int | None = None) -> bool:
        """Record one report of a page's player. Returns whether any tab is playing afterwards.

        `seq` (the tab's running report number) puts reports that overtook each other on the way back in order: one
        not newer than the newest seen for the tab is ignored, so a late "stopped" cannot end a playback that a
        newer "playing" already reported."""
        t = self._clock() if now is None else float(now)
        key = str(tab or "")[:64] or "?"
        with self._lock:
            self.reports += 1
            if seq is not None:
                last = self._seq.get(key)
                if last is not None and seq <= last:
                    self.stale += 1
                    return bool(self._open)
                self._seq.pop(key, None)
                self._seq[key] = int(seq)
                while len(self._seq) > MAX_TABS * 4:
                    self._seq.pop(next(iter(self._seq)))
            self._expire(t)
            cur = self._open.get(key)
            if playing:
                if cur is None:
                    if len(self._open) >= MAX_TABS:
                        stalest = min(self._open, key=lambda k: self._open[k][1])
                        s, seen = self._open.pop(stalest)
                        self._closed.append((s, seen))
                    self._open[key] = [t, t]
                else:
                    cur[1] = max(cur[1], t)
            elif cur is not None:
                del self._open[key]
                self._closed.append((cur[0], max(t, cur[0])))
            self._prune(t)
            return bool(self._open)

    def _expire(self, t: float) -> None:
        for key, (s, seen) in list(self._open.items()):
            if t - seen > TTL_S:
                del self._open[key]
                self._closed.append((s, seen + TTL_S))

    def _prune(self, t: float) -> None:
        cutoff = t - HISTORY_S
        if self._closed and self._closed[0][1] < cutoff:
            self._closed = [w for w in self._closed if w[1] >= cutoff]
        if len(self._closed) > MAX_WINDOWS:
            self._closed = self._closed[-MAX_WINDOWS:]

    def windows(self, now: float | None = None) -> list[Window]:
        """Every window still of interest; one that is still playing ends TTL_S after its last report."""
        t = self._clock() if now is None else float(now)
        with self._lock:
            self._expire(t)
            self._prune(t)
            return list(self._closed) + [(s, seen + TTL_S) for s, seen in self._open.values()]

    def playing(self, now: float | None = None) -> bool:
        t = self._clock() if now is None else float(now)
        with self._lock:
            self._expire(t)
            return bool(self._open)

    def covers(self, t: float, now: float | None = None) -> bool:
        """Should audio captured at wall-clock time `t` be left out?"""
        with self._lock:
            if not self._open and not self._closed:
                return False
        return covered(self.windows(now), t)

    def clear(self) -> None:
        with self._lock:
            self._open.clear()
            self._closed.clear()


#: the one guard of this process: the page's reports land here, the recorder in this process reads it
GUARD = PlaybackGuard()


def file_path(data_dir: Any) -> Path:
    return Path(data_dir) / FILE_NAME


def write_file(path: Path, windows: list[Window], now: float | None = None) -> None:
    """Mirror `windows` for a recorder in another process (atomic replace; never raises)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"written": time.time() if now is None else now,
                                   "windows": [[round(s, 3), round(e, 3)] for s, e in windows]}), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as e:  # noqa: BLE001 - the guard is a nicety; the page must keep playing
        log.debug("playback windows not written to %s: %s", path, e)


def _parse(text: str) -> list[Window]:
    data = json.loads(text)
    raw = data.get("windows") if isinstance(data, dict) else None
    out: list[Window] = []
    for w in raw[:MAX_WINDOWS] if isinstance(raw, list) else []:
        try:
            s, e = float(w[0]), float(w[1])
        except (TypeError, ValueError, IndexError):
            continue
        if s == s and e == e and e >= s:       # not NaN, not reversed
            out.append((s, e))
    return out


class FileWindows:
    """Windows a read-only page in another process mirrored to <data_dir>/playback.json (looked at no more often
    than every FILE_POLL_S; a missing or broken file means no windows)."""

    def __init__(self, path: Path, clock: Callable[[], float] = time.time):
        self.path = Path(path)
        self._clock = clock
        self._checked = float("-inf")
        self._windows: list[Window] = []

    def windows(self) -> list[Window]:
        now = self._clock()
        if now - self._checked < FILE_POLL_S:
            return self._windows
        self._checked = now
        # re-read every time (a few hundred bytes): a modification time can miss two writes in quick succession
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            self._windows = []
            return self._windows
        try:
            self._windows = _parse(text)
        except Exception as e:  # noqa: BLE001 - half-written / foreign file
            log.debug("playback windows in %s unreadable: %s", self.path, e)
            self._windows = []
        return self._windows

    def covers(self, t: float) -> bool:
        w = self.windows()
        return bool(w) and covered(w, t)


__all__ = ["DEFAULT_MODE", "FILE_NAME", "GUARD", "HEARTBEAT_S", "LEAD_S", "MODES", "TAIL_S", "TTL_S",
           "FileWindows", "PlaybackGuard", "covered", "file_path", "guard_mode", "write_file"]
