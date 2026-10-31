"""Tests for audio_note.audio.recorder with a scripted fake microphone: wall-clock re-anchoring after
a gap in delivered samples, stall detection, and surviving partial-file write failures."""
from __future__ import annotations

import threading
import time
from datetime import datetime

import numpy as np
import pytest

from audio_note.audio import recorder as rec_mod
from audio_note.audio.recorder import Recorder
from audio_note.audio.vad import SAMPLE_RATE, WINDOW

from .conftest import FakeVAD, noise


class FakeMic:
    """Replaces MicCapture: yields scripted (block, wall_time) pairs, then stops the recorder."""

    def __init__(self, script: list[tuple[np.ndarray, float] | None], stop_event: threading.Event, active: bool = True,
                 none_delay: float = 0.0):
        self.script = list(script)
        self.stop_event = stop_event
        self.active = active
        self.none_delay = none_delay
        self.last_block_wall = 0.0
        self.overflows = 0
        self.dropped = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def is_active(self) -> bool:
        return self.active

    def read(self, timeout: float = 1.0):
        if not self.script:
            self.stop_event.set()
            return None
        item = self.script.pop(0)
        if item is None:
            if self.none_delay:
                time.sleep(self.none_delay)
            return None
        block, wall = item
        self.last_block_wall = wall
        return block


def _blocks(n: int, t0: float, seed: int = 0) -> list[tuple[np.ndarray, float]]:
    a = noise(n * WINDOW / SAMPLE_RATE, amp=0.05, seed=seed)
    return [(a[i * WINDOW:(i + 1) * WINDOW], t0 + (i + 1) * WINDOW / SAMPLE_RATE) for i in range(n)]


@pytest.fixture
def rec_cfg(cfg_factory):
    # fast gate: 0.1 s onset, 0.5 s hangover; segments must hold >= 1 s of speech (storage default)
    return cfg_factory({"audio": {"min_speech_s": 0.1, "min_silence_s": 0.5, "pre_roll_s": 0.1, "post_roll_s": 0.1}})


def _make_recorder(cfg, db, monkeypatch, mic_factory, probs):
    stop = threading.Event()
    mic = mic_factory(stop)
    monkeypatch.setattr(rec_mod, "MicCapture", lambda *a, **k: mic)
    r = Recorder(cfg, db, stop)
    r.vad = FakeVAD(probs, default=0.0)
    return r, mic


def test_reanchors_after_gap_in_delivered_samples(rec_cfg, db, monkeypatch):
    t0 = time.mktime(datetime(2026, 10, 31, 23, 0, 0).timetuple())
    speech, quiet = 40, 40                       # 1.28 s speech, 1.28 s silence per phase
    gap = 8 * 3600.0                              # the laptop slept for 8 hours between the phases
    phase1 = _blocks(speech + quiet, t0, seed=1)
    t1 = phase1[-1][1] + gap
    phase2 = _blocks(speech + quiet, t1, seed=2)
    probs = [1.0] * speech + [0.0] * quiet + [1.0] * speech + [0.0] * quiet

    r, mic = _make_recorder(rec_cfg, db, monkeypatch, lambda stop: FakeMic(phase1 + phase2, stop), probs)
    r.run()

    rows = db.query("SELECT * FROM segments ORDER BY started_at")
    assert len(rows) == 2
    s0 = datetime.fromisoformat(rows[0]["started_at"])
    s1 = datetime.fromisoformat(rows[1]["started_at"])
    # first segment starts at (about) the first block's wall time; the second one after the gap, on the next day
    assert abs((s0 - datetime.fromtimestamp(t0)).total_seconds()) < 0.5
    assert abs((s1 - datetime.fromtimestamp(t1)).total_seconds()) < 0.5
    assert s1.strftime("%Y-%m-%d") == "2026-11-01"
    assert r.reanchors == 1
    assert r.segments_written == 2


def test_stall_raises_so_the_service_reopens_the_device(rec_cfg, db, monkeypatch):
    monkeypatch.setattr(rec_mod, "STALL_TIMEOUT_S", 0.05)
    probs = [1.0] * 40 + [0.0] * 5
    script = _blocks(45, time.time()) + [None] * 10
    r, mic = _make_recorder(rec_cfg, db, monkeypatch, lambda stop: FakeMic(script, stop, none_delay=0.03), probs)
    with pytest.raises(RuntimeError, match="stalled"):
        r.run()
    # the open segment (speech, never closed by silence) was flushed before raising, nothing was lost
    assert db.one("SELECT COUNT(*) AS n FROM segments")["n"] == 1
    assert list(rec_cfg.audio_dir.glob("*/*_partial.wav")) == []


def test_inactive_stream_raises_immediately(rec_cfg, db, monkeypatch):
    script = _blocks(2, time.time()) + [None]
    r, mic = _make_recorder(rec_cfg, db, monkeypatch, lambda stop: FakeMic(script, stop, active=False), [0.0] * 5)
    with pytest.raises(RuntimeError, match="active=False"):
        r.run()


def test_partial_write_failure_keeps_recording(rec_cfg, db, monkeypatch):
    class BrokenPartial:
        def __init__(self, cfg, started_at):
            self.started_at = started_at
            self.path = cfg.audio_dir / "x_partial.wav"
            self.writes = 0

        def write(self, block):
            self.writes += 1
            if self.writes >= 3:
                raise OSError("disk full")

        def close(self):
            pass

        def abandon(self):
            pass

    monkeypatch.setattr(rec_mod, "PartialWriter", BrokenPartial)
    probs = [1.0] * 40 + [0.0] * 40
    r, mic = _make_recorder(rec_cfg, db, monkeypatch, lambda stop: FakeMic(_blocks(80, time.time()), stop), probs)
    r.run()   # must not raise
    assert r._partial is None
    assert r.segments_written == 1
    assert db.one("SELECT COUNT(*) AS n FROM segments")["n"] == 1
