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


# --- channel levels with microphone + system audio (speakers v4 D.2) -------------------------------

class FakeLoop:
    """Replaces LoopbackCapture: one scripted chunk per microphone block (empty = nothing playing)."""

    dropped = 0
    device = None

    def __init__(self, chunks: list[np.ndarray]):
        self.chunks = list(chunks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def read_available(self) -> np.ndarray:
        return self.chunks.pop(0) if self.chunks else np.zeros(0, dtype=np.float32)


def _mixed_recorder(cfg, db, monkeypatch, script, loop_chunks, probs, loop_cls=FakeLoop):
    from audio_note.audio.capture import MixedCapture

    r, mic = _make_recorder(cfg, db, monkeypatch, lambda stop: FakeMic(script, stop), probs)
    monkeypatch.setattr(rec_mod, "open_capture", lambda c, m: MixedCapture(m, loop_cls(loop_chunks)))
    return r


def _split_blocks(n: int, size: int, t0: float, seed: int = 0) -> list[tuple[np.ndarray, float]]:
    a = noise(n * size / SAMPLE_RATE, amp=0.05, seed=seed)
    return [(a[i * size:(i + 1) * size], t0 + (i + 1) * size / SAMPLE_RATE) for i in range(n)]


def _pairs(row) -> np.ndarray:
    return np.frombuffer(row["channel_rms"], dtype=np.float16).astype(np.float32).reshape(-1, 2)


@pytest.fixture
def mix_cfg(cfg_factory):
    # 0.3 s pre-roll so the segment really starts with blocks recorded before the gate opened
    return cfg_factory({"audio": {"min_speech_s": 0.1, "min_silence_s": 0.5, "pre_roll_s": 0.3, "post_roll_s": 0.1}})


def test_segment_stores_one_channel_pair_per_block_including_pre_roll(mix_cfg, db, monkeypatch):
    lead, speech, quiet = 20, 40, 40
    n = lead + speech + quiet
    loud = set(range(30, 40))                                    # the far side of a call talks during blocks 30..39
    chunks = [np.full(WINDOW, 0.3, dtype=np.float32) if b in loud else np.zeros(0, dtype=np.float32) for b in range(n)]
    probs = [0.0] * lead + [1.0] * speech + [0.0] * quiet
    r = _mixed_recorder(mix_cfg, db, monkeypatch, _blocks(n, time.time()), chunks, probs)
    r.run()

    rows = db.query("SELECT * FROM segments")
    assert len(rows) == 1
    row = rows[0]
    assert row["channel_block_s"] == pytest.approx(WINDOW / SAMPLE_RATE)
    pairs = _pairs(row)
    n_samples = int(round(row["duration_s"] * SAMPLE_RATE))
    assert len(pairs) == -(-n_samples // WINDOW)                  # one pair per block of the segment (last one partial)
    # gate opens after 4 speech windows (block 23) with 10 windows of pre-roll -> the segment's first block is 14
    first = lead + 4 - 10
    sys_loud = {first + k for k in range(len(pairs)) if pairs[k, 1] > 0.1}
    assert sys_loud == loud
    assert np.all(pairs[:, 0] > 0.03) and np.all(pairs[:, 0] < 0.07)   # the microphone level is always there
    share = pairs[:, 1] ** 2 / (pairs[:, 1] ** 2 + pairs[:, 0] ** 2)
    idx = [b - first for b in sorted(loud)]
    assert np.all(share[idx] > 0.9) and np.all(np.delete(share, idx) == 0.0)
    # memory stays bounded: only what the (idle) gate's pre-roll ring may still need is kept
    assert len(r._channels) <= 11


def test_channel_pairs_cover_segments_that_do_not_align_to_blocks(mix_cfg, db, monkeypatch):
    size = 400                                                   # 25 ms blocks vs 32 ms VAD windows
    n = 160
    loud = set(range(60, 80))
    chunks = [np.full(size, 0.3, dtype=np.float32) if b in loud else np.zeros(0, dtype=np.float32) for b in range(n)]
    n_windows = n * size // WINDOW
    probs = [0.0] * 20 + [1.0] * 50 + [0.0] * (n_windows - 70)
    r = _mixed_recorder(mix_cfg, db, monkeypatch, _split_blocks(n, size, time.time()), chunks, probs)
    r.run()

    row = db.one("SELECT * FROM segments")
    assert row is not None and row["channel_block_s"] == pytest.approx(size / SAMPLE_RATE)
    pairs = _pairs(row)
    # segment starts at window 20 + 4 - 10 = 14; pair 0 is the block holding the segment's first sample
    start = 14 * WINDOW
    end = start + int(round(row["duration_s"] * SAMPLE_RATE))
    first_block, last_block = start // size, (end - 1) // size
    assert len(pairs) == last_block - first_block + 1
    sys_loud = {first_block + k for k in range(len(pairs)) if pairs[k, 1] > 0.1}
    assert sys_loud and sys_loud == loud & set(range(first_block, last_block + 1))


def test_two_segments_each_get_their_own_channel_pairs(mix_cfg, db, monkeypatch):
    phase = [0.0] * 20 + [1.0] * 40 + [0.0] * 40               # 100 blocks per phase, one segment each
    probs = phase + phase
    loud = set(range(130, 140))                                  # only during the second segment
    chunks = [np.full(WINDOW, 0.3, dtype=np.float32) if b in loud else np.zeros(0, dtype=np.float32) for b in range(200)]
    r = _mixed_recorder(mix_cfg, db, monkeypatch, _blocks(200, time.time()), chunks, probs)
    r.run()
    rows = db.query("SELECT * FROM segments ORDER BY started_at")
    assert len(rows) == 2
    p0, p1 = _pairs(rows[0]), _pairs(rows[1])
    for row, p in ((rows[0], p0), (rows[1], p1)):
        assert len(p) == -(-int(round(row["duration_s"] * SAMPLE_RATE)) // WINDOW)
    assert float(p0[:, 1].max()) == 0.0
    assert {114 + k for k in range(len(p1)) if p1[k, 1] > 0.1} == loud


def test_microphone_only_recording_has_no_channel_levels(rec_cfg, db, monkeypatch):
    probs = [1.0] * 40 + [0.0] * 40
    r, mic = _make_recorder(rec_cfg, db, monkeypatch, lambda stop: FakeMic(_blocks(80, time.time()), stop), probs)
    r.run()
    row = db.one("SELECT * FROM segments")
    assert row is not None and row["channel_rms"] is None and row["channel_block_s"] is None
    assert len(r._channels) == 0


def test_failed_loopback_records_microphone_only(mix_cfg, db, monkeypatch):
    class BrokenLoop(FakeLoop):
        def __enter__(self):
            raise RuntimeError("endpoint gone")

    probs = [1.0] * 40 + [0.0] * 40
    r = _mixed_recorder(mix_cfg, db, monkeypatch, _blocks(80, time.time()), [], probs, loop_cls=BrokenLoop)
    r.run()
    row = db.one("SELECT * FROM segments")
    assert row is not None and row["channel_rms"] is None


def test_channel_level_failure_never_loses_the_segment(mix_cfg, db, monkeypatch):
    probs = [1.0] * 40 + [0.0] * 40
    r = _mixed_recorder(mix_cfg, db, monkeypatch, _blocks(80, time.time()), [], probs)

    def boom(seg):
        raise ValueError("boom")

    monkeypatch.setattr(r, "_segment_channels", boom)
    r.run()
    row = db.one("SELECT * FROM segments")
    assert row is not None and row["channel_rms"] is None and r.segments_written == 1
