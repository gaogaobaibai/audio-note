"""Regression tests for the 24/7 robustness fixes (review findings 1-13):

  storage   transient-failure classifier, expiry keeping untranscribed failed audio, disk-cap eviction
            tiers, no orphan file when the DB insert fails, stale-partial listing with an exclusion
  capture   stream closed even when stop() raises, fallback closes the first stream and sets sr before
            start(), PortAudio device re-enumeration helper
  recorder  periodic recovery of partials that appear while running
  service   device refresh before every recorder restart (not the first start)
  scheduler hourly re-queue of transient failures (bounded), multi-day catch-up, midnight grace for the
            yesterday recheck, partial digest not marked done, unpushed digest retry, duplicate-push
            suppression, push of a stored digest without an LLM client
  daily     partial digest flagged, stored with a reduced count and regenerated on the next run
  logging   CLI role detection and the separate audio_note.cli.log file

No network, no real audio device: sounddevice is replaced by a fake module object.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
import types
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from audio_note import scheduler as sch
from audio_note import service
from audio_note.audio import capture
from audio_note.audio import recorder as rec_mod
from audio_note.audio import storage
from audio_note.audio.storage import (
    cleanup, dir_size_bytes, finalize_segment, is_transient_error_text, recover_partials, segment_path, stale_partials,
)
from audio_note.audio.vad import SAMPLE_RATE, WINDOW
from audio_note.config import PROJECT_ROOT
from audio_note.db import DB
from audio_note.llm.client import LLMError

from .conftest import FakeVAD, noise

MODELS_DIR = PROJECT_ROOT / "models"
DAY = "2026-10-30"


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
@pytest.fixture
def scfg(cfg_factory):
    """Config for scheduler/daily tests; root handlers installed by setup_logging are removed afterwards."""
    cfg = cfg_factory({
        "pipeline": {"process_interval_min": 10},
        "storage": {"cleanup_interval_min": 30},
        "analysis": {"daily_time": "22:30", "push_daily": True},
    })
    yield cfg
    from audio_note.logging_setup import _remove_own_handlers

    _remove_own_handlers(logging.getLogger())


def _seg(cfg, db, started: datetime, status: str = "new", *, error: str | None = None, speech_s=2.0,
         seconds: float = 2.0, with_file: bool = True):
    p = segment_path(cfg, started, ".wav")
    if with_file:
        sf.write(p, noise(seconds, amp=0.05), SAMPLE_RATE)
    sid = db.add_segment(p, started, seconds, speech_s=speech_s)
    if status != "new":
        db.set_segment_status(sid, status, error)
    return sid, p


def _utt(db: DB, seg: int, day: str, hhmm: str, text: str = "你好", who: str | None = "张三") -> int:
    return db.add_utterances([{
        "segment_id": seg, "start_s": 0.0, "end_s": 1.0, "abs_start": f"{day}T{hhmm}:00", "day": day,
        "cluster_label": 0, "speaker_id": None, "speaker_name": who, "speaker_score": None,
        "text_raw": text, "text": text,
    }])[0]


def _fake_jobs(calls: list):
    return {
        "process": lambda: calls.append(("process",)),
        "cleanup": lambda: calls.append(("cleanup",)),
        "daily": lambda day, push: calls.append(("daily", day, push)),
        "push": lambda day: calls.append(("push", day)) or {"toast": None},
    }


def _fake_push_module(monkeypatch, results: dict[str, str | None]) -> list:
    """Replace audio_note.push.base with a stub Notifier (no toast / HTTP)."""
    sent: list = []

    class Message:
        def __init__(self, title, body_md, url=None):
            self.title, self.body_md, self.url = title, body_md, url

    class Notifier:
        def __init__(self, cfg):
            self.cfg = cfg

        def send(self, msg):
            sent.append(msg)
            return dict(results)

    mod = types.ModuleType("audio_note.push.base")
    mod.Message, mod.Notifier = Message, Notifier
    monkeypatch.setitem(sys.modules, "audio_note.push.base", mod)
    return sent


# ----------------------------------------------------------------------------------------------
# storage: transient classifier, expiry, disk cap, orphan file, stale partials
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("err,transient", [
    ("LLMError: asr: giving up after 4 attempts: Connection error.", True),
    ("LLMError: chat: HTTP 503 overloaded", True),
    ("APITimeoutError: Request timed out.", True),
    ("LLMError: asr: HTTP 429 rate limit", True),
    ("LLMError: asr: HTTP 402 insufficient balance", True),
    ("FileNotFoundError: segment file missing: x.opus", False),
    ("LLMError: asr: HTTP 400 bad request", False),
    ("ValueError: audio too short", False),
    (None, False),
    ("", False),
])
def test_is_transient_error_text(err, transient):
    assert is_transient_error_text(err) is transient


def test_cleanup_expiry_keeps_failed_transient_audio_without_transcript(cfg_factory):
    cfg = cfg_factory({"storage": {"keep_audio_days": 30, "max_disk_gb": 0}})
    db = DB(cfg.db_path)
    old = datetime.now() - timedelta(days=45)
    kept, p_kept = _seg(cfg, db, old, "failed", error="LLMError: asr: giving up after 4 attempts: Connection error.")
    gone, p_gone = _seg(cfg, db, old + timedelta(hours=1), "failed", error="FileNotFoundError: segment file missing")
    transcribed, p_tr = _seg(cfg, db, old + timedelta(hours=2), "failed", error="LLMError: chat: HTTP 503")
    db.execute("INSERT INTO transcripts(segment_id, text_raw) VALUES (?, ?)", (transcribed, "有文本了"))
    stats = cleanup(cfg, db)
    assert stats["expired"] == 2
    assert db.get_segment(kept)["status"] == "failed" and p_kept.exists()        # retryable, never transcribed
    assert db.get_segment(gone)["status"] == "deleted" and not p_gone.exists()   # deterministic failure
    assert db.get_segment(transcribed)["status"] == "deleted" and not p_tr.exists()  # text exists: audio expendable


def test_cleanup_disk_cap_evicts_failed_then_old_new_as_last_resort(cfg_factory):
    cfg = cfg_factory({"storage": {"keep_audio_days": 30, "max_disk_gb": 1e-9}})
    db = DB(cfg.db_path)
    now = datetime.now()
    done, p_done = _seg(cfg, db, now - timedelta(days=3), "done")
    failed, p_failed = _seg(cfg, db, now - timedelta(days=2), "failed", error="LLMError: asr: giving up")
    old_new, p_old_new = _seg(cfg, db, now - timedelta(days=40), "new")
    fresh_new, p_fresh = _seg(cfg, db, now - timedelta(hours=1), "new")
    stats = cleanup(cfg, db)
    assert stats["disk_cap"] == 3
    assert db.get_segment(done)["status"] == "deleted" and not p_done.exists()
    assert db.get_segment(failed)["status"] == "deleted" and not p_failed.exists()
    assert "evicted" in (db.get_segment(failed)["error"] or "")
    assert db.get_segment(old_new)["status"] == "deleted" and not p_old_new.exists()
    assert db.get_segment(fresh_new)["status"] == "new" and p_fresh.exists()   # recent unprocessed audio survives


def test_cleanup_disk_cap_stops_at_transcribed_audio_when_that_is_enough(cfg_factory):
    one_file = 2 * SAMPLE_RATE * 2 + 44                      # 2 s PCM16 wav
    cap_gb = (one_file * 1.5) / (1024 ** 3)                  # three files exceed it, one fits (<= 0.9 * cap)
    cfg = cfg_factory({"storage": {"keep_audio_days": 30, "max_disk_gb": cap_gb}})
    db = DB(cfg.db_path)
    now = datetime.now()
    a, pa = _seg(cfg, db, now - timedelta(hours=5), "done")
    b, pb = _seg(cfg, db, now - timedelta(hours=4), "done")
    f, pf = _seg(cfg, db, now - timedelta(hours=3), "failed", error="LLMError: asr: giving up")
    assert dir_size_bytes(cfg.audio_dir) > cap_gb * 1024 ** 3
    stats = cleanup(cfg, db)
    assert stats["disk_cap"] == 2
    assert not pa.exists() and not pb.exists()
    assert db.get_segment(f)["status"] == "failed" and pf.exists()


def test_finalize_removes_file_when_db_insert_fails_and_keeps_partial(cfg, db, monkeypatch):
    t = datetime(2026, 10, 30, 9, 0, 0)
    partial = segment_path(cfg, t, "_partial.wav")
    sf.write(partial, noise(2.0, amp=0.05), SAMPLE_RATE)

    def boom(*a, **k):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(db, "add_segment", boom)
    with pytest.raises(RuntimeError, match="locked"):
        finalize_segment(cfg, db, noise(2.0, amp=0.05), t, speech_s=2.0, partial=partial)
    assert list(partial.parent.glob("*.opus")) == []          # no unregistered audio left behind
    assert partial.exists()                                    # the recovery copy is kept
    assert db.query("SELECT * FROM segments") == []


def test_stale_partials_skip_fresh_and_excluded_files(cfg, db):
    old = segment_path(cfg, datetime(2026, 10, 30, 9, 0, 0), "_partial.wav")
    fresh = segment_path(cfg, datetime(2026, 10, 30, 9, 5, 0), "_partial.wav")
    for p in (old, fresh):
        sf.write(p, noise(2.0, amp=0.05), SAMPLE_RATE)
    ago = time.time() - 60
    os.utime(old, (ago, ago))
    assert stale_partials(cfg) == [old]
    assert stale_partials(cfg, exclude=old) == []
    # the live recorder's own file is never touched even when its mtime is old
    assert recover_partials(cfg, db, FakeVAD(default=1.0), exclude=old) == 0
    assert old.exists() and fresh.exists()
    assert recover_partials(cfg, db, FakeVAD(default=1.0)) == 1
    assert not old.exists() and fresh.exists()


# ----------------------------------------------------------------------------------------------
# capture: fake sounddevice
# ----------------------------------------------------------------------------------------------
class FakeStream:
    """Stands in for sd.InputStream. start() may fail for some sample rates and may fire the callback
    right away (PortAudio's thread does that); stop() may raise like an invalidated WASAPI endpoint."""
    fail_start_rates: set[int] = set()
    raise_on_stop = False
    fire_on_start = False
    instances: list["FakeStream"] = []

    def __init__(self, device=None, channels=1, samplerate=16000, dtype="float32", blocksize=512, callback=None, latency=None):
        self.samplerate, self.blocksize, self.callback = int(samplerate), int(blocksize), callback
        self.started = self.stopped = self.closed = False
        self.active = False
        FakeStream.instances.append(self)

    def start(self):
        if self.samplerate in FakeStream.fail_start_rates:
            raise RuntimeError(f"Error opening InputStream: Invalid sample rate [PaErrorCode -9997] ({self.samplerate})")
        self.started = self.active = True
        if FakeStream.fire_on_start and self.callback is not None:
            self.callback(np.zeros((self.blocksize, 1), dtype=np.float32), self.blocksize, None, None)

    def stop(self):
        if FakeStream.raise_on_stop:
            raise RuntimeError("Error stopping stream: Unanticipated host error [PaErrorCode -9999]")
        self.stopped, self.active = True, False

    def close(self):
        self.closed = True


@pytest.fixture
def fake_sd(monkeypatch):
    FakeStream.instances = []
    FakeStream.fail_start_rates = set()
    FakeStream.raise_on_stop = False
    FakeStream.fire_on_start = False
    calls = {"terminate": 0, "initialize": 0}

    def _terminate():
        calls["terminate"] += 1

    def _initialize():
        calls["initialize"] += 1

    sd = SimpleNamespace(
        InputStream=FakeStream,
        query_devices=lambda device=None, kind=None: {"default_samplerate": 48000.0, "name": "Fake Mic", "max_input_channels": 1},
        _terminate=_terminate,
        _initialize=_initialize,
        calls=calls,
    )
    monkeypatch.setattr(capture, "sd", sd)
    return sd


def test_exit_closes_stream_even_when_stop_raises(fake_sd):
    FakeStream.raise_on_stop = True
    with capture.MicCapture(device=None) as mic:
        assert mic.is_active()
        s = mic._stream
    assert s.closed and mic._stream is None                   # no PortAudio stream leaked
    assert not mic.is_active()


def test_fallback_closes_first_stream_and_resamples_from_the_first_callback(fake_sd):
    FakeStream.fail_start_rates = {16000}
    FakeStream.fire_on_start = True
    with capture.MicCapture(device=None) as mic:
        assert mic.sr == 48000
        first, second = FakeStream.instances
        assert first.samplerate == 16000 and first.closed and not first.started
        assert second.samplerate == 48000 and second.started and second.blocksize == 1536
        # the callback fired inside start(): with self.sr already final the block was resampled to 16 kHz
        block, wall = mic.queue.get_nowait()
        assert block.shape == (WINDOW,) and wall > 0
    assert second.stopped and second.closed


def test_refresh_devices_reinitialises_portaudio_and_never_raises(fake_sd):
    assert capture.refresh_devices() is True
    assert fake_sd.calls == {"terminate": 1, "initialize": 1}

    def broken():
        raise RuntimeError("PaErrorCode -10000")

    fake_sd._terminate = broken
    assert capture.refresh_devices() is False


# ----------------------------------------------------------------------------------------------
# recorder: periodic partial recovery
# ----------------------------------------------------------------------------------------------
class FakeMic:
    def __init__(self, script, stop_event, on_first_read=None):
        self.script, self.stop_event, self.on_first_read = list(script), stop_event, on_first_read
        self.last_block_wall = 0.0
        self.overflows = self.dropped = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def is_active(self):
        return True

    def read(self, timeout=1.0):
        if self.on_first_read is not None:
            cb, self.on_first_read = self.on_first_read, None
            cb()
        if not self.script:
            self.stop_event.set()
            return None
        block, wall = self.script.pop(0)
        self.last_block_wall = wall
        return block


def test_recorder_recovers_partial_that_appears_while_running(cfg_factory, monkeypatch):
    if not (MODELS_DIR / "silero_vad.onnx").exists():
        pytest.skip("silero model missing")
    cfg = cfg_factory({"audio": {"min_speech_s": 0.1, "min_silence_s": 0.5, "pre_roll_s": 0.1, "post_roll_s": 0.1}})
    db = DB(cfg.db_path)
    monkeypatch.setattr(rec_mod, "PARTIAL_RECOVERY_INTERVAL_S", 0.0)
    leftover = segment_path(cfg, datetime(2026, 10, 30, 9, 0, 0), "_partial.wav")

    def drop_leftover():
        # a crash + restart within seconds left this file; it was too fresh at start-up
        sf.write(leftover, noise(2.0, amp=0.05), SAMPLE_RATE)
        ago = time.time() - 60
        os.utime(leftover, (ago, ago))

    a = noise(20 * WINDOW / SAMPLE_RATE, amp=0.05)
    t0 = time.time()
    script = [(a[i * WINDOW:(i + 1) * WINDOW], t0 + (i + 1) * WINDOW / SAMPLE_RATE) for i in range(20)]
    stop = threading.Event()
    mic = FakeMic(script, stop, on_first_read=drop_leftover)
    monkeypatch.setattr(rec_mod, "MicCapture", lambda *a, **k: mic)
    r = rec_mod.Recorder(cfg, db, stop)
    r.vad = FakeVAD(default=0.0)              # live stream: silence
    r._recover_vad = FakeVAD(default=1.0)     # recovery: all speech
    r.run()
    assert r.partials_recovered == 1 and not leftover.exists()
    rows = db.query("SELECT * FROM segments")
    assert len(rows) == 1 and rows[0]["started_at"].startswith("2026-10-30T09:00:00")
    assert r.vad.resets == 0                  # the live VAD state was never reset by the recovery


# ----------------------------------------------------------------------------------------------
# service: device refresh before restart
# ----------------------------------------------------------------------------------------------
def test_recorder_loop_refreshes_devices_before_each_restart(scfg, monkeypatch):
    db = DB(scfg.db_path)
    refreshed: list[int] = []
    runs: list[int] = []

    class FakeRecorder:
        def __init__(self, cfg, db, stop_event):
            self.stop_event = stop_event

        def run(self):
            runs.append(1)
            if len(runs) == 1:
                raise RuntimeError("input stream stalled")
            self.stop_event.set()

    import audio_note.audio.recorder as recorder_mod

    monkeypatch.setattr(recorder_mod, "Recorder", FakeRecorder)
    monkeypatch.setattr(capture, "refresh_devices", lambda: refreshed.append(1) or True)
    monkeypatch.setattr(service, "RECORDER_RESTART_DELAY_S", 0.01)
    stop = threading.Event()
    ref: dict = {}
    service._recorder_loop(scfg, db, stop, ref)
    assert len(runs) == 2
    assert len(refreshed) == 1                 # not before the first start, once before the restart
    assert "stalled" in ref["last_error"]


# ----------------------------------------------------------------------------------------------
# scheduler: transient failure re-queue
# ----------------------------------------------------------------------------------------------
def test_requeue_transient_failures_is_bounded_per_segment(scfg):
    db = DB(scfg.db_path)
    t = datetime(2026, 10, 30, 9, 0)
    transient, _ = _seg(scfg, db, t, "failed", error="LLMError: asr: giving up after 4 attempts: Connection error.", with_file=False)
    permanent, _ = _seg(scfg, db, t + timedelta(minutes=1), "failed", error="FileNotFoundError: segment file missing", with_file=False)
    assert sch.requeue_transient_failures(db, max_attempts=2) == 1
    assert db.get_segment(transient)["status"] == "new"
    assert db.get_segment(permanent)["status"] == "failed"
    assert db.get_kv(sch.KV_FAILED_ATTEMPTS) == {str(transient): 1}
    # it fails again the same way -> second (last) retry
    db.set_segment_status(transient, "failed", "LLMError: asr: HTTP 503 overloaded")
    assert sch.requeue_transient_failures(db, max_attempts=2) == 1
    db.set_segment_status(transient, "failed", "LLMError: asr: HTTP 503 overloaded")
    assert sch.requeue_transient_failures(db, max_attempts=2) == 0   # exhausted: left for --retry-failed
    assert db.get_segment(transient)["status"] == "failed"
    assert db.get_kv(sch.KV_FAILED_ATTEMPTS) == {str(transient): 2}


def test_requeue_counter_forgets_finished_segments(scfg):
    db = DB(scfg.db_path)
    sid, _ = _seg(scfg, db, datetime(2026, 10, 30, 9, 0), "failed", error="APITimeoutError: Request timed out.", with_file=False)
    sch.requeue_transient_failures(db)
    assert db.get_kv(sch.KV_FAILED_ATTEMPTS) == {str(sid): 1}
    db.set_segment_status(sid, "done")
    sch.requeue_transient_failures(db)
    assert db.get_kv(sch.KV_FAILED_ATTEMPTS) == {}


def test_scheduler_requeues_at_most_hourly_right_before_process(scfg):
    db = DB(scfg.db_path)
    calls: list = []
    s = sch.Scheduler(scfg, db, threading.Event(), jobs=_fake_jobs(calls))
    sid, _ = _seg(scfg, db, datetime(2026, 10, 31, 9, 0), "failed", error="LLMError: asr: giving up after 4 attempts", with_file=False)
    now = datetime(2026, 10, 31, 10, 0)
    assert "process" in s.run_once(now)
    assert db.get_segment(sid)["status"] == "new"              # re-queued before the process job ran
    db.set_segment_status(sid, "failed", "LLMError: asr: giving up after 4 attempts")
    assert "process" in s.run_once(now + timedelta(minutes=10))
    assert db.get_segment(sid)["status"] == "failed"           # within the hour: not again
    assert "process" in s.run_once(now + timedelta(minutes=61))
    assert db.get_segment(sid)["status"] == "new"
    assert db.get_kv(sch.KV_FAILED_ATTEMPTS) == {str(sid): 2}


# ----------------------------------------------------------------------------------------------
# scheduler: multi-day catch-up
# ----------------------------------------------------------------------------------------------
def test_daily_days_due_catches_up_several_missed_days():
    before = datetime(2026, 10, 31, 10, 0)
    after = datetime(2026, 10, 31, 22, 30)
    # default (1 day) keeps the old behaviour
    assert sch.daily_days_due(before, "22:30", "2026-10-26") == ["2026-10-30"]
    assert sch.daily_days_due(before, "22:30", "2026-10-26", max_catchup_days=7) == [
        "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30"]
    assert sch.daily_days_due(after, "22:30", "2026-10-26", max_catchup_days=7)[-1] == "2026-10-31"
    # capped to the most recent N days
    assert sch.daily_days_due(before, "22:30", "2026-10-01", max_catchup_days=2) == ["2026-10-29", "2026-10-30"]
    # no previous report / garbage value: yesterday only
    assert sch.daily_days_due(before, "22:30", None, max_catchup_days=7) == ["2026-10-30"]
    assert sch.daily_days_due(before, "22:30", "2026-0a-0b", max_catchup_days=7) == ["2026-10-30"]
    assert sch.daily_days_due(before, "22:30", "2026-10-30", max_catchup_days=7) == []


def test_scheduler_reports_every_missed_day_with_utterances(scfg):
    db = DB(scfg.db_path)
    calls: list = []
    s = sch.Scheduler(scfg, db, threading.Event(), jobs=_fake_jobs(calls))
    db.set_kv(sch.KV_DAILY_DAY, "2026-10-26")       # the service was down for four days
    for day in ("2026-10-28", "2026-10-29"):
        seg = db.add_segment(scfg.audio_dir / f"{day}.opus", datetime.fromisoformat(f"{day}T09:00"), 10.0)
        db.set_segment_status(seg, "done")
        _utt(db, seg, day, "09:00")
    ran = s.run_once(datetime(2026, 10, 31, 10, 0))
    assert ran.count("daily") == 2
    assert [c for c in calls if c[0] == "daily"] == [("daily", "2026-10-28", True), ("daily", "2026-10-29", True)]
    assert db.get_kv(sch.KV_DAILY_DAY) == "2026-10-30"       # empty days were marked done without a report


# ----------------------------------------------------------------------------------------------
# scheduler: midnight grace, partial digests, push retry, duplicate push
# ----------------------------------------------------------------------------------------------
def test_recheck_of_yesterday_waits_for_the_midnight_grace_period(scfg):
    db = DB(scfg.db_path)
    calls: list = []
    s = sch.Scheduler(scfg, db, threading.Event(), jobs=_fake_jobs(calls))
    db.set_kv(sch.KV_DAILY_DAY, "2026-10-30")
    db.set_kv(sch.KV_DAILY_RECHECK, "2026-10-29")
    db.set_kv(sch.KV_PROCESS, "2026-10-31T00:04:00")
    db.set_kv(sch.KV_CLEANUP, "2026-10-31T00:04:00")
    seg = db.add_segment(scfg.audio_dir / "y.opus", datetime(2026, 10, 30, 23, 50), 10.0)
    db.set_segment_status(seg, "done")
    _utt(db, seg, "2026-10-30", "23:50")
    _utt(db, seg, "2026-10-30", "23:51")
    db.execute("INSERT INTO daily_summaries(day, summary_md, utterance_count) VALUES (?,?,?)", ("2026-10-30", "x", 1))
    assert s._recheck_grace_s() == pytest.approx(300 + 20 + 600 + 60)
    assert s.run_once(datetime(2026, 10, 31, 0, 5)) == []          # a segment open at 00:00 may still be coming
    assert db.get_kv(sch.KV_DAILY_RECHECK) == "2026-10-29"
    ran = s.run_once(datetime(2026, 10, 31, 0, 20))
    assert "daily" in ran and calls[-1] == ("daily", "2026-10-30", True)
    assert db.get_kv(sch.KV_DAILY_RECHECK) == "2026-10-30"


def _today_with_utterance(scfg, db, day="2026-10-31"):
    seg = db.add_segment(scfg.audio_dir / f"{day}.opus", datetime.fromisoformat(f"{day}T09:00"), 10.0)
    db.set_segment_status(seg, "done")
    _utt(db, seg, day, "09:00")
    return seg


def test_partial_digest_is_not_marked_done_and_is_regenerated(scfg):
    db = DB(scfg.db_path)
    _today_with_utterance(scfg, db)
    outcomes = [("md", {"partial": True, "failed_blocks": [{"index": 2}]}), ("md", {"partial": True}), ("md", {})]
    daily_calls: list = []

    def daily(day, push):
        daily_calls.append(day)
        return outcomes.pop(0)

    jobs = _fake_jobs([])
    jobs["daily"] = daily
    s = sch.Scheduler(scfg, db, threading.Event(), jobs=jobs)
    evening = datetime(2026, 10, 31, 22, 30)
    ran = s.run_once(evening)
    assert "daily" not in ran and daily_calls == ["2026-10-31"]
    assert db.get_kv(sch.KV_DAILY_DAY) == "2026-10-30"             # today is NOT done
    assert db.get_kv(sch.KV_DAILY_PARTIAL) == {"2026-10-31": 1}
    assert s.run_once(evening + timedelta(minutes=1)) == []         # backed off
    s.run_once(evening + timedelta(minutes=31))
    assert daily_calls == ["2026-10-31"] * 2 and db.get_kv(sch.KV_DAILY_DAY) == "2026-10-30"
    ran = s.run_once(evening + timedelta(minutes=62))
    assert "daily" in ran and daily_calls == ["2026-10-31"] * 3
    assert db.get_kv(sch.KV_DAILY_DAY) == "2026-10-31"             # complete digest: done


def test_partial_digest_is_accepted_after_the_retry_budget(scfg, monkeypatch):
    db = DB(scfg.db_path)
    _today_with_utterance(scfg, db)
    monkeypatch.setattr(sch, "MAX_PARTIAL_RETRIES", 2)
    jobs = _fake_jobs([])
    jobs["daily"] = lambda day, push: ("md", {"partial": True})
    s = sch.Scheduler(scfg, db, threading.Event(), jobs=jobs)
    evening = datetime(2026, 10, 31, 22, 30)
    assert "daily" not in s.run_once(evening)
    assert "daily" in s.run_once(evening + timedelta(minutes=31))
    assert db.get_kv(sch.KV_DAILY_DAY) == "2026-10-31"
    assert db.get_kv(sch.KV_DAILY_PARTIAL) == {"2026-10-31": 2}


def test_note_push_result_tracks_days_whose_push_failed(scfg):
    db = DB(scfg.db_path)
    s = sch.Scheduler(scfg, db, threading.Event(), jobs=_fake_jobs([]))
    now = datetime(2026, 10, 31, 22, 31)
    assert s._note_push_result(DAY, {"toast": None, "feishu": "HTTP 500"}, now) is True
    assert db.get_kv(sch.KV_PUSH_PENDING) is None
    assert s._note_push_result(DAY, {"toast": "no toast", "feishu": "HTTP 500"}, now) is False
    assert db.get_kv(sch.KV_PUSH_PENDING) == {DAY: {"attempts": 0, "last": "2026-10-31T22:31:00"}}
    assert s._note_push_result(DAY, {}, now) is False            # no channel configured: nothing to retry
    assert db.get_kv(sch.KV_PUSH_PENDING) == {}


def test_failed_push_is_retried_later_and_cleared_on_success(scfg):
    db = DB(scfg.db_path)
    calls: list = []
    s = sch.Scheduler(scfg, db, threading.Event(), jobs=_fake_jobs(calls))
    now = datetime(2026, 10, 31, 22, 31)
    s._note_push_result(DAY, {"toast": "no network"}, now)
    assert "push" not in s.run_once(now)                            # just failed: not immediately
    assert "push" not in s.run_once(now + timedelta(minutes=10))
    ran = s.run_once(now + timedelta(minutes=31))
    assert "push" in ran and ("push", DAY) in calls
    assert db.get_kv(sch.KV_PUSH_PENDING) == {}
    assert "push" not in s.run_once(now + timedelta(minutes=70))


def test_push_retry_gives_up_after_budget(scfg, monkeypatch):
    db = DB(scfg.db_path)
    calls: list = []
    jobs = _fake_jobs(calls)
    jobs["push"] = lambda day: calls.append(("push", day)) or {"toast": "still down"}
    monkeypatch.setattr(sch, "MAX_PUSH_RETRIES", 2)
    s = sch.Scheduler(scfg, db, threading.Event(), jobs=jobs)
    now = datetime(2026, 10, 31, 22, 31)
    s._note_push_result(DAY, {"toast": "no network"}, now)
    s.run_once(now + timedelta(minutes=31))
    assert db.get_kv(sch.KV_PUSH_PENDING)[DAY]["attempts"] == 1
    s.run_once(now + timedelta(minutes=62))
    assert db.get_kv(sch.KV_PUSH_PENDING)[DAY]["attempts"] == 2
    s.run_once(now + timedelta(minutes=93))
    assert db.get_kv(sch.KV_PUSH_PENDING) == {}                     # dropped, no further attempt
    assert [c for c in calls if c[0] == "push"] == [("push", DAY)] * 2


def _patch_analyzer(monkeypatch, *, cached: bool, partial: bool):
    pushed: list[str] = []

    class FakeAnalyzer:
        def __init__(self, cfg, db, client=None):
            self.last_run_cached = False
            self.last_run_partial = False

        def run(self, day):
            self.last_run_cached, self.last_run_partial = cached, partial
            return "md", {"partial": partial}

        def push(self, day):
            pushed.append(day)
            return {"toast": None}

    import audio_note.analysis.daily as daily_mod

    monkeypatch.setattr(daily_mod, "DailyAnalyzer", FakeAnalyzer)
    return pushed


def test_default_daily_does_not_push_an_unchanged_already_pushed_digest(scfg, monkeypatch):
    db = DB(scfg.db_path)
    s = sch.Scheduler(scfg, db, threading.Event())
    monkeypatch.setattr(s, "_client", lambda: object())
    db.execute("INSERT INTO daily_summaries(day, summary_md, utterance_count, pushed_at) VALUES (?,?,?,?)",
               (DAY, "x", 1, "2026-10-30T22:31:00"))
    pushed = _patch_analyzer(monkeypatch, cached=True, partial=False)
    s._default_daily(DAY, True)
    assert pushed == []                                             # cached + pushed_at set: silence
    db.execute("UPDATE daily_summaries SET pushed_at=NULL WHERE day=?", (DAY,))
    s._default_daily(DAY, True)
    assert pushed == [DAY]                                          # cached but never delivered: push


def test_default_daily_pushes_a_partial_digest_only_once(scfg, monkeypatch):
    db = DB(scfg.db_path)
    s = sch.Scheduler(scfg, db, threading.Event())
    monkeypatch.setattr(s, "_client", lambda: object())
    pushed = _patch_analyzer(monkeypatch, cached=False, partial=True)
    s._default_daily(DAY, True)
    assert pushed == [DAY]
    s._bump_partial_attempts(DAY)                                   # what _run_daily does after a partial result
    s._default_daily(DAY, True)
    assert pushed == [DAY]                                          # the second partial is not pushed again


def test_default_push_sends_the_stored_digest_without_an_api_key(scfg, monkeypatch):
    db = DB(scfg.db_path)
    db.execute("INSERT INTO daily_summaries(day, summary_md, utterance_count) VALUES (?,?,?)", (DAY, "# 日报", 3))
    sent = _fake_push_module(monkeypatch, {"toast": None})
    s = sch.Scheduler(scfg, db, threading.Event())
    assert scfg.mimo.api_key == ""
    assert s._default_push(DAY) == {"toast": None}
    assert len(sent) == 1 and sent[0].body_md == "# 日报"
    assert db.one("SELECT pushed_at FROM daily_summaries WHERE day=?", (DAY,))["pushed_at"]


# ----------------------------------------------------------------------------------------------
# daily: partial digest
# ----------------------------------------------------------------------------------------------
class FlakyClient:
    """FakeClient whose map call fails for the block whose user prompt contains `fail_marker`."""

    def __init__(self, fail_marker: str | None = "第 2/2 块"):
        self.fail_marker = fail_marker
        self.map_calls = 0

    def json(self, system: str, user: str, **kw):
        if "open_loops" in system:
            return {"overview": "总览", "topics": ["话题"], "decisions": [], "todos": [], "people": [],
                    "mood": "平静", "open_loops": [], "timeline": []}
        self.map_calls += 1
        if self.fail_marker and self.fail_marker in user:
            raise LLMError("chat: giving up after 4 attempts: HTTP 503")
        return {"title": "块", "summary": "摘要", "topics": ["话题"], "decisions": [], "todos": [], "people": [],
                "mood": "平静", "open_questions": []}


def _two_sessions(db: DB) -> int:
    seg = db.add_segment("s.opus", datetime(2026, 10, 30, 9, 0), 3600.0)
    db.set_segment_status(seg, "done")
    for hhmm in ("09:00", "09:01", "09:10"):
        _utt(db, seg, DAY, hhmm, text="上午的内容")
    for hhmm in ("14:00", "14:02"):
        _utt(db, seg, DAY, hhmm, text="下午的内容")
    return 5


def test_partial_digest_is_flagged_stored_with_reduced_count_and_regenerated(cfg, db):
    from audio_note.analysis.daily import DailyAnalyzer

    n = _two_sessions(db)
    client = FlakyClient()
    an = DailyAnalyzer(cfg, db, client)
    md, js = an.run(DAY)
    assert an.last_run_partial and not an.last_run_cached
    assert js["partial"] is True and len(js["failed_blocks"]) == 1
    assert js["failed_blocks"][0]["time_range"] == "14:00–14:02" and "giving up" in js["failed_blocks"][0]["error"]
    assert "不完整" in md and "14:00–14:02" in md
    row = db.one("SELECT utterance_count, pushed_at FROM daily_summaries WHERE day=?", (DAY,))
    assert row["utterance_count"] == 3 < n                       # only the afternoon block is missing
    assert (cfg.daily_dir / f"{DAY}.md").read_text(encoding="utf-8") == md

    # the API recovers: the next run must regenerate instead of returning the cached partial digest
    client.fail_marker = None
    md2, js2 = an.run(DAY)
    assert not an.last_run_partial and not an.last_run_cached
    assert "partial" not in js2 and "不完整" not in md2
    assert db.one("SELECT utterance_count FROM daily_summaries WHERE day=?", (DAY,))["utterance_count"] == n
    # and from now on it is final
    an.run(DAY)
    assert an.last_run_cached


def test_render_markdown_adds_partial_note_only_when_flagged():
    from audio_note.analysis.daily import render_markdown

    js = {"overview": "x", "utterance_count": 5, "partial": True,
          "failed_blocks": [{"index": 2, "time_range": "14:00–14:02"}, {"index": 3, "time_range": "16:00–16:30"}]}
    md = render_markdown(DAY, js)
    assert "2 个时段分析失败" in md and "14:00–14:02、16:00–16:30" in md
    assert "不完整" not in render_markdown(DAY, {"overview": "x", "utterance_count": 5})


# ----------------------------------------------------------------------------------------------
# logging: CLI role and separate log file
# ----------------------------------------------------------------------------------------------
def test_detect_role_distinguishes_cli_commands_from_the_service():
    from audio_note.logging_setup import detect_role

    assert detect_role([r"C:\proj\.venv\Scripts\audio-note.exe", "status"]) == "cli"
    assert detect_role([r"C:\proj\.venv\Scripts\audio-note", "chat", "昨天说了什么"]) == "cli"
    assert detect_role(["audio-note", "--verbose", "-c", "x.yaml", "process", "--retry-failed"]) == "cli"
    assert detect_role(["audio-note", "run", "--tray"]) == "service"
    assert detect_role(["audio-note", "-c", "x.yaml", "record"]) == "service"
    assert detect_role(["audio-note"]) == "cli"
    assert detect_role([r"C:\proj\audio_note\__main__.py", "daily"]) == "cli"
    assert detect_role([r"C:\proj\audio_note\__main__.py", "run"]) == "service"
    # not the audio-note entry point at all (pytest, an embedding script): the service log
    assert detect_role([r"C:\x\site-packages\pytest\__main__.py", "-q"]) == "service"
    assert detect_role(["myscript.py", "status"]) == "service"
    assert detect_role([]) == "service"


def test_setup_logging_uses_a_separate_file_for_the_cli(scfg):
    from audio_note.logging_setup import log_file_name, setup_logging

    assert setup_logging(scfg) == scfg.data_dir / "audio_note.log"             # default/service unchanged
    assert setup_logging(scfg, role="cli") == scfg.data_dir / "audio_note.cli.log"
    logging.getLogger("audio_note.test").info("hello from the cli")
    for h in logging.getLogger().handlers:
        h.flush()
    assert "hello from the cli" in (scfg.data_dir / "audio_note.cli.log").read_text(encoding="utf-8")
    scfg["log"]["file"] = "notes.txt"
    assert log_file_name(scfg, "cli") == "notes.cli.txt" and log_file_name(scfg, "service") == "notes.txt"
