"""Regression tests for the review fixes that touch service.py, scheduler.py, memory/chat.py,
speech/speakers.py, cli.py and config.py (no network, no API key)."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from audio_note import scheduler as sch
from audio_note import service
from audio_note.config import DEFAULTS, PROJECT_ROOT, is_placeholder_key, load_config
from audio_note.db import DB

from .conftest import noise, silence

MODELS_DIR = PROJECT_ROOT / "models"


@pytest.fixture
def tmp_cfg(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "paths": {"data_dir": str(tmp_path / "data"), "models_dir": str(MODELS_DIR)},
        "mimo": {"api_key": ""},
        "pipeline": {"process_interval_min": 10},
        "storage": {"cleanup_interval_min": 30},
        "analysis": {"daily_time": "22:30", "push_daily": True},
    }), encoding="utf-8")
    monkeypatch.setenv("AUDIO_NOTE_CONFIG", str(cfg_path))
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    cfg = load_config(cfg_path)
    cfg.ensure_dirs()
    yield cfg
    from audio_note.logging_setup import _remove_own_handlers
    import logging
    _remove_own_handlers(logging.getLogger())


# --- service lock: recycled pid --------------------------------------------------------------

def test_lock_detects_recycled_pid(tmp_cfg, monkeypatch):
    other = 4321
    monkeypatch.setattr(service, "pid_alive", lambda pid: pid in (other, service.os.getpid()))
    monkeypatch.setattr(service, "process_start_time", lambda pid: 1000.0 if pid == other else 2000.0)
    # the pid in the lock is alive but was created at a different time -> a stranger, lock is stale
    service.lock_path(tmp_cfg).write_text(f"{other}\n2026-10-31T10:00:00\n5000.0\n", encoding="utf-8")
    assert service.running_pid(tmp_cfg) is None
    p = service.acquire_lock(tmp_cfg)
    pid, ctime = service.read_lock_info(tmp_cfg)
    assert pid == service.os.getpid() and ctime == 2000.0
    service.release_lock(tmp_cfg)
    assert not p.exists()
    # same pid with the matching creation time -> genuinely running
    service.lock_path(tmp_cfg).write_text(f"{other}\n2026-10-31T10:00:00\n1000.5\n", encoding="utf-8")
    assert service.running_pid(tmp_cfg) == other
    with pytest.raises(service.AlreadyRunning):
        service.acquire_lock(tmp_cfg)
    # old two-line lock without a creation time still counts as running (cannot tell)
    service.lock_path(tmp_cfg).write_text(f"{other}\n2026-10-31T10:00:00\n", encoding="utf-8")
    assert service.running_pid(tmp_cfg) == other


def test_process_start_time_of_self_is_recent():
    t = service.process_start_time(service.os.getpid())
    if t is None:
        pytest.skip("creation time unavailable on this platform")
    import time
    assert 0 <= time.time() - t < 24 * 3600


# --- scheduler: pending segments / late content ---------------------------------------------------

def _utt(db, seg, day, hhmm):
    db.add_utterances([{"segment_id": seg, "start_s": 0, "end_s": 1, "abs_start": f"{day}T{hhmm}:00", "day": day,
                        "text_raw": "你好", "text": "你好"}])


def test_daily_not_marked_done_while_segments_pending(tmp_cfg):
    db = DB(tmp_cfg.db_path)
    calls: list[tuple] = []
    jobs = {"process": lambda: calls.append(("process",)), "cleanup": lambda: None,
            "daily": lambda day, push: calls.append(("daily", day, push))}
    s = sch.Scheduler(tmp_cfg, db, threading.Event(), jobs=jobs)
    # yesterday has a recording that was never transcribed (no key / outage): status 'new'
    seg = db.add_segment(tmp_cfg.audio_dir / "y.opus", datetime(2026, 10, 30, 9, 0), 10.0)
    now = datetime(2026, 10, 31, 10, 0)
    assert s.run_once(now) == ["process", "cleanup"]
    assert db.get_kv(sch.KV_DAILY_DAY) is None          # not marked: utterances may still come
    # 30 min later the segment got processed -> catch-up report for yesterday runs
    _utt(db, seg, "2026-10-30", "09:00")
    db.set_segment_status(seg, "done")
    ran = s.run_once(now + timedelta(minutes=31))
    assert "daily" in ran and ("daily", "2026-10-30", True) in calls
    assert db.get_kv(sch.KV_DAILY_DAY) == "2026-10-30"


def test_daily_runs_process_first_and_rechecks_yesterday(tmp_cfg):
    db = DB(tmp_cfg.db_path)
    calls: list[tuple] = []
    jobs = {"process": lambda: calls.append(("process",)), "cleanup": lambda: None,
            "daily": lambda day, push: calls.append(("daily", day, push))}
    s = sch.Scheduler(tmp_cfg, db, threading.Event(), jobs=jobs)
    db.set_kv(sch.KV_DAILY_DAY, "2026-10-30")
    db.set_kv(sch.KV_DAILY_RECHECK, "2026-10-30")
    db.set_kv(sch.KV_PROCESS, datetime(2026, 10, 31, 22, 25).isoformat())   # process not due by itself
    db.set_kv(sch.KV_CLEANUP, datetime(2026, 10, 31, 22, 20).isoformat())
    seg = db.add_segment(tmp_cfg.audio_dir / "t.opus", datetime(2026, 10, 31, 9, 0), 10.0)
    db.set_segment_status(seg, "done")
    _utt(db, seg, "2026-10-31", "09:00")
    evening = datetime(2026, 10, 31, 22, 30)
    ran = s.run_once(evening)
    assert ran == ["process", "daily"]              # process forced right before the daily report
    assert calls[-2:] == [("process",), ("daily", "2026-10-31", True)]
    db.execute("INSERT INTO daily_summaries(day, summary_md, utterance_count) VALUES (?,?,?)", ("2026-10-31", "x", 1))

    # the user keeps talking after 22:30 and that segment is processed overnight
    late = db.add_segment(tmp_cfg.audio_dir / "late.opus", datetime(2026, 10, 31, 23, 10), 10.0)
    morning = datetime(2026, 11, 1, 8, 0)
    assert "daily" not in s.run_once(morning)       # 'late' is still new: wait for process
    db.set_segment_status(late, "done")
    _utt(db, late, "2026-10-31", "23:10")
    ran = s.run_once(morning + timedelta(minutes=1))
    assert "daily" in ran and calls[-1] == ("daily", "2026-10-31", True)   # regenerated + pushed once
    assert db.get_kv(sch.KV_DAILY_RECHECK) == "2026-10-31"
    assert "daily" not in s.run_once(morning + timedelta(minutes=2))


# --- memory chat: tool_call ids ----------------------------------------------------------------------

def test_chat_tool_calls_without_ids_pair_up(tmp_cfg):
    from audio_note.memory.chat import MemoryChat

    db = DB(tmp_cfg.db_path)

    def call(name, args):   # a gateway that omits `id`
        return SimpleNamespace(type="function", function=SimpleNamespace(name=name, arguments=json.dumps(args)))

    msgs = [SimpleNamespace(content=None, tool_calls=[call("list_days", {}), call("list_speakers", {})]),
            SimpleNamespace(content="done", tool_calls=None)]
    seen: list[dict] = []

    class FakeClient:
        def chat(self, messages, **kw):
            seen.append({"messages": [dict(m) for m in messages]})
            return SimpleNamespace(choices=[SimpleNamespace(message=msgs.pop(0), finish_reason="stop")], usage=None)

        def text(self, *a, **k): raise AssertionError
        def json(self, *a, **k): raise AssertionError
        def transcribe_bytes(self, *a, **k): raise AssertionError

    assert MemoryChat(tmp_cfg, db, FakeClient()).ask("有哪些天？") == "done"
    second = seen[1]["messages"]
    asst = [m for m in second if m["role"] == "assistant"][0]
    tool_ids = [m["tool_call_id"] for m in second if m["role"] == "tool"]
    assert [c["id"] for c in asst["tool_calls"]] == tool_ids
    assert len(set(tool_ids)) == 2


# --- speakers: label guard + sample copies ------------------------------------------------------------

def test_enroll_from_audio_rejects_short_speech_and_keeps_copies(speaker_engine, db):
    with pytest.raises(ValueError, match="语音"):
        speaker_engine.enroll_from_audio(db, "王五", silence(0.4), source="x#1.0-1.4")
    assert db.query("SELECT id FROM speakers") == [] and db.speaker_embeddings() == []
    sid = speaker_engine.enroll_from_audio(db, "王五", noise(2.0, amp=0.1, seed=9), source="seg.opus#1.0-3.0")
    copies = list((speaker_engine.cfg.speakers_dir / "王五").glob("*.wav"))
    assert len(copies) == 1 and copies[0].name.startswith("seg")
    assert db.one("SELECT source_path FROM speaker_embeddings WHERE speaker_id=?", (sid,))["source_path"] == "seg.opus#1.0-3.0"


def test_enroll_files_copies_samples_into_speakers_dir(speaker_engine, db, tmp_path):
    import soundfile as sf

    p = tmp_path / "zhangsan_1.wav"
    sf.write(p, noise(2.0, amp=0.1, seed=4), 16000)
    sid = speaker_engine.enroll(db, "张三", [p])
    row = db.one("SELECT source_path, duration_s FROM speaker_embeddings WHERE speaker_id=?", (sid,))
    copy = speaker_engine.cfg.speakers_dir / "张三" / "zhangsan_1.wav"
    assert copy.exists() and row["source_path"] == str(copy)
    p.unlink()   # original gone: the copy still allows re-embedding
    e = speaker_engine.embed_speech_only(sf.read(copy, dtype="float32")[0])[0]
    assert e.shape == (192,)


# --- cli: label --at -------------------------------------------------------------------------------------

def test_resolve_label_target_finds_segment_and_range(tmp_cfg):
    from audio_note.cli import CLIError, resolve_label_target

    db = DB(tmp_cfg.db_path)
    seg = db.add_segment(tmp_cfg.audio_dir / "a.opus", datetime(2026, 10, 31, 8, 19, 0), 60.0)
    db.add_utterances([
        {"segment_id": seg, "start_s": 10.0, "end_s": 14.0, "abs_start": "2026-10-31T08:19:10.000", "day": "2026-10-31",
         "text_raw": "a", "text": "a"},
        {"segment_id": seg, "start_s": 32.0, "end_s": 40.0, "abs_start": "2026-10-31T08:19:32.000", "day": "2026-10-31",
         "text_raw": "b", "text": "b"},
    ])
    row, a, b = resolve_label_target(db, datetime(2026, 10, 31, 8, 19, 32))
    assert row["id"] == seg and (a, b) == (32.0, 40.0)
    row, a, b = resolve_label_target(db, datetime(2026, 10, 31, 8, 19, 35), dur=3.0)   # inside the 2nd utterance
    assert (a, b) == (32.0, 35.0)
    row, a, b = resolve_label_target(db, datetime(2026, 10, 31, 8, 19, 16))             # nearest one
    assert (a, b) == (10.0, 14.0)
    with pytest.raises(CLIError):
        resolve_label_target(db, datetime(2026, 10, 31, 12, 0, 0))


def test_label_cli_argument_forms(tmp_cfg, monkeypatch):
    from audio_note.cli import build_parser

    p = build_parser()
    a = p.parse_args(["speakers", "label", "--at", "2026-10-31 08:19:32", "张三"])
    assert a.segment is None and a.name == "张三" and a.at == "2026-10-31 08:19:32"
    a = p.parse_args(["speakers", "label", "12", "张三", "--from", "3", "--to", "9"])
    assert a.segment == "12" and a.start == 3.0 and a.end == 9.0 and a.at is None
    a = p.parse_args(["speakers", "relabel", "--day", "2026-10-31"])
    assert a.day == "2026-10-31" and a.segment is None


# --- config -----------------------------------------------------------------------------------------------

def test_analysis_keys_are_the_ones_daily_reads():
    assert "lookback_days" not in DEFAULTS["analysis"]
    assert DEFAULTS["analysis"]["session_gap_min"] == 20 and DEFAULTS["analysis"]["max_block_chars"] == 6000


@pytest.mark.parametrize("key,placeholder", [("", True), ("   ", True), ("sk-填这里", True), ("<your-key>", True),
                                             ("your-api-key", True), ("sk-abc123realkey", False)])
def test_is_placeholder_key(key, placeholder):
    assert is_placeholder_key(key) is placeholder
