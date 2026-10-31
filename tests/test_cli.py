"""Tests for cli.py / scheduler.py / service.py / logging_setup.py (no network, no API key)."""
from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import yaml

from audio_note import scheduler as sch
from audio_note.config import PROJECT_ROOT, load_config
from audio_note.db import DB

MODELS_DIR = PROJECT_ROOT / "models"


@pytest.fixture
def tmp_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """config.yaml in tmp whose data_dir is a fresh temp directory; exported via AUDIO_NOTE_CONFIG."""
    data = tmp_path / "data"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "paths": {"data_dir": str(data), "models_dir": str(MODELS_DIR)},
        "mimo": {"api_key": ""},
        "pipeline": {"process_interval_min": 10},
        "storage": {"cleanup_interval_min": 30},
        "analysis": {"daily_time": "22:30", "push_daily": True},
    }), encoding="utf-8")
    monkeypatch.setenv("AUDIO_NOTE_CONFIG", str(cfg_path))
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    cfg = load_config(cfg_path)
    yield cfg
    # close file log handlers so tmp dirs can be removed on Windows
    from audio_note.logging_setup import _remove_own_handlers
    import logging
    _remove_own_handlers(logging.getLogger())


# --- scheduler pure functions ------------------------------------------------

def test_parse_hhmm():
    assert sch.parse_hhmm("22:30") == (22, 30)
    assert sch.parse_hhmm(" 7:05 ") == (7, 5)
    with pytest.raises(ValueError):
        sch.parse_hhmm("2230")
    with pytest.raises(ValueError):
        sch.parse_hhmm("25:00")


def test_next_daily_due():
    now = datetime(2026, 10, 31, 10, 0, 0)
    assert sch.next_daily_due(now, "22:30") == datetime(2026, 10, 31, 22, 30)
    assert sch.next_daily_due(datetime(2026, 10, 31, 22, 30), "22:30") == datetime(2026, 11, 1, 22, 30)
    assert sch.next_daily_due(datetime(2026, 10, 31, 23, 0), "22:30") == datetime(2026, 11, 1, 22, 30)


def test_is_due():
    now = datetime(2026, 10, 31, 10, 0, 0)
    assert sch.is_due(None, 10, now)
    assert sch.is_due("", 10, now)
    assert sch.is_due("garbage", 10, now)
    assert not sch.is_due(now - timedelta(minutes=9), 10, now)
    assert sch.is_due(now - timedelta(minutes=10), 10, now)
    assert sch.is_due((now - timedelta(minutes=11)).isoformat(), 10, now)
    assert sch.is_due(now + timedelta(hours=1), 10, now)       # clock went backwards
    assert not sch.is_due(None, 0, now)                         # disabled


def test_daily_days_due():
    before = datetime(2026, 10, 31, 10, 0)
    after = datetime(2026, 10, 31, 22, 31)
    assert sch.daily_days_due(before, "22:30", "2026-10-30") == []
    assert sch.daily_days_due(after, "22:30", "2026-10-30") == ["2026-10-31"]
    assert sch.daily_days_due(after, "22:30", "2026-10-31") == []
    # catch-up: last report older than yesterday -> yesterday first, then today if past the time
    assert sch.daily_days_due(before, "22:30", "2026-10-28") == ["2026-10-30"]
    assert sch.daily_days_due(after, "22:30", "2026-10-28") == ["2026-10-30", "2026-10-31"]
    assert sch.daily_days_due(before, "22:30", None) == ["2026-10-30"]


# --- Scheduler with fake jobs ------------------------------------------------

def test_scheduler_runs_jobs_with_fakes(tmp_cfg):
    db = DB(tmp_cfg.db_path)
    calls: list[tuple] = []
    jobs = {
        "process": lambda: calls.append(("process",)),
        "cleanup": lambda: calls.append(("cleanup",)),
        "daily": lambda day, push: calls.append(("daily", day, push)),
    }
    s = sch.Scheduler(tmp_cfg, db, threading.Event(), jobs=jobs, tick_s=0.01)
    now = datetime(2026, 10, 31, 10, 0)
    ran = s.run_once(now)
    assert ran == ["process", "cleanup"]           # no utterances -> daily catch-up is skipped silently
    assert db.get_kv(sch.KV_PROCESS) == now.isoformat(timespec="seconds")
    assert db.get_kv(sch.KV_DAILY_DAY) == "2026-10-30"
    # not due again 1 minute later
    assert s.run_once(now + timedelta(minutes=1)) == []
    # process due again after the interval
    assert s.run_once(now + timedelta(minutes=10)) == ["process"]

    # utterances for today -> daily runs at 22:30 with push flag from config
    seg = db.add_segment(tmp_cfg.audio_dir / "x.opus", now, 10.0)
    db.add_utterances([{"segment_id": seg, "start_s": 0, "end_s": 1, "abs_start": now.isoformat(), "day": "2026-10-31",
                        "text_raw": "你好", "text": "你好"}])
    evening = datetime(2026, 10, 31, 22, 30)
    ran = s.run_once(evening)
    assert "daily" in ran
    assert ("daily", "2026-10-31", True) in calls
    assert db.get_kv(sch.KV_DAILY_DAY) == "2026-10-31"
    assert "daily" not in s.run_once(evening + timedelta(minutes=1))


def test_scheduler_job_failure_is_logged_not_raised(tmp_cfg):
    db = DB(tmp_cfg.db_path)

    def boom():
        raise RuntimeError("x")

    s = sch.Scheduler(tmp_cfg, db, threading.Event(), jobs={"process": boom, "cleanup": lambda: None, "daily": lambda d, p: None})
    now = datetime(2026, 10, 31, 10, 0)
    assert "process" in s.run_once(now)
    assert s.errors["process"] == 1
    assert s.run_once(now + timedelta(minutes=1)) == []   # backed off to the interval


def test_scheduler_thread_stops(tmp_cfg):
    db = DB(tmp_cfg.db_path)
    stop = threading.Event()
    s = sch.Scheduler(tmp_cfg, db, stop, jobs={"process": lambda: None, "cleanup": lambda: None, "daily": lambda d, p: None}, tick_s=0.05)
    t = threading.Thread(target=s.run, daemon=True)
    t.start()
    stop.set()
    t.join(5)
    assert not t.is_alive()


# --- service helpers --------------------------------------------------------------

def test_pid_alive_and_lock(tmp_cfg):
    from audio_note import service

    assert service.pid_alive(os.getpid())
    assert not service.pid_alive(2 ** 22 + 12345)
    assert not service.pid_alive(0)
    assert service.read_lock(tmp_cfg) is None
    p = service.acquire_lock(tmp_cfg)
    assert p.exists() and service.read_lock(tmp_cfg) == os.getpid()
    assert service.running_pid(tmp_cfg) == os.getpid()
    service.release_lock(tmp_cfg)
    assert not p.exists()
    # stale lock from a dead pid is overwritten
    service.lock_path(tmp_cfg).write_text("4194304999\n", encoding="utf-8")
    assert service.running_pid(tmp_cfg) is None
    service.acquire_lock(tmp_cfg)
    assert service.read_lock(tmp_cfg) == os.getpid()
    service.release_lock(tmp_cfg)


def test_tray_icon_image():
    from audio_note.service import _make_icon_image

    img = _make_icon_image(32)
    assert img.size == (32, 32)


# --- logging ----------------------------------------------------------------------

def test_setup_logging_creates_file_and_no_duplicate_handlers(tmp_cfg):
    import logging
    from audio_note.logging_setup import setup_logging

    root = logging.getLogger()
    before = len(root.handlers)
    p1 = setup_logging(tmp_cfg)
    p2 = setup_logging(tmp_cfg, verbose=True)
    assert p1 == p2 == tmp_cfg.data_dir / "audio_note.log"
    assert len(root.handlers) == before + 2 or len(root.handlers) <= before + 2
    logging.getLogger("audio_note.test").info("hello 日志")
    for h in root.handlers:
        h.flush()
    assert "hello 日志" in p1.read_text(encoding="utf-8")


# --- cli ----------------------------------------------------------------------------

def test_help_exits_zero():
    from audio_note.cli import main

    with pytest.raises(SystemExit) as ex:
        main(["--help"])
    assert ex.value.code == 0


def test_speakers_help_exits_zero():
    from audio_note.cli import main

    with pytest.raises(SystemExit) as ex:
        main(["speakers", "--help"])
    assert ex.value.code == 0


def test_devices_runs(tmp_cfg):
    sd = pytest.importorskip("sounddevice")
    try:
        sd.query_devices()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no audio backend: {e}")
    from audio_note.cli import main

    assert main(["devices"]) == 0


def test_status_on_empty_data_dir(tmp_cfg):
    from audio_note.cli import main

    assert not tmp_cfg.db_path.exists()
    assert main(["status"]) == 0
    assert tmp_cfg.db_path.exists()
    # nothing was written into the repository's data dir by this test
    assert str(tmp_cfg.data_dir).startswith(str(tmp_cfg.db_path.parent))
    assert PROJECT_ROOT not in tmp_cfg.data_dir.parents


def test_init_creates_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from audio_note.cli import main

    cfg_path = tmp_path / "sub" / "config.yaml"
    monkeypatch.setenv("AUDIO_NOTE_CONFIG", str(cfg_path))
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    # the generated config uses relative paths (data/, models/) resolved against PROJECT_ROOT:
    # point that at tmp so nothing lands in the repository; models then show as missing (rc 1).
    import audio_note.config as config_mod
    from audio_note import cli as cli_mod
    monkeypatch.setattr(config_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli_mod, "PROJECT_ROOT", tmp_path)
    rc = main(["--config", str(cfg_path), "init"])
    assert (tmp_path / "data").is_dir()
    assert not (PROJECT_ROOT / "data" / "sub").exists()
    assert cfg_path.exists()
    text = cfg_path.read_text(encoding="utf-8")
    assert "mimo:" in text and "api_key" in text
    assert rc in (0, 1)  # 1 only when model files are missing on this machine
    data = yaml.safe_load(text)
    assert isinstance(data, dict) and "mimo" in data
    # running again does not overwrite
    assert main(["--config", str(cfg_path), "init"]) == rc
    from audio_note.logging_setup import _remove_own_handlers
    import logging
    _remove_own_handlers(logging.getLogger())


def test_missing_sibling_module_gives_clear_error(tmp_cfg, monkeypatch: pytest.MonkeyPatch):
    import importlib
    from audio_note import cli

    def fake_import(name: str):
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(cli.CLIError) as ex:
        cli._lazy("speech.pipeline", "Processor")
    assert "audio_note.speech.pipeline" in str(ex.value)


def test_process_without_api_key_is_a_cli_error(tmp_cfg, monkeypatch: pytest.MonkeyPatch):
    """`process` must fail fast (exit 2) on a missing key before touching the pipeline."""
    import sys
    import types
    from audio_note import cli

    fake = types.ModuleType("audio_note.speech.pipeline")
    fake.Processor = object
    monkeypatch.setitem(sys.modules, "audio_note.speech.pipeline", fake)
    assert cli.main(["process"]) == 2


def test_process_calls_processor_with_fake_client(tmp_cfg, monkeypatch: pytest.MonkeyPatch):
    import sys
    import types
    from audio_note import cli

    seen: dict = {}

    class FakeProcessor:
        def __init__(self, cfg, db, client):
            seen["client"] = client

        def run(self, retry_failed=False, limit=None):
            seen["args"] = (retry_failed, limit)
            return {"done": 0}

    fake_mod = types.ModuleType("audio_note.speech.pipeline")
    fake_mod.Processor = FakeProcessor
    monkeypatch.setitem(sys.modules, "audio_note.speech.pipeline", fake_mod)

    class FakeClient:
        usage = {"calls": 0, "audio_seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0}

        def chat(self, *a, **k): raise AssertionError("no network")
        def text(self, *a, **k): raise AssertionError("no network")
        def json(self, *a, **k): raise AssertionError("no network")
        def transcribe_bytes(self, *a, **k): raise AssertionError("no network")

    monkeypatch.setattr(cli, "_client", lambda cfg: FakeClient())
    assert cli.main(["process", "--retry-failed", "--limit", "3"]) == 0
    assert seen["args"] == (True, 3)
    assert isinstance(seen["client"], FakeClient)


def test_invoke_matches_by_name():
    from audio_note.cli import CLIError, _invoke

    def f(cfg, path, limit=None):
        return (cfg, path, limit)

    assert _invoke(f, cfg=1, path="p", client=3, file="p") == (1, "p", None)
    with pytest.raises(CLIError):
        _invoke(f, cfg=1)


def test_calibrate_threshold():
    from audio_note.cli import calibrate_threshold

    rng = np.random.default_rng(0)
    a = rng.normal(size=8).astype(np.float32)
    b = rng.normal(size=8).astype(np.float32)
    embs = [(1, "A", a), (1, "A", a + 0.05 * rng.normal(size=8).astype(np.float32)),
            (2, "B", b), (2, "B", b + 0.05 * rng.normal(size=8).astype(np.float32))]
    r = calibrate_threshold(embs)
    assert r["n_speakers"] == 2 and r["n_embeddings"] == 4
    assert r["same"]["n"] == 2 and r["cross"]["n"] == 4
    assert 0.3 <= r["suggested"] <= 0.7
    # only one speaker -> no cross distribution, still a suggestion
    r1 = calibrate_threshold(embs[:2])
    assert r1["cross"] is None and r1["suggested"] is not None


def test_speakers_calibrate_cli_needs_samples(tmp_cfg):
    from audio_note.cli import main

    assert main(["speakers", "calibrate"]) == 2   # fewer than 2 samples -> CLIError
    db = DB(tmp_cfg.db_path)
    rng = np.random.default_rng(1)
    for name in ("甲", "乙"):
        sid = db.upsert_speaker(name)
        base = rng.normal(size=16).astype(np.float32)
        for _ in range(3):
            db.add_speaker_embedding(sid, base + 0.1 * rng.normal(size=16).astype(np.float32), None, 5.0)
    assert main(["speakers", "calibrate"]) == 0
    assert main(["speakers", "list"]) == 0


def test_import_bad_file_is_cli_error(tmp_cfg):
    from audio_note.cli import main

    assert main(["import", str(tmp_cfg.data_dir / "nope.wav")]) == 2
