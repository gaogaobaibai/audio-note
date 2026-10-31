"""Regression tests for the defects found by the review of the web UI / profiles code:
profile switch robustness, DB connection lifetime, config PUT validation, speaker upload edge cases,
profile deletion while files are open, --profile in detect_role, the Markdown renderer (via node)."""
from __future__ import annotations

import gc
import json
import logging
import shutil
import sqlite3
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import yaml
from fastapi.testclient import TestClient

from audio_note import config
from audio_note import runtime as rt_mod
from audio_note.db import DB
from audio_note.runtime import JobError, Runtime, RuntimeHolder
from audio_note.ui.app import create_app

from .conftest import MODELS_DIR, SAMPLE_WAV

STATIC = Path(__file__).resolve().parent.parent / "audio_note" / "ui" / "static"
NODE = shutil.which("node")


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("AUDIO_NOTE_CONFIG", raising=False)
    monkeypatch.delenv("AUDIO_NOTE_PROFILE", raising=False)
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(config, "CURRENT_PROFILE_FILE", tmp_path / "profiles" / "current")
    (tmp_path / "config.yaml").write_text(
        f"mimo:\n  api_key: ''\npaths:\n  data_dir: data\n  models_dir: {MODELS_DIR.as_posix()}\n"
        "memory:\n  retrieval: fts\n  embedding_model: hash\n", encoding="utf-8")
    monkeypatch.setattr(rt_mod, "_recorder_loop", lambda cfg, db, ev, ref: ev.wait())
    yield tmp_path
    from audio_note.logging_setup import _remove_own_handlers

    _remove_own_handlers(logging.getLogger())


@pytest.fixture
def holder(root: Path):
    config.create_profile("demo")
    h = RuntimeHolder(Runtime(config.load_config()))
    yield h
    h.rt.shutdown()


# --- RuntimeHolder.switch ----------------------------------------------------------------------------

def test_switch_failure_leaves_the_old_runtime_intact(holder: RuntimeHolder, root: Path, monkeypatch: pytest.MonkeyPatch):
    old = holder.rt
    old.start_service(record=False)
    # the target profile's data dir cannot be created: a file sits where the directory should be
    bad = config.create_profile("bad")
    shutil.rmtree(root / "profiles" / "bad" / "data")
    (root / "profiles" / "bad" / "data").write_text("not a dir", encoding="utf-8")
    with pytest.raises(OSError):
        holder.switch("bad")
    assert holder.rt is old and old.alive and old.service_running() and old.scheduler_running()
    assert old.status()["profile"]["name"] == "default"            # the db is still open and usable
    assert config.current_profile_name() == "default"
    with TestClient(create_app(holder), raise_server_exceptions=False) as c:
        assert c.post("/api/profiles/bad/switch").status_code == 500
        assert c.get("/api/status").status_code == 200


def test_switch_refuses_while_a_job_is_running(holder: RuntimeHolder):
    rt = holder.rt
    gate = threading.Event()
    rt._job_cleanup = lambda: gate.wait(10) or {"ok": 1}       # type: ignore[assignment]
    rt.submit("cleanup")
    time.sleep(0.1)
    with pytest.raises(JobError) as ex:
        holder.switch("demo")
    assert "正在运行" in str(ex.value) and holder.rt is rt
    with TestClient(create_app(holder), raise_server_exceptions=False) as c:
        r = c.post("/api/profiles/demo/switch")
        assert r.status_code == 409 and "正在运行" in r.json()["error"]
    gate.set()
    assert rt.wait_jobs()
    assert holder.switch("demo").profile == "demo"


def test_switch_accepts_case_variant_of_an_existing_profile(holder: RuntimeHolder):
    new = holder.switch("DEMO")
    assert new.profile == "demo" and config.current_profile_name() == "demo"


def test_stop_then_start_does_not_run_two_schedulers(holder: RuntimeHolder, monkeypatch: pytest.MonkeyPatch):
    rt = holder.rt
    release = threading.Event()
    rt.sched.jobs["process"] = lambda: release.wait(20)           # a tick that takes "minutes"
    rt.sched.tick_s = 0.01
    rt.start_service(record=False)
    time.sleep(0.2)                                               # the tick is now inside the slow job
    rt.stop_service(timeout=0.2)
    assert rt._sched_thread is not None and rt._sched_thread.is_alive()
    with pytest.raises(JobError):
        rt.start_service(record=False)                            # refused instead of a second loop
    threads = [t for t in threading.enumerate() if t.name == "scheduler"]
    assert len(threads) == 1
    release.set()
    rt._sched_thread.join(5)
    assert not rt._sched_thread.is_alive()
    rt.start_service(record=False)                                # now fine, on a fresh stop event
    assert rt.scheduler_running() and rt.sched._processor is None
    rt.stop_service()


# --- DB connections -----------------------------------------------------------------------------------

def test_connections_of_finished_threads_are_closed(cfg):
    db = DB(cfg.db_path)
    assert db.open_connections() == 1

    def use() -> None:
        db.conn().execute("SELECT 1")

    for _ in range(20):
        t = threading.Thread(target=use)
        t.start()
        t.join()
    del t
    gc.collect()
    assert db.open_connections() == 1                              # only the main thread's one survives
    assert db.close_all() == 1
    with pytest.raises(sqlite3.ProgrammingError):
        db.conn().execute("SELECT 1")


def test_db_object_release_closes_its_connection(cfg):
    """A short-lived DB (e.g. `audio-note status`) must not keep the file open: profile deletion relies on it."""
    db = DB(cfg.db_path)
    db.conn().execute("SELECT 1")
    del db
    gc.collect()
    # on Windows an open handle would make the rename fail
    target = cfg.db_path.with_name("moved.sqlite3")
    cfg.db_path.rename(target)
    target.rename(cfg.db_path)


# --- profile deletion while files are open -----------------------------------------------------------

def test_delete_profile_in_use_is_all_or_nothing(root: Path):
    config.create_profile("p")
    cfg = config.load_config(profile="p")
    db = DB(cfg.db_path)
    (cfg.audio_dir / "2026-10-30").mkdir(parents=True)
    (cfg.audio_dir / "2026-10-30" / "x.opus").write_bytes(b"0" * 10)
    try:
        with pytest.raises(OSError) as ex:
            config.delete_profile("p")
        assert "正被别的程序使用" in str(ex.value) or "无法删除" in str(ex.value)
        # nothing was removed
        assert (cfg.audio_dir / "2026-10-30" / "x.opus").exists() and config.profile_exists("p")
    finally:
        db.close_all()
    config.delete_profile("p")
    assert not (root / "profiles" / "p").exists() and not list((root / "profiles").glob(".deleting-*"))


def test_api_delete_profile_in_use_returns_409(holder: RuntimeHolder, root: Path):
    cfg = config.load_config(profile="demo")
    db = DB(cfg.db_path)
    try:
        with TestClient(create_app(holder), raise_server_exceptions=False) as c:
            r = c.request("DELETE", "/api/profiles/demo", json={"confirm": "demo"})
            assert r.status_code == 409
        assert config.profile_exists("demo")
    finally:
        db.close_all()


# --- config PUT validation -----------------------------------------------------------------------------

@pytest.fixture
def client(holder: RuntimeHolder):
    with TestClient(create_app(holder), raise_server_exceptions=False) as c:
        yield c


@pytest.mark.parametrize("body", [
    {"audio": {"vad_threshold": "abc"}},
    {"audio": {"vad_threshold": 1.5}},
    {"mimo": {"chat_model": {"x": 1}}},
    {"ui": {"port": "nope"}},
    {"ui": {"port": 80}},
    {"push": {"channels": "feishu,nope"}},
    {"push": {"channels": [1, 2]}},
    {"storage": {"keep_audio_days": -5}},
    {"mimo": {"asr_concurrency": 0}},
    {"mimo": {"asr_language": "fr"}},
    {"mimo": {"base_url": "ftp://x"}},
    {"memory": {"strategy": "magic"}},
    {"mimo": "x"},
    {"audio": {"device": True}},
    {"correction": {"enabled": "maybe"}},
    {"push": {"feishu": "notadict"}},
])
def test_config_put_rejects_bad_values(client: TestClient, holder: RuntimeHolder, body: dict, root: Path):
    before = (root / "config.yaml").read_text(encoding="utf-8")
    r = client.put("/api/config", json=body)
    assert r.status_code == 400, r.text
    assert (root / "config.yaml").read_text(encoding="utf-8") == before
    assert holder.rt.cfg.audio.vad_threshold == 0.5 and holder.rt.cfg.ui.port == 8765


def test_config_put_coerces_form_strings(client: TestClient, holder: RuntimeHolder, root: Path):
    r = client.put("/api/config", json={"audio": {"vad_threshold": "0.6", "device": "2"}, "storage": {"cleanup_enabled": "false", "keep_audio_days": "15"},
                                        "push": {"channels": "toast, ntfy"}, "mimo": {"asr_language": " zh "}})
    assert r.status_code == 200, r.text
    cfg = holder.rt.cfg
    assert cfg.audio.vad_threshold == 0.6 and cfg.audio.device == 2 and cfg.storage.cleanup_enabled is False
    assert cfg.storage.keep_audio_days == 15 and cfg.push.channels == ["toast", "ntfy"] and cfg.mimo.asr_language == "zh"
    data = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    assert data["audio"]["device"] == 2 and data["storage"]["cleanup_enabled"] is False and data["push"]["channels"] == ["toast", "ntfy"]


def test_config_put_on_a_profile_writes_only_changed_keys(holder: RuntimeHolder, root: Path):
    holder.switch("demo")
    with TestClient(create_app(holder), raise_server_exceptions=False) as c:
        g = c.get("/api/config").json()
        # a "save everything" from the settings page: every value as the page currently shows it, one changed
        body = {sec: dict(vals) for sec, vals in g["values"].items()}
        body["audio"]["vad_threshold"] = 0.7
        body["mimo"]["api_key"] = ""            # untouched secret field
        r = c.put("/api/config", json=body)
        assert r.status_code == 200, r.text
        assert r.json()["saved"] == ["audio.vad_threshold"]
    data = yaml.safe_load((root / "profiles" / "demo" / "config.yaml").read_text(encoding="utf-8"))
    assert data["audio"] == {"vad_threshold": 0.7}
    assert set(data) == {"paths", "correction", "audio"}        # nothing else got pinned into the profile
    assert "vad_threshold" not in (root / "config.yaml").read_text(encoding="utf-8")


# --- speaker upload keeps a pre-existing speaker ---------------------------------------------------

@pytest.mark.skipif(not SAMPLE_WAV.exists(), reason="real sample recording not available")
def test_short_upload_does_not_delete_an_existing_speaker(holder: RuntimeHolder, client: TestClient):
    db = holder.rt.db
    sid = db.upsert_speaker("Alice")
    seg = db.add_segment(holder.rt.cfg.audio_dir / "a.opus", datetime(2026, 10, 30, 9, 0), 5.0)
    db.add_utterances([{"segment_id": seg, "start_s": 0, "end_s": 1, "abs_start": "2026-10-30T09:00:00", "day": "2026-10-30",
                        "speaker_id": sid, "speaker_name": "Alice", "text_raw": "hi", "text": "hi"}])
    import io

    buf = io.BytesIO()
    sf.write(buf, np.zeros(8000, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")
    r = client.post("/api/speakers/Alice/samples?filename=x.wav", content=buf.getvalue())
    assert r.status_code == 400
    assert db.one("SELECT id FROM speakers WHERE name='Alice'") is not None
    u = db.one("SELECT speaker_id, speaker_name FROM utterances")
    assert u["speaker_id"] == sid and u["speaker_name"] == "Alice"
    # a brand-new name with a useless sample leaves no empty speaker row behind
    r = client.post("/api/speakers/Nobody/samples?filename=x.wav", content=buf.getvalue())
    assert r.status_code == 400 and db.one("SELECT id FROM speakers WHERE name='Nobody'") is None


# --- misc ---------------------------------------------------------------------------------------------

def test_detect_role_skips_the_profile_value():
    from audio_note.logging_setup import detect_role

    assert detect_role(["audio-note", "--profile", "me", "run"]) == "service"
    assert detect_role(["audio-note", "--profile", "me", "ui"]) == "service"
    assert detect_role(["audio-note", "--profile", "me", "status"]) == "cli"


def test_init_targets_the_root_config(root: Path, monkeypatch: pytest.MonkeyPatch):
    from audio_note import cli

    monkeypatch.setattr(cli, "PROJECT_ROOT", root)
    (root / "config.yaml").unlink()
    config.create_profile("me")
    config.set_current_profile("me")
    assert cli.main(["init"]) in (0, 1)
    assert (root / "config.yaml").exists()                      # not profiles/me/config.yaml


@pytest.mark.skipif(not NODE, reason="node not installed")
def test_markdown_renderer_never_emits_raw_source():
    """md() in app.js: list items starting with '<input' used to be emitted unescaped (XSS)."""
    src = (STATIC / "app.js").read_text(encoding="utf-8")
    start = src.index("  function md(src) {")
    end = src.index("\n  }\n", start) + 4
    fn = src[start:end]
    helpers = 'const esc = (s) => String(s ?? "").replace(/[&<>"\']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", \'"\': "&quot;", "\'": "&#39;" }[c]));\n'
    cases = ["- <input onfocus=alert(1) autofocus> hi", "1. <input x", "- [x] <b>done</b>", "<script>x</script>", "| a | <i> |\n|---|---|\n| <u> | b |"]
    script = helpers + fn + "\nconst out = " + json.dumps(cases) + ".map(md); process.stdout.write(JSON.stringify(out));"
    res = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=30, check=True)
    outs = json.loads(res.stdout)
    for case, html in zip(cases, outs):
        assert "<input onfocus" not in html and "<script>" not in html and "<b>" not in html and "<u>" not in html, (case, html)
    assert '<input type="checkbox" disabled checked>' in outs[2] and "&lt;b&gt;done&lt;/b&gt;" in outs[2]
