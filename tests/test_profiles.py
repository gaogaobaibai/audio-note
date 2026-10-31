"""Profiles (档案): config layering + resolution, profile helpers, CLI `profile` commands, the UI
switcher endpoints and RuntimeHolder.switch. Everything runs in a sandboxed project root (PROJECT_ROOT /
PROFILES_DIR / CURRENT_PROFILE_FILE monkeypatched), never against the real profiles/ or config.yaml."""
from __future__ import annotations

import io
import logging
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from rich.console import Console

from audio_note import cli, config
from audio_note import runtime as rt_mod
from audio_note.runtime import Runtime, RuntimeHolder, profile_label
from audio_note.ui.app import create_app

from .conftest import MODELS_DIR


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake project root with a root config.yaml (shared settings) and an empty profiles dir."""
    monkeypatch.delenv("AUDIO_NOTE_CONFIG", raising=False)
    monkeypatch.delenv("AUDIO_NOTE_PROFILE", raising=False)
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(config, "CURRENT_PROFILE_FILE", tmp_path / "profiles" / "current")
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "# shared\nmimo:\n  api_key: sk-shared-key   # the key\n  chat_model: mimo-v2.6-flash\n"
        f"paths:\n  data_dir: data\n  models_dir: {MODELS_DIR.as_posix()}\nmemory:\n  retrieval: fts\n  embedding_model: hash\n",
        encoding="utf-8")
    yield tmp_path
    from audio_note.logging_setup import _remove_own_handlers

    _remove_own_handlers(logging.getLogger())


@pytest.fixture
def out(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    buf = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=buf, width=300, force_terminal=False, color_system=None,
                                                 highlight=False, soft_wrap=True))
    return buf


# --- helpers ----------------------------------------------------------------------------------------

def test_profile_name_validation():
    assert config.valid_profile_name("我")
    assert config.valid_profile_name("work-2026_a")
    for bad in ("", "default", "current", "custom", "data", ".hidden", "a b", "a/b", "..", "x" * 33, "名字!"):
        assert not config.valid_profile_name(bad), bad


def test_create_list_current_delete(root: Path):
    assert config.list_profiles() == ["default"]
    assert config.current_profile_name() == "default"
    p = config.create_profile("我")
    assert p == root / "profiles" / "我" / "config.yaml" and p.exists()
    assert (root / "profiles" / "我" / "data").is_dir()
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert data == {"paths": {"data_dir": "profiles/我/data"}, "correction": {"glossary_file": "profiles/我/glossary.txt"}}
    assert (root / "profiles" / "我" / "glossary.txt").exists()
    assert config.list_profiles() == ["default", "我"]
    with pytest.raises(FileExistsError):
        config.create_profile("我")
    with pytest.raises(ValueError):
        config.create_profile("default")
    config.set_current_profile("我")
    assert config.current_profile_name() == "我"
    assert (root / "profiles" / "current").read_text(encoding="utf-8").strip() == "我"
    with pytest.raises(ValueError):
        config.set_current_profile("nope")
    # a stray folder without config.yaml is not a profile
    (root / "profiles" / "junk").mkdir()
    assert config.list_profiles() == ["default", "我"]
    config.delete_profile("我")
    assert not (root / "profiles" / "我").exists()
    assert config.current_profile_name() == "default"      # current pointed at the deleted one
    with pytest.raises(ValueError):
        config.delete_profile("default")
    with pytest.raises(FileNotFoundError):
        config.delete_profile("我")


def test_create_profile_copy_from_copies_settings_but_not_secrets_or_paths(root: Path):
    src = config.create_profile("a")
    src.write_text("mimo:\n  chat_model: mimo-v2.6-pro\n  api_key: sk-a-secret\npush:\n  channels: [ntfy]\n  feishu: {webhook: 'https://x', secret: 's'}\n"
                   "ui:\n  port: 9999\npaths:\n  data_dir: profiles/a/data\ncorrection:\n  glossary_file: profiles/a/glossary.txt\n", encoding="utf-8")
    (root / "profiles" / "a" / "glossary.txt").write_text("科沃斯\n", encoding="utf-8")
    dst = config.create_profile("b", copy_from=src)
    data = yaml.safe_load(dst.read_text(encoding="utf-8"))
    assert data["paths"] == {"data_dir": "profiles/b/data"}
    assert data["correction"] == {"glossary_file": "profiles/b/glossary.txt"}
    assert data["mimo"] == {"chat_model": "mimo-v2.6-pro"}                 # api_key stays in the root file
    assert data["push"] == {"channels": ["ntfy"], "feishu": {}} and "ui" not in data
    assert (root / "profiles" / "b" / "glossary.txt").read_text(encoding="utf-8") == "科沃斯\n"
    # the shared key still reaches the copy through layering
    assert config.load_config(profile="b").mimo.api_key == "sk-shared-key"
    # Windows device names and case variants of an existing name are refused
    with pytest.raises(ValueError):
        config.create_profile("nul")
    with pytest.raises(FileExistsError):
        config.create_profile("A")
    assert config.canonical_profile_name("A") == "a" and config.canonical_profile_name("zzz") is None
    config.set_current_profile("A")
    assert config.current_profile_name() == "a"


def test_current_profile_falls_back_when_file_points_at_a_missing_profile(root: Path):
    (root / "profiles").mkdir()
    (root / "profiles" / "current").write_text("gone\n", encoding="utf-8")
    assert config.current_profile_name() == "default"
    assert config.resolve_config()[0] == "default"


# --- resolution / layering --------------------------------------------------------------------------

def test_resolve_config_precedence(root: Path, monkeypatch: pytest.MonkeyPatch):
    config.create_profile("p1")
    config.create_profile("p2")
    assert config.resolve_config() == ("default", root / "config.yaml", None)
    config.set_current_profile("p1")
    assert config.resolve_config() == ("p1", root / "profiles" / "p1" / "config.yaml", root / "config.yaml")
    assert config.resolve_config(profile="p2")[0] == "p2"
    monkeypatch.setenv("AUDIO_NOTE_PROFILE", "p2")
    assert config.resolve_config()[0] == "p2"
    monkeypatch.setenv("AUDIO_NOTE_CONFIG", str(root / "x.yaml"))
    assert config.resolve_config() == (None, root / "x.yaml", None)          # explicit file wins over profiles
    assert config.resolve_config(profile="p1")[0] == "p1"                      # ...unless a profile is asked for
    assert config.resolve_config(explicit=root / "y.yaml") == (None, root / "y.yaml", None)
    with pytest.raises(FileNotFoundError):
        config.resolve_config(profile="missing")


def test_load_config_layers_profile_over_root(root: Path):
    config.create_profile("me")
    (root / "profiles" / "me" / "config.yaml").write_text(
        "paths:\n  data_dir: profiles/me/data\nmimo:\n  chat_model: mimo-v2.6-pro\n", encoding="utf-8")
    cfg = config.load_config(profile="me")
    assert cfg.mimo.api_key == "sk-shared-key"                 # inherited from the root file
    assert cfg.mimo.chat_model == "mimo-v2.6-pro"             # overridden by the profile
    assert cfg.memory.retrieval == "fts"                       # inherited
    assert cfg.data_dir == root / "profiles" / "me" / "data"   # relative to the project root
    assert cfg["_profile"] == "me" and cfg["_base_config_path"] == str(root / "config.yaml")
    assert cfg["_config_path"] == str(root / "profiles" / "me" / "config.yaml")
    # the default profile is the root file alone
    d = config.load_config(profile="default")
    assert d["_profile"] == "default" and d["_base_config_path"] is None and d.data_dir == root / "data"
    # an explicit path never layers (tests, --config) unless base is given
    e = config.load_config(root / "profiles" / "me" / "config.yaml")
    assert e["_profile"] is None and e.mimo.api_key == "" and e.mimo.chat_model == "mimo-v2.6-pro"
    f = config.load_config(root / "profiles" / "me" / "config.yaml", base=root / "config.yaml")
    assert f.mimo.api_key == "sk-shared-key"


def test_runtime_reload_keeps_layering_and_profile(root: Path):
    config.create_profile("me")
    rt = Runtime(config.load_config(profile="me"))
    try:
        assert rt.profile == "me" and rt.cfg.mimo.api_key == "sk-shared-key"
        text = (root / "config.yaml").read_text(encoding="utf-8").replace("mimo-v2.6-flash", "mimo-v2.6-pro")
        (root / "config.yaml").write_text(text, encoding="utf-8")
        info = rt.reload_config()
        assert "mimo" in info["changed"]
        assert rt.cfg.mimo.chat_model == "mimo-v2.6-pro" and rt.cfg.mimo.api_key == "sk-shared-key"
        assert rt.profile == "me" and rt.config_path == root / "profiles" / "me" / "config.yaml"
        st = rt.status()
        assert st["profile"] == {"name": "me", "label": "me", "base_config_path": str(root / "config.yaml"),
                                 "profiles": ["default", "me"]}
    finally:
        rt.shutdown()


def test_profile_label():
    assert profile_label(None) == "自定义配置" and profile_label("default") == "默认" and profile_label("x") == "x"


# --- CLI --------------------------------------------------------------------------------------------

def test_cli_profile_commands(root: Path, out: io.StringIO):
    assert cli.main(["profile", "list"]) == 0
    assert "默认" in out.getvalue() and "● = 当前档案" in out.getvalue()
    assert cli.main(["profile", "create", "家里", "--use"]) == 0
    assert config.current_profile_name() == "家里"
    assert cli.main(["profile", "create", "家里"]) == 2            # exists
    assert cli.main(["profile", "create", "bad name"]) == 2
    out.truncate(0); out.seek(0)
    assert cli.main(["status"]) == 0
    assert "家里" in out.getvalue() and str(root / "profiles" / "家里" / "data") in out.getvalue()
    out.truncate(0); out.seek(0)
    assert cli.main(["--profile", "default", "status"]) == 0
    assert "默认" in out.getvalue()
    assert cli.main(["--profile", "missing", "status"]) == 2
    assert cli.main(["profile", "use", "default"]) == 0 and config.current_profile_name() == "default"
    assert cli.main(["profile", "use", "missing"]) == 2
    assert cli.main(["profile", "delete", "default", "-y"]) == 2
    assert cli.main(["profile", "delete", "家里", "-y"]) == 0
    assert config.list_profiles() == ["default"]
    assert cli.main(["profile", "delete", "家里", "-y"]) == 2


def test_cli_profile_delete_refuses_running_and_asks_for_the_name(root: Path, out: io.StringIO, monkeypatch: pytest.MonkeyPatch):
    config.create_profile("p")
    monkeypatch.setattr("audio_note.service.running_pid", lambda cfg: 4242)
    assert cli.main(["profile", "delete", "p", "-y"]) == 2 and "pid 4242" in out.getvalue()
    monkeypatch.setattr("audio_note.service.running_pid", lambda cfg: None)
    monkeypatch.setattr(cli.console, "input", lambda prompt="": "wrong")
    assert cli.main(["profile", "delete", "p"]) == 1 and (root / "profiles" / "p").exists()
    monkeypatch.setattr(cli.console, "input", lambda prompt="": "p")
    assert cli.main(["profile", "delete", "p"]) == 0 and not (root / "profiles" / "p").exists()


def test_cli_profile_list_shows_counts(root: Path, out: io.StringIO):
    config.create_profile("p")
    cfg = config.load_config(profile="p")
    from audio_note.db import DB

    db = DB(cfg.db_path)
    seg = db.add_segment(cfg.audio_dir / "x.opus", datetime(2026, 10, 30, 9, 0), 5.0)
    db.add_utterances([{"segment_id": seg, "start_s": 0, "end_s": 1, "abs_start": "2026-10-30T09:00:00", "day": "2026-10-30",
                        "text_raw": "a", "text": "a"}])
    db.close_all()
    assert cli.main(["profile", "list"]) == 0
    assert "1 条 / 1 天" in out.getvalue()
    assert cli._profile_counts(config.load_config(profile="default")) is None   # no database yet


# --- RuntimeHolder / API ----------------------------------------------------------------------------

def _fake_recorder_loop(cfg, db, stop_event, ref):
    ref["recorder"] = SimpleNamespace(is_speaking=False, segments_written=0, speech_total_s=0.0, started_at=datetime.now())
    stop_event.wait()


@pytest.fixture
def holder(root: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rt_mod, "_recorder_loop", _fake_recorder_loop)
    config.create_profile("demo")
    rt = Runtime(config.load_config())        # default profile
    rt.ui_url = "http://127.0.0.1:1"
    h = RuntimeHolder(rt)
    yield h
    h.rt.shutdown()


def test_holder_switch_replaces_runtime_and_closes_the_old_db(holder: RuntimeHolder, root: Path):
    old = holder.rt
    old.start_service(record=True)
    assert old.recording()
    new = holder.switch("demo")
    assert new is holder.rt and new is not old
    assert new.profile == "demo" and new.cfg.data_dir == root / "profiles" / "demo" / "data"
    assert new.ui_url == "http://127.0.0.1:1"
    assert config.current_profile_name() == "demo"
    assert new.recording() and new.service_running()          # recording resumed on the new profile
    assert not old.recording() and not old.service_running() and not old._job_thread.is_alive()
    with pytest.raises(sqlite3.ProgrammingError):
        old.db.conn().execute("SELECT 1")                       # old connections are closed
    assert holder.switch("demo") is new                         # same profile -> no-op
    with pytest.raises(FileNotFoundError):
        holder.switch("missing")
    # the lock moved with the runtime: the old data dir is free, the new one is held by us
    from audio_note.service import running_pid

    assert running_pid(old.cfg) is None and running_pid(new.cfg) == os.getpid()


def test_api_profiles_endpoints(holder: RuntimeHolder, root: Path):
    with TestClient(create_app(holder), raise_server_exceptions=False) as c:
        j = c.get("/api/profiles").json()
        assert j["current"] == "default" and [p["name"] for p in j["profiles"]] == ["default", "demo"]
        assert j["profiles"][0]["current"] and j["profiles"][0]["default"] and j["profiles"][1]["stats"] is None
        assert c.get("/api/status").json()["profile"]["name"] == "default"
        # create (+ switch), then the live runtime is the new one
        r = c.post("/api/profiles", json={"name": "我", "switch": True, "copy_from": "default"})
        assert r.status_code == 200 and r.json()["switched"]
        assert holder.rt.profile == "我" and c.get("/api/status").json()["profile"]["name"] == "我"
        copied = yaml.safe_load((root / "profiles" / "我" / "config.yaml").read_text(encoding="utf-8"))
        assert "api_key" not in copied.get("mimo", {}) and holder.rt.cfg.mimo.api_key == "sk-shared-key"
        assert c.post("/api/profiles", json={"name": "我"}).status_code == 409
        assert c.post("/api/profiles", json={"name": "bad name"}).status_code == 400
        assert c.post("/api/profiles", json={"name": "x", "copy_from": "nope"}).status_code == 404
        assert c.post("/api/profiles/missing/switch").status_code == 404
        # settings go to the profile file, not the shared one
        assert c.put("/api/config", json={"mimo": {"chat_model": "mimo-v2.6-pro"}}).status_code == 200
        assert "mimo-v2.6-pro" in (root / "profiles" / "我" / "config.yaml").read_text(encoding="utf-8")
        assert "mimo-v2.6-pro" not in (root / "config.yaml").read_text(encoding="utf-8")
        g = c.get("/api/config").json()
        assert g["profile"] == "我" and g["base_config_path"] == str(root / "config.yaml") and g["secrets"]["mimo.api_key"]["set"]
        # delete rules
        assert c.request("DELETE", "/api/profiles/我", json={"confirm": "我"}).status_code == 409      # current
        assert c.request("DELETE", "/api/profiles/default", json={"confirm": "default"}).status_code == 400
        assert c.request("DELETE", "/api/profiles/nope", json={"confirm": "nope"}).status_code == 404
        assert c.post("/api/profiles/demo/switch").status_code == 200 and holder.rt.profile == "demo"
        assert c.request("DELETE", "/api/profiles/我", json={"confirm": "wrong"}).status_code == 400
        assert c.request("DELETE", "/api/profiles/我", json={"confirm": "我"}).status_code == 200
        assert not (root / "profiles" / "我").exists()
        assert [p["name"] for p in c.get("/api/profiles").json()["profiles"]] == ["default", "demo"]


def test_api_delete_refuses_profile_running_elsewhere(holder: RuntimeHolder, root: Path, monkeypatch: pytest.MonkeyPatch):
    import audio_note.service as service

    monkeypatch.setattr(service, "running_pid", lambda cfg: 777)
    with TestClient(create_app(holder), raise_server_exceptions=False) as c:
        r = c.request("DELETE", "/api/profiles/demo", json={"confirm": "demo"})
        assert r.status_code == 409 and "777" in r.json()["error"]
        assert c.get("/api/profiles").json()["profiles"][1]["running_pid"] == 777


def test_api_switch_is_serialised(holder: RuntimeHolder):
    """Two concurrent switch requests never leave two runtimes holding locks."""
    config.create_profile("other")
    app = create_app(holder)
    results: list[int] = []

    def go(name: str) -> None:
        with TestClient(app, raise_server_exceptions=False) as c:
            results.append(c.post(f"/api/profiles/{name}/switch").status_code)

    ts = [threading.Thread(target=go, args=(n,)) for n in ("demo", "other", "demo")]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    assert results == [200, 200, 200]
    assert holder.rt.profile in ("demo", "other") and holder.rt.service_running()
    assert config.current_profile_name() == holder.rt.profile
