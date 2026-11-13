"""1.0 CLI package (audio_note/cli/): command registration, late binding to the facade, entry points and
log-role detection, `--json` output, the read-only `doctor` command.

No network, no microphone, no model loading: the model files are one-byte placeholders in tmp, the voices
module is a stub in sys.modules, every path lives under tmp_path. A few tests start a fresh interpreter
(`python -m audio_note[.cli]`, at most with a tmp config) to check what a real process does.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sqlite3
import subprocess
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from rich.console import Console

from audio_note import cli, config
from audio_note import db as db_module
from audio_note import logging_setup as ls
from audio_note.config import DEFAULTS, PROJECT_ROOT, load_config
from audio_note.db import DB

#: `audio-note --help` order of 0.3.x (the single-module cli.py); the split must reproduce it
BASELINE_COMMANDS = ["devices", "record", "run", "ui", "status", "import", "process", "transcribe", "speakers", "daily",
                     "push-test", "chat", "reindex", "cleanup", "loudness", "profile", "init"]
SPEAKER_COMMANDS = ["enroll", "record", "list", "label", "voices", "scan", "set-self", "relation", "relations", "relabel",
                    "calibrate", "audit", "reject", "negatives"]
BUILTIN_MODULES = ["service", "pipeline", "speakers", "memory", "maintenance", "profile", "system"]
#: handlers of the old module (+ doctor): all must stay importable / replaceable on `audio_note.cli`
HANDLERS = {
    "service": ["cmd_devices", "cmd_record", "cmd_run", "cmd_ui"],
    "pipeline": ["cmd_status", "cmd_import", "cmd_process", "cmd_transcribe"],
    "speakers": ["cmd_speakers_enroll", "cmd_speakers_record", "cmd_speakers_list", "cmd_speakers_label", "cmd_speakers_voices",
                 "cmd_speakers_scan", "cmd_speakers_set_self", "cmd_speakers_relation", "cmd_speakers_relations",
                 "cmd_speakers_relabel", "cmd_speakers_calibrate", "cmd_speakers_audit", "cmd_speakers_reject",
                 "cmd_speakers_negatives"],
    "memory": ["cmd_daily", "cmd_push_test", "cmd_chat", "cmd_reindex"],
    "maintenance": ["cmd_cleanup", "cmd_loudness"],
    "profile": ["cmd_profile_list", "cmd_profile_create", "cmd_profile_use", "cmd_profile_delete"],
    "system": ["cmd_init", "cmd_doctor"],
}
MODEL_KEYS = ("vad_model", "embedding_model", "segmentation_model")
FAKE_KEY = "sk-test-0123456789abcdef"      # not a real key; long enough to be recognisable in any output


# --- helpers ----------------------------------------------------------------------------------------

@pytest.fixture
def out(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Capture everything the CLI prints through the facade's console."""
    buf = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=buf, width=400, force_terminal=False, color_system=None,
                                                 highlight=False, soft_wrap=True))
    return buf


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """make(api_key="", models=True, extra=None) -> Config: a tmp config.yaml (exported through
    AUDIO_NOTE_CONFIG) whose data dir does not exist yet and whose models dir holds placeholder files."""
    def make(api_key: str = "", models: bool = True, extra: dict | None = None) -> Any:
        models_dir = tmp_path / "models"
        models_dir.mkdir(exist_ok=True)
        if models:
            for key in MODEL_KEYS:
                p = models_dir / DEFAULTS["speakers"][key]
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x")
        data = {"paths": {"data_dir": str(tmp_path / "data"), "models_dir": str(models_dir)}, "mimo": {"api_key": api_key},
                "memory": {"retrieval": "fts", "embedding_model": "hash"}}
        data.update(extra or {})
        p = tmp_path / "config.yaml"
        p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        monkeypatch.setenv("AUDIO_NOTE_CONFIG", str(p))
        monkeypatch.delenv("MIMO_API_KEY", raising=False)
        monkeypatch.delenv("AUDIO_NOTE_PROFILE", raising=False)
        return load_config(p)

    yield make
    ls._remove_own_handlers(logging.getLogger())      # close log files so tmp dirs can be removed on Windows


def doc(out: io.StringIO) -> Any:
    """The one JSON document a --json command printed."""
    return json.loads(out.getvalue())


def subparsers(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    return next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))


def seed(cfg: Any) -> dict[str, Any]:
    """Two segments (done / failed), two utterances (today + an older day), a daily, two speakers, one negative."""
    cfg.ensure_dirs()
    db = DB(cfg.db_path)
    today = datetime.now().replace(microsecond=0)
    done = db.add_segment(cfg.audio_dir / "a.opus", today, 10.0)
    db.set_segment_status(done, "done")
    failed = db.add_segment(cfg.audio_dir / "b.opus", today - timedelta(hours=1), 20.0)
    db.set_segment_status(failed, "failed", "boom")
    utts = db.add_utterances([
        {"segment_id": done, "start_s": 0, "end_s": 1, "abs_start": today.isoformat(), "day": today.strftime("%Y-%m-%d"),
         "text_raw": "今天开会", "text": "今天开会"},
        {"segment_id": done, "start_s": 1, "end_s": 2, "abs_start": "2026-10-01T10:00:00.000", "day": "2026-10-01",
         "text_raw": "以前的事", "text": "以前的事"},
    ])
    db.execute("INSERT INTO daily_summaries(day, summary_md, created_at, pushed_at) VALUES (?,?,?,?)",
               ("2026-10-01", "# x", "2026-10-01 22:30:00", None))
    zhang = db.upsert_speaker("张三", "朋友")
    li = db.upsert_speaker("李四")
    db.add_speaker_embedding(zhang, np.ones(4, dtype=np.float32), None, 7.0)
    db.add_speaker_embedding(zhang, np.ones(4, dtype=np.float32), None, 3.0)
    db.set_self_speaker(zhang)
    db.set_relation(li, "同事", "user")
    neg = db.add_speaker_negative(li, np.ones(4, dtype=np.float32), str(cfg.audio_dir / "a.opus") + "#0.0-1.0", 2.5,
                                  utterance_id=int(utts[0]), segment_id=done)
    (cfg.audio_dir / "blob.bin").write_bytes(b"\0" * 2048)
    return {"db": db, "done": done, "failed": failed, "utts": utts, "zhang": zhang, "li": li, "neg": neg, "today": today}


def run_module(module: str, *args: str) -> subprocess.CompletedProcess:
    """`python -m <module> <args>` in a fresh interpreter, from the project root."""
    child_env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-m", module, *args], cwd=str(PROJECT_ROOT), env=child_env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)


# --- package structure and registration ----------------------------------------------------------------

def test_help_order_is_the_old_order_followed_by_new_commands():
    assert cli.COMMAND_MODULES[:len(BUILTIN_MODULES)] == BUILTIN_MODULES
    sub = subparsers(cli.build_parser())
    names = list(sub.choices)
    assert names[:len(BASELINE_COMMANDS)] == BASELINE_COMMANDS            # nothing moved
    assert "doctor" in names[len(BASELINE_COMMANDS):]                     # new commands come after
    assert [a.dest for a in sub._choices_actions] == names                # the order --help prints
    speaker_subs = list(subparsers(sub.choices["speakers"]).choices)
    assert speaker_subs[:len(SPEAKER_COMMANDS)] == SPEAKER_COMMANDS          # nothing moved
    assert speaker_subs[len(SPEAKER_COMMANDS):] == ["rescore", "scoring"]    # speakers v5: appended after the old ones
    assert list(subparsers(sub.choices["profile"]).choices) == ["list", "create", "use", "delete"]
    assert sub.required and subparsers(sub.choices["speakers"]).required


def test_every_builtin_module_registers_its_handlers_and_the_facade_reexports_them():
    for name in BUILTIN_MODULES:
        mod = sys.modules[f"audio_note.cli.{name}"]
        assert callable(mod.register), name
        for fn in HANDLERS[name]:
            assert getattr(cli, fn) is getattr(mod, fn), fn
    for name in ("CLIError", "_lazy", "_lazy_module", "_client", "_invoke", "_db", "_expand_files", "_parse_datetime",
                 "_parse_day", "_human_bytes", "_print_result", "_profile_counts", "_optional_engine", "resolve_label_target",
                 "utterance_at", "calibrate_threshold", "build_parser", "main", "load_config", "PROJECT_ROOT",
                 "MINIMAL_CONFIG", "COMMAND_MODULES", "add_json_flag", "emit_json", "wants_json", "doctor_report"):
        assert hasattr(cli, name), name
    assert isinstance(cli.console, Console) and cli.log.name == "audio_note.cli"
    assert cli.PROJECT_ROOT == config.PROJECT_ROOT and cli.load_config is config.load_config


def test_every_leaf_command_dispatches_to_a_handler():
    def leaves(parser: argparse.ArgumentParser, path: tuple[str, ...] = ()):
        subs = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
        if not subs:
            yield path, parser
        for a in subs:
            for name, child in a.choices.items():
                yield from leaves(child, path + (name,))

    found = dict(leaves(cli.build_parser()))
    assert len(found) >= 34                                               # 33 commands of 0.3.x + doctor
    for path, parser in found.items():
        assert callable(parser.get_default("func")), path


def test_a_new_command_module_needs_a_file_with_register_and_one_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                     env, out: io.StringIO):
    """The documented recipe, with a real file imported on demand (found through the package path)."""
    env()
    (tmp_path / "plug").mkdir()
    (tmp_path / "plug" / "zz_extra.py").write_text(
        "from ._late import Late, handler\n"
        "from .helpers import CLIError\n"
        "console = Late('console')\n"
        "def cmd_zz(cfg, args):\n"
        "    if args.n < 0:\n"
        "        raise CLIError('n 不能是负数')\n"
        "    console.print(f'zz {args.n} {cfg.data_dir.name}')\n"
        "    return 7\n"
        "def register(sub):\n"
        "    s = sub.add_parser('zz', help='test command')\n"
        "    s.add_argument('--n', type=int, default=1)\n"
        "    s.set_defaults(func=handler(cmd_zz))\n", encoding="utf-8")
    monkeypatch.setattr(cli, "__path__", list(cli.__path__) + [str(tmp_path / "plug")])
    monkeypatch.setattr(cli, "COMMAND_MODULES", cli.COMMAND_MODULES + ["zz_extra"])
    try:
        assert "audio_note.cli.zz_extra" not in sys.modules
        assert cli.main(["zz", "--n", "3"]) == 7                          # not re-exported by the facade: still dispatched
        assert "zz 3 data" in out.getvalue()
        assert cli.main(["zz", "--n", "-1"]) == 2 and "n 不能是负数" in out.getvalue()
        assert list(subparsers(cli.build_parser()).choices)[-1] == "zz"   # appended name = last in --help
    finally:
        sys.modules.pop("audio_note.cli.zz_extra", None)
        if hasattr(cli, "zz_extra"):
            delattr(cli, "zz_extra")


def test_a_broken_command_module_is_skipped_with_a_warning(monkeypatch: pytest.MonkeyPatch, env, out: io.StringIO, caplog):
    env()
    bad = types.ModuleType("audio_note.cli.zz_bad")
    bad.register = lambda sub: 1 / 0
    monkeypatch.setitem(sys.modules, "audio_note.cli.zz_bad", bad)
    monkeypatch.setattr(cli, "COMMAND_MODULES", ["zz_missing"] + cli.COMMAND_MODULES + ["zz_bad"])
    with caplog.at_level(logging.WARNING, logger="audio_note.cli"):
        assert cli.main(["speakers", "list"]) == 0                        # the rest of the CLI works
    assert "已注册说话人" in out.getvalue()
    assert "audio_note.cli.zz_missing" in caplog.text and "audio_note.cli.zz_bad" in caplog.text
    assert "ZeroDivisionError" in caplog.text


def test_handlers_replaced_on_the_facade_win_when_the_parser_is_built(monkeypatch: pytest.MonkeyPatch, env):
    env()
    monkeypatch.setattr(cli, "cmd_status", lambda cfg, args: 5)
    monkeypatch.setattr(cli, "cmd_speakers_list", lambda cfg, args: 6)
    monkeypatch.setattr(cli, "cmd_profile_list", lambda cfg, args: 7)
    monkeypatch.setattr(cli, "cmd_doctor", lambda cfg, args: 8)
    assert [cli.main(a) for a in (["status"], ["speakers", "list"], ["profile", "list"], ["doctor"])] == [5, 6, 7, 8]


def test_command_modules_reach_patchable_names_through_the_facade(monkeypatch: pytest.MonkeyPatch, out: io.StringIO):
    from audio_note.cli import _late, helpers, maintenance, memory, pipeline, profile, service, speakers, system

    assert _late.pkg() is cli
    for mod in (helpers, maintenance, memory, pipeline, profile, service, speakers, system):
        assert isinstance(mod.console, _late.Late), mod.__name__           # never the object itself
    helpers.console.print("经过门面的输出")
    assert "经过门面的输出" in out.getvalue()
    with monkeypatch.context() as m:                                       # attribute writes land on the real console
        m.setattr(speakers.console, "input", lambda prompt="": "typed")
        m.setattr(speakers.console, "brand_new_attribute", 1, raising=False)
        assert cli.console.input() == "typed" and cli.console.brand_new_attribute == 1
    assert not hasattr(cli.console, "brand_new_attribute") and "Late('console')" in repr(speakers.console)
    sentinel = object()
    monkeypatch.setattr(cli, "_client", lambda cfg: sentinel)
    assert pipeline._client(None) is sentinel and memory._client(None) is sentinel and speakers._client(None) is sentinel
    monkeypatch.setattr(cli, "_optional_engine", lambda cfg: "engine")
    assert speakers._optional_engine(None) == "engine"
    monkeypatch.setattr(cli, "load_config", lambda *a, **k: ("cfg", a, k))
    assert profile.load_config(profile="x") == ("cfg", (), {"profile": "x"})
    assert system.load_config("p") == ("cfg", ("p",), {})
    # handler(): a handler the facade re-exports is dispatched through the facade, anything else to itself
    assert cli.EXPORTED_HANDLERS["cmd_cleanup"] is maintenance.cmd_cleanup
    assert _late.handler(maintenance.cmd_cleanup) is maintenance.cmd_cleanup
    monkeypatch.setattr(cli, "cmd_cleanup", sentinel)
    assert _late.handler(maintenance.cmd_cleanup) is sentinel

    def cmd_only_in_its_module(cfg, args):
        return 0

    def cmd_status(cfg, args):                    # same name as an exported handler, but another function
        return 0

    assert _late.handler(cmd_only_in_its_module) is cmd_only_in_its_module
    assert _late.handler(cmd_status) is cmd_status


def test_calibrate_threshold_moved_to_speech_and_is_still_exported_by_the_cli():
    from audio_note.speech import calibrate

    assert cli.calibrate_threshold is calibrate.calibrate_threshold
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    a2 = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    r = calibrate.calibrate_threshold([(1, "张三", a), (1, "张三", a2), (2, "李四", b)])
    assert r["n_embeddings"] == 3 and r["n_speakers"] == 2 and r["same"]["n"] == 1 and r["cross"]["n"] == 2
    assert not r["overlap"] and r["cross"]["max"] < r["suggested"] < r["same"]["min"]
    assert calibrate.calibrate_threshold([])["suggested"] is None


def test_import_boundaries_in_a_fresh_interpreter():
    """speech.calibrate does not pull in the CLI; importing the CLI does not import the modules that commands
    must load lazily (tests stub them through sys.modules, and start-up has to stay cheap)."""
    lazy = ["audio_note.audio.capture", "audio_note.speech.pipeline", "audio_note.speech.speakers", "audio_note.speech.voices",
            "audio_note.analysis.daily", "audio_note.push.base", "audio_note.memory.chat", "audio_note.memory.retrieval",
            "audio_note.runtime", "audio_note.service", "audio_note.logging_setup", "onnxruntime", "sounddevice"]
    code = ("import json, sys\n"
            "import audio_note.speech.calibrate\n"
            "early = 'audio_note.cli' in sys.modules\n"
            "import audio_note.cli as cli\n"
            "cli.build_parser()\n"
            f"print(json.dumps({{'early': early, 'loaded': [m for m in {lazy!r} if m in sys.modules]}}))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(PROJECT_ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=180,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1"))
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1]) == {"early": False, "loaded": []}


# --- entry points and log-role detection ------------------------------------------------------------------

@pytest.mark.parametrize("module", ["audio_note", "audio_note.cli"])
def test_module_entry_points_print_help(module: str):
    r = run_module(module, "--help")
    assert r.returncode == 0, r.stderr
    assert "usage: audio-note" in r.stdout and "doctor" in r.stdout and "speakers" in r.stdout


def test_python_m_audio_note_cli_is_a_cli_process_with_clean_json_stdout(env):
    """`python -m audio_note.cli` must be classified like the console script (own .cli.log, not the service log)
    and --json must leave nothing but the document on stdout even with DEBUG logging on."""
    cfg = env()
    r = run_module("audio_note.cli", "--config", str(cfg["_config_path"]), "-v", "--json", "status")
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)                                               # the whole stdout is one document
    assert d["data_dir"] == str(cfg.data_dir) and d["segments"]["total"] == 0 and d["api_key_configured"] is False
    assert (cfg.data_dir / "audio_note.cli.log").exists() and not (cfg.data_dir / "audio_note.log").exists()


def test_is_cli_entry_knows_both_module_entry_points():
    assert ls._is_cli_entry(r"C:\proj\audio_note\__main__.py")
    assert ls._is_cli_entry(r"C:\proj\audio_note\cli\__main__.py")
    assert ls._is_cli_entry("/srv/app/audio_note/cli/__main__.py")
    assert ls._is_cli_entry(r"C:\proj\.venv\Scripts\audio-note.exe")
    assert not ls._is_cli_entry(r"C:\proj\other\cli\__main__.py")
    assert not ls._is_cli_entry(r"C:\proj\audio_note\cli\helpers.py")
    assert not ls._is_cli_entry(r"C:\proj\audio_note\ui\__main__.py")
    assert not ls._is_cli_entry(r"C:\x\site-packages\pytest\__main__.py")


def test_detect_role_for_the_cli_package_and_new_options():
    main_py = r"C:\proj\audio_note\cli\__main__.py"
    assert ls.detect_role([main_py, "status"]) == "cli"
    assert ls.detect_role([main_py, "run", "--tray"]) == "service"
    assert ls.detect_role([main_py, "--profile", "me", "ui"]) == "service"
    assert ls.detect_role(["audio-note", "doctor"]) == "cli"
    assert ls.detect_role(["audio-note", "--json", "doctor"]) == "cli"
    assert ls.detect_role(["audio-note", "--json", "-v", "record"]) == "service"
    # value-taking options in every spelling argparse accepts: the value is never taken for the command
    assert ls.detect_role(["audio-note", "-c", "run", "status"]) == "cli"
    assert ls.detect_role(["audio-note", "--config", "status", "run"]) == "service"
    assert ls.detect_role(["audio-note", "--config=x.yaml", "run"]) == "service"
    assert ls.detect_role(["audio-note", "--conf", "x.yaml", "run"]) == "service"
    assert ls.detect_role(["audio-note", "--prof", "me", "record"]) == "service"
    assert ls.detect_role(["audio-note", "--profile=me", "status"]) == "cli"


def test_detect_role_and_the_parser_agree_on_global_options_and_service_commands():
    p = cli.build_parser()
    takes_value = {o for a in p._actions if a.option_strings and a.nargs != 0 for o in a.option_strings}
    assert takes_value == set(ls.GLOBAL_VALUE_OPTIONS)
    assert ls.SERVICE_COMMANDS <= set(subparsers(p).choices)
    a = p.parse_args(["-c", "x.yaml", "--profile", "me", "-v", "status"])
    assert a.config == "x.yaml" and a.profile == "me" and a.verbose and a.cmd == "status" and a.json is False


def test_short_config_option_selects_the_config_file(env, out: io.StringIO, monkeypatch: pytest.MonkeyPatch):
    cfg = env()
    monkeypatch.delenv("AUDIO_NOTE_CONFIG")
    assert cli.main(["-c", str(cfg["_config_path"]), "--json", "status"]) == 0
    assert doc(out)["config_path"] == str(cfg["_config_path"])


def test_setup_logging_can_log_to_stderr_and_skip_the_file(env):
    cfg = env()
    own = lambda: [h for h in logging.getLogger().handlers if getattr(h, ls._MARK, False)]   # noqa: E731
    assert ls.setup_logging(cfg, file=False, stderr=True) is None
    assert len(own()) == 1 and own()[0].console.stderr is True
    assert not cfg.data_dir.exists()                                       # nothing created
    assert ls.setup_logging(cfg) == cfg.data_dir / "audio_note.log"        # defaults unchanged: file + stdout console
    assert len(own()) == 2
    assert [h.console.stderr for h in own() if hasattr(h, "console")] == [False]


def test_main_survives_a_setup_logging_without_the_new_keywords(env, out: io.StringIO, monkeypatch: pytest.MonkeyPatch):
    cfg = env()
    calls: list[dict] = []

    def old_style(cfg, verbose=False):
        calls.append({"verbose": verbose})

    monkeypatch.setattr(ls, "setup_logging", old_style)
    assert cli.main(["status"]) == 0 and calls == [{"verbose": False}]     # plain commands pass no new keyword
    out.seek(0), out.truncate()
    assert cli.main(["--json", "status"]) == 0                             # TypeError -> basicConfig fallback
    assert doc(out)["data_dir"] == str(cfg.data_dir) and len(calls) == 1


# --- --json -----------------------------------------------------------------------------------------------

def test_status_json_on_an_empty_data_dir(env, out: io.StringIO):
    cfg = env()
    assert cli.main(["--json", "status"]) == 0
    d = doc(out)
    assert d["profile"] is None and d["profile_label"] == "自定义配置" and d["config_path"] == str(cfg["_config_path"])
    assert d["data_dir"] == str(cfg.data_dir) and d["service"] == {"running": False, "pid": None}
    assert d["api_key_configured"] is False and d["segments"] == {"total": 0, "by_status": {}, "failed": 0}
    assert d["utterances"] == {"today": 0, "total": 0, "days": 0} and d["speakers"] == []
    assert d["last_daily"] is None and d["last_daily_day"] is None and d["audio"]["bytes"] == 0


def test_status_json_matches_the_table(env, out: io.StringIO):
    cfg = env(api_key=FAKE_KEY)
    s = seed(cfg)
    s["db"].set_kv("last_daily_day", "2026-10-01")
    assert cli.main(["status", "--json"]) == 0                             # the flag also works after the command
    text = out.getvalue()
    d = json.loads(text)
    assert FAKE_KEY not in text and d["api_key_configured"] is True
    assert d["segments"] == {"total": 2, "failed": 1, "by_status": {"done": {"count": 1, "duration_s": 10.0},
                                                                    "failed": {"count": 1, "duration_s": 20.0}}}
    assert d["utterances"] == {"today": 1, "total": 2, "days": 2}
    assert d["audio"] == {"bytes": 2048, "max_disk_gb": float(cfg.storage.max_disk_gb)}
    assert d["last_daily"] == {"day": "2026-10-01", "created_at": "2026-10-01 22:30:00", "pushed_at": None,
                               "utterance_count": d["last_daily"]["utterance_count"]}
    assert d["last_daily_day"] == "2026-10-01" and d["day"] == s["today"].strftime("%Y-%m-%d")
    assert d["speakers"] == [{"id": s["zhang"], "name": "张三", "notes": "朋友", "samples": 2, "speech_s": 10.0},
                             {"id": s["li"], "name": "李四", "notes": None, "samples": 0, "speech_s": 0.0}]
    # the table is rendered from the same data
    out.seek(0), out.truncate()
    assert cli.main(["status"]) == 0
    table = out.getvalue()
    for needle in ("已配置", "done=1 (0 min)", "failed=1 (0 min)", "2 条 / 2 天", "2.0 KB", "调度 last_daily_day",
                   "2026-10-01（生成 2026-10-01 22:30:00，推送 未推送）", "失败段", "张三", "朋友", "10s"):
        assert needle in table, needle
    with pytest.raises(json.JSONDecodeError):
        json.loads(table)


def test_json_is_one_unwrapped_document_whatever_the_console_width(env, monkeypatch: pytest.MonkeyPatch):
    cfg = env()
    buf = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=buf, width=20, force_terminal=True, color_system="truecolor"))
    assert cli.main(["--json", "status"]) == 0
    text = buf.getvalue()
    assert "\x1b" not in text and json.loads(text)["data_dir"] == str(cfg.data_dir)
    assert text.startswith("{\n") and text.endswith("}\n")
    # values that look like rich markup or emoji codes are data, not formatting
    buf.seek(0), buf.truncate()
    cli.emit_json({"说明": "[red]不是标记[/red] :smile:", "path": Path("a") / "b", "when": datetime(2026, 11, 4, 9, 0),
                   "n": np.float32(1.5), "arr": np.arange(3), "set": {2, 1}})
    assert json.loads(buf.getvalue()) == {"说明": "[red]不是标记[/red] :smile:", "path": str(Path("a") / "b"),
                                          "when": "2026-11-04T09:00:00", "n": 1.5, "arr": [0, 1, 2], "set": [1, 2]}


def test_speakers_list_json(env, out: io.StringIO):
    cfg = env()
    s = seed(cfg)
    assert cli.main(["speakers", "list", "--json"]) == 0
    rows = doc(out)["speakers"]
    assert [r["name"] for r in rows] == ["张三", "李四"]                    # the user first, as in the table
    assert rows[0] == {"id": s["zhang"], "name": "张三", "is_self": True, "relation": None, "relation_source": None,
                       "samples": 2, "speech_s": 10.0, "created_at": rows[0]["created_at"], "notes": "朋友"}
    assert rows[1]["is_self"] is False and rows[1]["relation"] == "同事" and rows[1]["relation_source"] == "user"
    out.seek(0), out.truncate()
    assert cli.main(["--json", "speakers", "list"]) == 0 and len(doc(out)["speakers"]) == 2
    out.seek(0), out.truncate()
    assert cli.main(["speakers", "list"]) == 0 and "已注册说话人" in out.getvalue() and "同事" in out.getvalue()


def test_speakers_negatives_json(env, out: io.StringIO):
    cfg = env()
    s = seed(cfg)
    assert cli.main(["--json", "speakers", "negatives"]) == 0
    assert doc(out) == {"negatives": [{
        "id": s["neg"], "speaker_id": s["li"], "speaker": "李四", "duration_s": 2.5,
        "source_path": str(cfg.audio_dir / "a.opus") + "#0.0-1.0", "created_at": doc(out)["negatives"][0]["created_at"],
        "utterance_id": int(s["utts"][0]), "abs_start": s["today"].isoformat(), "text": "今天开会"}]}
    out.seek(0), out.truncate()
    assert cli.main(["speakers", "negatives", "张三", "--json"]) == 0 and doc(out) == {"negatives": []}
    out.seek(0), out.truncate()
    assert cli.main(["speakers", "negatives", "王敏", "--json"]) == 2      # unknown name: the usual error, as JSON
    assert "王敏" in doc(out)["error"]
    out.seek(0), out.truncate()
    assert cli.main(["speakers", "negatives", "--delete", str(s["neg"]), "--json"]) == 2
    assert "--delete" in doc(out)["error"]
    assert s["db"].one("SELECT COUNT(*) AS n FROM speaker_negatives")["n"] == 1      # nothing was deleted
    out.seek(0), out.truncate()
    assert cli.main(["speakers", "negatives"]) == 0 and "不是 TA 的声音" in out.getvalue() and "今天开会" in out.getvalue()


def _fake_voices(groups: list[dict], missing: list[int], calls: list[str]) -> types.ModuleType:
    mod = types.ModuleType("audio_note.speech.voices")
    mod.missing_segments = lambda db: list(missing)

    def backfill(db, cfg, engine):
        calls.append("backfill")
        return {"segments": len(missing), "clusters": 3, "skipped": 0}

    mod.backfill_cluster_embeddings = backfill
    mod.group_unknown_voices = lambda db, cfg: {"groups": groups, "missing_segments": 0, "threshold": 0.5,
                                                 "unknown_utterances": sum(g["utterances"] for g in groups)}
    mod.name_voice = lambda *a, **k: calls.append("name_voice")
    return mod


def test_speakers_voices_json_lists_groups_without_progress_lines(env, out: io.StringIO, monkeypatch: pytest.MonkeyPatch):
    env()
    groups = [
        {"key": "abc123def456", "members": [[1, 0], [2, 1]], "utterances": 5, "speech_s": 21.5, "segments": 2,
         "days": ["2026-11-03", "2026-11-04"], "n_days": 2, "closest": {"id": 1, "name": "张三", "score": np.float32(0.41)},
         "samples": [{"utterance_id": 9, "time": "09:00", "text": "下午再说", "seconds": 3.0}]},
        {"key": "ffe000aa1122", "members": [[3, 0]], "utterances": 1, "speech_s": 2.0, "segments": 1, "days": ["2026-11-04"],
         "n_days": 1, "closest": None, "samples": []},
    ]
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "audio_note.speech.voices", _fake_voices(groups, [7, 8], calls))
    from audio_note.cli import speakers as speakers_cmds

    # the placeholder model file cannot be loaded; the stubbed scan does not use the engine anyway
    monkeypatch.setattr(speakers_cmds, "_voice_engine_factory", lambda cfg: (lambda: object()))
    assert cli.main(["speakers", "voices", "--json"]) == 0
    d = doc(out)                                                           # valid JSON: no "正在扫描…" lines around it
    assert calls == ["backfill"] and d["scan"] == {"segments": 2, "clusters": 3, "skipped": 0}
    assert d["unknown_utterances"] == 6 and d["threshold"] == 0.5
    assert [(g["index"], g["key"], g["utterances"]) for g in d["groups"]] == [(1, "abc123def456", 5), (2, "ffe000aa1122", 1)]
    assert d["groups"][0]["members"] == [[1, 0], [2, 1]] and d["groups"][0]["closest"]["score"] == pytest.approx(0.41)
    assert d["groups"][0]["samples"][0]["text"] == "下午再说"
    # --assign changes data: not a JSON listing
    out.seek(0), out.truncate()
    assert cli.main(["--json", "speakers", "voices", "--assign", "1", "李四"]) == 2
    assert "--assign" in doc(out)["error"] and calls == ["backfill"]
    # the table output still prints the progress lines
    out.seek(0), out.truncate()
    assert cli.main(["speakers", "voices"]) == 0
    assert "正在扫描 2 段录音的声音" in out.getvalue() and "未识别的声音（2 组" in out.getvalue()


def test_speakers_voices_json_on_an_empty_database_uses_the_real_module(env, out: io.StringIO):
    env()
    assert cli.main(["--json", "speakers", "voices"]) == 0
    d = doc(out)
    assert d["groups"] == [] and d["scan"] == {"segments": 0, "clusters": 0, "skipped": 0} and d["unknown_utterances"] == 0


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake project root with a shared config.yaml and an empty profiles dir (as in test_profiles.py)."""
    for var in ("AUDIO_NOTE_CONFIG", "AUDIO_NOTE_PROFILE", "MIMO_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(config, "CURRENT_PROFILE_FILE", tmp_path / "profiles" / "current")
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    (tmp_path / "config.yaml").write_text(
        f"mimo:\n  api_key: {FAKE_KEY}\npaths:\n  data_dir: data\n  models_dir: models\n"
        "memory:\n  retrieval: fts\n  embedding_model: hash\n", encoding="utf-8")
    yield tmp_path
    ls._remove_own_handlers(logging.getLogger())


def test_profile_list_json(root: Path, out: io.StringIO):
    config.create_profile("work")
    c = load_config(profile="work")
    seed(c)
    config.create_profile("empty")
    assert cli.main(["--profile", "work", "profile", "list", "--json"]) == 0
    text = out.getvalue()
    d = json.loads(text)
    assert FAKE_KEY not in text
    assert d["current"] == "work" and d["config_path"] == str(config.profile_config_path("work"))
    rows = {r["name"]: r for r in d["profiles"]}
    assert set(rows) == {"default", "work", "empty"} and [r["name"] for r in d["profiles"]][0] == "default"
    assert rows["work"] == {"name": "work", "label": "work", "current": True, "data_dir": str(c.data_dir), "utterances": 2,
                            "days": 2, "speakers": 2, "last_day": rows["work"]["last_day"], "error": None}
    assert rows["default"]["label"] == "默认" and rows["default"]["current"] is False
    assert rows["empty"]["utterances"] is None and rows["empty"]["error"] is None     # no database yet: not created either
    assert not load_config(profile="empty").db_path.exists()
    out.seek(0), out.truncate()
    assert cli.main(["profile", "list"]) == 0
    table = out.getvalue()
    assert "2 条 / 2 天" in table and "（还没有数据）" in table and "● = 当前档案" in table


def test_global_json_is_rejected_by_commands_without_json_output(env, out: io.StringIO, monkeypatch: pytest.MonkeyPatch):
    env()
    called: list[str] = []
    monkeypatch.setattr(cli, "cmd_cleanup", lambda cfg, args: called.append("cleanup"))
    monkeypatch.setattr(cli, "cmd_speakers_scan", lambda cfg, args: called.append("scan"))
    assert cli.main(["--json", "cleanup"]) == 2 and "--json" in doc(out)["error"]
    out.seek(0), out.truncate()
    assert cli.main(["--json", "speakers", "scan"]) == 2 and "--json" in doc(out)["error"]
    assert called == []                                                    # never half-run with table output
    with pytest.raises(SystemExit) as ex:                                  # after the command it is an unknown option
        cli.main(["cleanup", "--json"])
    assert ex.value.code == 2
    assert cli.main(["cleanup"]) == 0 and called == ["cleanup"]            # without the flag nothing changed


def test_errors_are_json_documents_with_the_usual_exit_codes(env, out: io.StringIO, tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch):
    env()

    def boom(exc: BaseException):
        def handler(cfg, args):
            raise exc
        return handler

    for exc, code, text in ((cli.CLIError("故意失败"), 2, "故意失败"), (RuntimeError("kaboom"), 1, "RuntimeError: kaboom"),
                            (KeyboardInterrupt(), 130, "已中断")):
        monkeypatch.setattr(cli, "cmd_status", boom(exc))
        out.seek(0), out.truncate()
        assert cli.main(["--json", "status"]) == code
        assert doc(out) == {"error": text}
    bad = tmp_path / "bad.yaml"
    bad.write_text("mimo: [unclosed", encoding="utf-8")
    out.seek(0), out.truncate()
    assert cli.main(["--config", str(bad), "--json", "status"]) == 2
    assert doc(out)["error"].startswith("读取配置失败")
    # without --json the same failures print the same text as before
    monkeypatch.setattr(cli, "cmd_status", boom(cli.CLIError("故意失败")))
    out.seek(0), out.truncate()
    assert cli.main(["status"]) == 2 and "错误: 故意失败" in out.getvalue()


def test_json_mode_sends_log_lines_to_stderr(env, out: io.StringIO, capsys):
    env()
    assert cli.main(["-v", "--json", "status"]) == 0
    logging.getLogger("audio_note.test").warning("marker-for-stderr")
    captured = capsys.readouterr()
    assert "marker-for-stderr" in captured.err and "marker-for-stderr" not in captured.out
    json.loads(out.getvalue())


# --- doctor ------------------------------------------------------------------------------------------------

def checks(report: dict) -> dict[str, dict]:
    return {c["name"]: c for c in report["checks"]}


def test_doctor_is_read_only_on_a_fresh_install(env, out: io.StringIO, tmp_path: Path):
    cfg = env()
    before = sorted(p.name for p in tmp_path.iterdir())
    assert cli.main(["doctor"]) == 1                                       # no API key: a failed check
    assert sorted(p.name for p in tmp_path.iterdir()) == before            # no data dir, no log file, no database
    assert not cfg.data_dir.exists()
    text = out.getvalue()
    for needle in ("audio-note 体检", "配置文件", "数据目录", "还不存在", "模型文件", "MiMo API key", "未配置", "ffmpeg",
                   "服务", "未运行", "日志", "1 项有问题"):
        assert needle in text, needle
    assert "录音段" not in text                                             # no database: nothing to count


def test_doctor_json_on_a_healthy_profile(env, out: io.StringIO, monkeypatch: pytest.MonkeyPatch):
    cfg = env(api_key=FAKE_KEY)
    s = seed(cfg)
    s["db"].execute("DELETE FROM segments WHERE id=?", (s["failed"],))
    snapshot = sorted(p.name for p in cfg.data_dir.iterdir())
    assert cli.main(["doctor", "--json"]) == 0
    text = out.getvalue()
    d = json.loads(text)
    assert FAKE_KEY not in text                                            # the key is never printed
    assert d["ok"] is True and {c["status"] for c in d["checks"]} <= {"ok", "warn"}
    assert [c["name"] for c in d["checks"]] == ["config", "data_dir", "database", "segments", "processing", "models",
                                                "api_key", "ffmpeg", "service", "logs"]
    assert d["config"] == {"path": str(cfg["_config_path"]), "exists": True, "base_path": None, "profile": None,
                           "profile_label": "自定义配置"}
    assert d["data_dir"]["path"] == str(cfg.data_dir) and d["data_dir"]["exists"] and d["data_dir"]["writable"]
    assert d["data_dir"]["free_bytes"] > 0
    con = sqlite3.connect(cfg.db_path)
    try:
        user_version = con.execute("PRAGMA user_version").fetchone()[0]
    finally:
        con.close()
    assert d["database"]["schema_version"] == user_version == s["db"].schema_version()
    assert d["database"]["expected_schema_version"] == db_module.SCHEMA_VERSION and d["database"]["error"] is None
    assert d["database"]["path"] == str(cfg.db_path) and d["database"]["size_bytes"] > 0
    assert d["segments"] == {"total": 1, "by_status": {"done": {"count": 1, "duration_s": 10.0}}, "failed": 0,
                             "stuck_processing": {"count": 0, "older_than_s": 600, "items": []}}
    assert [(m["key"], m["exists"], m["size_bytes"]) for m in d["models"]] == [(k, True, 1) for k in MODEL_KEYS]
    assert d["api_key"] == {"configured": True, "source": "config"}
    assert d["service"] == {"running": False, "pid": None}
    assert [f["role"] for f in d["logs"]["files"]] == ["service", "cli"] and d["logs"]["errors"] == []
    assert d["logs"]["files"][1]["path"] == str(cfg.data_dir / "audio_note.cli.log")
    assert d["python"] and d["platform"] and d["time"]
    assert sorted(p.name for p in cfg.data_dir.iterdir()) == snapshot      # read-only: not even a log file
    # the key from the environment is reported as such, still without its value
    monkeypatch.setenv("MIMO_API_KEY", FAKE_KEY + "-env")
    out.seek(0), out.truncate()
    assert cli.main(["--json", "doctor"]) == 0
    assert doc(out)["api_key"] == {"configured": True, "source": "env"} and FAKE_KEY not in out.getvalue()
    out.seek(0), out.truncate()
    assert cli.main(["doctor"]) == 0
    table = out.getvalue()
    assert FAKE_KEY not in table and "已配置（来自环境变量 MIMO_API_KEY）" in table and "done=1" in table
    assert "有问题" not in table


def test_doctor_reports_stuck_and_failed_segments(env, out: io.StringIO):
    cfg = env(api_key=FAKE_KEY)
    s = seed(cfg)
    db = s["db"]
    now = datetime(2026, 11, 4, 12, 0, 0)
    ids = [db.add_segment(cfg.audio_dir / f"p{i}.opus", now - timedelta(minutes=30 + i), 5.0) for i in range(4)]
    stamps = ["2026-11-04T11:40:00",            # 20 min ago, ISO spelling
              "2026-11-04 11:30:00",            # 30 min ago, SQLite spelling
              "2026-11-04T11:55:00",            # 5 min ago: a run that is still working
              "not a time"]                     # unreadable: counted as stuck
    for seg_id, stamp in zip(ids, stamps):
        db.execute("UPDATE segments SET status='processing', updated_at=? WHERE id=?", (stamp, seg_id))
    r = cli.doctor_report(cfg, now=now)
    stuck = r["segments"]["stuck_processing"]
    assert stuck["count"] == 3 and stuck["older_than_s"] == 600
    assert [(i["id"], i["age_s"]) for i in stuck["items"]] == [(ids[0], 1200), (ids[1], 1800), (ids[3], None)]
    assert r["segments"]["by_status"]["processing"]["count"] == 4 and r["segments"]["failed"] == 1
    assert r["segments"]["total"] == 6
    c = checks(r)
    assert c["processing"]["status"] == "warn" and "3 段卡在 processing" in c["processing"]["detail"]
    assert f"#{ids[0]}" in c["processing"]["detail"] and "audio-note process" in c["processing"]["detail"]
    assert c["segments"]["status"] == "warn" and "--retry-failed" in c["segments"]["detail"]
    assert r["ok"] is True                                                 # warnings only
    assert cli.main(["doctor"]) == 0                                       # ... so the exit code stays 0
    assert "注意" in out.getvalue() and "项需要注意" in out.getvalue()
    # doctor only reports: the rows are still 'processing'
    assert db.one("SELECT COUNT(*) AS n FROM segments WHERE status='processing'")["n"] == 4


def _log_line(when: datetime, level: str, message: str, logger: str = "audio_note.scheduler") -> str:
    return f"{when.strftime('%Y-%m-%d %H:%M:%S')},123 {level:<7} MainThread {logger}: {message}\n"


def test_doctor_shows_the_last_errors_of_both_logs_without_secrets(env, out: io.StringIO):
    webhook = "https://push.example.invalid/hook/abcdef0123456789"
    cfg = env(api_key=FAKE_KEY, extra={"push": {"feishu": {"webhook": webhook, "secret": "x"}}})
    cfg.data_dir.mkdir(parents=True)
    now = datetime(2026, 11, 4, 12, 0, 0)
    service_log = (
        _log_line(now - timedelta(days=3), "ERROR", "old failure")
        + _log_line(now - timedelta(hours=5), "INFO", "job process done")
        + _log_line(now - timedelta(hours=4), "ERROR", f"job daily failed: boom (POST {webhook})")
        + "Traceback (most recent call last):\n  File \"x.py\", line 1, in <module>\nRuntimeError: boom\n"
        + _log_line(now - timedelta(hours=3), "WARNING", "retrying")
        + _log_line(now - timedelta(hours=2), "CRITICAL", f"request rejected for key {FAKE_KEY}", "audio_note.llm.client")
        + _log_line(now - timedelta(hours=1), "ERROR", "x" * 500))
    (cfg.data_dir / "audio_note.log").write_text(service_log, encoding="utf-8")
    (cfg.data_dir / "audio_note.cli.log").write_text(
        _log_line(now - timedelta(minutes=30), "ERROR", "command chat failed", "audio_note.cli"), encoding="utf-8")
    r = cli.doctor_report(cfg, now=now)
    errors = r["logs"]["errors"]
    assert [(e["source"], e["level"]) for e in errors] == [("service", "ERROR"), ("service", "ERROR"), ("service", "CRITICAL"),
                                                           ("service", "ERROR"), ("cli", "ERROR")]
    assert [e["time"] for e in errors] == sorted(e["time"] for e in errors)                  # oldest first
    assert errors[0]["message"].endswith("old failure") and "job daily failed: boom" in errors[1]["message"]
    assert errors[1]["message"].endswith("(POST ***)") and "request rejected for key ***" in errors[2]["message"]
    assert FAKE_KEY not in json.dumps(r, ensure_ascii=False) and webhook not in json.dumps(r, ensure_ascii=False)
    assert len(errors[3]["message"]) < 400 and errors[3]["message"].endswith("…")           # long lines are cut
    assert "command chat failed" in errors[4]["message"]
    c = checks(r)["logs"]
    assert c["status"] == "warn" and "最近 24 小时有 4 条错误" in c["detail"]
    assert [f["exists"] for f in r["logs"]["files"]] == [True, True]
    # only errors older than a day: informational
    quiet = cli.doctor_report(cfg, now=now + timedelta(days=5))
    assert checks(quiet)["logs"]["status"] == "ok" and len(quiet["logs"]["errors"]) == 5
    # only the newest LOG_ERRORS_SHOWN are kept, and a tail cut mid-line does not produce a bogus entry
    from audio_note.cli import system

    many = "".join(_log_line(now - timedelta(seconds=600 - i), "ERROR", f"e{i:03d} " + "y" * 2000) for i in range(300))
    (cfg.data_dir / "audio_note.log").write_text(many, encoding="utf-8")
    assert len(many.encode()) > system.LOG_TAIL_BYTES
    tail = cli.doctor_report(cfg, now=now)["logs"]["errors"]
    assert len(tail) == system.LOG_ERRORS_SHOWN and "e299" in tail[-1]["message"]
    assert cli.main(["doctor"]) == 0
    text = out.getvalue()
    assert "日志里最近的错误" in text and FAKE_KEY not in text and webhook not in text


def _raw_db(path: Path, user_version: int) -> None:
    """A database written by some other version of the program: one table, a stamped schema version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.execute("CREATE TABLE segments (id INTEGER PRIMARY KEY, path TEXT, started_at TEXT, duration_s REAL,"
                    " status TEXT, updated_at TEXT)")
        con.execute("INSERT INTO segments(path, started_at, duration_s, status, updated_at)"
                    " VALUES ('a.opus', '2026-11-04T09:00:00', 12.5, 'done', '2026-11-04 09:01:00')")
        con.execute(f"PRAGMA user_version={int(user_version)}")
        con.commit()
    finally:
        con.close()


def _db_state(path: Path) -> tuple[int, list[str]]:
    con = sqlite3.connect(path)
    try:
        return (con.execute("PRAGMA user_version").fetchone()[0],
                sorted(r[0] for r in con.execute("SELECT name FROM sqlite_master")))
    finally:
        con.close()


def test_doctor_never_migrates_and_flags_other_schema_versions(env, out: io.StringIO, monkeypatch: pytest.MonkeyPatch):
    cfg = env(api_key=FAKE_KEY)
    monkeypatch.setattr(db_module, "SCHEMA_VERSION", 2, raising=False)
    # an old database: reported, left exactly as it was (opening it with DB() would upgrade it)
    _raw_db(cfg.db_path, 0)
    before = _db_state(cfg.db_path)
    assert cli.main(["--json", "doctor"]) == 0
    d = doc(out)
    assert _db_state(cfg.db_path) == before == (0, ["segments"])
    assert d["database"]["schema_version"] == 0 and d["database"]["expected_schema_version"] == 2
    c = checks(d)["database"]
    assert c["status"] == "warn" and "自动升级" in c["detail"]
    assert d["segments"]["by_status"] == {"done": {"count": 1, "duration_s": 12.5}}
    # a database from a newer program: a failed check instead of a traceback
    cfg.db_path.unlink()
    _raw_db(cfg.db_path, 99)
    out.seek(0), out.truncate()
    assert cli.main(["--json", "doctor"]) == 1
    d = doc(out)
    assert d["ok"] is False and d["database"]["schema_version"] == 99
    assert checks(d)["database"]["status"] == "fail" and "更新的版本" in checks(d)["database"]["detail"]
    assert _db_state(cfg.db_path) == (99, ["segments"])
    # not a database at all
    cfg.db_path.write_bytes(b"this is not sqlite" * 100)
    out.seek(0), out.truncate()
    assert cli.main(["--json", "doctor"]) == 1
    d = doc(out)
    assert d["database"]["schema_version"] is None and d["database"]["error"] and d["segments"] is None
    assert checks(d)["database"]["status"] == "fail" and "打不开" in checks(d)["database"]["detail"]
    assert "segments" not in checks(d)
    # a readable database without the segment table
    cfg.db_path.unlink()
    con = sqlite3.connect(cfg.db_path)
    con.execute("CREATE TABLE other (x)")
    con.execute("PRAGMA user_version=2")
    con.commit()
    con.close()
    r = cli.doctor_report(cfg)
    assert r["database"]["schema_version"] == 2 and r["segments"] is None
    assert checks(r)["database"]["status"] == "fail" and "读不了录音段表" in checks(r)["database"]["detail"]


def test_doctor_lists_missing_models_and_ffmpeg_states(env, out: io.StringIO, monkeypatch: pytest.MonkeyPatch):
    from audio_note.cli import system

    cfg = env(api_key=FAKE_KEY, models=False)
    cfg.model_path("vad_model").write_bytes(b"x")
    monkeypatch.setattr(system, "_find_ffmpeg", lambda: {"found": False, "path": None, "source": None})
    assert cli.main(["--json", "doctor"]) == 1
    d = doc(out)
    assert [(m["key"], m["exists"]) for m in d["models"]] == [("vad_model", True), ("embedding_model", False),
                                                              ("segmentation_model", False)]
    c = checks(d)
    assert c["models"]["status"] == "fail" and f"{cfg.models_dir} 里缺少 2 个" in c["models"]["detail"]
    for key in ("embedding_model", "segmentation_model"):                  # named relative to the models dir
        assert Path(DEFAULTS["speakers"][key]).as_posix() in c["models"]["detail"]
    assert DEFAULTS["speakers"]["vad_model"] not in c["models"]["detail"] and "download_models" in c["models"]["detail"]
    assert d["models"][1]["path"] == str(cfg.model_path("embedding_model"))             # the data keeps the full path
    assert d["ffmpeg"] == {"found": False, "path": None, "source": None} and c["ffmpeg"]["status"] == "warn"
    assert d["ok"] is False
    out.seek(0), out.truncate()
    assert cli.main(["doctor"]) == 1 and "1 项有问题" in out.getvalue()
    monkeypatch.setattr(system, "_find_ffmpeg", lambda: {"found": True, "path": "/usr/bin/ffmpeg", "source": "PATH"})
    c = checks(cli.doctor_report(cfg))
    assert c["ffmpeg"]["status"] == "warn" and "imageio-ffmpeg" in c["ffmpeg"]["detail"]


def test_find_ffmpeg_prefers_the_bundled_binary(monkeypatch: pytest.MonkeyPatch):
    import shutil

    from audio_note.cli import system

    fake = types.ModuleType("imageio_ffmpeg")
    fake.get_ffmpeg_exe = lambda: r"C:\bundled\ffmpeg.exe"
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake)
    assert system._find_ffmpeg() == {"found": True, "path": r"C:\bundled\ffmpeg.exe", "source": "imageio-ffmpeg"}

    def no_binary():
        raise RuntimeError("no ffmpeg for this platform")

    fake.get_ffmpeg_exe = no_binary
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
    assert system._find_ffmpeg() == {"found": True, "path": "/usr/bin/ffmpeg", "source": "PATH"}
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert system._find_ffmpeg() == {"found": False, "path": None, "source": None}


def test_doctor_warns_about_a_missing_config_file_and_reports_the_profile(root: Path, out: io.StringIO):
    config.create_profile("work")
    data = root / "profiles" / "work" / "data"
    assert data.is_dir() and list(data.iterdir()) == []                    # create_profile makes the empty directory
    assert cli.main(["--profile", "work", "--json", "doctor"]) == 1        # models dir of the fake root is empty
    d = doc(out)
    assert d["config"]["profile"] == "work" and d["config"]["profile_label"] == "work" and d["config"]["exists"]
    assert d["config"]["base_path"] == str(root / "config.yaml")
    assert d["data_dir"]["path"] == str(data) and d["database"]["exists"] is False
    assert list(data.iterdir()) == []                                      # doctor wrote nothing into it
    (root / "config.yaml").unlink()
    out.seek(0), out.truncate()
    assert cli.main(["--json", "doctor"]) == 1
    d = doc(out)
    assert d["config"]["profile"] == "default" and d["config"]["profile_label"] == "默认" and not d["config"]["exists"]
    assert checks(d)["config"]["status"] == "warn" and "audio-note init" in checks(d)["config"]["detail"]
    assert checks(d)["api_key"]["status"] == "fail"


def test_doctor_escapes_rich_markup_in_paths_and_log_lines(env, out: io.StringIO):
    cfg = env(api_key=FAKE_KEY)
    cfg.data_dir.mkdir(parents=True)
    now = datetime.now()
    (cfg.data_dir / "audio_note.log").write_text(_log_line(now, "ERROR", "bad tag [/red] and [bold]text"), encoding="utf-8")
    assert cli.main(["doctor"]) == 0                                       # a MarkupError would be exit 1
    assert "bad tag [/red] and [bold]text" in out.getvalue()
