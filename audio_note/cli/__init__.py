"""Command-line entry point: `audio-note <cmd>` (also `python -m audio_note` and `python -m audio_note.cli`).

Global options (before the command)
  --config/-c PATH                one config file instead of the current profile
  --profile NAME                  run this one command against profiles/<NAME>/
  --verbose/-v                    DEBUG logging
  --json                          one JSON document on stdout instead of tables (status, doctor, speakers
                                  list|voices|negatives, profile list; also accepted after the command)

Commands, grouped by the module that registers them (COMMAND_MODULES order = `--help` order)
  service       devices                         list audio input devices / system-audio sources
                record                          record only (VAD-gated segments)
                run [--tray] [--no-ui] [--open] [--port N]    the service: record + scheduler + web UI
                ui [--record] [--port N] [--no-browser]       web UI without recording (read-only next to a running service)
  pipeline      status                          counts, today's utterances, disk usage, last daily, speakers
                import <files...> [--start]     import existing recordings
                process [--retry-failed] [--limit N]
                transcribe <file>               one-off transcription, printed, not stored
  speakers      speakers enroll|record|list|label|relabel|calibrate|audit   (enroll/label/voices --assign take
                                                --force to add a sample that sounds unlike the person)
                speakers voices|scan            unidentified voices in the recordings: list / name a group
                speakers set-self|relation|relations   who the user is, relations to the user (by hand / by the LLM)
                speakers reject|negatives       "not this person" for a sentence; list / delete negative samples
  memory        daily [--day] [--push] [--force]
                push-test <text>
                chat [question...]              one question, or an interactive REPL without arguments
                reindex [--no-vectors]          rebuild the keyword index, embed what has no vector yet
  maintenance   cleanup                         delete no-speech / expired audio
                loudness [--day D] [--apply --yes]   speech level per sentence; --apply deletes the faint ones
  profile       profile list|create|use|delete  separate recordings / records / speakers per person or purpose
  system        init                            create config.yaml, check model files
                doctor                          read-only health report (exit 1 when a check fails)
  lexicon       lexicon list|add|alias|import|export|apply   the vocabulary: canonical spellings, mis-hearings, history fixes
  inbox         inbox list|answer|dismiss       the questions the system wants to ask ("待确认")

Exit codes: 0 ok, 1 command-specific failure or an unexpected exception, 2 usage / configuration / user
error (CLIError), 130 Ctrl+C.

Package layout
  __init__.py   this facade: `console`, `log`, `PROJECT_ROOT`, `load_config`, `build_parser`, `main`, and
                re-exports of every helper and `cmd_*` handler (tests and embedders import and replace them here)
  _late.py      late binding: command modules reach `console`, `_client`, `_optional_engine`, `load_config`
                and `PROJECT_ROOT` through the facade at call time, so a replacement made here is seen
  helpers.py    CLIError, lazy imports, argument parsing, printing, the --json helpers
  <domain>.py   the handlers of one domain plus `register(sub)` which adds its argparse sub-commands

Adding commands (no placeholder modules exist on purpose)
  1. A new command tree: create `audio_note/cli/<name>.py` with handlers `cmd_x(cfg, args) -> int` and a
     `register(sub) -> None` that calls `sub.add_parser(...)` and `set_defaults(func=handler(cmd_x))`
     (`handler` is in `_late`), then append "<name>" to COMMAND_MODULES below. The position in the list
     is the position in `--help`. One more command in an existing domain goes into that module's `register`.
  2. In a command module import only `._late`, `.helpers`, `..config` / `..db` and the standard library /
     rich at module level. Every other project module (speech, memory, analysis, audio, service, runtime...)
     is imported inside the function that needs it: start-up stays fast, a broken optional module only
     breaks its own command, and tests can stub the module through `sys.modules`.
  3. Print through `console = Late("console")`, get the LLM through `_client = Late("_client")`, raise
     `CLIError` for user errors (exit 2), return the exit code (None = 0).
  4. Read-only listings should take `--json`: `add_json_flag(parser)` when registering, then
     `if wants_json(args): emit_json(data); return 0`. A command that must not create the data
     directory or a log file (like `doctor`) adds `read_only=True` to its `set_defaults`.
  5. A new global option that takes a value must be added to `logging_setup.GLOBAL_VALUE_OPTIONS`, a new
     long-running command to `logging_setup.SERVICE_COMMANDS` (they decide which log file the process owns).
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from typing import Any

from rich.console import Console

from ..config import (DEFAULT_PROFILE, PROJECT_ROOT, Config, create_profile, delete_profile,  # noqa: F401
                      list_profiles, load_config, profile_config_path, set_current_profile)
from ..db import DB  # noqa: F401
from ..speech.calibrate import calibrate_threshold  # noqa: F401  (moved; still importable from here)

log = logging.getLogger(__name__)
#: the one console every command prints through (command modules hold a late-bound proxy of it)
console = Console()

# The command modules are imported after `console` exists and with explicit name lists: a star import
# would overwrite the real `console` / `_client` with a sub-module's proxy of that very attribute.
from .helpers import (CLIError, _client, _db, _expand_files, _force_kwargs, _human_bytes, _invoke,  # noqa: E402,F401
                      _lazy, _lazy_module, _optional_engine, _parse_datetime, _parse_day, _print_result,
                      _reindex_after_enroll, _speaker_by_name, add_json_flag, emit_json, resolve_label_target,
                      utterance_at, wants_json)
from .maintenance import cmd_cleanup, cmd_loudness  # noqa: E402,F401
from .memory import cmd_chat, cmd_daily, cmd_push_test, cmd_reindex  # noqa: E402,F401
from .pipeline import cmd_import, cmd_process, cmd_status, cmd_transcribe, status_data  # noqa: E402,F401
from .profile import (_profile_counts, cmd_profile_create, cmd_profile_delete, cmd_profile_list,  # noqa: E402,F401
                      cmd_profile_use)
from .service import _wait_with_ctrl_c, cmd_devices, cmd_record, cmd_run, cmd_ui  # noqa: E402,F401
from .speakers import (KEY_SHOWN, SAMPLE_ORIGIN_LABELS, _print_voice_groups, _relabel_segment_quietly,  # noqa: E402,F401
                       _relation_text, _resolve_voice_group, _scan_voices, _voice_engine_factory,
                       cmd_speakers_audit, cmd_speakers_calibrate, cmd_speakers_enroll, cmd_speakers_label,
                       cmd_speakers_list, cmd_speakers_negatives, cmd_speakers_record, cmd_speakers_reject,
                       cmd_speakers_relabel, cmd_speakers_relation, cmd_speakers_relations, cmd_speakers_scan,
                       cmd_speakers_set_self, cmd_speakers_voices)
from .system import MINIMAL_CONFIG, MODEL_FILES, cmd_doctor, cmd_init, doctor_report  # noqa: E402,F401

#: The handlers re-exported above, by name: `_late.handler()` dispatches these through the facade attribute
#: of the same name (looked up when the parser is built), so replacing `audio_note.cli.cmd_x` takes effect.
EXPORTED_HANDLERS: dict[str, Any] = {name: fn for name, fn in list(globals().items())
                                     if name.startswith("cmd_") and callable(fn)}

#: Command modules of this package, in `--help` order. Each exposes `register(sub) -> None`.
#: To add commands: write the module, append its name here (see the recipe in the docstring).
COMMAND_MODULES: list[str] = ["service", "pipeline", "speakers", "memory", "maintenance", "profile", "system", "lexicon",
                              "inbox", "memory_tree"]


# --- argparse ----------------------------------------------------------------------

def _command_modules() -> list[Any]:
    """The modules named in COMMAND_MODULES. One that cannot be imported is skipped with a warning: its
    commands are missing, the rest of the CLI (the recorder service above all) keeps working."""
    mods: list[Any] = []
    for name in COMMAND_MODULES:
        full = f"{__name__}.{name}"
        mod = sys.modules.get(full)
        if mod is None:
            try:
                mod = importlib.import_module(full)
            except Exception as e:  # noqa: BLE001
                log.warning("command module %s is unavailable, its commands are missing: %s: %s", full, type(e).__name__, e)
                continue
        mods.append(mod)
    return mods


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="audio-note", description="全天无感录音 → 说话人识别 → MiMo 转写 → 日报与记忆问答")
    p.add_argument("--config", "-c", metavar="PATH", help="config.yaml 路径（默认 $AUDIO_NOTE_CONFIG 或项目根目录 config.yaml）")
    p.add_argument("--profile", metavar="NAME", help="用某个档案（profiles/<NAME>/）运行这一条命令；默认用 profile use 选的档案")
    p.add_argument("--verbose", "-v", action="store_true", help="DEBUG 日志")
    p.add_argument("--json", action="store_true",
                   help="输出一份 JSON 而不是表格（status / doctor / speakers list|voices|negatives / profile list；也可以写在命令后面）")
    sub = p.add_subparsers(dest="cmd", metavar="<cmd>")
    sub.required = True
    for mod in _command_modules():
        try:
            mod.register(sub)
        except Exception as e:  # noqa: BLE001 - one broken module must not take the whole CLI down
            log.warning("command module %s failed to register its commands: %s: %s", mod.__name__, type(e).__name__, e)
    return p


def _utf8_stdio() -> None:
    """When stdout/stderr are pipes (not a console) Python on Windows uses the ANSI code page;
    switch them to UTF-8 so Chinese help/output survives redirection and PowerShell pipelines."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and not stream.isatty() and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def _report_error(as_json: bool, text: str, markup: str) -> None:
    """Errors are printed as rich text; with --json the single document on stdout is {"error": text}."""
    if as_json:
        emit_json({"error": text})
    else:
        console.print(markup)


def main(argv: list[str] | None = None) -> int:
    _utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    as_json = wants_json(args)
    if as_json and not getattr(args, "json_ok", False):
        _report_error(True, "这个命令不支持 --json", "")
        return 2
    try:
        cfg = load_config(args.config, profile=args.profile)
    except Exception as e:  # noqa: BLE001
        _report_error(as_json, f"读取配置失败: {e}", f"[red]读取配置失败[/red]: {e}")
        return 2
    log_opts: dict[str, Any] = {}
    if as_json:
        log_opts["stderr"] = True          # stdout carries exactly one JSON document
    if getattr(args, "read_only", False):
        log_opts["file"] = False           # nothing may be created in the data dir behind the command's back
    try:
        from ..logging_setup import setup_logging

        setup_logging(cfg, verbose=bool(args.verbose), **log_opts)
    except Exception as e:  # noqa: BLE001
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
        log.warning("logging setup failed: %s", e)
    try:
        return int(args.func(cfg, args) or 0)
    except CLIError as e:
        _report_error(as_json, str(e), f"[red]错误[/red]: {e}")
        return 2
    except KeyboardInterrupt:
        _report_error(as_json, "已中断", "\n已中断")
        return 130
    except Exception as e:  # noqa: BLE001
        log.exception("command %s failed", args.cmd)
        _report_error(as_json, f"{type(e).__name__}: {e}", f"[red]{type(e).__name__}[/red]: {e}")
        return 1
