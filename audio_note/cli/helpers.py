"""Helpers shared by the command modules: errors, lazy imports, argument parsing, output.

`console` is a late-bound proxy of `audio_note.cli.console` (see `_late`): everything the CLI prints goes
through the one console object the package owns, so replacing it there captures all output.
"""
from __future__ import annotations

import argparse
import glob
import inspect
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from ..config import Config
from ..db import DB
from ._late import Late

log = logging.getLogger("audio_note.cli")
console = Late("console")


class CLIError(Exception):
    """User-facing error: printed in red, exit code 2."""


def _lazy_module(module: str) -> Any:
    """Import a submodule of audio_note (e.g. "memory.index") with the same clear error as _lazy()."""
    import importlib

    full = f"audio_note.{module}"
    try:
        return importlib.import_module(full)
    except ImportError as e:
        raise CLIError(f"模块 {full} 不可用（尚未实现或导入失败）: {e}") from e


def _lazy(module: str, attr: str) -> Any:
    """Import `attr` from `module` (relative to audio_note); clear message when it is missing."""
    import importlib

    full = f"audio_note.{module}"
    try:
        mod = importlib.import_module(full)
    except ImportError as e:
        raise CLIError(f"模块 {full} 不可用（尚未实现或导入失败）: {e}") from e
    try:
        return getattr(mod, attr)
    except AttributeError as e:
        raise CLIError(f"模块 {full} 中没有 {attr}") from e


def _client(cfg: Config) -> Any:
    from ..llm.client import MimoClient, MissingAPIKey

    try:
        return MimoClient(cfg, rpm=int(cfg.mimo.rpm))
    except MissingAPIKey as e:
        raise CLIError(f"{e}\n请在 config.yaml 的 mimo.api_key 填写 MiMo API key，或设置环境变量 MIMO_API_KEY") from e


def _meter(client: Any, db: Any) -> Any:
    """Write the calls of a CLI command's MiMo client to the usage_llm table like the service's, so
    /api/metrics also counts what CLI runs spent. Kept apart from `_client` (whose one-argument signature
    tests replace) and safe on any test double. Returns the client."""
    from .. import metrics

    metrics.install(client, db)
    return client


def _invoke(fn: Callable, **available: Any) -> Any:
    """Call `fn` passing only the parameters it declares, matched by name from `available`.
    Lets the CLI follow sibling modules whose exact signatures may differ slightly from DESIGN.md."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**available)
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for name, p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if name in available:
            if p.kind == p.POSITIONAL_ONLY:
                args.append(available[name])
            else:
                kwargs[name] = available[name]
        elif p.default is p.empty:
            raise CLIError(f"无法调用 {getattr(fn, '__qualname__', fn)}: 缺少参数 {name!r}")
    return fn(*args, **kwargs)


def _db(cfg: Config) -> DB:
    cfg.ensure_dirs()
    return DB(cfg.db_path)


def _reindex_after_enroll(cfg: Config, db: DB) -> None:
    """A newly enrolled name must be a single jieba token for FTS to find it: register it and
    recompute text_seg for existing utterances. Best effort; failures only log."""
    try:
        from ..memory import index as mem_index

        mem_index.register_from(cfg, db)
        n = db.one("SELECT COUNT(*) AS n FROM utterances")
        if n and n["n"]:
            k = mem_index.reindex(db)
            console.print(f"已更新检索索引（{k} 条）。")
    except Exception as e:  # noqa: BLE001
        log.warning("reindex after enroll failed: %s", e)


def _expand_files(patterns: list[str]) -> list[Path]:
    """Expand globs ourselves (PowerShell/cmd do not)."""
    out: list[Path] = []
    for pat in patterns:
        if any(ch in pat for ch in "*?["):
            hits = sorted(glob.glob(pat, recursive=True))
            if not hits:
                raise CLIError(f"没有匹配的文件: {pat}")
            out.extend(Path(h) for h in hits)
        else:
            p = Path(pat)
            if not p.exists():
                raise CLIError(f"文件不存在: {pat}")
            out.append(p)
    return out


def _parse_datetime(s: str) -> datetime:
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise CLIError(f"无法解析时间 {s!r}，格式如 \"2026-10-31 09:00\"")


def _parse_day(s: str | None) -> str:
    if not s:
        return datetime.now().strftime("%Y-%m-%d")
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as e:
        raise CLIError(f"日期格式应为 YYYY-MM-DD: {s!r}") from e


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def _print_result(obj: Any) -> None:
    """Pretty-print whatever a sibling module returns (str / dataclass / dict / list).

    Returned text is printed literally, never as rich markup: model answers cite sentences as [#123], which
    rich would take for a colour tag and silently drop (leaving "（、）" behind)."""
    from rich.markdown import Markdown
    from rich.markup import escape

    if obj is None:
        return
    if isinstance(obj, str):
        console.print(Markdown(obj) if obj.lstrip().startswith("#") else obj, markup=False)
        return
    if isinstance(obj, tuple) and obj and isinstance(obj[0], str):
        _print_result(obj[0])
        return
    text = getattr(obj, "text", None)
    if isinstance(text, str):
        console.print(text, markup=False)
        return
    if isinstance(obj, dict):
        if "text" in obj and isinstance(obj["text"], str):
            console.print(obj["text"], markup=False)
            return
        for k, v in obj.items():
            console.print(f"[bold]{escape(str(k))}[/bold]: {escape(str(v))}")
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            if isinstance(item, dict):
                t = item.get("text") or item.get("text_raw") or ""
                spk = item.get("speaker") or item.get("speaker_name") or ""
                start = item.get("start") if "start" in item else item.get("start_s")
                prefix = f"[{start:7.1f}s] " if isinstance(start, (int, float)) else ""
                console.print(f"{escape(prefix)}[bold]{escape(str(spk))}[/bold]{'：' if spk else ''}{escape(str(t))}")
            else:
                t = getattr(item, "text", None)
                spk = getattr(item, "speaker_name", None) or getattr(item, "speaker", None) or ""
                start = getattr(item, "start", None)
                if start is None:
                    start = getattr(item, "start_s", None)
                if t is not None:
                    prefix = f"[{start:7.1f}s] " if isinstance(start, (int, float)) else ""
                    console.print(f"{escape(prefix)}[bold]{escape(str(spk))}[/bold]{'：' if spk else ''}{escape(str(t))}")
                else:
                    console.print(str(item), markup=False)
        return
    console.print(str(obj), markup=False)


def _force_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """{"force": True} only when --force was given (engines without the keyword keep working)."""
    return {"force": True} if getattr(args, "force", False) else {}


def _speaker_by_name(db: DB, name: str) -> Any:
    row = db.one("SELECT * FROM speakers WHERE name=?", (str(name).strip(),))
    if row is None:
        raise CLIError(f"没有叫 {name} 的说话人（audio-note speakers list 查看）")
    return row


def resolve_label_target(db: DB, at: datetime, dur: float | None = None, tolerance_s: float = 60.0) -> tuple[Any, float, float]:
    """Find the utterance spoken at wall-clock time `at` (the timestamp shown in the transcript) and
    return (segment row, from_s, to_s) inside that segment's file. The utterance that contains `at`
    wins; otherwise the nearest one within tolerance_s. `dur` overrides the range length."""
    best = utterance_at(db, at, tolerance_s)
    seg = db.get_segment(int(best["segment_id"]))
    if seg is None:
        raise CLIError(f"语句所属的录音段 #{best['segment_id']} 已不存在")
    a = float(best["start_s"])
    b = a + float(dur) if dur else float(best["end_s"])
    return seg, a, b


def utterance_at(db: DB, at: datetime, tolerance_s: float = 60.0) -> Any:
    """The utterance row spoken at wall-clock time `at`: the one containing it, else the nearest within tolerance_s."""
    iso = at.isoformat(timespec="milliseconds")
    rows = db.query(
        "SELECT u.*, s.started_at AS seg_started_at FROM utterances u JOIN segments s ON s.id=u.segment_id"
        " WHERE u.abs_start BETWEEN ? AND ? ORDER BY u.abs_start",
        ((at - timedelta(seconds=tolerance_s)).isoformat(timespec="milliseconds"),
         (at + timedelta(seconds=tolerance_s)).isoformat(timespec="milliseconds")),
    )
    best = None
    best_key = None
    for r in rows:
        start = datetime.fromisoformat(r["abs_start"])
        length = max(0.0, float(r["end_s"]) - float(r["start_s"]))
        delta = (at - start).total_seconds()
        contains = 0.0 <= delta <= length
        key = (0 if contains else 1, abs(delta))
        if best_key is None or key < best_key:
            best, best_key = r, key
    if best is None:
        raise CLIError(f"{iso} 附近 {tolerance_s:.0f}s 内没有已转写的语句（看 transcripts/ 里的时间）")
    return best


def _optional_engine(cfg: Config) -> Any:
    """SpeakerEngine, or None when the voice model is missing / fails to load (a veto still works without it)."""
    try:
        if not cfg.model_path("embedding_model").exists():
            return None
    except Exception:  # noqa: BLE001 - odd config: let the engine decide
        pass
    try:
        from ..speech.speakers import SpeakerEngine

        return SpeakerEngine(cfg)
    except Exception as e:  # noqa: BLE001
        log.warning("speaker engine unavailable (%s): veto only, no negative sample", e)
        return None


# --- machine-readable output (--json) ---------------------------------------------

JSON_HELP = "输出一份 JSON（给脚本用），不打印表格"


def add_json_flag(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Declare that a (leaf) command supports `--json`. The flag is accepted before the sub-command
    (`audio-note --json status`, the global one) and after it (`audio-note status --json`); commands
    that did not call this reject the global flag instead of printing tables to a JSON consumer."""
    # SUPPRESS: without the flag the sub-parser must not reset what the global flag already stored
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help=JSON_HELP)
    parser.set_defaults(json_ok=True)
    return parser


def wants_json(args: Any) -> bool:
    return bool(getattr(args, "json", False))


def _json_default(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=str)
    for attr in ("tolist", "item"):          # numpy arrays / scalars
        fn = getattr(o, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:  # noqa: BLE001
                pass
    return str(o)


def emit_json(obj: Any) -> None:
    """Print one JSON document (UTF-8 text, 2-space indent) through the console: no markup, no
    highlighting, no wrapping or cropping whatever the terminal width."""
    console.out(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default), highlight=False)
