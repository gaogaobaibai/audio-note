"""Installation and health commands: `init`, `doctor`.

`doctor` is strictly read-only: it never creates a directory, opens the database in read-only mode (so it
cannot trigger a schema migration) and never prints the API key - only whether one is configured.
"""
from __future__ import annotations

import argparse
import logging
import os
import platform
import re
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from rich.table import Table

from ..config import DEFAULT_PROFILE, Config
from ._late import Late, handler, pkg
from .helpers import _human_bytes, add_json_flag, emit_json, wants_json

log = logging.getLogger("audio_note.cli")
console = Late("console")
load_config = Late("load_config")

#: (config key under `speakers`, what the model is for) - the files `init` and `doctor` look for
MODEL_FILES = (("vad_model", "VAD (Silero)"), ("embedding_model", "说话人 embedding (CAM++)"),
               ("segmentation_model", "说话人分离 (pyannote)"))


MINIMAL_CONFIG = """# audio_note 配置（由 `audio-note init` 生成）。
# 没有写出的键使用 audio_note/config.py 里的 DEFAULTS；完整示例见 config.example.yaml。

mimo:
  api_key: ""                      # 必填：MiMo API key（也可用环境变量 MIMO_API_KEY，优先级更高）
  base_url: https://api.xiaomimimo.com/v1
  chat_model: mimo-v2.6-flash      # 纠错 / 问答；mimo-v2.6-pro 更强更贵
  analysis_model: ""               # 每日分析用的模型，留空 = chat_model
  asr_model: mimo-v2.5-asr
  asr_language: auto               # auto | zh | en

paths:
  data_dir: data                   # 录音、数据库、转写、日报都在这里（相对项目根目录）
  models_dir: models

audio:
  device: null                     # null = 系统默认输入；或设备序号；或名称子串（audio-note devices 查看）
  vad_threshold: 0.5
  min_silence_s: 2.0
  max_segment_s: 300

storage:
  keep_audio_days: 30              # 音频保留天数（文字永久保留），0 = 永久
  max_disk_gb: 20

speakers:
  identify_threshold: 0.45         # 说话人识别阈值，`audio-note speakers calibrate` 给出建议值

pipeline:
  process_interval_min: 10

analysis:
  daily_time: "22:30"              # 每日分析时间 HH:MM
  push_daily: true

push:
  channels: [toast]                # toast | feishu | wecom | serverchan | pushplus | wxpusher | ntfy
  feishu: {webhook: "", secret: ""}
  wecom: {webhook: ""}
  serverchan: {sendkey: ""}
  pushplus: {token: ""}
  wxpusher: {spt: "", app_token: "", uids: []}
  ntfy: {topic: "", server: "https://ntfy.sh"}

log:
  level: INFO
"""


def cmd_init(cfg: Config, args: argparse.Namespace) -> int:
    """Create the root config.yaml (the file every profile inherits from) unless --config names another."""
    path = Path(args.config) if args.config else pkg().PROJECT_ROOT / "config.yaml"
    if path.exists():
        console.print(f"配置文件已存在: {path}（不覆盖）")
    else:
        example = pkg().PROJECT_ROOT / "config.example.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        if example.exists():
            shutil.copyfile(example, path)
            console.print(f"已从 config.example.yaml 生成 {path}")
        else:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(MINIMAL_CONFIG)
            console.print(f"已生成 {path}")
        cfg = load_config(path)
    cfg.ensure_dirs()
    console.print(f"数据目录: {cfg.data_dir}")

    missing: list[str] = []
    t = Table(title="模型文件")
    t.add_column("用途")
    t.add_column("路径")
    t.add_column("状态")
    for key, what in MODEL_FILES:
        p = cfg.model_path(key)
        ok = p.exists()
        if not ok:
            missing.append(str(p))
        t.add_row(what, str(p), "[green]OK[/green]" if ok else "[red]缺失[/red]")
    console.print(t)
    if missing:
        console.print("[red]缺失模型文件:[/red]")
        for m in missing:
            console.print(f"  {m}")
        console.print("请从 sherpa-onnx releases 下载并放到 models/ 目录（见 README）。")
    if not cfg.mimo.api_key:
        console.print(f"[yellow]mimo.api_key 未填写[/yellow]：编辑 {path} 或设置环境变量 MIMO_API_KEY")
    gl = cfg.glossary_path
    if not gl.exists():
        console.print(f"提示：可创建词表 {gl}（每行一个人名/术语，纠错时优先使用）")
    return 1 if missing else 0


# --- doctor ---------------------------------------------------------------------------

OK, WARN, FAIL = "ok", "warn", "fail"
STATUS_TEXT = {OK: "[green]正常[/green]", WARN: "[yellow]注意[/yellow]", FAIL: "[red]有问题[/red]"}
#: a 'processing' row untouched for this long belongs to a dead run (same age the pipeline reclaims them at)
STUCK_PROCESSING_S = 600
STUCK_SHOWN = 10
LOW_DISK_BYTES = 1024 ** 3
LOG_TAIL_BYTES = 256 * 1024
LOG_ERRORS_SHOWN = 5
LOG_MESSAGE_MAX = 300
#: config values that must never appear in a report: log lines are scrubbed of them before they are shown
SECRET_KEYS = (("mimo", "api_key"), ("push", "feishu", "webhook"), ("push", "feishu", "secret"), ("push", "wecom", "webhook"),
               ("push", "serverchan", "sendkey"), ("push", "pushplus", "token"), ("push", "wxpusher", "spt"),
               ("push", "wxpusher", "app_token"))
SECRET_MIN_LEN = 6
#: file log format of logging_setup: "<asctime> <LEVEL>   <thread> <logger>: <message>"
_LOG_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,.]\d{3} (ERROR|CRITICAL)\s+(.*)$")


def _profile_label(name: str | None) -> str:
    # same wording as runtime.profile_label, without importing the runtime (doctor must work when it cannot)
    if name is None:
        return "自定义配置"
    return "默认" if name == DEFAULT_PROFILE else str(name)


def _ro_connect(path: Path) -> sqlite3.Connection:
    """Read-only connection: never creates the file, never runs a migration."""
    quoted = path.as_posix().replace("%", "%25").replace("?", "%3f").replace("#", "%23")
    return sqlite3.connect(f"file:{quoted}?mode=ro", uri=True, timeout=5)


def _parse_ts(value: Any) -> datetime | None:
    """`updated_at` exists in two spellings: 'YYYY-MM-DD HH:MM:SS' (SQLite default) and 'YYYY-MM-DDTHH:MM:SS'.
    Always naive local time (an offset, should one ever be stored, is dropped rather than compared)."""
    try:
        return datetime.fromisoformat(str(value).strip().replace(" ", "T")).replace(tzinfo=None)
    except ValueError:
        return None


def _file_size(p: Path) -> int | None:
    try:
        return int(p.stat().st_size)
    except OSError:
        return None


def _short_path(path: str, base: Path) -> str:
    """`path` relative to `base` when it is inside it (the full path otherwise)."""
    try:
        return Path(path).relative_to(base).as_posix()
    except ValueError:
        return str(path)


def _find_ffmpeg() -> dict[str, Any]:
    """The ffmpeg the importer uses (through imageio-ffmpeg, which ships a binary), else one on PATH."""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe:
            return {"found": True, "path": str(exe), "source": "imageio-ffmpeg"}
    except Exception:  # noqa: BLE001 - package missing or no binary for this platform
        pass
    exe = shutil.which("ffmpeg")
    return {"found": bool(exe), "path": exe, "source": "PATH" if exe else None}


def _secrets(cfg: Config) -> list[str]:
    """The configured secret values (API key, push webhooks / tokens), longest first."""
    found: set[str] = set()
    for path in SECRET_KEYS:
        node: Any = cfg
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, str) and len(node.strip()) >= SECRET_MIN_LEN:
            found.add(node.strip())
    return sorted(found, key=len, reverse=True)


def _log_errors(path: Path, source: str, secrets: list[str] | tuple[str, ...] = ()) -> list[dict[str, str]]:
    """ERROR / CRITICAL lines in the last LOG_TAIL_BYTES of a log file (oldest first; traceback lines are
    skipped). Every occurrence of a value in `secrets` is replaced by ***."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - LOG_TAIL_BYTES))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = text.splitlines()
    if size > LOG_TAIL_BYTES and lines:
        lines = lines[1:]                       # the first line of the tail is cut somewhere in the middle
    out: list[dict[str, str]] = []
    for line in lines:
        m = _LOG_LINE.match(line)
        if not m:
            continue
        msg = m.group(3).strip()
        for secret in secrets:
            msg = msg.replace(secret, "***")
        if len(msg) > LOG_MESSAGE_MAX:
            msg = msg[:LOG_MESSAGE_MAX] + "…"
        out.append({"time": m.group(1), "level": m.group(2), "source": source, "message": msg})
    return out


def _database_report(cfg: Config, now: datetime) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """(database section, segments section or None when the segment table cannot be read)."""
    from .. import db as db_module

    path = cfg.db_path
    database: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "size_bytes": _file_size(path),
                                "schema_version": None, "expected_schema_version": getattr(db_module, "SCHEMA_VERSION", None),
                                "error": None}
    if not database["exists"]:
        return database, None
    segments: dict[str, Any] | None = None
    try:
        con = _ro_connect(path)
        try:
            database["schema_version"] = int(con.execute("PRAGMA user_version").fetchone()[0])
            try:
                rows = con.execute("SELECT status, COUNT(*), COALESCE(SUM(duration_s),0) FROM segments"
                                   " GROUP BY status ORDER BY status").fetchall()
                processing = con.execute("SELECT id, started_at, updated_at FROM segments WHERE status='processing'"
                                         " ORDER BY id").fetchall()
            except sqlite3.Error as e:          # a readable file without our tables, or a damaged one
                database["error"] = str(e)
            else:
                by_status = {str(r[0]): {"count": int(r[1]), "duration_s": float(r[2])} for r in rows}
                stuck = []
                for seg_id, started_at, updated_at in processing:
                    ts = _parse_ts(updated_at)
                    age = (now - ts).total_seconds() if ts else None
                    if age is None or age >= STUCK_PROCESSING_S:      # an unreadable timestamp counts as stuck
                        stuck.append({"id": int(seg_id), "started_at": started_at, "updated_at": updated_at,
                                      "age_s": round(age) if age is not None else None})
                segments = {"total": sum(s["count"] for s in by_status.values()), "by_status": by_status,
                            "failed": by_status.get("failed", {}).get("count", 0),
                            "stuck_processing": {"count": len(stuck), "older_than_s": STUCK_PROCESSING_S,
                                                 "items": stuck[:STUCK_SHOWN]}}
        finally:
            con.close()
    except sqlite3.Error as e:
        database["error"] = str(e)
    return database, segments


def doctor_report(cfg: Config, now: datetime | None = None) -> dict[str, Any]:
    """Health report of the profile `cfg` describes, as plain data (the table and `--json` render it).

    Read-only by construction: no directory is created, the database is opened with mode=ro, the API key
    is reduced to configured yes/no. `checks` is the verdict list (status ok | warn | fail); `ok` is False
    when at least one check failed."""
    import audio_note

    now = now or datetime.now()
    checks: list[dict[str, str]] = []

    def check(name: str, label: str, status: str, detail: str) -> None:
        checks.append({"name": name, "label": label, "status": status, "detail": detail})

    # --- config / profile
    cfg_path = Path(str(cfg.get("_config_path") or ""))
    profile = cfg.get("_profile")
    config = {"path": str(cfg_path), "exists": cfg_path.is_file(), "base_path": cfg.get("_base_config_path"),
              "profile": profile, "profile_label": _profile_label(profile)}
    if config["exists"]:
        check("config", "配置文件", OK, f"{cfg_path}（档案：{config['profile_label']}）")
    else:
        check("config", "配置文件", WARN, f"{cfg_path} 不存在，全部使用默认设置（audio-note init 生成）")

    # --- data dir
    data_dir = cfg.data_dir
    exists = data_dir.is_dir()
    free: int | None = None
    if exists:
        try:
            free = int(shutil.disk_usage(data_dir).free)
        except OSError:
            free = None
    data = {"path": str(data_dir), "exists": exists, "writable": os.access(data_dir, os.W_OK) if exists else None,
            "free_bytes": free}
    if not exists:
        check("data_dir", "数据目录", WARN, f"{data_dir} 还不存在（第一次录音或处理时创建）")
    elif not data["writable"]:
        check("data_dir", "数据目录", FAIL, f"{data_dir} 不可写")
    elif free is not None and free < LOW_DISK_BYTES:
        check("data_dir", "数据目录", WARN, f"{data_dir}：磁盘只剩 {_human_bytes(free)}")
    else:
        check("data_dir", "数据目录", OK, f"{data_dir}" + (f"（磁盘剩余 {_human_bytes(free)}）" if free is not None else ""))

    # --- database + segments
    database, segments = _database_report(cfg, now)
    ver, expected = database["schema_version"], database["expected_schema_version"]
    if not database["exists"]:
        check("database", "数据库", WARN, f"{database['path']} 还不存在（第一次录音或处理时创建）")
    elif ver is None:
        check("database", "数据库", FAIL, f"打不开 {database['path']}：{database['error']}")
    elif expected is not None and ver > int(expected):
        check("database", "数据库", FAIL, f"数据库来自更新的版本（库的结构版本 {ver}，程序是 {expected}）：请升级 audio-note")
    elif database["error"]:
        check("database", "数据库", FAIL, f"结构版本 {ver}，但读不了录音段表：{database['error']}")
    elif expected is not None and ver < int(expected):
        check("database", "数据库", WARN, f"结构版本 {ver}，程序需要 {expected}：下次启动服务或执行会写库的命令时自动升级（升级前自动备份）")
    else:
        check("database", "数据库", OK, f"结构版本 {ver}，{_human_bytes(database['size_bytes'] or 0)}")
    if segments is not None:
        counts = "  ".join(f"{k}={v['count']}" for k, v in segments["by_status"].items()) or "0"
        if segments["failed"]:
            check("segments", "录音段", WARN, f"{counts}；{segments['failed']} 段处理失败（audio-note process --retry-failed 重试）")
        else:
            check("segments", "录音段", OK, counts)
        stuck = segments["stuck_processing"]
        if stuck["count"]:
            ids = "、".join(f"#{s['id']}" for s in stuck["items"]) + ("…" if stuck["count"] > len(stuck["items"]) else "")
            check("processing", "处理中", WARN, f"{stuck['count']} 段卡在 processing 超过 {STUCK_PROCESSING_S // 60} 分钟（{ids}），"
                                               "多半是上次处理被中断：audio-note process 会收回重做")
        else:
            check("processing", "处理中", OK, "没有卡住的录音段")

    # --- model files
    models: list[dict[str, Any]] = []
    for key, what in MODEL_FILES:
        try:
            p = cfg.model_path(key)
        except Exception as e:  # noqa: BLE001 - odd config (key missing / not a string)
            models.append({"key": key, "label": what, "path": None, "exists": False, "size_bytes": None, "error": str(e)})
            continue
        models.append({"key": key, "label": what, "path": str(p), "exists": p.is_file(), "size_bytes": _file_size(p)})
    missing = [m for m in models if not m["exists"]]
    if missing:
        check("models", "模型文件", FAIL, f"{cfg.models_dir} 里缺少 {len(missing)} 个："
              + "、".join(_short_path(m["path"], cfg.models_dir) if m["path"] else m["key"] for m in missing)
              + "（运行 scripts/download_models.ps1）")
    else:
        check("models", "模型文件", OK, f"{len(models)} 个都在 {cfg.models_dir}")

    # --- API key: configured yes / no, never the value
    has_key = bool(cfg.mimo.api_key)
    api_key = {"configured": has_key, "source": ("env" if os.environ.get("MIMO_API_KEY") else "config") if has_key else None}
    if has_key:
        check("api_key", "MiMo API key", OK, "已配置（来自环境变量 MIMO_API_KEY）" if api_key["source"] == "env" else "已配置")
    else:
        check("api_key", "MiMo API key", FAIL, "未配置：填写 config.yaml 的 mimo.api_key 或设置环境变量 MIMO_API_KEY"
                                              "（没有它不能转写、生成日报和问答）")

    # --- ffmpeg (only needed to import formats libsndfile cannot read)
    ffmpeg = _find_ffmpeg()
    if ffmpeg["source"] == "imageio-ffmpeg":
        check("ffmpeg", "ffmpeg", OK, str(ffmpeg["path"]))
    elif ffmpeg["found"]:
        check("ffmpeg", "ffmpeg", WARN, f"系统里有 {ffmpeg['path']}，但 imageio-ffmpeg 不可用：导入 m4a 等格式会失败")
    else:
        check("ffmpeg", "ffmpeg", WARN, "没有找到：导入 wav / flac / ogg / mp3 以外的格式会失败（uv pip install imageio-ffmpeg）")

    # --- service
    pid: int | None = None
    try:
        from ..service import running_pid

        pid = running_pid(cfg)
    except Exception as e:  # noqa: BLE001
        log.debug("cannot tell whether the service runs: %s", e)
    service = {"running": bool(pid), "pid": pid}
    check("service", "服务", OK, f"运行中 (pid {pid})" if pid else "未运行")

    # --- log tail
    from ..logging_setup import log_file_name

    files: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    secrets = _secrets(cfg)
    for role in ("service", "cli"):
        p = data_dir / log_file_name(cfg, role)
        files.append({"role": role, "path": str(p), "exists": p.is_file(), "size_bytes": _file_size(p)})
        errors.extend(_log_errors(p, role, secrets))
    errors.sort(key=lambda e: e["time"])
    day_ago = (now - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
    recent = sum(1 for e in errors if e["time"] >= day_ago)
    errors = errors[-LOG_ERRORS_SHOWN:]
    if recent:
        check("logs", "日志", WARN, f"最近 24 小时有 {recent} 条错误（最后一条 {errors[-1]['time']}）")
    elif errors:
        check("logs", "日志", OK, f"最近 24 小时没有错误（上一条在 {errors[-1]['time']}）")
    else:
        check("logs", "日志", OK, "日志末尾没有错误")

    return {
        "ok": not any(c["status"] == FAIL for c in checks),
        "version": getattr(audio_note, "__version__", None),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "time": now.isoformat(timespec="seconds"),
        "checks": checks,
        "config": config,
        "data_dir": data,
        "database": database,
        "segments": segments,
        "models": models,
        "api_key": api_key,
        "ffmpeg": ffmpeg,
        "service": service,
        "logs": {"files": files, "errors": errors},
    }


def cmd_doctor(cfg: Config, args: argparse.Namespace) -> int:
    """Read-only health report; exit code 1 when a check failed (warnings alone keep it 0)."""
    from rich.markup import escape

    r = doctor_report(cfg)
    if wants_json(args):
        emit_json(r)
        return 0 if r["ok"] else 1
    t = Table(title="audio-note 体检" + (f"（版本 {r['version']}）" if r["version"] else ""))
    t.add_column("检查", no_wrap=True)
    t.add_column("状态", no_wrap=True)
    t.add_column("说明", overflow="fold")           # long paths are folded onto the next line, never cut with "…"
    for c in r["checks"]:
        t.add_row(c["label"], STATUS_TEXT[c["status"]], escape(c["detail"]))
    console.print(t)
    if r["logs"]["errors"]:
        console.print("日志里最近的错误：")
        for e in r["logs"]["errors"]:
            console.print(escape(f"  {e['time']} [{e['source']}] {e['message']}"))
    failed = sum(1 for c in r["checks"] if c["status"] == FAIL)
    warned = sum(1 for c in r["checks"] if c["status"] == WARN)
    if failed:
        console.print(f"[red]{failed} 项有问题[/red]" + (f"，{warned} 项需要注意。" if warned else "。"))
    elif warned:
        console.print(f"[yellow]{warned} 项需要注意[/yellow]，没有妨碍使用的问题。")
    else:
        console.print("[green]没有发现问题。[/green]")
    return 0 if r["ok"] else 1


def register(sub: Any) -> None:
    sub.add_parser("init", help="生成 config.yaml 并检查模型文件").set_defaults(func=handler(cmd_init))
    s = sub.add_parser("doctor", help="体检：配置、数据目录、数据库版本、模型、API key、ffmpeg、卡住的录音段、最近的错误日志（只读）")
    # read_only: main() must not create the data dir / a log file before the report has looked at them
    add_json_flag(s).set_defaults(func=handler(cmd_doctor), read_only=True)
