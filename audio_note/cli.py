"""Command-line entry point: `audio-note <cmd>` (also `python -m audio_note`).

Commands
  devices                         list audio input devices
  record                          record only (VAD-gated segments)
  run [--tray]                    record + scheduler (process / cleanup / daily)
  status                          counts, today's utterances, disk usage, last daily, speakers
  import <files...> [--start]     import existing recordings
  process [--retry-failed] [--limit N]
  transcribe <file>               one-off transcription, printed, not stored
  speakers enroll|list|label|relabel|calibrate
  daily [--day] [--push] [--force]
  push-test <text>
  chat [question...]              one question, or an interactive REPL without arguments
  cleanup
  init                            create config.yaml, check model files

Modules written by other parts of the project (speech.pipeline, speech.correct, memory.chat,
analysis.daily, push.base) are imported lazily inside the command functions, so a missing one
only breaks that command with a clear message.
"""
from __future__ import annotations

import argparse
import glob
import inspect
import logging
import os
import shutil
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
from rich.console import Console
from rich.table import Table

from .config import (DEFAULT_PROFILE, PROJECT_ROOT, Config, create_profile, delete_profile,
                     list_profiles, load_config, profile_config_path, set_current_profile)
from .db import DB

log = logging.getLogger(__name__)
console = Console()


class CLIError(Exception):
    """User-facing error: printed in red, exit code 2."""


# --- helpers --------------------------------------------------------------

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
    from .llm.client import MimoClient, MissingAPIKey

    try:
        return MimoClient(cfg, rpm=int(cfg.mimo.rpm))
    except MissingAPIKey as e:
        raise CLIError(f"{e}\n请在 config.yaml 的 mimo.api_key 填写 MiMo API key，或设置环境变量 MIMO_API_KEY") from e


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
        from .memory import index as mem_index

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
    """Pretty-print whatever a sibling module returns (str / dataclass / dict / list)."""
    from rich.markdown import Markdown

    if obj is None:
        return
    if isinstance(obj, str):
        console.print(Markdown(obj) if obj.lstrip().startswith("#") else obj)
        return
    if isinstance(obj, tuple) and obj and isinstance(obj[0], str):
        _print_result(obj[0])
        return
    text = getattr(obj, "text", None)
    if isinstance(text, str):
        console.print(text)
        return
    if isinstance(obj, dict):
        if "text" in obj and isinstance(obj["text"], str):
            console.print(obj["text"])
            return
        for k, v in obj.items():
            console.print(f"[bold]{k}[/bold]: {v}")
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            if isinstance(item, dict):
                t = item.get("text") or item.get("text_raw") or ""
                spk = item.get("speaker") or item.get("speaker_name") or ""
                start = item.get("start") if "start" in item else item.get("start_s")
                prefix = f"[{start:7.1f}s] " if isinstance(start, (int, float)) else ""
                console.print(f"{prefix}[bold]{spk}[/bold]{'：' if spk else ''}{t}")
            else:
                t = getattr(item, "text", None)
                spk = getattr(item, "speaker_name", None) or getattr(item, "speaker", None) or ""
                start = getattr(item, "start", None)
                if start is None:
                    start = getattr(item, "start_s", None)
                if t is not None:
                    prefix = f"[{start:7.1f}s] " if isinstance(start, (int, float)) else ""
                    console.print(f"{prefix}[bold]{spk}[/bold]{'：' if spk else ''}{t}")
                else:
                    console.print(str(item))
        return
    console.print(str(obj))


# --- pure helpers used by commands (unit-testable) ----------------------------

def calibrate_threshold(embeddings: list[tuple[int, str, np.ndarray]], lo: float = 0.3, hi: float = 0.7) -> dict[str, Any]:
    """Cosine scores between enrolled embeddings: same-speaker vs cross-speaker distributions and a
    suggested identify_threshold = midpoint(max cross, min same) clamped to [lo, hi]."""
    embs = []
    for sid, name, e in embeddings:
        e = np.asarray(e, dtype=np.float32)
        n = float(np.linalg.norm(e))
        embs.append((int(sid), name, e / n if n > 0 else e))
    same: list[float] = []
    cross: list[float] = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            s = float(np.dot(embs[i][2], embs[j][2]))
            (same if embs[i][0] == embs[j][0] else cross).append(s)

    def stats(xs: list[float]) -> dict[str, float] | None:
        if not xs:
            return None
        a = np.asarray(xs)
        return {"n": int(a.size), "min": float(a.min()), "mean": float(a.mean()), "max": float(a.max())}

    out: dict[str, Any] = {
        "n_embeddings": len(embs),
        "n_speakers": len({e[0] for e in embs}),
        "same": stats(same),
        "cross": stats(cross),
        "suggested": None,
        "overlap": False,
    }
    if same and cross:
        mid = (max(cross) + min(same)) / 2.0
        out["overlap"] = max(cross) >= min(same)
        out["suggested"] = float(min(hi, max(lo, mid)))
    elif same:
        out["suggested"] = float(min(hi, max(lo, min(same) - 0.05)))
    elif cross:
        out["suggested"] = float(min(hi, max(lo, max(cross) + 0.05)))
    return out


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


# --- commands -------------------------------------------------------------------

def cmd_devices(cfg: Config, args: argparse.Namespace) -> int:
    from .audio.capture import list_input_devices

    devices = list_input_devices()
    t = Table(title="音频输入设备")
    t.add_column("#", justify="right")
    t.add_column("名称")
    t.add_column("Host API")
    t.add_column("声道", justify="right")
    t.add_column("默认采样率", justify="right")
    for d in devices:
        t.add_row(str(d.index), d.name, d.hostapi, str(d.channels), f"{d.default_sr:.0f}")
    console.print(t)
    console.print(f"配置 audio.device = {cfg.audio.device!r}（null = 系统默认）")
    return 0


def _wait_with_ctrl_c(stop_event: threading.Event, threads: list[threading.Thread]) -> None:
    try:
        while not stop_event.is_set() and any(t.is_alive() for t in threads):
            stop_event.wait(0.5)
    except KeyboardInterrupt:
        console.print("\n正在停止…")
    stop_event.set()
    for t in threads:
        t.join(15.0)


def cmd_record(cfg: Config, args: argparse.Namespace) -> int:
    from .audio.recorder import Recorder

    db = _db(cfg)
    stop_event = threading.Event()
    rec = Recorder(cfg, db, stop_event)
    errors: list[BaseException] = []

    def target() -> None:
        try:
            rec.run()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
            log.exception("recorder failed: %s", e)
            stop_event.set()

    t = threading.Thread(target=target, name="recorder", daemon=True)
    t.start()
    console.print("录音中（只保存检测到语音的片段），Ctrl+C 停止。")
    _wait_with_ctrl_c(stop_event, [t])
    console.print(f"本次写入 {rec.segments_written} 段，共 {rec.speech_total_s:.0f}s 语音。")
    return 1 if errors else 0


def cmd_run(cfg: Config, args: argparse.Namespace) -> int:
    from .service import run_service

    if args.port:
        cfg["ui"]["port"] = int(args.port)
    return run_service(cfg, tray=bool(args.tray), ui=False if args.no_ui else None, open_browser=True if args.open else None)


def cmd_ui(cfg: Config, args: argparse.Namespace) -> int:
    """Web UI without recording (recording can be started from the page). When another process already
    runs the service, the page works read-only (browse + chat)."""
    from .runtime import Runtime, RuntimeHolder, profile_label
    from .service import AlreadyRunning, _install_signal_handlers, start_ui

    if args.port:
        cfg["ui"]["port"] = int(args.port)
    rt = Runtime(cfg)
    try:
        rt.start_service(record=bool(args.record))
    except AlreadyRunning as e:
        console.print(f"[yellow]{e}[/yellow]\n界面以只读模式启动：可以看记录和问答，录音和处理请在那个进程的界面操作。")
    holder = RuntimeHolder(rt)
    srv = start_ui(holder, open_browser=not args.no_browser)
    if srv is None:
        rt.shutdown()
        raise CLIError("界面启动失败（端口被占用？看上面的日志）")
    console.print(f"界面地址 [bold]{srv.url}[/bold]  档案：{profile_label(rt.profile)}"
                  f"  （Ctrl+C 退出{'' if args.record else '；录音可在页面上开始'}）")
    stop = threading.Event()
    _install_signal_handlers(stop)
    try:
        while not stop.is_set():
            stop.wait(0.5)
    except KeyboardInterrupt:
        console.print("\n正在停止…")
    srv.stop()
    holder.rt.shutdown()
    return 0


def cmd_profile_list(cfg: Config, args: argparse.Namespace) -> int:
    from .runtime import profile_label

    cur = cfg.get("_profile")
    t = Table(title="档案（每个档案有自己的录音、记录、说话人；API key 等共享设置放根目录 config.yaml）")
    for col in ("", "档案", "数据目录", "记录"):
        t.add_column(col)
    for name in list_profiles():
        try:
            c = load_config(profile=name)
            n = _profile_counts(c)
            info = f"{n['utterances']} 条 / {n['days']} 天" if n else "（还没有数据）"
            data_dir = str(c.data_dir)
        except Exception as e:  # noqa: BLE001
            info, data_dir = f"[red]{e}[/red]", "?"
        t.add_row("●" if name == cur else "", profile_label(name), data_dir, info)
    console.print(t)
    if cur is None:
        console.print(f"当前使用的是自定义配置 {cfg.get('_config_path')}（--config / AUDIO_NOTE_CONFIG），不属于任何档案。")
    else:
        console.print("● = 当前档案。切换：audio-note profile use <名字>；一次性：audio-note --profile <名字> <命令>")
    return 0


def _profile_counts(c: Config) -> dict[str, int] | None:
    """Row counts of a profile's database without creating it (None when the db does not exist)."""
    import sqlite3

    if not c.db_path.exists():
        return None
    con = sqlite3.connect(f"file:{c.db_path.as_posix()}?mode=ro", uri=True, timeout=5)
    try:
        row = con.execute("SELECT COUNT(*), COUNT(DISTINCT day), MAX(day) FROM utterances").fetchone()
        spk = con.execute("SELECT COUNT(*) FROM speakers").fetchone()
        return {"utterances": int(row[0] or 0), "days": int(row[1] or 0), "speakers": int(spk[0] or 0), "last_day": row[2]}
    except sqlite3.Error:
        return None
    finally:
        con.close()


def cmd_profile_create(cfg: Config, args: argparse.Namespace) -> int:
    src = None
    if args.copy_from:
        src = profile_config_path(args.copy_from)
        if not src.exists():
            raise CLIError(f"档案不存在: {args.copy_from}")
    try:
        path = create_profile(args.name, copy_from=src)
    except (ValueError, FileExistsError, OSError) as e:
        raise CLIError(str(e)) from e
    console.print(f"已创建档案 [bold]{args.name}[/bold]：{path}")
    if args.use:
        set_current_profile(args.name)
        console.print(f"已切换到 {args.name}。")
    else:
        console.print(f"切换：audio-note profile use {args.name}")
    return 0


def cmd_profile_use(cfg: Config, args: argparse.Namespace) -> int:
    try:
        set_current_profile(args.name)
    except ValueError as e:
        raise CLIError(str(e)) from e
    c = load_config(profile=args.name)
    console.print(f"当前档案：[bold]{args.name}[/bold]（数据目录 {c.data_dir}）。正在运行的服务 / 界面需要重启或在页面上切换。")
    return 0


def cmd_profile_delete(cfg: Config, args: argparse.Namespace) -> int:
    from .service import running_pid

    if args.name == DEFAULT_PROFILE:
        raise CLIError("默认档案不能删除")
    try:
        c = load_config(profile=args.name)
    except FileNotFoundError as e:
        raise CLIError(str(e)) from e
    pid = running_pid(c)
    if pid:
        raise CLIError(f"档案 {args.name} 的服务正在运行 (pid {pid})，先停止再删除")
    if not args.yes:
        n = _profile_counts(c)
        console.print(f"将删除档案 [bold]{args.name}[/bold] 的全部数据：{c.data_dir}"
                      + (f"（{n['utterances']} 条记录 / {n['days']} 天 / {n['speakers']} 个说话人）" if n else ""))
        try:
            ans = console.input("确认请输入档案名: ").strip()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        if ans != args.name:
            console.print("已取消。")
            return 1
    try:
        delete_profile(args.name)
    except (ValueError, FileNotFoundError, OSError) as e:
        raise CLIError(str(e)) from e
    console.print(f"已删除档案 {args.name}。")
    return 0


def cmd_status(cfg: Config, args: argparse.Namespace) -> int:
    from .audio.storage import dir_size_bytes
    from .service import running_pid

    db = _db(cfg)
    today = datetime.now().strftime("%Y-%m-%d")

    t = Table(title="audio-note 状态", show_header=False)
    t.add_column("项")
    t.add_column("值")
    from .runtime import profile_label

    t.add_row("档案", profile_label(cfg.get("_profile")))
    t.add_row("配置文件", str(cfg.get("_config_path", "")))
    t.add_row("数据目录", str(cfg.data_dir))
    pid = running_pid(cfg)
    t.add_row("服务", f"运行中 (pid {pid})" if pid else "未运行")
    t.add_row("MiMo API key", "已配置" if cfg.mimo.api_key else "[red]未配置[/red]")

    rows = db.query("SELECT status, COUNT(*) AS n, COALESCE(SUM(duration_s),0) AS dur FROM segments GROUP BY status ORDER BY status")
    if rows:
        t.add_row("录音段", "  ".join(f"{r['status']}={r['n']} ({r['dur'] / 60:.0f} min)" for r in rows))
    else:
        t.add_row("录音段", "0")
    utt_today = db.one("SELECT COUNT(*) AS n FROM utterances WHERE day=?", (today,))
    utt_all = db.one("SELECT COUNT(*) AS n, COUNT(DISTINCT day) AS days FROM utterances")
    t.add_row("今日语句", str(utt_today["n"] if utt_today else 0))
    t.add_row("语句总数", f"{utt_all['n']} 条 / {utt_all['days']} 天" if utt_all else "0")
    size = dir_size_bytes(cfg.audio_dir) if cfg.audio_dir.exists() else 0
    cap = float(cfg.storage.max_disk_gb or 0)
    t.add_row("音频占用", f"{_human_bytes(size)}" + (f" / 上限 {cap:g} GB" if cap else ""))
    last = db.one("SELECT day, created_at, pushed_at, utterance_count FROM daily_summaries ORDER BY day DESC LIMIT 1")
    if last:
        t.add_row("最近日报", f"{last['day']}（生成 {last['created_at']}，推送 {last['pushed_at'] or '未推送'}）")
    else:
        t.add_row("最近日报", "无")
    kv_day = db.get_kv("last_daily_day")
    if kv_day:
        t.add_row("调度 last_daily_day", str(kv_day))
    failed = db.one("SELECT COUNT(*) AS n FROM segments WHERE status='failed'")
    if failed and failed["n"]:
        t.add_row("失败段", f"[yellow]{failed['n']}[/yellow]（audio-note process --retry-failed 重试）")
    console.print(t)

    spk = db.query(
        "SELECT s.id, s.name, s.notes, COUNT(e.id) AS n, COALESCE(SUM(e.duration_s),0) AS dur"
        " FROM speakers s LEFT JOIN speaker_embeddings e ON e.speaker_id=s.id GROUP BY s.id ORDER BY s.name"
    )
    st = Table(title="已注册说话人")
    st.add_column("id", justify="right")
    st.add_column("姓名")
    st.add_column("样本数", justify="right")
    st.add_column("语音时长", justify="right")
    st.add_column("备注")
    for r in spk:
        st.add_row(str(r["id"]), r["name"], str(r["n"]), f"{r['dur']:.0f}s", r["notes"] or "")
    if spk:
        console.print(st)
    else:
        console.print("未注册说话人（audio-note speakers enroll <姓名> <wav...>）")
    return 0


def cmd_import(cfg: Config, args: argparse.Namespace) -> int:
    from .audio.importer import import_file
    from .audio.vad import SileroVAD

    files = _expand_files(args.files)
    start = _parse_datetime(args.start) if args.start else None
    if start is not None and len(files) > 1:
        console.print("[yellow]--start 指定了多个文件的同一起始时间；后续文件请分别导入。[/yellow]")
    db = _db(cfg)
    vad = SileroVAD(cfg.model_path("vad_model"), threads=2)
    total = 0
    for f in files:
        try:
            ids = import_file(cfg, db, f, start_time=start, vad=vad)
        except Exception as e:  # noqa: BLE001
            log.exception("import %s failed", f)
            console.print(f"[red]导入失败[/red] {f}: {e}")
            continue
        total += len(ids)
        console.print(f"{f.name}: {len(ids)} 段")
    console.print(f"共导入 {total} 段，运行 audio-note process 进行转写。")
    return 0


def cmd_process(cfg: Config, args: argparse.Namespace) -> int:
    Processor = _lazy("speech.pipeline", "Processor")
    db = _db(cfg)
    client = _client(cfg)
    proc = Processor(cfg, db, client)
    run = None
    for name in ("run", "run_once", "process_pending", "process_all"):
        fn = getattr(proc, name, None)
        if callable(fn):
            run = fn
            break
    if run is None:
        raise CLIError("Processor 没有 run()/run_once()/process_pending() 方法")
    result = _invoke(run, retry_failed=bool(args.retry_failed), limit=args.limit)
    _print_result(result)
    usage = getattr(client, "usage", None)
    if usage:
        console.print(f"API: {usage.get('calls', 0)} 次调用, 音频 {usage.get('audio_seconds', 0):.0f}s, "
                      f"tokens {usage.get('prompt_tokens', 0)}+{usage.get('completion_tokens', 0)}")
    return 0


def cmd_transcribe(cfg: Config, args: argparse.Namespace) -> int:
    transcribe_file = _lazy("speech.pipeline", "transcribe_file")
    path = Path(args.file)
    if not path.exists():
        raise CLIError(f"文件不存在: {path}")
    client = _client(cfg)
    db = _db(cfg)
    result = _invoke(transcribe_file, cfg=cfg, config=cfg, client=client, llm=client, db=db,
                     path=path, file=path, file_path=path, src=path, source=path, audio_path=path)
    _print_result(result)
    return 0


def cmd_speakers_enroll(cfg: Config, args: argparse.Namespace) -> int:
    from .speech.speakers import SpeakerEngine

    files = _expand_files(args.files)
    db = _db(cfg)
    engine = SpeakerEngine(cfg)
    sid = engine.enroll(db, args.name, files, notes=args.notes)
    n = db.one("SELECT COUNT(*) AS n FROM speaker_embeddings WHERE speaker_id=?", (sid,))
    console.print(f"已注册 {args.name} (id {sid})，共 {n['n'] if n else 0} 个样本（副本保存在 {cfg.speakers_dir}）。")
    _reindex_after_enroll(cfg, db)
    done = db.one("SELECT COUNT(*) AS n FROM segments WHERE status='done'")
    if done and done["n"]:
        console.print("已转写的记录可用 audio-note speakers relabel [--day YYYY-MM-DD] 按新声纹重新识别（不重复转写）。")
    return 0


def cmd_speakers_record(cfg: Config, args: argparse.Namespace) -> int:
    """Record an enrollment sample from the microphone, save it under data/speakers/<name>/, enroll it."""
    import time

    import numpy as np
    import soundfile as sf

    from .audio.capture import MicCapture
    from .audio.vad import SAMPLE_RATE
    from .speech.speakers import SpeakerEngine

    seconds = max(5, int(args.seconds))
    out_dir = cfg.speakers_dir / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    console.print(f"将录制 {seconds} 秒作为 [bold]{args.name}[/bold] 的声纹样本，请用平时说话的音量自然地说话（念一段文字即可）。")
    for i in (3, 2, 1):
        console.print(f"  {i} ...")
        time.sleep(1)
    console.print("[green]开始录音[/green]")
    blocks: list[np.ndarray] = []
    got = 0
    with MicCapture(cfg.audio.device, int(cfg.audio.block_ms), float(cfg.audio.gain)) as mic:
        t0 = time.time()
        last_tick = 0
        while got < seconds * SAMPLE_RATE:
            b = mic.read(timeout=1.0)
            if b is None:
                if time.time() - t0 > seconds + 5:
                    console.print("[red]麦克风没有数据，检查设备（audio-note devices）[/red]")
                    return 2
                continue
            blocks.append(b)
            got += len(b)
            tick = int(got / SAMPLE_RATE)
            if tick != last_tick and tick % 5 == 0:
                last_tick = tick
                console.print(f"  已录 {tick}s")
    audio = np.concatenate(blocks)[: seconds * SAMPLE_RATE]
    sf.write(str(out), audio, SAMPLE_RATE, subtype="PCM_16")
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    console.print(f"样本已保存：{out}（峰值 {peak:.2f}）")
    if peak < 0.02:
        console.print("[yellow]音量很低，可能没录到声音；请检查麦克风或调大 audio.gain[/yellow]")
    db = _db(cfg)
    engine = SpeakerEngine(cfg)
    emb, speech_s = engine.embed_speech_only(audio)
    if speech_s < 3.0:
        console.print(f"[red]只检测到 {speech_s:.1f} 秒语音，样本不可用；请重录并持续说话[/red]")
        return 2
    sid = db.upsert_speaker(args.name, args.notes)
    db.add_speaker_embedding(sid, emb, str(out), speech_s)
    n = db.one("SELECT COUNT(*) AS n FROM speaker_embeddings WHERE speaker_id=?", (sid,))
    console.print(f"已注册 {args.name} (id {sid})，语音 {speech_s:.1f}s，共 {n['n'] if n else 0} 个样本。建议在不同场合再录 1–2 次以提高稳定性。")
    _reindex_after_enroll(cfg, db)
    return 0


def cmd_speakers_list(cfg: Config, args: argparse.Namespace) -> int:
    db = _db(cfg)
    rows = db.query(
        "SELECT s.id, s.name, s.notes, s.created_at, COUNT(e.id) AS n, COALESCE(SUM(e.duration_s),0) AS dur"
        " FROM speakers s LEFT JOIN speaker_embeddings e ON e.speaker_id=s.id GROUP BY s.id ORDER BY s.name"
    )
    t = Table(title="已注册说话人")
    for col in ("id", "姓名", "样本数", "语音时长", "注册时间", "备注"):
        t.add_column(col)
    for r in rows:
        t.add_row(str(r["id"]), r["name"], str(r["n"]), f"{r['dur']:.0f}s", r["created_at"], r["notes"] or "")
    console.print(t)
    if not rows:
        console.print("（空）")
    return 0


def resolve_label_target(db: DB, at: datetime, dur: float | None = None, tolerance_s: float = 60.0) -> tuple[Any, float, float]:
    """Find the utterance spoken at wall-clock time `at` (the timestamp shown in the transcript) and
    return (segment row, from_s, to_s) inside that segment's file. The utterance that contains `at`
    wins; otherwise the nearest one within tolerance_s. `dur` overrides the range length."""
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
    seg = db.get_segment(int(best["segment_id"]))
    if seg is None:
        raise CLIError(f"语句所属的录音段 #{best['segment_id']} 已不存在")
    a = float(best["start_s"])
    b = a + float(dur) if dur else float(best["end_s"])
    return seg, a, b


def _relabel_segment_quietly(cfg: Config, db: DB, engine: Any, seg_id: int) -> None:
    """After a label, re-identify that segment's utterances so the transcript shows the new name."""
    try:
        from .speech.pipeline import Processor

        changed = Processor(cfg, db, engine=engine).relabel_segment(seg_id)
        console.print(f"已按新声纹重新识别录音段 #{seg_id}：{changed} 条语句的说话人有变化。")
    except Exception as e:  # noqa: BLE001
        log.warning("relabel after label failed: %s", e)


def cmd_speakers_label(cfg: Config, args: argparse.Namespace) -> int:
    from .speech.speakers import SpeakerEngine

    db = _db(cfg)
    seg_id: int | None = None
    if args.at:
        if args.segment:
            raise CLIError("--at 与 <segment_id|path> 二选一")
        seg, a, b = resolve_label_target(db, _parse_datetime(args.at), args.dur)
        seg_id = int(seg["id"])
        path = Path(seg["path"])
    else:
        if not args.segment:
            raise CLIError("请给出 <segment_id|path>（并用 --from/--to 指定秒数），或者用 --at \"YYYY-MM-DD HH:MM:SS\"")
        if args.start is None or args.end is None:
            raise CLIError("使用 <segment_id|path> 时必须给出 --from 和 --to（秒）")
        ref = str(args.segment)
        if ref.isdigit():
            row = db.get_segment(int(ref))
            if row is None:
                raise CLIError(f"没有 id={ref} 的录音段")
            seg_id = int(row["id"])
            path = Path(row["path"])
        else:
            path = Path(ref)
            row = db.one("SELECT id FROM segments WHERE path=?", (str(path),))
            seg_id = int(row["id"]) if row else None
        a, b = float(args.start), float(args.end)
    if not path.exists():
        raise CLIError(f"音频文件不存在（可能已过期删除）: {path}")
    if b <= a:
        raise CLIError("--to 必须大于 --from")
    engine = SpeakerEngine(cfg)
    try:
        sid = engine.enroll_from_segment_turns(db, args.name, path, [(a, b)])
    except ValueError as e:
        raise CLIError(str(e)) from e
    console.print(f"已把 {path.name} 的 {a:.1f}s–{b:.1f}s 标记为 {args.name} (id {sid})。")
    _reindex_after_enroll(cfg, db)
    if seg_id is not None:
        _relabel_segment_quietly(cfg, db, engine, seg_id)
    console.print("历史记录可用 audio-note speakers relabel [--day YYYY-MM-DD] 按新声纹重新识别（不重复转写）。")
    return 0


def cmd_speakers_relabel(cfg: Config, args: argparse.Namespace) -> int:
    """Re-identify speakers on already transcribed segments (no ASR, no cost)."""
    from .speech.speakers import SpeakerEngine

    Processor = _lazy("speech.pipeline", "Processor")
    db = _db(cfg)
    if not db.speaker_embeddings():
        raise CLIError("还没有注册任何说话人（先 speakers enroll 或 speakers label）")
    day = _parse_day(args.day) if args.day else None
    proc = Processor(cfg, db, engine=SpeakerEngine(cfg))
    stats = proc.relabel(day=day, segment_id=args.segment)
    console.print(f"重新识别 {stats['segments']} 段：{stats['changed']} 条语句的说话人有变化，{stats['skipped']} 段跳过（音频已删除或失败）。")
    return 0


def cmd_speakers_calibrate(cfg: Config, args: argparse.Namespace) -> int:
    db = _db(cfg)
    embs = db.speaker_embeddings()
    if len(embs) < 2:
        raise CLIError("至少需要 2 个注册样本才能校准（建议每人 ≥ 3 个样本、≥ 2 个人）")
    r = calibrate_threshold(embs)
    t = Table(title="说话人相似度分布（cosine）")
    for col in ("类型", "对数", "最小", "平均", "最大"):
        t.add_column(col, justify="right")
    for label, key in (("同一人", "same"), ("不同人", "cross")):
        s = r[key]
        if s:
            t.add_row(label, str(s["n"]), f"{s['min']:.3f}", f"{s['mean']:.3f}", f"{s['max']:.3f}")
        else:
            t.add_row(label, "0", "-", "-", "-")
    console.print(t)
    console.print(f"样本 {r['n_embeddings']} 个 / 说话人 {r['n_speakers']} 人；当前阈值 identify_threshold = {cfg.speakers.identify_threshold}")
    if r["same"] is None:
        console.print("[yellow]没有同一人的多个样本，无法估计同人分布；请每人至少注册 2 个样本。[/yellow]")
    if r["cross"] is None:
        console.print("[yellow]只有一个说话人，无法估计不同人分布；请再注册一个人。[/yellow]")
    if r["overlap"]:
        console.print("[yellow]同人与异人分数区间重叠，识别会有误判；建议增加样本或检查录音质量。[/yellow]")
    if r["suggested"] is not None:
        console.print(f"建议阈值: [bold]{r['suggested']:.2f}[/bold]（写入 config.yaml 的 speakers.identify_threshold）")
    return 0


def cmd_daily(cfg: Config, args: argparse.Namespace) -> int:
    from rich.markdown import Markdown

    day = _parse_day(args.day)
    db = _db(cfg)
    existing = db.one("SELECT summary_md, created_at, pushed_at FROM daily_summaries WHERE day=?", (day,))
    DailyAnalyzer = _lazy("analysis.daily", "DailyAnalyzer")
    analyzer = None
    if existing and not args.force:
        console.print(f"{day} 的日报已存在（{existing['created_at']}），--force 可重新生成。")
        console.print(Markdown(existing["summary_md"]))
    else:
        client = _client(cfg)
        analyzer = DailyAnalyzer(cfg, db, client)
        result = _invoke(analyzer.run, day=day, force=bool(args.force))
        md = result[0] if isinstance(result, tuple) else result
        if isinstance(md, str):
            console.print(Markdown(md))
        else:
            _print_result(result)
    if args.push:
        if analyzer is None:
            analyzer = DailyAnalyzer(cfg, db, _client(cfg))
        analyzer.push(day)
        console.print(f"已推送 {day} 的日报。")
    return 0


def cmd_push_test(cfg: Config, args: argparse.Namespace) -> int:
    Notifier = _lazy("push.base", "Notifier")
    Message = _lazy("push.base", "Message")
    text = " ".join(args.text) if isinstance(args.text, list) else str(args.text)
    msg = Message("audio-note 测试", text)
    results = Notifier(cfg).send(msg)
    ok = True
    if isinstance(results, dict):
        for ch, err in results.items():
            if err:
                ok = False
                console.print(f"[red]{ch}[/red]: {err}")
            else:
                console.print(f"[green]{ch}[/green]: 已发送")
    else:
        console.print(str(results))
    return 0 if ok else 1


def cmd_chat(cfg: Config, args: argparse.Namespace) -> int:
    MemoryChat = _lazy("memory.chat", "MemoryChat")
    Retriever = _lazy("memory.retrieval", "Retriever")
    db = _db(cfg)
    client = _client(cfg)
    retriever = Retriever(cfg, db, client, mode=args.retrieval, expand=args.expand)
    chat = MemoryChat(cfg, db, client, retriever=retriever, strategy=args.strategy)
    question = " ".join(args.question).strip() if args.question else ""
    if question:
        answer = _invoke(chat.ask, question=question, history=[])
        _print_result(answer)
        return 0
    chat.repl()
    return 0


def cmd_reindex(cfg: Config, args: argparse.Namespace) -> int:
    """Rebuild the keyword index (after glossary / speaker changes) and embed utterances without a vector."""
    mem_index = _lazy_module("memory.index")
    db = _db(cfg)
    mem_index.register_from(cfg, db)
    n = mem_index.reindex(db)
    console.print(f"关键词索引已重建：{n} 条。")
    if args.no_vectors or cfg.memory.retrieval == "fts":
        console.print("向量索引跳过（memory.retrieval = fts 或 --no-vectors）。")
        return 0
    vectors = _lazy_module("memory.vectors")
    get_embedder = _lazy("memory.embed", "get_embedder")
    emb = get_embedder(cfg)
    missing = vectors.missing_count(db, emb.name)
    if not missing:
        console.print(f"向量索引已是最新（模型 {emb.name}，{db.vector_count(emb.name)} 条）。")
        return 0
    console.print(f"正在用 {emb.name} 嵌入 {missing} 条记录…")
    k = vectors.index_missing(db, emb)
    console.print(f"向量索引完成：新增 {k} 条，共 {db.vector_count(emb.name)} 条。")
    return 0


def cmd_cleanup(cfg: Config, args: argparse.Namespace) -> int:
    from .audio.storage import cleanup
    from .audio.vad import SileroVAD

    db = _db(cfg)
    vad = None
    try:
        vad = SileroVAD(cfg.model_path("vad_model"), threads=2)
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]VAD 不可用，跳过语音复检: {e}[/yellow]")
    stats = cleanup(cfg, db, vad)
    for k, v in stats.items():
        console.print(f"{k}: {v}")
    return 0


def cmd_init(cfg: Config, args: argparse.Namespace) -> int:
    """Create the root config.yaml (the file every profile inherits from) unless --config names another."""
    path = Path(args.config) if args.config else PROJECT_ROOT / "config.yaml"
    if path.exists():
        console.print(f"配置文件已存在: {path}（不覆盖）")
    else:
        example = PROJECT_ROOT / "config.example.yaml"
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
    for key, what in (("vad_model", "VAD (Silero)"), ("embedding_model", "说话人 embedding (CAM++)"),
                      ("segmentation_model", "说话人分离 (pyannote)")):
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


# --- argparse ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="audio-note", description="全天无感录音 → 说话人识别 → MiMo 转写 → 日报与记忆问答")
    p.add_argument("--config", metavar="PATH", help="config.yaml 路径（默认 $AUDIO_NOTE_CONFIG 或项目根目录 config.yaml）")
    p.add_argument("--profile", metavar="NAME", help="用某个档案（profiles/<NAME>/）运行这一条命令；默认用 profile use 选的档案")
    p.add_argument("--verbose", "-v", action="store_true", help="DEBUG 日志")
    sub = p.add_subparsers(dest="cmd", metavar="<cmd>")
    sub.required = True

    sub.add_parser("devices", help="列出音频输入设备").set_defaults(func=cmd_devices)
    sub.add_parser("record", help="只录音（VAD 门控，只存语音段）").set_defaults(func=cmd_record)
    s = sub.add_parser("run", help="常驻服务：录音 + 定时处理/清理/日报 + 网页界面")
    s.add_argument("--tray", action="store_true", help="显示系统托盘图标")
    s.add_argument("--no-ui", action="store_true", help="不提供网页界面")
    s.add_argument("--open", action="store_true", help="启动后在浏览器打开界面")
    s.add_argument("--port", type=int, default=None, help="界面端口（默认 ui.port = 8765）")
    s.set_defaults(func=cmd_run)
    s = sub.add_parser("ui", help="打开网页界面（不自动录音；录音可在页面上开始）")
    s.add_argument("--record", action="store_true", help="启动时就开始录音（等于 run --open）")
    s.add_argument("--port", type=int, default=None, help="界面端口（默认 ui.port = 8765）")
    s.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    s.set_defaults(func=cmd_ui)
    sub.add_parser("status", help="状态总览").set_defaults(func=cmd_status)

    s = sub.add_parser("import", help="导入已有录音文件")
    s.add_argument("files", nargs="+", help="音频文件或通配符")
    s.add_argument("--start", help='录音开始时间，如 "2026-10-31 09:00"（默认从文件名/修改时间推断）')
    s.set_defaults(func=cmd_import)

    s = sub.add_parser("process", help="处理待转写的录音段")
    s.add_argument("--retry-failed", action="store_true", help="重试 failed 状态的段")
    s.add_argument("--limit", type=int, default=None, help="最多处理 N 段")
    s.set_defaults(func=cmd_process)

    s = sub.add_parser("transcribe", help="一次性转写一个文件并打印（不入库）")
    s.add_argument("file")
    s.set_defaults(func=cmd_transcribe)

    sp = sub.add_parser("speakers", help="说话人注册 / 列表 / 标注 / 校准")
    ssub = sp.add_subparsers(dest="speakers_cmd", metavar="<sub>")
    ssub.required = True
    s = ssub.add_parser("enroll", help="用音频样本注册说话人")
    s.add_argument("name")
    s.add_argument("files", nargs="+", help="wav/mp3/... 样本文件")
    s.add_argument("--notes", default=None)
    s.set_defaults(func=cmd_speakers_enroll)
    s = ssub.add_parser("record", help="用麦克风现场录一段样本并注册说话人（注册自己最方便）")
    s.add_argument("name")
    s.add_argument("--seconds", type=int, default=20, help="录制秒数（默认 20，至少 5）")
    s.add_argument("--notes", default=None)
    s.set_defaults(func=cmd_speakers_record)
    ssub.add_parser("list", help="列出说话人").set_defaults(func=cmd_speakers_list)
    s = ssub.add_parser("label", help="把某段录音的一个时间范围标记为某人（并重新识别该段）")
    s.add_argument("segment", nargs="?", default=None, help="segment id 或音频文件路径（记录文件的 ## 标题里有 #id 和文件名）")
    s.add_argument("name")
    s.add_argument("--from", dest="start", type=float, default=None, metavar="S", help="起始秒（配合 segment）")
    s.add_argument("--to", dest="end", type=float, default=None, metavar="S", help="结束秒（配合 segment）")
    s.add_argument("--at", default=None, metavar="TIME", help='记录里那句话的时间，如 "2026-10-31 08:19:32"（自动找到对应录音段与区间）')
    s.add_argument("--dur", type=float, default=None, metavar="S", help="配合 --at：从该句开始取多少秒（默认取整句）")
    s.set_defaults(func=cmd_speakers_label)
    s = ssub.add_parser("relabel", help="对已转写的录音段按当前声纹库重新识别说话人（不重复 ASR）")
    s.add_argument("--day", default=None, help="只处理某天 YYYY-MM-DD")
    s.add_argument("--segment", type=int, default=None, help="只处理某个 segment id")
    s.set_defaults(func=cmd_speakers_relabel)
    ssub.add_parser("calibrate", help="根据注册样本估计识别阈值").set_defaults(func=cmd_speakers_calibrate)

    s = sub.add_parser("daily", help="生成（并推送）某天的日报")
    s.add_argument("--day", default=None, help="YYYY-MM-DD，默认今天")
    s.add_argument("--push", action="store_true")
    s.add_argument("--force", action="store_true", help="已存在也重新生成")
    s.set_defaults(func=cmd_daily)

    s = sub.add_parser("push-test", help="向配置的推送渠道发测试消息")
    s.add_argument("text", nargs="+")
    s.set_defaults(func=cmd_push_test)

    s = sub.add_parser("chat", help="记忆问答（无参数进入交互模式）")
    s.add_argument("question", nargs="*")
    s.add_argument("--strategy", choices=["agentic", "prefetch", "rag"], default=None, help="覆盖 memory.strategy")
    s.add_argument("--retrieval", choices=["fts", "vector", "hybrid"], default=None, help="覆盖 memory.retrieval")
    s.add_argument("--expand", dest="expand", action="store_true", default=None, help="开启 LLM 查询扩展")
    s.add_argument("--no-expand", dest="expand", action="store_false", help="关闭 LLM 查询扩展")
    s.set_defaults(func=cmd_chat)

    s = sub.add_parser("reindex", help="重建关键词索引并补齐向量索引")
    s.add_argument("--no-vectors", action="store_true", help="只重建关键词索引")
    s.set_defaults(func=cmd_reindex)

    sub.add_parser("cleanup", help="清理无语音/过期音频").set_defaults(func=cmd_cleanup)
    pp = sub.add_parser("profile", help="档案：给不同的人 / 用途分开的录音、记录和说话人")
    psub = pp.add_subparsers(dest="profile_cmd", metavar="<sub>")
    psub.required = True
    psub.add_parser("list", help="列出档案").set_defaults(func=cmd_profile_list)
    s = psub.add_parser("create", help="新建档案（profiles/<名字>/，API key 等从根目录 config.yaml 继承）")
    s.add_argument("name")
    s.add_argument("--copy-from", default=None, metavar="NAME", help="复制某个档案的设置")
    s.add_argument("--use", action="store_true", help="创建后立即切换过去")
    s.set_defaults(func=cmd_profile_create)
    s = psub.add_parser("use", help="切换当前档案（之后的命令和界面都用它）")
    s.add_argument("name")
    s.set_defaults(func=cmd_profile_use)
    s = psub.add_parser("delete", help="删除档案及其全部数据")
    s.add_argument("name")
    s.add_argument("--yes", "-y", action="store_true", help="不询问")
    s.set_defaults(func=cmd_profile_delete)
    sub.add_parser("init", help="生成 config.yaml 并检查模型文件").set_defaults(func=cmd_init)
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


def main(argv: list[str] | None = None) -> int:
    _utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config, profile=args.profile)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]读取配置失败[/red]: {e}")
        return 2
    try:
        from .logging_setup import setup_logging

        setup_logging(cfg, verbose=bool(args.verbose))
    except Exception as e:  # noqa: BLE001
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
        log.warning("logging setup failed: %s", e)
    try:
        return int(args.func(cfg, args) or 0)
    except CLIError as e:
        console.print(f"[red]错误[/red]: {e}")
        return 2
    except KeyboardInterrupt:
        console.print("\n已中断")
        return 130
    except Exception as e:  # noqa: BLE001
        log.exception("command %s failed", args.cmd)
        console.print(f"[red]{type(e).__name__}[/red]: {e}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
