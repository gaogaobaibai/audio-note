"""Command-line entry point: `audio-note <cmd>` (also `python -m audio_note`).

Commands
  devices                         list audio input devices
  record                          record only (VAD-gated segments)
  run [--tray]                    record + scheduler (process / cleanup / daily)
  status                          counts, today's utterances, disk usage, last daily, speakers
  import <files...> [--start]     import existing recordings
  process [--retry-failed] [--limit N]
  transcribe <file>               one-off transcription, printed, not stored
  speakers enroll|record|list|label|relabel|calibrate|audit   (enroll/label/voices --assign take --force to add a sample that sounds unlike the person)
  speakers voices|scan            unidentified voices in the recordings: list / name a group (no enrollment recording needed)
  speakers set-self|relation|relations   who the user is, relations to the user (by hand / inferred by the LLM)
  daily [--day] [--push] [--force]
  push-test <text>
  chat [question...]              one question, or an interactive REPL without arguments
  cleanup
  loudness [--day D] [--apply --yes]   speech level per sentence; --apply deletes the ones below audio.min_speech_dbfs
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
    from .audio.loopback import list_loopback_devices

    lbs = list_loopback_devices()
    if lbs:
        t2 = Table(title="系统声音源（录会议 / 通话对方的声音：audio.system_audio: true）")
        t2.add_column("#", justify="right")
        t2.add_column("输出设备")
        t2.add_column("声道", justify="right")
        t2.add_column("采样率", justify="right")
        t2.add_column("默认")
        for d in lbs:
            t2.add_row(str(d.index), d.name, str(d.channels), f"{d.default_sr:.0f}", "✓" if d.is_default else "")
        console.print(t2)
        console.print(f"配置 audio.system_audio = {bool(cfg.audio.get('system_audio', False))}，"
                      f"audio.system_device = {cfg.audio.get('system_device')!r}（null = 默认输出设备）")
    else:
        console.print("[yellow]系统声音源不可用（需要 Windows 和 PyAudioWPatch：uv pip install PyAudioWPatch）[/yellow]")
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
    before = db.one("SELECT COUNT(*) AS n FROM speaker_embeddings e JOIN speakers s ON s.id=e.speaker_id WHERE s.name=?", (args.name,))
    n0 = int(before["n"]) if before else 0
    sid = engine.enroll(db, args.name, files, notes=args.notes, **_force_kwargs(args))
    n = db.one("SELECT COUNT(*) AS n FROM speaker_embeddings WHERE speaker_id=?", (sid,))
    n1 = int(n["n"]) if n else 0
    console.print(f"已注册 {args.name} (id {sid})，共 {n1} 个样本（副本保存在 {cfg.speakers_dir}）。")
    skipped = len(files) - (n1 - n0)
    if skipped > 0:
        console.print(f"[yellow]{skipped} 个文件没有加入：语音太短，或和 {args.name} 已有的样本不像（确定是同一个人可加 --force）。[/yellow]")
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
    db.add_speaker_embedding(sid, emb, str(out), speech_s, origin="mic")
    n = db.one("SELECT COUNT(*) AS n FROM speaker_embeddings WHERE speaker_id=?", (sid,))
    console.print(f"已注册 {args.name} (id {sid})，语音 {speech_s:.1f}s，共 {n['n'] if n else 0} 个样本。建议在不同场合再录 1–2 次以提高稳定性。")
    _reindex_after_enroll(cfg, db)
    return 0


def _force_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """{"force": True} only when --force was given (engines without the keyword keep working)."""
    return {"force": True} if getattr(args, "force", False) else {}


SAMPLE_ORIGIN_LABELS = {"mic": "麦克风", "file": "文件", "label": "标记", "voice": "声音组", "cluster": "簇"}


def cmd_speakers_audit(cfg: Config, args: argparse.Namespace) -> int:
    """Table of every enrolled sample with its consistency to the person's other samples; --fix deletes
    the outliers (samples that probably hold somebody else's voice)."""
    from .speech.speakers import audit_samples, sample_consistency_min

    db = _db(cfg)
    min_cos = sample_consistency_min(cfg)
    speakers = db.query("SELECT id, name FROM speakers ORDER BY is_self DESC, name")
    t = Table(title=f"声纹样本一致性（阈值 {min_cos:.2f}）")
    for col in ("人", "样本", "来源", "时长", "一致性", "离群"):
        t.add_column(col)
    outliers: list[tuple[int, str, int]] = []
    ambiguous: list[str] = []
    total = 0
    for sp in speakers:
        rows = audit_samples(db, int(sp["id"]), min_cos)
        if any(a["ambiguous"] for a in rows):
            ambiguous.append(sp["name"])
        for a in rows:
            total += 1
            cons = "-" if a["consistency"] is None else f"{a['consistency']:.2f}"
            flag = "[red]是[/red]" if a["outlier"] else ("[yellow]歧义[/yellow]" if a["ambiguous"] else "")
            if a["outlier"]:
                outliers.append((int(sp["id"]), sp["name"], int(a["id"])))
            src = Path(str(a["source_path"]).split("#")[0]).name if a["source_path"] else "（扫描时保存的声纹）"
            dur = f"{float(a['duration_s']):.1f}s" if a["duration_s"] is not None else "-"
            t.add_row(sp["name"], f"#{a['id']} {src}", SAMPLE_ORIGIN_LABELS.get(a.get("origin") or "", "-"), dur, cons, flag)
    console.print(t)
    if not total:
        console.print("（没有样本）")
        return 0
    if ambiguous:
        console.print(f"[yellow]{'、'.join(ambiguous)} 的样本分成两组互不相像，没有麦克风 / 文件样本可以判断哪组才是本人："
                      "请在说话人页试听后删掉不是 TA 的（或用 speakers record 录一段作为基准）。[/yellow]")
    if not outliers:
        if not ambiguous:
            console.print("所有样本都和同一人的其他样本一致。")
        return 0
    names = sorted({n for _, n, _ in outliers})
    console.print(f"[yellow]{len(outliers)} 个样本和其他样本不像（可能混了别人的声音）：{'、'.join(names)}[/yellow]")
    if not getattr(args, "fix", False):
        console.print("删除它们：audio-note speakers audit --fix")
        return 0
    ids = [e for _, _, e in outliers]
    cur = db.execute(f"DELETE FROM speaker_embeddings WHERE id IN ({','.join('?' * len(ids))})", ids)
    console.print(f"已删除 {int(cur.rowcount or 0)} 个离群样本。")
    for sp in speakers:
        left = db.one("SELECT COUNT(*) AS n FROM speaker_embeddings WHERE speaker_id=?", (int(sp["id"]),))
        if sp["name"] in names and left and int(left["n"]) == 0:
            console.print(f"[yellow]{sp['name']} 已没有样本：在说话人页重新标记一段，或 speakers enroll 补充。[/yellow]")
    return 0


def _relation_text(relation: str | None, source: str | None) -> str:
    """'同事' / '朋友（自动推断）' / '' for the speaker tables."""
    if not relation:
        return ""
    return f"{relation}（自动推断）" if source == "auto" else str(relation)


def cmd_speakers_list(cfg: Config, args: argparse.Namespace) -> int:
    db = _db(cfg)
    rows = db.query(
        "SELECT s.id, s.name, s.notes, s.created_at, s.is_self, s.relation, s.relation_source,"
        " COUNT(e.id) AS n, COALESCE(SUM(e.duration_s),0) AS dur"
        " FROM speakers s LEFT JOIN speaker_embeddings e ON e.speaker_id=s.id GROUP BY s.id ORDER BY s.is_self DESC, s.name"
    )
    t = Table(title="已注册说话人")
    for col in ("id", "姓名", "本人", "关系", "样本数", "语音时长", "注册时间", "备注"):
        t.add_column(col)
    for r in rows:
        t.add_row(str(r["id"]), r["name"], "我" if r["is_self"] else "", _relation_text(r["relation"], r["relation_source"]),
                  str(r["n"]), f"{r['dur']:.0f}s", r["created_at"], r["notes"] or "")
    console.print(t)
    if not rows:
        console.print("（空）")
    elif not any(r["is_self"] for r in rows):
        console.print("还没有标出哪位是本人：audio-note speakers set-self <姓名>（日报 / 问答会站在本人视角）")
    return 0


def _speaker_by_name(db: DB, name: str) -> Any:
    row = db.one("SELECT * FROM speakers WHERE name=?", (str(name).strip(),))
    if row is None:
        raise CLIError(f"没有叫 {name} 的说话人（audio-note speakers list 查看）")
    return row


def cmd_speakers_set_self(cfg: Config, args: argparse.Namespace) -> int:
    """Mark an enrolled speaker as the user ("我"): digests, chat and pushes take that person's point of view."""
    db = _db(cfg)
    row = _speaker_by_name(db, args.name)
    prev = db.self_speaker()
    db.set_self_speaker(int(row["id"]))
    if prev is not None and int(prev["id"]) != int(row["id"]):
        console.print(f"本人已从 {prev['name']} 改为 [bold]{row['name']}[/bold] (id {row['id']})。")
    else:
        console.print(f"已把 [bold]{row['name']}[/bold] (id {row['id']}) 设为本人。")
    return 0


def cmd_speakers_relation(cfg: Config, args: argparse.Namespace) -> int:
    """Store (or clear with '-') a speaker's relation to the user; `--note` keeps a remark next to it."""
    db = _db(cfg)
    row = _speaker_by_name(db, args.name)
    rel = str(args.relation or "").strip()
    if row["is_self"]:
        raise CLIError(f"{row['name']} 是本人，不需要标注关系")
    if rel in ("", "-"):
        db.set_relation(int(row["id"]), None, None)
        console.print(f"已清除 {row['name']} 的关系标注。")
        return 0
    db.set_relation(int(row["id"]), rel, "user", (args.note or "").strip() or None)
    console.print(f"已标注 {row['name']}：{rel}" + (f"（{args.note}）" if args.note else "") + "。")
    return 0


def cmd_speakers_relations(cfg: Config, args: argparse.Namespace) -> int:
    """Ask the LLM how each enrolled speaker relates to the user (hand-labelled ones are never touched)."""
    relations = _lazy_module("analysis.relations")
    db = _db(cfg)
    ids = None
    if args.speaker:
        ids = [int(_speaker_by_name(db, args.speaker)["id"])]
    client = _client(cfg)
    results = list(relations.infer_relations(cfg, db, client, speaker_ids=ids, force=bool(args.force)) or [])
    t = Table(title="关系推断")
    for col in ("id", "姓名", "关系", "置信度", "结果", "理由"):
        t.add_column(col)
    for r in results:
        conf = r.get("confidence")
        if r.get("skipped"):
            outcome = f"跳过：{r['skipped']}"
        elif r.get("applied"):
            outcome = "[green]已写入[/green]"
        else:
            outcome = "未写入（不确定或置信度不够）"
        t.add_row(str(r.get("id", "")), str(r.get("name", "")), str(r.get("relation") or ""),
                  f"{float(conf):.2f}" if isinstance(conf, (int, float)) else "", outcome, str(r.get("reason") or ""))
    console.print(t)
    applied = sum(1 for r in results if r.get("applied"))
    if not results:
        console.print("没有需要推断的说话人（本人和已手动标注的不推断；语句太少的也跳过）。")
    else:
        console.print(f"共 {len(results)} 人，写入 {applied} 个关系。手动标注：audio-note speakers relation <姓名> <关系>")
    return 0


def _scan_voices(cfg: Config, db: DB, voices: Any, engine_factory: Callable[[], Any]) -> dict[str, int]:
    """backfill_cluster_embeddings for the segments that still lack cluster embeddings (loads the engine only when needed)."""
    missing = voices.missing_segments(db)
    if not missing:
        return {"segments": 0, "clusters": 0, "skipped": 0}
    console.print(f"正在扫描 {len(missing)} 段录音的声音…")
    stats = dict(voices.backfill_cluster_embeddings(db, cfg, engine_factory()))
    console.print(f"扫描完成：{stats['segments']} 段 / {stats['clusters']} 个声音簇，{stats['skipped']} 段跳过。")
    return stats


KEY_SHOWN = 6   # characters of a voice group's stable key shown in the table (the full key is accepted too)


def _print_voice_groups(res: dict[str, Any]) -> None:
    groups = res.get("groups") or []
    if not groups:
        console.print("没有待确认的声音（所有语句都已识别，或还没有已转写的记录）。")
        return
    t = Table(title=f"未识别的声音（{len(groups)} 组，{res.get('unknown_utterances', 0)} 句未识别）")
    for col in ("#", "编号", "句数", "时长", "日期", "可能是", "示例"):
        t.add_column(col)
    for i, g in enumerate(groups, start=1):
        days = g.get("days") or []
        span = days[0] if len(days) <= 1 else f"{days[0]} ~ {days[-1]}（{len(days)} 天）"
        closest = g.get("closest") or {}
        maybe = f"{closest['name']}（{closest['score']:.2f}）" if closest and float(closest.get("score") or 0) >= 0.3 else ""
        samples = "\n".join(f"{s.get('time', '')} {s.get('text', '')}".strip() for s in (g.get("samples") or [])[:3])
        t.add_row(str(i), str(g.get("key") or "")[:KEY_SHOWN], str(g.get("utterances", 0)), f"{float(g.get('speech_s', 0)):.0f}s",
                  span or "", maybe, samples)
    console.print(t)
    console.print("命名：audio-note speakers voices --assign <#> <姓名> [--self] [--relation 同事]（<#> 也可以写编号，编号在重新扫描后不变）")


def _resolve_voice_group(groups: list[dict[str, Any]], ref: Any) -> tuple[int, dict[str, Any]]:
    """`--assign` takes the 1-based index of the listing or the group's stable key (full or a prefix of at
    least a few characters). Returns (index, group)."""
    ref = str(ref or "").strip().lower()
    if not ref:
        raise CLIError("--assign 需要给出序号或编号：speakers voices --assign <#或编号> <姓名>")
    if ref.isdigit() and 1 <= int(ref) <= len(groups):
        return int(ref), groups[int(ref) - 1]
    hits = [(i, g) for i, g in enumerate(groups, start=1) if len(ref) >= 3 and str(g.get("key") or "").lower().startswith(ref)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise CLIError(f"编号 {ref} 不唯一（匹配 {len(hits)} 组），请多写几位")
    if ref.isdigit():
        raise CLIError(f"没有第 {int(ref)} 组（共 {len(groups)} 组，先运行 audio-note speakers voices 查看）")
    raise CLIError(f"没有编号为 {ref} 的声音组（共 {len(groups)} 组，先运行 audio-note speakers voices 查看）")


def _voice_engine_factory(cfg: Config) -> Callable[[], Any]:
    """Lazy SpeakerEngine for `voices` / `scan`: only built when a scan or a naming needs it; a missing
    model file or a failing load is a CLIError (Chinese message) instead of a traceback."""
    def engine() -> Any:
        if not cfg.model_path("embedding_model").exists():
            raise CLIError(f"模型文件缺失，无法提取声纹（运行 scripts/download_models.ps1）：{cfg.model_path('embedding_model')}")
        from .speech.speakers import SpeakerEngine

        try:
            return SpeakerEngine(cfg)
        except CLIError:
            raise
        except Exception as e:  # noqa: BLE001 - onnxruntime / model file problems
            raise CLIError(f"声纹模型加载失败：{e}（检查 models/ 目录，或重新运行 scripts/download_models.ps1）") from e

    return engine


def cmd_speakers_voices(cfg: Config, args: argparse.Namespace) -> int:
    """List the unidentified voices grouped across recordings; `--assign <#|key> <name>` names one group."""
    voices = _lazy_module("speech.voices")
    db = _db(cfg)
    engine = _voice_engine_factory(cfg)
    scan = _scan_voices(cfg, db, voices, engine)
    res = voices.group_unknown_voices(db, cfg)
    if args.assign is None:
        _print_voice_groups(res)
        return 0
    if not args.name:
        raise CLIError("--assign 需要给出姓名：speakers voices --assign <#或编号> <姓名>")
    groups = res.get("groups") or []
    if scan["clusters"]:
        _print_voice_groups(res)
        console.print("[yellow]刚扫描到新的录音，分组可能变了：请按上面的新序号或编号重新运行 --assign。[/yellow]")
        return 1
    idx, g = _resolve_voice_group(groups, args.assign)
    console.print(f"第 {idx} 组 = 编号 {g.get('key', '')}（{g.get('utterances', 0)} 句）")
    try:
        out = voices.name_voice(db, cfg, engine(), g["members"], args.name, is_self=bool(args.self),
                                relation=(args.relation or "").strip() or None, **_force_kwargs(args))
    except ValueError as e:
        raise CLIError(str(e) + ("" if getattr(args, "force", False) or "不像" not in str(e) else "（命令行用 --force）")) from e
    console.print(f"已标记为 [bold]{out['name']}[/bold] (id {out['id']})：{out['utterances_assigned']} 句，"
                  f"补充 {out['samples_added']} 个声纹样本（{len(out['segments'])} 段 / {len(out['days'])} 天）。")
    if out.get("samples_rejected"):
        console.print(f"[yellow]{out['samples_rejected']} 个样本和 {out['name']} 已有的样本不像，没有加入（确定是同一个人可加 --force）。[/yellow]")
    if out.get("members_inconsistent"):
        console.print(f"[yellow]这组里有 {out['members_inconsistent']} 段声音和其他成员不像，没有用来做样本。[/yellow]")
    if out.get("samples_from_clusters"):
        console.print(f"这组的录音已过期，{out['samples_from_clusters']} 个样本直接取自扫描时保存的声纹。")
    if out.get("skipped_user_labels"):
        console.print(f"{out['skipped_user_labels']} 句已手动标为别人，保持不变。")
    if args.self:
        console.print("已设为本人。")
    if args.relation:
        console.print(f"关系：{args.relation}")
    _reindex_after_enroll(cfg, db)
    from .audio.loudness import min_speech_dbfs

    left = voices.unknown_utterance_count(db, min_speech_dbfs(cfg))
    if left and out["samples_added"]:
        console.print(f"还有 {left} 句未识别：audio-note speakers relabel 可用新声纹重新识别其他记录。")
    return 0


def cmd_speakers_scan(cfg: Config, args: argparse.Namespace) -> int:
    """Only compute the missing cluster embeddings (what `voices` does before listing)."""
    voices = _lazy_module("speech.voices")
    db = _db(cfg)
    stats = _scan_voices(cfg, db, voices, _voice_engine_factory(cfg))
    if not stats["segments"] and not stats["skipped"]:
        console.print("没有需要扫描的录音段。")
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
        min_s = float(cfg.speakers.get("min_label_sample_s", 2.5))
    except (TypeError, ValueError, AttributeError):
        min_s = 2.5
    try:
        sid = engine.enroll_from_segment_turns(db, args.name, path, [(a, b)], min_speech_s=min_s, **_force_kwargs(args))
    except ValueError as e:
        if getattr(e, "min_s", None) is not None:       # speakers.TooShortSample (checked by attribute: tests stub the module)
            raise CLIError(f"{e}（命令行的 label 只能提取声纹，请选 ≥ {min_s:g} 秒语音的范围，或在界面的「记录」页标记这句）") from e
        raise CLIError(str(e) + ("" if getattr(args, "force", False) or "不像" not in str(e) else "（命令行用 --force）")) from e
    console.print(f"已把 {path.name} 的 {a:.1f}s–{b:.1f}s 标记为 {args.name} (id {sid})。")
    if getattr(args, "self", False):
        db.set_self_speaker(int(sid))
        console.print(f"已把 {args.name} 设为本人。")
    relation = (getattr(args, "relation", None) or "").strip()
    if relation:
        db.set_relation(int(sid), relation, "user")
        console.print(f"关系：{relation}")
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


def cmd_loudness(cfg: Config, args: argparse.Namespace) -> int:
    """Speech level of the stored sentences: fill in the missing loudness_db (reads the audio), print the
    distribution per speaker and how many sentences fall below audio.min_speech_dbfs; `--apply --yes` deletes
    those (far-away voices recorded before the gate existed) and rewrites the day's Markdown."""
    from .audio.loudness import backfill_loudness, loudness_stats, min_speech_dbfs

    db = _db(cfg)
    day = _parse_day(args.day) if args.day else None
    thr = min_speech_dbfs(cfg)
    computed = backfill_loudness(db, day=day)
    if computed:
        console.print(f"补算了 {computed} 句的音量。")
    st = loudness_stats(db, thr, day=day, unknown_label=str(cfg.speakers.unknown_label or "未知"))
    title = "每句音量分布（dBFS）" + (f"，门槛 {thr:g}" if thr is not None else "，门槛已关闭") + (f"，{day}" if day else "")
    t = Table(title=title)
    for col in ("人", "句数", "中位", "p10", "p90", "低于门槛"):
        t.add_column(col)
    for s in st["speakers"]:
        t.add_row(s["name"], str(s["n"]), f"{s['median']:.0f}", f"{s['p10']:.0f}", f"{s['p90']:.0f}", str(s["below"]))
    console.print(t)
    if not st["total"]:
        console.print("（没有记录）")
        return 0
    if st["missing"]:
        console.print(f"[yellow]{st['missing']} 句的音频已删除，算不出音量。[/yellow]")
    if thr is None:
        console.print("audio.min_speech_dbfs 为 null（关闭），不会过滤。")
        return 0
    console.print(f"{st['below']} 句低于 {thr:g} dBFS（远处 / 不清晰的声音）。")
    if not getattr(args, "apply", False):
        if st["below"]:
            console.print("删除它们：audio-note loudness --apply --yes（手动标过说话人的句子不删）")
        return 0
    rows = db.query("SELECT id, day FROM utterances WHERE loudness_db IS NOT NULL AND loudness_db < ?"
                    " AND (speaker_source IS NULL OR speaker_source<>'user')" + (" AND day=?" if day else ""),
                    (float(thr), day) if day else (float(thr),))
    if not rows:
        console.print("没有可删除的句子（低于门槛的都是手动标过的）。")
        return 0
    console.print(f"将删除 {len(rows)} 句（文字、索引和向量一起删除，音频不动）。")
    if not getattr(args, "yes", False):
        console.print("[yellow]确认请加 --yes。[/yellow]")
        return 1
    ids = [int(r["id"]) for r in rows]
    days = sorted({str(r["day"]) for r in rows})
    deleted = 0
    for k in range(0, len(ids), 500):
        chunk = ids[k:k + 500]
        cur = db.execute(f"DELETE FROM utterances WHERE id IN ({','.join('?' * len(chunk))})", chunk)
        deleted += int(cur.rowcount or 0)
    write_day_markdown = _lazy("speech.pipeline", "write_day_markdown")
    for d in days:
        try:
            write_day_markdown(db, cfg, d)
        except Exception as e:  # noqa: BLE001
            log.warning("cannot write markdown for %s: %s", d, e)
    console.print(f"已删除 {deleted} 句，重写了 {len(days)} 天的记录文件。")
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
    s.add_argument("--force", action="store_true", help="样本和此人已有样本不像时仍然加入")
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
    s.add_argument("--self", dest="self", action="store_true", help="这个人就是我（本人）")
    s.add_argument("--relation", default=None, metavar="R", help="和本人的关系，如 同事 / 朋友 / 领导")
    s.add_argument("--force", action="store_true", help="这段声音和此人已有样本不像时仍然加入")
    s.set_defaults(func=cmd_speakers_label)
    s = ssub.add_parser("voices", help="列出记录里还没认出来的声音（按声音分组），--assign 给一组起名字")
    s.add_argument("--assign", default=None, metavar="N|编号", help="给第 N 组（或编号为 N 的组）命名；序号和编号见不带参数的输出")
    s.add_argument("name", nargs="?", default=None, help="配合 --assign：这组声音是谁")
    s.add_argument("--self", dest="self", action="store_true", help="配合 --assign：这组声音是我自己")
    s.add_argument("--relation", default=None, metavar="R", help="配合 --assign：和本人的关系")
    s.add_argument("--force", action="store_true", help="配合 --assign：这组声音和此人已有样本不像时仍然加入")
    s.set_defaults(func=cmd_speakers_voices)
    ssub.add_parser("scan", help="只扫描还没分析过声音的录音段（voices 会自动做）").set_defaults(func=cmd_speakers_scan)
    s = ssub.add_parser("set-self", help="标出哪位说话人是本人（日报 / 问答以本人视角）")
    s.add_argument("name")
    s.set_defaults(func=cmd_speakers_set_self)
    s = ssub.add_parser("relation", help="标注某人和本人的关系（- 清除）")
    s.add_argument("name")
    s.add_argument("relation", help="如 同事 / 朋友 / 领导 / 导师，或 - 清除")
    s.add_argument("--note", default=None, help="备注")
    s.set_defaults(func=cmd_speakers_relation)
    s = ssub.add_parser("relations", help="让模型从记录里推断各说话人和本人的关系（手动标过的不动）")
    s.add_argument("--force", action="store_true", help="忽略上次推断的缓存")
    s.add_argument("--speaker", default=None, metavar="NAME", help="只推断某一个人")
    s.set_defaults(func=cmd_speakers_relations)
    s = ssub.add_parser("relabel", help="对已转写的录音段按当前声纹库重新识别说话人（不重复 ASR）")
    s.add_argument("--day", default=None, help="只处理某天 YYYY-MM-DD")
    s.add_argument("--segment", type=int, default=None, help="只处理某个 segment id")
    s.set_defaults(func=cmd_speakers_relabel)
    ssub.add_parser("calibrate", help="根据注册样本估计识别阈值").set_defaults(func=cmd_speakers_calibrate)
    s = ssub.add_parser("audit", help="检查每个声纹样本和同一人其他样本是否一致（--fix 删除离群样本）")
    s.add_argument("--fix", action="store_true", help="删除离群样本")
    s.set_defaults(func=cmd_speakers_audit)

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
    s = sub.add_parser("loudness", help="每句音量分布；--apply --yes 删除低于 audio.min_speech_dbfs 的句子（远处的声音）")
    s.add_argument("--day", default=None, help="只看 / 只处理某天 YYYY-MM-DD")
    s.add_argument("--apply", action="store_true", help="删除低于门槛且没有手动标过说话人的句子，并重写当天记录")
    s.add_argument("--yes", "-y", action="store_true", help="配合 --apply：确认删除")
    s.set_defaults(func=cmd_loudness)
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
