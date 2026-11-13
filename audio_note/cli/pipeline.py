"""Recording pipeline commands: `status`, `import`, `process`, `transcribe`."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.table import Table

from ..config import Config
from ..db import DB
from ._late import Late, handler
from .helpers import (CLIError, _db, _expand_files, _human_bytes, _invoke, _lazy, _meter, _parse_datetime, _print_result,
                      add_json_flag, emit_json, wants_json)

log = logging.getLogger("audio_note.cli")
console = Late("console")
_client = Late("_client")


def status_data(cfg: Config, db: DB) -> dict[str, Any]:
    """Everything `status` shows, as plain data: the tables and `--json` are two renderings of it."""
    from ..audio.storage import dir_size_bytes
    from ..runtime import profile_label
    from ..service import running_pid

    today = datetime.now().strftime("%Y-%m-%d")
    pid = running_pid(cfg)
    seg_rows = db.query("SELECT status, COUNT(*) AS n, COALESCE(SUM(duration_s),0) AS dur FROM segments GROUP BY status ORDER BY status")
    utt_today = db.one("SELECT COUNT(*) AS n FROM utterances WHERE day=?", (today,))
    utt_all = db.one("SELECT COUNT(*) AS n, COUNT(DISTINCT day) AS days FROM utterances")
    size = dir_size_bytes(cfg.audio_dir) if cfg.audio_dir.exists() else 0
    last = db.one("SELECT day, created_at, pushed_at, utterance_count FROM daily_summaries ORDER BY day DESC LIMIT 1")
    kv_day = db.get_kv("last_daily_day")
    by_status = {str(r["status"]): {"count": int(r["n"]), "duration_s": float(r["dur"])} for r in seg_rows}
    spk = db.query(
        "SELECT s.id, s.name, s.notes, COUNT(e.id) AS n, COALESCE(SUM(e.duration_s),0) AS dur"
        " FROM speakers s LEFT JOIN speaker_embeddings e ON e.speaker_id=s.id GROUP BY s.id ORDER BY s.name"
    )
    return {
        "day": today,
        "profile": cfg.get("_profile"),
        "profile_label": profile_label(cfg.get("_profile")),
        "config_path": str(cfg.get("_config_path", "")),
        "data_dir": str(cfg.data_dir),
        "service": {"running": bool(pid), "pid": pid},
        "api_key_configured": bool(cfg.mimo.api_key),
        # same shape as the `segments` section of `doctor --json`
        "segments": {"total": sum(v["count"] for v in by_status.values()), "by_status": by_status,
                     "failed": by_status.get("failed", {}).get("count", 0)},
        "utterances": {"today": int(utt_today["n"]) if utt_today else 0,
                       "total": int(utt_all["n"]) if utt_all else 0, "days": int(utt_all["days"]) if utt_all else 0},
        "audio": {"bytes": int(size), "max_disk_gb": float(cfg.storage.max_disk_gb or 0)},
        "last_daily": ({"day": last["day"], "created_at": last["created_at"], "pushed_at": last["pushed_at"],
                        "utterance_count": last["utterance_count"]} if last else None),
        "last_daily_day": kv_day or None,
        "speakers": [{"id": int(r["id"]), "name": r["name"], "notes": r["notes"], "samples": int(r["n"]),
                      "speech_s": float(r["dur"])} for r in spk],
    }


def cmd_status(cfg: Config, args: argparse.Namespace) -> int:
    db = _db(cfg)
    d = status_data(cfg, db)
    if wants_json(args):
        emit_json(d)
        return 0

    t = Table(title="audio-note 状态", show_header=False)
    t.add_column("项")
    t.add_column("值")
    t.add_row("档案", d["profile_label"])
    t.add_row("配置文件", d["config_path"])
    t.add_row("数据目录", d["data_dir"])
    pid = d["service"]["pid"]
    t.add_row("服务", f"运行中 (pid {pid})" if pid else "未运行")
    t.add_row("MiMo API key", "已配置" if d["api_key_configured"] else "[red]未配置[/red]")

    segs = d["segments"]["by_status"]
    if segs:
        t.add_row("录音段", "  ".join(f"{status}={s['count']} ({s['duration_s'] / 60:.0f} min)" for status, s in segs.items()))
    else:
        t.add_row("录音段", "0")
    utt = d["utterances"]
    t.add_row("今日语句", str(utt["today"]))
    t.add_row("语句总数", f"{utt['total']} 条 / {utt['days']} 天")
    cap = d["audio"]["max_disk_gb"]
    t.add_row("音频占用", f"{_human_bytes(d['audio']['bytes'])}" + (f" / 上限 {cap:g} GB" if cap else ""))
    last = d["last_daily"]
    if last:
        t.add_row("最近日报", f"{last['day']}（生成 {last['created_at']}，推送 {last['pushed_at'] or '未推送'}）")
    else:
        t.add_row("最近日报", "无")
    if d["last_daily_day"]:
        t.add_row("调度 last_daily_day", str(d["last_daily_day"]))
    if d["segments"]["failed"]:
        t.add_row("失败段", f"[yellow]{d['segments']['failed']}[/yellow]（audio-note process --retry-failed 重试）")
    console.print(t)

    st = Table(title="已注册说话人")
    st.add_column("id", justify="right")
    st.add_column("姓名")
    st.add_column("样本数", justify="right")
    st.add_column("语音时长", justify="right")
    st.add_column("备注")
    for r in d["speakers"]:
        st.add_row(str(r["id"]), r["name"], str(r["samples"]), f"{r['speech_s']:.0f}s", r["notes"] or "")
    if d["speakers"]:
        console.print(st)
    else:
        console.print("未注册说话人（audio-note speakers enroll <姓名> <wav...>）")
    return 0


def cmd_import(cfg: Config, args: argparse.Namespace) -> int:
    from ..audio.importer import import_file
    from ..audio.vad import SileroVAD

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
    client = _meter(_client(cfg), db)
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
    db = _db(cfg)
    client = _meter(_client(cfg), db)
    result = _invoke(transcribe_file, cfg=cfg, config=cfg, client=client, llm=client, db=db,
                     path=path, file=path, file_path=path, src=path, source=path, audio_path=path)
    _print_result(result)
    return 0


def register(sub: Any) -> None:
    add_json_flag(sub.add_parser("status", help="状态总览")).set_defaults(func=handler(cmd_status))

    s = sub.add_parser("import", help="导入已有录音文件")
    s.add_argument("files", nargs="+", help="音频文件或通配符")
    s.add_argument("--start", help='录音开始时间，如 "2026-10-31 09:00"（默认从文件名/修改时间推断）')
    s.set_defaults(func=handler(cmd_import))

    s = sub.add_parser("process", help="处理待转写的录音段")
    s.add_argument("--retry-failed", action="store_true", help="重试 failed 状态的段")
    s.add_argument("--limit", type=int, default=None, help="最多处理 N 段")
    s.set_defaults(func=handler(cmd_process))

    s = sub.add_parser("transcribe", help="一次性转写一个文件并打印（不入库）")
    s.add_argument("file")
    s.set_defaults(func=handler(cmd_transcribe))
