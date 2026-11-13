"""Housekeeping commands: `cleanup`, `loudness`."""
from __future__ import annotations

import argparse
import logging
from typing import Any

from rich.table import Table

from ..config import Config
from ._late import Late, handler
from .helpers import _db, _lazy, _parse_day

log = logging.getLogger("audio_note.cli")
console = Late("console")


def cmd_cleanup(cfg: Config, args: argparse.Namespace) -> int:
    from ..audio.storage import cleanup
    from ..audio.vad import SileroVAD

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
    from ..audio.loudness import backfill_loudness, loudness_stats, min_speech_dbfs

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


def register(sub: Any) -> None:
    sub.add_parser("cleanup", help="清理无语音/过期音频").set_defaults(func=handler(cmd_cleanup))
    s = sub.add_parser("loudness", help="每句音量分布；--apply --yes 删除低于 audio.min_speech_dbfs 的句子（远处的声音）")
    s.add_argument("--day", default=None, help="只看 / 只处理某天 YYYY-MM-DD")
    s.add_argument("--apply", action="store_true", help="删除低于门槛且没有手动标过说话人的句子，并重写当天记录")
    s.add_argument("--yes", "-y", action="store_true", help="配合 --apply：确认删除")
    s.set_defaults(func=handler(cmd_loudness))
