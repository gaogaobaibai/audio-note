"""transcripts/YYYY-MM-DD.md: the day's utterances as Markdown, regenerated from the DB (the DB is the source of
truth; the file is a convenience copy). Part of the audio_note.speech.pipeline facade."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

from ..config import Config
from ..db import DB
from .chunking import F, log


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def segment_heading(seg_id: int, seg_path: str, started_at: datetime, duration_s: float,
                    source: str | None = None, origin: str | None = None) -> str:
    """`## HH:MM:SS–HH:MM:SS (#id file.opus)[（导入：name）]` — the segment reference lets the user run
    `speakers label <id> ... --from/--to` or `speakers label --at` straight from the transcript."""
    end = started_at + timedelta(seconds=float(duration_s or 0.0))
    label = ""
    if source == "import" and origin:
        label = f"（导入：{Path(origin).name}）"
    return f"## {started_at.strftime('%H:%M:%S')}–{end.strftime('%H:%M:%S')} (#{seg_id} {Path(seg_path).name}){label}"


def write_day_markdown(db: DB, cfg: Config, day: str) -> Path:
    """Regenerate transcripts/YYYY-MM-DD.md from the DB (grouped by segment)."""
    unknown = cfg.speakers.unknown_label
    rows = db.query(
        "SELECT u.*, s.path AS seg_path, s.started_at AS seg_started_at, s.duration_s AS seg_duration_s,"
        " s.source AS seg_source, s.origin AS seg_origin"
        " FROM utterances u JOIN segments s ON s.id=u.segment_id WHERE u.day=? ORDER BY u.abs_start, u.start_s, u.id",
        (day,),
    )
    lines: list[str] = [f"# {day} 记录", ""]
    cur_seg: int | None = None
    for r in rows:
        if r["segment_id"] != cur_seg:
            cur_seg = r["segment_id"]
            if lines[-1] != "":
                lines.append("")
            lines.append(F.segment_heading(int(r["segment_id"]), r["seg_path"], F._parse_iso(r["seg_started_at"]),
                                           r["seg_duration_s"], r["seg_source"], r["seg_origin"]))
        ts = F._parse_iso(r["abs_start"]).strftime("%H:%M:%S")
        name = r["speaker_name"] or unknown
        text = (r["text"] or "").replace("\r", "").replace("\n", " ").strip()
        lines.append(f"- **{ts} {name}**：{text}")
    out_dir = cfg.transcripts_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{day}.md"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines).rstrip("\n") + "\n")
    return path


def write_days_markdown(db: DB, cfg: Config, days: Iterable[str]) -> None:
    """write_day_markdown for each day (sorted); a failure is only logged: the rows are in the DB already."""
    for day in sorted(set(days)):
        try:
            F.write_day_markdown(db, cfg, day)
        except Exception as e:  # noqa: BLE001 - the DB is the source of truth
            log.warning("cannot write markdown for %s: %s", day, e)


__all__ = ["segment_heading", "write_day_markdown", "write_days_markdown"]
