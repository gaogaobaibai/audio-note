"""Reading what was said: days, one day's transcript, segments, search, context around a sentence.

    GET /api/days
    GET /api/transcript/{day}?speaker=
    GET /api/segments?status=&day=&limit=200
    GET /api/search?q=&day_from=&day_to=&speaker=&limit=30&mode=
    GET /api/context/{uid}?before=6&after=6

Every utterance goes through serializers.utt_dict (the one place for its JSON shape); since 1.0 it carries
`marks` (speech.revisions.marks: what was fixed, what is being asked) next to session_id / text_source / flags.

GET /api/transcript/{day} keeps its `sessions` list (the day cut at analysis.session_gap_min, each with its
utterances) and adds `session_meta`: the memory layer's sessions of that day (audio_note.memory.sessions.for_day:
id, title, kind, summary ... - join on utterance.session_id), [] while that module is absent or has nothing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from ..deps import RT, day_arg, unknown_label
from ..serializers import utt_dict, utt_dicts, utterance_extras

router = APIRouter(prefix="/api")


@router.get("/days")
def days(rt: RT) -> list[dict[str, Any]]:
    db = rt.db
    unk = unknown_label(rt)
    out: dict[str, dict[str, Any]] = {}
    for r in db.query("SELECT day, COUNT(*) AS n, MIN(abs_start) AS first, MAX(abs_start) AS last,"
                      " COALESCE(SUM(end_s-start_s),0) AS speech FROM utterances GROUP BY day"):
        out[r["day"]] = {"day": r["day"], "utterances": int(r["n"]), "first": (r["first"] or "")[11:16],
                         "last": (r["last"] or "")[11:16], "speech_min": round(float(r["speech"]) / 60.0, 1),
                         "segments": 0, "pending": 0, "failed": 0, "daily": None, "speakers": []}
    for r in db.query("SELECT substr(started_at,1,10) AS d, status, COUNT(*) AS n FROM segments"
                      " WHERE status != 'deleted' GROUP BY d, status"):
        d = out.setdefault(r["d"], {"day": r["d"], "utterances": 0, "first": "", "last": "", "speech_min": 0.0,
                                    "segments": 0, "pending": 0, "failed": 0, "daily": None, "speakers": []})
        d["segments"] += int(r["n"])
        if r["status"] in ("new", "processing"):
            d["pending"] += int(r["n"])
        elif r["status"] == "failed":
            d["failed"] += int(r["n"])
    for r in db.query("SELECT day, created_at, pushed_at, utterance_count FROM daily_summaries"):
        if r["day"] in out:
            out[r["day"]]["daily"] = {"created_at": r["created_at"], "pushed_at": r["pushed_at"],
                                      "utterance_count": r["utterance_count"]}
    for r in db.query("SELECT day, speaker_name, COUNT(*) AS n FROM utterances GROUP BY day, speaker_name ORDER BY n DESC"):
        if r["day"] in out:
            out[r["day"]]["speakers"].append({"name": r["speaker_name"] or unk, "n": int(r["n"])})
    return sorted(out.values(), key=lambda d: d["day"], reverse=True)


@router.get("/transcript/{day}")
def transcript(day: str, rt: RT, speaker: str | None = None) -> dict[str, Any]:
    from ...analysis.daily import DailyAnalyzer

    day_arg(day)
    db = rt.db
    unk = unknown_label(rt)
    sessions = DailyAnalyzer(rt.cfg, db, None).sessions(day)
    seg_ok: dict[int, bool] = {}

    def audio_ok(seg_id: int) -> bool:
        if seg_id not in seg_ok:
            row = db.get_segment(seg_id)
            seg_ok[seg_id] = bool(row and Path(row["path"]).exists())
        return seg_ok[seg_id]

    kept: list[tuple[int, list[Any]]] = []
    for i, rows in enumerate(sessions, start=1):
        if speaker:
            rows = [r for r in rows if (r["speaker_name"] or unk) == speaker]
            if not rows:
                continue
        kept.append((i, list(rows)))
    # names of guessed speakers + UI marks, once for the whole day
    names, marks = utterance_extras(db, [r for _, rows in kept for r in rows])
    out_sessions = []
    for i, rows in kept:
        utts = [utt_dict(r, unk, audio_ok(int(r["segment_id"])), speaker_names=names,
                         marks=None if marks is None else list(marks.get(int(r["id"])) or []))
                for r in rows]
        out_sessions.append({"index": i, "start": utts[0]["time"][:5], "end": utts[-1]["time"][:5], "utterances": utts})
    segs = [dict(r) for r in db.query(
        "SELECT id, started_at, duration_s, speech_s, status, error, source FROM segments"
        " WHERE started_at >= ? AND started_at < ? AND status != 'deleted' ORDER BY started_at",
        (f"{day}T00:00:00", f"{day}T23:59:59.999"))]
    counts: dict[str, int] = {}
    for r in db.query("SELECT speaker_name, COUNT(*) AS n FROM utterances WHERE day=? GROUP BY speaker_name", (day,)):
        counts[r["speaker_name"] or unk] = int(r["n"])
    md = rt.cfg.transcripts_dir / f"{day}.md"
    from ...audio.loudness import min_speech_dbfs

    return {"day": day, "sessions": out_sessions, "segments": segs, "speakers": counts,
            "markdown_path": str(md) if md.exists() else None, "unknown_label": unk,
            "min_speech_dbfs": min_speech_dbfs(rt.cfg), "session_meta": _session_meta(db, day, unk)}


def _session_meta(db: Any, day: str, unknown: str | None = None) -> list[dict[str, Any]]:
    """The memory layer's sessions of `day` (package B3: audio_note.memory.sessions.for_day), or [] when the
    module / function is not there or fails - the transcript must never depend on it."""
    import importlib
    import inspect

    try:
        fn = getattr(importlib.import_module("audio_note.memory.sessions"), "for_day", None)
        if not callable(fn):
            return []
        try:
            takes_unknown = "unknown" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            takes_unknown = False
        out = fn(db, day, unknown=unknown) if takes_unknown else fn(db, day)
        return [dict(s) for s in (out or [])]
    except ImportError:
        return []
    except Exception as e:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).warning("session metadata unavailable for %s: %s: %s", day, type(e).__name__, e)
        return []


@router.get("/segments")
def segments(rt: RT, status: str | None = None, day: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    where, params = ["status != 'deleted'"], []
    if status:
        where.append("status=?")
        params.append(status)
    if day:
        where.append("started_at >= ? AND started_at < ?")
        params += [f"{day_arg(day)}T00:00:00", f"{day}T23:59:59.999"]
    rows = rt.db.query(f"SELECT id, path, started_at, duration_s, speech_s, status, error, source, updated_at FROM segments"
                       f" WHERE {' AND '.join(where)} ORDER BY started_at DESC LIMIT ?", [*params, max(1, min(int(limit), 2000))])
    out = []
    for r in rows:
        d = dict(r)
        d["audio"] = Path(d["path"]).exists()
        d["name"] = Path(d.pop("path")).name
        out.append(d)
    return out


@router.get("/search")
def search(q: str, rt: RT, day_from: str | None = None, day_to: str | None = None, speaker: str | None = None,
           limit: int = 30, mode: str | None = None) -> dict[str, Any]:
    q = (q or "").strip()
    if not q:
        return {"hits": [], "how": ""}
    if day_from:
        day_arg(day_from)
    if day_to:
        day_arg(day_to)
    unk = unknown_label(rt)
    retr = rt.retriever(mode, False) if mode else rt.retriever(None, False)
    res = retr.search(q, day_from=day_from, day_to=day_to, speaker=speaker, limit=max(1, min(int(limit), 200)))
    hits = [{"id": h.id, "segment_id": h.segment_id, "day": h.day, "abs_start": h.abs_start, "time": h.abs_start[11:16],
             "speaker": h.speaker_name or unk, "text": h.text, "score": round(float(h.score), 4), "source": h.source}
            for h in res.hits]
    return {"hits": hits, "how": res.how_label, "expansions": res.expansions}


@router.get("/context/{uid}")
def context(uid: int, rt: RT, before: int = 6, after: int = 6) -> dict[str, Any]:
    from ...memory import index

    rows = index.window(rt.db, int(uid), max(0, int(before)), max(0, int(after)))
    if not rows:
        raise HTTPException(404, "没有这条记录")
    return {"id": uid, "utterances": utt_dicts(rt.db, rows, unknown_label(rt))}
