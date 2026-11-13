"""Sessions / episodes: the conversations a day is made of.

    GET /api/sessions?day=YYYY-MM-DD&limit=50     -> {"day", "sessions": [session, ...]}
    GET /api/sessions/{id}                        -> session + "utterances": [the transcript rows of it]

A session is {"id", "day", "started_at", "ended_at", "start", "end" (HH:MM), "duration_s", "crosses_midnight",
"utterance_count", "speech_s", "chars", "status" (open | closed), "kind", "title", "summary", "topics": [],
"participants": [], "refined", "digested", "refined_at", "digested_at", "meta", "updated_at", "speakers":
[{"id", "name", "n", "speech_s"}]}. kind / title / summary / topics / participants stay empty until the episode
step has looked at the session; "speakers" is always there (counted from the sentences).

Without `day` the newest sessions are returned (newest first). Sessions are cut by memory.sessions.sync, which
the post-processing chain runs after every processed segment; these handlers only read.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from ..deps import RT, day_arg, unknown_label

router = APIRouter(prefix="/api")


@router.get("/sessions")
def sessions_list(rt: RT, day: str | None = None, limit: int = 50) -> dict[str, Any]:
    from ...memory import sessions

    unk = unknown_label(rt)
    if day:
        return {"day": day_arg(day), "sessions": sessions.for_day(rt.db, day, unknown=unk)}
    ids = [int(r["id"]) for r in rt.db.query("SELECT id FROM sessions ORDER BY started_at DESC, id DESC LIMIT ?",
                                             (max(1, min(int(limit), 500)),))]
    out = [s for s in (sessions.get(rt.db, sid, unknown=unk) for sid in ids) if s is not None]
    return {"day": None, "sessions": out}


@router.get("/sessions/{session_id}")
def session_get(session_id: int, rt: RT) -> dict[str, Any]:
    from ...memory import sessions
    from ..serializers import utt_dicts

    db = rt.db
    unk = unknown_label(rt)
    out = sessions.get(db, int(session_id), unknown=unk)
    if out is None:
        raise HTTPException(404, "没有这个会话")
    rows = db.query("SELECT * FROM utterances WHERE session_id=? ORDER BY abs_start, id", (int(session_id),))
    seg_ok: dict[int, bool] = {}

    def audio_ok(seg_id: int) -> bool:
        if seg_id not in seg_ok:
            row = db.get_segment(seg_id)
            seg_ok[seg_id] = bool(row and Path(row["path"]).exists())
        return seg_ok[seg_id]

    out["utterances"] = utt_dicts(db, rows, unk, audio_ok)        # the same shape as GET /api/transcript/{day}
    out["items"] = [dict(r) for r in db.query(
        "SELECT id, kind, title, content, status, due, owner FROM memory_items WHERE session_id=?"
        " AND status IN ('active', 'done') ORDER BY id", (int(session_id),))]
    return out
