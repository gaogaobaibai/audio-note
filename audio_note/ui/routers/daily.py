"""Daily digests and the memory notes extracted from them.

    GET    /api/daily
    GET    /api/daily/{day}
    GET    /api/notes?kind=&q=&limit=200          compatibility: reads memory_items (see routers/memory.py)
    DELETE /api/notes/{note_id}                   compatibility: the item becomes a tombstone
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter

from ..deps import RT, day_arg

router = APIRouter(prefix="/api")


@router.get("/daily")
def daily_list(rt: RT) -> list[dict[str, Any]]:
    rows = rt.db.query("SELECT d.day, d.created_at, d.pushed_at, d.utterance_count, d.model, d.summary_json,"
                       " (SELECT COUNT(*) FROM utterances u WHERE u.day=d.day) AS now_count"
                       " FROM daily_summaries d ORDER BY d.day DESC")
    out = []
    for r in rows:
        partial = False
        overview = ""
        try:
            js = json.loads(r["summary_json"] or "{}")
            partial = bool(js.get("partial"))
            overview = str(js.get("overview") or "")[:160]
        except ValueError:
            pass
        out.append({"day": r["day"], "created_at": r["created_at"], "pushed_at": r["pushed_at"], "model": r["model"],
                    "utterance_count": r["utterance_count"], "now_count": int(r["now_count"]), "partial": partial,
                    "stale": int(r["now_count"]) != int(r["utterance_count"] or 0), "overview": overview})
    return out


@router.get("/daily/{day}")
def daily_get(day: str, rt: RT) -> dict[str, Any]:
    day_arg(day)
    r = rt.db.one("SELECT * FROM daily_summaries WHERE day=?", (day,))
    now = rt.db.one("SELECT COUNT(*) AS n FROM utterances WHERE day=?", (day,))
    n_now = int(now["n"]) if now else 0
    if r is None:
        return {"day": day, "exists": False, "now_count": n_now}
    try:
        js = json.loads(r["summary_json"] or "{}")
    except ValueError:
        js = {}
    js.pop("blocks", None)
    return {"day": day, "exists": True, "markdown": r["summary_md"], "json": js, "created_at": r["created_at"],
            "pushed_at": r["pushed_at"], "model": r["model"], "utterance_count": r["utterance_count"],
            "now_count": n_now, "stale": n_now != int(r["utterance_count"] or 0)}


@router.get("/notes")
def notes(rt: RT, kind: str | None = None, q: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    """The 0.3 notes list, served from memory_items (ids of migrated notes are unchanged), in the old shape:
    id, kind, title, content, day_from, day_to, updated_at. Deleted (tombstoned), merged-away and archived
    items are left out. New code uses GET /api/memory/items, which has every field."""
    where, params = ["status IN ('active', 'done')"], []
    if kind:
        where.append("kind=?")
        params.append(kind)
    if q:
        where.append("(title LIKE ? OR content LIKE ?)")
        pat = f"%{q.strip()}%"
        params += [pat, pat]
    sql = ("SELECT id, kind, title, content, COALESCE(valid_from, day) AS day_from, day AS day_to, updated_at"
           " FROM memory_items WHERE " + " AND ".join(where) + " ORDER BY updated_at DESC, id DESC LIMIT ?")
    return [dict(r) for r in rt.db.query(sql, [*params, max(1, min(int(limit), 2000))])]


@router.delete("/notes/{note_id}")
def note_delete(note_id: int, rt: RT) -> dict[str, Any]:
    """Delete = tombstone: the item disappears from every list and is never generated again."""
    from ...memory import items as mem_items

    row = rt.db.one("SELECT status FROM memory_items WHERE id=?", (int(note_id),))
    if row is None or row["status"] == "rejected":
        return {"ok": True, "deleted": 0}
    mem_items.reject(rt.db, int(note_id))
    rt.events.publish("memory", change="deleted", id=int(note_id))
    return {"ok": True, "deleted": 1}
