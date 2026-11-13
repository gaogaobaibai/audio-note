"""Memory: items (facts, todos, decisions, people ...), people profiles, forgetting.

    GET    /api/memory/items?kind=&status=&q=&subject=&limit=&offset=   -> {"items": [item], "counts": {...}}
    POST   /api/memory/items            {kind, title, content?, subject?, due?, owner?, importance?, pinned?, day?}
                                                                        -> {"ok", "item", "action": added | updated}
    GET    /api/memory/items/{id}                                       -> item + "evidence_utterances": [rows]
    PATCH  /api/memory/items/{id}       {title?, content?, subject?, kind?, status?, pinned?, importance?, due?, owner?}
                                                                        -> {"ok", "item"}
    DELETE /api/memory/items/{id}                                       -> {"ok", "deleted": 0 | 1}   (a tombstone)
    POST   /api/memory/items/{id}/restore                               -> {"ok", "item"}
    POST   /api/memory/items/{id}/merge {into}                          -> {"ok", "item"}   ({id} is folded into `into`)
    GET    /api/memory/todos?status=active&due_before=                  -> {"todos": [item]}
    GET    /api/memory/people                                           -> {"people": [person]}
    GET    /api/memory/people/{name}                                    -> person + facts / todos / recent_sessions
    POST   /api/memory/forget           {day?, time_from?, time_to?, utterance_ids?, session_id?, dry_run?=true}
                                                                        -> the report of memory.forget.forget()

An item is the memory_items row as JSON: {"id", "kind", "title", "content", "subject", "status", "source",
"confidence", "importance", "pinned": bool, "day", "valid_from", "valid_to", "due", "owner", "session_id",
"evidence": [utterance ids], "superseded_by", "hits", "last_used_at", "created_at", "updated_at", "lexicon_id",
"speaker_id", "fingerprint"}; search results add "score" and "match".

`status` filter: one of active (default) | done | archived | superseded | rejected, a comma-separated list, or
`all` (everything but the deleted ones). Every change publishes a `memory` event {change, id?}; a real forget
also publishes `utterances` / `daily` for the days it touched.

Item edits are counted in Runtime.busy() (a profile switch waits for them) but are allowed in a read-only UI
process, like the old notes delete: they never touch the rows the pipeline works on. Forgetting does, so it goes
through deps.rewriting (409 next to a running service) and holds the scheduler lock.

(The "about me" profile - /api/memory/profile - is added by the second half of package B3.)
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException

from ..deps import RT, bool_arg, int_arg, rewriting, unknown_label

router = APIRouter(prefix="/api")


def _mod() -> Any:
    from ...memory import items

    return items


def _status_arg(status: str | None) -> Any:
    items = _mod()
    if status is None or status == "":
        return ("active",)
    if status == "all":
        return "all"
    wanted = [s.strip() for s in str(status).split(",") if s.strip()]
    bad = [s for s in wanted if s not in items.STATUSES]
    if bad or not wanted:
        raise HTTPException(400, f"status 应为 {' / '.join(items.STATUSES)} 或 all")
    return tuple(wanted)


def _kinds_arg(kind: str | None) -> list[str] | None:
    items = _mod()
    if not kind:
        return None
    wanted = [k.strip() for k in str(kind).split(",") if k.strip()]
    bad = [k for k in wanted if k not in items.KINDS]
    if bad:
        raise HTTPException(400, f"kind 应为 {' / '.join(items.KINDS)}")
    return wanted or None


def _embedder(rt: Any) -> Any:
    """The retriever's embedding model, loading it when needed (None in keyword-only mode or when it cannot be
    loaded): a search typed into the memory page should find a paraphrase too."""
    try:
        return rt.retriever().embedder()
    except Exception:  # noqa: BLE001 - no model: keyword search still works
        return None


# ---------------------------------------------------------------------------------------------- items
@router.get("/memory/items")
def items_list(rt: RT, kind: str | None = None, status: str | None = None, q: str | None = None,
               subject: str | None = None, limit: int = 100, offset: int = 0) -> dict[str, Any]:
    items = _mod()
    kinds, st = _kinds_arg(kind), _status_arg(status)
    limit = max(1, min(int(limit), 1000))
    if q and q.strip():
        rows = items.search(rt.db, rt.cfg, q, kinds=kinds, status=st, subject=subject or None, limit=limit + max(0, int(offset)),
                            embedder=_embedder(rt))[max(0, int(offset)):]
    else:
        rows = items.list_items(rt.db, kinds=kinds, status=st, subject=subject or None, limit=limit, offset=offset)
    return {"items": rows, "counts": items.counts(rt.db)}


@router.post("/memory/items")
def item_create(rt: RT, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """The user writes a memory down by hand. An item of the same kind and title is updated instead
    (action "updated"), a deleted one comes back."""
    items = _mod()
    kind = str(body.get("kind") or "note").strip().lower()
    if kind not in items.KINDS:
        raise HTTPException(400, f"kind 应为 {' / '.join(items.KINDS)}")
    if not str(body.get("title") or "").strip():
        raise HTTPException(400, "需要标题")
    data = {k: body[k] for k in ("title", "content", "subject", "due", "owner", "importance", "pinned", "day", "valid_from",
                                 "valid_to", "status") if k in body}
    data["kind"] = kind
    if "pinned" in data:
        data["pinned"] = bool_arg(data["pinned"], "pinned")
    try:
        with rt.busy():
            item_id, action = items.upsert(rt.db, rt.cfg, data, source="user")
    except ValueError as e:
        raise HTTPException(400, str(e))
    rt.events.publish("memory", change=action, id=item_id, kind=kind)
    return {"ok": True, "item": items.get(rt.db, item_id), "action": action}


@router.get("/memory/items/{item_id}")
def item_get(item_id: int, rt: RT) -> dict[str, Any]:
    items = _mod()
    from ..serializers import utt_dicts

    it = items.get(rt.db, int(item_id))
    if it is None:
        raise HTTPException(404, "没有这条记忆")
    ev = it["evidence"][:50]
    rows = rt.db.query(f"SELECT * FROM utterances WHERE id IN ({','.join('?' * len(ev))}) ORDER BY abs_start, id", ev) if ev else []
    it["evidence_utterances"] = utt_dicts(rt.db, rows, unknown_label(rt))
    return it


@router.patch("/memory/items/{item_id}")
def item_patch(item_id: int, rt: RT, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    items = _mod()
    fields = {k: v for k, v in body.items() if k in items.EDITABLE}
    unknown = sorted(set(body) - set(items.EDITABLE))
    if unknown:
        raise HTTPException(400, f"不能修改这些字段：{', '.join(unknown)}")
    if not fields:
        raise HTTPException(400, "没有要修改的内容")
    if "pinned" in fields:
        fields["pinned"] = bool_arg(fields["pinned"], "pinned")
    try:
        with rt.busy():
            it = items.update(rt.db, int(item_id), **fields)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    rt.events.publish("memory", change="updated", id=int(item_id), kind=it["kind"])
    return {"ok": True, "item": it}


@router.delete("/memory/items/{item_id}")
def item_delete(item_id: int, rt: RT) -> dict[str, Any]:
    """Delete = tombstone: gone from every list and search, and never generated again."""
    items = _mod()
    it = items.get(rt.db, int(item_id))
    if it is None:
        raise HTTPException(404, "没有这条记忆")
    if it["status"] == "rejected":
        return {"ok": True, "deleted": 0}
    with rt.busy():
        items.reject(rt.db, int(item_id))
    rt.events.publish("memory", change="deleted", id=int(item_id), kind=it["kind"])
    return {"ok": True, "deleted": 1}


@router.post("/memory/items/{item_id}/restore")
def item_restore(item_id: int, rt: RT) -> dict[str, Any]:
    items = _mod()
    try:
        with rt.busy():
            it = items.restore(rt.db, int(item_id))
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    rt.events.publish("memory", change="restored", id=int(item_id), kind=it["kind"])
    return {"ok": True, "item": it}


@router.post("/memory/items/{item_id}/merge")
def item_merge(item_id: int, rt: RT, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Fold this item into another one: {"into": id of the item that stays}."""
    items = _mod()
    into = int_arg(body.get("into"), "into", minimum=1)
    if into is None:
        raise HTTPException(400, "需要 into：要并入的那条记忆的 id")
    try:
        with rt.busy():
            it = items.merge(rt.db, into, int(item_id))
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    rt.events.publish("memory", change="merged", id=into, merged=int(item_id), kind=it["kind"])
    return {"ok": True, "item": it}


@router.get("/memory/todos")
def todos_list(rt: RT, status: str | None = "active", due_before: str | None = None) -> dict[str, Any]:
    items = _mod()
    try:
        return {"todos": items.todos(rt.db, status=_status_arg(status), due_before=due_before or None)}
    except ValueError as e:
        raise HTTPException(400, str(e))


# ---------------------------------------------------------------------------------------------- people
@router.get("/memory/people")
def people_list(rt: RT) -> dict[str, Any]:
    return {"people": _mod().people(rt.db, rt.cfg)}


@router.get("/memory/people/{name}")
def person_get(name: str, rt: RT) -> dict[str, Any]:
    try:
        return _mod().person(rt.db, rt.cfg, name)
    except LookupError as e:
        raise HTTPException(404, str(e))


# ---------------------------------------------------------------------------------------------- forget
@router.post("/memory/forget")
def memory_forget(rt: RT, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Forget a day / a time range / a session / some sentences across every layer. `dry_run` defaults to
    true: the answer then only says what WOULD be removed. A real run holds the scheduler lock, so it never
    overlaps the pipeline working on the same rows."""
    from ...memory.forget import forget

    dry = bool_arg(body.get("dry_run"), "dry_run", default=True)
    ids = body.get("utterance_ids")
    if ids is not None and (not isinstance(ids, list) or any(isinstance(x, bool) or not isinstance(x, int) for x in ids)):
        raise HTTPException(400, "utterance_ids 应为句子 id 的列表")
    kw = dict(day=body.get("day") or None, time_from=body.get("time_from") or None, time_to=body.get("time_to") or None,
              utterance_ids=ids or None, session_id=int_arg(body.get("session_id"), "session_id", minimum=1))
    try:
        if dry:
            return forget(rt.db, rt.cfg, dry_run=True, **kw)
        with rewriting(rt):
            with rt.sched.lock:
                report = forget(rt.db, rt.cfg, dry_run=False, **kw)
    except ValueError as e:
        raise HTTPException(400, str(e))
    for d in report["days"]:
        rt.events.publish("utterances", day=d, segment_id=None, ids=[], change="deleted")
    for d in report["digests"]:
        rt.events.publish("daily", day=d)
    rt.events.publish("memory", change="forgot", days=report["days"], utterances=report["utterances"])
    return report
