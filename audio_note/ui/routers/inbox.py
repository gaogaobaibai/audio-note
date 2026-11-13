"""Inbox ("待确认"): the questions the system wants to ask the user.

    GET  /api/inbox?status=open&kind=&limit=50&offset=0     -> {items, counts}
    POST /api/inbox/{id}/answer   {choice?, text?, as_correct?, apply_history?}  -> {ok, item, result, counts}
    POST /api/inbox/{id}/dismiss  {never?}                                       -> {ok, item, counts}
    POST /api/inbox/{id}/snooze   {days?}                                        -> {ok, item, counts}
    POST /api/inbox/{id}/reopen                                                  -> {ok, item, reverted, counts}

All the logic is in audio_note.inbox (see its docstring for kinds, payloads and what an answer does). An item is
the dict of inbox.list_items plus "utterance": the sentence it is about as the transcript shows it (speaker,
time, text, audio available?) or null when that sentence no longer exists.

Answering and re-opening rewrite stored sentences, so they go through deps.rewriting (409 in a read-only
process); every change publishes an `inbox` event (and `utterance` / `lexicon` events for what it touched).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from ..deps import RT, bool_arg, int_arg, rewriting, unknown_label
from ..serializers import utt_dicts

router = APIRouter(prefix="/api")

STATUSES = ("open", "answered", "dismissed", "snoozed", "queued", "expired", "all")


def _with_utterances(rt: Any, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach the sentence each question is about (one query for the whole page)."""
    db = rt.db
    ids = sorted({int(it["utterance_id"]) for it in items if it.get("utterance_id") is not None})
    by_id: dict[int, dict[str, Any]] = {}
    if ids:
        rows = db.query(f"SELECT * FROM utterances WHERE id IN ({','.join('?' * len(ids))})", ids)
        seg_ok: dict[int, bool] = {}

        def audio_ok(seg_id: int) -> bool:
            if seg_id not in seg_ok:
                row = db.get_segment(seg_id)
                seg_ok[seg_id] = bool(row and Path(row["path"]).exists())
            return seg_ok[seg_id]

        by_id = {u["id"]: u for u in utt_dicts(db, rows, unknown_label(rt), audio_ok)}
    for it in items:
        uid = it.get("utterance_id")
        it["utterance"] = by_id.get(int(uid)) if uid is not None else None
    return items


def _one(rt: Any, item_id: int) -> dict[str, Any] | None:
    from ... import inbox

    it = inbox.get(rt.db, item_id)
    return _with_utterances(rt, [it])[0] if it else None


@router.get("/inbox")
def inbox_list(rt: RT, status: str = "open", kind: str | None = None, limit: int = 50, offset: int = 0) -> dict[str, Any]:
    from ... import inbox

    status = (status or "open").strip()
    if status not in STATUSES:
        raise HTTPException(400, f"status 只能是 {' / '.join(STATUSES)}")
    items = inbox.list_items(rt.db, status=status, kind=(kind or None), limit=max(1, min(int(limit), 200)),
                             offset=max(0, int(offset)), cfg=rt.cfg)
    return {"items": _with_utterances(rt, items), "counts": inbox.counts(rt.db, rt.cfg)}


@router.post("/inbox/{item_id}/answer")
def inbox_answer(item_id: int, rt: RT, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Answer a question: {choice?: index of a candidate (or the candidate itself), text?: a typed answer,
    as_correct?: "it is right as it is", apply_history?: true}. What happens is up to the kind - for a word
    question see inbox._answer_term: the lexicon learns it, the history is corrected through revisions."""
    from ... import inbox

    choice = body.get("choice")
    if isinstance(choice, (dict, list)):
        raise HTTPException(400, "choice 应为选项序号或选项文字")
    text = body.get("text")
    if text is not None and not isinstance(text, str):
        raise HTTPException(400, "text 应为文字")
    as_correct = bool_arg(body.get("as_correct"), "as_correct")
    apply_history = bool_arg(body.get("apply_history"), "apply_history", True)
    with rewriting(rt):
        try:
            result = inbox.answer(rt.db, rt.cfg, int(item_id), choice=choice, text=text, as_correct=as_correct,
                                  apply_history=apply_history, events=rt.events)
        except LookupError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
    return {"ok": True, "item": _one(rt, item_id), "result": result, "counts": inbox.counts(rt.db, rt.cfg)}


@router.post("/inbox/{item_id}/dismiss")
def inbox_dismiss(item_id: int, rt: RT, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Close a question without answering it. {never: true}: do not ask about this again."""
    from ... import inbox

    never = bool_arg(body.get("never"), "never")
    with rt.busy():
        try:
            inbox.dismiss(rt.db, int(item_id), never=never, events=rt.events)
        except LookupError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
    return {"ok": True, "item": _one(rt, item_id), "counts": inbox.counts(rt.db, rt.cfg)}


@router.post("/inbox/{item_id}/snooze")
def inbox_snooze(item_id: int, rt: RT, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Hide a question for {days: 1} days."""
    from ... import inbox

    days = int_arg(body.get("days"), "days", 1) or 1
    with rt.busy():
        try:
            inbox.snooze(rt.db, int(item_id), days=days, events=rt.events)
        except LookupError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
    return {"ok": True, "item": _one(rt, item_id), "counts": inbox.counts(rt.db, rt.cfg)}


@router.post("/inbox/{item_id}/reopen")
def inbox_reopen(item_id: int, rt: RT) -> dict[str, Any]:
    """Take an answer / dismissal / snooze back ("撤销"): the sentences an answer rewrote get their text back,
    the alias and entry it created are removed, the question is open again."""
    from ... import inbox

    with rewriting(rt):
        try:
            res = inbox.reopen(rt.db, rt.cfg, int(item_id), events=rt.events)
        except LookupError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
    return {"ok": True, "item": _one(rt, item_id), "reverted": res["reverted"], "counts": inbox.counts(rt.db, rt.cfg)}
