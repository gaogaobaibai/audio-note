"""The inbox ("待确认"): what the system would rather ask than guess (SPEC v1 section 6.6).

Anything can put a question here with add(); the user answers it once and the answer is generalised by the
handler of its kind - to the history and to everything recorded later.

    add(db, kind, title=..., payload=..., dedupe_key=...)     -> id (0 = not asked: already settled for good)
    list_items(db, status="open") / counts(db) / get(db, id)
    answer(db, cfg, id, choice=..., text=..., as_correct=..., apply_history=...)
    dismiss(db, id, never=False) / snooze(db, id, days=1) / reopen(db, cfg, id)
    register_handler(kind, fn)        fn(db, cfg, item, answer) -> dict

Kinds and who answers them: `term` and `rule` here (package B2); `voice` and `hint` are registered by the speaker
code, `conflict` by the memory code - at import time of their module, through register_handler. This module
imports nothing from those packages; its own two handlers import the lexicon lazily. When an answer arrives for
a kind nobody registered yet, the modules named in HANDLER_MODULES[kind] are imported (quietly) first, so the
owner does not depend on having been imported earlier in this process.

One question per thing
----------------------
(kind, dedupe_key) identifies what is being asked. While such a question is waiting (open, queued or snoozed), a
second add() does not create another one: the occurrence is counted, its sentence joins payload.examples (five
at most), new candidates are merged and the priority rises. A key the user closed with "never ask again"
(dismiss(never=True), or "the text is right") is not asked again: add() returns 0. For kind 'term' the default
key is the pinyin of the span, so differently written same-sounding spellings are one question, and a span that
is a known lexicon entry is never asked about.

Not too many per day
--------------------
At most `inbox.daily_max` new 'term' questions become visible per day. The rest wait with status 'queued' (not
counted, not listed as open) and are released on later days, highest priority first. A snoozed question comes
back when its time is up. Both happen lazily, whenever the inbox is read or written.

Statuses: open | answered | dismissed | snoozed | queued | expired.

`term` payload (see SPEC 6.6): {"span", "sentence", "utterance_id", "start", "end", "candidates": [...], "why",
"source", "heard": {...}, "examples": [{"utterance_id", "sentence"}], "term_kind": "name|term|number|other"}.
`rule` payload: {"alias_id", "lexicon_id", "alias", "term", "count", "utterance_id", "examples": [...]}.
"""
from __future__ import annotations

import importlib
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Callable

from .db import now_iso

log = logging.getLogger(__name__)

KINDS = ("term", "voice", "hint", "conflict", "rule")
LIVE = ("open", "queued", "snoozed")
MAX_EXAMPLES = 5
MAX_CANDIDATES = 6
DEFAULT_DAILY_MAX = 12
KV_DAY = "inbox_day"            # {"day": "YYYY-MM-DD", "n": term questions made visible that day}

Handler = Callable[[Any, Any, dict[str, Any], dict[str, Any]], dict[str, Any]]
_HANDLERS: dict[str, Handler] = {}
#: kind -> modules whose import registers its handler (tried in order when no handler is registered yet).
#: The owning package may append its own module name here instead of using one of these.
HANDLER_MODULES: dict[str, list[str]] = {
    "voice": ["audio_note.speech.voices", "audio_note.speech.tracks"],
    "hint": ["audio_note.speech.scoring", "audio_note.speech.review"],
    "conflict": ["audio_note.memory.consolidate", "audio_note.memory.items"],
}


# ---------------------------------------------------------------------------------------------- plumbing
def register_handler(kind: str, fn: Handler) -> Handler:
    """Make `fn(db, cfg, item, answer) -> dict` the one who carries out answers to questions of `kind`.

    item: the question as list_items() returns it (payload parsed). answer: {"choice", "text", "as_correct",
    "apply_history", "events"} exactly as answer() received them. The returned dict is stored with the answer
    and returned to the caller. Raise ValueError (message for the user) to refuse: the question stays open."""
    _HANDLERS[str(kind)] = fn
    return fn


def handler_for(kind: str) -> Handler | None:
    """The registered handler of `kind`; on a miss the kind's owner modules are imported once more (a missing
    or broken module is skipped without noise) before giving up."""
    fn = _HANDLERS.get(kind)
    if fn is not None:
        return fn
    for name in HANDLER_MODULES.get(kind, ()):
        try:
            importlib.import_module(name)
        except Exception as e:  # noqa: BLE001 - not installed yet / failed to import: not our problem here
            log.debug("inbox: %s not importable for kind %s: %s", name, kind, e)
            continue
        fn = _HANDLERS.get(kind)
        if fn is not None:
            return fn
    return None


def _loads(v: Any, default: Any) -> Any:
    if isinstance(v, (dict, list)):
        return v
    if not v:
        return default
    try:
        out = json.loads(v)
    except (TypeError, ValueError):
        return default
    return out if isinstance(out, type(default)) else default


def _item(r: Any) -> dict[str, Any]:
    return {"id": int(r["id"]), "kind": r["kind"], "status": r["status"], "dedupe_key": r["dedupe_key"],
            "priority": float(r["priority"] or 0.0), "title": r["title"], "payload": _loads(r["payload"], {}),
            "utterance_id": r["utterance_id"], "session_id": r["session_id"], "speaker_id": r["speaker_id"],
            "lexicon_id": r["lexicon_id"], "occurrences": int(r["occurrences"] or 1),
            "answer": _loads(r["answer"], {}) or None, "snooze_until": r["snooze_until"], "created_at": r["created_at"],
            "updated_at": r["updated_at"], "answered_at": r["answered_at"]}


def _daily_max(cfg: Any) -> int:
    try:
        return max(0, int(cfg.inbox.daily_max))
    except (AttributeError, KeyError, TypeError, ValueError):
        return DEFAULT_DAILY_MAX


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _day_count(con: Any) -> int:
    row = con.execute("SELECT value FROM kv WHERE key=?", (KV_DAY,)).fetchone()
    d = _loads(row[0], {}) if row else {}
    return int(d.get("n") or 0) if d.get("day") == _today() else 0


def _set_day_count(con: Any, n: int) -> None:
    con.execute("INSERT INTO kv(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (KV_DAY, json.dumps({"day": _today(), "n": int(n)})))


def _announce(db: Any, events: Any, kind: str | None = None, **data: Any) -> None:
    if events is None:
        return
    try:
        row = db.one("SELECT COUNT(*) AS n FROM inbox_items WHERE status='open'")
        events.publish("inbox", open=int(row["n"]) if row else 0, kind=kind, **data)
    except Exception:  # noqa: BLE001
        log.debug("inbox event not published", exc_info=True)


def _merge_examples(payload: dict[str, Any], new: dict[str, Any]) -> bool:
    """Add the evidence of `new` to payload.examples (five at most). True when it is a sentence not seen yet."""
    examples = [e for e in (payload.get("examples") or []) if isinstance(e, dict)]
    seen = {e.get("utterance_id") for e in examples} | {payload.get("utterance_id")}
    fresh = False
    for e in [{"utterance_id": new.get("utterance_id"), "sentence": new.get("sentence")}] + list(new.get("examples") or []):
        if not isinstance(e, dict) or e.get("utterance_id") is None or e.get("utterance_id") in seen:
            continue
        seen.add(e.get("utterance_id"))
        fresh = True
        if len(examples) < MAX_EXAMPLES:
            examples.append({"utterance_id": e.get("utterance_id"), "sentence": e.get("sentence")})
    payload["examples"] = examples
    return fresh


def _merge_candidates(payload: dict[str, Any], new: list[Any] | None) -> None:
    span = str(payload.get("span") or "")
    out: list[str] = []
    for c in list(payload.get("candidates") or []) + list(new or []):
        c = str(c or "").strip()
        if c and c != span and c not in out:
            out.append(c)
    payload["candidates"] = out[:MAX_CANDIDATES]


def _release(db: Any, cfg: Any = None) -> int:
    """Wake what is due: snoozed questions whose time is up, queued ones while today's allowance lasts.
    Returns how many became open. Never raises (reading the inbox must not fail because of housekeeping)."""
    try:
        # reading the inbox happens all the time (status polls): look without a write lock first, and take one
        # only when something really is due
        now = now_iso()
        cap = _daily_max(cfg)
        due = db.one("SELECT 1 FROM inbox_items WHERE status='snoozed' AND (snooze_until IS NULL OR snooze_until <= ?) LIMIT 1",
                     (now,)) is not None
        waiting = db.one("SELECT 1 FROM inbox_items WHERE status='queued' LIMIT 1") is not None
        if not due and not (waiting and (not cap or _day_count(db.conn()) < cap)):
            return 0
        n = 0
        with db.transaction() as con:
            for r in con.execute("SELECT id, kind, dedupe_key FROM inbox_items WHERE status='snoozed' AND"
                                 " (snooze_until IS NULL OR snooze_until <= ?)", (now,)).fetchall():
                n += _open(con, r, now)
            used = _day_count(con)
            opened = 0
            for r in con.execute("SELECT id, kind, dedupe_key FROM inbox_items WHERE status='queued'"
                                 " ORDER BY priority DESC, id").fetchall():
                if cap and used + opened >= cap:
                    break
                got = _open(con, r, now)
                opened += got
                n += got
            if opened:
                _set_day_count(con, used + opened)
        return n
    except Exception as e:  # noqa: BLE001
        log.warning("inbox housekeeping skipped: %s: %s", type(e).__name__, e)
        return 0


def _open(con: Any, r: Any, now: str) -> int:
    """Set one waiting question to open; when another open question with the same key exists (the unique
    index), fold this one into it instead."""
    try:
        con.execute("UPDATE inbox_items SET status='open', snooze_until=NULL, updated_at=? WHERE id=?", (now, int(r["id"])))
        return 1
    except sqlite3.IntegrityError:
        con.execute("UPDATE inbox_items SET occurrences=occurrences+1, updated_at=? WHERE kind=? AND dedupe_key=? AND status='open'",
                    (now, r["kind"], r["dedupe_key"]))
        con.execute("UPDATE inbox_items SET status='expired', updated_at=? WHERE id=?", (now, int(r["id"])))
        return 0


# ---------------------------------------------------------------------------------------------- asking
def add(db: Any, kind: str, *, title: str, payload: dict[str, Any] | None, dedupe_key: str | None = None,
        priority: float = 0.0, utterance_id: int | None = None, session_id: int | None = None,
        speaker_id: int | None = None, lexicon_id: int | None = None, events: Any = None, cfg: Any = None) -> int:
    """Ask the user something. Returns the id of the question that now carries it - a new one or the waiting
    one it was merged into - or 0 when it must not be asked (see "One question per thing" above).

    `cfg` is only needed for the daily allowance of 'term' questions (inbox.daily_max; the default without)."""
    kind = str(kind)
    payload = dict(payload or {})
    title = " ".join(str(title or "").split())
    if not title:
        raise ValueError("问题不能没有标题")
    if utterance_id is None and payload.get("utterance_id") is not None:
        utterance_id = payload.get("utterance_id")
    if kind == "term":
        span = str(payload.get("span") or "").strip()
        if span:
            try:
                lexicon = importlib.import_module("audio_note.speech.lexicon")
                pinyin = importlib.import_module("audio_note.speech.pinyin")
                if lexicon.Lexicon.load(db).is_term(span):
                    return 0                       # a spelling the lexicon knows is right
                if dedupe_key is None:
                    dedupe_key = pinyin.key(span) or span
            except ImportError:
                dedupe_key = dedupe_key or span
        if utterance_id is not None and payload.get("utterance_id") is None:
            payload["utterance_id"] = utterance_id
        _merge_candidates(payload, [])
        payload.setdefault("examples", [])
    _release(db, cfg)
    now = now_iso()
    created = False
    with db.transaction() as con:
        if dedupe_key is not None:
            for r in con.execute("SELECT payload FROM inbox_items WHERE kind=? AND dedupe_key=? AND status IN ('dismissed','answered')",
                                 (kind, dedupe_key)).fetchall():
                if _loads(r["payload"], {}).get("never"):
                    return 0
            row = con.execute(f"SELECT * FROM inbox_items WHERE kind=? AND dedupe_key=? AND status IN ({','.join('?' * len(LIVE))})"
                              " ORDER BY status='open' DESC, id LIMIT 1", (kind, dedupe_key, *LIVE)).fetchone()
            if row is not None:
                old = _loads(row["payload"], {})
                fresh = _merge_examples(old, payload)
                _merge_candidates(old, payload.get("candidates"))
                con.execute("UPDATE inbox_items SET payload=?, occurrences=occurrences+?, priority=?, updated_at=? WHERE id=?",
                            (json.dumps(old, ensure_ascii=False), 1 if fresh else 0,
                             max(float(row["priority"] or 0.0), float(priority or 0.0)) + (1.0 if fresh else 0.0), now, int(row["id"])))
                item_id = int(row["id"])
            else:
                item_id = 0
        else:
            item_id = 0
        if not item_id:
            status = "open"
            if kind == "term":
                cap, used = _daily_max(cfg), _day_count(con)
                if cap and used >= cap:
                    status = "queued"              # today's allowance is used up: it waits, by priority
                else:
                    _set_day_count(con, used + 1)
            cur = con.execute(
                "INSERT INTO inbox_items(kind, status, dedupe_key, priority, title, payload, utterance_id, session_id, speaker_id,"
                " lexicon_id, occurrences, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,1,?,?)",
                (kind, status, dedupe_key, float(priority or 0.0), title, json.dumps(payload, ensure_ascii=False, default=str),
                 utterance_id, session_id, speaker_id, lexicon_id, now, now))
            item_id = int(cur.lastrowid)
            created = True
    _announce(db, events, kind, id=item_id, change="added" if created else "merged")
    return item_id


# ---------------------------------------------------------------------------------------------- reading
def get(db: Any, item_id: int) -> dict[str, Any] | None:
    r = db.one("SELECT * FROM inbox_items WHERE id=?", (int(item_id),))
    return _item(r) if r else None


def list_items(db: Any, *, status: str | None = "open", kind: str | None = None, limit: int = 50, offset: int = 0,
               cfg: Any = None) -> list[dict[str, Any]]:
    """Questions as dicts (payload / answer parsed), the most pressing first. status: one status, or "all" /
    None for every status. Keys: id, kind, status, dedupe_key, priority, title, payload, utterance_id,
    session_id, speaker_id, lexicon_id, occurrences, answer, snooze_until, created_at, updated_at, answered_at."""
    _release(db, cfg)
    where, params = [], []
    if status and status != "all":
        where.append("status=?")
        params.append(status)
    if kind:
        where.append("kind=?")
        params.append(kind)
    sql = "SELECT * FROM inbox_items" + (" WHERE " + " AND ".join(where) if where else "")
    sql += " ORDER BY priority DESC, updated_at DESC, id DESC LIMIT ? OFFSET ?"
    return [_item(r) for r in db.query(sql, [*params, max(1, min(int(limit), 500)), max(0, int(offset))])]


def counts(db: Any, cfg: Any = None) -> dict[str, Any]:
    """{"open": n, "by_kind": {kind: n}, "queued": n, "snoozed": n}; open is what the bell shows."""
    _release(db, cfg)
    by_kind: dict[str, int] = {}
    other = {"queued": 0, "snoozed": 0}
    for r in db.query("SELECT status, kind, COUNT(*) AS n FROM inbox_items WHERE status IN ('open','queued','snoozed')"
                      " GROUP BY status, kind"):
        if r["status"] == "open":
            by_kind[r["kind"]] = int(r["n"])
        else:
            other[r["status"]] += int(r["n"])
    return {"open": sum(by_kind.values()), "by_kind": by_kind, **other}


# ---------------------------------------------------------------------------------------------- answering
def _require(db: Any, item_id: int, statuses: tuple[str, ...] = LIVE) -> dict[str, Any]:
    it = get(db, item_id)
    if it is None:
        raise LookupError("没有这个问题")
    if it["status"] not in statuses:
        raise ValueError("这个问题已经处理过了")
    return it


def answer(db: Any, cfg: Any, item_id: int, *, choice: Any = None, text: str | None = None, as_correct: bool = False,
           apply_history: bool = True, events: Any = None) -> dict[str, Any]:
    """Answer a waiting question and carry the answer out through the handler of its kind.

    choice: which offered candidate (its index, or the candidate itself); text: an answer typed by the user;
    as_correct: "it is right as it is"; apply_history: also change what is already stored (not only this
    sentence and the future). What these mean exactly is up to the kind (see the handlers below).
    Returns {"id", "kind", "status": "answered", **what the handler returned}. Raises LookupError (no such
    question) / ValueError (already handled, unusable answer, no handler) - the question then stays as it was."""
    it = _require(db, item_id)
    fn = handler_for(it["kind"])
    if fn is None:
        raise ValueError("这类问题现在还不能在这里回答")
    given = {"choice": choice, "text": (text.strip() if isinstance(text, str) else text) or None,
             "as_correct": bool(as_correct), "apply_history": bool(apply_history)}
    result = fn(db, cfg, it, {**given, "events": events}) or {}
    payload = dict(it["payload"])
    if result.get("never"):
        payload["never"] = True                # "the text is right": this key is settled for good
    now = now_iso()
    db.execute("UPDATE inbox_items SET status='answered', answer=?, payload=?, answered_at=?, updated_at=?, lexicon_id=COALESCE(?, lexicon_id)"
               " WHERE id=?", (json.dumps({**given, "result": result}, ensure_ascii=False, default=str),
                               json.dumps(payload, ensure_ascii=False, default=str), now, now, result.get("lexicon_id"), it["id"]))
    _announce(db, events, it["kind"], id=it["id"], change="answered")
    return {"id": it["id"], "kind": it["kind"], "status": "answered", **result}


def dismiss(db: Any, item_id: int, *, never: bool = False, events: Any = None) -> dict[str, Any]:
    """Close a question without answering. never=True: do not ask about this (kind, key) again."""
    it = _require(db, item_id)
    payload = dict(it["payload"])
    if never:
        payload["never"] = True
    now = now_iso()
    db.execute("UPDATE inbox_items SET status='dismissed', payload=?, answered_at=?, updated_at=? WHERE id=?",
               (json.dumps(payload, ensure_ascii=False, default=str), now, now, it["id"]))
    _announce(db, events, it["kind"], id=it["id"], change="dismissed")
    return get(db, it["id"]) or it


def snooze(db: Any, item_id: int, days: float = 1, *, events: Any = None) -> dict[str, Any]:
    """Hide a question for `days` days (it comes back on its own)."""
    it = _require(db, item_id, ("open", "queued", "snoozed"))
    try:
        days = float(days)
    except (TypeError, ValueError):
        raise ValueError("天数应为数字") from None
    if not 0 < days <= 365:
        raise ValueError("天数应在 1 到 365 之间")
    until = (datetime.now() + timedelta(days=days)).isoformat(timespec="seconds")
    db.execute("UPDATE inbox_items SET status='snoozed', snooze_until=?, updated_at=? WHERE id=?", (until, now_iso(), it["id"]))
    _announce(db, events, it["kind"], id=it["id"], change="snoozed")
    return get(db, it["id"]) or it


def reopen(db: Any, cfg: Any, item_id: int, *, events: Any = None) -> dict[str, Any]:
    """Take an answer (or a dismissal / snooze) back: the question is open again.

    For an answered 'term' / 'rule' question everything the answer did is reverted: the sentences it rewrote
    get their text back (each through revisions.undo), and the alias / entry the answer created is removed.
    Returns {"id", "status": "open", "reverted": sentences restored}."""
    it = get(db, item_id)
    if it is None:
        raise LookupError("没有这个问题")
    if it["status"] == "open":
        return {"id": it["id"], "status": "open", "reverted": 0}
    reverted = 0
    if it["status"] == "answered" and it["kind"] in ("term", "rule"):
        reverted = _revert_answer(db, cfg, it, events)
    payload = dict(it["payload"])
    payload.pop("never", None)
    now = now_iso()
    try:
        db.execute("UPDATE inbox_items SET status='open', answer=NULL, answered_at=NULL, snooze_until=NULL, payload=?, updated_at=?"
                   " WHERE id=?", (json.dumps(payload, ensure_ascii=False, default=str), now, it["id"]))
    except sqlite3.IntegrityError:
        raise ValueError("已经有一个同样的问题在等你回答了") from None
    _announce(db, events, it["kind"], id=it["id"], change="reopened")
    return {"id": it["id"], "status": "open", "reverted": reverted}


def expire_for_utterances(db: Any, utterance_ids: Any, *, events: Any = None) -> int:
    """The sentences were deleted (forget, loudness clean-up): waiting questions whose only evidence they were
    are closed (status 'expired'); other questions lose them from their examples. Returns how many expired."""
    gone = {int(x) for x in utterance_ids}
    if not gone:
        return 0
    n = 0
    now = now_iso()
    with db.transaction() as con:
        for r in con.execute(f"SELECT * FROM inbox_items WHERE status IN ({','.join('?' * len(LIVE))})", LIVE).fetchall():
            p = _loads(r["payload"], {})
            ev = [e for e in (p.get("examples") or []) if isinstance(e, dict)]
            ids = {r["utterance_id"], p.get("utterance_id")} | {e.get("utterance_id") for e in ev}
            ids.discard(None)
            if not ids or not (ids & gone):
                continue
            left = [e for e in ev if e.get("utterance_id") not in gone]
            primary = r["utterance_id"] if r["utterance_id"] not in gone else None
            if primary is None and not left:
                con.execute("UPDATE inbox_items SET status='expired', updated_at=? WHERE id=?", (now, int(r["id"])))
                n += 1
                continue
            if primary is None:
                first = left.pop(0)
                primary = first.get("utterance_id")
                p["utterance_id"], p["sentence"] = primary, first.get("sentence")
                p.pop("start", None)
                p.pop("end", None)
            p["examples"] = left
            con.execute("UPDATE inbox_items SET utterance_id=?, payload=?, updated_at=? WHERE id=?",
                        (primary, json.dumps(p, ensure_ascii=False, default=str), now, int(r["id"])))
    if n:
        _announce(db, events, None, change="expired")
    return n


# ---------------------------------------------------------------------------------------------- handlers: term, rule
def _evidence_ids(item: dict[str, Any]) -> list[int]:
    p = item["payload"]
    out: list[int] = []
    for v in [p.get("utterance_id"), item.get("utterance_id")] + [e.get("utterance_id") for e in (p.get("examples") or [])
                                                                   if isinstance(e, dict)]:
        try:
            v = int(v)
        except (TypeError, ValueError):
            continue
        if v not in out:
            out.append(v)
    return out


def _pick(item: dict[str, Any], answer: dict[str, Any]) -> str | None:
    """The spelling the user chose: typed text first, else the candidate named by `choice`."""
    if answer.get("text"):
        return " ".join(str(answer["text"]).split()) or None
    choice = answer.get("choice")
    if choice is None or isinstance(choice, bool):
        return None
    cands = [str(c) for c in (item["payload"].get("candidates") or [])]
    if isinstance(choice, int) or (isinstance(choice, str) and choice.strip().isdigit() and choice.strip() not in cands):
        k = int(choice)
        if not 0 <= k < len(cands):
            raise ValueError("没有这个选项")
        return cands[k]
    return " ".join(str(choice).split()) or None


def _replace_in_sentences(db: Any, cfg: Any, ids: list[int], span: str, to: str, item: dict[str, Any], events: Any) -> list[int]:
    """Replace `span` by `to` in the given sentences only (no lexicon rule): through revisions, source 'answer'."""
    revisions = importlib.import_module("audio_note.speech.revisions")
    changed: list[int] = []
    p = item["payload"]
    for uid in ids:
        row = db.one("SELECT text FROM utterances WHERE id=?", (uid,))
        if row is None:
            continue
        text = row["text"] or ""
        hint = p.get("start") if uid == p.get("utterance_id") and isinstance(p.get("start"), int) else None
        at = text.find(span) if hint is None else min((i for i in range(len(text)) if text.startswith(span, i)),
                                                      key=lambda i: abs(i - hint), default=-1)
        if at < 0:
            continue
        new_text = text[:at] + to + text[at + len(span):]
        d = {"edits": [{"from": span, "to": to, "reason": "回答", "start": at, "end": at + len(to)}], "inbox_id": item["id"]}
        if revisions.update_text(db, cfg, uid, new_text, source="answer", detail=d, events=events, flush=False) is not None:
            changed.append(uid)
    if changed:
        revisions.flush_markdown(db, cfg)
    return changed


def _answer_term(db: Any, cfg: Any, item: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    """"Which word was this?" - span X of a sentence, with candidates.

    * a candidate / typed word Y != X: Y becomes a lexicon entry (hot word from now on) and X its mishear
      alias. With apply_history and an alias safe to apply automatically (scope 'auto') every stored X becomes
      Y; with a 'hint' alias only the question's own sentences; without apply_history only the sentence asked
      about. Everything through revisions (source 'answer', detail.inbox_id), so each change can be undone.
    * as_correct (or Y == X): X is right - it becomes a lexicon entry itself and is never asked about again.
    * numbers (term_kind 'number', or X and Y differ in their digits) are not vocabulary: only the question's
      sentences are corrected, no entry and no alias is made.
    Returns {"term", "alias", "lexicon_id", "alias_id", "scope", "changed": [utterance ids], "as_correct",
    "created_term", "created_alias", "never"?}."""
    lexicon = importlib.import_module("audio_note.speech.lexicon")
    correct = importlib.import_module("audio_note.speech.correct")
    p = item["payload"]
    span = " ".join(str(p.get("span") or "").split())
    if not span:
        raise ValueError("这个问题里没有要确认的词")
    chosen = _pick(item, answer)
    as_correct = bool(answer.get("as_correct")) or (chosen is not None and chosen == span)
    if not as_correct and not chosen:
        raise ValueError("请选一个写法，或者输入正确的写法")
    events = answer.get("events")
    term_kind = str(p.get("term_kind") or "other")
    kind = "person" if term_kind == "name" else "term"
    numeric = term_kind == "number" or (chosen is not None and correct.digit_runs(span) != correct.digit_runs(chosen))
    base = {"term": span if as_correct else chosen, "alias": None if as_correct else span, "lexicon_id": None, "alias_id": None,
            "scope": None, "changed": [], "as_correct": as_correct, "created_term": False, "created_alias": False}
    if as_correct:
        if not numeric:
            lex = lexicon.Lexicon.load(db)
            base["created_term"] = not lex.is_term(span)
            base["lexicon_id"] = lexicon.upsert_term(db, span, kind=kind, source="answer", events=events)
        return {**base, "never": True}
    evidence = _evidence_ids(item)
    if numeric:
        targets = evidence if answer.get("apply_history") else evidence[:1]
        return {**base, "changed": _replace_in_sentences(db, cfg, targets, span, chosen, item, events)}
    lex = lexicon.Lexicon.load(db)
    base["created_term"] = not lex.is_term(chosen)
    lexicon_id = lexicon.upsert_term(db, chosen, kind=kind, source="answer", events=events)
    had = {a.alias for a in (lexicon.Lexicon.load(db).by_id[lexicon_id].aliases)}
    alias_id = lexicon.add_alias(db, lexicon_id, span, source="answer", cfg=cfg, events=events)
    scope = (lexicon.get_alias(db, alias_id) or {}).get("scope", "auto")
    if answer.get("apply_history") and scope == "auto":
        targets = None                                   # the whole history
    elif answer.get("apply_history"):
        targets = evidence                               # a hint alias: only the sentences that were asked about
    else:
        targets = evidence[:1]
    # the sentences the question was about are changed whatever was undone in them before: this IS the user's word
    res = lexicon.apply_alias_to_history(db, cfg, alias_id, source="answer", detail={"inbox_id": item["id"]}, events=events,
                                         utterance_ids=targets, respect_undone=targets is None)
    return {**base, "lexicon_id": lexicon_id, "alias_id": alias_id, "scope": scope, "changed": res["changed"],
            "created_alias": span not in had}


def _answer_rule(db: Any, cfg: Any, item: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    """"Change the other N places too?" after a hand edit taught the lexicon an alias. Yes (choice 0 / "yes" /
    true, or no choice at all with apply_history) rewrites them through revisions (source 'answer'); no
    (choice 1 / "no" / false, as_correct, or apply_history=false) leaves the history alone - the alias keeps
    working for new recordings either way. Returns {"alias_id", "lexicon_id", "applied", "changed": [ids]}."""
    lexicon = importlib.import_module("audio_note.speech.lexicon")
    p = item["payload"]
    choice = answer.get("choice")
    if isinstance(choice, str):
        choice = {"yes": 0, "y": 0, "是": 0, "no": 1, "n": 1, "否": 1}.get(choice.strip().lower(), choice)
        if isinstance(choice, str) and choice.strip().isdigit():
            choice = int(choice)
    if isinstance(choice, bool):
        choice = 0 if choice else 1
    yes = not answer.get("as_correct") and bool(answer.get("apply_history")) and choice in (None, 0)
    if choice not in (None, 0, 1):
        raise ValueError("请回答「改」或「不改」")
    base = {"alias_id": p.get("alias_id"), "lexicon_id": p.get("lexicon_id"), "applied": 0, "changed": []}
    if not yes:
        return base
    try:
        res = lexicon.apply_alias_to_history(db, cfg, int(p.get("alias_id")), source="answer", detail={"inbox_id": item["id"]},
                                             events=answer.get("events"))
    except (LookupError, TypeError):
        raise ValueError("这条规则已经不在词库里了") from None
    return {**base, "applied": res["applied"], "changed": res["changed"]}


def _revert_answer(db: Any, cfg: Any, item: dict[str, Any], events: Any) -> int:
    """Undo what answering `item` did (see reopen)."""
    revisions = importlib.import_module("audio_note.speech.revisions")
    lexicon = importlib.import_module("audio_note.speech.lexicon")
    n = 0
    rows = db.query("SELECT id, utterance_id, detail FROM utterance_revisions WHERE source='answer' AND undone_at IS NULL"
                    " AND detail LIKE ? ORDER BY id DESC", (f'%"inbox_id": {int(item["id"])}%',))
    for r in rows:
        if _loads(r["detail"], {}).get("inbox_id") != item["id"]:
            continue
        try:
            revisions.undo(db, cfg, int(r["utterance_id"]), int(r["id"]), events=events, veto=False)
            n += 1
        except (LookupError, ValueError) as e:       # the sentence is gone, or was edited again since: leave it
            log.info("answer #%s: revision %s not reverted (%s)", item["id"], r["id"], e)
    result = (item.get("answer") or {}).get("result") or {}
    if item["kind"] == "term":
        try:
            if result.get("created_alias") and result.get("alias_id"):
                lexicon.delete_alias(db, int(result["alias_id"]), events=events)
            if result.get("created_term") and result.get("lexicon_id"):
                t = lexicon.get_term(db, int(result["lexicon_id"]))
                if t and not t["aliases"] and t["source"] == "answer":
                    lexicon.delete_term(db, cfg, int(t["id"]), events=events)
        except (LookupError, ValueError) as e:
            log.info("answer #%s: lexicon not reverted (%s)", item["id"], e)
    return n


register_handler("term", _answer_term)
register_handler("rule", _answer_rule)

__all__ = ["HANDLER_MODULES", "KINDS", "add", "answer", "counts", "dismiss", "expire_for_utterances", "get", "handler_for",
           "list_items", "register_handler", "reopen", "snooze"]
