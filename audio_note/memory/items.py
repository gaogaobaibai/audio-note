"""Memory items: what the system remembers beyond the raw sentences - facts, todos, decisions, people,
preferences, projects, events - each with provenance (the sentences it rests on), a state and a tombstone.
This table replaces the 0.3 `memory_notes` (the migration copied those in with source='legacy').

    upsert(db, cfg, item)         add or merge one item (the only way automatic producers write)
    search(db, cfg, query)        keyword + vector search with recency / importance
    related_facts(db, cfg, text)  short fact lines about what a text mentions (refine / digest prompts)
    update / reject / restore / merge / touch      what the user (or the chat on their behalf) does
    todos / people / person       ready-made views
    reindex(db, cfg, embedder)    tokens and vectors for items that lack them (legacy rows)

Rules that keep automatic producers from fighting the user:

* identity is `fingerprint(kind, title)` (db.memory_fingerprint: case, spaces and punctuation ignored);
* a rejected item is a tombstone: the row stays, and nothing with the same fingerprint is ever added again;
* an item the user edited (source='user') or pinned only gains evidence and a newer "last seen" day;
* an item merged into another one (status='superseded', superseded_by) forwards later sightings to its target,
  so a merge is not undone by the next digest that phrases it the old way;
* a todo that is done stays done when a later digest mentions it again.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import date, datetime, timedelta
from typing import Any, Iterable, Sequence

from ..db import DB, memory_fingerprint, now_iso
from . import index

log = logging.getLogger(__name__)

KINDS = ("fact", "todo", "decision", "person", "preference", "project", "event", "topic", "note")
STATUSES = ("active", "done", "superseded", "archived", "rejected")
SOURCES = ("auto", "digest", "user", "chat", "voice", "legacy")
KIND_LABELS = {"fact": "事实", "todo": "待办", "decision": "决定", "person": "人物", "preference": "偏好",
               "project": "项目", "event": "事件", "topic": "话题", "note": "备注"}
TITLE_MAX = 120
CONTENT_MAX = 2000
EVIDENCE_MAX = 100
#: a person item is a dated log ("2026-11-03：…" per line); this many newest lines are kept
PERSON_LINES_MAX = 12
#: columns update() may change (anything else is refused)
EDITABLE = ("kind", "title", "content", "subject", "status", "pinned", "importance", "due", "owner", "valid_from",
            "valid_to", "day", "confidence", "speaker_id", "lexicon_id")
#: half-life of the recency boost in search(), in days
RECENCY_HALF_LIFE_DAYS = 30.0

_DATED_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2})[：:]")
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def fingerprint(kind: str, title: str) -> str:
    """Identity of an item for de-duplication and tombstones (the same function the migration used)."""
    return memory_fingerprint(kind, title)


# ------------------------------------------------------------------------------------------- small helpers
def _clean(v: Any, limit: int | None = None) -> str:
    s = " ".join(str(v or "").split())
    return s[:limit] if limit else s


def _clean_block(v: Any, limit: int = CONTENT_MAX) -> str:
    """Content keeps its line structure (a person item is a list of dated lines)."""
    lines = [" ".join(ln.split()) for ln in str(v or "").replace("\r", "").split("\n")]
    return "\n".join(ln for ln in lines if ln)[:limit]


def _kind(v: Any) -> str:
    k = str(v or "").strip().lower()
    return k if k in KINDS else "note"


def _importance(v: Any, default: float = 0.5) -> float:
    """0..1. Producers that grade on the 1-5 scale of the episode prompt are mapped onto it (3 -> 0.5)."""
    if v is None or v == "":
        return default
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if x > 1.0:
        x = (min(x, 5.0) - 1.0) / 4.0
    return max(0.0, min(1.0, x))


def _evidence(v: Any) -> list[int]:
    if v is None or v == "":
        return []
    if isinstance(v, (str, bytes)):
        try:
            v = json.loads(v)
        except (TypeError, ValueError):
            return []
    out: list[int] = []
    for x in v if isinstance(v, (list, tuple, set)) else []:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if i not in out:
            out.append(i)
    return out


def _day(v: Any) -> str | None:
    s = str(v or "").strip()[:10]
    return s if _DAY.match(s) else None


def parse_due(text: Any, base_day: str | date | None = None) -> str | None:
    """A deadline as people say it ("周三", "明天", "11月6号", "2026-11-06") -> YYYY-MM-DD, or None when it is
    not a date. Relative words count from `base_day` (the day it was said); a weekday means the next one."""
    s = _clean(text)
    if not s:
        return None
    if _day(s) and len(s) >= 10:
        return _day(s)
    base = date.fromisoformat(base_day) if isinstance(base_day, str) and _day(base_day) else (
        base_day if isinstance(base_day, date) else date.today())
    for word, off in (("大后天", 3), ("后天", 2), ("明天", 1), ("明早", 1), ("明晚", 1), ("今天", 0), ("今晚", 0), ("今早", 0)):
        if word in s:
            return (base + timedelta(days=off)).isoformat()
    m = re.search(r"(下下|下|这|本)?(?:周|星期|礼拜)([一二三四五六日天])", s)
    if m:
        wd = "一二三四五六日天".index(m[2])
        wd = 6 if wd == 7 else wd
        monday = base - timedelta(days=base.weekday())
        d = monday + timedelta(days=wd)
        if m[1] == "下":
            d += timedelta(days=7)
        elif m[1] == "下下":
            d += timedelta(days=14)
        elif d < base:                       # a bare weekday that already passed this week: the next one
            d += timedelta(days=7)
        return d.isoformat()
    m = re.search(r"(?:(\d{4})[年/\-.])?(\d{1,2})[月/\-.](\d{1,2})[日号]?", s)
    if m:
        try:
            d = date(int(m[1]) if m[1] else base.year, int(m[2]), int(m[3]))
        except ValueError:
            return None
        if not m[1] and d < base - timedelta(days=30):     # "1月5号" said in December
            d = d.replace(year=d.year + 1)
        return d.isoformat()
    if "月底" in s:
        nxt = (base.replace(day=28) + timedelta(days=4)).replace(day=1)
        return (nxt - timedelta(days=1)).isoformat()
    return None


def _as_dict(r: Any) -> dict[str, Any]:
    d = dict(r)
    d["evidence"] = _evidence(d.get("evidence"))
    d["pinned"] = bool(d.get("pinned"))
    return d


def _token_text(title: Any, content: Any, subject: Any = None, owner: Any = None) -> str:
    return index.tokenize(" ".join(str(x) for x in (title, content, subject, owner) if x))


def _write_tokens(c: Any, item_id: int, title: Any, content: Any, subject: Any = None, owner: Any = None) -> None:
    c.execute("INSERT INTO memory_item_tokens(item_id, text_seg) VALUES (?,?)"
              " ON CONFLICT(item_id) DO UPDATE SET text_seg=excluded.text_seg",
              (int(item_id), _token_text(title, content, subject, owner)))


def _drop_index(c: Any, item_id: int, tokens: bool = True) -> None:
    """Take an item out of the search indexes (its vector always: the text it was computed from is gone)."""
    if tokens:
        c.execute("DELETE FROM memory_item_tokens WHERE item_id=?", (int(item_id),))
    c.execute("DELETE FROM memory_vectors WHERE kind='item' AND ref_id=?", (int(item_id),))


def _merge_content(kind: str, old: str, new: str) -> str:
    """Content of an automatic item seen again. A person item is a log of dated lines: a line for a day that
    is already there replaces it, a new day is appended (the 0.3 notes overwrote the person every day).
    Everything else: the newer wording wins unless it says nothing new."""
    old, new = _clean_block(old), _clean_block(new)
    if not new or new == old:
        return old
    if not old:
        return new
    if kind == "person":
        lines = old.split("\n")
        for ln in new.split("\n"):
            m = _DATED_LINE.match(ln)
            if m:
                lines = [x for x in lines if not x.startswith(m[1])]
            if ln not in lines:
                lines.append(ln)
        return "\n".join(lines[-PERSON_LINES_MAX:])[:CONTENT_MAX]
    return old if new in old else new


def _resolve(c: Any, row: Any) -> Any:
    """Follow superseded_by to the item that now stands for `row` (a few hops at most; loops end the walk)."""
    seen = {int(row["id"])}
    while row["status"] == "superseded" and row["superseded_by"]:
        nxt = c.execute("SELECT * FROM memory_items WHERE id=?", (int(row["superseded_by"]),)).fetchone()
        if nxt is None or int(nxt["id"]) in seen:
            break
        seen.add(int(nxt["id"]))
        row = nxt
    return row


# ------------------------------------------------------------------------------------------- reading
def get(db: DB, item_id: int) -> dict[str, Any] | None:
    r = db.one("SELECT * FROM memory_items WHERE id=?", (int(item_id),))
    return _as_dict(r) if r is not None else None


def _where(kinds: Any, status: Any, subject: str | None, alias: str = "") -> tuple[str, list[Any]]:
    p = f"{alias}." if alias else ""
    sql, params = "", []
    kinds = [kinds] if isinstance(kinds, str) else list(kinds or [])
    if kinds:
        sql += f" AND {p}kind IN ({','.join('?' * len(kinds))})"
        params += [str(k) for k in kinds]
    if status is None or status == "all" or status == ("all",):
        sql += f" AND {p}status <> 'rejected'"              # tombstones are never listed unless asked for by name
    else:
        st = [status] if isinstance(status, str) else list(status)
        sql += f" AND {p}status IN ({','.join('?' * len(st))})"
        params += [str(s) for s in st]
    if subject:
        sql += f" AND {p}subject = ?"
        params.append(str(subject))
    return sql, params


def list_items(db: DB, *, kinds: Any = None, status: Any = ("active",), subject: str | None = None, limit: int = 100,
               offset: int = 0) -> list[dict[str, Any]]:
    """Items newest first (pinned ones on top). `status`: a name, a list of names, or "all" (= not rejected)."""
    wsql, params = _where(kinds, status, subject)
    rows = db.query("SELECT * FROM memory_items WHERE 1=1" + wsql +
                    " ORDER BY pinned DESC, COALESCE(day, substr(updated_at, 1, 10)) DESC, updated_at DESC, id DESC LIMIT ? OFFSET ?",
                    [*params, max(1, int(limit)), max(0, int(offset))])
    return [_as_dict(r) for r in rows]


def counts(db: DB) -> dict[str, Any]:
    """{"total": active + done items, "by_kind": {kind: n}, "by_status": {status: n}} (tombstones only in by_status)."""
    by_status = {r["status"]: int(r["n"]) for r in db.query("SELECT status, COUNT(*) AS n FROM memory_items GROUP BY status")}
    by_kind = {r["kind"]: int(r["n"]) for r in db.query(
        "SELECT kind, COUNT(*) AS n FROM memory_items WHERE status IN ('active', 'done') GROUP BY kind")}
    return {"total": sum(by_kind.values()), "by_kind": by_kind, "by_status": by_status}


def format_line(item: Any, content_chars: int = 200) -> str:
    """'[todo] 标题 (2026-10-28~2026-10-29): 内容' - the one-line form the chat prompt and the tools use."""
    first, last = item["valid_from"] or item["day"] or "", item["day"] or ""
    span = first + (f"~{last}" if last and last != first else "")
    content = " ".join(str(item["content"] or "").split())[:content_chars]
    return f"[{item['kind']}] {item['title']}{(' (' + span + ')') if span else ''}: {content}"


# ------------------------------------------------------------------------------------------- writing
def upsert(db: DB, cfg: Any, item: dict[str, Any], *, source: str = "auto", embedder: Any = None) -> tuple[int, str]:
    """Add one item or fold it into the item with the same fingerprint. Returns (id, "added" | "updated" | "skipped").

    item: {"kind", "title"} plus optional "content", "subject", "evidence" (utterance ids), "day" (the day it
    was seen), "valid_from", "valid_to", "due", "owner", "session_id", "importance" (0..1, or 1-5), "confidence",
    "pinned", "status", "speaker_id", "lexicon_id".

    * tombstone (status 'rejected')            -> skipped, nothing changes;
    * merged away / replaced ('superseded')    -> skipped (the id of the item that stands for it is returned);
    * edited by the user or pinned             -> only evidence and the last-seen day are added;
    * otherwise                                -> content / fields refreshed, evidence united, days widened.

    source='user' (the user adds an item by hand) overrides all of that: what they typed wins, and a
    tombstone or merged-away row with the same title comes back to life.

    Tokens for the keyword index are always written; the vector only when an `embedder` is passed (search()
    and reindex() fill missing vectors later)."""
    kind = _kind(item.get("kind"))
    title = _clean(item.get("title"), TITLE_MAX)
    if not title:
        raise ValueError("记忆条目需要标题")
    content = _clean_block(item.get("content"))
    subject = _clean(item.get("subject"), 80) or None
    owner = _clean(item.get("owner"), 80) or None
    day = _day(item.get("day"))
    due = _day(item.get("due")) or parse_due(item.get("due"), day)
    evidence = _evidence(item.get("evidence"))
    source = source if source in SOURCES else "auto"
    fp = fingerprint(kind, title)
    now = now_iso()
    with db.transaction() as c:
        by_user = source == "user"
        # an active row first for the user (they edit what they see), a tombstone first for everybody else
        rows = c.execute("SELECT * FROM memory_items WHERE fingerprint=? ORDER BY (status='rejected') "
                         + ("ASC" if by_user else "DESC") + ", (status='superseded'), id", (fp,)).fetchall()
        if rows and not by_user and rows[0]["status"] in ("rejected", "superseded"):
            return int(_resolve(c, rows[0])["id"]), "skipped"
        if rows:
            old = rows[0]
            oid = int(old["id"])
            ev = _evidence(old["evidence"])
            merged_ev = (ev + [e for e in evidence if e not in ev])[-EVIDENCE_MAX:]
            new_day = max(d for d in (old["day"], day) if d) if (old["day"] or day) else None
            first = min(d for d in (old["valid_from"], old["day"], item.get("valid_from") and _day(item.get("valid_from")), day)
                        if d) if (old["valid_from"] or old["day"] or day) else None
            if (old["source"] == "user" or bool(old["pinned"])) and not by_user:
                if merged_ev == ev and new_day == old["day"]:
                    return oid, "skipped"
                c.execute("UPDATE memory_items SET evidence=?, day=?, valid_from=?, updated_at=? WHERE id=?",
                          (json.dumps(merged_ev) if merged_ev else None, new_day, first, now, oid))
                return oid, "updated"
            new_content = (content or old["content"]) if by_user else _merge_content(kind, old["content"], content)
            status = "active" if by_user and old["status"] in ("rejected", "superseded") else old["status"]
            if item.get("status") == "done" and status == "active":
                status = "done"
            c.execute(
                "UPDATE memory_items SET title=?, content=?, subject=?, owner=?, due=?, evidence=?, day=?, valid_from=?,"
                " valid_to=?, importance=?, confidence=?, session_id=?, speaker_id=?, lexicon_id=?, status=?, source=?,"
                " pinned=?, superseded_by=?, updated_at=? WHERE id=?",
                (title if by_user else old["title"], new_content, subject or old["subject"], owner or old["owner"],
                 due or old["due"],
                 json.dumps(merged_ev) if merged_ev else None, new_day, first, _day(item.get("valid_to")) or old["valid_to"],
                 max(float(old["importance"] or 0.0), _importance(item.get("importance"), 0.0)) or 0.5,
                 item.get("confidence") if item.get("confidence") is not None else old["confidence"],
                 item.get("session_id") if item.get("session_id") is not None else old["session_id"],
                 item.get("speaker_id") if item.get("speaker_id") is not None else old["speaker_id"],
                 item.get("lexicon_id") if item.get("lexicon_id") is not None else old["lexicon_id"],
                 status, "user" if by_user else old["source"], 1 if (old["pinned"] or item.get("pinned")) else 0,
                 old["superseded_by"] if status == "superseded" else None, now, oid))
            if (new_content != old["content"] or (subject and subject != old["subject"]) or (owner and owner != old["owner"])
                    or status != old["status"]):
                _write_tokens(c, oid, title if by_user else old["title"], new_content, subject or old["subject"],
                              owner or old["owner"])
                c.execute("DELETE FROM memory_vectors WHERE kind='item' AND ref_id=?", (oid,))
            item_id, action = oid, "updated"
        else:
            status = str(item.get("status") or "active")
            status = status if status in ("active", "done", "archived") else "active"
            cur = c.execute(
                "INSERT INTO memory_items(kind, title, content, subject, lexicon_id, speaker_id, status, source, confidence,"
                " importance, pinned, day, valid_from, valid_to, due, owner, session_id, evidence, fingerprint, created_at,"
                " updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (kind, title, content, subject, item.get("lexicon_id"), item.get("speaker_id"), status, source,
                 item.get("confidence"), _importance(item.get("importance")), 1 if item.get("pinned") else 0, day,
                 _day(item.get("valid_from")) or day, _day(item.get("valid_to")), due, owner, item.get("session_id"),
                 json.dumps(evidence[-EVIDENCE_MAX:]) if evidence else None, fp, now, now))
            item_id, action = int(cur.lastrowid), "added"
            _write_tokens(c, item_id, title, content, subject, owner)
    if embedder is not None:
        _embed(db, embedder, [item_id])
    return item_id, action


def _embed(db: DB, embedder: Any, ids: Sequence[int]) -> int:
    """(Re)compute the vectors of the given items; failures only cost recall and are logged."""
    from . import vectors

    rows = db.query("SELECT id, title, content FROM memory_items WHERE status NOT IN ('rejected', 'superseded')"
                    f" AND id IN ({','.join('?' * len(ids))})",
                    [int(i) for i in ids]) if ids else []
    if not rows:
        return 0
    try:
        return vectors.put(db, "item", embedder.name,
                           zip((int(r["id"]) for r in rows), embedder.embed_passages([vectors.item_text(r) for r in rows])))
    except Exception as e:  # noqa: BLE001
        log.warning("memory item vectors not written: %s", e)
        return 0


def update(db: DB, item_id: int, **fields: Any) -> dict[str, Any]:
    """The user edits an item: any of EDITABLE (title, content, subject, kind, status active / done / archived,
    pinned, importance, due, owner, ...). The item becomes source='user', which automatic producers respect.
    Raises LookupError for an unknown id and ValueError for a bad field / value or a title another item has."""
    row = db.one("SELECT * FROM memory_items WHERE id=?", (int(item_id),))
    if row is None:
        raise LookupError("没有这条记忆")
    unknown = sorted(set(fields) - set(EDITABLE))
    if unknown:
        raise ValueError(f"不能修改这些字段：{', '.join(unknown)}")
    new: dict[str, Any] = {}
    for k, v in fields.items():
        if k == "kind":
            if str(v or "").strip().lower() not in KINDS:
                raise ValueError(f"类型必须是 {' / '.join(KINDS)} 之一")
            new[k] = str(v).strip().lower()
        elif k == "title":
            title = _clean(v, TITLE_MAX)
            if not title:
                raise ValueError("标题不能为空")
            new[k] = title
        elif k == "content":
            new[k] = _clean_block(v)
        elif k in ("subject", "owner"):
            new[k] = _clean(v, 80) or None
        elif k == "status":
            if v not in ("active", "done", "archived"):
                raise ValueError("状态只能改成 active / done / archived（删除请用删除操作）")
            new[k] = v
        elif k == "pinned":
            new[k] = 1 if v else 0
        elif k == "importance":
            new[k] = _importance(v)
        elif k in ("due", "valid_from", "valid_to", "day"):
            if v in (None, ""):
                new[k] = None
            else:
                d = _day(v) or (parse_due(v) if k == "due" else None)
                if d is None:
                    raise ValueError(f"{k} 应为 YYYY-MM-DD")
                new[k] = d
        else:
            new[k] = v
    kind, title = new.get("kind", row["kind"]), new.get("title", row["title"])
    fp = fingerprint(kind, title)
    with db.transaction() as c:
        if fp != row["fingerprint"]:
            clash = c.execute("SELECT id, status FROM memory_items WHERE fingerprint=? AND id<>? AND status<>'superseded'",
                              (fp, int(item_id))).fetchone()
            if clash is not None and clash["status"] != "rejected":
                raise ValueError("已经有一条同名的记忆了，可以把两条合并")
        sets = ", ".join(f"{k}=?" for k in new)
        c.execute(f"UPDATE memory_items SET {sets + ', ' if sets else ''}fingerprint=?, source='user', updated_at=? WHERE id=?",
                  [*new.values(), fp, now_iso(), int(item_id)])
        after = c.execute("SELECT * FROM memory_items WHERE id=?", (int(item_id),)).fetchone()
        if {"title", "content", "subject", "owner"} & set(new):
            _write_tokens(c, int(item_id), after["title"], after["content"], after["subject"], after["owner"])
            c.execute("DELETE FROM memory_vectors WHERE kind='item' AND ref_id=?", (int(item_id),))
    return _as_dict(after)


def reject(db: DB, item_id: int) -> None:
    """Delete an item for good: it becomes a tombstone (status 'rejected') that no automatic producer brings
    back, and leaves the search indexes. Its text stays in the row so restore() can undo the delete;
    memory.forget scrubs the text as well. Raises LookupError for an unknown id."""
    with db.transaction() as c:
        cur = c.execute("UPDATE memory_items SET status='rejected', pinned=0, updated_at=? WHERE id=?", (now_iso(), int(item_id)))
        if not cur.rowcount:
            raise LookupError("没有这条记忆")
        _drop_index(c, int(item_id))


def restore(db: DB, item_id: int) -> dict[str, Any]:
    """Undo reject() (or bring an archived / done item back): status 'active', indexed again."""
    with db.transaction() as c:
        row = c.execute("SELECT * FROM memory_items WHERE id=?", (int(item_id),)).fetchone()
        if row is None:
            raise LookupError("没有这条记忆")
        if not str(row["title"] or "").strip():
            raise ValueError("这条记忆的内容已经被彻底清除，无法恢复")
        c.execute("UPDATE memory_items SET status='active', superseded_by=NULL, updated_at=? WHERE id=?", (now_iso(), int(item_id)))
        _write_tokens(c, int(item_id), row["title"], row["content"], row["subject"], row["owner"])
        after = c.execute("SELECT * FROM memory_items WHERE id=?", (int(item_id),)).fetchone()
    return _as_dict(after)


def merge(db: DB, keep_id: int, drop_id: int) -> dict[str, Any]:
    """Fold `drop_id` into `keep_id`: evidence united, the dropped wording appended when it says something else,
    days / importance / hits combined; the dropped row stays as status 'superseded' pointing at the kept one,
    so later sightings of its wording land on the kept item. Returns the kept item."""
    keep_id, drop_id = int(keep_id), int(drop_id)
    if keep_id == drop_id:
        raise ValueError("不能把一条记忆和它自己合并")
    with db.transaction() as c:
        keep = c.execute("SELECT * FROM memory_items WHERE id=?", (keep_id,)).fetchone()
        drop = c.execute("SELECT * FROM memory_items WHERE id=?", (drop_id,)).fetchone()
        if keep is None or drop is None:
            raise LookupError("没有这条记忆")
        if keep["status"] in ("rejected", "superseded") or drop["status"] == "rejected":
            raise ValueError("已删除或已合并的记忆不能再合并")
        ev = _evidence(keep["evidence"])
        ev = (ev + [e for e in _evidence(drop["evidence"]) if e not in ev])[-EVIDENCE_MAX:]
        content = _clean_block(keep["content"])
        extra = _clean_block(drop["content"]) or _clean(drop["title"])
        if keep["kind"] == "person":
            content = _merge_content("person", content, extra)
        elif extra and extra not in content and extra != _clean(keep["title"]):
            content = (content + "\n" + extra).strip()[:CONTENT_MAX]
        days = [d for d in (keep["day"], drop["day"]) if d]
        firsts = [d for d in (keep["valid_from"], drop["valid_from"], keep["day"], drop["day"]) if d]
        c.execute(
            "UPDATE memory_items SET content=?, evidence=?, day=?, valid_from=?, importance=?, pinned=?, hits=?, due=?, owner=?,"
            " subject=?, updated_at=? WHERE id=?",
            (content, json.dumps(ev) if ev else None, max(days) if days else None, min(firsts) if firsts else None,
             max(float(keep["importance"] or 0), float(drop["importance"] or 0)), 1 if (keep["pinned"] or drop["pinned"]) else 0,
             int(keep["hits"] or 0) + int(drop["hits"] or 0), keep["due"] or drop["due"], keep["owner"] or drop["owner"],
             keep["subject"] or drop["subject"], now_iso(), keep_id))
        c.execute("UPDATE memory_items SET status='superseded', superseded_by=?, pinned=0, updated_at=? WHERE id=?",
                  (keep_id, now_iso(), drop_id))
        c.execute("UPDATE memory_items SET superseded_by=? WHERE superseded_by=?", (keep_id, drop_id))
        _drop_index(c, drop_id)
        after = c.execute("SELECT * FROM memory_items WHERE id=?", (keep_id,)).fetchone()
        _write_tokens(c, keep_id, after["title"], after["content"], after["subject"], after["owner"])
        c.execute("DELETE FROM memory_vectors WHERE kind='item' AND ref_id=?", (keep_id,))
    return _as_dict(after)


def touch(db: DB, item_ids: Iterable[int]) -> int:
    """Count a use (the chat cited the item): hits += 1, last_used_at = now. Returns the rows touched."""
    ids = sorted({int(i) for i in item_ids})
    if not ids:
        return 0
    cur = db.execute(f"UPDATE memory_items SET hits = hits + 1, last_used_at=? WHERE id IN ({','.join('?' * len(ids))})",
                     [now_iso(), *ids])
    return int(cur.rowcount or 0)


# ------------------------------------------------------------------------------------------- indexes
def reindex_tokens(db: DB, force: bool = False) -> int:
    """Keyword tokens for items that have none (all items with `force`: jieba learned new words)."""
    sql = ("SELECT i.id, i.title, i.content, i.subject, i.owner FROM memory_items i"
           " LEFT JOIN memory_item_tokens t ON t.item_id = i.id WHERE i.status NOT IN ('rejected', 'superseded')")
    rows = db.query(sql if force else sql + " AND t.item_id IS NULL")
    if rows:
        with db.transaction() as c:
            for r in rows:
                _write_tokens(c, int(r["id"]), r["title"], r["content"], r["subject"], r["owner"])
    return len(rows)


def reindex(db: DB, cfg: Any = None, embedder: Any = None, *, force: bool = False) -> dict[str, int]:
    """Bring the item indexes up to date: tokens for items without any, vectors for items without one for
    `embedder` (skipped when no embedder is given). The legacy notes the migration copied start with neither."""
    out = {"tokens": reindex_tokens(db, force=force), "vectors": 0}
    if embedder is not None:
        from . import vectors

        try:
            out["vectors"] = vectors.index_items(db, embedder)
        except Exception as e:  # noqa: BLE001
            log.warning("memory item vectors not built: %s", e)
    return out


# ------------------------------------------------------------------------------------------- search
def min_similarity(embedder: Any) -> float:
    """Cosine below which a vector neighbour of a question is not "about" it. Depends on the model: the real
    sentence model puts unrelated short Chinese texts around 0.3-0.45, the hash embedder of the tests near 0."""
    v = getattr(embedder, "min_item_similarity", None)
    if v is not None:
        return float(v)
    return 0.25 if getattr(embedder, "name", "") == "hash" else 0.5


def _recency(day: Any, updated_at: Any, today: date) -> float:
    s = str(day or updated_at or "")[:10]
    try:
        age = max(0, (today - date.fromisoformat(s)).days)
    except ValueError:
        return 0.0
    return 0.5 ** (age / RECENCY_HALF_LIFE_DAYS)


def search(db: DB, cfg: Any, query: str, *, kinds: Any = None, status: Any = ("active",), subject: str | None = None,
           limit: int = 20, embedder: Any = None, now: datetime | date | None = None) -> list[dict[str, Any]]:
    """Items about `query`, best first: the keyword list (content words of the query, lexicon aliases included)
    and the vector list (only neighbours above min_similarity) are fused by reciprocal rank, then important,
    pinned and recent items are nudged up; substring matches on title / content / subject join at half weight.
    Each result is the item dict plus "score" and "match" ("fts", "vector", "like" or a "+"-joined combination).
    An empty query lists the newest items.

    Without `embedder` only keywords are used; with one, items that have no vector yet are embedded first."""
    query = (query or "").strip()
    limit = max(1, int(limit))
    if not query:
        return list_items(db, kinds=kinds, status=status, subject=subject, limit=limit)
    if db.one("SELECT 1 FROM memory_items i LEFT JOIN memory_item_tokens t ON t.item_id = i.id"
              " WHERE t.item_id IS NULL AND i.status NOT IN ('rejected', 'superseded') LIMIT 1") is not None:
        reindex_tokens(db)
    wsql, wparams = _where(kinds, status, subject, alias="i")
    cand = max(3 * limit, 30)
    ranked: dict[int, float] = {}
    how: dict[int, list[str]] = {}
    rrf_k = 60
    try:
        rrf_k = max(1, int(cfg.memory.get("rrf_k", 60)))
    except Exception:  # noqa: BLE001 - a bare config in a test
        pass

    def add(ids: list[int], label: str, weight: float = 1.0) -> None:
        for rank, i in enumerate(ids, start=1):
            ranked[i] = ranked.get(i, 0.0) + weight / (rrf_k + rank)
            how.setdefault(i, [])
            if label not in how[i]:
                how[i].append(label)

    terms = index.query_terms(query, db)
    if terms:
        try:
            rows = db.query(
                "SELECT i.id FROM memory_items_fts JOIN memory_items i ON i.id = memory_items_fts.rowid"
                " WHERE memory_items_fts MATCH ?" + wsql + " ORDER BY bm25(memory_items_fts), i.id LIMIT ?",
                [" OR ".join(t.match() for t in terms), *wparams, cand])
            add([int(r["id"]) for r in rows], "fts")
        except sqlite3.OperationalError as e:
            log.warning("memory item keyword query failed: %s", e)
    if embedder is not None:
        from . import vectors

        try:
            vectors.index_items(db, embedder)
            allowed = [int(r["id"]) for r in db.query("SELECT i.id FROM memory_items i WHERE 1=1" + wsql, wparams)]
            floor = min_similarity(embedder)
            add([i for i, sim in vectors.nearest(db, embedder, "item", query, cand, ids=allowed) if sim >= floor], "vector")
        except Exception as e:  # noqa: BLE001 - the keyword list still answers
            log.warning("memory item vector search failed: %s", e)
    # substring matches on the visible text: what jieba cut differently ("预算" inside "预算表"), and - when nothing
    # else matched - a query that is one character or has no content word at all
    needles = [t.text for t in terms if len(t.text) >= 2] or ([] if ranked else [query])
    if needles:
        def pat(s: str) -> str:
            return "%" + s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"

        cond = " OR ".join("i.title LIKE ? ESCAPE '\\' OR i.content LIKE ? ESCAPE '\\' OR i.subject LIKE ? ESCAPE '\\'" for _ in needles)
        rows = db.query(f"SELECT i.id FROM memory_items i WHERE ({cond})" + wsql + " ORDER BY i.updated_at DESC, i.id DESC LIMIT ?",
                        [p for n in needles for p in (pat(n),) * 3] + [*wparams, cand])
        add([int(r["id"]) for r in rows], "like", 0.5)
    if not ranked:
        return []
    today = now.date() if isinstance(now, datetime) else (now or date.today())
    ids = list(ranked)
    rows = db.query(f"SELECT * FROM memory_items WHERE id IN ({','.join('?' * len(ids))})", ids)
    out = []
    for r in rows:
        d = _as_dict(r)
        boost = (0.75 + 0.5 * float(r["importance"] or 0.0)) * (0.85 + 0.15 * _recency(r["day"], r["updated_at"], today))
        d["score"] = round(ranked[int(r["id"])] * boost * (1.1 if r["pinned"] else 1.0), 6)
        d["match"] = "+".join(how[int(r["id"])])
        out.append(d)
    out.sort(key=lambda d: (-d["score"], -int(d["id"])))
    return out[:limit]


def related_facts(db: DB, cfg: Any, text: str, limit: int = 8) -> list[str]:
    """Short lines of what memory knows about the things `text` mentions, for prompts that correct or
    summarise that text ("[事实] 标题：内容（2026-11-03）"). Cheap (no model): items whose subject occurs
    in the text - under any spelling the lexicon knows - come first, then items sharing a name, number or
    Latin term with it."""
    text = str(text or "")
    limit = max(0, int(limit))
    if not text.strip() or limit == 0:
        return []
    low = text.lower()
    names: set[str] = set()
    for _matched, group in index.alias_groups(db, text):
        names.update(group)
    picked: list[Any] = []
    seen: set[int] = set()

    def take(rows: Iterable[Any]) -> None:
        for r in rows:
            if int(r["id"]) not in seen and len(picked) < limit:
                seen.add(int(r["id"]))
                picked.append(r)

    subjects = [str(r["subject"]) for r in db.query(
        "SELECT DISTINCT subject FROM memory_items WHERE status='active' AND COALESCE(subject, '') <> ''")]
    wanted = [s for s in subjects if (len(s) >= 2 and s.lower() in low) or s in names]
    if wanted:
        take(db.query(f"SELECT * FROM memory_items WHERE status='active' AND subject IN ({','.join('?' * len(wanted))})"
                      " ORDER BY pinned DESC, importance DESC, COALESCE(day, '') DESC, id DESC LIMIT ?", [*wanted, limit]))
    if len(picked) < limit:
        strong = [t for t in index.query_terms(text[:4000], db) if t.strong][:40]
        if strong:
            try:
                take(db.query(
                    "SELECT i.* FROM memory_items_fts JOIN memory_items i ON i.id = memory_items_fts.rowid"
                    " WHERE memory_items_fts MATCH ? AND i.status='active' ORDER BY bm25(memory_items_fts), i.id LIMIT ?",
                    [" OR ".join(t.match() for t in strong), limit * 2]))
            except sqlite3.OperationalError as e:
                log.debug("related_facts keyword query failed: %s", e)
    out = []
    for r in picked:
        title = _clean(r["title"])
        content = _clean(r["content"], 100)
        body = title if not content or content == title or content in title else f"{title}：{content}"
        out.append(f"[{KIND_LABELS.get(r['kind'], r['kind'])}] {body[:140]}" + (f"（{r['day']}）" if r["day"] else ""))
    return out


# ------------------------------------------------------------------------------------------- todos
def todos(db: DB, *, status: Any = "active", due_before: str | None = None) -> list[dict[str, Any]]:
    """Todo items: those with a deadline first (earliest on top), then by importance and recency.
    `status`: 'active' (open), 'done', or "all"; `due_before`: only todos due on or before that day."""
    wsql, params = _where("todo", status, None)
    if due_before:
        d = _day(due_before)
        if d is None:
            raise ValueError("due_before 应为 YYYY-MM-DD")
        wsql += " AND due IS NOT NULL AND due <= ?"
        params.append(d)
    rows = db.query("SELECT * FROM memory_items WHERE 1=1" + wsql +
                    " ORDER BY (due IS NULL), due, pinned DESC, importance DESC, COALESCE(day, '') DESC, id DESC", params)
    return [_as_dict(r) for r in rows]


# ------------------------------------------------------------------------------------------- people
def _people_index(db: DB, cfg: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Everybody the system knows about, merged by alias: enrolled speakers, person entries of the lexicon
    (with their aliases) and the people memory items are about. Returns (entries, {lower-cased name: entry})."""
    entries: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}

    def link(e: dict[str, Any], name: Any) -> None:
        n = _clean(name)
        if n:
            by_name.setdefault(n.lower(), e)
            if n != e["name"] and n not in e["aliases"]:
                e["aliases"].append(n)

    def new(name: str) -> dict[str, Any]:
        e = {"name": name, "aliases": [], "speaker_id": None, "is_self": False, "relation": None, "relation_source": None,
             "relation_note": None, "lexicon_id": None, "note": None, "utterances": 0}
        entries.append(e)
        by_name.setdefault(name.lower(), e)
        return e

    for p in db.people_table():
        e = new(str(p["name"]))
        e.update(speaker_id=int(p["id"]), is_self=bool(p.get("is_self")), relation=p.get("relation"),
                 relation_source=p.get("relation_source"), relation_note=p.get("relation_note"),
                 utterances=int(p.get("utterances") or 0))
    by_speaker = {e["speaker_id"]: e for e in entries}
    try:
        lex = db.query("SELECT id, term, note, speaker_id FROM lexicon WHERE kind='person' AND status <> 'disabled' ORDER BY id")
        aliases: dict[int, list[str]] = {}
        for a in db.query("SELECT a.lexicon_id, a.alias FROM lexicon_aliases a JOIN lexicon l ON l.id = a.lexicon_id"
                          " WHERE l.kind='person' AND l.status <> 'disabled' ORDER BY a.id"):
            aliases.setdefault(int(a["lexicon_id"]), []).append(str(a["alias"]))
    except sqlite3.Error:
        lex, aliases = [], {}
    for r in lex:
        names = [str(r["term"]), *aliases.get(int(r["id"]), [])]
        e = by_speaker.get(r["speaker_id"]) if r["speaker_id"] is not None else None
        if e is None:
            e = next((by_name[n.lower()] for n in names if n.lower() in by_name), None)
        if e is None:
            e = new(str(r["term"]))
        e["lexicon_id"] = e["lexicon_id"] or int(r["id"])
        e["note"] = e["note"] or r["note"]
        for n in names:
            link(e, n)
    for r in db.query("SELECT DISTINCT title, subject FROM memory_items WHERE kind='person' AND status IN ('active', 'done')"):
        for n in (r["subject"], r["title"]):
            n = _clean(n)
            if n and n.lower() not in by_name:
                new(n)
                break
    return entries, by_name


def _names(e: dict[str, Any]) -> list[str]:
    return [e["name"], *e["aliases"]]


def people(db: DB, cfg: Any) -> list[dict[str, Any]]:
    """People profiles, the user first: {"name", "aliases", "speaker_id", "is_self", "relation", "relation_source",
    "relation_note", "lexicon_id", "note", "utterances", "speech_s", "last_seen", "facts", "todos"} ("facts" /
    "todos" are counts of active items about / owned by the person)."""
    entries, _ = _people_index(db, cfg)
    speech = {r["speaker_id"]: r for r in db.query(
        "SELECT speaker_id, SUM(end_s - start_s) AS s, MAX(day) AS last FROM utterances WHERE speaker_id IS NOT NULL"
        " GROUP BY speaker_id")}
    items = db.query("SELECT kind, title, subject, owner, day FROM memory_items WHERE status='active'")
    for e in entries:
        names = {n.lower() for n in _names(e)}
        sp = speech.get(e["speaker_id"])
        e["speech_s"] = round(float(sp["s"] or 0.0), 1) if sp is not None else 0.0
        last = sp["last"] if sp is not None else None
        facts = todos_n = 0
        for it in items:
            about = _clean(it["subject"]).lower() in names or (it["kind"] == "person" and _clean(it["title"]).lower() in names)
            owns = it["kind"] == "todo" and _clean(it["owner"]).lower() in names
            if it["kind"] == "todo" and (about or owns):
                todos_n += 1
            elif about:
                facts += 1
            if (about or owns) and it["day"] and (last is None or it["day"] > last):
                last = it["day"]
        e.update(facts=facts, todos=todos_n, last_seen=last)
    entries.sort(key=lambda e: (not e["is_self"], -e["utterances"], -(e["facts"] + e["todos"]), e["name"]))
    return entries


def person(db: DB, cfg: Any, name: str) -> dict[str, Any]:
    """One person's profile by any of their names: the people() entry plus "facts" and "todos" (item dicts),
    "recent_sessions" (the newest sessions they spoke in or were mentioned in), "first_seen", "days".
    Raises LookupError when nobody is known under that name."""
    entries, by_name = _people_index(db, cfg)
    e = by_name.get(_clean(name).lower())
    if e is None:
        raise LookupError(f"没有找到「{_clean(name)}」这个人")
    names = _names(e)
    low = {n.lower() for n in names}
    out = dict(e)
    facts: list[dict[str, Any]] = []
    todo_list: list[dict[str, Any]] = []
    for r in db.query("SELECT * FROM memory_items WHERE status IN ('active', 'done')"
                      " ORDER BY pinned DESC, COALESCE(day, '') DESC, importance DESC, id DESC"):
        about = _clean(r["subject"]).lower() in low or (r["kind"] == "person" and _clean(r["title"]).lower() in low)
        owns = r["kind"] == "todo" and _clean(r["owner"]).lower() in low
        if r["kind"] == "todo" and (about or owns):
            todo_list.append(_as_dict(r))
        elif about and r["status"] == "active":
            facts.append(_as_dict(r))
    todo_list.sort(key=lambda t: t["status"] != "active")      # open ones first (the sort is stable)
    out["facts"], out["todos"] = facts, todo_list
    sid = e["speaker_id"]
    if sid is not None:
        st = db.one("SELECT COUNT(*) AS n, SUM(end_s - start_s) AS s, MIN(day) AS first, MAX(day) AS last,"
                    " COUNT(DISTINCT day) AS days FROM utterances WHERE speaker_id=?", (sid,))
        out.update(utterances=int(st["n"] or 0), speech_s=round(float(st["s"] or 0.0), 1), first_seen=st["first"],
                   last_seen=st["last"], days=int(st["days"] or 0))
        srows = db.query("SELECT session_id, MAX(abs_start) AS t FROM utterances WHERE speaker_id=? AND session_id IS NOT NULL"
                         " GROUP BY session_id ORDER BY t DESC LIMIT 5", (sid,))
    else:
        like = " OR ".join("text LIKE ?" for _ in names)
        pats = ["%" + n.replace("%", "").replace("_", "") + "%" for n in names]
        st = db.one(f"SELECT COUNT(*) AS n, MIN(day) AS first, MAX(day) AS last, COUNT(DISTINCT day) AS days FROM utterances"
                    f" WHERE {like}", pats)
        out.update(utterances=0, speech_s=0.0, mentions=int(st["n"] or 0), first_seen=st["first"], last_seen=st["last"],
                   days=int(st["days"] or 0))
        srows = db.query(f"SELECT session_id, MAX(abs_start) AS t FROM utterances WHERE session_id IS NOT NULL AND ({like})"
                         " GROUP BY session_id ORDER BY t DESC LIMIT 5", pats)
    from . import sessions as _sessions

    out["recent_sessions"] = [s for s in (_sessions.get(db, int(r["session_id"])) for r in srows) if s is not None]
    for it in facts + todo_list:
        if it["day"] and (out.get("last_seen") is None or it["day"] > out["last_seen"]):
            out["last_seen"] = it["day"]
    return out


__all__ = ["KINDS", "STATUSES", "SOURCES", "KIND_LABELS", "EDITABLE", "fingerprint", "parse_due", "get", "list_items",
           "counts", "format_line", "upsert", "update", "reject", "restore", "merge", "touch", "reindex_tokens", "reindex",
           "min_similarity", "search", "related_facts", "todos", "people", "person"]
