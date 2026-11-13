"""Every change of a sentence's text, recorded and reversible (SPEC v1 section 6.3).

`utterances.text_raw` is what the recogniser wrote and never changes. `utterances.text` is what the user reads;
whoever changes it - the lexicon's first pass, an LLM, an answered question, the user's own hand - goes through
this module, which keeps one row per change in `utterance_revisions` (old text, new text, who, why).

    update_text(db, cfg, utterance_id, new_text, source=..., detail=...)    the ONE way to change stored text
    undo(db, cfg, utterance_id, revision_id=None)                           take a change back (itself a revision)
    history(db, utterance_id)                                               the revisions, newest first
    marks(db, rows)                                                         what the page underlines, per sentence
    learn_from_edit(db, cfg, utterance_id, old, new, remember=False)        a hand edit becomes a lexicon rule
    record(con, ...) / diff_edits(old, new) / flush_markdown(db, cfg)       building blocks (pipeline, batches)

What update_text keeps consistent, in one transaction: the revision row; utterances.text / text_source /
edited_at; the FTS tokens (utterance_tokens - the triggers move the index); the sentence's vectors are dropped
(utterance_vectors and memory_vectors kind 'utt': the next indexing pass re-embeds the new text); the session's
text_hash (its digest is stale); the day is noted for a Markdown rewrite. After the commit the day's Markdown is
rewritten (unless the caller batches: flush=False, then flush_markdown once) and an `utterance` event is sent.

revision.detail is JSON: {"edits": [{"from", "to", "reason"?, "alias_id"?, "lexicon_id"?, "start"?, "end"?}],
"inbox_id"?, "model"?, "undo_of"?}. `edits` is what marks() and a selective undo work from: [start, end) is where
`to` stood in the new text when the revision was made.

Undo and automatic rules: when the user takes back a rewrite that a lexicon alias made, that alias is not applied
to that sentence again by later history passes (undone_aliases) - a tombstone per sentence, in the spirit of
"nothing the user removed comes back by itself".
"""
from __future__ import annotations

import difflib
import importlib
import json
import logging
import re
from typing import Any, Iterable

from ..db import now_iso
from . import pinyin

log = logging.getLogger(__name__)

SOURCES = ("lexicon", "llm", "refine", "answer", "user", "guard", "undo")
KV_MD_DIRTY = "markdown_dirty_days"
MAX_TEXT_CHARS = 20000
#: marks() never returns more than this many marks for one sentence (a cleaned-up loop would be all underline)
MAX_MARKS = 16
#: a learned alias / term is a word, not a clause
MAX_LEARN_CHARS = 12
#: learn_from_edit without "remember": the corrected word must be rarer than this in the dictionary to count as a
#: name / term (project names made of rare words pass, 再见 does not); known lexicon entries always count
LEARN_MAX_RIGHT_FREQ = 50

_PUNCT_ONLY = re.compile(r"^[\W_]*$", re.UNICODE)
_CJK = re.compile(r"[㐀-䶿一-鿿]")
#: single characters that end a made-up word when a one-character edit is widened to a word (function words)
_FUNCTION_CHARS = frozenset("的了在是我你他她它们这那和与跟就也都还又很吗吧呢啊呀哦嗯把被给对从到去来有没不要会能说想看让个么着过得地上下里")
#: what follows / precedes a given name in a form of address: 岚姐, 王总, 李老师 / 老秦, 小周, 阿梨
_TITLE_SUFFIX = ("老师", "师傅", "姐", "哥", "总", "工", "叔", "姨", "爷", "奶", "嫂", "弟", "妹", "兄", "老", "导", "董", "局", "处", "队")
_NAME_PREFIX = ("老", "小", "阿", "大")


# ---------------------------------------------------------------------------------------------- helpers
def _col(r: Any, name: str, default: Any = None) -> Any:
    try:
        return r[name]
    except (IndexError, KeyError):
        return default


def _json(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return v
    if not v:
        return None
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return None


def _tokenize(db: Any, text: str) -> str:
    """FTS tokens of `text`, cut the way the index cuts everything else (lexicon words kept whole)."""
    try:
        from . import lexicon

        lexicon.ensure_jieba(db)
        index = importlib.import_module("audio_note.memory.index")
        return index.tokenize(text)
    except Exception:  # noqa: BLE001 - no jieba / no memory package: words by whitespace, like the pipeline's fallback
        return " ".join(text.split())


def _publish(events: Any, type_: str, **data: Any) -> None:
    if events is None:
        return
    try:
        events.publish(type_, **data)
    except Exception:  # noqa: BLE001
        log.debug("%s event not published", type_, exc_info=True)


def diff_edits(old: str, new: str) -> list[dict[str, Any]]:
    """The changed stretches between two texts, left to right: [{"from", "to", "pos", "start", "end"}] where
    `pos` is the offset of `from` in `old` and [start, end) the place of `to` in `new`. Characters, not words;
    neighbouring changes are one stretch."""
    old, new = old or "", new or ""
    if old == new:
        return []
    if len(old) + len(new) > 4000:
        # a quadratic diff of a huge text is not worth it: one stretch between the common prefix and suffix
        a = 0
        while a < min(len(old), len(new)) and old[a] == new[a]:
            a += 1
        b = 0
        while b < min(len(old), len(new)) - a and old[len(old) - 1 - b] == new[len(new) - 1 - b]:
            b += 1
        return [{"from": old[a:len(old) - b], "to": new[a:len(new) - b], "pos": a, "start": a, "end": len(new) - b}]
    out: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if out and out[-1]["end"] == j1 and out[-1]["pos"] + len(out[-1]["from"]) == i1:
            out[-1]["from"] += old[i1:i2]
            out[-1]["to"] += new[j1:j2]
            out[-1]["end"] = j2
        else:
            out.append({"from": old[i1:i2], "to": new[j1:j2], "pos": i1, "start": j1, "end": j2})
    return out


# ---------------------------------------------------------------------------------------------- writing
def record(con: Any, utterance_id: int, old_text: str, new_text: str, source: str, detail: Any = None,
           created_at: str | None = None) -> int:
    """INSERT one revision row on `con` (the caller's transaction) without touching the sentence: for code that
    stores a sentence together with the changes already made to it (the pipeline's first pass)."""
    cur = con.execute(
        "INSERT INTO utterance_revisions(utterance_id, old_text, new_text, source, detail, created_at) VALUES (?,?,?,?,?,?)",
        (int(utterance_id), old_text, new_text, source, json.dumps(detail, ensure_ascii=False) if detail is not None else None,
         created_at or now_iso()))
    return int(cur.lastrowid)


def _mark_day_dirty(con: Any, day: str | None) -> None:
    if not day:
        return
    row = con.execute("SELECT value FROM kv WHERE key=?", (KV_MD_DIRTY,)).fetchone()
    days = _json(row[0]) if row else None
    days = [d for d in days if isinstance(d, str)] if isinstance(days, list) else []
    if day not in days:
        days.append(day)
        con.execute("INSERT INTO kv(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (KV_MD_DIRTY, json.dumps(days)))


def flush_markdown(db: Any, cfg: Any) -> list[str]:
    """Rewrite transcripts/<day>.md for every day whose text changed since the last flush. Returns the days.
    The database is the source of truth: a file that cannot be written is logged, not raised."""
    if cfg is None:
        return []
    with db.transaction() as con:
        row = con.execute("SELECT value FROM kv WHERE key=?", (KV_MD_DIRTY,)).fetchone()
        days = _json(row[0]) if row else None
        days = sorted({d for d in days if isinstance(d, str)}) if isinstance(days, list) else []
        if days:
            con.execute("DELETE FROM kv WHERE key=?", (KV_MD_DIRTY,))
    if not days:
        return []
    try:
        from .markdown import write_days_markdown

        write_days_markdown(db, cfg, days)
    except Exception as e:  # noqa: BLE001
        log.warning("cannot rewrite markdown for %s: %s", ", ".join(days), e)
    return days


def _source_after(con: Any, utterance_id: int, text: str, text_raw: str) -> str | None:
    """utterances.text_source once a change was taken back: whoever made the latest change still in effect."""
    if text == text_raw:
        return None
    row = con.execute("SELECT source FROM utterance_revisions WHERE utterance_id=? AND undone_at IS NULL AND source<>'undo'"
                      " ORDER BY id DESC LIMIT 1", (int(utterance_id),)).fetchone()
    return str(row[0]) if row else "llm"       # changed, but before revisions existed: the old per-segment corrector


def _write(db: Any, utterance_id: int, new_text: Any, source: str, detail: Any, *, undo_of: int | None = None,
           redo: int | None = None, veto: bool = True) -> dict[str, Any] | None:
    """update_text without the after-commit part. undo_of: the revision this write takes back (stamped undone
    in the same transaction, also when the text turns out to be unchanged); redo: a revision that is in effect
    again because the undo that had taken it back is being undone; veto=False: the undone revision is noted as
    "taken back by the system" (see undone_aliases)."""
    if source not in SOURCES:
        raise ValueError(f"未知的修改来源: {source}")
    if not isinstance(new_text, str):
        raise ValueError("文字应为字符串")
    new_text = new_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not new_text:
        raise ValueError("文字不能为空")
    if len(new_text) > MAX_TEXT_CHARS:
        raise ValueError("文字太长了")
    uid = int(utterance_id)
    seg = _tokenize(db, new_text)            # outside the transaction: the first call may load jieba's dictionary
    with db.transaction() as con:
        row = con.execute("SELECT id, text, text_raw, day, session_id FROM utterances WHERE id=?", (uid,)).fetchone()
        if row is None:
            raise LookupError("没有这句话")
        old_text = row["text"] or ""
        now = now_iso()
        if undo_of is not None:
            con.execute("UPDATE utterance_revisions SET undone_at=? WHERE id=? AND undone_at IS NULL", (now, int(undo_of)))
            if not veto:
                old = con.execute("SELECT detail FROM utterance_revisions WHERE id=?", (int(undo_of),)).fetchone()
                d = _json(old[0]) if old else None
                d = d if isinstance(d, dict) else {}
                d["veto"] = False
                con.execute("UPDATE utterance_revisions SET detail=? WHERE id=?", (json.dumps(d, ensure_ascii=False), int(undo_of)))
        if redo is not None:
            con.execute("UPDATE utterance_revisions SET undone_at=NULL WHERE id=? AND utterance_id=?", (int(redo), uid))
        if old_text == new_text:
            return None
        rev_id = record(con, uid, old_text, new_text, source, detail, now)
        text_source = _source_after(con, uid, new_text, row["text_raw"] or "") if source == "undo" else source
        con.execute("UPDATE utterances SET text=?, text_source=?, edited_at=? WHERE id=?", (new_text, text_source, now, uid))
        con.execute("INSERT INTO utterance_tokens(utterance_id, text_seg) VALUES (?,?)"
                    " ON CONFLICT(utterance_id) DO UPDATE SET text_seg=excluded.text_seg", (uid, seg))
        con.execute("DELETE FROM utterance_vectors WHERE utterance_id=?", (uid,))
        con.execute("DELETE FROM memory_vectors WHERE kind='utt' AND ref_id=?", (uid,))
        _mark_day_dirty(con, row["day"])
        session_id = row["session_id"]
        if session_id is not None:
            db.refresh_session_hashes(int(session_id))      # same connection, inside this transaction
    return {"utterance_id": uid, "revision_id": rev_id, "old_text": old_text, "new_text": new_text, "source": source,
            "text_source": text_source, "day": row["day"], "session_id": session_id}


def update_text(db: Any, cfg: Any, utterance_id: int, new_text: str, *, source: str, detail: dict[str, Any] | None = None,
                events: Any = None, flush: bool = True) -> dict[str, Any] | None:
    """Change the text of one sentence (see the module docstring for everything that follows from it).

    Returns {"utterance_id", "revision_id", "old_text", "new_text", "source", "text_source", "day",
    "session_id"}, or None when the text is already `new_text` (nothing is written then).
    Raises LookupError (no such sentence) / ValueError (empty text, unknown source) with a message for the user.
    `flush=False`: the caller changes many sentences and calls flush_markdown(db, cfg) once at the end."""
    res = _write(db, utterance_id, new_text, source, detail)
    if res is None:
        return None
    if flush:
        flush_markdown(db, cfg)
    _publish(events, "utterance", id=res["utterance_id"], change="text", day=res["day"], source=source)
    return res


# ---------------------------------------------------------------------------------------------- undo / history
def _row_dict(r: Any) -> dict[str, Any]:
    return {"id": int(r["id"]), "utterance_id": int(r["utterance_id"]), "old_text": r["old_text"], "new_text": r["new_text"],
            "source": r["source"], "detail": _json(r["detail"]) or {}, "created_at": r["created_at"], "undone_at": r["undone_at"]}


def history(db: Any, utterance_id: int) -> list[dict[str, Any]]:
    """The revisions of a sentence, newest first: [{id, utterance_id, old_text, new_text, source, detail,
    created_at, undone_at}] (detail parsed; undone_at set once the change was taken back)."""
    return [_row_dict(r) for r in db.query("SELECT * FROM utterance_revisions WHERE utterance_id=? ORDER BY id DESC",
                                           (int(utterance_id),))]


def _reverse(text: str, rev: dict[str, Any]) -> str | None:
    """`text` with the edits of `rev` taken back, or None when they cannot be located any more."""
    if text == rev["new_text"]:
        return rev["old_text"]
    edits = [e for e in (rev["detail"].get("edits") or []) if isinstance(e, dict) and e.get("to") and e.get("from") is not None]
    if not edits:
        return None
    cur = text
    # right to left, so the recorded offsets of the edits further left are still good hints
    for e in sorted(edits, key=lambda e: -(e.get("start") if isinstance(e.get("start"), int) else 0)):
        to, frm = str(e["to"]), str(e["from"])
        at = _nearest(cur, to, e.get("start"))
        if at < 0:
            return None
        cur = cur[:at] + frm + cur[at + len(to):]
    return cur


def _nearest(text: str, needle: str, hint: Any, taken: Iterable[tuple[int, int]] = ()) -> int:
    """Start of the occurrence of `needle` in `text` closest to offset `hint` (the first one without a hint)
    that does not overlap a span in `taken`; -1 when there is none."""
    if not needle:
        return -1
    taken = list(taken)
    best, best_d = -1, None
    pos = text.find(needle)
    while pos >= 0:
        end = pos + len(needle)
        if not any(pos < t1 and t0 < end for t0, t1 in taken):
            d = abs(pos - hint) if isinstance(hint, int) else pos
            if best_d is None or d < best_d:
                best, best_d = pos, d
        pos = text.find(needle, pos + 1)
    return best


def undo(db: Any, cfg: Any, utterance_id: int, revision_id: int | None = None, *, events: Any = None,
         veto: bool = True) -> dict[str, Any]:
    """Take a change back: the latest one still in effect, or the one named by `revision_id`.

    Writes a revision with source 'undo' (so the undo is itself in the history) and stamps the undone revision.
    An older revision can be undone on its own when its edits can still be found in the current text; otherwise
    ValueError asks to undo the later changes first. Returns {"utterance_id", "undone", "revision_id", "text"}.

    `veto` (default): the user rejects this change for this sentence - a lexicon alias it applied is not
    applied to the sentence again by later history passes. veto=False is for taking back a whole operation
    (inbox.reopen): the sentences return to how they were, without any statement about the rule."""
    uid = int(utterance_id)
    row = db.one("SELECT id, text FROM utterances WHERE id=?", (uid,))
    if row is None:
        raise LookupError("没有这句话")
    if revision_id is None:
        r = db.one("SELECT * FROM utterance_revisions WHERE utterance_id=? AND undone_at IS NULL AND source<>'undo'"
                   " ORDER BY id DESC LIMIT 1", (uid,))
        if r is None:
            raise ValueError("这句话没有可以撤销的修改")
    else:
        r = db.one("SELECT * FROM utterance_revisions WHERE id=? AND utterance_id=?", (int(revision_id), uid))
        if r is None:
            raise LookupError("没有这条修改记录")
        if r["undone_at"] is not None:
            raise ValueError("这次修改已经撤销过了")
    rev = _row_dict(r)
    target = _reverse(row["text"] or "", rev)
    if target is None:
        raise ValueError("这次修改之后文字又变过，没法单独撤销：请先撤销后面的修改")
    # undoing an undo puts the change it had taken back in effect again
    redo = rev["detail"].get("undo_of") if rev["source"] == "undo" else None
    res = _write(db, uid, target, "undo", {"undo_of": rev["id"], "edits": [
        {"from": e.get("to"), "to": e.get("from")} for e in (rev["detail"].get("edits") or []) if isinstance(e, dict)]},
        undo_of=rev["id"], redo=int(redo) if isinstance(redo, int) else None, veto=veto)
    if res is not None:
        flush_markdown(db, cfg)
        _publish(events, "utterance", id=uid, change="text", day=res["day"], source="undo")
    return {"utterance_id": uid, "undone": rev["id"], "revision_id": res["revision_id"] if res else None,
            "text": target}


def undone_aliases(db: Any, utterance_ids: Iterable[int]) -> dict[int, set[int]]:
    """{utterance id: lexicon alias ids the user took back in that sentence}: history passes skip those.

    Only undos that were a veto count (see undo), and only while the alias is still the same rule: ids of
    deleted aliases are handed out again by SQLite, so the alias must still spell what the undone edit replaced."""
    ids = [int(x) for x in utterance_ids]
    out: dict[int, set[int]] = {}
    known: dict[int, Any] | None = None
    for k in range(0, len(ids), 500):
        chunk = ids[k:k + 500]
        for r in db.query(f"SELECT utterance_id, detail FROM utterance_revisions WHERE undone_at IS NOT NULL AND source<>'undo'"
                          f" AND utterance_id IN ({','.join('?' * len(chunk))})", chunk):
            d = _json(r["detail"]) or {}
            if d.get("veto") is False:
                continue
            for e in (d.get("edits") or []):
                if not isinstance(e, dict) or e.get("alias_id") is None:
                    continue
                try:
                    aid = int(e["alias_id"])
                except (TypeError, ValueError):
                    continue
                if known is None:
                    from . import lexicon

                    known = lexicon.Lexicon.load(db).alias_by_id
                got = known.get(aid)
                if got is not None and got[1].alias.lower() == str(e.get("from") or "").lower():
                    out.setdefault(int(r["utterance_id"]), set()).add(aid)
    return out


# ---------------------------------------------------------------------------------------------- marks
def marks(db: Any, rows: Iterable[Any]) -> dict[int, list[dict[str, Any]]]:
    """What the page underlines in each sentence of `rows` (utterance rows with id, text, text_raw):

        {utterance_id: [{"start", "end", "kind": "fixed", "from", "source"},
                        {"start", "end", "kind": "uncertain", "inbox_id", "candidates"}, ...]}

    Offsets are code points into the row's current text; the marks of a sentence do not overlap and are sorted.
    fixed: where the text differs from what the recogniser wrote - the span and original of the revision edit
    that explains it when there is one (so a whole word is marked, with who changed it), else the bare
    character difference. uncertain: the span of an open 'term' question, in its evidence sentences and - for
    names and terms - wherever else the same spelling occurs. A sentence without marks has no key."""
    rows = list(rows)
    out: dict[int, list[dict[str, Any]]] = {}
    changed = [r for r in rows if _col(r, "text_raw") is not None and (r["text"] or "") != (_col(r, "text_raw") or "")]
    revs: dict[int, list[dict[str, Any]]] = {}
    ids = [int(r["id"]) for r in changed]
    for k in range(0, len(ids), 500):
        chunk = ids[k:k + 500]
        for rv in db.query(f"SELECT * FROM utterance_revisions WHERE undone_at IS NULL AND source<>'undo'"
                           f" AND utterance_id IN ({','.join('?' * len(chunk))}) ORDER BY id", chunk):
            revs.setdefault(int(rv["utterance_id"]), []).append(_row_dict(rv))
    for r in changed:
        uid = int(r["id"])
        text, raw = r["text"] or "", _col(r, "text_raw") or ""
        regions = [d for d in diff_edits(raw, text)]
        spans = [(d["start"], d["end"]) for d in regions]
        got: list[dict[str, Any]] = []
        taken: list[tuple[int, int]] = []
        for rv in revs.get(uid, []):
            for e in rv["detail"].get("edits") or []:
                if not isinstance(e, dict) or not e.get("to") or e.get("to") == e.get("from"):
                    continue
                to = str(e["to"])
                # only where the text really differs from the raw text: an edit the user typed back is no mark
                at = -1
                pos = text.find(to)
                best = None
                while pos >= 0:
                    end = pos + len(to)
                    free = not any(pos < t1 and t0 < end for t0, t1 in taken)
                    real = any(pos < max(s1, s0 + 1) and s0 < end for s0, s1 in spans)
                    if free and real:
                        d = abs(pos - e["start"]) if isinstance(e.get("start"), int) else pos
                        if best is None or d < best:
                            best, at = d, pos
                    pos = text.find(to, pos + 1)
                if at < 0:
                    continue
                taken.append((at, at + len(to)))
                got.append({"start": at, "end": at + len(to), "kind": "fixed", "from": str(e.get("from") or ""),
                            "source": rv["source"]})
        fallback = _col(r, "text_source") or "llm"
        for d in regions:
            if d["end"] <= d["start"] or any(d["start"] < t1 and t0 < d["end"] for t0, t1 in taken):
                continue
            if _PUNCT_ONLY.match(d["from"]) and _PUNCT_ONLY.match(d["to"]):
                continue                                   # a comma that became a full stop is not worth a mark
            got.append({"start": d["start"], "end": d["end"], "kind": "fixed", "from": d["from"], "source": fallback})
        if got:
            out[uid] = got
    # --- open questions about a word
    try:
        items = db.query("SELECT id, utterance_id, payload FROM inbox_items WHERE kind='term' AND status='open' ORDER BY priority DESC, id")
    except Exception:  # noqa: BLE001 - a database without the table
        items = []
    for it in items:
        p = _json(it["payload"]) or {}
        span = str(p.get("span") or "")
        if not span:
            continue
        evidence = {it["utterance_id"], p.get("utterance_id")} | {ex.get("utterance_id") for ex in (p.get("examples") or [])
                                                                 if isinstance(ex, dict)}
        everywhere = p.get("term_kind") in ("name", "term")
        cands = [str(c) for c in (p.get("candidates") or []) if c][:6]
        for r in rows:
            uid = int(r["id"])
            text = r["text"] or ""
            if span not in text or not (everywhere or uid in evidence):
                continue
            cur = out.setdefault(uid, [])
            if everywhere:
                places, pos = [], text.find(span)
                while pos >= 0 and len(places) < 3:
                    places.append(pos)
                    pos = text.find(span, pos + len(span))
            else:                      # about this one spot: the occurrence that was asked about
                places = [_nearest(text, span, p.get("start") if uid == p.get("utterance_id") else None)]
            for pos in places:
                end = pos + len(span)
                # the question wins over a "fixed" mark on the same characters: it is the thing to act on
                cur[:] = [m for m in cur if not (m["kind"] == "fixed" and m["start"] < end and pos < m["end"])]
                if not any(m["start"] < end and pos < m["end"] for m in cur):
                    cur.append({"start": pos, "end": end, "kind": "uncertain", "inbox_id": int(it["id"]), "candidates": cands})
            if not cur:
                out.pop(uid, None)
    for uid, ms in out.items():
        ms.sort(key=lambda m: (m["start"], m["end"]))
        del ms[MAX_MARKS:]
    return out


# ---------------------------------------------------------------------------------------------- learning from a hand edit
def _token_spans(text: str) -> list[tuple[int, int]]:
    """[start, end) of the dictionary words of `text` (jieba without its new-word guessing: an unknown word
    comes out as single characters, which _widen then joins by its own, stricter rules - the guesser happily
    glues a preceding 和 to a name); one span per character without jieba."""
    try:
        import jieba

        jieba.setLogLevel(logging.WARNING)
        spans, pos = [], 0
        for tok in jieba.lcut(text, HMM=False):
            spans.append((pos, pos + len(tok)))
            pos += len(tok)
        return spans
    except Exception:  # noqa: BLE001
        return [(i, i + 1) for i in range(len(text))]


def _widen(text: str, a: int, b: int, lex: Any) -> tuple[int, int]:
    """The word of `text` the changed stretch [a, b) belongs to: a known lexicon spelling around it, else the
    tokenizer's word(s); a lone character is joined with a form of address next to it (岚+姐, 老+秦) or with
    the single-character neighbours that are not function words (an unknown word comes out of the tokenizer
    as single characters)."""
    for h in (lex.find_in(text) if lex is not None else []):
        if h.start <= a and b <= h.end and h.alias is None:
            return h.start, h.end
    spans = _token_spans(text)
    lo, hi = a, max(b, a)
    for s, e in spans:
        if s < max(b, a + 1) and a < e:
            lo, hi = min(lo, s), max(hi, e)
    if hi - lo != 1 or not _CJK.match(text[lo:hi]):
        return lo, hi
    for suf in _TITLE_SUFFIX:
        if text.startswith(suf, hi):
            return lo, hi + len(suf)
    if lo > 0 and text[lo - 1] in _NAME_PREFIX:
        return lo - 1, hi
    single = {s for s, e in spans if e - s == 1}

    def joinable(i: int) -> bool:
        return 0 <= i < len(text) and i in single and bool(_CJK.match(text[i])) and text[i] not in _FUNCTION_CHARS

    while hi - lo < 4 and joinable(hi):
        hi += 1
    while hi - lo < 4 and joinable(lo - 1):
        lo -= 1
    return lo, hi


def edit_pair(old_text: str, new_text: str, lex: Any = None) -> tuple[str, str] | None:
    """(wrong, right): the one word a hand edit replaced, widened to a whole word on both sides. None when the
    edit is not a single replacement (several places changed, something only inserted or deleted, too long)."""
    edits = diff_edits(old_text or "", new_text or "")
    if len(edits) != 1 or not edits[0]["from"] or not edits[0]["to"]:
        return None
    e = edits[0]
    lo, hi = _widen(new_text, e["start"], e["end"], lex)
    # the same number of characters to the left / right of the change exist in the old text
    left = e["start"] - lo
    right = hi - e["end"]
    olo, ohi = e["pos"] - left, e["pos"] + len(e["from"]) + right
    if olo < 0 or ohi > len(old_text):
        return None
    wrong, right_word = old_text[olo:ohi].strip(), new_text[lo:hi].strip()
    if not wrong or not right_word or wrong == right_word:
        return None
    if len(wrong) > MAX_LEARN_CHARS or len(right_word) > MAX_LEARN_CHARS:
        return None
    if _PUNCT_ONLY.match(wrong) or _PUNCT_ONLY.match(right_word):
        return None
    return wrong, right_word


def _guess_kind(word: str) -> str:
    if any(word.endswith(s) for s in _TITLE_SUFFIX) and len(word) <= 4 or (word[:1] in _NAME_PREFIX and len(word) == 2):
        return "person"
    return "term"


def learn_from_edit(db: Any, cfg: Any, utterance_id: int, old_text: str, new_text: str, *, remember: bool = False,
                    events: Any = None) -> dict[str, Any] | None:
    """Turn the user's hand edit of one sentence into a lexicon rule, when it is one.

    The edit must replace a single word (edit_pair). A rule is made when the user asked for it (`remember`), or
    when it looks like a name / term the recogniser mis-heard: the two spellings sound alike (pinyin.close),
    the right one is a known entry or a word the dictionary hardly knows (LEARN_MAX_RIGHT_FREQ), and the wrong
    one is not itself an entry. The rule is entry `right` (created with source 'edit' when new) + mishear alias `wrong`; its scope
    is decided by lexicon.add_alias (auto / hint).

    Returns None when nothing was learned, else {"lexicon_id", "alias_id", "term", "alias", "scope",
    "created_term", "other_matches", "inbox_id"}: other_matches = how many OTHER stored sentences contain the
    wrong spelling; when there are some, a 'rule' question ("change the other N places too?") is put into the
    inbox and inbox_id names it (None otherwise). Nothing else is rewritten here."""
    from . import lexicon

    lex = lexicon.Lexicon.load(db)
    pair = edit_pair(old_text, new_text, lex)
    if pair is None:
        return None
    wrong, right = pair
    known = lex.by_term.get(right)
    if not remember:
        if lex.is_term(wrong):
            return None                                   # one known word replaced by another: a different word was meant
        if not pinyin.close(wrong, right):
            return None                                   # reworded, not mis-heard
        if known is None and (len(right) < 2 or lexicon.word_frequency(right) > LEARN_MAX_RIGHT_FREQ):
            return None                                   # an everyday word (在见 -> 再见): not vocabulary
    created = known is None or known.status != "active"
    try:
        lexicon_id = lexicon.upsert_term(db, right, kind=_guess_kind(right) if known is None else known.kind, source="edit",
                                         events=events)
        alias_id = lexicon.add_alias(db, lexicon_id, wrong, source="edit", cfg=cfg, events=events)
    except (ValueError, LookupError) as e:
        log.info("hand edit not learned (%s)", e)
        return None
    alias = lexicon.get_alias(db, alias_id) or {}
    scope = alias.get("scope", "auto")
    res = lexicon.apply_alias_to_history(db, None, alias_id, dry_run=True, max_listed=6)
    others = [m for m in res["matches"] if m["utterance_id"] != int(utterance_id)]
    n_other = sum(1 for i in res["match_ids"] if i != int(utterance_id))
    inbox_id = None
    if n_other > 0:
        try:
            inbox = importlib.import_module("audio_note.inbox")
            inbox_id = inbox.add(
                db, "rule", title=f"其他 {n_other} 处「{wrong}」也改成「{right}」吗？",
                payload={"alias_id": alias_id, "lexicon_id": lexicon_id, "alias": wrong, "term": right, "count": n_other,
                         "utterance_id": int(utterance_id),
                         "examples": [{"utterance_id": m["utterance_id"], "sentence": m["before"]} for m in others[:3]]},
                dedupe_key=f"alias:{alias_id}", priority=float(min(n_other, 20)), utterance_id=int(utterance_id),
                lexicon_id=lexicon_id, events=events) or None
        except Exception as e:  # noqa: BLE001 - the rule exists either way; the question is a convenience
            log.warning("rule question not created: %s", e)
    return {"lexicon_id": lexicon_id, "alias_id": alias_id, "term": right, "alias": wrong, "scope": scope,
            "created_term": bool(created), "other_matches": n_other, "inbox_id": inbox_id}


__all__ = ["KV_MD_DIRTY", "SOURCES", "diff_edits", "edit_pair", "flush_markdown", "history", "learn_from_edit", "marks",
           "record", "undo", "undone_aliases", "update_text"]
