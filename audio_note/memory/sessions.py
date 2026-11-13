"""Memory sessions: stretches of continuous conversation, cut where nobody spoke for a while.

    sync(cfg, db)                 first step of the post-processing chain (audio_note.chain.STEPS)
    for_day(db, day)              the sessions of one day (those that started the evening before and ran past
                                  midnight included)
    get(db, session_id)           one session, optionally with its sentences
    lines(db, session_id)         numbered lines for an LLM prompt (refine / episodes)

A session is a row of `sessions` plus `utterances.session_id`. Two sentences belong to the same session when
the silence between them is at most `memory.session_gap_min` (default 5 minutes; measured on the owner's data
97% of the gaps between consecutive sentences are shorter, and a day then falls into 9-22 sessions). This is a
finer cut than `analysis.session_gap_min` (20 minutes), which only shapes the blocks of the daily digest.

What the other chain steps rely on (the column comments in db.SCHEMA are the contract):

* `status`      'open' while sentences may still arrive, 'closed' once the last sentence ended more than the
                gap ago. Refine and episodes only touch closed sessions. A closed session that receives a
                sentence after all (late import, a split) is open again until the gap has passed once more.
* `raw_hash` / `text_hash`   always db.refresh_session_hashes(): ids + text_raw / ids + text. This module never
                writes refined_hash / digested_hash, so "hash differs" keeps meaning "work to do".
* `ended_at`    ISO local time ('T' form) of the end of the last sentence.

sync() is idempotent and cheap when nothing changed (three indexed queries), because the chain runs it after
every processed segment.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Iterable

from ..db import DB, now_iso

log = logging.getLogger(__name__)

DEFAULT_GAP_MIN = 5.0
#: sentences assigned per transaction (a stop request is honoured between two batches)
BATCH = 2000
#: session columns handed out by for_day() / get()
_COLS = ("id, day, started_at, ended_at, utterance_count, speech_s, chars, status, kind, title, summary, topics,"
         " participants, raw_hash, refined_hash, refined_at, text_hash, digested_hash, digested_at, meta, updated_at")


def gap_minutes(cfg: Any) -> float:
    """memory.session_gap_min as a positive number of minutes (the default when missing or unusable)."""
    try:
        v = float(cfg.memory.get("session_gap_min", DEFAULT_GAP_MIN))
    except (TypeError, ValueError, AttributeError):
        return DEFAULT_GAP_MIN
    return v if v > 0 else DEFAULT_GAP_MIN


def _parse(ts: Any) -> datetime:
    """utterances.abs_start / sessions.started_at -> datetime ('T' or space, with or without fractions)."""
    s = str(ts or "").strip().replace(" ", "T")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds")


def _span(row: Any) -> tuple[datetime, datetime]:
    """(start, end) of an utterance row: abs_start plus its length inside the segment."""
    start = _parse(row["abs_start"])
    dur = max(0.0, float(row["end_s"] or 0.0) - float(row["start_s"] or 0.0))
    return start, start + timedelta(seconds=dur)


def _stopped(stop_event: Any) -> bool:
    return stop_event is not None and stop_event.is_set()


# ------------------------------------------------------------------------------------------- bookkeeping
def _drop(db: DB, sid: int, into: int | None = None) -> None:
    """Remove a session row that has no sentences left (or was merged into `into`). What pointed at it either
    follows the merge or is detached; the sentences themselves are the caller's business."""
    db.execute("UPDATE memory_items SET session_id=? WHERE session_id=?", (into, int(sid)))
    db.execute("UPDATE inbox_items SET session_id=? WHERE session_id=?", (into, int(sid)))
    db.execute("DELETE FROM memory_vectors WHERE kind='session' AND ref_id=?", (int(sid),))
    db.execute("DELETE FROM sessions WHERE id=?", (int(sid),))


def refresh(db: DB, session_id: int) -> bool:
    """Recompute a session's time span, counts and the two content hashes from its sentences as they are now.
    A session without sentences is removed. Returns False when the session is gone afterwards.
    For anybody who added, removed or moved sentences of a session (forget, split, a re-processed segment)."""
    sid = int(session_id)
    rows = db.query("SELECT id, abs_start, start_s, end_s, text FROM utterances WHERE session_id=? ORDER BY abs_start, id",
                    (sid,))
    if not rows:
        _drop(db, sid)
        return False
    spans = [_span(r) for r in rows]
    start = min(s for s, _ in spans)
    end = max(e for _, e in spans)
    db.execute(
        "UPDATE sessions SET day=?, started_at=?, ended_at=?, utterance_count=?, speech_s=?, chars=?, updated_at=? WHERE id=?",
        (start.strftime("%Y-%m-%d"), _iso(start), _iso(end), len(rows),
         round(sum((e - s).total_seconds() for s, e in spans), 3), sum(len(r["text"] or "") for r in rows), now_iso(), sid))
    db.refresh_session_hashes(sid)
    return True


def _needs_reconcile(db: DB) -> bool:
    """True when sentences left or joined sessions behind our back: the stored counts no longer add up.
    Two aggregate queries; the expensive look only happens when they disagree."""
    have = db.one("SELECT COUNT(*) AS n FROM utterances WHERE session_id IS NOT NULL")["n"]
    want = db.one("SELECT COALESCE(SUM(utterance_count), 0) AS n FROM sessions")["n"]
    return int(have) != int(want)


def repair(db: DB, session_id: int, gap: timedelta) -> bool:
    """Bring one session back in line with its sentences after some were removed: when the removal left a hole
    longer than the gap, the part after the hole is detached (session_id NULL; the next sync() makes it a
    session of its own), then refresh(). Returns False when the session is gone afterwards."""
    sid = int(session_id)
    rows = db.query("SELECT id, abs_start, start_s, end_s FROM utterances WHERE session_id=? ORDER BY abs_start, id", (sid,))
    end: datetime | None = None
    for i, u in enumerate(rows):
        s, e = _span(u)
        if end is not None and s - end > gap:
            db.conn().executemany("UPDATE utterances SET session_id=NULL WHERE id=?", [(int(x["id"]),) for x in rows[i:]])
            break
        end = e if end is None or e > end else end
    return refresh(db, sid)


def _reconcile(db: DB, gap: timedelta, out: dict[str, Any], changes: dict[int, str]) -> None:
    """Repair sessions after deletions made elsewhere (a segment processed again, faint sentences removed, a
    split): sentences of a vanished session become unassigned, emptied sessions are removed, and a session
    that lost its middle is cut at the hole (repair)."""
    with db.transaction():
        db.execute("UPDATE utterances SET session_id=NULL WHERE session_id IS NOT NULL"
                   " AND session_id NOT IN (SELECT id FROM sessions)")
        stale = db.query(
            "SELECT s.id FROM sessions s LEFT JOIN (SELECT session_id, COUNT(*) AS n FROM utterances"
            " WHERE session_id IS NOT NULL GROUP BY session_id) u ON u.session_id = s.id"
            " WHERE COALESCE(u.n, 0) <> s.utterance_count")
        for r in stale:
            sid = int(r["id"])
            if repair(db, sid, gap):
                out["updated"] += 1
                changes[sid] = "updated"
            else:
                out["removed"] += 1
                changes[sid] = "removed"


def _assign(db: DB, rows: list[Any], gap: timedelta, out: dict[str, Any], changes: dict[int, str],
            reopened: set[int]) -> int:
    """Give every row of `rows` (utterances without a session) a session. Sentences closer than the gap chain
    into runs; a run joins every existing session it comes within the gap of, and two sessions joined by the
    same run are merged into the earlier one. Returns the number of sentences assigned."""
    items = sorted(((*_span(r), int(r["id"])) for r in rows), key=lambda x: (x[0], x[2]))
    runs: list[list[Any]] = []                                 # [start, end, [utterance ids]]
    for s, e, uid in items:
        if runs and s - runs[-1][1] <= gap:
            runs[-1][1] = max(runs[-1][1], e)
            runs[-1][2].append(uid)
        else:
            runs.append([s, e, [uid]])
    lo = _iso(runs[0][0] - gap - timedelta(seconds=1))
    hi = _iso(max(r[1] for r in runs) + gap + timedelta(seconds=1))
    existing = [(int(r["id"]), _parse(r["started_at"]), _parse(r["ended_at"]), str(r["status"]))
                for r in db.query("SELECT id, started_at, ended_at, status FROM sessions"
                                  " WHERE replace(ended_at, ' ', 'T') >= ? AND replace(started_at, ' ', 'T') <= ?"
                                  " ORDER BY started_at, id", (lo, hi))]

    # union-find over runs (0..R-1) and existing sessions (R..R+S-1); sessions never link to each other directly
    parent = list(range(len(runs) + len(existing)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, (rs, re_, _ids) in enumerate(runs):
        for j, (_sid, ss, se, _st) in enumerate(existing):
            if se >= rs - gap and ss <= re_ + gap:
                parent[find(len(runs) + j)] = find(i)
    groups: dict[int, tuple[list[int], list[int]]] = {}
    for i in range(len(runs)):
        groups.setdefault(find(i), ([], []))[0].append(i)
    for j in range(len(existing)):
        root = find(len(runs) + j)
        if root in groups:
            groups[root][1].append(j)

    assigned = 0
    for run_idx, sess_idx in groups.values():
        ids = [uid for i in run_idx for uid in runs[i][2]]
        if sess_idx:
            keep, _ks, _ke, keep_status = existing[sess_idx[0]]         # sorted by started_at: the earliest stays
            for j in sess_idx[1:]:
                other = existing[j][0]
                db.execute("UPDATE utterances SET session_id=? WHERE session_id=?", (keep, other))
                _drop(db, other, into=keep)
                out["merged"] += 1
                changes[other] = "removed"
                reopened.discard(other)
            if keep_status != "open":
                db.execute("UPDATE sessions SET status='open' WHERE id=?", (keep,))
                reopened.add(keep)
            if keep not in changes:
                out["updated"] += 1
                changes[keep] = "updated"
        else:
            start = runs[run_idx[0]][0]
            keep = int(db.execute(
                "INSERT INTO sessions(day, started_at, ended_at, status) VALUES (?,?,?, 'open')",
                (start.strftime("%Y-%m-%d"), _iso(start), _iso(runs[run_idx[-1]][1]))).lastrowid)
            out["created"] += 1
            changes[keep] = "created"
        db.conn().executemany("UPDATE utterances SET session_id=? WHERE id=?", [(keep, uid) for uid in ids])
        refresh(db, keep)
        assigned += len(ids)
    return assigned


# ------------------------------------------------------------------------------------------- the chain step
def sync(cfg: Any, db: DB, client: Any = None, *, stop_event: Any = None, events: Any = None,
         limit: int | None = None, now: datetime | None = None) -> dict[str, Any]:
    """Assign every sentence without a session, refresh the sessions that changed, close the ones whose gap has
    passed. Idempotent; needs no LLM (`client` is part of the fixed step signature and unused).

    `limit` is ignored on purpose: the scheduler hands every chain step the same small number, meant for LLM
    work, while everything here is a few SQL statements. `now` fixes the clock (tests, benchmarks).

    Returns {"assigned", "created", "updated", "merged", "removed", "closed", "reopened", "open", "changed",
    "wall_ms"}: sentences assigned; sessions created / extended or repaired / merged away / removed because
    they were empty / closed in this call / open again after having been closed; sessions open afterwards."""
    t0 = time.perf_counter()
    gap = timedelta(minutes=gap_minutes(cfg))
    out: dict[str, Any] = {"assigned": 0, "created": 0, "updated": 0, "merged": 0, "removed": 0, "closed": 0, "reopened": 0}
    changes: dict[int, str] = {}
    reopened: set[int] = set()

    if _needs_reconcile(db):
        _reconcile(db, gap, out, changes)
    while not _stopped(stop_event):
        rows = db.query("SELECT id, abs_start, start_s, end_s FROM utterances WHERE session_id IS NULL"
                        " ORDER BY abs_start, id LIMIT ?", (BATCH,))
        if not rows:
            break
        with db.transaction():
            n = _assign(db, rows, gap, out, changes, reopened)
        out["assigned"] += n
        if n == 0 or len(rows) < BATCH:
            break

    # close what has been quiet for longer than the gap (sessions just created from old recordings included)
    cutoff = _iso((now or datetime.now()) - gap)
    due = [int(r["id"]) for r in db.query(
        "SELECT id FROM sessions WHERE status='open' AND replace(ended_at, ' ', 'T') <= ?", (cutoff,))]
    if due:
        with db.transaction() as c:
            c.executemany("UPDATE sessions SET status='closed', updated_at=? WHERE id=? AND status='open'",
                          [(now_iso(), sid) for sid in due])
        for sid in due:
            changes[sid] = "closed"
            reopened.discard(sid)
        out["closed"] = len(due)
    for sid in reopened:
        if changes.get(sid) != "removed":
            changes[sid] = "reopened"
    out["reopened"] = len(reopened)
    out["open"] = int(db.one("SELECT COUNT(*) AS n FROM sessions WHERE status='open'")["n"])
    out["changed"] = bool(changes)
    out["wall_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)

    if events is not None and changes:
        days = {int(r["id"]): r["day"] for r in db.query(
            f"SELECT id, day FROM sessions WHERE id IN ({','.join('?' * len(changes))})", list(changes))}
        for sid, change in changes.items():
            try:
                events.publish("session", id=sid, change=change, day=days.get(sid))
            except Exception as e:  # noqa: BLE001 - an event never fails the step
                log.debug("session event dropped: %s", e)
    if changes:
        log.info("sessions: %d sentences assigned, %d new, %d updated, %d merged, %d closed (%.0f ms)",
                 out["assigned"], out["created"], out["updated"], out["merged"], out["closed"], out["wall_ms"])
    return out


# ------------------------------------------------------------------------------------------- reading
def _json_list(raw: Any) -> list[Any]:
    if not raw:
        return []
    try:
        v = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return v if isinstance(v, list) else []


def _hhmm(ts: Any) -> str:
    return str(ts or "").replace("T", " ")[11:16]


def _as_dict(r: Any) -> dict[str, Any]:
    start, end = _parse(r["started_at"]), _parse(r["ended_at"])
    try:
        meta = json.loads(r["meta"]) if r["meta"] else None
    except (TypeError, ValueError):
        meta = None
    return {
        "id": int(r["id"]), "day": r["day"], "started_at": r["started_at"], "ended_at": r["ended_at"],
        "start": _hhmm(r["started_at"]), "end": _hhmm(r["ended_at"]),
        "duration_s": round(max(0.0, (end - start).total_seconds()), 1),
        "crosses_midnight": end.date() != start.date(),
        "utterance_count": int(r["utterance_count"] or 0), "speech_s": float(r["speech_s"] or 0.0),
        "chars": int(r["chars"] or 0), "status": r["status"], "kind": r["kind"], "title": r["title"],
        "summary": r["summary"], "topics": _json_list(r["topics"]), "participants": _json_list(r["participants"]),
        # "refined" / "digested": the step has seen the session exactly as it is now
        "refined": bool(r["refined_hash"]) and r["refined_hash"] == r["raw_hash"],
        "digested": bool(r["digested_hash"]) and r["digested_hash"] == r["text_hash"],
        "refined_at": r["refined_at"], "digested_at": r["digested_at"], "meta": meta, "updated_at": r["updated_at"],
    }


def _add_speakers(db: DB, sessions: list[dict[str, Any]], unknown: str | None = None) -> None:
    """sessions[i]["speakers"] = [{"id", "name", "n", "speech_s"}], most talkative first (one grouped query)."""
    if not sessions:
        return
    by_id = {s["id"]: s for s in sessions}
    for s in sessions:
        s["speakers"] = []
    rows = db.query(
        "SELECT session_id, speaker_id, speaker_name, COUNT(*) AS n, SUM(end_s - start_s) AS speech_s FROM utterances"
        f" WHERE session_id IN ({','.join('?' * len(by_id))}) GROUP BY session_id, speaker_id, speaker_name"
        " ORDER BY n DESC, (speaker_name IS NULL), speaker_name", list(by_id))
    for r in rows:
        by_id[int(r["session_id"])]["speakers"].append({
            "id": r["speaker_id"], "name": r["speaker_name"] or unknown, "n": int(r["n"]),
            "speech_s": round(float(r["speech_s"] or 0.0), 1)})


def for_day(db: DB, day: str, *, unknown: str | None = None) -> list[dict[str, Any]]:
    """Sessions that have sentences on `day` (YYYY-MM-DD), in time order: the ones that started that day and
    one that started the evening before and ran past midnight. Each carries "speakers"."""
    day = str(day)
    rows = db.query(
        f"SELECT {_COLS} FROM sessions WHERE day = ? OR (day < ? AND substr(replace(ended_at, ' ', 'T'), 1, 10) >= ?)"
        " ORDER BY started_at, id", (day, day, day))
    out = [_as_dict(r) for r in rows]
    _add_speakers(db, out, unknown)
    return out


def get(db: DB, session_id: int, *, with_utterances: bool = False, unknown: str | None = None) -> dict[str, Any] | None:
    """One session as a dict (see for_day), or None. `with_utterances` adds "utterances": the rows of the
    session as plain dicts in time order (all columns the transcript shows; no audio paths)."""
    r = db.one(f"SELECT {_COLS} FROM sessions WHERE id=?", (int(session_id),))
    if r is None:
        return None
    out = _as_dict(r)
    _add_speakers(db, [out], unknown)
    if with_utterances:
        out["utterances"] = [dict(u) for u in db.query(
            "SELECT id, segment_id, start_s, end_s, abs_start, day, speaker_id, speaker_name, speaker_score, speaker_source,"
            " text, text_raw, text_source, edited_at, flags, speaker_guess_id, speaker_guess_score"
            " FROM utterances WHERE session_id=? ORDER BY abs_start, id", (int(session_id),))]
    return out


def lines(db: DB, session_id: int, *, unknown: str = "未知") -> list[dict[str, Any]]:
    """The session as numbered lines for an LLM prompt: [{"i", "id", "time", "speaker", "text"}] ("i" counts
    from 0 in time order, "id" is the utterance id the model's answer is mapped back to, "time" is HH:MM)."""
    rows = db.query("SELECT id, abs_start, speaker_name, text FROM utterances WHERE session_id=? ORDER BY abs_start, id",
                    (int(session_id),))
    return [{"i": i, "id": int(r["id"]), "time": _hhmm(r["abs_start"]), "speaker": r["speaker_name"] or unknown,
             "text": " ".join(str(r["text"] or "").split())} for i, r in enumerate(rows)]


def ids_for(db: DB, utterance_ids: Iterable[int]) -> list[int]:
    """The distinct sessions the given sentences belong to (unassigned sentences contribute nothing)."""
    ids = [int(u) for u in utterance_ids]
    out: set[int] = set()
    for i in range(0, len(ids), 500):
        part = ids[i:i + 500]
        out.update(int(r["session_id"]) for r in db.query(
            f"SELECT DISTINCT session_id FROM utterances WHERE session_id IS NOT NULL AND id IN ({','.join('?' * len(part))})",
            part))
    return sorted(out)


__all__ = ["DEFAULT_GAP_MIN", "gap_minutes", "sync", "refresh", "repair", "for_day", "get", "lines", "ids_for"]
