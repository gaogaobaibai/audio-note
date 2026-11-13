"""Forgetting: remove a stretch of what was recorded from EVERY layer, and say exactly what was done.

    report = forget(db, cfg, day="2026-11-03")                                    # dry run: counts only
    report = forget(db, cfg, day="2026-11-03", time_from="14:00", time_to="15:30", dry_run=False)
    report = forget(db, cfg, utterance_ids=[812, 813], dry_run=False)
    report = forget(db, cfg, session_id=41, dry_run=False)

Deleting a sentence used to leave it in the segment's stored transcript, the day's Markdown, the digest, the
notes and the indexes. forget() walks all of them:

    sentences            rows deleted; keyword tokens / FTS, text vectors (all views), voice vectors, revisions
                         and vetoes go with them
    segment transcripts  the raw ASR items of the forgotten sentences are cut out (the row is removed when the
                         whole segment is forgotten)
    audio                the file is deleted when the WHOLE segment is forgotten; a segment that keeps other
                         sentences keeps its file, and the report names it ("audio_kept")
    cluster embeddings   of clusters that lose all their sentences
    sessions             counts / hashes refreshed, summary dropped (it was written from the old content);
                         an emptied session is removed
    memory items         evidence trimmed; an item with no evidence left becomes a scrubbed tombstone (never
                         generated again, except a person: other days may name them); automatic items of the day
                         that never had evidence are removed so the next digest rebuilds them from what is left
                         (forgetting part of a day: only those that mention words of the forgotten sentences)
    digests, Markdown    the day's digest (row and file) is removed and listed for rebuilding (the scheduler writes
                         it again from what is left, without pushing); audio outside the profile is never deleted; the transcript
                         Markdown is rewritten (or removed when the day is empty)
    inbox, stored chats  questions about the forgotten sentences and answers that cite them are removed
    backups              pre-upgrade copies of the database that still contain the sentences are deleted
    the file itself      deleted pages are zeroed (PRAGMA secure_delete), the FTS indexes are merged, the WAL is
                         folded back, and a small database is vacuumed so nothing lingers in free space

What it cannot reach is listed in report["notes"]: digests already pushed to a phone or a chat tool, voice
samples the user cut from that audio (kept, named in the report), log files.

Nothing here calls an LLM. The caller serialises it against the pipeline (the web handler holds the scheduler
lock): a segment that is being transcribed right now is skipped and reported.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from ..db import BACKUP_DIR_NAME, DB, now_iso

log = logging.getLogger(__name__)

#: databases up to this size are vacuumed after a forget (rewrites the file: nothing survives in free pages)
VACUUM_MAX_BYTES = 256 * 1024 * 1024
#: kv key: days whose digest was removed by forget() and can be generated again from what is left
KV_REBUILD_DAYS = "forget_rebuild_days"
_CHUNK = 500


# ------------------------------------------------------------------------------------------- scope
def _parse_time(value: Any, day: str | None, name: str) -> datetime | None:
    """'2026-11-03T14:00' | '2026-11-03 14:00:05' | '14:00' (needs `day`) -> datetime."""
    if value is None or value == "":
        return None
    s = str(value).strip().replace(" ", "T")
    if len(s) <= 8 and ":" in s:
        if not day:
            raise ValueError(f"{name} 只有时间没有日期：请同时给出 day，或写成 YYYY-MM-DDTHH:MM")
        if s.startswith("24:00"):                   # "until midnight"
            return datetime.fromisoformat(day) + timedelta(days=1)
        s = f"{day}T{s}"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        raise ValueError(f"{name} 不是有效的时间：{value!r}（用 YYYY-MM-DDTHH:MM 或 HH:MM）") from None


def _day_arg(day: Any) -> str | None:
    if day is None or day == "":
        return None
    try:
        return datetime.strptime(str(day).strip(), "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        raise ValueError(f"day 应为 YYYY-MM-DD：{day!r}") from None


def _in(ids: Iterable[int]) -> Iterable[list[int]]:
    ids = list(ids)
    for i in range(0, len(ids), _CHUNK):
        yield ids[i:i + _CHUNK]


def _select(db: DB, day: str | None, t0: datetime | None, t1: datetime | None, utterance_ids: Any,
            session_id: Any) -> tuple[list[Any], set[int]]:
    """(utterance rows to forget, ids of segments that lie completely inside a day / time scope)."""
    rows: dict[int, Any] = {}
    cols = "id, segment_id, day, abs_start, start_s, end_s, cluster_label, session_id, text, text_raw, created_at"

    def take(found: Iterable[Any]) -> None:
        for r in found:
            rows[int(r["id"])] = r

    if utterance_ids:
        ids = sorted({int(u) for u in utterance_ids})
        for part in _in(ids):
            take(db.query(f"SELECT {cols} FROM utterances WHERE id IN ({','.join('?' * len(part))})", part))
    if session_id is not None:
        take(db.query(f"SELECT {cols} FROM utterances WHERE session_id=?", (int(session_id),)))
    whole_segments: set[int] = set()
    if day or t0 or t1:
        lo = t0 or (datetime.fromisoformat(day) if day else None)
        hi = t1 or (datetime.fromisoformat(day) + timedelta(days=1) if day else None)
        cond, params = [], []
        if lo is not None:
            cond.append("replace(abs_start, ' ', 'T') >= ?")
            params.append(lo.isoformat(timespec="seconds"))
        if hi is not None:
            cond.append("replace(abs_start, ' ', 'T') < ?")
            params.append(hi.isoformat(timespec="seconds"))
        take(db.query(f"SELECT {cols} FROM utterances WHERE " + " AND ".join(cond), params))
        # recordings that lie completely inside the window go as a whole, sentences or not (silence, faint speech)
        scond, sparams = ["status <> 'deleted'"], []
        if lo is not None:
            scond.append("replace(started_at, ' ', 'T') >= ?")
            sparams.append(lo.isoformat(timespec="seconds"))
        for s in db.query("SELECT id, started_at, duration_s FROM segments WHERE " + " AND ".join(scond), sparams):
            try:
                end = datetime.fromisoformat(str(s["started_at"]).replace(" ", "T")) + timedelta(seconds=float(s["duration_s"] or 0))
            except ValueError:
                continue
            if hi is None or end <= hi:
                whole_segments.add(int(s["id"]))
    return [rows[k] for k in sorted(rows)], whole_segments


# ------------------------------------------------------------------------------------------- helpers
def _json(raw: Any, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def _bounds(start_s: Any, end_s: Any) -> tuple[float, float] | None:
    try:
        return round(float(start_s), 3), round(float(end_s), 3)
    except (TypeError, ValueError):
        return None


def _scrub_transcript(c: sqlite3.Connection, seg_id: int, gone: list[Any], apply: bool) -> int:
    """Cut the forgotten sentences out of the segment's transcripts rows (items are matched by utterance id when
    they carry one, else by their rounded bounds, as speech.split does). Returns the number of rows changed."""
    gone_ids = {int(u["id"]) for u in gone}
    gone_bounds = {b for b in (_bounds(u["start_s"], u["end_s"]) for u in gone) if b is not None}
    changed = 0
    for tr in c.execute("SELECT id, raw_json, correction_json FROM transcripts WHERE segment_id=?", (int(seg_id),)).fetchall():
        items = [it for it in _json(tr["raw_json"], []) if isinstance(it, dict)]
        keep, index_map = [], {}
        for pos, it in enumerate(items):
            uid = it.get("utterance_id")
            hit = (isinstance(uid, int) and uid in gone_ids) or (not isinstance(uid, int) and
                                                                 _bounds(it.get("start_s"), it.get("end_s")) in gone_bounds)
            if hit:
                continue
            old_i = it.get("i") if isinstance(it.get("i"), int) else pos
            index_map[old_i] = len(keep)
            keep.append({**it, "i": len(keep)})
        if len(keep) == len(items):
            continue
        changed += 1
        if not apply:
            continue
        edits = [{**e, "i": index_map[e.get("i")]} for e in _json(tr["correction_json"], [])
                 if isinstance(e, dict) and e.get("i") in index_map]
        c.execute("UPDATE transcripts SET raw_json=?, text_raw=?, text_corrected=?, correction_json=? WHERE id=?",
                  (json.dumps(keep, ensure_ascii=False), "\n".join(str(it.get("text_raw") or "") for it in keep),
                   "\n".join(str(it.get("text") or "") for it in keep),
                   json.dumps(edits, ensure_ascii=False) if edits else None, int(tr["id"])))
    return changed


def _cited_ids(meta: Any) -> set[int]:
    """Utterance ids a stored chat message cites (meta.citations: ids, or dicts with id / utterance_id)."""
    out: set[int] = set()
    cites = meta.get("citations") if isinstance(meta, dict) else None
    for cit in cites if isinstance(cites, list) else []:
        v = cit.get("utterance_id", cit.get("id")) if isinstance(cit, dict) else cit
        try:
            out.add(int(v))
        except (TypeError, ValueError):
            continue
    return out


def _terms(texts: Iterable[str]) -> set[str]:
    from . import index as _index

    out: set[str] = set()
    for t in texts:
        out.update(x for x in _index.content_tokens(_index.tokenize(t).split()) if len(x) > 1)
    return out


def _forgotten_terms(c: sqlite3.Connection, rows: list[Any], gone: set[int], days: Iterable[str]) -> set[str]:
    """Content words of the forgotten sentences that no remaining sentence of those days uses: a day's
    evidence-less automatic item mentioning one of them was (probably) written from what is being forgotten."""
    mine = _terms(f"{u['text'] or ''} {u['text_raw'] or ''}" for u in rows)
    if not mine:
        return set()
    left: set[str] = set()
    for d in days:
        left |= _terms(f"{r['text'] or ''} {r['text_raw'] or ''}" for r in
                       c.execute("SELECT id, text, text_raw FROM utterances WHERE day=?", (d,)).fetchall() if int(r["id"]) not in gone)
    return mine - left


def _owned(cfg: Any, p: Path) -> bool:
    """Is `p` inside this profile's audio or data folder? A database copied or restored elsewhere keeps the
    absolute segment paths of the profile it came from; forgetting in that copy must never delete the other
    profile's recordings (Path.unlink bypasses the Recycle Bin)."""
    try:
        rp = p.resolve()
    except OSError:
        return False
    for attr in ("audio_dir", "data_dir"):
        try:
            root = Path(getattr(cfg, attr)).resolve()
        except Exception:  # noqa: BLE001 - a config double without the folder
            continue
        if rp == root or root in rp.parents:
            return True
    return False


def _sample_overlaps(source_path: Any, seg_path: str, gone: list[Any]) -> bool:
    """Was this voice sample ("<segment file>#a-b") cut from audio of a forgotten sentence?"""
    sp = str(source_path or "")
    if "#" not in sp:
        return False
    path, _, span = sp.rpartition("#")
    if Path(path).name != Path(seg_path).name:
        return False
    try:
        a, b = (float(x) for x in span.split("-", 1))
    except ValueError:
        return True
    return any(a < float(u["end_s"]) and float(u["start_s"]) < b for u in gone)


# ------------------------------------------------------------------------------------------- forget
def forget(db: DB, cfg: Any, *, day: str | None = None, time_from: str | None = None, time_to: str | None = None,
           utterance_ids: Iterable[int] | None = None, session_id: int | None = None, dry_run: bool = True,
           purge_backups: bool = True, vacuum: bool | None = None) -> dict[str, Any]:
    """Forget everything in the scope (the union of what the arguments select) across all layers; see the
    module docstring for the layers.

    day                      a whole day (YYYY-MM-DD), or the day `time_from` / `time_to` given as HH:MM belong to
    time_from / time_to      sentences that START in [time_from, time_to); ISO date-time or HH:MM with `day`
    utterance_ids            these sentences
    session_id               every sentence of that session
    dry_run                  True (default): change nothing, report what would happen
    purge_backups            also delete pre-upgrade database copies that still contain the sentences
    vacuum                   rewrite the database file afterwards; None = only when it is small (VACUUM_MAX_BYTES)

    Returns the report (same shape for a dry run): {"dry_run", "scope", "utterances", "chars", "days", "segments":
    {"whole", "partial", "skipped"}, "audio_files", "audio_bytes", "audio_kept": [segment ids], "transcripts",
    "cluster_embeddings", "vectors", "voice_vectors", "revisions", "sessions": {"removed", "updated"},
    "items": {"tombstoned", "trimmed", "removed", "unverified"}, "item_ids": {...}, "digests": [days],
    "markdown": [days], "inbox", "chat_messages", "voice_samples_kept": [sample ids], "backups": [file names],
    "physical": {...}, "notes": [Chinese remarks], "errors": [...]}.
    Raises ValueError when no scope is given or a value cannot be read."""
    day = _day_arg(day)
    t0, t1 = _parse_time(time_from, day, "time_from"), _parse_time(time_to, day, "time_to")
    ids_given = [int(u) for u in utterance_ids] if utterance_ids is not None else []
    if not (day or t0 or t1 or ids_given or session_id is not None):
        raise ValueError("请指定要忘记的范围：某一天、一段时间、一个会话或若干句话")
    if t0 and t1 and t1 <= t0:
        raise ValueError("time_to 必须晚于 time_from")
    apply = not dry_run
    # only a whole day may take the day's evidence-less automatic items with it: forgetting ten minutes must not
    # wipe every todo and decision of the day (the digest stores them without evidence)
    whole_day = bool(day) and not (t0 or t1 or ids_given or session_id is not None)
    report: dict[str, Any] = {
        "dry_run": bool(dry_run),
        "scope": {"day": day, "time_from": t0.isoformat(timespec="seconds") if t0 else None,
                  "time_to": t1.isoformat(timespec="seconds") if t1 else None, "utterance_ids": len(ids_given),
                  "session_id": int(session_id) if session_id is not None else None},
        "utterances": 0, "chars": 0, "days": [], "segments": {"whole": 0, "partial": 0, "skipped": 0},
        "audio_files": 0, "audio_bytes": 0, "audio_kept": [], "transcripts": 0, "cluster_embeddings": 0, "vectors": 0,
        "voice_vectors": 0, "revisions": 0, "sessions": {"removed": 0, "updated": 0}, "audio_outside": [],
        "items": {"tombstoned": 0, "trimmed": 0, "removed": 0, "unverified": 0},
        "item_ids": {"tombstoned": [], "trimmed": [], "removed": [], "unverified": []},
        "digests": [], "markdown": [], "inbox": 0, "chat_messages": 0, "voice_samples_kept": [], "backups": [],
        "physical": {}, "notes": [], "errors": [],
    }
    conn = db.conn()
    if apply:
        try:
            conn.execute("PRAGMA secure_delete=ON")      # freed cells and pages are overwritten with zeros
        except sqlite3.Error as e:
            report["errors"].append(f"secure_delete: {e}")

    files_to_delete: list[tuple[int, Path]] = []
    with db.transaction() as c:
        rows, window_segments = _select(db, day, t0, t1, ids_given, session_id)
        by_seg: dict[int, list[Any]] = {}
        for u in rows:
            by_seg.setdefault(int(u["segment_id"]), []).append(u)
        seg_ids = sorted(set(by_seg) | window_segments)
        segs = {}
        for part in _in(seg_ids):
            for s in c.execute(f"SELECT id, path, status, started_at FROM segments WHERE id IN ({','.join('?' * len(part))})",
                               part).fetchall():
                segs[int(s["id"])] = s
        # a segment in the middle of transcription would get its sentences back a moment later: leave it alone
        busy = {sid for sid, s in segs.items() if s["status"] == "processing"}
        if busy:
            report["segments"]["skipped"] = len(busy)
            report["errors"].append(f"{len(busy)} 段录音正在转写，这次没有处理：稍后再试一次")
            rows = [u for u in rows if int(u["segment_id"]) not in busy]
            for sid in busy:
                by_seg.pop(sid, None)
        gone = {int(u["id"]) for u in rows}
        days = sorted({str(u["day"]) for u in rows} | {str(segs[s]["started_at"])[:10] for s in window_segments if s in segs
                                                         and s not in busy})
        report["utterances"], report["chars"], report["days"] = len(rows), sum(len(u["text"] or "") for u in rows), days

        # --- segments: whole (audio, transcript, clusters go) or partial (only the forgotten items are cut out)
        whole: list[int] = []
        for sid in seg_ids:
            if sid in busy or sid not in segs:
                continue
            left = c.execute("SELECT COUNT(*) FROM utterances WHERE segment_id=?", (sid,)).fetchone()[0] - len(by_seg.get(sid, []))
            if sid in window_segments or left == 0:
                whole.append(sid)
            else:
                report["segments"]["partial"] += 1
                report["audio_kept"].append(sid)
                report["transcripts"] += _scrub_transcript(c, sid, by_seg[sid], apply)
                labels = {u["cluster_label"] for u in by_seg[sid] if u["cluster_label"] is not None}
                for label in labels:
                    still = c.execute("SELECT COUNT(*) FROM utterances WHERE segment_id=? AND cluster_label=?",
                                      (sid, label)).fetchone()[0] - sum(1 for u in by_seg[sid] if u["cluster_label"] == label)
                    if still == 0:
                        n = c.execute("SELECT COUNT(*) FROM cluster_embeddings WHERE segment_id=? AND cluster_label=?",
                                      (sid, label)).fetchone()[0]
                        report["cluster_embeddings"] += int(n)
                        if apply:
                            c.execute("DELETE FROM cluster_embeddings WHERE segment_id=? AND cluster_label=?", (sid, label))
            for smp in c.execute("SELECT id, source_path FROM speaker_embeddings WHERE source_path LIKE ?",
                                 ("%" + Path(str(segs[sid]["path"])).name + "#%",)).fetchall():
                if sid in whole or _sample_overlaps(smp["source_path"], str(segs[sid]["path"]), by_seg.get(sid, [])):
                    report["voice_samples_kept"].append(int(smp["id"]))
        report["segments"]["whole"] = len(whole)
        for sid in whole:
            p = Path(str(segs[sid]["path"]))
            try:
                if p.is_file() and not _owned(cfg, p):
                    report["audio_outside"].append(sid)
                elif p.is_file():
                    report["audio_files"] += 1
                    report["audio_bytes"] += p.stat().st_size
                    files_to_delete.append((sid, p))
            except OSError:
                pass
            report["transcripts"] += int(c.execute("SELECT COUNT(*) FROM transcripts WHERE segment_id=?", (sid,)).fetchone()[0])
            report["cluster_embeddings"] += int(c.execute("SELECT COUNT(*) FROM cluster_embeddings WHERE segment_id=?",
                                                          (sid,)).fetchone()[0])
            if apply:
                c.execute("DELETE FROM transcripts WHERE segment_id=?", (sid,))
                c.execute("DELETE FROM cluster_embeddings WHERE segment_id=?", (sid,))
                c.execute("UPDATE segments SET status='deleted', error=NULL, speech_s=NULL, updated_at=? WHERE id=?",
                          (now_iso(), sid))

        # --- what hangs on the sentences
        sessions_hit: set[int] = {int(u["session_id"]) for u in rows if u["session_id"] is not None}
        neighbours: set[int] = set()
        for part in _in(sorted(gone)):
            marks = ",".join("?" * len(part))
            report["vectors"] += int(c.execute(f"SELECT COUNT(*) FROM utterance_vectors WHERE utterance_id IN ({marks})", part).fetchone()[0])
            report["vectors"] += int(c.execute(f"SELECT COUNT(*) FROM memory_vectors WHERE kind='utt' AND ref_id IN ({marks})", part).fetchone()[0])
            report["voice_vectors"] += int(c.execute(f"SELECT COUNT(*) FROM utterance_voice WHERE utterance_id IN ({marks})", part).fetchone()[0])
            report["revisions"] += int(c.execute(f"SELECT COUNT(*) FROM utterance_revisions WHERE utterance_id IN ({marks})", part).fetchone()[0])
        if apply:
            # the "window" vectors of the sentences next to a forgotten one were computed with its text
            for u in rows:
                for sql in ("SELECT id FROM utterances WHERE abs_start < ? OR (abs_start = ? AND id < ?) ORDER BY abs_start DESC, id DESC LIMIT 1",
                            "SELECT id FROM utterances WHERE abs_start > ? OR (abs_start = ? AND id > ?) ORDER BY abs_start, id LIMIT 1"):
                    nb = c.execute(sql, (u["abs_start"], u["abs_start"], int(u["id"]))).fetchone()
                    if nb is not None and int(nb["id"]) not in gone:
                        neighbours.add(int(nb["id"]))
            for part in _in(sorted(gone)):
                marks = ",".join("?" * len(part))
                c.execute(f"DELETE FROM memory_vectors WHERE kind='utt' AND ref_id IN ({marks})", part)
                c.execute(f"DELETE FROM utterances WHERE id IN ({marks})", part)      # tokens / FTS, vectors, voice, revisions cascade
            for part in _in(sorted(neighbours)):
                c.execute(f"DELETE FROM memory_vectors WHERE kind='utt' AND view='window' AND ref_id IN ({','.join('?' * len(part))})", part)

        # --- sessions: which ones lose everything (the rows are changed further down, after the memory items
        # that point at them were looked at)
        from . import sessions as _sessions

        removed_sessions: set[int] = set()
        for sid in sorted(sessions_hit):
            left = c.execute("SELECT COUNT(*) FROM utterances WHERE session_id=?", (sid,)).fetchone()[0]
            left -= 0 if apply else sum(1 for u in rows if u["session_id"] == sid)
            if left <= 0:
                removed_sessions.add(sid)
                report["sessions"]["removed"] += 1
            else:
                report["sessions"]["updated"] += 1

        # --- memory items
        # a partial forget removes only the items that mention what is being forgotten (see _forgotten_terms)
        hint = set() if whole_day else _forgotten_terms(c, rows, gone, days)
        today_items = c.execute("SELECT id, evidence, session_id, source, pinned, day, valid_from, status, title, content FROM memory_items"
                                " WHERE status <> 'rejected' OR COALESCE(title, '') <> ''").fetchall()
        affected_days = set(days)
        for it in today_items:
            iid = int(it["id"])
            ev = [int(x) for x in _json(it["evidence"], []) if isinstance(x, (int, float))]
            lost = [e for e in ev if e in gone]
            if lost:
                left_ev = [e for e in ev if e not in gone]
                if left_ev:
                    key = "trimmed"
                    if apply:
                        c.execute("UPDATE memory_items SET evidence=?, updated_at=? WHERE id=?", (json.dumps(left_ev), now_iso(), iid))
                else:
                    key = "tombstoned"
            elif not ev and it["session_id"] is not None and int(it["session_id"]) in removed_sessions:
                key = "tombstoned"
            elif (not ev and it["status"] != "rejected" and it["source"] != "user" and not it["pinned"]
                  and it["day"] in affected_days and (it["valid_from"] or it["day"]) == it["day"]
                  and (whole_day or any(t in f"{it['title'] or ''} {it['content'] or ''}" for t in hint))):
                key = "removed"         # an evidence-less automatic item of that day: the next digest rebuilds what still holds
            elif not ev and it["status"] != "rejected" and it["day"] in affected_days:
                key = "unverified"      # cannot be traced to sentences (older than the day, or the user's own): left alone
            else:
                continue
            report["items"][key] += 1
            report["item_ids"][key].append(iid)
            if not apply or key in ("trimmed", "unverified"):
                continue
            c.execute("DELETE FROM memory_item_tokens WHERE item_id=?", (iid,))
            c.execute("DELETE FROM memory_vectors WHERE kind='item' AND ref_id=?", (iid,))
            if key == "removed":
                c.execute("UPDATE memory_items SET superseded_by=NULL WHERE superseded_by=?", (iid,))
                c.execute("DELETE FROM memory_items WHERE id=?", (iid,))
            else:
                # a scrubbed tombstone: the text is gone and the fingerprint keeps automatic producers from writing the
                # same fact again - except for a person, whose fingerprint is just the name: other days may still
                # mention them, and a name-wide block would keep them out of memory for good
                c.execute("UPDATE memory_items SET status='rejected', title='', content='', subject=NULL, owner=NULL, due=NULL,"
                          " fingerprint=CASE WHEN kind='person' THEN NULL ELSE fingerprint END,"
                          " evidence=NULL, session_id=NULL, pinned=0, superseded_by=NULL, updated_at=? WHERE id=?", (now_iso(), iid))

        if apply:
            gap = timedelta(minutes=_sessions.gap_minutes(cfg))
            for sid in sorted(sessions_hit):
                # title / summary were written from content that is partly gone: the episode step writes them again
                c.execute("UPDATE sessions SET kind=NULL, title=NULL, summary=NULL, topics=NULL, participants=NULL, meta=NULL,"
                          " digested_hash=NULL, digested_at=NULL WHERE id=?", (sid,))
                c.execute("DELETE FROM memory_vectors WHERE kind='session' AND ref_id=?", (sid,))
                _sessions.repair(db, sid, gap)

        # --- digests (row now, file after the commit)
        for d in days:
            if c.execute("SELECT 1 FROM daily_summaries WHERE day=?", (d,)).fetchone() is not None or (
                    cfg.daily_dir / f"{d}.md").exists():
                report["digests"].append(d)
            if apply:
                c.execute("DELETE FROM daily_summaries WHERE day=?", (d,))

        # --- inbox and stored chats
        for it in c.execute("SELECT id, utterance_id, session_id, payload FROM inbox_items").fetchall():
            payload = _json(it["payload"], {})
            payload = payload if isinstance(payload, dict) else {}
            main = it["utterance_id"] if it["utterance_id"] is not None else payload.get("utterance_id")
            try:
                main = int(main) if main is not None else None
            except (TypeError, ValueError):
                main = None
            examples = payload.get("examples") if isinstance(payload.get("examples"), list) else []
            kept = [e for e in examples if not (isinstance(e, dict) and e.get("utterance_id") in gone)]
            if (main is not None and main in gone) or (it["session_id"] is not None and int(it["session_id"]) in removed_sessions) \
                    or (examples and not kept and main is None):
                report["inbox"] += 1
                if apply:
                    c.execute("DELETE FROM inbox_items WHERE id=?", (int(it["id"]),))
            elif len(kept) != len(examples):
                report["inbox"] += 1
                if apply:
                    payload["examples"] = kept
                    c.execute("UPDATE inbox_items SET payload=?, updated_at=? WHERE id=?",
                              (json.dumps(payload, ensure_ascii=False), now_iso(), int(it["id"])))
        for m in c.execute("SELECT id, meta FROM chat_messages WHERE meta IS NOT NULL").fetchall():
            if _cited_ids(_json(m["meta"], {})) & gone:
                report["chat_messages"] += 1
                if apply:
                    c.execute("DELETE FROM chat_messages WHERE id=?", (int(m["id"]),))
        if apply and report["digests"]:
            pending = sorted(set(db.get_kv(KV_REBUILD_DAYS, []) or []) | set(report["digests"]))
            db.set_kv(KV_REBUILD_DAYS, pending)

    # --- backups that still hold the sentences (taken after the earliest of them was stored)
    if purge_backups and rows:
        earliest = min((str(u["created_at"] or "") for u in rows), default="")
        bdir = Path(db.path).parent / BACKUP_DIR_NAME
        try:
            first = datetime.fromisoformat(earliest.replace(" ", "T")).timestamp() if earliest else 0.0
        except ValueError:
            first = 0.0
        for f in sorted(bdir.glob("*.sqlite3")) if bdir.is_dir() else []:
            try:
                if f.stat().st_mtime >= first:
                    report["backups"].append(f.name)
                    if apply:
                        f.unlink()
            except OSError as e:
                report["errors"].append(f"备份 {f.name} 没能删除：{e}")

    # --- Markdown of the affected days
    md_days = [d for d in days if (cfg.transcripts_dir / f"{d}.md").exists()]
    report["markdown"] = md_days
    if apply:
        for sid, p in files_to_delete:
            try:
                p.unlink(missing_ok=True)
            except OSError as e:
                report["errors"].append(f"录音文件 {p.name} 没能删除（可能正在播放）：{e}")
        for d in days:
            try:
                (cfg.daily_dir / f"{d}.md").unlink(missing_ok=True)
                md = cfg.transcripts_dir / f"{d}.md"
                if db.one("SELECT 1 FROM utterances WHERE day=? LIMIT 1", (d,)) is None:
                    md.unlink(missing_ok=True)
                elif md.exists():
                    from ..speech.markdown import write_day_markdown

                    write_day_markdown(db, cfg, d)
            except Exception as e:  # noqa: BLE001 - reported, the database is already clean
                report["errors"].append(f"{d} 的 Markdown 没能更新：{e}")
        # sentences cut loose by a hole in their session become sessions of their own
        try:
            _sessions.sync(cfg, db)
        except Exception as e:  # noqa: BLE001
            report["errors"].append(f"会话没有重新整理：{e}")
        report["physical"] = _scrub_file(db, vacuum)

    if report["audio_kept"]:
        report["notes"].append(f"有 {len(report['audio_kept'])} 段录音里还有别的话，录音文件保留着（里面仍能听到被忘记的那几句）；"
                               "想连声音一起删，请把整段都选上。")
    if report["voice_samples_kept"]:
        report["notes"].append(f"有 {len(report['voice_samples_kept'])} 个声音样本是从这些录音里截取的，没有动；可以在「说话人」里删除。")
    if report["items"]["trimmed"]:
        report["notes"].append(f"有 {report['items']['trimmed']} 条记忆还有别的出处，保留了下来，建议看一眼内容是否还合适。")
    if report["audio_outside"]:
        report["notes"].append(f"有 {len(report['audio_outside'])} 个录音文件不在这个档案的数据目录里（多半是从别处复制来的数据库），没有删除。")
    if any("-v0-" in b for b in report["backups"]):
        report["notes"].append("要删除的备份里有升级前的数据库备份；删掉后就不能再退回旧版本了。")
    if report["items"]["unverified"]:
        report["notes"].append(f"有 {report['items']['unverified']} 条当天的记忆没有出处记录，无法判断是否相关，没有动。")
    if report["digests"]:
        report["notes"].append("这些天的日报已删除，可以重新生成；已经推送出去的日报收不回来。")
    if apply:
        log.info("forget: %d sentences on %s; %d whole / %d partial segments, %d audio files, items %s",
                 report["utterances"], ",".join(days) or "-", report["segments"]["whole"], report["segments"]["partial"],
                 report["audio_files"], report["items"])
    return report


def _scrub_file(db: DB, vacuum: bool | None) -> dict[str, Any]:
    """Make the deleted text unreadable in the database FILE as well: merge the FTS indexes (deleted entries
    leave their b-trees), vacuum a small database (free pages and slack inside pages disappear), fold the WAL."""
    out: dict[str, Any] = {"secure_delete": True}
    for fts in ("utterances_fts", "memory_items_fts"):
        try:
            db.execute(f"INSERT INTO {fts}({fts}) VALUES ('optimize')")
            out["fts_optimized"] = True
        except sqlite3.Error as e:
            out["fts_optimized"] = False
            log.warning("forget: optimize %s failed: %s", fts, e)
    try:
        size = Path(db.path).stat().st_size
    except OSError:
        size = 0
    want = vacuum if vacuum is not None else size <= VACUUM_MAX_BYTES
    out["vacuum"] = False
    if want:
        try:
            db.execute("VACUUM")
            out["vacuum"] = True
        except sqlite3.Error as e:          # another connection is in the middle of something: not fatal
            out["vacuum_error"] = str(e)
    try:
        row = db.one("PRAGMA wal_checkpoint(TRUNCATE)")
        out["wal_checkpoint"] = "busy" if row is not None and int(row[0]) else "ok"
    except sqlite3.Error as e:
        out["wal_checkpoint"] = f"failed: {e}"
    return out


__all__ = ["forget", "KV_REBUILD_DAYS", "VACUUM_MAX_BYTES"]
