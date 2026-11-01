"""Unknown voices: group the not-yet-identified diarization clusters of all recordings by voice, so the
user can listen to one group and give it a name (no separate enrollment recording needed).

    group_unknown_voices(db, cfg)                  -> {"groups": [...], "missing_segments": m, ...}   (numpy only)
    backfill_cluster_embeddings(db, cfg, engine)   -> compute cluster_embeddings for 'done' segments that lack them
    name_voice(db, cfg, engine, members, name)     -> enroll samples from the group, assign its utterances, mark self / relation

A "member" is one (segment_id, cluster_label) pair of cluster_embeddings whose cluster still has at
least one utterance with speaker_id IS NULL. Groups are built by greedy average-linkage clustering of
the members' embeddings (centroid = speech_s-weighted mean, re-normalised; merge while the best pair's
cosine >= cfg.speakers.voice_group_threshold).

A segment that was scanned but yielded no embedding (unreadable file, utterances outside the audio) gets a
sentinel cluster_embeddings row with cluster_label=-1 (db.mark_cluster_embeddings_empty), so it is not
reported as "missing" and re-scanned every session; grouping and naming ignore such rows.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..analysis.relations import RELATION_CHOICES  # noqa: F401  (re-exported for the API / CLI)
from ..audio.storage import read_segment
from ..config import Config
from ..db import DB
from .speakers import MIN_CLUSTER_SPEECH_S, Turn

log = logging.getLogger(__name__)

MAX_SAMPLES_PER_GROUP = 5
DEFAULT_GROUP_THRESHOLD = 0.55


# --------------------------------------------------------------------------- helpers
def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _hhmm(abs_start: str | None) -> str:
    try:
        return datetime.fromisoformat(str(abs_start)).strftime("%H:%M")
    except (TypeError, ValueError):
        return ""


def _group_key(members: list[tuple[int, int]]) -> str:
    s = ",".join(f"{seg}:{cl}" for seg, cl in sorted(members))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def _threshold(cfg: Config, threshold: float | None) -> float:
    if threshold is not None:
        return float(threshold)
    try:
        return float(cfg.speakers.get("voice_group_threshold", DEFAULT_GROUP_THRESHOLD))
    except (TypeError, ValueError, AttributeError):
        return DEFAULT_GROUP_THRESHOLD


def missing_segments(db: DB) -> list[Any]:
    """'done' segments with unidentified utterances (at least one long enough to embed) whose audio file
    still exists but that have no cluster_embeddings row yet (the sentinel of a scan that found nothing
    counts as a row): the ones backfill_cluster_embeddings scans."""
    rows = db.query(
        "SELECT s.id, s.path FROM segments s WHERE s.status='done'"
        " AND EXISTS (SELECT 1 FROM utterances u WHERE u.segment_id=s.id AND u.speaker_id IS NULL AND u.end_s - u.start_s >= ?)"
        " AND NOT EXISTS (SELECT 1 FROM cluster_embeddings ce WHERE ce.segment_id=s.id)"
        " ORDER BY s.started_at", (MIN_CLUSTER_SPEECH_S,))
    return [r for r in rows if Path(r["path"]).exists()]


def unknown_utterance_count(db: DB) -> int:
    row = db.one("SELECT COUNT(*) AS n FROM utterances WHERE speaker_id IS NULL")
    return int(row["n"]) if row else 0


# --------------------------------------------------------------------------- clustering
def cluster_members(embs: np.ndarray, weights: np.ndarray, threshold: float) -> list[list[int]]:
    """Greedy average-linkage clustering. embs: (n, d) unit vectors; weights: (n,) > 0.
    Returns groups of row indices (every row in exactly one group)."""
    n = int(embs.shape[0])
    if n == 0:
        return []
    groups: list[list[int]] = [[i] for i in range(n)]
    cent = embs.astype(np.float32, copy=True)
    active = np.ones(n, dtype=bool)
    sims = cent @ cent.T
    np.fill_diagonal(sims, -np.inf)
    while True:
        flat = int(np.argmax(sims))
        i, j = divmod(flat, n)
        if not np.isfinite(sims[i, j]) or sims[i, j] < threshold:
            break
        groups[i].extend(groups[j])
        groups[j] = []
        active[j] = False
        sims[j, :] = -np.inf
        sims[:, j] = -np.inf
        idx = groups[i]
        cent[i] = _unit((embs[idx] * weights[idx][:, None]).sum(axis=0))
        row = cent @ cent[i]
        row[~active] = -np.inf
        row[i] = -np.inf
        sims[i, :] = row
        sims[:, i] = row
    return [g for g in groups if g]


def _closest_enrolled(centroid: np.ndarray, enrolled: list[tuple[int, str, np.ndarray]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for sid, name, e in enrolled:
        if e.shape[0] != centroid.shape[0]:
            continue
        s = float(np.dot(centroid, _unit(e)))
        if best is None or s > best["score"]:
            best = {"id": int(sid), "name": name, "score": round(s, 3)}
    return best


# --------------------------------------------------------------------------- public: grouping
def group_unknown_voices(db: DB, cfg: Config, threshold: float | None = None, max_members: int = 2000) -> dict[str, Any]:
    """Group the unidentified clusters of all recordings by voice (pure numpy, no model).

    Returns {"groups": [...], "missing_segments": m, "threshold": t, "unknown_utterances": total}; see the
    module docstring. Groups are sorted by speech_s desc; each has "key", "members", "utterances", "speech_s",
    "segments", "days", "n_days", "samples" (<= 5, with audio and longest first) and "closest" (the most
    similar enrolled speaker, a hint "maybe X") or None.
    """
    thr = _threshold(cfg, threshold)
    out: dict[str, Any] = {"groups": [], "missing_segments": len(missing_segments(db)), "threshold": thr,
                           "unknown_utterances": unknown_utterance_count(db)}
    rows = db.query(
        "SELECT ce.segment_id, ce.cluster_label, ce.embedding, ce.dim, ce.speech_s, s.path"
        " FROM cluster_embeddings ce JOIN segments s ON s.id = ce.segment_id"
        " WHERE ce.cluster_label >= 0"
        " AND EXISTS (SELECT 1 FROM utterances u WHERE u.segment_id=ce.segment_id AND u.cluster_label=ce.cluster_label"
        "             AND u.speaker_id IS NULL)"
        " ORDER BY s.started_at DESC, ce.segment_id DESC, ce.cluster_label LIMIT ?", (int(max_members),))
    if not rows:
        return out
    # members must share one embedding dimension (a model change leaves old rows behind): keep the majority
    dims: dict[int, int] = {}
    for r in rows:
        dims[int(r["dim"])] = dims.get(int(r["dim"]), 0) + 1
    dim = max(dims.items(), key=lambda kv: kv[1])[0]
    members = [r for r in rows if int(r["dim"]) == dim]
    embs = np.stack([_unit(np.frombuffer(r["embedding"], dtype=np.float32)) for r in members])
    weights = np.array([max(float(r["speech_s"]), 0.1) for r in members], dtype=np.float32)
    has_audio = {int(r["segment_id"]): Path(r["path"]).exists() for r in members}

    # unidentified utterances of the member clusters, grouped by (segment, cluster)
    seg_ids = sorted({int(r["segment_id"]) for r in members})
    utts: dict[tuple[int, int], list[Any]] = {}
    for k in range(0, len(seg_ids), 500):
        chunk = seg_ids[k:k + 500]
        for u in db.query(
                "SELECT id, segment_id, cluster_label, day, abs_start, start_s, end_s, text FROM utterances"
                f" WHERE speaker_id IS NULL AND cluster_label IS NOT NULL AND segment_id IN ({','.join('?' * len(chunk))})"
                " ORDER BY abs_start, start_s, id", chunk):
            utts.setdefault((int(u["segment_id"]), int(u["cluster_label"])), []).append(u)

    enrolled = db.speaker_embeddings()
    groups: list[dict[str, Any]] = []
    for idx in cluster_members(embs, weights, thr):
        mem = [(int(members[i]["segment_id"]), int(members[i]["cluster_label"])) for i in idx]
        rows_u = [u for m in mem for u in utts.get(m, [])]
        if not rows_u:
            continue
        centroid = _unit((embs[idx] * weights[idx][:, None]).sum(axis=0))
        days = sorted({str(u["day"]) for u in rows_u})
        speech_s = float(sum(max(0.0, float(u["end_s"]) - float(u["start_s"])) for u in rows_u))
        cands = sorted(rows_u, key=lambda u: (not has_audio.get(int(u["segment_id"]), False),
                                               -(float(u["end_s"]) - float(u["start_s"]))))
        samples = [{"utterance_id": int(u["id"]), "day": str(u["day"]), "time": _hhmm(u["abs_start"]),
                    "text": (u["text"] or "").strip(), "seconds": round(float(u["end_s"]) - float(u["start_s"]), 1),
                    "audio": bool(has_audio.get(int(u["segment_id"]), False))}
                   for u in cands[:MAX_SAMPLES_PER_GROUP]]
        groups.append({
            "key": _group_key(mem), "members": [[seg, cl] for seg, cl in sorted(mem)],
            "utterances": len(rows_u), "speech_s": round(speech_s, 1), "segments": len({seg for seg, _ in mem}),
            "days": days, "n_days": len(days), "samples": samples,
            "closest": _closest_enrolled(centroid, enrolled),
        })
    groups.sort(key=lambda g: (-g["speech_s"], -g["utterances"], g["key"]))
    out["groups"] = groups
    return out


# --------------------------------------------------------------------------- public: backfill
def _segment_turns(db: DB, seg_id: int) -> list[Turn]:
    """The stored utterances of a segment as diarization turns (same mapping as Processor.relabel_segment)."""
    utts = db.query("SELECT start_s, end_s, cluster_label FROM utterances WHERE segment_id=? ORDER BY start_s, id", (int(seg_id),))
    return [Turn(float(u["start_s"]), float(u["end_s"]), int(u["cluster_label"] if u["cluster_label"] is not None else i))
            for i, u in enumerate(utts)]


def backfill_cluster_embeddings(db: DB, cfg: Config, engine: Any, limit: int | None = None, stop_event: Any = None) -> dict[str, int]:
    """Compute and store cluster_embeddings for the segments missing_segments() lists (no identification).
    Returns {"segments": scanned, "clusters": stored, "skipped": segments that failed or had nothing to embed}.
    A skipped segment gets the sentinel row, so it is not listed as missing again (process_segment clears it)."""
    stats = {"segments": 0, "clusters": 0, "skipped": 0}
    rows = missing_segments(db)
    if limit:
        rows = rows[: int(limit)]
    for r in rows:
        if stop_event is not None and stop_event.is_set():
            log.info("voices backfill: stop requested, %d segment(s) left", len(rows) - stats["segments"] - stats["skipped"])
            break
        seg_id = int(r["id"])
        stored = 0
        try:
            audio = read_segment(r["path"])
            clusters: dict[int, list[Turn]] = {}
            for t in _segment_turns(db, seg_id):
                clusters.setdefault(t.cluster, []).append(t)
            for cid, ts in clusters.items():
                emb = engine.cluster_embedding(audio, ts)
                if emb is None:
                    continue
                db.upsert_cluster_embedding(seg_id, cid, emb, sum(t.duration for t in ts))
                stored += 1
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the scan
            log.warning("voices backfill: segment #%d skipped: %s", seg_id, e)
        if stored:
            stats["segments"] += 1
            stats["clusters"] += stored
        else:
            stats["skipped"] += 1
            _mark_scanned_empty(db, seg_id)
    log.info("voices backfill: %s", stats)
    return stats


def _mark_scanned_empty(db: DB, seg_id: int) -> None:
    try:
        db.mark_cluster_embeddings_empty(seg_id)
    except Exception as e:  # noqa: BLE001 - the scan result still stands; the segment is just re-scanned later
        log.warning("voices backfill: cannot mark segment #%d as scanned: %s", seg_id, e)


# --------------------------------------------------------------------------- public: naming
def _parse_members(members: list[Any]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for m in members or []:
        try:
            seg, cl = int(m[0]), int(m[1])
        except (TypeError, ValueError, IndexError, KeyError) as e:
            raise ValueError(f"members 格式错误: {m!r}") from e
        if cl < 0:
            raise ValueError(f"members 格式错误: {m!r}")
        if (seg, cl) not in out:
            out.append((seg, cl))
    if not out:
        raise ValueError("members 不能为空")
    return out


def name_voice(db: DB, cfg: Config, engine: Any, members: list[Any], name: str, *, is_self: bool = False,
               relation: str | None = None, max_samples: int = 3) -> dict[str, Any]:
    """Give a voice group a name: enroll up to `max_samples` embeddings from its longest members (audio
    concatenated per cluster), assign every utterance of every member to the speaker (speaker_source='user'),
    optionally mark the speaker as the user / store the relation, and rewrite the affected days' Markdown.

    When none of the members' recordings can be read any more (expired), the stored cluster embeddings of
    the top members are enrolled directly instead (they are the same L2-normalised CAM++ vectors an
    enrolment produces); 'samples_from_clusters' counts those (included in 'samples_added').
    Utterances the user already labelled by hand as somebody else are kept and counted in
    'skipped_user_labels'. Raises ValueError for an empty name, malformed members, or when a brand-new
    speaker got no usable sample (the speaker row is removed again in that case). FTS rebuild is the
    caller's job."""
    from .pipeline import write_day_markdown

    name = (name or "").strip()
    if not name:
        raise ValueError("姓名不能为空")
    mem = _parse_members(members)
    unknown = str(cfg.speakers.get("unknown_label") or "未知")
    existed = db.one("SELECT id FROM speakers WHERE name=?", (name,)) is not None
    sid = db.upsert_speaker(name)

    # 2. samples from the members with the most speech (and an audio file)
    member_rows = {(seg, cl): (emb, s) for seg, cl, emb, s in db.cluster_embeddings(sorted({seg for seg, _ in mem}))}
    speech_by_member = {k: v[1] for k, v in member_rows.items()}
    seg_rows = {seg: db.get_segment(seg) for seg in {seg for seg, _ in mem}}
    ranked = sorted(mem, key=lambda m: -speech_by_member.get(m, 0.0))
    samples_added = 0
    samples_from_clusters = 0
    tried = 0           # members with an audio file (and utterances) we attempted
    audio_read = 0      # ... of which the file could be read (the engine saw a clip)
    for seg, cl in ranked:
        if tried >= int(max_samples):
            break
        row = seg_rows.get(seg)
        if row is None or not Path(row["path"]).exists():
            continue
        utts = db.query("SELECT start_s, end_s FROM utterances WHERE segment_id=? AND cluster_label=? ORDER BY start_s, id", (seg, cl))
        if not utts:
            continue
        tried += 1
        try:
            audio = read_segment(row["path"])
            from ..audio.vad import SAMPLE_RATE

            parts = [audio[int(float(u["start_s"]) * SAMPLE_RATE): int(float(u["end_s"]) * SAMPLE_RATE)] for u in utts]
            clip = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
            audio_read += 1
            # the clip is a concatenation of non-contiguous ranges: the sample row must point at the
            # speech-only copy (playable, survives audio expiry), not at a "#a-b" slice of the segment
            source = f"{row['path']}#{float(utts[0]['start_s']):.1f}-{float(utts[-1]['end_s']):.1f}"
            engine.enroll_from_audio(db, name, clip, source=source, copy_as_source=True)
            samples_added += 1
        except ValueError as e:              # too little speech in this member
            log.info("name_voice: member %s/%s skipped: %s", seg, cl, e)
        except Exception as e:  # noqa: BLE001 - unreadable file etc.
            log.warning("name_voice: member %s/%s failed: %s", seg, cl, e)
    if samples_added == 0 and audio_read == 0:
        # the group's recordings are gone (expired / unreadable): enrol the stored cluster embeddings
        for seg, cl in ranked[: max(0, int(max_samples))]:
            got = member_rows.get((seg, cl))
            if got is None or got[0].size == 0:
                continue
            db.add_speaker_embedding(sid, got[0], None, got[1])
            samples_from_clusters += 1
        samples_added += samples_from_clusters
        if samples_from_clusters:
            log.info("name_voice: %s: no readable recording, enrolled %d stored cluster embedding(s)", name, samples_from_clusters)
    if samples_added == 0 and not existed:
        db.delete_speaker(sid, unknown)      # nothing assigned yet: just drops the empty row (+ relation cache)
        if audio_read == 0:
            raise ValueError("这组声音的录音已过期，也没有可用的声纹，无法登记")
        raise ValueError("这组声音里的语音太短，无法提取声纹")

    # 3. assign every utterance of every member (the user confirmed the group by ear) - except the ones
    #    they already labelled by hand as somebody else
    assigned = 0
    skipped_user = 0
    days: set[str] = set()
    segments: set[int] = set()
    for seg, cl in mem:
        other = db.one("SELECT COUNT(*) AS n FROM utterances WHERE segment_id=? AND cluster_label=?"
                       " AND speaker_source='user' AND speaker_id IS NOT NULL AND speaker_id<>?", (seg, cl, sid))
        skipped_user += int(other["n"]) if other else 0
        cur = db.execute(
            "UPDATE utterances SET speaker_id=?, speaker_name=?, speaker_score=NULL, speaker_source='user'"
            " WHERE segment_id=? AND cluster_label=?"
            " AND (speaker_source IS NULL OR speaker_source<>'user' OR speaker_id IS NULL OR speaker_id=?)",
            (sid, name, seg, cl, sid))
        n = int(cur.rowcount or 0)
        if n:
            assigned += n
            segments.add(seg)
            for d in db.query("SELECT DISTINCT day FROM utterances WHERE segment_id=? AND cluster_label=? AND speaker_id=?", (seg, cl, sid)):
                days.add(str(d["day"]))

    # 4. self / relation
    if is_self:
        db.set_self_speaker(sid)
    if relation and str(relation).strip():
        db.set_relation(sid, str(relation).strip(), "user")

    # 5. markdown
    for day in sorted(days):
        try:
            write_day_markdown(db, cfg, day)
        except Exception as e:  # noqa: BLE001 - the DB is the source of truth
            log.warning("cannot write markdown for %s: %s", day, e)
    log.info("name_voice: %s <- %d member(s), %d sample(s) (%d from stored clusters), %d utterance(s), %d hand-labelled kept",
             name, len(mem), samples_added, samples_from_clusters, assigned, skipped_user)
    return {"id": int(sid), "name": name, "samples_added": samples_added, "samples_from_clusters": samples_from_clusters,
            "utterances_assigned": assigned, "skipped_user_labels": skipped_user,
            "segments": sorted(segments), "days": sorted(days)}


__all__ = ["RELATION_CHOICES", "group_unknown_voices", "backfill_cluster_embeddings", "name_voice", "cluster_members",
           "missing_segments", "unknown_utterance_count"]
