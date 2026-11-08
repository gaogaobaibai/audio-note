"""Unknown voices: group the not-yet-identified diarization clusters of all recordings by voice, so the
user can listen to one group and give it a name (no separate enrollment recording needed).

    group_unknown_voices(db, cfg)                  -> {"groups": [...], "missing_segments": m, ...}   (numpy only)
    backfill_cluster_embeddings(db, cfg, engine)   -> compute cluster_embeddings for 'done' segments that lack them
    name_voice(db, cfg, engine, members, name)     -> enroll samples from the group, assign its utterances, mark self / relation
    reject_voice(db, cfg, engine, sid, utt_ids)    -> "not X": veto + negative samples from sentences, relabel (v4)
    reject_voice_group(db, cfg, engine, members, sid) -> the same for a voice group's unidentified sentences

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
from ..audio.loudness import min_speech_dbfs
from ..audio.storage import read_segment
from ..config import Config
from ..db import DB
from .speakers import (MIN_CLUSTER_SPEECH_S, SampleMismatch, Turn, check_sample, load_negatives, min_label_sample_s,
                       negative_margin, negative_scores, sample_consistency_min)

log = logging.getLogger(__name__)

MAX_SAMPLES_PER_GROUP = 5
DEFAULT_GROUP_THRESHOLD = 0.55


def _loud_clause(min_dbfs: float | None, alias: str = "u") -> tuple[str, list[Any]]:
    """SQL fragment keeping only utterances at or above the loudness gate (rows without a stored level pass).
    Faint (far-away) sentences are not grouped, sampled or counted: their voice cannot be told apart."""
    if min_dbfs is None:
        return "", []
    return f" AND ({alias}.loudness_db IS NULL OR {alias}.loudness_db >= ?)", [float(min_dbfs)]


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


def unknown_utterance_count(db: DB, min_dbfs: float | None = None) -> int:
    """Unidentified utterances, leaving out the ones below the loudness gate `min_dbfs` (None = count all)."""
    clause, params = _loud_clause(min_dbfs)
    row = db.one(f"SELECT COUNT(*) AS n FROM utterances u WHERE u.speaker_id IS NULL{clause}", params)
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


def _closest_enrolled(centroid: np.ndarray, enrolled: list[tuple[int, str, np.ndarray]],
                      negatives: list[tuple[int, np.ndarray]] | None = None, margin: float = 0.0,
                      exclude: set[int] | frozenset[int] | None = None) -> dict[str, Any] | None:
    """The most similar enrolled speaker (the "maybe X" hint). Speakers in `exclude` (vetoed for the group's
    sentences) and speakers whose negative samples the centroid resembles at least as much (p - n < margin) are
    never suggested."""
    per: dict[int, tuple[str, float]] = {}
    excl = set(exclude or ())
    for sid, name, e in enrolled:
        if e.shape[0] != centroid.shape[0] or int(sid) in excl:
            continue
        s = float(np.dot(centroid, _unit(e)))
        if int(sid) not in per or s > per[int(sid)][1]:
            per[int(sid)] = (name, s)
    neg = negative_scores(centroid, negatives)
    best: dict[str, Any] | None = None
    for sid, (name, s) in per.items():
        if negatives and s - neg.get(sid, -1.0) < float(margin):
            continue
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
    min_db = min_speech_dbfs(cfg)
    loud, loud_params = _loud_clause(min_db)
    out: dict[str, Any] = {"groups": [], "missing_segments": len(missing_segments(db)), "threshold": thr,
                           "unknown_utterances": unknown_utterance_count(db, min_db)}
    rows = db.query(
        "SELECT ce.segment_id, ce.cluster_label, ce.embedding, ce.dim, ce.speech_s, s.path"
        " FROM cluster_embeddings ce JOIN segments s ON s.id = ce.segment_id"
        " WHERE ce.cluster_label >= 0"
        " AND EXISTS (SELECT 1 FROM utterances u WHERE u.segment_id=ce.segment_id AND u.cluster_label=ce.cluster_label"
        f"             AND u.speaker_id IS NULL{loud})"
        " ORDER BY s.started_at DESC, ce.segment_id DESC, ce.cluster_label LIMIT ?", (*loud_params, int(max_members)))
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
                "SELECT u.id, u.segment_id, u.cluster_label, u.day, u.abs_start, u.start_s, u.end_s, u.text FROM utterances u"
                f" WHERE u.speaker_id IS NULL AND u.cluster_label IS NOT NULL AND u.segment_id IN ({','.join('?' * len(chunk))}){loud}"
                " ORDER BY u.abs_start, u.start_s, u.id", [*chunk, *loud_params]):
            utts.setdefault((int(u["segment_id"]), int(u["cluster_label"])), []).append(u)

    enrolled = db.speaker_embeddings()
    negatives = load_negatives(db)
    nmargin = negative_margin(cfg)
    all_ids = [int(u["id"]) for us in utts.values() for u in us]
    vetoed = db.rejected_speakers(all_ids) if hasattr(db, "rejected_speakers") else {}
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
            "closest": _closest_enrolled(centroid, enrolled, negatives, nmargin,
                                         {s for u in rows_u for s in vetoed.get(int(u["id"]), ())}),
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


def _consistent_members(member_rows: dict[tuple[int, int], tuple[np.ndarray, float]], mem: list[tuple[int, int]],
                        min_cos: float) -> tuple[set[tuple[int, int]], int]:
    """Members whose stored cluster embedding resembles the anchor (the member with the most speech):
    (usable members, number of inconsistent ones). Members without a comparable embedding count as usable."""
    anchor: tuple[int, int] | None = None
    for m in mem:
        got = member_rows.get(m)
        if got is None or got[0].size == 0:
            continue
        if anchor is None or got[1] > member_rows[anchor][1]:
            anchor = m
    if anchor is None:
        return set(mem), 0
    a = _unit(member_rows[anchor][0])
    ok: set[tuple[int, int]] = set()
    bad = 0
    for m in mem:
        got = member_rows.get(m)
        if got is None or got[0].size == 0 or got[0].shape != a.shape:
            ok.add(m)
            continue
        if float(np.dot(a, _unit(got[0]))) >= min_cos:
            ok.add(m)
        else:
            bad += 1
            log.info("name_voice: member %s/%s does not resemble the group's anchor; not used as a sample", m[0], m[1])
    return ok, bad


def name_voice(db: DB, cfg: Config, engine: Any, members: list[Any], name: str, *, is_self: bool = False,
               relation: str | None = None, max_samples: int = 3, force: bool = False) -> dict[str, Any]:
    """Give a voice group a name: enroll up to `max_samples` embeddings from its longest members (audio
    concatenated per cluster), assign every utterance of every member to the speaker (speaker_source='user'),
    optionally mark the speaker as the user / store the relation, and rewrite the affected days' Markdown.

    When none of the members' recordings can be read any more (expired), the stored cluster embeddings of
    the top members are enrolled directly instead (they are the same L2-normalised CAM++ vectors an
    enrolment produces); 'samples_from_clusters' counts those (included in 'samples_added').
    Utterances the user already labelled by hand as somebody else are kept and counted in
    'skipped_user_labels'; the ones the user vetoed for this speaker (「不是 TA」, utterance_rejections) are kept
    too and counted in 'skipped_vetoed'. Raises ValueError for an empty name, malformed members, or when a brand-new
    speaker got no usable sample (the speaker row is removed again in that case). FTS rebuild is the
    caller's job.

    Consistency (v3): members whose stored embedding does not resemble the group's anchor (the member
    with the most speech) are not used as samples ('members_inconsistent'); a sample that does not resemble
    the person's existing samples is skipped ('samples_rejected') unless `force`. When every candidate was
    rejected that way and nothing was added, SampleMismatch is raised before any utterance is assigned."""
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
    min_cos = sample_consistency_min(cfg)
    usable, members_inconsistent = _consistent_members(member_rows, mem, min_cos)
    ranked = sorted((m for m in mem if m in usable), key=lambda m: -speech_by_member.get(m, 0.0))
    samples_added = 0
    samples_from_clusters = 0
    samples_rejected = 0
    worst_mismatch: float | None = None
    tried = 0           # members with an audio file (and utterances) we attempted
    audio_read = 0      # ... of which the file could be read (the engine saw a clip)
    faint_only: set[tuple[int, int]] = set()     # members whose sentences all lie below the loudness gate
    enroll_kwargs: dict[str, Any] = {"force": True} if force else {}
    enroll_kwargs["min_speech_s"] = min_label_sample_s(cfg)      # short far-field snippets make unreliable samples
    enroll_kwargs["origin"] = "voice"
    loud, loud_params = _loud_clause(min_speech_dbfs(cfg))       # faint sentences never become samples
    for seg, cl in ranked:
        if tried >= int(max_samples):
            break
        row = seg_rows.get(seg)
        if row is None or not Path(row["path"]).exists():
            continue
        utts = db.query(f"SELECT u.start_s, u.end_s FROM utterances u WHERE u.segment_id=? AND u.cluster_label=?{loud}"
                        " ORDER BY u.start_s, u.id", (seg, cl, *loud_params))
        if not utts:
            if loud and db.one("SELECT 1 FROM utterances WHERE segment_id=? AND cluster_label=? LIMIT 1", (seg, cl)):
                faint_only.add((seg, cl))
                log.info("name_voice: member %s/%s has only faint sentences; not used as a sample", seg, cl)
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
            engine.enroll_from_audio(db, name, clip, source=source, copy_as_source=True, **enroll_kwargs)
            samples_added += 1
        except SampleMismatch as e:          # the group's voice does not resemble this person's samples
            samples_rejected += 1
            worst_mismatch = e.score if worst_mismatch is None else max(worst_mismatch, e.score)
            log.info("name_voice: member %s/%s rejected: %s", seg, cl, e)
        except ValueError as e:              # too little speech in this member
            log.info("name_voice: member %s/%s skipped: %s", seg, cl, e)
        except Exception as e:  # noqa: BLE001 - unreadable file etc.
            log.warning("name_voice: member %s/%s failed: %s", seg, cl, e)
    if samples_added == 0 and audio_read == 0:
        # the group's recordings are gone (expired / unreadable): enrol the stored cluster embeddings
        # (a faint-only member's embedding is a far-away voice: never a sample)
        for seg, cl in ranked[: max(0, int(max_samples))]:
            got = member_rows.get((seg, cl))
            if got is None or got[0].size == 0 or (seg, cl) in faint_only:
                continue
            score, ok = check_sample(db, sid, got[0], min_cos)
            if not ok and not force:
                samples_rejected += 1
                worst_mismatch = score if worst_mismatch is None else max(worst_mismatch, float(score))
                log.info("name_voice: stored embedding of member %s/%s rejected (cosine %.2f < %.2f)", seg, cl, score, min_cos)
                continue
            db.add_speaker_embedding(sid, got[0], None, got[1], origin="cluster")
            samples_from_clusters += 1
        samples_added += samples_from_clusters
        if samples_from_clusters:
            log.info("name_voice: %s: no readable recording, enrolled %d stored cluster embedding(s)", name, samples_from_clusters)
    if samples_added == 0 and samples_rejected and not force:
        # every usable sample of the group sounds unlike this person: nothing was added and nothing is
        # assigned; the caller may retry with force=True ("仍然加入")
        raise SampleMismatch(name, float(worst_mismatch or 0.0), min_cos)
    if samples_added == 0 and not existed:
        db.delete_speaker(sid, unknown)      # nothing assigned yet: just drops the empty row (+ relation cache)
        if faint_only and audio_read == 0:
            raise ValueError("这组声音都太轻（远处 / 不清晰），没有可用的声纹样本，无法登记")
        if audio_read == 0:
            raise ValueError("这组声音的录音已过期，也没有可用的声纹，无法登记")
        raise ValueError("这组声音里的语音太短，无法提取声纹")

    # 3. assign every utterance of every member (the user confirmed the group by ear) - except the ones
    #    they already labelled by hand as somebody else
    #    and the ones they vetoed for this person (utterance_rejections: 「不是 TA」 wins over the group)
    assigned = 0
    skipped_user = 0
    skipped_vetoed = 0
    days: set[str] = set()
    segments: set[int] = set()
    veto = "EXISTS (SELECT 1 FROM utterance_rejections r WHERE r.utterance_id=utterances.id AND r.speaker_id=?)"
    for seg, cl in mem:
        other = db.one("SELECT COUNT(*) AS n FROM utterances WHERE segment_id=? AND cluster_label=?"
                       " AND speaker_source='user' AND speaker_id IS NOT NULL AND speaker_id<>?", (seg, cl, sid))
        skipped_user += int(other["n"]) if other else 0
        vetoed = db.one(f"SELECT COUNT(*) AS n FROM utterances WHERE segment_id=? AND cluster_label=? AND {veto}"
                        " AND NOT (speaker_source='user' AND speaker_id IS NOT NULL AND speaker_id<>?)", (seg, cl, sid, sid))
        skipped_vetoed += int(vetoed["n"]) if vetoed else 0
        cur = db.execute(
            "UPDATE utterances SET speaker_id=?, speaker_name=?, speaker_score=NULL, speaker_source='user'"
            " WHERE segment_id=? AND cluster_label=?"
            " AND (speaker_source IS NULL OR speaker_source<>'user' OR speaker_id IS NULL OR speaker_id=?)"
            f" AND NOT {veto}",
            (sid, name, seg, cl, sid, sid))
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
    log.info("name_voice: %s <- %d member(s), %d sample(s) (%d from stored clusters, %d rejected, %d members inconsistent),"
             " %d utterance(s), %d hand-labelled kept, %d vetoed kept",
             name, len(mem), samples_added, samples_from_clusters, samples_rejected, members_inconsistent, assigned, skipped_user,
             skipped_vetoed)
    return {"id": int(sid), "name": name, "samples_added": samples_added, "samples_from_clusters": samples_from_clusters,
            "samples_rejected": samples_rejected, "members_inconsistent": members_inconsistent,
            "utterances_assigned": assigned, "skipped_user_labels": skipped_user, "skipped_vetoed": skipped_vetoed,
            "segments": sorted(segments), "days": sorted(days)}




# --------------------------------------------------------------------------- public: "not X" (negative samples, v4)
NEGATIVE_MIN_SPEECH_S = 1.0       # a rejected clip needs this much speech to become a negative sample
NEGATIVE_MAX_CLIPS = 3            # at most this many negative samples per rejection
NEGATIVE_MAX_S = 30.0             # each negative sample is cut to this many seconds of audio
NEGATIVE_WARN_SIMILARITY = 0.75   # a "not X" voice this similar to X's reference voice is probably a mis-click


def _speaker_row(db: DB, speaker_id: Any) -> Any:
    try:
        sid = int(speaker_id)
    except (TypeError, ValueError):
        raise ValueError("speaker_id 应为说话人 id")
    row = db.one("SELECT id, name FROM speakers WHERE id=?", (sid,))
    if row is None:
        raise LookupError("没有这个说话人")
    return row


def _reference_similarity(db: DB, cfg: Config, speaker_id: int, emb: np.ndarray) -> float | None:
    """Cosine between `emb` and the speaker's core reference voice (speakers._core_reference), None without samples."""
    from .speakers import _core_reference

    e = _unit(emb)
    embs, durs, origins = [], [], []
    for r in db.query("SELECT embedding, dim, duration_s, origin FROM speaker_embeddings WHERE speaker_id=? ORDER BY id", (int(speaker_id),)):
        if int(r["dim"]) != e.shape[0]:
            continue
        embs.append(_unit(np.frombuffer(r["embedding"], dtype=np.float32)))
        durs.append(r["duration_s"])
        origins.append(r["origin"])
    ref = _core_reference(embs, durs, sample_consistency_min(cfg), origins)
    return float(np.dot(e, ref)) if ref is not None else None


def _embed_negative(engine: Any, clip: np.ndarray) -> tuple[np.ndarray, float]:
    """(embedding, speech seconds) of a rejected clip: speech only when the engine can gate it."""
    from ..audio.vad import SAMPLE_RATE

    fn = getattr(engine, "embed_speech_only", None)
    if callable(fn):
        emb, speech_s = fn(clip, max_s=NEGATIVE_MAX_S)
        return np.asarray(emb, dtype=np.float32), float(speech_s)
    return np.asarray(engine.embed(clip), dtype=np.float32), len(clip) / SAMPLE_RATE


def _save_negative_clip(cfg: Config, name: str, clip: np.ndarray, stem: str) -> Path | None:
    """Keep a 16 kHz copy of a negative clip stitched from several sentences (a "#a-b" slice would not play it)."""
    import soundfile as sf

    from ..audio.vad import SAMPLE_RATE
    from .speakers import _safe_name

    try:
        d = cfg.speakers_dir / _safe_name(name) / "negatives"
        d.mkdir(parents=True, exist_ok=True)
        dest = d / f"{_safe_name(stem)}.wav"
        k = 0
        while dest.exists():
            k += 1
            dest = d / f"{_safe_name(stem)}_{k}.wav"
        sf.write(str(dest), np.asarray(clip, dtype=np.float32), SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return dest
    except Exception as e:  # noqa: BLE001 - the negative embedding still counts without a playable copy
        log.warning("cannot save negative clip for %s: %s", name, e)
        return None


def _reject_rows(db: DB, cfg: Config, engine: Any, sp: Any, rows: list[Any], relabel: bool = True) -> dict[str, Any]:
    """Shared part of reject_voice / reject_voice_group: negative samples from the rows grouped by (segment, cluster),
    vetoes + released attribution, markdown, relabel of the affected segments."""
    from ..audio.vad import SAMPLE_RATE
    from .pipeline import Processor, write_day_markdown

    sid, name = int(sp["id"]), str(sp["name"])
    unknown = str(cfg.speakers.get("unknown_label") or "未知")
    groups: dict[tuple[int, int | None], list[Any]] = {}
    for r in rows:
        groups.setdefault((int(r["segment_id"]), r["cluster_label"]), []).append(r)
    ranked = sorted(groups.items(), key=lambda kv: -sum(max(0.0, float(u["end_s"]) - float(u["start_s"])) for u in kv[1]))
    seg_rows = {seg: db.get_segment(seg) for seg in {k[0] for k in groups}}
    negatives_added = 0
    clips_tried = 0
    warning: str | None = None
    warn_score: float | None = None
    for (seg, cl), us in ranked:
        if clips_tried >= NEGATIVE_MAX_CLIPS:
            break
        row = seg_rows.get(seg)
        if row is None or not Path(row["path"]).exists() or engine is None:
            continue
        us = sorted(us, key=lambda u: (float(u["start_s"]), int(u["id"])))
        if sum(max(0.0, float(u["end_s"]) - float(u["start_s"])) for u in us) < NEGATIVE_MIN_SPEECH_S:
            continue
        try:
            audio = read_segment(row["path"])
            parts = [audio[int(float(u["start_s"]) * SAMPLE_RATE): int(float(u["end_s"]) * SAMPLE_RATE)] for u in us]
            clip = np.concatenate(parts)[: int(NEGATIVE_MAX_S * SAMPLE_RATE)] if parts else np.zeros(0, dtype=np.float32)
            if len(clip) < int(NEGATIVE_MIN_SPEECH_S * SAMPLE_RATE):
                continue
            clips_tried += 1
            emb, speech_s = _embed_negative(engine, clip)
        except Exception as e:  # noqa: BLE001 - unreadable file / engine failure: the veto still stands
            log.warning("reject: no negative sample from segment #%s: %s", seg, e)
            continue
        if speech_s < NEGATIVE_MIN_SPEECH_S:
            log.info("reject: segment #%s cluster %s has only %.1fs speech; veto without a negative sample", seg, cl, speech_s)
            continue
        sim = _reference_similarity(db, cfg, sid, emb)
        if sim is not None and sim >= NEGATIVE_WARN_SIMILARITY:
            # probably a mis-click: veto only, the voice does not become a negative sample
            if warn_score is None or sim > warn_score:
                warn_score = sim
                warning = f"这段声音和 {name} 的样本非常像（{sim:.2f}），确定不是 TA 吗？"
            log.info("reject: segment #%s voice resembles %s (%.2f); vetoed but not stored as a negative", seg, name, sim)
            continue
        if len(us) == 1:
            source: str | None = f"{row['path']}#{float(us[0]['start_s']):.1f}-{float(us[0]['end_s']):.1f}"
        else:
            saved = _save_negative_clip(cfg, name, clip, f"neg_{seg}_{float(us[0]['start_s']):.0f}")
            source = str(saved) if saved else None
        db.add_speaker_negative(sid, emb, source, round(float(speech_s), 2), utterance_id=int(us[0]["id"]), segment_id=seg)
        negatives_added += 1

    ids = [int(r["id"]) for r in rows]
    released = db.reject_utterances(sid, ids, unknown)
    days = sorted({str(r["day"]) for r in rows})
    for day in days:
        try:
            write_day_markdown(db, cfg, day)
        except Exception as e:  # noqa: BLE001 - the DB is the source of truth
            log.warning("cannot write markdown for %s: %s", day, e)
    segments = sorted({k[0] for k in groups})
    relabeled = 0
    if relabel and engine is not None:
        proc = Processor(cfg, db, engine=engine)
        for seg in segments:
            row = seg_rows.get(seg)
            if row is None or row["status"] != "done" or not Path(row["path"]).exists():
                continue
            try:
                relabeled += proc.relabel_segment(seg)
            except Exception as e:  # noqa: BLE001 - the veto is in; re-identification is a bonus
                log.warning("reject: relabel of segment #%s failed: %s", seg, e)
    log.info("reject %s: %d sentence(s) vetoed (%d released), %d negative sample(s), %d relabeled",
             name, len(ids), released, negatives_added, relabeled)
    out: dict[str, Any] = {"speaker_id": sid, "name": name, "rejected": len(ids), "released": released,
                           "utterance_ids": sorted(set(ids)),
                           "negatives_added": negatives_added, "relabeled_changed": relabeled,
                           "segments": segments, "days": days}
    if warning:
        out["warning"] = warning
        out["warning_score"] = round(float(warn_score or 0.0), 3)
    return out


def reject_voice(db: DB, cfg: Config, engine: Any, speaker_id: int, utterance_ids: list[int],
                 whole_cluster: bool = False) -> dict[str, Any]:
    """"These sentences are NOT speaker_id": veto the speaker for them (utterance_rejections), release the ones
    attributed to that speaker, take negative samples from their audio, then re-identify the affected segments.

    With `whole_cluster` the sentences of the same segment and cluster that are also attributed to the speaker
    automatically (not hand-labelled as that speaker) are included. Negative samples: the sentences grouped by
    (segment, cluster) and stitched together, at most NEGATIVE_MAX_CLIPS clips of NEGATIVE_MAX_S each, only with
    at least NEGATIVE_MIN_SPEECH_S of speech and a readable recording (otherwise veto only). A clip that
    resembles the speaker's reference voice >= NEGATIVE_WARN_SIMILARITY is not stored (veto only) and the result
    carries a `warning`. Raises LookupError (unknown speaker / sentences) and ValueError (bad input).

    Returns {speaker_id, name, rejected, released, utterance_ids (every vetoed sentence, whole_cluster ones included),
    negatives_added, relabeled_changed, segments, days, warning?}."""
    sp = _speaker_row(db, speaker_id)
    try:
        ids = sorted({int(i) for i in (utterance_ids or [])})
    except (TypeError, ValueError):
        raise ValueError("utterance_ids 应为句子 id 列表")
    if not ids:
        raise ValueError("没有要否决的句子")
    cols = "id, segment_id, cluster_label, start_s, end_s, day, speaker_id, speaker_source"
    rows = list(db.query(f"SELECT {cols} FROM utterances WHERE id IN ({','.join('?' * len(ids))})", ids))
    if not rows:
        raise LookupError("没有这条记录")
    if whole_cluster:
        have = {int(r["id"]) for r in rows}
        for seg, cl in sorted({(int(r["segment_id"]), int(r["cluster_label"])) for r in rows if r["cluster_label"] is not None}):
            for r in db.query(f"SELECT {cols} FROM utterances WHERE segment_id=? AND cluster_label=? AND speaker_id=?"
                              " AND (speaker_source IS NULL OR speaker_source<>'user')", (seg, cl, int(sp["id"]))):
                if int(r["id"]) not in have:
                    have.add(int(r["id"]))
                    rows.append(r)
    return _reject_rows(db, cfg, engine, sp, rows)


def reject_voice_group(db: DB, cfg: Config, engine: Any, members: list[Any], speaker_id: int) -> dict[str, Any]:
    """「不是 X」 on a voice group: veto speaker_id for every unidentified sentence of the members and take negative
    samples per member (as reject_voice). Raises ValueError (bad members / no unidentified sentence) and LookupError."""
    sp = _speaker_row(db, speaker_id)
    mem = _parse_members(members)
    rows: list[Any] = []
    for seg, cl in mem:
        rows.extend(db.query("SELECT id, segment_id, cluster_label, start_s, end_s, day, speaker_id, speaker_source FROM utterances"
                             " WHERE segment_id=? AND cluster_label=? AND speaker_id IS NULL ORDER BY start_s, id", (seg, cl)))
    if not rows:
        raise ValueError("这组声音里没有未识别的句子")
    return _reject_rows(db, cfg, engine, sp, rows)


__all__ = ["RELATION_CHOICES", "group_unknown_voices", "backfill_cluster_embeddings", "name_voice", "cluster_members",
           "missing_segments", "unknown_utterance_count", "reject_voice", "reject_voice_group"]
