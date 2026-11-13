"""Re-identification of stored utterances, without ASR: Processor.relabel_segment / Processor.relabel.

The functions take the Processor as `proc` (they use its db / cfg / stop_event and the voice-stage helpers
`_identify`, `_self_id`); Processor keeps the public methods and delegates here. Names that were globals of the old
pipeline.py are reached through `F` (see chunking.py).

relabel reads the audio again, so it is the way to (re)compute the voice vectors of a segment; the decision itself
is the scoring layer's (speakers v5): with an engine that decides through scoring.py, relabel_segment stores every
sentence's own vector (utterance_voice), its "maybe X" and its decision details, and then lets scoring.rescore
look at the segment together with its neighbours in time - exactly what the first pass does. `scoring.rescore`
alone re-decides history from the stored vectors without touching any audio.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..audio.vad import SAMPLE_RATE
from .chunking import F, log, stage_timer
from .speakers import Turn


def _row_get(r: Any, key: str) -> Any:
    try:
        return r[key]
    except (IndexError, KeyError):
        return None


def relabel_segment(proc: Any, seg_id: int) -> int:
    """Re-run speaker identification on a 'done' segment using the stored utterances as turns
    (no diarization, no ASR, no billing) and update speaker_id/name/score + the day Markdown.
    Utterances the user labelled by hand (speaker_source='user' with a speaker_id; a deleted speaker
    leaves speaker_id NULL and the row is automatic again) are left untouched. Returns the number of
    utterances whose speaker changed. Used after `speakers enroll/label` so history picks up a newly
    enrolled voice; the segment's cluster_embeddings are refreshed on the way.

    A sentence is never identified as a speaker the user vetoed for it (utterance_rejections), and a mostly
    system-audio sentence (segments.channel_rms, speakers.system_turn_share) never as the self speaker; the
    engine tries the next candidate, and the write-back enforces it (unknown) whatever the engine did.
    utterances.system_share is (re)computed when the segment has channel data.

    The write is conditional: between reading the rows and the UPDATE (the embedding takes seconds)
    a request thread may hand-label one of them (name_voice / label); such a row is skipped by the
    WHERE clause and not counted as changed."""
    db, cfg = proc.db, proc.cfg
    seg = db.get_segment(seg_id)
    if seg is None:
        raise ValueError(f"segment {seg_id} not found")
    path = Path(seg["path"])
    if not path.exists():
        raise FileNotFoundError(f"segment file missing (expired?): {path}")
    utts = db.query("SELECT * FROM utterances WHERE segment_id=? ORDER BY start_s, id", (seg_id,))
    if not utts:
        return 0
    audio = F.read_segment(path)
    # loudness: fill in what older rows lack, and keep the faint (far-away) utterances out of identification
    min_db = F.min_speech_dbfs(cfg)
    levels: list[float] = []
    for u in utts:
        lv = u["loudness_db"]
        if lv is None:
            lv = F.speech_dbfs(F._clip(audio, u["start_s"], u["end_s"]))
            db.execute("UPDATE utterances SET loudness_db=? WHERE id=? AND loudness_db IS NULL", (lv, u["id"]))
        levels.append(float(lv))
    loud = [i for i, lv in enumerate(levels) if not F.is_faint(lv, min_db)]
    # vetoes (utterance_rejections) and system-audio sentences (never the self speaker)
    pairs, block_s = F._segment_channels(seg)
    shares: list[float | None] = [None] * len(utts)
    if pairs is not None:
        for i, u in enumerate(utts):
            sh = F.system_share(pairs, block_s, float(u["start_s"]), float(u["end_s"]))
            shares[i] = sh
            if sh is not None and F._row_get(u, "system_share") != sh:
                db.execute("UPDATE utterances SET system_share=? WHERE id=?", (sh, u["id"]))
    rejected = db.rejected_speakers([int(u["id"]) for u in utts]) if hasattr(db, "rejected_speakers") else {}
    me = proc._self_id() if pairs is not None else None
    sys_thr = F.system_turn_share(cfg)
    excludes: list[frozenset[int]] = []
    for i, u in enumerate(utts):
        ex = set(rejected.get(int(u["id"]), ()))
        if me is not None and shares[i] is not None and shares[i] >= sys_thr:
            ex.add(me)
        excludes.append(frozenset(ex))
    turns = [Turn(float(utts[i]["start_s"]), float(utts[i]["end_s"]),
                  int(utts[i]["cluster_label"] if utts[i]["cluster_label"] is not None else i), exclude=excludes[i]) for i in loud]
    if turns:
        with stage_timer(db, "identify", len(audio) / SAMPLE_RATE):
            turns = proc._identify(audio, turns, seg_id, per_utterance=True)
    by_row = dict(zip(loud, turns))
    unknown = cfg.speakers.unknown_label
    changed = 0
    days: set[str] = set()
    # an engine that decides through the scoring layer leaves a Decision (and the vectors it computed) on the turns
    scored = any(getattr(t, "decision", None) is not None for t in turns)
    voice_rows: list[tuple[int, str, int, bytes, float]] = []
    model = None
    if scored:
        from . import scoring

        model = scoring.model_id(cfg)
    for i, u in enumerate(utts):
        if u["speaker_source"] == "user" and u["speaker_id"] is not None:
            continue
        t = by_row.get(i)
        if t is None:
            # too quiet to identify: left as it is, just without a score -- unless it is attributed to a speaker it
            # may not be (a veto, or the self speaker on system audio): then it is released to unknown
            if u["speaker_id"] is not None and int(u["speaker_id"]) in excludes[i]:
                cur = db.execute(
                    "UPDATE utterances SET speaker_id=NULL, speaker_name=?, speaker_score=NULL, speaker_source=NULL"
                    " WHERE id=? AND speaker_id=? AND (speaker_source IS NULL OR speaker_source<>'user')",
                    (unknown, u["id"], int(u["speaker_id"])))
                if int(cur.rowcount or 0) == 1:
                    changed += 1
                    days.add(u["day"])
                continue
            db.execute("UPDATE utterances SET speaker_score=NULL WHERE id=?", (u["id"],))
            continue
        sid_new, name, score_new = t.speaker_id, (t.speaker_name or unknown), t.score
        if sid_new is not None and sid_new in excludes[i]:
            sid_new, name = None, unknown        # the engine ignored the veto: enforce it
        d = getattr(t, "decision", None)
        emb = getattr(t, "emb", None)
        if scored and emb is not None:
            v = np.asarray(emb, dtype=np.float32).ravel()
            n = float(np.linalg.norm(v))
            if v.shape[0] >= 2 and np.isfinite(n) and n > 0:
                voice_rows.append((int(u["id"]), model, int(v.shape[0]), (v / n).astype(np.float32).tobytes(),
                                   round(float(t.duration), 3)))
        if d is not None:
            # "maybe X" and how the sentence was decided travel with it (a guess never goes to speaker_id)
            gid = d.guess_id if (sid_new is None and d.guess_id is not None and d.guess_id not in excludes[i]) else None
            gscore = round(float(d.guess_score), 4) if (gid is not None and d.guess_score is not None) else None
            meta = json.dumps(d.meta() if sid_new == d.speaker_id else {"m": "veto"}, ensure_ascii=False, sort_keys=True)
            cur = db.execute(
                "UPDATE utterances SET speaker_id=?, speaker_name=?, speaker_score=?, speaker_source=NULL,"
                " speaker_guess_id=?, speaker_guess_score=?, speaker_meta=? WHERE id=?"
                " AND (speaker_source IS NULL OR speaker_source<>'user' OR speaker_id IS NULL)"
                " AND NOT EXISTS (SELECT 1 FROM utterance_rejections r WHERE r.utterance_id=utterances.id AND r.speaker_id=?)",
                (sid_new, name, score_new, gid, gscore, meta, u["id"], sid_new))
        else:
            cur = db.execute(
                "UPDATE utterances SET speaker_id=?, speaker_name=?, speaker_score=?, speaker_source=NULL WHERE id=?"
                " AND (speaker_source IS NULL OR speaker_source<>'user' OR speaker_id IS NULL)"
                " AND NOT EXISTS (SELECT 1 FROM utterance_rejections r WHERE r.utterance_id=utterances.id AND r.speaker_id=?)",
                (sid_new, name, score_new, u["id"], sid_new))
        if int(cur.rowcount or 0) != 1:
            continue                        # hand-labelled meanwhile: leave it alone
        if u["speaker_id"] != sid_new or (u["speaker_name"] or unknown) != name:
            changed += 1
            days.add(u["day"])
    if scored:
        changed, days = _refine(proc, seg_id, utts, voice_rows, unknown)
    if changed:
        F.write_days_markdown(db, cfg, days)
    log.info("relabel segment #%d: %d/%d utterances changed", seg_id, changed, len(utts))
    return changed


def _refine(proc: Any, seg_id: int, before: list[Any], voice_rows: list[tuple[int, str, int, bytes, float]],
            unknown: str) -> tuple[int, set[str]]:
    """Second half of a scoring-layer relabel: store the sentences' own vectors, let scoring.rescore decide the
    segment together with its neighbours (tracks), and count what changed against the rows as they were before
    the relabel started. Returns (changed sentences, their days)."""
    from . import scoring

    db, cfg = proc.db, proc.cfg
    if voice_rows:
        with db.transaction() as con:
            for row in voice_rows:
                # the sentence may have been split / deleted while the audio was embedded: its vector goes with it
                con.execute("INSERT OR REPLACE INTO utterance_voice(utterance_id, model, dim, embedding, speech_s)"
                            " SELECT ?,?,?,?,? WHERE EXISTS (SELECT 1 FROM utterances WHERE id=?)", (*row, row[0]))
    try:
        scoring.rescore(db, cfg, segment_ids=[int(seg_id)])
    except Exception as e:  # noqa: BLE001 - the per-sentence decisions are in; pooling with the neighbours is a bonus
        log.warning("relabel segment #%d: neighbours not considered: %s", seg_id, e)
    now = {int(r["id"]): r for r in db.query("SELECT id, speaker_id, speaker_name, day FROM utterances WHERE segment_id=?",
                                             (int(seg_id),))}
    changed = 0
    days: set[str] = set()
    for u in before:
        r = now.get(int(u["id"]))
        if r is None or (u["speaker_source"] == "user" and u["speaker_id"] is not None):
            continue
        if u["speaker_id"] != r["speaker_id"] or (u["speaker_name"] or unknown) != (r["speaker_name"] or unknown):
            changed += 1
            days.add(str(r["day"]))
    return changed, days


def relabel(proc: Any, day: str | None = None, segment_id: int | None = None) -> dict[str, int]:
    """relabel_segment over every 'done' segment whose audio still exists (optionally one day / one id)."""
    db = proc.db
    stats = {"segments": 0, "changed": 0, "skipped": 0}
    if segment_id is not None:
        rows = [db.get_segment(int(segment_id))]
    elif day:
        rows = db.query("SELECT * FROM segments WHERE status='done' AND started_at >= ? AND started_at < ? ORDER BY started_at",
                        (f"{day}T00:00:00", f"{day}T23:59:59.999"))
    else:
        rows = db.query("SELECT * FROM segments WHERE status='done' ORDER BY started_at")
    for row in rows:
        if row is None:
            continue
        if proc.stop_event is not None and proc.stop_event.is_set():
            break
        if not Path(row["path"]).exists():
            stats["skipped"] += 1
            continue
        try:
            stats["changed"] += proc.relabel_segment(int(row["id"]))
            stats["segments"] += 1
        except Exception as e:  # noqa: BLE001
            stats["skipped"] += 1
            log.warning("relabel segment #%d failed: %s", row["id"], e)
    log.info("relabel: %s", stats)
    return stats


__all__ = ["relabel", "relabel_segment"]
