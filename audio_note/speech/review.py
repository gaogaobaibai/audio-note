"""Review queue (speakers v4, docs/SPEC_speakers_v4_negatives.md section B): the sentences automatically attributed to
a speaker, least certain first, so the user can confirm (「是 TA」) or reject (「不是 TA」, voices.reject_voice) them
by ear at low cost.

    review_queue(db, cfg, speaker_id, limit=20, offset=0) -> [item, ...]
    review_total(db, cfg, speaker_id)                      -> number of sentences the queue holds
    confirm_utterance(db, cfg, engine, utterance_id, speaker_id, whole_cluster=False) -> {...}

Item: {utterance_id, day, time, text, seconds, score, loudness_db, system_share, audio}. Queue = speaker_source IS NULL,
audio file present, at least REVIEW_MIN_S long, not vetoed for the speaker, not below the loudness gate
(audio.min_speech_dbfs: faint / far-away voices the user asked to drop; rows without a stored level stay); ordered by
speaker_score ascending (NULL = least certain, first), then by duration ascending, then id.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..audio.loudness import min_speech_dbfs
from ..config import Config
from ..db import DB

log = logging.getLogger(__name__)

REVIEW_MIN_S = 0.8


def _candidates(db: DB, cfg: Config, speaker_id: int) -> list[Any]:
    min_db = min_speech_dbfs(cfg)
    loud, loud_params = ("", []) if min_db is None else (" AND (u.loudness_db IS NULL OR u.loudness_db >= ?)", [float(min_db)])
    rows = db.query(
        "SELECT u.id, u.segment_id, u.day, u.abs_start, u.start_s, u.end_s, u.text, u.speaker_score, u.loudness_db, u.system_share,"
        " s.path AS seg_path FROM utterances u JOIN segments s ON s.id = u.segment_id"
        " WHERE u.speaker_id=? AND u.speaker_source IS NULL AND (u.end_s - u.start_s) >= ?"
        " AND NOT EXISTS (SELECT 1 FROM utterance_rejections r WHERE r.utterance_id=u.id AND r.speaker_id=u.speaker_id)"
        f"{loud}"
        " ORDER BY (u.speaker_score IS NOT NULL), u.speaker_score, (u.end_s - u.start_s), u.id",
        (int(speaker_id), REVIEW_MIN_S, *loud_params))
    exists: dict[int, bool] = {}
    out = []
    for r in rows:
        seg = int(r["segment_id"])
        if seg not in exists:
            exists[seg] = bool(r["seg_path"]) and Path(r["seg_path"]).exists()
        if exists[seg]:
            out.append(r)
    return out


def _item(r: Any) -> dict[str, Any]:
    return {"utterance_id": int(r["id"]), "day": r["day"], "time": str(r["abs_start"] or "")[11:19],
            "text": (r["text"] or "").strip(), "seconds": round(float(r["end_s"]) - float(r["start_s"]), 1),
            "score": round(float(r["speaker_score"]), 3) if r["speaker_score"] is not None else None,
            "loudness_db": round(float(r["loudness_db"]), 1) if r["loudness_db"] is not None else None,
            "system_share": round(float(r["system_share"]), 3) if r["system_share"] is not None else None,
            "audio": True}


def review_queue(db: DB, cfg: Config, speaker_id: int, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
    """One page of the speaker's review queue (see the module docstring)."""
    limit = max(0, int(limit))
    offset = max(0, int(offset))
    return [_item(r) for r in _candidates(db, cfg, speaker_id)[offset: offset + limit]]


def review_total(db: DB, cfg: Config, speaker_id: int) -> int:
    return len(_candidates(db, cfg, speaker_id))


def confirm_utterance(db: DB, cfg: Config, engine: Any, utterance_id: int, speaker_id: int,
                      whole_cluster: bool = False) -> dict[str, Any]:
    """「是 TA」: the sentence belongs to speaker_id (speaker_source='user', relabel leaves it alone; a veto of the
    speaker for it is lifted). With `whole_cluster` the sentences of the same segment and cluster automatically
    attributed to the speaker are confirmed too. When the sentence holds at least speakers.min_label_sample_s of
    speech, its audio is still there and it resembles the person's samples, it also becomes a voice sample (origin
    'label'); otherwise `sample_added` is False (never an error). Raises LookupError for an unknown sentence /
    speaker.

    Returns {utterance_id, speaker_id, name, confirmed, sample_added, sample_note, segment_id, days}."""
    from .pipeline import write_day_markdown
    from .speakers import SampleMismatch, TooShortSample, min_label_sample_s

    try:
        sid = int(speaker_id)
        uid = int(utterance_id)
    except (TypeError, ValueError):
        raise ValueError("speaker_id 应为说话人 id")
    sp = db.one("SELECT id, name FROM speakers WHERE id=?", (sid,))
    if sp is None:
        raise LookupError("没有这个说话人")
    u = db.one("SELECT id, segment_id, cluster_label, start_s, end_s, day, speaker_id FROM utterances WHERE id=?", (uid,))
    if u is None:
        raise LookupError("没有这条记录")
    name = str(sp["name"])
    seg_id = int(u["segment_id"])
    changed_owner = u["speaker_id"] != sid
    db.execute("DELETE FROM utterance_rejections WHERE utterance_id=? AND speaker_id=?", (uid, sid))
    if changed_owner:
        db.execute("UPDATE utterances SET speaker_id=?, speaker_name=?, speaker_score=NULL, speaker_source='user' WHERE id=?",
                   (sid, name, uid))
    else:
        db.execute("UPDATE utterances SET speaker_name=?, speaker_source='user' WHERE id=?", (name, uid))
    confirmed = 1
    if whole_cluster and u["cluster_label"] is not None:
        cur = db.execute("UPDATE utterances SET speaker_source='user' WHERE segment_id=? AND cluster_label=? AND speaker_id=?"
                         " AND speaker_source IS NULL AND id<>?", (seg_id, int(u["cluster_label"]), sid, uid))
        confirmed += int(cur.rowcount or 0)

    sample_added = False
    sample_note: str | None = None
    a, b = float(u["start_s"]), float(u["end_s"])
    min_s = min_label_sample_s(cfg)
    seg = db.get_segment(seg_id)
    if b - a < min_s:
        sample_note = f"这句只有 {b - a:.1f}s（至少 {min_s:g}s 才取声纹样本）"
    elif seg is None or not Path(seg["path"]).exists():
        sample_note = "这段音频已不存在，没有取声纹样本"
    elif engine is None:
        sample_note = "声纹模型不可用，没有取声纹样本"
    else:
        try:
            from ..audio.storage import read_segment
            from ..audio.vad import SAMPLE_RATE

            audio = read_segment(seg["path"])
            clip = np.asarray(audio[int(a * SAMPLE_RATE): int(b * SAMPLE_RATE)], dtype=np.float32)
            engine.enroll_from_audio(db, name, clip, source=f"{seg['path']}#{a:.1f}-{b:.1f}", min_speech_s=min_s, origin="label")
            sample_added = True
        except (SampleMismatch, TooShortSample) as e:
            sample_note = str(e)
        except ValueError as e:
            sample_note = str(e)
        except Exception as e:  # noqa: BLE001 - the confirmation stands without a sample
            log.warning("confirm #%d: no sample: %s", uid, e)
            sample_note = f"取声纹样本失败：{e}"
    days = sorted({str(r["day"]) for r in db.query("SELECT DISTINCT day FROM utterances WHERE segment_id=? AND speaker_id=?", (seg_id, sid))}
                  | {str(u["day"])})
    if changed_owner:
        for day in days:
            try:
                write_day_markdown(db, cfg, day)
            except Exception as e:  # noqa: BLE001 - the DB is the source of truth
                log.warning("cannot write markdown for %s: %s", day, e)
    log.info("confirm #%d as %s (%d sentence(s)), sample added: %s", uid, name, confirmed, sample_added)
    return {"utterance_id": uid, "speaker_id": sid, "name": name, "confirmed": confirmed, "sample_added": sample_added,
            "sample_note": sample_note, "segment_id": seg_id, "days": days}


__all__ = ["review_queue", "review_total", "confirm_utterance", "REVIEW_MIN_S"]
