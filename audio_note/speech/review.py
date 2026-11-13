"""Review queue (speakers v4 section B, extended by v5 / SPEC_v1 5.1): the sentences worth a second look for one
speaker, so the user can settle them by ear at low cost - 「是 TA」 (confirm_utterance) or 「不是 TA」
(voices.reject_voice).

    review_queue(db, cfg, speaker_id, limit=20, offset=0) -> [item, ...]
    review_total(db, cfg, speaker_id)                      -> number of sentences the queue holds
    review_counts(db, cfg, speaker_id)                     -> {"guess": n, "sure": m}
    confirm_utterance(db, cfg, engine, utterance_id, speaker_id, whole_cluster=False) -> {...}

The queue asks from both sides:

* kind "guess": sentences nobody is assigned to that MAY be this speaker (utterances.speaker_guess_id: the scoring
  layer was not sure enough to decide). They come first, the most influential first: `influence` is how many
  other "maybe" sentences of this speaker sound like this one (stored voice vectors, cosine >=
  speakers.track_link_cos) - one answer settles them all, because a confirmation is evidence for the speaker's
  centroid (scoring.adapt) and a rejection becomes a negative sample. Then the higher guess score, then the
  longer sentence (easier to judge by ear).
* kind "sure": sentences automatically attributed to the speaker, least certain first (speaker_score ascending,
  NULL first, then the shorter one) - the order of v4.

Item: {utterance_id, day, time, text, seconds, score, loudness_db, system_share, audio, kind, influence,
guess_score}. Both kinds: audio file present, at least REVIEW_MIN_S long, not vetoed for the speaker, not below the
loudness gate (audio.min_speech_dbfs; rows without a stored level stay), never hand-labelled.
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
_COLS = ("u.id, u.segment_id, u.cluster_label, u.day, u.abs_start, u.start_s, u.end_s, u.text, u.speaker_score, u.loudness_db,"
         " u.system_share, u.speaker_guess_score, s.path AS seg_path")


def _with_audio(rows: list[Any]) -> list[Any]:
    exists: dict[int, bool] = {}
    out = []
    for r in rows:
        seg = int(r["segment_id"])
        if seg not in exists:
            exists[seg] = bool(r["seg_path"]) and Path(r["seg_path"]).exists()
        if exists[seg]:
            out.append(r)
    return out


def _sure_rows(db: DB, cfg: Config, speaker_id: int) -> list[Any]:
    min_db = min_speech_dbfs(cfg)
    loud, loud_params = ("", []) if min_db is None else (" AND (u.loudness_db IS NULL OR u.loudness_db >= ?)", [float(min_db)])
    rows = db.query(
        f"SELECT {_COLS} FROM utterances u JOIN segments s ON s.id = u.segment_id"
        " WHERE u.speaker_id=? AND u.speaker_source IS NULL AND (u.end_s - u.start_s) >= ?"
        " AND NOT EXISTS (SELECT 1 FROM utterance_rejections r WHERE r.utterance_id=u.id AND r.speaker_id=u.speaker_id)"
        f"{loud}"
        " ORDER BY (u.speaker_score IS NOT NULL), u.speaker_score, (u.end_s - u.start_s), u.id",
        (int(speaker_id), REVIEW_MIN_S, *loud_params))
    return _with_audio(rows)


def _guess_rows(db: DB, cfg: Config, speaker_id: int) -> list[Any]:
    min_db = min_speech_dbfs(cfg)
    loud, loud_params = ("", []) if min_db is None else (" AND (u.loudness_db IS NULL OR u.loudness_db >= ?)", [float(min_db)])
    try:
        rows = db.query(
            f"SELECT {_COLS} FROM utterances u JOIN segments s ON s.id = u.segment_id"
            " WHERE u.speaker_id IS NULL AND u.speaker_guess_id=? AND (u.end_s - u.start_s) >= ?"
            " AND (u.speaker_source IS NULL OR u.speaker_source<>'user')"
            " AND NOT EXISTS (SELECT 1 FROM utterance_rejections r WHERE r.utterance_id=u.id AND r.speaker_id=?)"
            f"{loud} ORDER BY u.id",
            (int(speaker_id), REVIEW_MIN_S, int(speaker_id), *loud_params))
    except Exception as e:  # noqa: BLE001 - a database from before the 1.0 columns: no guesses yet
        log.debug("no guess rows: %s", e)
        return []
    return _with_audio(rows)


def _vectors(db: DB, rows: list[Any]) -> dict[int, np.ndarray]:
    """{utterance id: unit voice vector}: the sentence's own vector, else the one of its cluster."""
    out: dict[int, np.ndarray] = {}
    ids = [int(r["id"]) for r in rows]
    try:
        for k in range(0, len(ids), 500):
            chunk = ids[k:k + 500]
            for v in db.query(f"SELECT utterance_id, embedding FROM utterance_voice WHERE utterance_id IN ({','.join('?' * len(chunk))})", chunk):
                e = np.frombuffer(v["embedding"], dtype=np.float32)
                n = float(np.linalg.norm(e))
                if n > 0:
                    out[int(v["utterance_id"])] = e / n
    except Exception as e:  # noqa: BLE001
        log.debug("no utterance_voice: %s", e)
    rest = [r for r in rows if int(r["id"]) not in out and r["cluster_label"] is not None]
    if rest:
        segs = sorted({int(r["segment_id"]) for r in rest})
        cl: dict[tuple[int, int], np.ndarray] = {}
        for k in range(0, len(segs), 500):
            for seg, label, emb, _s in db.cluster_embeddings(segs[k:k + 500]):
                n = float(np.linalg.norm(emb))
                if n > 0:
                    cl[(seg, label)] = emb / n
        for r in rest:
            v = cl.get((int(r["segment_id"]), int(r["cluster_label"])))
            if v is not None:
                out[int(r["id"])] = v
    return out


def _influence(db: DB, cfg: Config, rows: list[Any]) -> dict[int, int]:
    """Per sentence: how many OTHER sentences of `rows` sound like it (cosine >= speakers.track_link_cos between
    the stored vectors). Sentences without a vector count 0. Pure numpy."""
    out = {int(r["id"]): 0 for r in rows}
    if len(rows) < 2:
        return out
    vecs = _vectors(db, rows)
    by_dim: dict[int, list[int]] = {}
    for uid, v in vecs.items():
        by_dim.setdefault(int(v.shape[0]), []).append(uid)
    try:
        link = float(cfg.speakers.get("track_link_cos", 0.55))
    except (TypeError, ValueError, AttributeError):
        link = 0.55
    for ids in by_dim.values():
        if len(ids) < 2:
            continue
        M = np.stack([vecs[i] for i in ids])
        near = (M @ M.T) >= link
        counts = near.sum(axis=1) - 1
        for uid, c in zip(ids, counts):
            out[uid] = int(c)
    return out


def _item(r: Any, kind: str, influence: int = 0) -> dict[str, Any]:
    gs = r["speaker_guess_score"] if kind == "guess" else None
    return {"utterance_id": int(r["id"]), "day": r["day"], "time": str(r["abs_start"] or "")[11:19],
            "text": (r["text"] or "").strip(), "seconds": round(float(r["end_s"]) - float(r["start_s"]), 1),
            "score": round(float(r["speaker_score"]), 3) if r["speaker_score"] is not None else None,
            "loudness_db": round(float(r["loudness_db"]), 1) if r["loudness_db"] is not None else None,
            "system_share": round(float(r["system_share"]), 3) if r["system_share"] is not None else None,
            "audio": True, "kind": kind, "influence": int(influence),
            "guess_score": round(float(gs), 3) if gs is not None else None}


def _queue(db: DB, cfg: Config, speaker_id: int) -> list[dict[str, Any]]:
    guesses = _guess_rows(db, cfg, speaker_id)
    infl = _influence(db, cfg, guesses)
    guesses.sort(key=lambda r: (-infl[int(r["id"])], -(float(r["speaker_guess_score"]) if r["speaker_guess_score"] is not None else -1.0),
                                -(float(r["end_s"]) - float(r["start_s"])), int(r["id"])))
    return ([_item(r, "guess", infl[int(r["id"])]) for r in guesses]
            + [_item(r, "sure") for r in _sure_rows(db, cfg, speaker_id)])


def review_queue(db: DB, cfg: Config, speaker_id: int, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
    """One page of the speaker's review queue (see the module docstring)."""
    limit = max(0, int(limit))
    offset = max(0, int(offset))
    return _queue(db, cfg, speaker_id)[offset: offset + limit]


def review_counts(db: DB, cfg: Config, speaker_id: int) -> dict[str, int]:
    return {"guess": len(_guess_rows(db, cfg, speaker_id)), "sure": len(_sure_rows(db, cfg, speaker_id))}


def review_total(db: DB, cfg: Config, speaker_id: int) -> int:
    c = review_counts(db, cfg, speaker_id)
    return c["guess"] + c["sure"]


def confirm_utterance(db: DB, cfg: Config, engine: Any, utterance_id: int, speaker_id: int,
                      whole_cluster: bool = False, events: Any = None) -> dict[str, Any]:
    """「是 TA」: the sentence belongs to speaker_id (speaker_source='user', never re-decided automatically; a veto of
    the speaker for it is lifted, a "maybe" on it is answered). With `whole_cluster` the sentences of the same
    segment and cluster automatically attributed to the speaker - or guessed to be the speaker - are confirmed
    too. When the sentence holds at least speakers.min_label_sample_s of speech, its audio is still there and it
    resembles the person's samples, it also becomes a voice sample (origin 'label'); otherwise `sample_added` is
    False (never an error). Raises LookupError for an unknown sentence / speaker.

    A confirmation is evidence: afterwards the speaker's centroids are rebuilt and the stored decisions brought
    in step (scoring.refresh), so sentences that sound like this one follow. `rescored` says how many changed.

    Returns {utterance_id, speaker_id, name, confirmed, sample_added, sample_note, segment_id, days, rescored}."""
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
    _clear_guess(db, "id=?", (uid,))
    confirmed = 1
    if whole_cluster and u["cluster_label"] is not None:
        cur = db.execute("UPDATE utterances SET speaker_source='user' WHERE segment_id=? AND cluster_label=? AND speaker_id=?"
                         " AND speaker_source IS NULL AND id<>?", (seg_id, int(u["cluster_label"]), sid, uid))
        confirmed += int(cur.rowcount or 0)
        taken = _confirm_guesses(db, seg_id, int(u["cluster_label"]), sid, name, uid)
        confirmed += taken
        changed_owner = changed_owner or taken > 0

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
    rescored = 0
    try:
        from . import scoring

        res = scoring.refresh(db, cfg, events=events, source="review")
        rescored = int((res.get("rescore") or {}).get("changed") or 0)
    except Exception as e:  # noqa: BLE001 - the confirmation is in; the chain step brings the rest in step later
        log.warning("confirm #%d: decisions not refreshed: %s", uid, e)
    log.info("confirm #%d as %s (%d sentence(s)), sample added: %s, %d other sentence(s) re-decided",
             uid, name, confirmed, sample_added, rescored)
    return {"utterance_id": uid, "speaker_id": sid, "name": name, "confirmed": confirmed, "sample_added": sample_added,
            "sample_note": sample_note, "segment_id": seg_id, "days": days, "rescored": rescored}


def _clear_guess(db: DB, where: str, params: tuple[Any, ...]) -> None:
    """An answered sentence carries no "maybe" any more (databases without the 1.0 columns have none to clear)."""
    try:
        db.execute(f"UPDATE utterances SET speaker_guess_id=NULL, speaker_guess_score=NULL WHERE {where}", params)
    except Exception as e:  # noqa: BLE001
        log.debug("no guess columns: %s", e)


def _confirm_guesses(db: DB, seg_id: int, cluster: int, sid: int, name: str, but: int) -> int:
    """whole_cluster for a "maybe": the unassigned sentences of the cluster guessed to be the speaker become theirs
    (never the ones vetoed for the speaker)."""
    try:
        cur = db.execute(
            "UPDATE utterances SET speaker_id=?, speaker_name=?, speaker_score=NULL, speaker_source='user',"
            " speaker_guess_id=NULL, speaker_guess_score=NULL"
            " WHERE segment_id=? AND cluster_label=? AND speaker_id IS NULL AND speaker_guess_id=? AND id<>?"
            " AND (speaker_source IS NULL OR speaker_source<>'user')"
            " AND NOT EXISTS (SELECT 1 FROM utterance_rejections r WHERE r.utterance_id=utterances.id AND r.speaker_id=?)",
            (sid, name, seg_id, cluster, sid, but, sid))
        return int(cur.rowcount or 0)
    except Exception as e:  # noqa: BLE001
        log.debug("no guess columns: %s", e)
        return 0


__all__ = ["review_queue", "review_total", "review_counts", "confirm_utterance", "REVIEW_MIN_S"]
