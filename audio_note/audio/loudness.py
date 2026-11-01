"""Speech loudness (dBFS) and the "faint voice" threshold (docs/SPEC_speakers_v3b.md §B).

Far-away / unclear voices (somebody two metres from the microphone) cannot be identified reliably and are
not worth transcribing; the user asked for a level gate instead. Measured on their data (50 ms frames, RMS of
the louder half of the frames): the user's own near-field turns sit at a median of -26 dBFS (p10 -30), a
friend two metres away at -53 dBFS, the user's occasional turned-away turns at -35 .. -45 dBFS. The default
gate of -40 dBFS keeps the user and drops the distant voices; system audio captured through loopback has an
AGC (peak 0.5) and always clears it.

    speech_dbfs(x)                 loudness of a clip in dBFS (-120.0 for an empty / silent clip)
    min_speech_dbfs(cfg)           the configured gate (audio.min_speech_dbfs) or None when switched off
    is_faint(level, threshold)     True when `level` is below the gate (None level / None gate -> False)
    backfill_loudness(db, ...)     compute the missing utterances.loudness_db of 'done' segments whose audio exists
    loudness_stats(db, threshold)  per-speaker distribution + count below the gate (the `loudness` CLI / UI job)
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from .vad import SAMPLE_RATE

log = logging.getLogger(__name__)

DEFAULT_MIN_SPEECH_DBFS = -40.0
SILENCE_DBFS = -120.0          # reported for an empty or all-zero clip (20*log10(0) is -inf)


def speech_dbfs(x: np.ndarray, frame_s: float = 0.05, top_frac: float = 0.5) -> float:
    """Loudness of a speech clip in dBFS: the clip is cut into `frame_s` frames, the most energetic `top_frac`
    of them are kept (pauses and breaths inside a turn do not drag the level down) and the RMS over those
    frames is converted to dB relative to full scale (1.0). An empty / silent clip gives SILENCE_DBFS."""
    x = np.asarray(x, dtype=np.float32).ravel()
    if x.size == 0:
        return SILENCE_DBFS
    n = max(1, int(round(float(frame_s) * SAMPLE_RATE)))
    frames = x.size // n
    if frames >= 1:
        energy = (x[: frames * n].reshape(frames, n).astype(np.float64) ** 2).mean(axis=1)
    else:
        energy = np.array([float(np.mean(x.astype(np.float64) ** 2))])
    k = max(1, int(math.ceil(len(energy) * float(top_frac))))
    top = np.sort(energy)[::-1][:k]
    rms = float(math.sqrt(max(float(top.mean()), 0.0)))
    if rms <= 0.0:
        return SILENCE_DBFS
    return max(SILENCE_DBFS, 20.0 * math.log10(rms))


def min_speech_dbfs(cfg: Any) -> float | None:
    """audio.min_speech_dbfs as a float, or None when the gate is switched off (null / empty / unparsable)."""
    try:
        v = cfg.audio.get("min_speech_dbfs", DEFAULT_MIN_SPEECH_DBFS)
    except AttributeError:
        return DEFAULT_MIN_SPEECH_DBFS
    if v is None or v == "" or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def is_faint(level: float | None, threshold: float | None) -> bool:
    """True when a stored / measured level lies below the gate. Unknown levels (None) are never faint."""
    if level is None or threshold is None:
        return False
    return float(level) < float(threshold)


# --------------------------------------------------------------------------- existing records
def backfill_loudness(db: Any, day: str | None = None, segment_id: int | None = None, stop_event: Any = None) -> int:
    """Compute utterances.loudness_db where it is NULL, for 'done' segments whose audio file still exists
    (one day, one segment, or everything). Returns the number of utterances filled in."""
    from .storage import read_segment

    where, params = ["s.status='done'"], []
    if segment_id is not None:
        where.append("s.id=?")
        params.append(int(segment_id))
    elif day:
        where.append("s.started_at >= ? AND s.started_at < ?")
        params += [f"{day}T00:00:00", f"{day}T23:59:59.999"]
    segs = db.query(
        f"SELECT DISTINCT s.id, s.path FROM segments s JOIN utterances u ON u.segment_id=s.id"
        f" WHERE {' AND '.join(where)} AND u.loudness_db IS NULL ORDER BY s.started_at", params)
    done = 0
    for s in segs:
        if stop_event is not None and stop_event.is_set():
            break
        p = Path(s["path"])
        if not p.exists():
            continue
        try:
            audio = read_segment(p)
        except Exception as e:  # noqa: BLE001 - one unreadable file must not stop the pass
            log.warning("loudness: cannot read segment #%s (%s): %s", s["id"], p.name, e)
            continue
        for u in db.query("SELECT id, start_s, end_s FROM utterances WHERE segment_id=? AND loudness_db IS NULL", (int(s["id"]),)):
            clip = audio[int(float(u["start_s"]) * SAMPLE_RATE): int(float(u["end_s"]) * SAMPLE_RATE)]
            db.execute("UPDATE utterances SET loudness_db=? WHERE id=?", (speech_dbfs(clip), int(u["id"])))
            done += 1
    if done:
        log.info("loudness: computed loudness_db for %d utterance(s)", done)
    return done


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def loudness_stats(db: Any, threshold: float | None, day: str | None = None, unknown_label: str = "未知") -> dict[str, Any]:
    """Distribution of the stored levels: {"speakers": [{name, n, median, p10, p90, below}], "total", "missing",
    "below", "threshold"}; `below` counts utterances under the gate (0 when the gate is off)."""
    where, params = [], []
    if day:
        where.append("day=?")
        params.append(day)
    sql = "SELECT speaker_name, loudness_db FROM utterances"
    if where:
        sql += " WHERE " + " AND ".join(where)
    by: dict[str, list[float]] = {}
    total = missing = 0
    for r in db.query(sql, params):
        total += 1
        if r["loudness_db"] is None:
            missing += 1
            continue
        by.setdefault(r["speaker_name"] or unknown_label, []).append(float(r["loudness_db"]))
    speakers = []
    below = 0
    for name, vals in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        b = sum(1 for v in vals if is_faint(v, threshold))
        below += b
        speakers.append({"name": name, "n": len(vals), "median": round(_percentile(vals, 50), 1),
                         "p10": round(_percentile(vals, 10), 1), "p90": round(_percentile(vals, 90), 1), "below": b})
    return {"speakers": speakers, "total": total, "missing": missing, "below": below, "threshold": threshold}


__all__ = ["DEFAULT_MIN_SPEECH_DBFS", "SILENCE_DBFS", "speech_dbfs", "min_speech_dbfs", "is_faint", "backfill_loudness",
           "loudness_stats"]
