"""JSON shapes shared by several routers. `utt_dict` is THE one place an utterance row becomes JSON.

    utt_dict(row, unknown_label, audio_ok=None, *, speaker_names=None, marks=None) -> dict
    utterance_extras(db, rows) -> (speaker_names, marks_by_id)     # what utt_dict needs beyond the row itself
    loudness_of(row) / system_share_of(row) / hint(secret) / wav_bytes(audio)

Utterance JSON (existing keys never change, SPEC v1 section 8 only adds):

    id, segment_id, day, abs_start, time (HH:MM:SS), start_s, end_s, speaker (name or the unknown label),
    speaker_id, score (3 dp | null), text, cluster, loudness_db (1 dp | null), system_share (3 dp | null),
    text_raw?      only when it differs from text
    audio?         bool, only when the caller knows whether the recording still exists
    session_id     int | null     } present when the row has the column (schema v2); a row from an
    text_source    str | null     } older database or a narrow SELECT simply lacks these keys
    flags          object         }   (flags: the JSON column parsed, {} when empty / unreadable)
    speaker_guess? {id, name, score}   "可能是 X": only for a sentence without a speaker that has a guess
    marks?         list           only when the caller passes them (speech.revisions.marks, package B2)
"""
from __future__ import annotations

import io
import json
import logging
from typing import Any, Iterable, Mapping

log = logging.getLogger(__name__)

_MISSING = object()


def col(r: Any, name: str, default: Any = _MISSING) -> Any:
    """`r[name]`, or `default` when the row (sqlite3.Row or dict) has no such column."""
    try:
        return r[name]
    except (IndexError, KeyError):
        return default


def loudness_of(r: Any) -> float | None:
    """utterances.loudness_db of a row, rounded for display (None when not computed / column absent)."""
    v = col(r, "loudness_db", None)
    return round(float(v), 1) if v is not None else None


def system_share_of(r: Any) -> float | None:
    """utterances.system_share of a row (share of system-audio energy, 0..1; None when unknown / column absent)."""
    v = col(r, "system_share", None)
    return round(float(v), 3) if v is not None else None


def hint(secret: str | None) -> str:
    """Masked form of a secret for the settings page: `abc…wxyz` (never the secret itself)."""
    s = str(secret or "")
    if not s:
        return ""
    return s[:3] + "…" + s[-4:] if len(s) > 10 else "…" + s[-2:]


def parse_flags(v: Any) -> dict[str, Any]:
    """utterances.flags (JSON object text) -> dict; {} for NULL, broken JSON or anything that is not an object."""
    if isinstance(v, dict):
        return dict(v)
    if not v:
        return {}
    try:
        d = json.loads(v)
    except (TypeError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def utt_dict(r: Any, unknown: str, audio_ok: bool | None = None, *, speaker_names: Mapping[int, str] | None = None,
             marks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """One utterance row as JSON (see the module docstring). `speaker_names` (id -> name) resolves the name
    of a guessed speaker; `marks` are the UI marks of this sentence when the caller computed them."""
    text = r["text"] or ""
    raw = r["text_raw"] or ""
    d: dict[str, Any] = {
        "id": int(r["id"]), "segment_id": int(r["segment_id"]), "day": r["day"], "abs_start": r["abs_start"],
        "time": (r["abs_start"] or "")[11:19], "start_s": float(r["start_s"]), "end_s": float(r["end_s"]),
        "speaker": r["speaker_name"] or unknown, "speaker_id": r["speaker_id"],
        "score": round(float(r["speaker_score"]), 3) if r["speaker_score"] is not None else None,
        "text": text, "cluster": r["cluster_label"], "loudness_db": loudness_of(r),
        "system_share": system_share_of(r),
    }
    if raw and raw != text:
        d["text_raw"] = raw
    if audio_ok is not None:
        d["audio"] = audio_ok
    # --- schema v2 columns: emitted only when the row carries them
    v = col(r, "session_id")
    if v is not _MISSING:
        d["session_id"] = int(v) if v is not None else None
    v = col(r, "text_source")
    if v is not _MISSING:
        d["text_source"] = str(v) if v else None
    v = col(r, "flags")
    if v is not _MISSING:
        d["flags"] = parse_flags(v)
    gid = col(r, "speaker_guess_id", None)
    if gid is not None and r["speaker_id"] is None:          # a guess only matters while nobody is assigned
        score = col(r, "speaker_guess_score", None)
        d["speaker_guess"] = {"id": int(gid), "name": (speaker_names or {}).get(int(gid)),
                              "score": round(float(score), 3) if score is not None else None}
    # --- hook for package B2: per-sentence UI marks (fixed / uncertain spans)
    if marks is not None:
        d["marks"] = marks
    return d


def speaker_names(db: Any) -> dict[int, str]:
    """id -> name of every enrolled speaker (one small query)."""
    return {int(r["id"]): r["name"] for r in db.query("SELECT id, name FROM speakers")}


def marks_for(db: Any, rows: Iterable[Any]) -> dict[int, list[dict[str, Any]]] | None:
    """HOOK (package B2): UI marks for a batch of utterance rows, `{utterance_id: [mark, ...]}`.

    Delegates to `audio_note.speech.revisions.marks(db, rows)` (SPEC v1 section 6.3) once that module exists;
    until then - and whenever it fails - returns None, and utt_dict emits no "marks" key at all. Marks are
    decoration: they must never break reading a transcript.
    """
    try:
        from ..speech import revisions
    except ImportError:
        return None
    fn = getattr(revisions, "marks", None)
    if fn is None:
        return None
    try:
        out = fn(db, list(rows))
    except Exception as e:  # noqa: BLE001
        log.warning("utterance marks unavailable: %s: %s", type(e).__name__, e)
        return None
    return out if isinstance(out, dict) else None


def utterance_extras(db: Any, rows: Iterable[Any]) -> tuple[dict[int, str], dict[int, list[dict[str, Any]]] | None]:
    """(speaker_names, marks) for a batch of rows about to go through utt_dict. The speakers table is only
    read when some row actually carries a guess; marks is None while package B2's module is absent."""
    rows = list(rows)
    names: dict[int, str] = {}
    if any(col(r, "speaker_guess_id", None) is not None for r in rows):
        names = speaker_names(db)
    return names, marks_for(db, rows)


def utt_dicts(db: Any, rows: Iterable[Any], unknown: str, audio_ok: Any = None) -> list[dict[str, Any]]:
    """utt_dict for a batch: `audio_ok` is None, or a callable segment_id -> bool."""
    rows = list(rows)
    names, marks = utterance_extras(db, rows)
    out = []
    for r in rows:
        ok = audio_ok(int(r["segment_id"])) if audio_ok is not None else None
        m = None if marks is None else list(marks.get(int(r["id"])) or [])
        out.append(utt_dict(r, unknown, ok, speaker_names=names, marks=m))
    return out


def wav_bytes(audio: Any) -> bytes:
    """float32 samples -> WAV (PCM 16-bit, the pipeline's sample rate)."""
    import numpy as np
    import soundfile as sf

    from ..audio.vad import SAMPLE_RATE

    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()
