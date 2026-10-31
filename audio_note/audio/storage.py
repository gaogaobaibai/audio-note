"""Segment files on disk + housekeeping.

Layout: data/audio/YYYY-MM-DD/HHMMSS_mmm.opus   (16 kHz mono)
While a live segment is being captured it is streamed to  ..._partial.wav  so a crash
loses nothing; on close it is transcoded to Opus and the wav removed. recover_partials()
finishes any leftovers at startup.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import soundfile as sf

from ..config import Config
from ..db import DB
from .vad import SAMPLE_RATE, SileroVAD, speech_seconds

log = logging.getLogger(__name__)

RATIO_MIN_DURATION_S = 10.0   # min_speech_ratio only applies to segments at least this long (short replies are fine)


def opus_compression_level(bitrate_bps: int) -> float:
    """libsndfile maps compression_level (0..1) to an Opus VBR bitrate roughly linearly:
    measured on speech  0.5->132k, 0.7->82k, 0.85->44k, 0.95->19k  => kbps ~= 257 - 250*level."""
    kbps = bitrate_bps / 1000
    return float(min(0.99, max(0.0, (257 - kbps) / 250)))


def segment_path(cfg: Config, started_at: datetime, ext: str) -> Path:
    d = cfg.audio_dir / started_at.strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    return d / (started_at.strftime("%H%M%S_") + f"{started_at.microsecond // 1000:03d}" + ext)


def unique_segment_path(cfg: Config, started_at: datetime, ext: str, db: DB | None = None) -> Path:
    """segment_path() that is neither on disk nor registered in `segments` (appends _1, _2, ...).

    Two imports with the same --start, a re-import after cleanup expired the earlier files, or two
    phone recordings carrying the same timestamp must never overwrite each other's audio or trip
    the UNIQUE(path) constraint half-way through an import.
    """
    base = segment_path(cfg, started_at, ext)
    stem = base.name[: -len(ext)] if ext and base.name.endswith(ext) else base.stem
    cand = base
    k = 0
    while cand.exists() or (db is not None and db.one("SELECT id FROM segments WHERE path=?", (str(cand),)) is not None):
        k += 1
        cand = base.parent / f"{stem}_{k}{ext}"
    return cand


def write_segment_file(cfg: Config, audio: np.ndarray, started_at: datetime, db: DB | None = None) -> Path:
    """Encode `audio` to a fresh file named after `started_at`; never overwrites an existing segment."""
    codec = cfg.audio.codec
    if codec == "opus":
        p = unique_segment_path(cfg, started_at, ".opus", db)
        sf.write(p, audio, SAMPLE_RATE, format="OGG", subtype="OPUS",
                 compression_level=opus_compression_level(int(cfg.audio.opus_bitrate)))
    elif codec == "flac":
        p = unique_segment_path(cfg, started_at, ".flac", db)
        sf.write(p, audio, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    else:
        p = unique_segment_path(cfg, started_at, ".wav", db)
        sf.write(p, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return p


def read_segment(path: Path | str) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        n_out = int(round(len(audio) * SAMPLE_RATE / sr))
        audio = np.interp(np.arange(n_out) / SAMPLE_RATE, np.arange(len(audio)) / sr, audio).astype(np.float32)
    return audio


class PartialWriter:
    """Incremental WAV writer for the segment currently being captured."""

    def __init__(self, cfg: Config, started_at: datetime):
        self.cfg = cfg
        self.started_at = started_at
        self.path = segment_path(cfg, started_at, "_partial.wav")
        self._f = sf.SoundFile(str(self.path), mode="w", samplerate=SAMPLE_RATE, channels=1, subtype="PCM_16", format="WAV")
        self.frames = 0

    def write(self, block: np.ndarray) -> None:
        self._f.write(block)
        self.frames += len(block)

    def close(self) -> None:
        self._f.close()

    def abandon(self) -> None:
        self.close()
        try:
            self.path.unlink()
        except OSError:
            pass


def finalize_segment(cfg: Config, db: DB, audio: np.ndarray, started_at: datetime, speech_s: float,
                     source: str = "mic", origin: str | None = None, partial: Path | None = None) -> int | None:
    """Encode + register one segment. Returns the segment id (or None if dropped as non-speech).

    Dropped when speech_s < storage.min_speech_total_s, or (for segments longer than
    RATIO_MIN_DURATION_S) when speech_s / duration < storage.min_speech_ratio: a gate kept open by
    hysteresis-band noise (TV, HVAC, distant chatter) can otherwise run to max_segment_s with a
    second of real speech in it.
    """
    dur = len(audio) / SAMPLE_RATE
    if speech_s < float(cfg.storage.min_speech_total_s):
        log.debug("dropping %.1fs segment with only %.2fs speech", dur, speech_s)
        if partial and partial.exists():
            partial.unlink()
        return None
    ratio = speech_s / max(dur, 1e-6)
    if dur >= RATIO_MIN_DURATION_S and ratio < float(cfg.storage.min_speech_ratio):
        log.info("dropping %.1fs segment: speech ratio %.3f < %.2f", dur, ratio, float(cfg.storage.min_speech_ratio))
        if partial and partial.exists():
            partial.unlink()
        return None
    path = write_segment_file(cfg, audio, started_at, db)
    seg_id = db.add_segment(path, started_at, dur, SAMPLE_RATE, source=source, origin=origin, speech_s=speech_s)
    if partial and partial.exists():
        partial.unlink()
    log.info("segment #%d %s %.1fs (speech %.1fs, %d KB)", seg_id, path.name, dur, speech_s, path.stat().st_size // 1024)
    return seg_id


def recover_partials(cfg: Config, db: DB, vad: SileroVAD) -> int:
    """Finish *_partial.wav files left behind by a crash."""
    n = 0
    for p in sorted(cfg.audio_dir.glob("*/*_partial.wav")):
        try:
            if time.time() - p.stat().st_mtime < 5:
                continue  # may still be written by another instance
            audio = read_segment(p)
            if len(audio) == 0:
                p.unlink()
                continue
            day = p.parent.name
            hms, ms = p.name.split("_")[0], p.name.split("_")[1]
            started = datetime.strptime(f"{day} {hms}", "%Y-%m-%d %H%M%S") + timedelta(milliseconds=int(ms))
            sp = speech_seconds(audio, vad, float(cfg.audio.vad_threshold))
            if finalize_segment(cfg, db, audio, started, sp, partial=p) is not None:
                n += 1
            elif p.exists():
                p.unlink()
        except Exception as e:  # noqa: BLE001
            log.exception("recover %s failed: %s", p, e)
    if n:
        log.info("recovered %d partial segments", n)
    return n


def dir_size_bytes(d: Path) -> int:
    total = 0
    for root, _, files in os.walk(d):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _delete_file(path: str) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def cleanup(cfg: Config, db: DB, vad: SileroVAD | None = None) -> dict[str, int]:
    """Housekeeping pass:
    1. VAD re-check of segments without speech_s, plus 'new' segments whose stored speech ratio is
       already below min_speech_ratio -> delete those with (almost) no speech.
    2. Delete audio of transcribed (or permanently failed) segments older than keep_audio_days
       (transcripts are kept).
    3. If audio dir exceeds max_disk_gb, delete oldest transcribed audio first.
    4. Remove empty day folders.
    """
    stats = {"rechecked": 0, "no_speech": 0, "expired": 0, "disk_cap": 0}
    st = cfg.storage

    # 1. recheck
    if vad is not None:
        rows = db.query(
            "SELECT * FROM segments WHERE (speech_s IS NULL AND status IN ('new','done'))"
            " OR (status='new' AND speech_s IS NOT NULL AND duration_s >= ? AND speech_s / MAX(duration_s, 1e-6) < ?)",
            (RATIO_MIN_DURATION_S, float(st.min_speech_ratio)),
        )
        for row in rows:
            try:
                audio = read_segment(row["path"])
            except Exception as e:  # noqa: BLE001
                log.warning("cannot read %s: %s", row["path"], e)
                db.set_segment_status(row["id"], "failed", f"unreadable: {e}")
                continue
            sp = speech_seconds(audio, vad, float(cfg.audio.vad_threshold))
            db.execute("UPDATE segments SET speech_s=? WHERE id=?", (sp, row["id"]))
            stats["rechecked"] += 1
            ratio = sp / max(1e-6, row["duration_s"])
            low_ratio = ratio < float(st.min_speech_ratio) and float(row["duration_s"]) >= RATIO_MIN_DURATION_S
            if row["status"] == "new" and (sp < float(st.min_speech_total_s) or low_ratio):
                _delete_file(row["path"])
                db.set_segment_status(row["id"], "no_speech")
                stats["no_speech"] += 1

    # 2. expiry
    keep_days = int(st.keep_audio_days or 0)
    if keep_days > 0:
        cutoff = (datetime.now() - timedelta(days=keep_days)).isoformat(timespec="seconds")
        for row in db.query("SELECT id, path FROM segments WHERE status IN ('done','failed') AND started_at < ?", (cutoff,)):
            _delete_file(row["path"])
            db.set_segment_status(row["id"], "deleted")
            stats["expired"] += 1

    # 3. disk cap
    cap = float(st.max_disk_gb or 0) * (1024 ** 3)
    if cap > 0:
        used = dir_size_bytes(cfg.audio_dir)
        if used > cap:
            for row in db.query("SELECT id, path FROM segments WHERE status='done' ORDER BY started_at"):
                try:
                    sz = os.path.getsize(row["path"])
                except OSError:
                    sz = 0
                _delete_file(row["path"])
                db.set_segment_status(row["id"], "deleted")
                stats["disk_cap"] += 1
                used -= sz
                if used <= cap * 0.9:
                    break

    # 4. empty folders
    for d in cfg.audio_dir.glob("*"):
        if d.is_dir() and not any(d.iterdir()):
            shutil.rmtree(d, ignore_errors=True)
    log.info("cleanup: %s", stats)
    return stats
