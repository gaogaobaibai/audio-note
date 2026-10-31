"""Import existing recordings: decode -> VAD-gate into speech segments -> Opus files + DB rows."""
from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import soundfile as sf

from ..config import Config
from ..db import DB
from .recorder import gate_config_from
from .storage import finalize_segment, read_segment
from .vad import SAMPLE_RATE, SileroVAD, segment_audio

log = logging.getLogger(__name__)


def decode_any(path: Path) -> np.ndarray:
    """Return float32 mono 16 kHz. libsndfile handles wav/flac/ogg/opus/mp3; anything else goes through ffmpeg."""
    try:
        return read_segment(path)
    except Exception as e:  # noqa: BLE001
        log.info("libsndfile cannot read %s (%s); using ffmpeg", path.name, e)
    import imageio_ffmpeg  # bundled ffmpeg binary, no system install needed

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "decoded.wav"
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "wav", str(out)]
        subprocess.run(cmd, check=True)
        audio, sr = sf.read(str(out), dtype="float32")
        return audio if audio.ndim == 1 else audio.mean(axis=1)


_NAME_TIME_RE = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})[ _T-]?(\d{2})[-_:]?(\d{2})(?:[-_:]?(\d{2}))?")


def parse_name_time(path: Path) -> datetime | None:
    """Start time encoded in the file name (YYYYMMDD[_-]HHMM[SS], YYYY-MM-DD[ _T]HH-MM[-SS], ...),
    or None when absent or not a valid date. Seconds are optional (voice recorders often omit them)."""
    m = _NAME_TIME_RE.search(Path(path).stem)
    if not m:
        return None
    y, mo, d, h, mi, se = (int(g) if g is not None else 0 for g in m.groups())
    try:
        return datetime(y, mo, d, h, mi, se)
    except ValueError:
        return None


def guess_start_time(path: Path) -> datetime:
    """Time from the file name (parse_name_time) else the file mtime (usually the END of a recording:
    import_file subtracts the duration in that case, see _name_has_time)."""
    t = parse_name_time(path)
    return t if t is not None else datetime.fromtimestamp(path.stat().st_mtime)


def import_file(cfg: Config, db: DB, path: Path, start_time: datetime | None = None,
                vad: SileroVAD | None = None) -> list[int]:
    path = Path(path)
    cfg.ensure_dirs()
    vad = vad or SileroVAD(cfg.model_path("vad_model"), threads=2)
    audio = decode_any(path)
    dur = len(audio) / SAMPLE_RATE
    if start_time is None:
        start_time = guess_start_time(path)
        # mtime is usually the END of a recording
        if not _name_has_time(path):
            start_time = start_time - timedelta(seconds=dur)
    segs = segment_audio(audio, vad, gate_config_from(cfg))
    ids: list[int] = []
    for seg in segs:
        started = start_time + timedelta(seconds=seg.start_sample / SAMPLE_RATE)
        sid = finalize_segment(cfg, db, seg.audio, started, seg.speech_s, source="import", origin=str(path))
        if sid is not None:
            ids.append(sid)
    log.info("imported %s: %.0fs -> %d speech segments", path.name, dur, len(ids))
    return ids


def _name_has_time(path: Path) -> bool:
    """True exactly when guess_start_time() took the time from the file name (same parser, so an
    invalid date such as month 13 counts as 'no time in name' and the mtime path is used)."""
    return parse_name_time(path) is not None
