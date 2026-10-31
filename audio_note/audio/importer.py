"""Import existing recordings: decode -> VAD-gate into speech segments -> Opus files + DB rows."""
from __future__ import annotations

import logging
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


def guess_start_time(path: Path) -> datetime:
    """Try YYYYMMDD[_-]HHMMSS or YYYY-MM-DD[ _]HH-MM-SS in the file name, else file mtime minus duration (set later)."""
    import re

    s = path.stem
    m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})[ _T-]?(\d{2})[-_:]?(\d{2})[-_:]?(\d{2})", s)
    if m:
        y, mo, d, h, mi, se = (int(g) for g in m.groups())
        try:
            return datetime(y, mo, d, h, mi, se)
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime)


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
    import re

    return re.search(r"20\d{2}[-_]?\d{2}[-_]?\d{2}[ _T-]?\d{2}[-_:]?\d{2}", path.stem) is not None
