"""ASR over audio arrays via MiMo (mimo-v2.5-asr).

The API returns punctuated text only (no timestamps), so callers pass speaker-homogeneous
chunks bounded by diarization/VAD and keep their own timing. Chunks longer than
`mimo.asr_max_chunk_s` are split at the quietest point (VAD) and the texts joined.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from ..audio.vad import SAMPLE_RATE, WINDOW, SileroVAD
from ..config import Config
from ..llm.client import MimoClient

log = logging.getLogger(__name__)


@dataclass
class ASRResult:
    text: str
    seconds: float      # billed seconds reported by the API (0 if absent)
    parts: int          # number of API calls used


def wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio.astype(np.float32, copy=False), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def split_at_quietest(audio: np.ndarray, vad: SileroVAD, max_s: float, min_s: float = 5.0) -> list[np.ndarray]:
    """Recursively split audio > max_s at the lowest-VAD-probability window in the middle 60% of the clip."""
    if len(audio) <= int(max_s * SAMPLE_RATE):
        return [audio]
    vad.reset()
    n = len(audio) // WINDOW
    probs = np.array([vad.prob(audio[i * WINDOW:(i + 1) * WINDOW]) for i in range(n)])
    lo, hi = int(n * 0.2), int(n * 0.8)
    if hi - lo < 2:
        mid = len(audio) // 2
    else:
        k = lo + int(np.argmin(probs[lo:hi]))
        mid = k * WINDOW + WINDOW // 2
    left, right = audio[:mid], audio[mid:]
    out: list[np.ndarray] = []
    for part in (left, right):
        if len(part) >= int(min_s * SAMPLE_RATE) or len(out) == 0:
            out.extend(split_at_quietest(part, vad, max_s, min_s))
        else:
            out[-1] = np.concatenate([out[-1], part])
    return out


class ASR:
    def __init__(self, cfg: Config, client: MimoClient, vad: SileroVAD | None = None):
        self.cfg = cfg
        self.client = client
        self.vad = vad or SileroVAD(cfg.model_path("vad_model"))
        self.language = cfg.mimo.asr_language or "auto"
        self.max_chunk_s = float(cfg.mimo.asr_max_chunk_s)

    def transcribe(self, audio: np.ndarray) -> ASRResult:
        if len(audio) < int(0.3 * SAMPLE_RATE):
            return ASRResult("", 0.0, 0)
        parts = split_at_quietest(audio, self.vad, self.max_chunk_s)
        texts: list[str] = []
        secs = 0.0
        for p in parts:
            t, s = self.client.transcribe_bytes(wav_bytes(p), "audio/wav", self.language)
            texts.append(t)
            secs += s
        return ASRResult(_join(texts), secs, len(parts))


def _join(texts: list[str]) -> str:
    out = ""
    for t in texts:
        t = t.strip()
        if not t:
            continue
        if out and not out[-1] in "。！？!?.,，；;…":
            out += "，"
        out += t
    return out
