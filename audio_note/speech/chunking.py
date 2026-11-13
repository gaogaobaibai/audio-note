"""Turns -> ASR chunks, plus the small primitives every other part of the split pipeline shares.

`audio_note.speech.pipeline` used to be one 1200-line module; it is now a facade over

    chunking.py       Chunk, build_chunks, _clip + Stopped, stage_timer, F   (this file: a leaf, it imports none
                                                                               of the modules below)
    channels.py       system-audio share of a time range (segments.channel_rms)
    markdown.py       transcripts/YYYY-MM-DD.md
    glossary_file.py  the one glossary.txt parser
    stage_voice.py    VoiceStage: audio -> identified turns
    stage_text.py     TextStage: chunks -> raw text -> corrected text
    processor.py      Processor(VoiceStage, TextStage): orchestration, storage, hooks; transcribe_file
    relabel.py        re-identification of stored utterances (no ASR)
    split.py          splitting one mixed utterance

The facade and monkeypatching
-----------------------------
Every name that was a global of the old pipeline.py is still an attribute of `audio_note.speech.pipeline`, and the
code in the modules above reaches such names through `F` (`F.read_segment(path)`, `F.SPLIT_PAD_S`, ...): a
late-bound view of the facade module. So `monkeypatch.setattr(pipeline, "write_day_markdown", fake)` still reaches
every caller, exactly as it did when all of this lived in one module. Classes and exceptions (Chunk, Turn,
Stopped, LLMError, ...) are imported directly: they are used for construction / isinstance / except clauses.

All of these modules log as "audio_note.speech.pipeline" (the logger name is part of the old module's observable
behaviour: log filters, tests).
"""
from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np

from ..audio.vad import SAMPLE_RATE
from .speakers import Turn

LOGGER_NAME = "audio_note.speech.pipeline"
log = logging.getLogger(LOGGER_NAME)


# --------------------------------------------------------------------------- facade access
class _Facade:
    """Attribute access on the `audio_note.speech.pipeline` module, resolved at call time (see the module docstring).
    The import is late because the facade imports this module."""

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        from . import pipeline

        return getattr(pipeline, name)


F = _Facade()


# --------------------------------------------------------------------------- stop requests
class Stopped(RuntimeError):
    """A stop request (service shutdown, `stop_event`) cut the work on a segment short. It is not a failure:
    Processor.process_segment puts the segment back to 'new' so the next run does it again from the start.
    Raise it as `Stopped("stopped")` (the message older code matched on is kept)."""


# --------------------------------------------------------------------------- stage timing
@contextmanager
def stage_timer(db: Any, name: str, audio_s: float = 0.0, n: int = 1) -> Iterator[None]:
    """`audio_note.metrics.stage(db, name, audio_s=..., n=...)` around a pipeline stage when that module exists;
    otherwise, or when the metrics code itself fails, a no-op. Accounting never changes what the stage does:
    exceptions of the wrapped block always propagate unchanged, exceptions of the metrics code never do.
    Stage names used here: gate, diarize, identify, asr, correct, store."""
    cm = None
    try:
        from .. import metrics

        cm = metrics.stage(db, name, audio_s=float(audio_s or 0.0), n=int(n))
        cm.__enter__()
    except Exception as e:  # noqa: BLE001 - module absent / being written / broken: run the stage without accounting
        if not isinstance(e, ImportError):
            log.debug("metrics.stage(%s) unavailable: %s", name, e)
        cm = None
    try:
        yield
    except BaseException as ev:
        _leave_stage(cm, name, ev)
        raise
    else:
        _leave_stage(cm, name, None)


def _leave_stage(cm: Any, name: str, ev: BaseException | None) -> None:
    if cm is None:
        return
    try:
        if ev is None:
            cm.__exit__(None, None, None)
        else:
            cm.__exit__(type(ev), ev, ev.__traceback__)      # its verdict is ignored: the caller re-raises `ev`
    except Exception as e:  # noqa: BLE001
        if e is not ev:      # the stage's own exception coming back out of the metrics code is not a metrics failure
            log.debug("metrics.stage(%s) failed on exit: %s", name, e)


# --------------------------------------------------------------------------- chunks
@dataclass
class Chunk:
    """A stretch of audio from one diarization cluster, bounded for ASR.

    start_s / end_s are the speech bounds (used for timestamps); audio_start_s / audio_end_s
    include the padding actually sent to ASR (clamped to the file).
    """
    start_s: float
    end_s: float
    cluster: int
    speaker_id: int | None = None
    speaker_name: str | None = None
    score: float | None = None
    audio_start_s: float = 0.0
    audio_end_s: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def slice(self, audio: np.ndarray) -> np.ndarray:
        a = max(0, int(round(self.audio_start_s * SAMPLE_RATE)))
        b = min(len(audio), int(round(self.audio_end_s * SAMPLE_RATE)))
        return audio[a:b]


def _clip(audio: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    a = max(0, int(float(start_s) * SAMPLE_RATE))
    b = min(len(audio), int(float(end_s) * SAMPLE_RATE))
    return audio[a:b] if b > a else audio[:0]


def build_chunks(turns: Sequence[Turn], audio_len_s: float, max_chunk_s: float, gap_s: float = 1.0,
                 pad_s: float = 0.15, min_chunk_s: float = 0.6) -> list[Chunk]:
    """Turn diarization turns into ASR chunks (pure function).

    1. Sort turns by start; merge consecutive turns of the same cluster whose gap is <= gap_s,
       as long as the merged span stays <= max_chunk_s (otherwise a new chunk starts).
    2. A single turn longer than max_chunk_s is cut into equal pieces <= max_chunk_s.
    3. Chunks shorter than min_chunk_s are dropped.
    4. pad_s is added on both sides for the audio slice, clamped to [0, audio_len_s].
    """
    max_chunk_s = float(max_chunk_s) if max_chunk_s and max_chunk_s > 0 else float("inf")
    pieces: list[Turn] = []
    for t in sorted(turns, key=lambda t: (t.start, t.end)):
        start, end = float(t.start), min(float(t.end), float(audio_len_s))
        if end <= start:
            continue
        dur = end - start
        if dur > max_chunk_s:
            n = int(math.ceil(dur / max_chunk_s))
            step = dur / n
            for k in range(n):
                pieces.append(Turn(start + k * step, start + (k + 1) * step, t.cluster, t.speaker_id, t.speaker_name, t.score))
        else:
            pieces.append(Turn(start, end, t.cluster, t.speaker_id, t.speaker_name, t.score))

    merged: list[Chunk] = []
    for p in pieces:
        last = merged[-1] if merged else None
        if (last is not None and last.cluster == p.cluster and p.start - last.end_s <= gap_s
                and p.end - last.start_s <= max_chunk_s):
            last.end_s = max(last.end_s, p.end)
        else:
            merged.append(Chunk(p.start, p.end, p.cluster, p.speaker_id, p.speaker_name, p.score))

    out: list[Chunk] = []
    for c in merged:
        if c.duration_s < min_chunk_s:
            continue
        c.audio_start_s = max(0.0, c.start_s - pad_s)
        c.audio_end_s = min(float(audio_len_s), c.end_s + pad_s)
        out.append(c)
    return out


__all__ = ["Chunk", "F", "LOGGER_NAME", "Stopped", "build_chunks", "stage_timer"]
