"""Always-on recorder: mic -> VAD gate -> speech-only segment files.

Run in a thread (see service.py) or directly via `audio-note record`.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta

import numpy as np

from ..config import Config
from ..db import DB
from .capture import MicCapture
from .storage import PartialWriter, finalize_segment, recover_partials
from .vad import SAMPLE_RATE, WINDOW, GateConfig, SileroVAD, SpeechGate

log = logging.getLogger(__name__)


def gate_config_from(cfg: Config) -> GateConfig:
    a = cfg.audio
    return GateConfig(threshold=float(a.vad_threshold), min_silence_s=float(a.min_silence_s),
                      min_speech_s=float(a.min_speech_s), max_segment_s=float(a.max_segment_s),
                      pre_roll_s=float(a.pre_roll_s), post_roll_s=float(a.post_roll_s))


class Recorder:
    def __init__(self, cfg: Config, db: DB, stop_event: threading.Event | None = None):
        self.cfg = cfg
        self.db = db
        self.stop_event = stop_event or threading.Event()
        self.vad = SileroVAD(cfg.model_path("vad_model"), threads=1)
        self.gate = SpeechGate(gate_config_from(cfg))
        self.segments_written = 0
        self.speech_total_s = 0.0
        self.started_at: datetime | None = None
        self.stream_t0: datetime | None = None   # wall time of absolute sample 0
        self._partial: PartialWriter | None = None
        self._partial_start_sample = 0
        self.last_prob = 0.0
        self.is_speaking = False

    # wall-clock time of an absolute sample index
    def _time_of(self, sample: int) -> datetime:
        assert self.stream_t0 is not None
        return self.stream_t0 + timedelta(seconds=sample / SAMPLE_RATE)

    def run(self) -> None:
        cfg = self.cfg
        cfg.ensure_dirs()
        recover_partials(cfg, self.db, self.vad)
        self.started_at = datetime.now()
        carry = np.zeros(0, dtype=np.float32)
        with MicCapture(cfg.audio.device, int(cfg.audio.block_ms), float(cfg.audio.gain)) as mic:
            self.stream_t0 = datetime.now()
            consumed = 0  # absolute samples fed to the gate
            last_log = time.time()
            while not self.stop_event.is_set():
                block = mic.read(timeout=0.5)
                if block is None:
                    continue
                buf = np.concatenate([carry, block]) if len(carry) else block
                n = len(buf) // WINDOW
                for i in range(n):
                    w = buf[i * WINDOW:(i + 1) * WINDOW]
                    p = self.vad.prob(w)
                    self.last_prob = p
                    was_active = self.gate.active
                    segs = self.gate.push(w, p)
                    consumed += WINDOW
                    if self.gate.active and not was_active:
                        self._open_partial(consumed - self.gate._buf_len)
                    if self._partial is not None and self.gate.active:
                        self._partial.write(w)
                    for seg in segs:
                        self._emit(seg)
                    self.is_speaking = self.gate.active
                carry = buf[n * WINDOW:]
                # periodic drift correction: re-anchor stream_t0 if queue backlog grew
                if time.time() - last_log > 300:
                    last_log = time.time()
                    log.info("recorder alive: %d segments, %.0fs speech, overflows=%d dropped=%d",
                             self.segments_written, self.speech_total_s, mic.overflows, mic.dropped)
            for seg in self.gate.flush():
                self._emit(seg)

    def _open_partial(self, start_sample: int) -> None:
        self._partial_start_sample = start_sample
        started = self._time_of(start_sample)
        try:
            self._partial = PartialWriter(self.cfg, started)
            # pre-roll already sits in the gate buffer; write it now
            if self.gate._buf:
                for b in self.gate._buf[:-1]:
                    self._partial.write(b)
        except Exception as e:  # noqa: BLE001
            log.warning("partial writer failed: %s", e)
            self._partial = None

    def _emit(self, seg) -> None:
        started = self._time_of(seg.start_sample)
        partial_path = None
        if self._partial is not None:
            self._partial.close()
            partial_path = self._partial.path
            self._partial = None
        try:
            seg_id = finalize_segment(self.cfg, self.db, seg.audio, started, seg.speech_s, partial=partial_path)
        except Exception as e:  # noqa: BLE001
            log.exception("finalize failed: %s", e)
            return
        if seg_id is not None:
            self.segments_written += 1
            self.speech_total_s += seg.speech_s


def run_recorder(cfg: Config, db: DB, stop_event: threading.Event | None = None) -> Recorder:
    rec = Recorder(cfg, db, stop_event)
    rec.run()
    return rec
