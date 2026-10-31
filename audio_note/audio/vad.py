"""Silero VAD (v4 ONNX, from sherpa-onnx releases) run directly with onnxruntime.

Two layers:
  SileroVAD        – per-window speech probability with internal RNN state (streaming).
  SpeechGate       – state machine turning probabilities into speech segments with
                     pre-roll, hangover, min-speech and max-segment rules. Works both
                     for live recording (feed blocks as they arrive) and offline files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

WINDOW = 512          # samples per VAD window @16 kHz (32 ms)
SAMPLE_RATE = 16000


class SileroVAD:
    def __init__(self, model_path: Path | str, threads: int = 1):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        names = {i.name for i in self.sess.get_inputs()}
        self._v4 = "h" in names  # v4: (x, h, c) ; v5: (input, state, sr)
        self.reset()

    def reset(self) -> None:
        if self._v4:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)
        else:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._carry = np.zeros(0, dtype=np.float32)

    def prob(self, window: np.ndarray) -> float:
        """window: exactly WINDOW float32 samples in [-1, 1]."""
        x = window.reshape(1, WINDOW).astype(np.float32, copy=False)
        if self._v4:
            p, self._h, self._c = self.sess.run(None, {"x": x, "h": self._h, "c": self._c})
        else:
            p, self._state = self.sess.run(None, {"input": x, "state": self._state, "sr": np.array(SAMPLE_RATE, dtype=np.int64)})
        return float(p.reshape(-1)[0])

    def probs(self, audio: np.ndarray) -> list[float]:
        """Stream an arbitrary-length block; returns one prob per full window consumed.
        Leftover samples are carried to the next call."""
        buf = np.concatenate([self._carry, audio.astype(np.float32, copy=False)])
        n = len(buf) // WINDOW
        out = [self.prob(buf[i * WINDOW:(i + 1) * WINDOW]) for i in range(n)]
        self._carry = buf[n * WINDOW:]
        return out


@dataclass
class Segment:
    start_sample: int            # absolute sample index in the stream
    audio: np.ndarray            # float32 mono 16 kHz
    speech_s: float              # seconds of windows judged speech
    reason: str = "silence"      # silence | max_len | flush

    @property
    def duration_s(self) -> float:
        return len(self.audio) / SAMPLE_RATE


@dataclass
class GateConfig:
    threshold: float = 0.5
    min_silence_s: float = 1.0
    min_speech_s: float = 0.3
    max_segment_s: float = 300.0
    pre_roll_s: float = 0.5
    post_roll_s: float = 0.5
    hysteresis: float = 0.15     # speech ends when prob < threshold - hysteresis


@dataclass
class SpeechGate:
    """Feed 512-sample windows with their VAD probability; emits Segment objects.

    State: idle -> (speech >= min_speech) -> active -> (silence >= min_silence or max_len) -> idle
    """
    cfg: GateConfig = field(default_factory=GateConfig)
    _pre: list[np.ndarray] = field(default_factory=list)      # pre-roll ring (idle only)
    _pre_len: int = 0
    _active: bool = False
    _buf: list[np.ndarray] = field(default_factory=list)
    _buf_len: int = 0
    _seg_start: int = 0
    _speech_windows: int = 0
    _silence_windows: int = 0
    _onset_windows: int = 0
    _pos: int = 0                                              # absolute sample counter

    @property
    def active(self) -> bool:
        return self._active

    def _win_s(self) -> float:
        return WINDOW / SAMPLE_RATE

    def push(self, window: np.ndarray, prob: float) -> list[Segment]:
        out: list[Segment] = []
        c = self.cfg
        on = prob >= c.threshold
        off = prob < (c.threshold - c.hysteresis)
        start_pos = self._pos
        self._pos += len(window)

        if not self._active:
            # pre-roll ring
            self._pre.append(window)
            self._pre_len += len(window)
            max_pre = int(c.pre_roll_s * SAMPLE_RATE) + WINDOW
            while self._pre_len > max_pre and len(self._pre) > 1:
                self._pre_len -= len(self._pre.pop(0))
            if on:
                self._onset_windows += 1
                if self._onset_windows * self._win_s() >= c.min_speech_s:
                    self._active = True
                    self._buf = list(self._pre)
                    self._buf_len = self._pre_len
                    self._seg_start = self._pos - self._pre_len
                    self._speech_windows = self._onset_windows
                    self._silence_windows = 0
                    self._pre, self._pre_len = [], 0
            else:
                self._onset_windows = 0
            return out

        # active
        self._buf.append(window)
        self._buf_len += len(window)
        if on:
            self._speech_windows += 1
            self._silence_windows = 0
        elif off:
            self._silence_windows += 1
        # (in the hysteresis band: neither counter moves)

        if self._silence_windows * self._win_s() >= c.min_silence_s:
            out.append(self._close("silence"))
        elif self._buf_len >= int(c.max_segment_s * SAMPLE_RATE):
            out.append(self._close("max_len"))
        return out

    def _close(self, reason: str) -> Segment:
        c = self.cfg
        audio = np.concatenate(self._buf) if self._buf else np.zeros(0, dtype=np.float32)
        if reason == "silence":
            # trim trailing silence down to post_roll
            trailing = self._silence_windows * WINDOW
            keep_tail = int(c.post_roll_s * SAMPLE_RATE)
            cut = max(0, trailing - keep_tail)
            if cut:
                audio = audio[:-cut]
        seg = Segment(start_sample=self._seg_start, audio=audio,
                      speech_s=self._speech_windows * self._win_s(), reason=reason)
        self._active = False
        self._buf, self._buf_len = [], 0
        self._speech_windows = self._silence_windows = self._onset_windows = 0
        return seg

    def flush(self) -> list[Segment]:
        """Close any open segment (on shutdown)."""
        if self._active and self._buf_len > 0:
            return [self._close("flush")]
        return []


def segment_audio(audio: np.ndarray, vad: SileroVAD, cfg: GateConfig) -> list[Segment]:
    """Offline: run the gate over a whole 16 kHz float32 array."""
    vad.reset()
    gate = SpeechGate(cfg)
    segs: list[Segment] = []
    n = len(audio) // WINDOW
    for i in range(n):
        w = audio[i * WINDOW:(i + 1) * WINDOW]
        segs.extend(gate.push(w, vad.prob(w)))
    tail = audio[n * WINDOW:]
    if len(tail):
        w = np.zeros(WINDOW, dtype=np.float32)
        w[:len(tail)] = tail
        segs.extend(gate.push(w, vad.prob(w)))
    segs.extend(gate.flush())
    return segs


def speech_seconds(audio: np.ndarray, vad: SileroVAD, threshold: float = 0.5) -> float:
    """Total seconds of windows above threshold (for storage re-checks)."""
    vad.reset()
    n = len(audio) // WINDOW
    cnt = 0
    for i in range(n):
        if vad.prob(audio[i * WINDOW:(i + 1) * WINDOW]) >= threshold:
            cnt += 1
    return cnt * WINDOW / SAMPLE_RATE
