"""Microphone capture via sounddevice (PortAudio). Produces float32 mono 16 kHz blocks on a queue."""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)
TARGET_SR = 16000


@dataclass
class DeviceInfo:
    index: int
    name: str
    hostapi: str
    channels: int
    default_sr: float


def list_input_devices() -> list[DeviceInfo]:
    out: list[DeviceInfo] = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            out.append(DeviceInfo(i, d["name"], sd.query_hostapis(d["hostapi"])["name"], d["max_input_channels"], d["default_samplerate"]))
    return out


def resolve_device(spec: int | str | None) -> int | None:
    """None -> default; int -> index; str -> first input device whose name contains it (case-insensitive).
    Prefers WASAPI/DirectSound over MME when several entries share a name."""
    if spec is None or spec == "":
        return None
    if isinstance(spec, int):
        return spec
    s = str(spec)
    if s.isdigit():
        return int(s)
    cands = [d for d in list_input_devices() if s.lower() in d.name.lower()]
    if not cands:
        raise ValueError(f"no input device matching {spec!r}; run `audio-note devices`")
    prio = {"Windows WASAPI": 0, "Windows DirectSound": 1, "MME": 2}
    cands.sort(key=lambda d: prio.get(d.hostapi, 3))
    return cands[0].index


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    t_in = np.arange(len(x)) / sr_in
    t_out = np.arange(n_out) / sr_out
    return np.interp(t_out, t_in, x).astype(np.float32)


class MicCapture:
    """Context manager; blocks arrive via .queue as float32 mono @16k arrays."""

    def __init__(self, device: int | str | None = None, block_ms: int = 32, gain: float = 1.0, max_queue: int = 2000):
        self.device = resolve_device(device)
        self.block_ms = block_ms
        self.gain = gain
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue)
        self.sr = TARGET_SR
        self._stream: sd.InputStream | None = None
        self.dropped = 0
        self.overflows = 0
        self._lock = threading.Lock()

    def _open(self, sr: int) -> sd.InputStream:
        blocksize = int(sr * self.block_ms / 1000)
        return sd.InputStream(device=self.device, channels=1, samplerate=sr, dtype="float32",
                              blocksize=blocksize, callback=self._cb, latency="high")

    def __enter__(self) -> "MicCapture":
        try:
            self._stream = self._open(TARGET_SR)
            self._stream.start()
            self.sr = TARGET_SR
        except Exception as e:  # device refuses 16 kHz -> open at its default and resample
            info = sd.query_devices(self.device, "input")
            sr = int(info["default_samplerate"])
            log.warning("16 kHz capture failed (%s); falling back to %d Hz + resample", e, sr)
            self._stream = self._open(sr)
            self._stream.start()
            self.sr = sr
        log.info("capture started on device %s (%s) @ %d Hz", self.device, sd.query_devices(self.device, "input")["name"], self.sr)
        return self

    def __exit__(self, *exc) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _cb(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status and status.input_overflow:
            self.overflows += 1
        x = indata[:, 0].copy()
        if self.gain != 1.0:
            x *= self.gain
        if self.sr != TARGET_SR:
            x = _resample(x, self.sr, TARGET_SR)
        try:
            self.queue.put_nowait(x)
        except queue.Full:
            self.dropped += 1

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
