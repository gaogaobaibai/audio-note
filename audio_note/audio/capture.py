"""Microphone capture via sounddevice (PortAudio). Produces float32 mono 16 kHz blocks on a queue.

Every block is stamped with the wall-clock time of its PortAudio callback (``last_block_wall``
after ``read()``), so the consumer can anchor sample indices to real time even when blocks were
dropped, the machine slept, or the queue built up a backlog. ``last_callback_at`` / ``is_active()``
let the consumer detect a stream that silently stopped delivering (device unplugged, WASAPI
endpoint invalidated after sleep/resume).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
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


def refresh_devices() -> bool:
    """Re-enumerate PortAudio devices. PortAudio takes its device table (indices, names, default
    device) once at initialisation, so a microphone plugged in / replugged / made default after the
    process started is invisible until Pa_Terminate + Pa_Initialize. Returns True on success; never raises."""
    try:
        sd._terminate()
        sd._initialize()
        log.info("audio device list refreshed")
        return True
    except Exception as e:  # noqa: BLE001 - best effort; the old table keeps working
        log.warning("device refresh failed: %s", e)
        return False


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
        self.queue: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(maxsize=max_queue)
        self.sr = TARGET_SR
        self._stream: sd.InputStream | None = None
        self.dropped = 0
        self.overflows = 0
        self.last_callback_at: float = 0.0   # time.time() of the most recent PortAudio callback
        self.last_block_wall: float = 0.0    # time.time() stamp of the block most recently returned by read()
        self._lock = threading.Lock()

    def _open(self, sr: int) -> sd.InputStream:
        blocksize = int(sr * self.block_ms / 1000)
        return sd.InputStream(device=self.device, channels=1, samplerate=sr, dtype="float32",
                              blocksize=blocksize, callback=self._cb, latency="high")

    def __enter__(self) -> "MicCapture":
        # `self.sr` must be final BEFORE start(): the PortAudio thread may run _cb immediately after
        # start() returns and reads self.sr to decide whether a block needs resampling.
        try:
            self.sr = TARGET_SR
            self._stream = self._open(TARGET_SR)
            self._stream.start()
        except Exception as e:  # device refuses 16 kHz -> open at its default and resample
            self._close_stream()   # an opened-but-not-started stream must not leak
            info = sd.query_devices(self.device, "input")
            sr = int(info["default_samplerate"])
            log.warning("16 kHz capture failed (%s); falling back to %d Hz + resample", e, sr)
            self.sr = sr
            try:
                self._stream = self._open(sr)
                self._stream.start()
            except Exception:
                self._close_stream()
                raise
        log.info("capture started on device %s (%s) @ %d Hz", self.device, sd.query_devices(self.device, "input")["name"], self.sr)
        self.last_callback_at = time.time()
        return self

    def _close_stream(self) -> None:
        """stop() + close() the current stream; a failing stop() (endpoint invalidated by unplug or
        sleep/resume) must not prevent close(), or the PortAudio stream leaks on every restart."""
        s = self._stream
        self._stream = None
        if s is None:
            return
        try:
            s.stop()
        except Exception as e:  # noqa: BLE001
            log.warning("stream stop failed (%s); closing anyway", e)
        finally:
            try:
                s.close()
            except Exception as e:  # noqa: BLE001
                log.warning("stream close failed: %s", e)

    def is_active(self) -> bool:
        """True while the PortAudio stream reports itself running."""
        s = self._stream
        if s is None:
            return False
        try:
            return bool(s.active)
        except Exception:  # noqa: BLE001 - closed/invalid stream
            return False

    def __exit__(self, *exc) -> None:
        self._close_stream()

    def _cb(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        now = time.time()
        self.last_callback_at = now
        if status and status.input_overflow:
            self.overflows += 1
        x = indata[:, 0].copy()
        if self.gain != 1.0:
            x *= self.gain
        if self.sr != TARGET_SR:
            x = _resample(x, self.sr, TARGET_SR)
        try:
            self.queue.put_nowait((x, now))
        except queue.Full:
            self.dropped += 1

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        """Next block (float32 mono 16 kHz) or None on timeout; its callback wall time is left in
        ``last_block_wall``."""
        try:
            block, wall = self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
        self.last_block_wall = wall
        return block
