"""System-audio capture ("what is playing through the speakers / headphones": the other side of an
online meeting or call) via WASAPI loopback, using PyAudioWPatch (a PyAudio build whose PortAudio
exposes every output device as an additional input device named "<output> [Loopback]").

    with LoopbackCapture(device=None) as lb:        # None = the default output device's loopback
        chunk = lb.read_available()                 # float32 mono 16 kHz, whatever arrived since last call

Windows only delivers loopback data while something is actually rendering: silence produces no
callbacks at all, so a consumer must treat "nothing available" as silence (MixedCapture does).
The import of pyaudiowpatch is lazy; on other platforms or without the wheel, `available()` is
False and opening raises a clear RuntimeError.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

TARGET_SR = 16000
LOOPBACK_SUFFIX = " [Loopback]"
AGC_TARGET_PEAK = 0.5
AGC_MAX_GAIN = 40.0
AGC_MIN_PEAK = 1e-4      # below this the chunk is digital silence / noise floor: leave the gain alone
AGC_DECAY = 0.999        # per chunk (~21 ms at 48 kHz / 1024 frames): the tracked peak halves in ~15 s


@dataclass
class LoopbackDevice:
    index: int           # PyAudioWPatch device index of the loopback input
    name: str            # output device name without the " [Loopback]" suffix
    channels: int
    default_sr: float
    is_default: bool


def _pa() -> Any:
    import pyaudiowpatch as pa  # noqa: WPS433 - optional dependency

    return pa


def available() -> bool:
    try:
        _pa()
        return True
    except Exception:  # noqa: BLE001
        return False


def list_loopback_devices() -> list[LoopbackDevice]:
    """Every output device that can be captured (name = the output's name). Empty when unsupported."""
    try:
        pa = _pa()
    except Exception as e:  # noqa: BLE001
        log.debug("loopback unavailable: %s", e)
        return []
    p = pa.PyAudio()
    try:
        try:
            default_idx = p.get_default_wasapi_loopback()["index"]
        except Exception:  # noqa: BLE001
            default_idx = None
        out: list[LoopbackDevice] = []
        for d in p.get_loopback_device_info_generator():
            name = str(d["name"])
            if name.endswith(LOOPBACK_SUFFIX):
                name = name[: -len(LOOPBACK_SUFFIX)]
            out.append(LoopbackDevice(int(d["index"]), name, int(d["maxInputChannels"]), float(d["defaultSampleRate"]),
                                      int(d["index"]) == default_idx))
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("cannot list loopback devices: %s", e)
        return []
    finally:
        p.terminate()


def resolve_loopback(spec: int | str | None, devices: list[LoopbackDevice] | None = None) -> LoopbackDevice:
    """None -> default output's loopback; int -> loopback index (or the output device's index, matched
    by name); str -> first loopback whose name contains it (case-insensitive)."""
    devices = list_loopback_devices() if devices is None else devices
    if not devices:
        raise RuntimeError("没有可用的系统声音源（需要 Windows + PyAudioWPatch；运行 uv pip install PyAudioWPatch）")
    if spec is None or spec == "":
        for d in devices:
            if d.is_default:
                return d
        return devices[0]
    if isinstance(spec, int) or (isinstance(spec, str) and spec.strip().isdigit()):
        idx = int(spec)
        for d in devices:
            if d.index == idx:
                return d
        # an output-device index from sounddevice's table: match by name
        try:
            import sounddevice as sd

            name = str(sd.query_devices(idx)["name"])
            for d in devices:
                if d.name == name or name.startswith(d.name) or d.name.startswith(name):
                    return d
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(f"没有序号为 {idx} 的系统声音源；可用：" + "、".join(f"{d.index}: {d.name}" for d in devices))
    s = str(spec).lower()
    for d in devices:
        if s in d.name.lower():
            return d
    raise RuntimeError(f"没有名字包含 {spec!r} 的系统声音源；可用：" + "、".join(d.name for d in devices))


class StreamResampler:
    """Linear-interpolation resampler that keeps its phase across chunks. Resampling each 1024-frame
    chunk on its own (1024 @ 48 kHz -> 341.33 @ 16 kHz) would drop a third of a sample per chunk and
    put a tiny discontinuity every 21 ms, which the VAD and ASR hear as a buzz."""

    def __init__(self, sr_in: int, sr_out: int):
        self.ratio = sr_in / sr_out          # input samples per output sample
        self._carry = np.zeros(0, dtype=np.float32)
        self._pos = 0.0                      # position of the next output sample, in input samples of the buffer

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.ratio == 1.0:
            return x
        buf = np.concatenate([self._carry, x]) if len(self._carry) else x
        if len(buf) < 2:
            self._carry = buf
            return np.zeros(0, dtype=np.float32)
        last = len(buf) - 1
        n = int((last - self._pos) // self.ratio) + 1 if self._pos <= last else 0
        if n <= 0:
            self._carry = buf
            return np.zeros(0, dtype=np.float32)
        idx = self._pos + self.ratio * np.arange(n)
        y = np.interp(idx, np.arange(len(buf)), buf).astype(np.float32)
        next_pos = self._pos + self.ratio * n
        # keep at least the last input sample so the first output of the next chunk can interpolate
        # between it and the new data; `_pos` stays relative to the kept samples
        keep = min(int(np.floor(next_pos)), last)
        self._carry = buf[keep:]
        self._pos = next_pos - keep
        return y


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """One-shot resample (whole clips, tests)."""
    if sr_in == sr_out:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    return np.interp(np.arange(n_out) / sr_out, np.arange(len(x)) / sr_in, x).astype(np.float32)


class LoopbackCapture:
    """Context manager around one WASAPI loopback stream. Chunks (float32 mono 16 kHz) queue up as
    they arrive; `read_available()` drains the queue."""

    def __init__(self, device: int | str | None = None, gain: float = 1.0, frames_per_buffer: int = 1024, max_queue: int = 4000):
        self.spec = device
        self.gain = float(gain)
        self.frames_per_buffer = int(frames_per_buffer)
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue)
        self.device: LoopbackDevice | None = None
        self.sr = 0
        self.channels = 0
        self.dropped = 0
        self.chunks = 0
        self.last_callback_at = 0.0
        # automatic gain: loopback audio is post-volume (a quiet Windows volume slider gives a tiny
        # signal the VAD ignores), so chunks are scaled towards AGC_TARGET_PEAK with a slowly decaying
        # peak tracker, up to AGC_MAX_GAIN
        self._peak = 0.0
        self.agc_gain = 1.0
        self._p: Any = None
        self._stream: Any = None
        self._lock = threading.Lock()
        self._resampler: StreamResampler | None = None

    def __enter__(self) -> "LoopbackCapture":
        pa = _pa()   # ImportError -> clear message from the caller
        self.device = resolve_loopback(self.spec)
        self.sr = int(self.device.default_sr) or 48000
        self.channels = max(1, int(self.device.channels))
        self._resampler = StreamResampler(self.sr, TARGET_SR)
        self._p = pa.PyAudio()
        try:
            self._stream = self._p.open(format=pa.paFloat32, channels=self.channels, rate=self.sr, input=True,
                                        input_device_index=self.device.index, frames_per_buffer=self.frames_per_buffer,
                                        stream_callback=self._cb)
            self._stream.start_stream()
        except Exception:
            self._close()
            raise
        self.last_callback_at = time.time()
        log.info("system audio capture started: %s @ %d Hz x%d", self.device.name, self.sr, self.channels)
        return self

    def _cb(self, data: bytes, frames: int, time_info: Any, status: int) -> tuple[None, int]:
        self.last_callback_at = time.time()
        try:
            x = np.frombuffer(data, dtype=np.float32)
            if self.channels > 1:
                x = x.reshape(-1, self.channels).mean(axis=1)
            x = x.astype(np.float32, copy=False)
            peak = float(np.abs(x).max()) if len(x) else 0.0
            self._peak = max(peak, self._peak * AGC_DECAY)
            if self._peak > AGC_MIN_PEAK:
                self.agc_gain = float(min(AGC_MAX_GAIN, max(1.0, AGC_TARGET_PEAK / self._peak)))
            g = self.gain * self.agc_gain
            if g != 1.0:
                x = x * g
            if self.sr != TARGET_SR and self._resampler is not None:
                x = self._resampler.process(x)
                if not len(x):
                    return (None, 0)
            self.queue.put_nowait(x)
            self.chunks += 1
        except queue.Full:
            self.dropped += 1
        except Exception as e:  # noqa: BLE001 - never raise inside the PortAudio callback
            log.debug("loopback callback error: %s", e)
        return (None, 0)   # paContinue

    def read_available(self) -> np.ndarray:
        """Everything that arrived since the last call (possibly empty)."""
        parts: list[np.ndarray] = []
        while True:
            try:
                parts.append(self.queue.get_nowait())
            except queue.Empty:
                break
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts) if len(parts) > 1 else parts[0]

    def is_active(self) -> bool:
        s = self._stream
        if s is None:
            return False
        try:
            return bool(s.is_active())
        except Exception:  # noqa: BLE001
            return False

    def _close(self) -> None:
        with self._lock:
            s, self._stream = self._stream, None
            p, self._p = self._p, None
        if s is not None:
            try:
                s.stop_stream()
            except Exception as e:  # noqa: BLE001
                log.debug("loopback stop failed: %s", e)
            try:
                s.close()
            except Exception as e:  # noqa: BLE001
                log.debug("loopback close failed: %s", e)
        if p is not None:
            try:
                p.terminate()
            except Exception as e:  # noqa: BLE001
                log.debug("pyaudio terminate failed: %s", e)

    def __exit__(self, *exc: Any) -> None:
        self._close()
