"""Pure numpy audio helpers for the bench build: resampling, level measurement, far-field simulation
(attenuation + synthetic reverb), noise floor and per-block channel levels. No model, no I/O, deterministic
for a given seed. Unit-tested in tests/test_bench_audio.py.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

SR = 16000
BLOCK = 512                 # samples per capture block (32 ms at 16 kHz), same as the live recorder
SILENCE_DB = -120.0


def stable_seed(*parts: object) -> int:
    """A 32-bit seed derived from the parts (the same on every machine and run: no hash randomisation)."""
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


def stable_unit(*parts: object) -> float:
    """A deterministic number in [0, 1) derived from the parts."""
    return stable_seed(*parts) / 2 ** 32


def resample(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    """Band-limited resampling by spectrum truncation / zero padding (ideal low-pass; fine for clips that
    start and end in silence). Returns float32."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if sr_in == sr_out or x.size == 0:
        return x.astype(np.float32)
    n_out = int(round(x.size * sr_out / sr_in))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    spec = np.fft.rfft(x)
    out = np.zeros(n_out // 2 + 1, dtype=np.complex128)
    k = min(spec.size, out.size)
    out[:k] = spec[:k]
    if sr_out < sr_in and k == out.size and n_out % 2 == 0:
        out[-1] = out[-1].real            # the new Nyquist bin of an even-length signal must be real
    y = np.fft.irfft(out, n=n_out) * (n_out / x.size)
    return y.astype(np.float32)


def active_level_db(x: np.ndarray, frame_s: float = 0.05, top_frac: float = 0.5, sr: int = SR) -> float:
    """Speech level in dBFS: RMS over the most energetic `top_frac` of `frame_s` frames. Same definition as
    audio_note.audio.loudness.speech_dbfs (the level the pipeline's faint-voice gate looks at)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return SILENCE_DB
    n = max(1, int(round(frame_s * sr)))
    frames = x.size // n
    if frames >= 1:
        energy = (x[: frames * n].reshape(frames, n) ** 2).mean(axis=1)
    else:
        energy = np.array([float(np.mean(x ** 2))])
    k = max(1, int(math.ceil(len(energy) * top_frac)))
    top = np.sort(energy)[::-1][:k]
    rms = math.sqrt(max(float(top.mean()), 0.0))
    return SILENCE_DB if rms <= 0 else max(SILENCE_DB, 20.0 * math.log10(rms))


def rms_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return SILENCE_DB
    r = math.sqrt(float(np.mean(x ** 2)))
    return SILENCE_DB if r <= 0 else max(SILENCE_DB, 20.0 * math.log10(r))


def gain_db(x: np.ndarray, db: float) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32) * np.float32(10.0 ** (db / 20.0))).astype(np.float32)


def set_active_level(x: np.ndarray, target_db: float) -> np.ndarray:
    """Scale so that active_level_db(x) == target_db (silent clips are returned unchanged)."""
    cur = active_level_db(x)
    if cur <= SILENCE_DB + 1:
        return np.asarray(x, dtype=np.float32)
    return gain_db(x, target_db - cur)


def set_peak(x: np.ndarray, peak: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return x if m <= 0 else (x * np.float32(peak / m)).astype(np.float32)


def trim_silence(x: np.ndarray, rel_db: float = -40.0, margin_s: float = 0.03, sr: int = SR) -> np.ndarray:
    """Cut leading / trailing samples quieter than `rel_db` below the clip's peak (10 ms frames), keeping a
    small margin: TTS clips come with a few hundred ms of padding that would blur the truth timestamps."""
    x = np.asarray(x, dtype=np.float32).ravel()
    n = max(1, int(0.01 * sr))
    frames = x.size // n
    if frames < 3:
        return x
    env = np.sqrt((x[: frames * n].reshape(frames, n).astype(np.float64) ** 2).mean(axis=1))
    peak = float(env.max())
    if peak <= 0:
        return x
    on = np.nonzero(env >= peak * 10.0 ** (rel_db / 20.0))[0]
    m = int(margin_s * sr)
    a = max(0, int(on[0]) * n - m)
    b = min(x.size, (int(on[-1]) + 1) * n + m)
    return x[a:b]


def fade(x: np.ndarray, ms: float = 5.0, sr: int = SR) -> np.ndarray:
    """Short raised-cosine fade in / out so a trimmed clip never starts with a click."""
    x = np.array(x, dtype=np.float32, copy=True)
    k = min(int(ms / 1000.0 * sr), x.size // 2)
    if k > 0:
        w = (0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, k))).astype(np.float32)
        x[:k] *= w
        x[-k:] *= w[::-1]
    return x


def reverb_ir(rt60_s: float = 0.45, drr_db: float = 0.0, seed: int = 0, sr: int = SR, predelay_s: float = 0.004) -> np.ndarray:
    """Synthetic room impulse response: a unit direct path followed (after `predelay_s`) by Gaussian noise with
    an exponential decay reaching -60 dB at `rt60_s`. `drr_db` is the direct-to-reverberant energy ratio."""
    n = int(round(rt60_s * 1.5 * sr))
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    tail = rng.standard_normal(n) * np.exp(-6.907755 * t / max(rt60_s, 1e-3))     # ln(1000) = 6.9078 -> -60 dB
    d = int(round(predelay_s * sr))
    ir = np.zeros(d + n, dtype=np.float64)
    ir[0] = 1.0
    e_tail = float(np.sum(tail ** 2))
    if e_tail > 0:
        ir[d:] += tail * math.sqrt(10.0 ** (-drr_db / 10.0) / e_tail)
    return ir.astype(np.float32)


def convolve(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Full convolution via FFT (len(x) + len(ir) - 1 samples), float32."""
    x = np.asarray(x, dtype=np.float64).ravel()
    ir = np.asarray(ir, dtype=np.float64).ravel()
    if x.size == 0 or ir.size == 0:
        return np.zeros(0, dtype=np.float32)
    n = x.size + ir.size - 1
    nfft = 1 << (n - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)[:n]
    return y.astype(np.float32)


def far_field(x: np.ndarray, *, level_db: float, rt60_s: float = 0.45, drr_db: float = 0.0, seed: int = 0) -> np.ndarray:
    """A far-away talker: `x` convolved with a synthetic reverberant impulse response, then scaled so the
    speech level (active_level_db over the original span) is `level_db`. The output is longer than the input
    by the reverberation tail."""
    y = convolve(x, reverb_ir(rt60_s, drr_db, seed))
    cur = active_level_db(y[: max(1, len(np.asarray(x).ravel()))])
    if cur <= SILENCE_DB + 1:
        return y
    return gain_db(y, level_db - cur)


def noise_floor(n: int, level_db: float, seed: int = 0, colour: float = 0.9) -> np.ndarray:
    """Stationary room noise: white Gaussian noise through a one-pole low-pass (`colour` in [0, 1): 0 = white),
    scaled to an RMS of `level_db` dBFS."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    if colour > 0:
        # y[k] = colour * y[k-1] + (1 - colour) * w[k], done in the frequency domain (no Python loop)
        nfft = 1 << (n - 1).bit_length()
        f = np.fft.rfftfreq(nfft)
        h = (1.0 - colour) / (1.0 - colour * np.exp(-2j * math.pi * f))
        w = np.fft.irfft(np.fft.rfft(w, nfft) * h, nfft)[:n]
    r = math.sqrt(float(np.mean(w ** 2)))
    if r > 0:
        w = w * (10.0 ** (level_db / 20.0) / r)
    return w.astype(np.float32)


def mix_into(buf: np.ndarray, x: np.ndarray, start: int) -> None:
    """buf[start : start + len(x)] += x, clipped to the buffer."""
    if start >= buf.size or x.size == 0:
        return
    a = max(0, start)
    b = min(buf.size, start + x.size)
    if b > a:
        buf[a:b] += x[a - start: b - start]


def soft_limit(x: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    """Transparent below 80 % of the ceiling, tanh knee above: overlapping talkers never wrap around."""
    x = np.asarray(x, dtype=np.float32)
    knee = 0.8 * ceiling
    out = x.copy()
    hot = np.abs(x) > knee
    if np.any(hot):
        over = (np.abs(x[hot]) - knee) / (ceiling - knee)
        out[hot] = np.sign(x[hot]) * (knee + (ceiling - knee) * np.tanh(over))
    return out


def block_rms(x: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """RMS of every `block` samples (the last partial block is zero padded), float32."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = int(math.ceil(x.size / block)) if x.size else 0
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    pad = np.zeros(n * block, dtype=np.float64)
    pad[: x.size] = x
    return np.sqrt((pad.reshape(n, block) ** 2).mean(axis=1)).astype(np.float32)


def channel_pairs(mic: np.ndarray, system: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """(n_blocks, 2) float32 array of (mic_rms, system_rms) per capture block: what the live recorder stores in
    segments.channel_rms when it mixes the microphone with system audio."""
    a, b = block_rms(mic, block), block_rms(system, block)
    n = max(a.size, b.size)
    out = np.zeros((n, 2), dtype=np.float32)
    out[: a.size, 0] = a
    out[: b.size, 1] = b
    return out
