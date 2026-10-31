"""Tests for audio_note.audio.vad: SpeechGate state machine (synthetic probabilities) and SileroVAD basics."""
from __future__ import annotations

import numpy as np
import pytest

from audio_note.audio.vad import SAMPLE_RATE, WINDOW, GateConfig, SpeechGate, segment_audio, speech_seconds

from .conftest import FakeVAD, noise, silence

WIN_S = WINDOW / SAMPLE_RATE  # 0.032 s


def win(idx: int) -> np.ndarray:
    """A window whose samples all equal its index, so we can tell which windows ended up in a segment."""
    return np.full(WINDOW, float(idx), dtype=np.float32)


def feed(gate: SpeechGate, probs: list[float], start_idx: int = 0):
    """Push a list of probabilities (one per window); returns (segments, next_idx)."""
    segs = []
    i = start_idx
    for p in probs:
        segs.extend(gate.push(win(i), p))
        i += 1
    return segs, i


def windows_for(seconds: float) -> int:
    """Smallest n with n * WIN_S >= seconds (how the gate counts onset / silence)."""
    n = 0
    while n * WIN_S < seconds:
        n += 1
    return n


# ---------------------------------------------------------------------------
# SpeechGate
# ---------------------------------------------------------------------------

def default_gate() -> SpeechGate:
    return SpeechGate(GateConfig(threshold=0.5, min_silence_s=1.0, min_speech_s=0.3,
                                 max_segment_s=300.0, pre_roll_s=0.5, post_roll_s=0.5, hysteresis=0.15))


def test_onset_requires_min_speech():
    g = default_gate()
    need = windows_for(0.3)  # 10 windows
    assert need == 10
    segs, _ = feed(g, [0.0] * 20 + [0.9] * (need - 1))
    assert segs == [] and not g.active
    segs, _ = feed(g, [0.9], start_idx=20 + need - 1)
    assert segs == [] and g.active


def test_onset_counter_resets_on_silence_window():
    g = default_gate()
    # 5 speech, 1 silence, 5 speech: never 10 consecutive -> stays idle
    feed(g, [0.9] * 5 + [0.0] + [0.9] * 5)
    assert not g.active
    # a further 5 consecutive (total 10 in a row) opens the gate
    feed(g, [0.9] * 5, start_idx=11)
    assert g.active


def test_pre_roll_included_and_start_sample():
    g = default_gate()
    n_sil, n_on = 50, 10
    feed(g, [0.0] * n_sil + [0.9] * n_on)
    assert g.active
    # ring holds at most pre_roll_s*SR + WINDOW samples -> 16 windows (8192 <= 8512 < 8704)
    max_pre_windows = (int(0.5 * SAMPLE_RATE) + WINDOW) // WINDOW
    assert max_pre_windows == 16
    first_idx = n_sil + n_on - max_pre_windows
    assert g._seg_start == first_idx * WINDOW
    assert g._buf_len == max_pre_windows * WINDOW
    assert float(g._buf[0][0]) == first_idx


def test_hangover_closes_and_trailing_silence_trimmed_to_post_roll():
    g = default_gate()
    n_sil, n_on, n_more = 50, 10, 20
    segs, i = feed(g, [0.0] * n_sil + [0.9] * n_on + [0.9] * n_more)
    assert segs == []
    n_hang = windows_for(1.0)  # 32 windows
    segs, i = feed(g, [0.0] * (n_hang - 1), start_idx=i)
    assert segs == [] and g.active
    segs, i = feed(g, [0.0], start_idx=i)
    assert len(segs) == 1 and not g.active
    seg = segs[0]
    assert seg.reason == "silence"
    pre = 16
    buf_len = (pre + n_more + n_hang) * WINDOW
    cut = n_hang * WINDOW - int(0.5 * SAMPLE_RATE)
    assert len(seg.audio) == buf_len - cut
    assert seg.start_sample == (n_sil + n_on - pre) * WINDOW
    assert seg.speech_s == pytest.approx((n_on + n_more) * WIN_S)
    # audio begins with the pre-roll window and ends inside the trailing silence region
    assert float(seg.audio[0]) == n_sil + n_on - pre
    assert float(seg.audio[-1]) >= n_sil + n_on + n_more  # a silence window index
    assert abs(seg.duration_s - len(seg.audio) / SAMPLE_RATE) < 1e-9


def test_hysteresis_band_does_not_count_as_silence():
    g = default_gate()
    segs, i = feed(g, [0.9] * 10)
    assert g.active
    # 0.4 is below threshold (0.5) but above threshold-hysteresis (0.35): no counter moves
    segs, i = feed(g, [0.4] * 200, start_idx=i)
    assert segs == [] and g.active
    segs, i = feed(g, [0.0] * windows_for(1.0), start_idx=i)
    assert len(segs) == 1 and segs[0].reason == "silence"
    # a speech window in the middle of silence resets the hangover
    g2 = default_gate()
    segs, i = feed(g2, [0.9] * 10)
    segs, i = feed(g2, [0.0] * 31 + [0.9] + [0.0] * 31, start_idx=i)
    assert segs == [] and g2.active


def test_max_len_split_without_trim():
    g = SpeechGate(GateConfig(threshold=0.5, min_silence_s=1.0, min_speech_s=0.3,
                              max_segment_s=1.0, pre_roll_s=0.5, post_roll_s=0.5))
    segs, i = feed(g, [0.9] * 100)
    assert len(segs) >= 1
    first = segs[0]
    assert first.reason == "max_len"
    assert len(first.audio) >= int(1.0 * SAMPLE_RATE)
    assert len(first.audio) == 32 * WINDOW  # first window count whose length >= 16000
    # after a max_len close the gate is idle again and re-opens after min_speech
    assert all(s.reason == "max_len" for s in segs)
    # segments are contiguous: next start = previous start + windows consumed (no pre-roll overlap
    # beyond what the ring kept)
    for a, b in zip(segs, segs[1:]):
        assert b.start_sample >= a.start_sample + len(a.audio) - 16 * WINDOW


def test_flush_closes_open_segment():
    g = default_gate()
    assert g.flush() == []
    feed(g, [0.9] * 15)
    assert g.active
    segs = g.flush()
    assert len(segs) == 1 and segs[0].reason == "flush" and not g.active
    assert len(segs[0].audio) == 15 * WINDOW  # nothing trimmed on flush
    assert g.flush() == []


def test_min_speech_zero_opens_on_first_window():
    g = SpeechGate(GateConfig(min_speech_s=0.0))
    feed(g, [0.9])
    assert g.active


# ---------------------------------------------------------------------------
# segment_audio / speech_seconds with a scripted VAD
# ---------------------------------------------------------------------------

def test_segment_audio_scripted_probs():
    # 100 windows: 20 silence, 30 speech, 50 silence
    probs = [0.0] * 20 + [0.95] * 30 + [0.0] * 50
    audio = np.arange(100 * WINDOW, dtype=np.float32) / 1e6
    vad = FakeVAD(probs)
    cfg = GateConfig(threshold=0.5, min_silence_s=1.0, min_speech_s=0.3, pre_roll_s=0.5, post_roll_s=0.5)
    segs = segment_audio(audio, vad, cfg)
    assert vad.resets == 1
    assert len(segs) == 1
    s = segs[0]
    assert s.reason == "silence"
    assert s.speech_s == pytest.approx(30 * WIN_S)
    assert s.start_sample == (20 + 10 - 16) * WINDOW
    # audio slice really comes from the input at start_sample
    np.testing.assert_array_equal(s.audio[:WINDOW], audio[s.start_sample:s.start_sample + WINDOW])


def test_segment_audio_tail_padding_and_flush():
    # speech until the very end, plus a partial last window -> flushed segment
    probs = [0.9] * 40
    audio = np.ones(30 * WINDOW + 100, dtype=np.float32) * 0.1
    segs = segment_audio(audio, FakeVAD(probs), GateConfig(min_speech_s=0.3, min_silence_s=1.0))
    assert len(segs) == 1 and segs[0].reason == "flush"
    # 30 full windows + 1 zero-padded tail window
    assert len(segs[0].audio) == 31 * WINDOW
    assert float(segs[0].audio[-1]) == 0.0


def test_segment_audio_no_speech_returns_empty():
    audio = np.zeros(50 * WINDOW, dtype=np.float32)
    assert segment_audio(audio, FakeVAD(default=0.0), GateConfig()) == []


def test_speech_seconds_counts_windows_at_threshold():
    probs = [0.5, 0.49, 0.9, 0.0, 1.0]
    audio = np.zeros(5 * WINDOW + 10, dtype=np.float32)
    assert speech_seconds(audio, FakeVAD(probs), 0.5) == pytest.approx(3 * WIN_S)
    assert speech_seconds(audio, FakeVAD(probs), 0.95) == pytest.approx(1 * WIN_S)


# ---------------------------------------------------------------------------
# Real Silero model
# ---------------------------------------------------------------------------

def test_silero_silence_prob_near_zero(silero):
    silero.reset()
    ps = [silero.prob(np.zeros(WINDOW, dtype=np.float32)) for _ in range(20)]
    assert max(ps) < 0.1
    assert speech_seconds(silence(2.0), silero) == 0.0
    assert segment_audio(silence(2.0), silero, GateConfig()) == []


def test_silero_probs_streaming_carry(silero):
    silero.reset()
    out = silero.probs(np.zeros(WINDOW + 100, dtype=np.float32))
    assert len(out) == 1 and len(silero._carry) == 100
    out = silero.probs(np.zeros(WINDOW - 100, dtype=np.float32))
    assert len(out) == 1 and len(silero._carry) == 0
    out = silero.probs(np.zeros(10, dtype=np.float32))
    assert out == [] and len(silero._carry) == 10
    silero.reset()
    assert len(silero._carry) == 0


def test_silero_outputs_valid_probabilities_on_noise(silero):
    silero.reset()
    ps = silero.probs(noise(1.0, amp=0.05))
    assert len(ps) == 31
    assert all(0.0 <= p <= 1.0 for p in ps)
