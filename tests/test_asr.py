"""Tests for audio_note.speech.asr: chunk splitting, text joining, ASR.transcribe with a fake MiMo client."""
from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from audio_note.audio.vad import SAMPLE_RATE, WINDOW
from audio_note.speech.asr import ASR, ASRResult, _join, split_at_quietest, wav_bytes

from .conftest import FakeVAD, noise, silence


class FakeClient:
    """Mimics MimoClient.transcribe_bytes; records every call."""

    def __init__(self, text_fn=None, seconds: float = 1.5):
        self.calls: list[tuple[bytes, str, str]] = []
        self.text_fn = text_fn or (lambda i, audio: f"第{i}段")
        self.seconds = seconds

    def transcribe_bytes(self, data: bytes, mime: str = "audio/wav", language: str = "auto", model=None):
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
        assert sr == SAMPLE_RATE and audio.ndim == 1
        self.calls.append((data, mime, language))
        return self.text_fn(len(self.calls) - 1, audio), self.seconds

    # the other MimoClient methods exist so duck typing elsewhere does not break
    def chat(self, *a, **k):
        raise AssertionError("chat() must not be called by ASR")

    def text(self, *a, **k):
        raise AssertionError("text() must not be called by ASR")

    def json(self, *a, **k):
        raise AssertionError("json() must not be called by ASR")


def test_wav_bytes_roundtrip():
    a = noise(0.5, amp=0.2)
    b = wav_bytes(a)
    assert b[:4] == b"RIFF"
    back, sr = sf.read(io.BytesIO(b), dtype="float32")
    assert sr == SAMPLE_RATE and len(back) == len(a)
    np.testing.assert_allclose(back, a, atol=1e-4)


def test_split_short_audio_untouched():
    a = noise(3.0)
    vad = FakeVAD()
    parts = split_at_quietest(a, vad, max_s=10.0)
    assert len(parts) == 1 and parts[0] is a
    assert vad.calls == 0  # no VAD run when no split needed


def test_split_lengths_bounded_and_total_preserved():
    a = noise(25.0, amp=0.05)
    parts = split_at_quietest(a, FakeVAD(default=0.5), max_s=10.0, min_s=5.0)
    assert len(parts) >= 3
    assert sum(len(p) for p in parts) == len(a)
    assert all(len(p) <= int(10.0 * SAMPLE_RATE) for p in parts)
    # min_s is only a soft bound (a short LEFT part is kept when nothing precedes it), so we do
    # not assert it; we do require that no part is absurdly tiny
    assert all(len(p) >= SAMPLE_RATE for p in parts)
    np.testing.assert_array_equal(np.concatenate(parts), a)


def test_split_happens_at_quietest_window():
    a = noise(20.0, amp=0.05)
    n = len(a) // WINDOW
    probs = [0.9] * n
    k = int(n * 0.6)  # inside the middle 60 %
    probs[k] = 0.01
    parts = split_at_quietest(a, FakeVAD(probs), max_s=15.0, min_s=2.0)
    assert len(parts) == 2
    assert len(parts[0]) == k * WINDOW + WINDOW // 2


def test_split_ignores_quiet_windows_outside_middle_band():
    a = noise(20.0, amp=0.05)
    n = len(a) // WINDOW
    probs = [0.9] * n
    probs[2] = 0.0  # in the first 20 %: must be ignored
    probs[n // 2] = 0.3
    parts = split_at_quietest(a, FakeVAD(probs), max_s=15.0, min_s=2.0)
    assert len(parts) == 2
    assert len(parts[0]) == (n // 2) * WINDOW + WINDOW // 2


@pytest.mark.parametrize("texts,expected", [
    ([], ""),
    (["你好"], "你好"),
    (["你好", "世界"], "你好，世界"),
    (["你好。", "世界"], "你好。世界"),
    (["你好，", "世界"], "你好，世界"),
    (["Hello?", "World"], "Hello?World"),
    (["a.", "b!", "c"], "a.b!c"),
    (["", "  ", "x", "", "y"], "x，y"),
    (["  你好  ", "世界…", "再见"], "你好，世界…再见"),
])
def test_join(texts, expected):
    assert _join(texts) == expected


def make_asr(cfg_factory, client, max_chunk_s: float = 10.0, language: str = "zh", vad=None) -> ASR:
    cfg = cfg_factory({"mimo": {"asr_max_chunk_s": max_chunk_s, "asr_language": language}})
    return ASR(cfg, client, vad=vad or FakeVAD(default=0.5))


def test_transcribe_too_short_returns_empty_without_calls(cfg_factory):
    client = FakeClient()
    asr = make_asr(cfg_factory, client)
    r = asr.transcribe(noise(0.2))
    assert r == ASRResult("", 0.0, 0)
    assert client.calls == []


def test_transcribe_single_chunk(cfg_factory):
    client = FakeClient(seconds=2.5)
    asr = make_asr(cfg_factory, client, language="zh")
    r = asr.transcribe(noise(2.0, amp=0.05))
    assert r.text == "第0段" and r.seconds == 2.5 and r.parts == 1
    assert len(client.calls) == 1
    data, mime, lang = client.calls[0]
    assert mime == "audio/wav" and lang == "zh"
    back, _ = sf.read(io.BytesIO(data), dtype="float32")
    assert len(back) == 2 * SAMPLE_RATE


def test_transcribe_splits_long_audio_and_joins(cfg_factory):
    texts = ["今天天气不错", "我们去开会吧。", "好的"]
    client = FakeClient(text_fn=lambda i, a: texts[i % len(texts)], seconds=1.0)
    asr = make_asr(cfg_factory, client, max_chunk_s=10.0)
    r = asr.transcribe(noise(25.0, amp=0.05))
    assert r.parts == len(client.calls) >= 3
    assert r.seconds == pytest.approx(1.0 * r.parts)
    assert r.text.startswith("今天天气不错，我们去开会吧。好的")
    assert "，，" not in r.text and "。，" not in r.text
    # every API call stays within the chunk limit and nothing is lost
    lens = [len(sf.read(io.BytesIO(d), dtype="float32")[0]) for d, _, _ in client.calls]
    assert all(n <= 10 * SAMPLE_RATE for n in lens)
    assert sum(lens) == 25 * SAMPLE_RATE


def test_transcribe_split_exactly_two_parts(cfg_factory):
    # 12 s, max 10 s, one quiet window at 50 % -> exactly two 6 s chunks
    a = noise(12.0, amp=0.05)
    n = len(a) // WINDOW
    probs = [0.9] * n
    probs[n // 2] = 0.0
    client = FakeClient(text_fn=lambda i, a: ["前半", "后半。"][i], seconds=1.0)
    asr = make_asr(cfg_factory, client, max_chunk_s=10.0, vad=FakeVAD(probs))
    r = asr.transcribe(a)
    assert r.parts == 2 and r.seconds == 2.0
    assert r.text == "前半，后半。"


def test_transcribe_skips_empty_part_text(cfg_factory):
    client = FakeClient(text_fn=lambda i, a: "" if i == 0 else "只有这句")
    asr = make_asr(cfg_factory, client, max_chunk_s=10.0)
    r = asr.transcribe(noise(15.0, amp=0.05))
    assert r.parts >= 2
    assert r.text.startswith("只有这句")
    assert r.text == "，".join(["只有这句"] * (r.parts - 1))


def test_language_default_auto(cfg_factory):
    client = FakeClient()
    cfg = cfg_factory({"mimo": {"asr_language": ""}})
    asr = ASR(cfg, client, vad=FakeVAD())
    assert asr.language == "auto"
    asr.transcribe(noise(1.0))
    assert client.calls[0][2] == "auto"


def test_transcribe_padded_max_chunk_is_one_request(cfg_factory):
    """build_chunks caps the speech span at max_chunk_s and pads 0.15 s per side: that must still be
    a single ASR call, while clearly longer audio is split into <= max_chunk_s pieces."""
    client = FakeClient()
    vad = FakeVAD(default=0.5)
    asr = make_asr(cfg_factory, client, max_chunk_s=10.0, vad=vad)
    r = asr.transcribe(noise(10.3, amp=0.05))
    assert r.parts == 1 and len(client.calls) == 1 and vad.calls == 0
    r = asr.transcribe(noise(11.0, amp=0.05))
    assert r.parts == 2
    lens = [len(sf.read(io.BytesIO(d), dtype="float32")[0]) for d, _, _ in client.calls[1:]]
    assert all(n <= 10 * SAMPLE_RATE for n in lens) and sum(lens) == 11 * SAMPLE_RATE


def test_transcribe_with_real_vad_split(cfg_factory, silero):
    client = FakeClient()
    # NB: max_chunk_s must stay well above split_at_quietest's default min_s (5 s), otherwise
    # short right-hand parts are merged back and the chunk limit is not honoured
    cfg = cfg_factory({"mimo": {"asr_max_chunk_s": 12.0}})
    asr = ASR(cfg, client, vad=silero)
    r = asr.transcribe(noise(30.0, amp=0.05))
    assert r.parts >= 3 and len(client.calls) == r.parts
    lens = [len(sf.read(io.BytesIO(d), dtype="float32")[0]) for d, _, _ in client.calls]
    assert all(n <= 12 * SAMPLE_RATE for n in lens) and sum(lens) == 30 * SAMPLE_RATE
