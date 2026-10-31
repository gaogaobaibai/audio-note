"""System-audio (WASAPI loopback) capture: the stream resampler, the microphone + loopback mix,
graceful fallback when loopback is unavailable, config / status / API plumbing. No real audio
devices are touched: the microphone and loopback captures are fakes."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from audio_note.audio import capture, loopback
from audio_note.audio.loopback import StreamResampler, _resample, resolve_loopback
from audio_note.audio.capture import MixedCapture, open_capture


# --- resampler ------------------------------------------------------------------------------------

@pytest.mark.parametrize("sr_in,chunk", [(48000, 1024), (44100, 1000), (48000, 7), (16000, 512)])
def test_stream_resampler_matches_one_shot_resample(sr_in: int, chunk: int):
    t = np.arange(sr_in * 2) / sr_in
    x = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    r = StreamResampler(sr_in, 16000)
    y = np.concatenate([r.process(x[i:i + chunk]) for i in range(0, len(x), chunk)])
    ref = _resample(x, sr_in, 16000)
    n = min(len(y), len(ref))
    assert abs(len(y) - len(ref)) <= 1
    assert float(np.abs(y[:n] - ref[:n]).max()) < 1e-5          # no phase jump at chunk boundaries


def test_stream_resampler_handles_tiny_chunks():
    r = StreamResampler(48000, 16000)
    out = [r.process(np.full(1, 0.5, dtype=np.float32)) for _ in range(10)]
    assert sum(len(o) for o in out) == 3 or sum(len(o) for o in out) == 4
    assert all(abs(float(v) - 0.5) < 1e-6 for o in out for v in o)


# --- mixing ---------------------------------------------------------------------------------------

class FakeMic:
    overflows = 0
    dropped = 0
    last_callback_at = 1.0
    last_block_wall = 2.0

    def __init__(self, blocks: list[np.ndarray]):
        self.blocks = list(blocks)
        self.entered = self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *a: Any):
        self.exited = True

    def is_active(self) -> bool:
        return True

    def read(self, timeout: float = 1.0):
        return self.blocks.pop(0) if self.blocks else None


class FakeLoop:
    dropped = 0

    def __init__(self, chunks: list[np.ndarray], fail: bool = False):
        self.chunks = list(chunks)
        self.fail = fail
        self.device = SimpleNamespace(name="spk")
        self.exited = False

    def __enter__(self):
        if self.fail:
            raise RuntimeError("no loopback here")
        return self

    def __exit__(self, *a: Any):
        self.exited = True

    def read_available(self) -> np.ndarray:
        return self.chunks.pop(0) if self.chunks else np.zeros(0, dtype=np.float32)


def blk(v: float, n: int = 512) -> np.ndarray:
    return np.full(n, v, dtype=np.float32)


def test_mixed_capture_adds_system_audio_with_gain_and_clips():
    mic = FakeMic([blk(0.1), blk(0.1), blk(0.1), blk(0.1)])
    loop = FakeLoop([blk(0.2, 300), np.zeros(0, dtype=np.float32), blk(0.6, 1000)])
    with MixedCapture(mic, loop, system_gain=2.0) as m:
        b1 = m.read()
        assert np.allclose(b1[:300], 0.5) and np.allclose(b1[300:], 0.1)   # 300 samples mixed, rest mic only
        b2 = m.read()
        assert np.allclose(b2, 0.1)                                        # nothing playing -> mic only
        b3 = m.read()
        assert np.allclose(b3, 1.0)                                        # 0.1 + 0.6*2 clipped to 1
        b4 = m.read()
        assert np.allclose(b4[:488], 1.0) and np.allclose(b4[488:], 0.1) and len(m._buf) == 0   # leftover 488 carried over
        assert m.system_samples == 300 + 512 + 488
        assert m.system_active and m.system_name == "spk" and m.system_error is None
        assert m.read() is None
    assert mic.exited and loop.exited
    assert m.last_callback_at == 1.0 and m.last_block_wall == 2.0 and m.overflows == 0 and m.dropped == 0


def test_mixed_capture_caps_a_runaway_loopback_backlog():
    mic = FakeMic([blk(0.0), blk(0.0)])
    loop = FakeLoop([blk(0.3, 16000 * 5)])          # 5 s arrived at once (mic stalled / clock drift)
    with MixedCapture(mic, loop, max_lead_s=2.0) as m:
        m.read()
        assert m.lead_drops == 1 and len(m._buf) <= 16000      # trimmed to half of max_lead minus one block
        m.read()
        assert m.lead_drops == 1


def test_mixed_capture_falls_back_to_microphone_when_loopback_fails():
    mic = FakeMic([blk(0.2)])
    with MixedCapture(mic, FakeLoop([], fail=True)) as m:
        assert not m.system_active and "no loopback" in (m.system_error or "")
        assert np.allclose(m.read(), 0.2)
    assert mic.exited


def test_open_capture_without_system_audio_is_plain_mic(cfg):
    cap = open_capture(cfg)
    assert isinstance(cap, capture.MicCapture)


def test_open_capture_with_system_audio_reports_missing_backend(cfg_factory, monkeypatch: pytest.MonkeyPatch):
    cfg = cfg_factory({"audio": {"system_audio": True}})
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "pyaudiowpatch":
            raise ImportError("no module")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    cap = open_capture(cfg)
    assert isinstance(cap, MixedCapture) and cap.system is None and "no module" in (cap.system_error or "")


def test_open_capture_with_system_audio_builds_loopback(cfg_factory, monkeypatch: pytest.MonkeyPatch):
    cfg = cfg_factory({"audio": {"system_audio": True, "system_device": "Speaker", "system_gain": 1.5}})
    made: dict[str, Any] = {}

    class FakeLB:
        def __init__(self, device=None, gain=1.0):
            made.update(device=device, gain=gain)

    monkeypatch.setattr(loopback, "LoopbackCapture", FakeLB)
    cap = open_capture(cfg)
    assert isinstance(cap, MixedCapture) and isinstance(cap.system, FakeLB)
    assert made == {"device": "Speaker", "gain": 1.5} and cap.system_gain == 1.5


# --- device resolution ------------------------------------------------------------------------------

def _devs():
    return [loopback.LoopbackDevice(13, "扬声器 (Realtek Speaker)", 2, 48000.0, True),
            loopback.LoopbackDevice(14, "BenQ RD280UG", 2, 48000.0, False)]


def test_resolve_loopback_by_default_index_and_name():
    assert resolve_loopback(None, _devs()).index == 13
    assert resolve_loopback(14, _devs()).name == "BenQ RD280UG"
    assert resolve_loopback("14", _devs()).index == 14
    assert resolve_loopback("benq", _devs()).index == 14
    with pytest.raises(RuntimeError):
        resolve_loopback("nope", _devs())
    with pytest.raises(RuntimeError):
        resolve_loopback(99, _devs())
    with pytest.raises(RuntimeError):
        resolve_loopback(None, [])


def test_list_loopback_devices_is_empty_without_backend(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(loopback, "_pa", lambda: (_ for _ in ()).throw(ImportError("x")))
    assert loopback.list_loopback_devices() == [] and loopback.available() is False


# --- AGC ------------------------------------------------------------------------------------------

def test_loopback_agc_lifts_quiet_audio(monkeypatch: pytest.MonkeyPatch):
    lb = loopback.LoopbackCapture(None, 1.0)
    lb.sr, lb.channels = 48000, 2
    lb._resampler = StreamResampler(48000, 16000)
    quiet = (np.random.default_rng(0).standard_normal(1024 * 2) * 0.004).astype(np.float32)   # ~2 % volume
    lb._cb(quiet.tobytes(), 1024, None, 0)
    chunk = lb.read_available()
    assert 1.0 < lb.agc_gain <= loopback.AGC_MAX_GAIN
    assert 0.2 < float(np.abs(chunk).max()) <= 1.0
    # digital silence does not change the gain
    g = lb.agc_gain
    lb._cb(np.zeros(2048, dtype=np.float32).tobytes(), 1024, None, 0)
    assert lb.agc_gain == g


# --- runtime status / api plumbing -------------------------------------------------------------------

def test_status_reports_system_audio_fields(cfg_factory):
    from audio_note.runtime import Runtime

    cfg = cfg_factory({"audio": {"system_audio": True, "system_device": 13}})
    rt = Runtime(cfg)
    try:
        s = rt.status()["service"]
        assert s["system_audio"] is True and s["system_device"] == 13 and s["system_active"] is False and s["system_error"] is None
        rt._rec_ref["recorder"] = SimpleNamespace(is_speaking=False, segments_written=0, speech_total_s=0.0, started_at=None,
                                                  capture=SimpleNamespace(system_active=True, system_name="spk", system_error=None))
        s = rt.status()["service"]
        assert s["system_active"] is True and s["system_name"] == "spk"
    finally:
        rt.shutdown()


def test_api_system_devices_and_config_toggle(cfg_factory, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    from audio_note.runtime import Runtime
    from audio_note.ui.app import create_app

    cfg = cfg_factory({"memory": {"retrieval": "fts", "embedding_model": "hash"}})
    monkeypatch.setattr(loopback, "available", lambda: True)
    monkeypatch.setattr(loopback, "list_loopback_devices", _devs)
    rt = Runtime(cfg)
    try:
        with TestClient(create_app(rt), raise_server_exceptions=False) as c:
            j = c.get("/api/devices/system").json()
            assert j["available"] and [d["index"] for d in j["devices"]] == [13, 14] and j["devices"][0]["is_default"]
            r = c.put("/api/config", json={"audio": {"system_audio": True, "system_device": "13", "system_gain": "2"}})
            assert r.status_code == 200 and sorted(r.json()["saved"]) == ["audio.system_audio", "audio.system_device", "audio.system_gain"]
            assert rt.cfg.audio.system_audio is True and rt.cfg.audio.system_device == 13 and rt.cfg.audio.system_gain == 2.0
            assert c.get("/api/status").json()["service"]["system_audio"] is True
            assert c.put("/api/config", json={"audio": {"system_gain": 0}}).status_code == 400
            assert c.put("/api/config", json={"audio": {"system_device": [1]}}).status_code == 400
    finally:
        rt.shutdown()
