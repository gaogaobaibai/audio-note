"""scripts/bench/audiofx.py and the pure parts of build.py / story.py / common.py: synthetic acoustics, the fixed
story's integrity, profile configs without secrets, the ASR cache. No models, no network, no real audio."""
from __future__ import annotations

import importlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

BENCH = Path(__file__).resolve().parent.parent / "scripts" / "bench"
PKG = "audio_note_bench"


def bench(module: str):
    if PKG not in sys.modules:
        spec = importlib.util.spec_from_file_location(PKG, BENCH / "__init__.py", submodule_search_locations=[str(BENCH)])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[PKG] = pkg
        spec.loader.exec_module(pkg)
    return importlib.import_module(f"{PKG}.{module}")


fx = bench("audiofx")
story = bench("story")
build = bench("build")
common = bench("common")
SR = 16000


def tone(freq: float, seconds: float, amp: float = 0.3, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * math.pi * freq * t)).astype(np.float32)


# --- audiofx ---------------------------------------------------------------------------------------
def test_resample_keeps_duration_pitch_and_level():
    x = tone(440.0, 1.0, sr=24000)
    y = fx.resample(x, 24000, 16000)
    assert y.dtype == np.float32 and len(y) == 16000
    spec = np.abs(np.fft.rfft(y))
    assert int(np.argmax(spec)) == 440
    assert fx.rms_db(y) == pytest.approx(fx.rms_db(x), abs=0.1)
    assert len(fx.resample(x, 24000, 24000)) == len(x)
    assert len(fx.resample(np.zeros(0, np.float32), 24000, 16000)) == 0
    # content above the new Nyquist frequency is removed, not aliased
    hi = fx.resample(tone(10000.0, 1.0, sr=24000), 24000, 16000)
    assert fx.rms_db(hi) < -60


def test_active_level_matches_the_pipeline_definition():
    from audio_note.audio.loudness import speech_dbfs

    rng = np.random.default_rng(0)
    x = np.concatenate([np.zeros(8000, np.float32), (0.1 * rng.standard_normal(16000)).astype(np.float32), np.zeros(8000, np.float32)])
    assert fx.active_level_db(x) == pytest.approx(speech_dbfs(x), abs=1e-6)
    assert fx.active_level_db(np.zeros(100, np.float32)) == fx.SILENCE_DB
    assert fx.active_level_db(np.zeros(0, np.float32)) == fx.SILENCE_DB
    y = fx.set_active_level(x, -30.0)
    assert fx.active_level_db(y) == pytest.approx(-30.0, abs=0.01)
    assert fx.rms_db(fx.gain_db(x, -20.0)) == pytest.approx(fx.rms_db(x) - 20.0, abs=0.01)
    assert float(np.max(np.abs(fx.set_peak(x, 0.5)))) == pytest.approx(0.5, abs=1e-6)


def test_trim_and_fade():
    x = np.concatenate([np.zeros(4000, np.float32), tone(300.0, 0.5), np.zeros(6000, np.float32)])
    y = fx.trim_silence(x)
    assert 8000 <= len(y) <= 8000 + 2 * int(0.03 * SR) + 320
    assert len(fx.trim_silence(np.zeros(16000, np.float32))) == 16000          # nothing to anchor on: unchanged
    z = fx.fade(np.ones(1600, np.float32))
    assert z[0] == 0.0 and z[-1] == 0.0 and z[800] == 1.0


def test_reverb_ir_decays_and_far_field_hits_the_level():
    ir = fx.reverb_ir(rt60_s=0.4, drr_db=0.0, seed=1)
    assert ir[0] == 1.0
    tail = ir[int(0.004 * SR):]
    assert float(np.sum(tail.astype(np.float64) ** 2)) == pytest.approx(1.0, rel=1e-3)      # DRR 0 dB
    e_early = float(np.sum(tail[: int(0.1 * SR)].astype(np.float64) ** 2))
    e_late = float(np.sum(tail[int(0.3 * SR): int(0.4 * SR)].astype(np.float64) ** 2))
    assert 10 * math.log10(e_late / e_early) < -25                                # energy really falls off
    assert np.array_equal(fx.reverb_ir(seed=5), fx.reverb_ir(seed=5))
    assert not np.array_equal(fx.reverb_ir(seed=5), fx.reverb_ir(seed=6))

    rng = np.random.default_rng(1)
    dry = fx.set_active_level((rng.standard_normal(SR) * np.hanning(SR)).astype(np.float32), -18.0)
    far = fx.far_field(dry, level_db=-38.0, rt60_s=0.4, seed=2)
    assert len(far) > len(dry)                                                   # the tail rings on
    assert fx.active_level_db(far[: len(dry)]) == pytest.approx(-38.0, abs=0.05)
    assert fx.rms_db(far[len(dry): len(dry) + 1600]) > -80                        # ... audibly


def test_convolve_matches_numpy():
    rng = np.random.default_rng(2)
    a, b = rng.standard_normal(200), rng.standard_normal(37)
    assert np.allclose(fx.convolve(a, b), np.convolve(a, b), atol=1e-4)
    assert len(fx.convolve(np.zeros(0), b)) == 0


def test_noise_floor_level_colour_and_determinism():
    n = fx.noise_floor(SR * 2, -58.0, seed=3)
    assert fx.rms_db(n) == pytest.approx(-58.0, abs=0.05)
    assert np.array_equal(n, fx.noise_floor(SR * 2, -58.0, seed=3))
    spec = np.abs(np.fft.rfft(n.astype(np.float64))) ** 2
    assert spec[: len(spec) // 8].mean() > 5 * spec[-len(spec) // 8:].mean()       # low-pass coloured
    white = fx.noise_floor(SR, -40.0, seed=3, colour=0.0)
    assert fx.rms_db(white) == pytest.approx(-40.0, abs=0.05)
    assert len(fx.noise_floor(0, -40.0)) == 0


def test_mix_limit_and_channel_pairs():
    buf = np.zeros(10, np.float32)
    fx.mix_into(buf, np.ones(4, np.float32), 8)
    fx.mix_into(buf, np.ones(4, np.float32), 20)
    assert buf.tolist() == [0] * 8 + [1, 1]
    lim = fx.soft_limit(np.array([0.1, -0.5, 1.5, -3.0], np.float32))
    assert lim[0] == pytest.approx(0.1) and lim[1] == pytest.approx(-0.5)
    assert 0.78 < lim[2] < 0.98 and -0.98 < lim[3] < -0.78

    mic = np.concatenate([np.full(512, 0.1, np.float32), np.zeros(512, np.float32), np.full(100, 0.2, np.float32)])
    sysa = np.concatenate([np.zeros(512, np.float32), np.full(512, 0.3, np.float32)])
    pairs = fx.channel_pairs(mic, sysa)
    assert pairs.shape == (3, 2) and pairs.dtype == np.float32
    assert pairs[0].tolist() == pytest.approx([0.1, 0.0])
    assert pairs[1].tolist() == pytest.approx([0.0, 0.3])
    assert pairs[2, 0] == pytest.approx(0.2 * math.sqrt(100 / 512)) and pairs[2, 1] == 0.0
    # the pipeline reads these pairs back as "mostly system audio" for the second block
    from audio_note.audio.storage import encode_channel_rms
    from audio_note.speech.pipeline import channel_pairs, system_share

    back = channel_pairs(encode_channel_rms(pairs))
    assert system_share(back, 0.032, 0.032, 0.064) == pytest.approx(1.0)
    assert system_share(back, 0.032, 0.0, 0.032) == pytest.approx(0.0)


def test_stable_seed_is_stable():
    assert fx.stable_seed("a", 1, "x") == fx.stable_seed("a", 1, "x") == 2806722479
    assert fx.stable_seed("a", 1) != fx.stable_seed("a", 2)
    assert 0.0 <= fx.stable_unit("bench", 3) < 1.0


# --- the story --------------------------------------------------------------------------------------
def test_story_is_well_formed():
    assert story.validate() == []
    s = story.summary()
    assert s["sessions"] == 12 and s["lines"] >= 150
    assert set(s["by_condition"]) == {"near", "far", "overlap", "system"}
    assert min(s["by_condition"].values()) >= 12
    assert s["conflicts"] == 2 and s["todos_done_later"] >= 10
    assert set(story.CAST) == {"我", "周岚", "陈默", "阿梨", "老秦", "旁白"}
    assert story.enrolled_names() == ["我", "周岚", "陈默", "阿梨"]
    assert sum(1 for c in story.CAST.values() if c.get("self")) == 1


def test_story_is_deterministic_data_with_asr_style_numerals():
    assert story.as_json() == story.as_json()
    json.dumps(story.as_json(), ensure_ascii=False)                       # plain data
    import re

    for ln in story.all_lines():
        # Arabic digits only inside alphanumeric codes (T30, X7, X7S, P0)
        stripped = re.sub(r"[A-Za-z]+\d+[A-Za-z]*", "", ln["text"])
        assert not re.search(r"\d", stripped), ln["text"]
    # the narrator is only ever system audio and never enrolled
    assert all(ln["sys"] for ln in story.all_lines() if ln["speaker"] == story.NARRATOR)
    assert not story.CAST[story.NARRATOR]["enrolled"]


def test_story_terms_cover_names_terms_and_numbers():
    terms = story.term_list()
    groups = {t["group"] for t in terms}
    assert groups == {"name", "term", "number"}
    assert 20 <= len(story.LEXICON) <= 30
    textnorm = bench("textnorm")
    align = bench("align")
    lines = [textnorm.normalize(ln["text"]) for ln in story.all_lines()]
    for t in terms:
        norm = textnorm.normalize(t["term"])
        assert any(align.term_occurrences(ln, norm) for ln in lines), t["term"]
    # searched line by line: a figure that ends one line never fuses with the figure opening the next one
    ref, owner = align.concat(lines)
    st = align.term_stats(ref, align.align(ref, ref), [textnorm.normalize("十二万五")], owner)
    assert st["125000"] == {"n": 2, "wrong": 0}
    assert align.term_stats(ref, align.align(ref, ref), ["125000"])["125000"]["n"] == 0      # ... which it would without owners
    assert len(story.term_list(include_figures=False)) == len(story.LEXICON)


def test_story_memory_truth_links():
    items = {m["id"]: m for m in story.memory_items()}
    done = [m for m in items.values() if m["kind"] == "todo" and m.get("done_in")]
    still_open = [m for m in items.values() if m["kind"] == "todo" and not m.get("done_in")]
    assert done and still_open
    for c in story.CONFLICTS:
        a, b = items[c["a"]], items[c["b"]]
        assert a["day"] != b["day"] and a["kind"] == b["kind"] == "fact"
    superseded = [m for m in items.values() if m.get("superseded_in")]
    assert len(superseded) == 2 and all(m["kind"] == "decision" for m in superseded)
    kinds = {s["kind"] for s in story.sessions()}
    assert kinds == {"meeting", "conversation", "media"}


# --- build: pure helpers ----------------------------------------------------------------------------
def test_plan_lines_per_variant():
    sess = story.session("d1s3")                                  # the remote meeting
    mixed = build.plan_lines(sess, "mixed")
    assert {ln["condition"] for ln in mixed} == {"near", "system"}
    clean = build.plan_lines(sess, "clean")
    assert {ln["condition"] for ln in clean} == {"near"}
    far = build.plan_lines(story.session("d1s2"), "far")
    assert {ln["condition"] for ln in far} == {"far"} and all(ln["overlap"] == 0 for ln in far)
    ov = build.plan_lines(story.session("d1s2"), "mixed")
    assert [ln["condition"] for ln in ov[3:5]] == ["overlap", "overlap"]      # both sides of an overlap


def test_gaps_and_tts_keys_are_deterministic():
    assert build.default_gap(0, 1) == build.default_gap(0, 1)
    gaps = [build.default_gap(s, k) for s in range(12) for k in range(1, 14)]
    assert any(g >= 2.0 for g in gaps) and min(gaps) >= 0.4            # some pauses close a segment
    assert build.tts_key("我", "你好") == build.tts_key("我", "你好")
    assert build.tts_key("我", "你好") != build.tts_key("周岚", "你好")
    assert build.tts_key("老秦", "你好", "ref1") != build.tts_key("老秦", "你好", "ref2")
    jobs = build.tts_jobs()
    assert len(jobs) == len(set(jobs)) and len(jobs) < 400            # inside the TTS budget
    assert ("旁白", story.ENROLL_TEXT[0]) not in jobs
    assert build.plausible_duration("好，散会。", 1.2)
    assert not build.plausible_duration("好，散会。", 12.0)
    assert not build.plausible_duration("今天主要过一下青禾App灰度发布的安排。", 0.5)


def test_profile_config_has_no_key_and_absolute_paths(tmp_path):
    p = build.write_profile_config(tmp_path / "profiles" / "x", mimo={"base_url": "https://example.invalid/v1"},
                                   models_dir=tmp_path / "models", threads=2,
                                   overrides=build._parse_overrides(["speakers.identify_threshold=0.5", "correction.mode=session"]))
    conf = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert "api_key" not in conf.get("mimo", {})
    assert Path(conf["paths"]["data_dir"]).is_absolute() and Path(conf["paths"]["models_dir"]).is_absolute()
    assert conf["speakers"] == {"num_threads": 2, "identify_threshold": 0.5}
    assert conf["correction"]["mode"] == "session" and conf["correction"]["enabled"] is True
    assert conf["push"]["channels"] == [] and conf["ui"]["enabled"] is False
    with pytest.raises(ValueError):
        build.write_profile_config(tmp_path / "profiles" / "y", mimo={"api_key": "sk-secret"})
    with pytest.raises(SystemExit):
        build._parse_overrides(["nonsense"])


def test_safe_rmtree_only_touches_bench_profiles(tmp_path):
    root = tmp_path / "bench"
    good = root / "profiles" / "old"
    good.mkdir(parents=True)
    other = tmp_path / "elsewhere" / "old"
    other.mkdir(parents=True)
    with pytest.raises(RuntimeError):
        build._safe_rmtree(other, root)
    with pytest.raises(RuntimeError):
        build._safe_rmtree(root / "profiles", root)
    build._safe_rmtree(good, root)
    assert not good.exists() and other.exists()


class _FakeMimo:
    def __init__(self):
        self.calls = []
        self.usage = {"calls": 0}
        self.cfg = type("C", (), {"mimo": type("M", (), {"asr_model": "fake-asr"})()})()

    def transcribe_bytes(self, wav, mime="audio/wav", language="auto", model=None, **kw):
        self.calls.append((len(wav), language, model, tuple(sorted(kw.items()))))
        return f"text{len(self.calls)}", 1.5

    def scoped(self, purpose):
        return self

    def chat(self, messages, **kw):
        return "chat"


def test_caching_client_caches_by_exact_request(tmp_path):
    inner = _FakeMimo()
    c = build.CachingClient(inner, tmp_path / "asr")
    wav = b"RIFF" + b"\0" * 40 + b"\1\0" * 16000                     # 1 s of 16 kHz PCM16
    assert c.transcribe_bytes(wav, "audio/wav", "auto") == ("text1", 1.5)
    assert c.transcribe_bytes(wav, "audio/wav", "auto") == ("text1", 1.5)            # served from disk
    assert len(inner.calls) == 1
    assert c.transcribe_bytes(wav, "audio/wav", "zh") == ("text2", 1.5)              # another language: another request
    assert c.transcribe_bytes(wav, "audio/wav", "auto", prompt="可能出现的专有名词：青禾") == ("text3", 1.5)
    assert c.transcribe_bytes(wav, "audio/wav", "auto", prompt="可能出现的专有名词：青禾") == ("text3", 1.5)
    assert c.transcribe_bytes(wav, "audio/wav", "auto", prompt=None) == ("text1", 1.5)   # None == not given
    assert len(inner.calls) == 3
    st = c.bench_stats
    assert st["asr_requests"] == 6 and st["asr_cache_hits"] == 3
    assert st["asr_live_audio_s"] == pytest.approx(3.0, abs=0.01) and st["asr_billed_s"] == pytest.approx(4.5)
    # everything else is the wrapped client; attributes set by the application land on it
    assert c.chat([]) == "chat" and c.usage == {"calls": 0}
    c.on_usage = "hook"
    assert inner.on_usage == "hook"
    assert isinstance(c.scoped("asr"), build.CachingClient)
    # a second proxy on the same directory starts warm
    c2 = build.CachingClient(_FakeMimo(), tmp_path / "asr")
    assert c2.transcribe_bytes(wav, "audio/wav", "auto") == ("text1", 1.5)


def test_caching_client_enforces_the_live_budget(tmp_path):
    c = build.CachingClient(_FakeMimo(), tmp_path / "asr", max_live_s=1.5)
    one = b"RIFF" + b"\0" * 40 + b"\1\0" * 16000
    two = b"RIFF" + b"\0" * 40 + b"\2\0" * 16000
    c.transcribe_bytes(one)
    with pytest.raises(RuntimeError):
        c.transcribe_bytes(two)
    assert c.transcribe_bytes(one)[0] == "text1"                       # cached answers stay free


def test_stub_asr_returns_text_for_real_audio_only():
    asr = build.StubASR()
    assert asr.transcribe(np.zeros(SR, np.float32)).text == "测试"
    assert asr.transcribe(np.zeros(100, np.float32)).text == ""
    assert asr.transcribe(np.zeros(SR, np.float32), prompt="x").seconds == 0.0


# --- common -----------------------------------------------------------------------------------------
def test_stage_timer_is_exclusive_and_wraps_methods():
    t = common.StageTimer()
    with t.stage("outer"):
        time.sleep(0.03)
        with t.stage("inner"):
            time.sleep(0.05)
    snap = t.snapshot()
    assert snap["inner"]["n"] == 1 and snap["inner"]["wall_s"] >= 0.045
    assert 0.02 <= snap["outer"]["wall_s"] < 0.05                      # the inner 50 ms are not counted twice

    class Engine:
        def diarize(self, audio):
            return [audio]

        def embed(self, audio, scale=1):
            return audio * scale

        def identify_turns(self, audio, turns, db, segment_id=None):
            return [self.embed(audio)]

    e = Engine()
    wrapped = common.time_engine(e, t)
    assert set(wrapped) == {"diarize", "embed", "identify_turns"}
    assert e.identify_turns(2, [], None, segment_id=3) == [2] and e.diarize(1) == [1]
    snap = t.snapshot()
    assert snap["identify"]["n"] == 1 and snap["embed"]["n"] == 1 and snap["diarize"]["n"] == 1
    import inspect

    assert "segment_id" in inspect.signature(e.identify_turns).parameters     # signature probing still works


def test_call_supported_drops_unknown_keywords():
    def old(a, b=1):
        return (a, b)

    def new(a, b=1, **kw):
        return (a, b, kw)

    assert common.call_supported(old, 1, b=2, channel_rms="x") == (1, 2)
    assert common.call_supported(new, 1, b=2, channel_rms="x") == (1, 2, {"channel_rms": "x"})


def test_load_key_never_returns_the_key(tmp_path, monkeypatch):
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    monkeypatch.delenv("BENCH_KEY_CONFIG", raising=False)
    f = tmp_path / "c.yaml"
    f.write_text("mimo:\n  api_key: sk-test-123\n  base_url: https://example.invalid/v1\n", encoding="utf-8")
    extra = common.load_key(f)
    assert extra == {"base_url": "https://example.invalid/v1"}
    import os

    assert os.environ["MIMO_API_KEY"] == "sk-test-123"
    assert common.scrub("Bearer sk-test-123 failed") == "Bearer *** failed"
    monkeypatch.delenv("MIMO_API_KEY")
    with pytest.raises(SystemExit):
        common.load_key(None)


def test_open_ro_refuses_writes_and_finds_schema(tmp_path):
    import sqlite3

    p = tmp_path / "x.sqlite3"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE t (a INTEGER, b TEXT)")
    c.commit()
    c.close()
    ro = common.open_ro(p)
    assert common.table_exists(ro, "t") and not common.table_exists(ro, "nope")
    assert common.columns(ro, "t") == {"a", "b"} and common.columns(ro, "nope") == set()
    with pytest.raises(sqlite3.OperationalError):
        ro.execute("INSERT INTO t VALUES (1, 'x')")
    ro.close()
    with pytest.raises(FileNotFoundError):
        common.open_ro(tmp_path / "missing.sqlite3")


def test_use_project_prefers_pythonpath_then_repo(tmp_path, monkeypatch):
    monkeypatch.delenv("AUDIO_NOTE_SRC", raising=False)
    fake = tmp_path / "old"
    (fake / "audio_note").mkdir(parents=True)
    (fake / "audio_note" / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(fake))
    assert common._pythonpath_src() == fake
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "nothing"))
    assert common._pythonpath_src() is None
    # the package under test in this process is the repository's own
    assert common.use_project() == common.REPO_ROOT.resolve()
    info = common.code_info()
    assert Path(info["src"]) == common.REPO_ROOT.resolve()


# --- build.render on synthetic clips (no TTS) -----------------------------------------------------------
def _fake_tts_cache(root: Path) -> None:
    """One short synthetic 'voice' clip per TTS job, so render() can run without the network."""
    import soundfile as sf

    (root / "tts_cache").mkdir(parents=True)
    (root / "refs").mkdir()
    for name in story.CAST:
        if build.voice_spec(name)[0] == "design":
            sf.write(str(root / "refs" / f"{name}.wav"), tone(200.0, 0.2), SR, subtype="PCM_16")
    for k, (speaker, text) in enumerate(build.tts_jobs()):
        seconds = 0.4 + 0.02 * len(text)
        rng = np.random.default_rng(k)
        clip = (0.2 * rng.standard_normal(int(seconds * SR)) * np.hanning(int(seconds * SR))).astype(np.float32)
        clip = np.concatenate([np.zeros(1600, np.float32), clip, np.zeros(1600, np.float32)])
        sf.write(str(build.clip_path(root, speaker, text)), clip, SR, subtype="PCM_16")


def test_render_writes_sessions_truth_and_channel_levels(tmp_path):
    import soundfile as sf

    root = tmp_path / "bench"
    _fake_tts_cache(root)
    truth = build.render(root, "mixed")
    assert truth == json.loads((root / "sessions" / "mixed" / "truth.json").read_text(encoding="utf-8"))
    assert len(truth["sessions"]) == 12 and len(truth["lines"]) == len(story.all_lines())
    assert {ln["condition"] for ln in truth["lines"]} == {"near", "far", "overlap", "system"}
    assert set(truth["lines"][0]) >= {"speaker", "text", "t0", "t1", "condition", "session"}
    assert truth["speakers"]["我"] == {"enrolled": True, "self": True, "system_only": False, "voice": "preset"}
    assert truth["enroll"]["我"] == ["我_1.wav", "我_2.wav"] and "旁白" not in truth["enroll"]
    assert (root / "enroll" / "老秦_2.wav").exists()
    assert len(truth["memory"]["items"]) == len(story.memory_items()) and len(truth["memory"]["conflicts"]) == 2
    assert truth["lexicon"] == story.LEXICON
    by_id = {s["id"]: s for s in truth["sessions"]}
    for s in truth["sessions"]:
        audio, sr = sf.read(str(root / "sessions" / "mixed" / s["file"]), dtype="float32")
        assert sr == SR and len(audio) / SR == pytest.approx(s["duration_s"], abs=0.01)
        assert float(np.max(np.abs(audio))) <= 0.99
        prev = None
        for ln in s["lines"]:
            assert 0.5 <= ln["t0"] < ln["t1"] <= s["duration_s"]
            if prev is not None and ln["condition"] != "overlap":
                assert ln["t0"] >= prev["t1"] + 0.39          # a real pause between lines that do not overlap
            prev = ln
        assert s["has_system"] == any(ln["channel"] == "system" for ln in s["lines"])
        assert (s["channels_file"] is not None) == s["has_system"]
    # overlap: the second line starts before the first one ends
    d1s2 = by_id["d1s2"]["lines"]
    assert d1s2[4]["t0"] < d1s2[3]["t1"] and d1s2[4]["t0"] > d1s2[3]["t0"]
    # levels: near around the target, far about 20 dB below
    near = [ln["level_db"] for ln in truth["lines"] if ln["condition"] == "near"]
    far = [ln["level_db"] for ln in truth["lines"] if ln["condition"] == "far"]
    assert -19.6 <= min(near) and max(near) <= -16.4
    assert -39.6 <= min(far) and max(far) <= -36.4
    # channel pairs: system lines are system-dominated, microphone lines are not
    s = by_id["d1s3"]
    pairs = np.load(str(root / "sessions" / "mixed" / s["channels_file"]))
    assert pairs.shape[1] == 2 and pairs.shape[0] == int(np.ceil(s["duration_s"] * SR / fx.BLOCK))
    for ln in s["lines"]:
        blk = pairs[int(ln["t0"] * SR) // fx.BLOCK + 1: int(ln["t1"] * SR) // fx.BLOCK]
        share = float((blk[:, 1] ** 2).sum() / ((blk ** 2).sum()))
        assert (share > 0.9) if ln["channel"] == "system" else (share < 0.1), (ln["i"], share)
    # deterministic: rendering again gives the same audio and the same truth
    first = (root / "sessions" / "mixed" / by_id["d2s4"]["file"]).read_bytes()
    assert build.render(root, "mixed") == truth
    assert (root / "sessions" / "mixed" / by_id["d2s4"]["file"]).read_bytes() == first

    clean = build.render(root, "clean")
    assert {ln["condition"] for ln in clean["lines"]} == {"near"} and not any(s["has_system"] for s in clean["sessions"])
    far_v = build.render(root, "far")
    assert {ln["condition"] for ln in far_v["lines"]} == {"far", "system"}
    with pytest.raises(ValueError):
        build.render(root, "nope")
    with pytest.raises(FileNotFoundError):
        build.load_clip(root, "我", "这句话没有合成过")
