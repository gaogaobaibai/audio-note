"""Tests for audio_note.speech.speakers: embeddings (real CAM++ model), match() logic, short-clip diarize, enrollment."""
from __future__ import annotations

import numpy as np
import pytest

from audio_note.audio.vad import SAMPLE_RATE
from audio_note.speech.speakers import SpeakerEngine, Turn

from .conftest import chirp, noise, silence


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_turn_duration():
    t = Turn(1.5, 4.0, 2)
    assert t.duration == pytest.approx(2.5)
    assert t.speaker_id is None and t.speaker_name is None and t.score is None


# ---------------------------------------------------------------------------
# match(): pure numpy, no model needed
# ---------------------------------------------------------------------------

def test_match_empty_enrolled():
    assert SpeakerEngine.match(unit(np.ones(192)), [], 0.45) == (None, None, None)


def test_match_picks_best_speaker_above_threshold():
    rng = np.random.default_rng(0)
    e1, e2 = unit(rng.standard_normal(192)), unit(rng.standard_normal(192))
    enrolled = [(1, "张三", e1), (2, "李四", e2)]
    probe = unit(e1 + 0.01 * rng.standard_normal(192))  # ||noise|| ~ 0.14 -> cosine ~ 0.99
    sid, name, score = SpeakerEngine.match(probe, enrolled, 0.45)
    assert (sid, name) == (1, "张三") and score > 0.9


def test_match_below_threshold_returns_score_but_no_identity():
    rng = np.random.default_rng(1)
    e1 = unit(rng.standard_normal(192))
    probe = unit(rng.standard_normal(192))  # ~orthogonal in 192-d
    sid, name, score = SpeakerEngine.match(probe, [(1, "张三", e1)], 0.45)
    assert sid is None and name is None
    assert score is not None and score < 0.45
    assert score == pytest.approx(float(np.dot(probe, e1)))


def test_match_threshold_boundary_and_multiple_samples_per_speaker():
    base = np.zeros(192, dtype=np.float32)
    base[0] = 1.0
    probe = base.copy()
    far = np.zeros(192, dtype=np.float32)
    far[1] = 1.0
    # speaker 1 has a far and a close sample: max over samples is used
    enrolled = [(1, "A", far), (1, "A", base), (2, "B", far)]
    assert SpeakerEngine.match(probe, enrolled, 1.0) == (1, "A", pytest.approx(1.0))
    assert SpeakerEngine.match(probe, enrolled, 1.0001)[0] is None
    # score exactly at threshold is accepted
    sid, name, score = SpeakerEngine.match(probe, [(3, "C", unit([1, 1] + [0] * 190))], 0.70710677)
    assert sid == 3 and score == pytest.approx(0.70710677, abs=1e-6)


def test_match_normalises_unnormalised_enrolled():
    base = np.zeros(192, dtype=np.float32)
    base[0] = 1.0
    sid, name, score = SpeakerEngine.match(base, [(1, "A", base * 50.0)], 0.9)
    assert sid == 1 and score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# real embedding model
# ---------------------------------------------------------------------------

def test_engine_dim_and_lazy_diarizer(speaker_engine):
    assert speaker_engine.dim == 192
    assert speaker_engine._diar is None


def test_embed_dim_and_unit_norm(speaker_engine):
    e = speaker_engine.embed(noise(1.0, amp=0.1, seed=3))
    assert e.shape == (192,) and e.dtype == np.float32
    assert np.linalg.norm(e) == pytest.approx(1.0, abs=1e-4)
    # deterministic
    e2 = speaker_engine.embed(noise(1.0, amp=0.1, seed=3))
    np.testing.assert_allclose(e, e2, atol=1e-5)
    # float64 input accepted
    e3 = speaker_engine.embed(noise(1.0, amp=0.1, seed=3).astype(np.float64))
    np.testing.assert_allclose(e, e3, atol=1e-5)


def test_embed_different_inputs_differ(speaker_engine):
    a = speaker_engine.embed(noise(1.5, amp=0.1, seed=10))
    b = speaker_engine.embed(noise(1.5, amp=0.1, seed=11))
    assert float(np.dot(a, b)) < 0.999


def test_diarize_short_clip_single_turn_without_diarizer(speaker_engine):
    turns = speaker_engine.diarize(noise(0.5, amp=0.1))
    assert turns == [Turn(0.0, 0.5, 0)]
    assert speaker_engine._diar is None  # the heavy pyannote model was not loaded
    assert speaker_engine.diarize(np.zeros(0, dtype=np.float32)) == [Turn(0.0, 0.0, 0)]


def test_identify_turns_matches_enrolled_and_marks_unknown(speaker_engine, db):
    voice = noise(3.0, amp=0.1, seed=42)
    sid = db.upsert_speaker("张三")
    db.add_speaker_embedding(sid, speaker_engine.embed(voice), "sample.wav", 3.0)
    # two white-noise clips embed almost identically (cosine ~0.92), a chirp does not (~0.02)
    other = chirp(3.0)
    audio = np.concatenate([voice, other, np.zeros(int(0.3 * SAMPLE_RATE), dtype=np.float32)])
    turns = [Turn(0.0, 3.0, 0), Turn(3.0, 6.0, 1), Turn(6.0, 6.3, 2)]
    out = speaker_engine.identify_turns(audio, turns, db)
    assert out is turns
    t0, t1, t2 = out
    assert (t0.speaker_id, t0.speaker_name) == (sid, "张三") and t0.score == pytest.approx(1.0, abs=1e-3)
    # the second cluster does not resemble the enrolled voice; it still gets a score
    assert t1.speaker_id is None and t1.speaker_name == "未知" and t1.score is not None and t1.score < 0.45
    # too-short cluster (< 0.5 s) is not embedded at all
    assert t2.speaker_id is None and t2.speaker_name == "未知" and t2.score is None


def test_identify_turns_without_enrolled_speakers(speaker_engine, db):
    turns = [Turn(0.0, 1.0, 0)]
    out = speaker_engine.identify_turns(noise(1.0, amp=0.1), turns, db)
    assert out[0].speaker_name == "未知" and out[0].speaker_id is None and out[0].score is None


def test_identify_turns_same_cluster_shares_identity(speaker_engine, db):
    voice = noise(2.0, amp=0.1, seed=5)
    sid = db.upsert_speaker("李四")
    db.add_speaker_embedding(sid, speaker_engine.embed(voice), None, 2.0)
    audio = np.concatenate([voice[:SAMPLE_RATE], np.zeros(SAMPLE_RATE, dtype=np.float32), voice[SAMPLE_RATE:]])
    turns = [Turn(0.0, 1.0, 0), Turn(2.0, 3.0, 0)]
    out = speaker_engine.identify_turns(audio, turns, db)
    assert all(t.speaker_name == "李四" and t.speaker_id == sid for t in out)
    assert out[0].score == out[1].score


def test_enroll_from_audio_stores_embedding(speaker_engine, db):
    audio = noise(2.0, amp=0.1, seed=9)
    sid = speaker_engine.enroll_from_audio(db, "王五", audio, source="live")
    rows = db.speaker_embeddings()
    assert len(rows) == 1 and rows[0][0] == sid and rows[0][1] == "王五" and rows[0][2].shape == (192,)
    r = db.one("SELECT source_path, duration_s, dim FROM speaker_embeddings")
    assert r["source_path"] == "live" and r["dim"] == 192 and 0 < r["duration_s"] <= 2.0 + 0.1


def test_embed_speech_only_silence_falls_back_to_whole_audio(speaker_engine):
    emb, secs = speaker_engine.embed_speech_only(silence(1.0))
    assert emb.shape == (192,)
    assert secs == pytest.approx(1.0)


def test_enroll_from_files_skips_short_speech(speaker_engine, db, tmp_path):
    import soundfile as sf

    p = tmp_path / "short.wav"
    sf.write(p, silence(0.5), SAMPLE_RATE)
    sid = speaker_engine.enroll(db, "赵六", [p])
    assert db.one("SELECT name FROM speakers WHERE id=?", (sid,))["name"] == "赵六"
    assert db.speaker_embeddings() == []  # 0.5 s < 1.0 s speech -> skipped
