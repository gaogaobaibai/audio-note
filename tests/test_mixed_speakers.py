"""Speakers v3 - mixed sentences (docs/SPEC_speakers_v3_mixed.md): sample consistency (check_sample / SampleMismatch /
force / audit_samples), the turn-level pass of identify_turns, per-utterance relabel, Processor.split_utterance
(manual `at` and `auto`) and the API endpoints around them.

No network (ASR is a fake), no microphone. The real CAM++ engine is used where the decision depends on real
embeddings: chirp and white noise embed nearly orthogonally and stand in for two people.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from audio_note.audio.storage import segment_path
from audio_note.audio.vad import SAMPLE_RATE as SR
from audio_note.db import DB
from audio_note.llm.client import LLMError
from audio_note.runtime import Runtime
from audio_note.speech import voices
from audio_note.speech.asr import ASRResult
from audio_note.speech.pipeline import Processor
from audio_note.speech.speakers import SampleMismatch, SpeakerEngine, Turn, audit_samples, check_sample
from audio_note.ui.app import create_app

from .conftest import chirp, noise

DAY = "2026-10-30"


# --- helpers (same shape as tests/test_voices.py) ----------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _voice(seed: int, dim: int = 192) -> np.ndarray:
    return _unit(np.random.default_rng(seed).standard_normal(dim))


def _near(base: np.ndarray, seed: int, scale: float = 0.03) -> np.ndarray:
    return _unit(base + scale * np.random.default_rng(seed).standard_normal(base.shape[0]))


def _seg(cfg, db: DB, when: datetime, audio: np.ndarray, status: str = "done") -> tuple[int, Path]:
    p = segment_path(cfg, when, ".wav")
    sf.write(p, audio, SR, subtype="PCM_16")
    sid = db.add_segment(p, when, len(audio) / SR, speech_s=len(audio) / SR)
    if status != "new":
        db.set_segment_status(sid, status)
    return sid, p


def _utt(db: DB, seg: int, day: str, hhmm: str, text: str, who: str | None = None, speaker_id: int | None = None,
         start_s: float = 0.0, end_s: float = 1.0, score: float | None = None, cluster: int = 0, source: str | None = None) -> int:
    return db.add_utterances([{
        "segment_id": seg, "start_s": start_s, "end_s": end_s, "abs_start": f"{day}T{hhmm}:00", "day": day,
        "cluster_label": cluster, "speaker_id": speaker_id, "speaker_name": who, "speaker_score": score,
        "text_raw": text, "text": text, "text_seg": text, "speaker_source": source,
    }])[0]


def _dc(seconds: float, level: float) -> np.ndarray:
    """A constant-level clip: the fake engine below 'recognises' positive clips as A and negative ones as nobody."""
    return np.full(int(seconds * SR), level, dtype=np.float32)


class FakeASR:
    def __init__(self, text: str = "重新转写", fail: bool = False):
        self.text, self.fail = text, fail
        self.calls: list[float] = []

    def transcribe(self, audio):
        self.calls.append(round(len(audio) / SR, 2))
        if self.fail:
            raise LLMError("asr down")
        return ASRResult(f"{self.text}{len(self.calls)}", round(len(audio) / SR, 1), 1)


class FakeSplitEngine:
    """SpeakerEngine stand-in for split_utterance: `diarize` returns scripted turns, `identify_turns` labels by
    cluster (cluster 0 -> speaker `a_id`), `embed` maps a positive-DC clip to voice 1 (= speaker A's sample)."""

    match = staticmethod(SpeakerEngine.match)

    def __init__(self, a_id: int | None, turns: list[Turn] | None = None):
        self.a_id = a_id
        self.turns = turns or []
        self.diarize_calls = 0

    def diarize(self, audio):
        self.diarize_calls += 1
        return [Turn(t.start, t.end, t.cluster) for t in self.turns]

    def identify_turns(self, audio, turns, db, segment_id=None):
        for t in turns:
            if t.cluster == 0 and self.a_id is not None:
                t.speaker_id, t.speaker_name, t.score = self.a_id, "甲", 0.8
            else:
                t.speaker_id, t.speaker_name, t.score = None, "未知", 0.1
        return turns

    def embed(self, audio):
        return _voice(1) if float(np.mean(audio)) > 0 else _voice(2)

    def cluster_embedding(self, audio, turns):
        return _voice(1)


# =========================================================================================================
# 1. sample consistency
# =========================================================================================================

def test_sample_mismatch_is_a_value_error_with_attributes():
    e = SampleMismatch("张三", 0.3123, 0.5)
    assert isinstance(e, ValueError) and (e.name, e.score, e.threshold) == ("张三", pytest.approx(0.3123), 0.5)
    assert "张三" in str(e) and "0.31 < 0.50" in str(e) and "仍然加入" in str(e)


def test_check_sample_and_audit_samples_pure_numpy(db):
    sid = db.upsert_speaker("甲")
    assert check_sample(db, sid, _voice(1), 0.5) == (None, True)                        # no samples yet: anything goes
    assert audit_samples(db, sid, 0.5) == []
    base = _voice(1)
    e1 = db.add_speaker_embedding(sid, base, "a.wav", 3.0)
    assert audit_samples(db, sid, 0.5)[0]["consistency"] is None                        # single sample: nothing to compare
    e2 = db.add_speaker_embedding(sid, _near(base, 2), "b.wav", 4.0)
    e3 = db.add_speaker_embedding(sid, _voice(9), None, 5.0)                            # somebody else's voice
    score, ok = check_sample(db, sid, _near(base, 3), 0.5)
    assert ok and score > 0.8
    score, ok = check_sample(db, sid, _voice(7), 0.5)
    assert not ok and score < 0.5
    score_lo, ok_lo = check_sample(db, sid, _voice(7), -1.0)                            # threshold -1 accepts everything
    assert ok_lo and score_lo < 0.5                                                    # (the core reference itself depends on the threshold)
    rows = audit_samples(db, sid, 0.5)
    assert [r["id"] for r in rows] == [e1, e2, e3]
    assert rows[0]["consistency"] > 0.8 and rows[1]["consistency"] > 0.8 and not rows[0]["outlier"] and not rows[1]["outlier"]
    assert rows[2]["consistency"] < 0.5 and rows[2]["outlier"] is True and rows[2]["source_path"] is None
    assert rows[2]["duration_s"] == 5.0
    # a sample of another embedding size (older model) is neither compared nor an outlier
    e4 = db.add_speaker_embedding(sid, _voice(4, dim=16), "old.wav", 1.0)
    rows = audit_samples(db, sid, 0.5)
    assert rows[3]["id"] == e4 and rows[3]["consistency"] is None and rows[3]["outlier"] is False
    assert check_sample(db, sid, _voice(4, dim=16), 0.5) == (pytest.approx(1.0), True)
    # another speaker's samples never count
    other = db.upsert_speaker("乙")
    db.add_speaker_embedding(other, _voice(7), None, 1.0)
    assert check_sample(db, sid, _voice(7), 0.5)[1] is False


def test_enroll_from_audio_rejects_a_voice_unlike_the_existing_samples_unless_forced(speaker_engine, db):
    voice = noise(3.0, amp=0.1, seed=42)
    sid = speaker_engine.enroll_from_audio(db, "甲", voice, source="first")
    assert speaker_engine.check_sample(db, sid, speaker_engine.embed(voice)) == (pytest.approx(1.0, abs=1e-3), True)
    with pytest.raises(SampleMismatch) as ex:
        speaker_engine.enroll_from_audio(db, "甲", chirp(3.0), source="second")
    assert ex.value.name == "甲" and ex.value.score < 0.5 and ex.value.threshold == 0.5 and "仍然加入" in str(ex.value)
    assert len(db.speaker_embeddings()) == 1                                             # nothing was added
    assert db.one("SELECT COUNT(*) AS n FROM speakers")["n"] == 1
    # another white-noise clip resembles the first sample (~0.9): accepted
    assert speaker_engine.enroll_from_audio(db, "甲", noise(2.0, amp=0.1, seed=7), source="third") == sid
    assert len(db.speaker_embeddings()) == 2
    # force adds the mismatching clip, and the audit then flags it as the outlier
    assert speaker_engine.enroll_from_audio(db, "甲", chirp(3.0), source="forced", force=True) == sid
    rows = speaker_engine.audit_samples(db, sid)
    assert [r["outlier"] for r in rows] == [False, False, True] and rows[2]["consistency"] < 0.5
    # a brand-new name has nothing to compare against: never a mismatch
    assert speaker_engine.enroll_from_audio(db, "乙", chirp(2.0), source="x") != sid


def test_enroll_files_skips_mismatching_file_and_segment_turns_take_force(speaker_engine, db, tmp_path):
    p_noise, p_chirp = tmp_path / "noise.wav", tmp_path / "chirp.wav"
    sf.write(p_noise, noise(3.0, amp=0.1, seed=42), SR)
    sf.write(p_chirp, chirp(3.0), SR)
    sid = speaker_engine.enroll(db, "甲", [p_noise])
    assert len(db.speaker_embeddings()) == 1
    assert speaker_engine.enroll(db, "甲", [p_chirp]) == sid                              # skipped, logged, no error
    assert len(db.speaker_embeddings()) == 1
    assert speaker_engine.enroll(db, "甲", [p_chirp], force=True) == sid
    assert len(db.speaker_embeddings()) == 2
    seg = tmp_path / "seg.wav"
    sf.write(seg, np.concatenate([chirp(2.0), noise(2.0, amp=0.1, seed=3)]), SR)
    other = speaker_engine.enroll_from_audio(db, "乙", noise(3.0, amp=0.1, seed=5), source="s")
    with pytest.raises(SampleMismatch):
        speaker_engine.enroll_from_segment_turns(db, "乙", seg, [(0.0, 2.0)])              # chirp part for the noise person
    assert speaker_engine.enroll_from_segment_turns(db, "乙", seg, [(2.0, 4.0)]) == other     # noise part is fine
    assert speaker_engine.enroll_from_segment_turns(db, "乙", seg, [(0.0, 2.0)], force=True) == other
    assert db.one("SELECT COUNT(*) AS n FROM speaker_embeddings WHERE speaker_id=?", (other,))["n"] == 3


# =========================================================================================================
# 2. turn-level identification
# =========================================================================================================

def test_identify_turns_moves_a_foreign_turn_out_of_a_known_cluster(speaker_engine, db):
    a = db.upsert_speaker("甲")
    db.add_speaker_embedding(a, speaker_engine.embed(chirp(3.0)), None, 3.0)
    audio = np.concatenate([chirp(3.0), noise(3.0, amp=0.1, seed=42)])
    turns = [Turn(0.0, 3.0, 0), Turn(3.0, 6.0, 0)]                                      # both in one cluster (the bug)
    seg = db.add_segment("x.wav", datetime(2026, 10, 30, 9, 0), 6.0)
    out = speaker_engine.identify_turns(audio, turns, db, segment_id=seg)
    assert out is turns
    t0, t1 = out
    assert (t0.speaker_id, t0.speaker_name) == (a, "甲") and t0.score is not None and t0.score >= 0.45
    assert t1.speaker_id is None and t1.speaker_name == "未知" and t1.score is not None and t1.score < 0.45
    assert t0.cluster != t1.cluster and {t0.cluster, t1.cluster} == {0, 1}              # new id = max(existing) + 1
    stored = {cl: s for _seg, cl, _e, s in db.cluster_embeddings([seg])}
    assert stored == {0: pytest.approx(3.0), 1: pytest.approx(3.0)}                        # embeddings follow the corrected clusters


def test_identify_turns_recognises_a_known_turn_inside_an_unknown_cluster(speaker_engine, db):
    a = db.upsert_speaker("甲")
    db.add_speaker_embedding(a, speaker_engine.embed(noise(3.0, amp=0.1, seed=42)), None, 3.0)
    audio = np.concatenate([chirp(4.0), noise(1.5, amp=0.1, seed=42), chirp(2.5)])
    turns = [Turn(0.0, 4.0, 0), Turn(4.0, 5.5, 0), Turn(5.5, 8.0, 0)]
    out = speaker_engine.identify_turns(audio, turns, db)
    assert [t.speaker_id for t in out] == [None, a, None]
    assert out[1].cluster == 1 and out[0].cluster == out[2].cluster == 0
    assert out[1].score >= 0.45 and out[1].speaker_name == "甲"


def test_identify_turns_two_enrolled_speakers_in_one_cluster_share_new_ids_per_speaker(speaker_engine, db):
    a = db.upsert_speaker("甲")
    b = db.upsert_speaker("乙")
    db.add_speaker_embedding(a, speaker_engine.embed(chirp(3.0)), None, 3.0)
    db.add_speaker_embedding(b, speaker_engine.embed(noise(3.0, amp=0.1, seed=42)), None, 3.0)
    audio = np.concatenate([chirp(3.0), noise(2.0, amp=0.1, seed=42), chirp(2.0), noise(2.0, amp=0.1, seed=43)])
    turns = [Turn(0.0, 3.0, 5), Turn(3.0, 5.0, 5), Turn(5.0, 7.0, 5), Turn(7.0, 9.0, 5)]
    out = speaker_engine.identify_turns(audio, turns, db)
    assert [t.speaker_id for t in out] == [a, b, a, b]
    assert out[0].cluster == out[2].cluster and out[1].cluster == out[3].cluster and out[0].cluster != out[1].cluster
    assert {out[0].cluster, out[1].cluster} <= {5, 6, 7}                                   # original id and/or new ids from max+1
    assert all(t.score is not None and t.score >= 0.45 for t in out)


def test_identify_turns_keeps_short_turns_and_can_be_switched_off(speaker_engine, db, monkeypatch):
    a = db.upsert_speaker("甲")
    db.add_speaker_embedding(a, speaker_engine.embed(chirp(3.0)), None, 3.0)
    audio = np.concatenate([chirp(3.0), noise(0.8, amp=0.1, seed=42), chirp(0.5)])
    turns = [Turn(0.0, 3.0, 0), Turn(3.0, 3.8, 0), Turn(3.8, 4.3, 0)]
    out = speaker_engine.identify_turns(audio, turns, db)
    assert out[0].speaker_id == a                                                           # the 3 s turn is checked
    # the 0.8 s / 0.5 s turns are not: they keep the cluster decision (whatever it was) and its cluster id
    assert out[1].cluster == out[2].cluster and (out[1].speaker_id, out[1].score) == (out[2].speaker_id, out[2].score)
    # min 0.5 s: now the 0.8 s noise turn is checked and moved
    monkeypatch.setitem(speaker_engine.cfg["speakers"], "turn_check_min_s", 0.5)
    out = speaker_engine.identify_turns(audio[: int(3.8 * SR)], [Turn(0.0, 3.0, 0), Turn(3.0, 3.8, 0)], db)
    assert out[1].speaker_id is None and out[0].speaker_id == a and out[0].cluster != out[1].cluster
    assert {out[0].cluster, out[1].cluster} == {0, 1}
    # 0 switches the layer off entirely
    monkeypatch.setitem(speaker_engine.cfg["speakers"], "turn_check_min_s", 0)
    out = speaker_engine.identify_turns(np.concatenate([chirp(3.0), noise(3.0, amp=0.1, seed=42)]), [Turn(0.0, 3.0, 0), Turn(3.0, 6.0, 0)], db)
    assert out[0].cluster == out[1].cluster == 0 and out[0].speaker_id == out[1].speaker_id


def test_identify_utterances_scores_each_long_utterance_on_its_own(speaker_engine, db):
    a = db.upsert_speaker("甲")
    db.add_speaker_embedding(a, speaker_engine.embed(chirp(3.0)), None, 3.0)
    audio = np.concatenate([chirp(3.0), noise(3.0, amp=0.1, seed=42), noise(0.6, amp=0.1, seed=1)])
    turns = [Turn(0.0, 3.0, 0), Turn(3.0, 6.0, 0), Turn(6.0, 6.6, 0)]
    seg = db.add_segment("y.wav", datetime(2026, 10, 30, 9, 0), 6.6)
    out = speaker_engine.identify_utterances(audio, turns, db, segment_id=seg)
    assert [t.cluster for t in out] == [0, 0, 0]                                            # ids never change here
    assert (out[0].speaker_id, out[0].speaker_name) == (a, "甲") and out[0].score >= 0.45
    assert out[1].speaker_id is None and out[1].speaker_name == "未知" and out[1].score < 0.45
    cluster_score = speaker_engine.match(speaker_engine.cluster_embedding(audio, turns), db.speaker_embeddings(), 0.45)[2]
    assert out[2].score == pytest.approx(cluster_score)                                     # 0.6 s: the cluster decision
    assert out[0].score != out[1].score
    assert [cl for _s, cl, _e, _d in db.cluster_embeddings([seg])] == [0]
    # without enrolled speakers everything is unknown and nothing is embedded per turn
    db.execute("DELETE FROM speaker_embeddings")
    out = speaker_engine.identify_utterances(audio, [Turn(0.0, 3.0, 0), Turn(3.0, 6.0, 0)], db)
    assert all(t.speaker_id is None and t.score is None for t in out)


def test_relabel_segment_identifies_each_utterance_separately(cfg, db, speaker_engine):
    a = db.upsert_speaker("甲")
    db.add_speaker_embedding(a, speaker_engine.embed(chirp(3.0)), None, 3.0)
    seg, _ = _seg(cfg, db, datetime(2026, 10, 30, 9, 0), np.concatenate([chirp(3.0), noise(3.0, amp=0.1, seed=42), chirp(2.0)]))
    u1 = _utt(db, seg, DAY, "09:00", "第一句", "未知", None, 0.0, 3.0, cluster=0)
    u2 = _utt(db, seg, DAY, "09:00", "第二句", "未知", None, 3.0, 6.0, cluster=0)
    u3 = _utt(db, seg, DAY, "09:00", "手标的", "乙", db.upsert_speaker("乙"), 6.0, 8.0, cluster=0, source="user")
    proc = Processor(cfg, db, engine=speaker_engine)
    assert proc.relabel_segment(seg) == 1
    rows = {r["id"]: r for r in db.query("SELECT * FROM utterances")}
    assert (rows[u1]["speaker_id"], rows[u1]["speaker_name"]) == (a, "甲") and rows[u1]["speaker_score"] >= 0.45
    assert rows[u2]["speaker_id"] is None and rows[u2]["speaker_name"] == "未知" and rows[u2]["speaker_score"] < 0.45
    assert rows[u1]["speaker_score"] != rows[u2]["speaker_score"]                           # each sentence carries its own score
    assert rows[u3]["speaker_name"] == "乙" and rows[u3]["speaker_source"] == "user"       # hand label untouched
    assert rows[u1]["cluster_label"] == rows[u2]["cluster_label"] == 0
    md = (cfg.transcripts_dir / f"{DAY}.md").read_text(encoding="utf-8")
    assert "甲**：第一句" in md and "未知**：第二句" in md
    assert proc.relabel_segment(seg) == 0


# =========================================================================================================
# name_voice: member consistency + sample rejection
# =========================================================================================================

class ConsistencyEngine:
    """enroll_from_audio that stores a fixed voice per member clip length and applies the real check_sample."""

    def __init__(self, emb: np.ndarray):
        self.emb = emb
        self.calls: list[tuple[str, bool]] = []

    def enroll_from_audio(self, db, name, audio, source=None, copy_as_source=False, force=False, **kw):
        self.calls.append((name, force))
        row = db.one("SELECT id FROM speakers WHERE name=?", (name,))
        if row is not None:
            score, ok = check_sample(db, int(row["id"]), self.emb, 0.5)
            if not ok and not force:
                raise SampleMismatch(name, float(score), 0.5)
        sid = db.upsert_speaker(name)
        db.add_speaker_embedding(sid, self.emb, source, len(audio) / SR)
        return sid


def test_name_voice_skips_inconsistent_members_and_rejects_a_group_unlike_the_person(cfg, db):
    a = _voice(11)
    seg1, _ = _seg(cfg, db, datetime(2026, 10, 30, 9, 0), noise(10.0, amp=0.1))
    seg2, _ = _seg(cfg, db, datetime(2026, 10, 31, 9, 0), noise(10.0, amp=0.1, seed=2))
    seg3, _ = _seg(cfg, db, datetime(2026, 11, 1, 9, 0), noise(10.0, amp=0.1, seed=3))
    for seg, day in ((seg1, DAY), (seg2, "2026-10-31"), (seg3, "2026-11-01")):
        _utt(db, seg, day, "09:00", "一句", None, None, 0.0, 4.0, cluster=0)
        _utt(db, seg, day, "09:00", "两句", None, None, 4.0, 6.0, cluster=0)
    db.upsert_cluster_embedding(seg1, 0, a, 6.0)                    # anchor (most speech)
    db.upsert_cluster_embedding(seg2, 0, _near(a, 5), 5.0)          # consistent
    db.upsert_cluster_embedding(seg3, 0, _voice(99), 4.0)           # somebody else slipped into the group
    members = [[seg1, 0], [seg2, 0], [seg3, 0]]
    eng = ConsistencyEngine(a)
    res = voices.name_voice(db, cfg, eng, members, "甲", max_samples=3)
    assert res["members_inconsistent"] == 1 and res["samples_added"] == 2 and res["samples_rejected"] == 0
    assert len(eng.calls) == 2                                      # the odd member was never used as a sample
    assert res["utterances_assigned"] == 6                          # ... but its utterances are still assigned
    # naming another group as 甲 whose voice is unlike 甲's samples: nothing added, nothing assigned, 400-able error
    seg4, _ = _seg(cfg, db, datetime(2026, 11, 2, 9, 0), noise(10.0, amp=0.1, seed=4))
    u4 = _utt(db, seg4, "2026-11-02", "09:00", "别人", None, None, 0.0, 5.0, cluster=0)
    db.upsert_cluster_embedding(seg4, 0, _voice(99), 5.0)
    eng2 = ConsistencyEngine(_voice(99))
    with pytest.raises(SampleMismatch) as ex:
        voices.name_voice(db, cfg, eng2, [[seg4, 0]], "甲")
    assert ex.value.name == "甲" and ex.value.score < 0.5
    assert db.one("SELECT speaker_id FROM utterances WHERE id=?", (u4,))["speaker_id"] is None
    assert db.one("SELECT COUNT(*) AS n FROM speaker_embeddings")["n"] == 2
    res = voices.name_voice(db, cfg, eng2, [[seg4, 0]], "甲", force=True)
    assert res["samples_added"] == 1 and res["samples_rejected"] == 0 and res["utterances_assigned"] == 1
    assert eng2.calls[-1] == ("甲", True)
    assert db.one("SELECT speaker_name FROM utterances WHERE id=?", (u4,))["speaker_name"] == "甲"


def test_name_voice_real_engine_rejects_then_forces(cfg, db, speaker_engine):
    a = db.upsert_speaker("甲")
    db.add_speaker_embedding(a, speaker_engine.embed(chirp(3.0)), None, 3.0)
    seg, _ = _seg(cfg, db, datetime(2026, 10, 30, 9, 0), noise(6.0, amp=0.1, seed=42))
    _utt(db, seg, DAY, "09:00", "噪声的人", None, None, 0.0, 6.0, cluster=0)
    assert voices.backfill_cluster_embeddings(db, cfg, speaker_engine)["clusters"] == 1
    with pytest.raises(SampleMismatch):
        voices.name_voice(db, cfg, speaker_engine, [[seg, 0]], "甲")
    assert db.one("SELECT COUNT(*) AS n FROM speaker_embeddings")["n"] == 1
    res = voices.name_voice(db, cfg, speaker_engine, [[seg, 0]], "甲", force=True)
    assert res["samples_added"] == 1 and res["utterances_assigned"] == 1
    assert sum(1 for r in speaker_engine.audit_samples(db, a) if r["outlier"]) == 1        # the two samples disagree: the medoid stays, the other is the outlier
    # the stored-embedding fallback (audio expired) applies the same check
    seg2, p2 = _seg(cfg, db, datetime(2026, 10, 31, 9, 0), noise(6.0, amp=0.1, seed=42))
    _utt(db, seg2, "2026-10-31", "09:00", "过期的", None, None, 0.0, 6.0, cluster=0)
    b = db.upsert_speaker("乙")
    db.add_speaker_embedding(b, speaker_engine.embed(chirp(3.0)), None, 3.0)
    db.upsert_cluster_embedding(seg2, 0, speaker_engine.embed(noise(6.0, amp=0.1, seed=42)), 6.0)
    p2.unlink()
    with pytest.raises(SampleMismatch):
        voices.name_voice(db, cfg, speaker_engine, [[seg2, 0]], "乙")
    res = voices.name_voice(db, cfg, speaker_engine, [[seg2, 0]], "乙", force=True)
    assert res["samples_from_clusters"] == 1 and res["samples_added"] == 1


# =========================================================================================================
# 3. split_utterance
# =========================================================================================================

@pytest.fixture
def split_setup(cfg, db):
    """8 s segment: utterance #1 covers 0-8 s in cluster 0 (positive DC for 0-4.5 s, negative after), plus a
    second utterance in cluster 1 so new cluster labels start after the segment's maximum."""
    audio = np.concatenate([_dc(4.5, 0.2), _dc(3.5, -0.2), _dc(2.0, 0.1)])
    seg, path = _seg(cfg, db, datetime(2026, 10, 30, 9, 0), audio)
    a = db.upsert_speaker("甲")
    db.add_speaker_embedding(a, _voice(1), None, 3.0)
    mixed = _utt(db, seg, DAY, "09:00", "混在一起的一句", "甲", a, 0.0, 8.0, 0.7, cluster=0)
    other = _utt(db, seg, DAY, "09:00", "另一句", "未知", None, 8.0, 10.0, cluster=1)
    db.upsert_cluster_embedding(seg, 0, _voice(1), 8.0)
    return {"seg": seg, "path": path, "a": a, "mixed": mixed, "other": other, "cfg": cfg, "db": db}


def _proc(setup, engine, asr=None) -> Processor:
    p = Processor(setup["cfg"], setup["db"], client=object(), engine=engine, asr=asr or FakeASR(), corrector=None)
    p._corrector_tried = True
    return p


def test_split_utterance_at_points(split_setup):
    s = split_setup
    db, a = s["db"], s["a"]
    asr = FakeASR()
    proc = _proc(s, FakeSplitEngine(a), asr)
    before = db.one("SELECT COUNT(*) AS n FROM utterance_tokens")["n"]
    res = proc.split_utterance(s["mixed"], at=[3.0, 4.5, 4.5, 6.0])
    assert res["removed"] == s["mixed"] and len(res["pieces"]) == 4 and 8.5 < res["billed_s"] < 9.5
    p = res["pieces"]
    assert [(x["start_s"], x["end_s"]) for x in p] == [(0.0, 3.0), (3.0, 4.5), (4.5, 6.0), (6.0, 8.0)]
    assert [x["speaker"] for x in p] == ["甲", "甲", "未知", "未知"]                     # positive DC -> 甲, negative -> nobody
    assert p[0]["speaker_id"] == a and p[2]["speaker_id"] is None and p[0]["score"] == pytest.approx(1.0, abs=1e-3)
    assert [x["text"] for x in p] == ["重新转写1", "重新转写2", "重新转写3", "重新转写4"]
    assert asr.calls == [pytest.approx(3.15, abs=0.02), pytest.approx(1.8, abs=0.02), pytest.approx(1.8, abs=0.02), pytest.approx(2.3, abs=0.02)]
    assert db.one("SELECT id FROM utterances WHERE id=?", (s["mixed"],)) is None
    rows = db.query("SELECT * FROM utterances WHERE segment_id=? ORDER BY start_s", (s["seg"],))
    assert [r["id"] for r in rows[:4]] == [x["id"] for x in p] and rows[4]["id"] == s["other"]
    assert [r["cluster_label"] for r in rows] == [0, 2, 3, 4, 1]                            # piece 0 keeps 0; others after max (1)
    assert [r["speaker_source"] for r in rows[:4]] == [None] * 4
    assert rows[1]["abs_start"] == "2026-10-30T09:00:03.000" and rows[3]["abs_start"] == "2026-10-30T09:00:06.000"
    assert all(r["day"] == DAY for r in rows)
    assert db.one("SELECT COUNT(*) AS n FROM utterance_tokens")["n"] == before - 1 + 4   # old tokens cascaded, new ones in
    assert {cl for _s, cl, _e, _d in db.cluster_embeddings([s["seg"]])} == {0, 2, 3, 4}
    md = (s["cfg"].transcripts_dir / f"{DAY}.md").read_text(encoding="utf-8")
    assert "09:00:03 甲**：重新转写2" in md and "09:00:06 未知**：重新转写4" in md and "混在一起" not in md


def test_split_utterance_validates_points_and_merges_short_pieces(split_setup):
    s = split_setup
    proc = _proc(s, FakeSplitEngine(s["a"]))
    for bad in ([0.2], [7.8], [-1.0], [9.0], [5.0, 3.0], ["x"]):
        with pytest.raises(ValueError):
            proc.split_utterance(s["mixed"], at=bad)
    with pytest.raises(ValueError, match="拆分点"):
        proc.split_utterance(s["mixed"], at=[])
    with pytest.raises(ValueError, match="没有这条记录"):
        proc.split_utterance(99999, at=[2.0])
    assert db_count(s["db"]) == 2                                                           # nothing changed so far
    # a 0.3 s piece is merged into the previous one; a short first piece joins the next
    res = proc.split_utterance(s["mixed"], at=[0.4, 3.0, 3.3])
    assert [(x["start_s"], x["end_s"]) for x in res["pieces"]] == [(0.0, 3.3), (3.3, 8.0)]
    # the audio must still exist
    s["path"].unlink()
    with pytest.raises(FileNotFoundError):
        proc.split_utterance(res["pieces"][0]["id"], at=[1.0])


def db_count(db) -> int:
    return int(db.one("SELECT COUNT(*) AS n FROM utterances")["n"])


def test_split_utterance_auto_with_scripted_diarization(split_setup):
    s = split_setup
    engine = FakeSplitEngine(s["a"], [Turn(0.0, 2.0, 0), Turn(2.0, 2.5, 1), Turn(2.6, 4.4, 0), Turn(4.6, 8.0, 1)])
    proc = _proc(s, engine)
    preview = proc.preview_split(s["mixed"], auto=True)
    assert preview["utterance_id"] == s["mixed"] and preview["segment_id"] == s["seg"] and preview["end_s"] == 8.0
    assert [(x["rel_start"], x["rel_end"], x["speaker"]) for x in preview["pieces"]] == [(0.0, 4.4, "甲"), (4.6, 8.0, "未知")]
    assert preview["pieces"][0]["speaker_id"] == s["a"] and preview["pieces"][0]["score"] == 0.8 and preview["pieces"][1]["score"] == 0.1
    assert db_count(s["db"]) == 2 and engine.diarize_calls == 1                               # preview changes nothing
    res = proc.split_utterance(s["mixed"], auto=True)
    assert [(x["start_s"], x["end_s"], x["speaker"], x["speaker_id"]) for x in res["pieces"]] == [(0.0, 4.4, "甲", s["a"]), (4.6, 8.0, "未知", None)]
    assert db_count(s["db"]) == 3
    # one voice only -> error, nothing changed
    engine.turns = [Turn(0.0, 4.4, 0)]
    with pytest.raises(ValueError, match="没有分出第二个声音"):
        proc.split_utterance(res["pieces"][0]["id"], auto=True)
    engine.turns = []
    with pytest.raises(ValueError, match="没有分出第二个声音"):
        proc.preview_split(res["pieces"][0]["id"], auto=True)
    with pytest.raises(ValueError, match="at"):
        proc.split_utterance(res["pieces"][0]["id"])
    assert db_count(s["db"]) == 3


def test_split_utterance_asr_failure_leaves_the_database_untouched(split_setup):
    s = split_setup
    proc = _proc(s, FakeSplitEngine(s["a"]), FakeASR(fail=True))
    with pytest.raises(LLMError):
        proc.split_utterance(s["mixed"], at=[3.0])
    row = s["db"].one("SELECT * FROM utterances WHERE id=?", (s["mixed"],))
    assert row is not None and row["text"] == "混在一起的一句" and db_count(s["db"]) == 2
    assert {cl for _s, cl, _e, _d in s["db"].cluster_embeddings([s["seg"]])} == {0}


def test_split_utterance_auto_real_engine_separates_two_voices(cfg, db, speaker_engine, monkeypatch):
    """The real embedding / turn-level identification on chirp + noise; diarization is scripted because the
    pyannote model does not treat synthetic signals as speech."""
    a = db.upsert_speaker("甲")
    db.add_speaker_embedding(a, speaker_engine.embed(chirp(3.0)), None, 3.0)
    seg, _ = _seg(cfg, db, datetime(2026, 10, 30, 9, 0), np.concatenate([chirp(3.0), noise(3.0, amp=0.1, seed=42)]))
    mixed = _utt(db, seg, DAY, "09:00", "混合", "甲", a, 0.0, 6.0, 0.5, cluster=0)
    monkeypatch.setattr(speaker_engine, "diarize", lambda audio: [Turn(0.0, 3.0, 0), Turn(3.0, 6.0, 0)])
    proc = Processor(cfg, db, client=object(), engine=speaker_engine, asr=FakeASR(), corrector=None)
    proc._corrector_tried = True
    res = proc.split_utterance(mixed, auto=True)
    p = res["pieces"]
    assert [(x["start_s"], x["end_s"]) for x in p] == [(0.0, 3.0), (3.0, 6.0)]
    assert (p[0]["speaker_id"], p[0]["speaker"]) == (a, "甲") and p[0]["score"] >= 0.45
    assert p[1]["speaker_id"] is None and p[1]["speaker"] == "未知" and p[1]["score"] < 0.45
    rows = db.query("SELECT cluster_label, speaker_name FROM utterances ORDER BY start_s")
    assert [tuple(r) for r in rows] == [(0, "甲"), (1, "未知")]
    stored = {cl: d for _s, cl, _e, d in db.cluster_embeddings([seg])}
    assert stored == {0: pytest.approx(3.0), 1: pytest.approx(3.0)}


# =========================================================================================================
# API
# =========================================================================================================

@pytest.fixture
def ui_cfg(cfg_factory, tmp_path: Path):
    cfg = cfg_factory({"memory": {"retrieval": "fts", "embedding_model": "hash"},
                       "correction": {"enabled": False, "glossary_file": str(tmp_path / "glossary.txt")}})
    yield cfg
    from audio_note.logging_setup import _remove_own_handlers

    _remove_own_handlers(logging.getLogger())


@pytest.fixture
def rt(ui_cfg) -> Runtime:
    r = Runtime(ui_cfg)
    yield r
    r.shutdown()


@pytest.fixture
def client(rt: Runtime) -> TestClient:
    with TestClient(create_app(rt), raise_server_exceptions=False) as c:
        yield c


class MismatchEngine:
    """API-side engine: enroll* raise SampleMismatch for an existing speaker unless forced; split helpers as
    FakeSplitEngine."""

    match = staticmethod(SpeakerEngine.match)

    def __init__(self, a_id: int | None = None):
        self.a_id = a_id
        self.calls: list[tuple] = []
        self.turns = [Turn(0.0, 3.0, 0), Turn(3.0, 6.0, 1)]

    def _enroll(self, db, name, force, source):
        self.calls.append((name, bool(force)))
        row = db.one("SELECT id FROM speakers WHERE name=?", (name,))
        if row is not None and db.one("SELECT 1 FROM speaker_embeddings WHERE speaker_id=?", (row["id"],)) and not force:
            raise SampleMismatch(name, 0.21, 0.5)
        sid = db.upsert_speaker(name)
        db.add_speaker_embedding(sid, _voice(1), source, 2.0)
        return sid

    def enroll(self, db, name, sources, notes=None, force=False):
        sid = db.upsert_speaker(name, notes)
        try:
            return self._enroll(db, name, force, str(sources[0]))
        except SampleMismatch:
            return sid                                                  # enroll() skips, like the real one

    def enroll_from_audio(self, db, name, audio, source=None, copy_as_source=False, force=False, **kw):
        return self._enroll(db, name, force, source)

    def enroll_from_segment_turns(self, db, name, seg_path, turns, force=False, **kw):
        return self._enroll(db, name, force, f"{seg_path}#{turns[0][0]:.1f}-{turns[-1][1]:.1f}")

    def speech_only(self, audio, max_s=60.0):
        return audio

    def embed(self, audio):
        return _voice(1) if float(np.mean(audio)) > 0 else _voice(2)

    def check_sample(self, db, speaker_id, emb):
        return 0.21, False

    def cluster_embedding(self, audio, turns):
        return _voice(1)

    def diarize(self, audio):
        return [Turn(t.start, t.end, t.cluster) for t in self.turns]

    def identify_turns(self, audio, turns, db, segment_id=None):
        for t in turns:
            if t.cluster == 0 and self.a_id is not None:
                t.speaker_id, t.speaker_name, t.score = self.a_id, "甲", 0.8
            else:
                t.speaker_id, t.speaker_name, t.score = None, "未知", 0.1
        return turns


def _wav(audio: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_api_samples_consistency_outliers_and_bulk_delete(client: TestClient, rt: Runtime):
    db = rt.db
    sid = db.upsert_speaker("甲")
    base = _voice(1)
    e1 = db.add_speaker_embedding(sid, base, "a.wav", 3.0)
    e2 = db.add_speaker_embedding(sid, _near(base, 2), "b.wav", 4.0)
    e3 = db.add_speaker_embedding(sid, _voice(9), None, 5.0)
    e4 = db.add_speaker_embedding(sid, _voice(10), None, 5.0)
    single = db.upsert_speaker("乙")
    db.add_speaker_embedding(single, _voice(3), None, 1.0)
    rows = client.get(f"/api/speakers/{sid}/samples").json()
    assert [r["id"] for r in rows] == [e1, e2, e3, e4]
    assert rows[0]["consistency"] > 0.8 and rows[0]["outlier"] is False and rows[1]["outlier"] is False
    assert rows[2]["outlier"] is True and rows[2]["consistency"] < 0.5 and rows[3]["outlier"] is True
    one = client.get(f"/api/speakers/{single}/samples").json()
    assert len(one) == 1 and one[0]["consistency"] is None and one[0]["outlier"] is False
    sp = {s["name"]: s for s in client.get("/api/speakers").json()["speakers"]}
    assert sp["甲"]["outlier_samples"] == 2 and sp["乙"]["outlier_samples"] == 0
    r = client.delete(f"/api/speakers/{sid}/samples/outliers")
    assert r.status_code == 200 and r.json() == {"ok": True, "deleted": 2, "remaining": 2}
    assert [r["id"] for r in client.get(f"/api/speakers/{sid}/samples").json()] == [e1, e2]
    assert client.delete(f"/api/speakers/{sid}/samples/outliers").json() == {"ok": True, "deleted": 0, "remaining": 2}
    assert client.delete("/api/speakers/999/samples/outliers").status_code == 404
    # two samples that disagree are both outliers; deleting leaves the speaker row with no samples
    db.execute("DELETE FROM speaker_embeddings WHERE id=?", (e2,))
    db.add_speaker_embedding(sid, _voice(5), None, 1.0)
    assert client.delete(f"/api/speakers/{sid}/samples/outliers").json() == {"ok": True, "deleted": 1, "remaining": 1}   # two strangers: the (longer) medoid is kept
    assert db.one("SELECT name FROM speakers WHERE id=?", (sid,))["name"] == "甲"
    assert sp["甲"]["samples"] == 4


def test_api_label_and_voices_name_report_mismatch_and_accept_force(client: TestClient, rt: Runtime, monkeypatch):
    db = rt.db
    eng = MismatchEngine()
    monkeypatch.setattr(rt, "engine", lambda: eng)
    seg, _ = _seg(rt.cfg, db, datetime(2026, 10, 30, 9, 0), _dc(6.0, 0.2))
    u1 = _utt(db, seg, DAY, "09:00", "一", None, None, 0.0, 3.0, cluster=0)
    u2 = _utt(db, seg, DAY, "09:00", "二", None, None, 3.0, 6.0, cluster=1)
    # first label of a new name: no samples to compare -> fine
    r = client.post("/api/speakers/label", json={"utterance_id": u1, "name": "甲"})
    assert r.status_code == 200, r.text
    sid = r.json()["id"]
    # another range for the same name mismatches -> 400 with the flag and the score
    r = client.post("/api/speakers/label", json={"utterance_id": u2, "name": "甲"})
    assert r.status_code == 400
    body = r.json()
    assert body["mismatch"] is True and body["score"] == 0.21 and body["threshold"] == 0.5 and body["name"] == "甲"
    assert "甲" in body["error"] and "仍然加入" in body["error"]
    assert db.one("SELECT speaker_id FROM utterances WHERE id=?", (u2,))["speaker_id"] is None
    r = client.post("/api/speakers/label", json={"utterance_id": u2, "name": "甲", "force": True})
    assert r.status_code == 200 and r.json()["samples"] == 2 and eng.calls[-1] == ("甲", True)
    assert db.one("SELECT speaker_id, speaker_source FROM utterances WHERE id=?", (u2,))["speaker_id"] == sid
    # voices/name: a group that sounds unlike 甲
    seg2, _ = _seg(rt.cfg, db, datetime(2026, 10, 31, 9, 0), _dc(6.0, -0.2))
    u3 = _utt(db, seg2, "2026-10-31", "09:00", "三", None, None, 0.0, 5.0, cluster=0)
    db.upsert_cluster_embedding(seg2, 0, _voice(9), 5.0)
    r = client.post("/api/voices/name", json={"members": [[seg2, 0]], "name": "甲", "auto_relabel": False})
    assert r.status_code == 400 and r.json()["mismatch"] is True and r.json()["score"] == 0.21
    assert db.one("SELECT speaker_id FROM utterances WHERE id=?", (u3,))["speaker_id"] is None
    r = client.post("/api/voices/name", json={"members": [[seg2, 0]], "name": "甲", "auto_relabel": False, "force": True})
    assert r.status_code == 200, r.text
    assert r.json()["samples_added"] == 1 and r.json()["samples_rejected"] == 0 and r.json()["utterances_assigned"] == 1
    # ordinary ValueErrors keep the plain {"error"} body
    r = client.post("/api/voices/name", json={"members": [], "name": "甲"})
    assert r.status_code == 400 and r.json() == {"error": "members 不能为空"}
    assert rt.wait_jobs(5)


def test_api_upload_sample_reports_mismatch_and_force_adds(client: TestClient, rt: Runtime, monkeypatch):
    eng = MismatchEngine()
    monkeypatch.setattr(rt, "engine", lambda: eng)
    body = _wav(_dc(2.0, 0.2))
    r = client.post("/api/speakers/甲/samples", params={"filename": "a.wav"}, content=body)
    assert r.status_code == 200 and r.json()["samples"] == 1
    r = client.post("/api/speakers/甲/samples", params={"filename": "b.wav"}, content=body)
    assert r.status_code == 400 and r.json()["mismatch"] is True and r.json()["score"] == 0.21 and "甲" in r.json()["error"]
    assert rt.db.one("SELECT COUNT(*) AS n FROM speaker_embeddings")["n"] == 1
    r = client.post("/api/speakers/甲/samples", params={"filename": "b.wav", "force": "true"}, content=body)
    assert r.status_code == 200 and r.json()["samples"] == 2 and eng.calls[-1] == ("甲", True)


def test_api_split_needs_api_key_then_splits(client: TestClient, rt: Runtime, monkeypatch):
    db = rt.db
    a = db.upsert_speaker("甲")
    db.add_speaker_embedding(a, _voice(1), None, 3.0)
    eng = MismatchEngine(a)
    monkeypatch.setattr(rt, "engine", lambda: eng)
    seg, path = _seg(rt.cfg, db, datetime(2026, 10, 30, 9, 0), np.concatenate([_dc(3.0, 0.2), _dc(3.0, -0.2)]))
    mixed = _utt(db, seg, DAY, "09:00", "混合", "甲", a, 0.0, 6.0, 0.6, cluster=0)
    # preview needs no key
    r = client.get(f"/api/utterances/{mixed}/preview-split", params={"auto": 1})
    assert r.status_code == 200, r.text
    pv = r.json()
    assert pv["utterance_id"] == mixed and pv["segment_id"] == seg and pv["start_s"] == 0.0 and pv["end_s"] == 6.0
    assert [(p["start_s"], p["end_s"], p["rel_start"], p["rel_end"], p["speaker"], p["speaker_id"], p["score"]) for p in pv["pieces"]] == \
        [(0.0, 3.0, 0.0, 3.0, "甲", a, 0.8), (3.0, 6.0, 3.0, 6.0, "未知", None, 0.1)]
    r = client.get(f"/api/utterances/{mixed}/preview-split", params={"at": "2.0,4.0"})
    assert r.status_code == 200 and [(p["start_s"], p["end_s"]) for p in r.json()["pieces"]] == [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]
    assert client.get("/api/utterances/99999/preview-split").status_code == 404
    eng.turns = [Turn(0.0, 6.0, 0)]
    r = client.get(f"/api/utterances/{mixed}/preview-split")
    assert r.status_code == 400 and "没有分出第二个声音" in r.json()["error"]
    eng.turns = [Turn(0.0, 3.0, 0), Turn(3.0, 6.0, 1)]
    # the split itself needs the API key: 409 like /api/chat
    r = client.post(f"/api/utterances/{mixed}/split", json={"auto": True})
    assert r.status_code == 409 and "API key" in r.json()["error"]
    assert db.one("SELECT id FROM utterances WHERE id=?", (mixed,)) is not None
    # with a client (fake) and a fake ASR class
    monkeypatch.setattr(rt, "client", lambda: object())

    class FakeASRClass:
        def __init__(self, cfg, client, vad=None):
            self.inner = FakeASR("拆")

        def transcribe(self, audio):
            return self.inner.transcribe(audio)

    import audio_note.speech.asr as asr_mod

    monkeypatch.setattr(asr_mod, "ASR", FakeASRClass)
    assert client.post("/api/utterances/99999/split", json={"auto": True}).status_code == 404
    r = client.post(f"/api/utterances/{mixed}/split", json={})
    assert r.status_code == 400 and "at" in r.json()["error"]
    r = client.post(f"/api/utterances/{mixed}/split", json={"at": "1.0"})
    assert r.status_code == 400
    r = client.post(f"/api/utterances/{mixed}/split", json={"at": [0.1]})
    assert r.status_code == 400 and "拆分点" in r.json()["error"]
    r = client.post(f"/api/utterances/{mixed}/split", json={"at": [2.0, 4.0]})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["ok"] is True and out["removed"] == mixed and len(out["pieces"]) == 3 and out["billed_s"] > 0
    assert [(p["start_s"], p["end_s"], p["speaker"], p["speaker_id"]) for p in out["pieces"]] == \
        [(0.0, 2.0, "甲", a), (2.0, 4.0, "未知", None), (4.0, 6.0, "未知", None)]
    assert [p["text"] for p in out["pieces"]] == ["拆1", "拆2", "拆3"]
    assert all(set(p) == {"id", "start_s", "end_s", "speaker", "speaker_id", "score", "text"} for p in out["pieces"])
    # the original row is gone (SQLite may hand its rowid to the first new piece when it was the last row)
    assert sorted(r["id"] for r in db.query("SELECT id FROM utterances WHERE segment_id=?", (seg,))) == sorted(p["id"] for p in out["pieces"])
    assert db.one("SELECT COUNT(*) AS n FROM utterances WHERE text='混合'")["n"] == 0
    assert client.get(f"/api/transcript/{DAY}").json()["sessions"][0]["utterances"][0]["text"] == "拆1"
    # auto on one of the pieces
    first = out["pieces"][0]["id"]
    eng.turns = [Turn(0.0, 1.0, 0), Turn(1.0, 2.0, 1)]
    r = client.post(f"/api/utterances/{first}/split", json={"auto": True})
    assert r.status_code == 200, r.text
    assert [(p["start_s"], p["end_s"], p["speaker"]) for p in r.json()["pieces"]] == [(0.0, 1.0, "甲"), (1.0, 2.0, "未知")]
    # an ASR failure answers 502 and changes nothing
    monkeypatch.setattr(FakeASRClass, "transcribe", lambda self, audio: (_ for _ in ()).throw(LLMError("down")))
    n = db_count(db)
    r = client.post(f"/api/utterances/{out['pieces'][2]['id']}/split", json={"at": [1.0]})
    assert r.status_code == 502 and "重新转写失败" in r.json()["error"] and db_count(db) == n
    assert rt.wait_jobs(5)


def test_api_config_exposes_the_v3_keys(client: TestClient, rt: Runtime):
    cfg = client.get("/api/config").json()
    spk = cfg["values"]["speakers"]
    assert spk["sample_consistency_min"] == 0.5 and spk["turn_check_min_s"] == 1.0 and spk["turn_reject_margin"] == 0.15
    assert cfg["defaults"]["speakers"]["turn_check_min_s"] == 1.0
    r = client.put("/api/config", json={"speakers": {"sample_consistency_min": 1.5}})
    assert r.status_code == 400 and "0–1" in r.json()["error"]
    r = client.put("/api/config", json={"speakers": {"turn_check_min_s": 0}})
    assert r.status_code == 200, r.text
    assert rt.cfg.speakers.turn_check_min_s == 0


# --- medoid-core consistency (two stray samples cannot vouch for each other) -------------------------------

def test_audit_consistency_is_measured_against_the_core_not_mutual_vouching(db):
    """Three long clean samples of one voice plus two short snippets that resemble each other (0.6) but not
    the voice (0.4): the old 'best cosine to any other sample' let the snippets vouch for one another;
    the core reference (duration-weighted medoid + its neighbours) flags both."""
    sid = db.upsert_speaker("核")
    v = _voice(1)
    # v3b: the clean samples are uploads (origin 'file'), which anchors the core; without such an anchor two
    # factions would be reported as ambiguous instead (see test_audit_v3b_* below)
    good = [db.add_speaker_embedding(sid, _near(v, 10 + i), f"g{i}.wav", 25.0, origin="file") for i in range(3)]
    junk_base = _voice(9)
    junk = [db.add_speaker_embedding(sid, _unit(0.78 * junk_base + 0.62 * _voice(20 + i)), None, 2.0) for i in range(2)]
    rows = {r["id"]: r for r in audit_samples(db, sid, 0.5)}
    assert all(rows[g]["consistency"] > 0.9 and not rows[g]["outlier"] for g in good)
    assert all(rows[j]["consistency"] < 0.5 and rows[j]["outlier"] for j in junk)
    # a new clip that resembles the junk but not the voice is refused; one that resembles the voice is accepted
    assert check_sample(db, sid, _near(junk_base, 3), 0.5)[1] is False
    assert check_sample(db, sid, _near(v, 4), 0.5)[1] is True
    # with only two samples that disagree the longer one is the reference
    sid2 = db.upsert_speaker("二")
    long_id = db.add_speaker_embedding(sid2, _voice(2), "long.wav", 20.0)
    short_id = db.add_speaker_embedding(sid2, _voice(3), "short.wav", 1.5)
    rows2 = {r["id"]: r for r in audit_samples(db, sid2, 0.5)}
    assert rows2[long_id]["outlier"] is False and rows2[short_id]["outlier"] is True


# =========================================================================================================
# v3b A1: anchor priority / ambiguity (docs/SPEC_speakers_v3b.md)
# =========================================================================================================

def test_audit_v3b_mic_samples_anchor_the_core_over_a_larger_voice_faction(db):
    """Reviewer's scenario: 4 similar 'voice' samples (cut from a mixed group) + 3 similar 'mic' samples, the two
    factions unlike each other. The majority used to become the core and the clean mic samples the outliers;
    now the anchor is chosen among mic / file samples, so the voice faction is the outlier."""
    sid = db.upsert_speaker("本人")
    me, them = _voice(1), _voice(2)
    voice = [db.add_speaker_embedding(sid, _near(them, 10 + i), None, 20.0, origin="voice") for i in range(4)]
    mic = [db.add_speaker_embedding(sid, _near(me, 20 + i), f"record_{i}.wav", 10.0, origin="mic") for i in range(3)]
    rows = {r["id"]: r for r in audit_samples(db, sid, 0.5)}
    assert all(rows[m]["outlier"] is False and rows[m]["ambiguous"] is False and rows[m]["consistency"] > 0.9 for m in mic)
    assert all(rows[v]["outlier"] is True and rows[v]["ambiguous"] is False and rows[v]["consistency"] < 0.5 for v in voice)
    assert {rows[m]["origin"] for m in mic} == {"mic"} and {rows[v]["origin"] for v in voice} == {"voice"}
    assert check_sample(db, sid, _near(me, 5), 0.5)[1] is True and check_sample(db, sid, _near(them, 5), 0.5)[1] is False
    # 'file' anchors as well; 'label' / 'cluster' / NULL do not
    sid2 = db.upsert_speaker("二")
    for i in range(4):
        db.add_speaker_embedding(sid2, _near(them, 30 + i), None, 20.0, origin="label")
    f = db.add_speaker_embedding(sid2, _near(me, 40), "up.wav", 3.0, origin="file")
    rows = {r["id"]: r for r in audit_samples(db, sid2, 0.5)}
    assert rows[f]["outlier"] is False and sum(1 for r in rows.values() if r["outlier"]) == 4


def test_audit_v3b_two_factions_without_an_anchor_are_ambiguous_not_outliers(db):
    from audio_note.speech.speakers import speaker_is_ambiguous

    sid = db.upsert_speaker("谁")
    a, b = _voice(3), _voice(4)
    fa = [db.add_speaker_embedding(sid, _near(a, 50 + i), None, 20.0, origin="voice") for i in range(4)]
    fb = [db.add_speaker_embedding(sid, _near(b, 60 + i), None, 5.0, origin="voice") for i in range(2)]
    lone = db.add_speaker_embedding(sid, _voice(5), None, 5.0, origin="label")
    rows = {r["id"]: r for r in audit_samples(db, sid, 0.5)}
    assert all(rows[i]["ambiguous"] is True and rows[i]["outlier"] is False for i in fa + fb)
    assert rows[lone]["ambiguous"] is False and rows[lone]["outlier"] is True       # a lone stranger is still an outlier
    assert speaker_is_ambiguous(db, sid, 0.5)
    # ambiguous: check_sample falls back to the best cosine over all samples and never refuses
    s, ok = check_sample(db, sid, _near(b, 70), 0.5)
    assert ok and s > 0.8
    s, ok = check_sample(db, sid, _voice(6), 0.5)
    assert ok and s < 0.5
    # one mic sample resolves it: faction a (which it resembles) is the core, faction b the outliers
    db.add_speaker_embedding(sid, _near(a, 80), "rec.wav", 8.0, origin="mic")
    rows = {r["id"]: r for r in audit_samples(db, sid, 0.5)}
    assert not any(r["ambiguous"] for r in rows.values()) and not speaker_is_ambiguous(db, sid, 0.5)
    assert all(rows[i]["outlier"] for i in fb + [lone]) and not any(rows[i]["outlier"] for i in fa)
    assert check_sample(db, sid, _near(b, 70), 0.5)[1] is False


def test_analyse_samples_duration_cap_without_anchor():
    """Without a trusted anchor the duration weight is capped at 10 s: a 60 s mixed clip cannot drag the medoid
    into its faction (uncapped, faction A below would be the core)."""
    from audio_note.speech.speakers import ANCHOR_DURATION_CAP_S, _analyse_samples

    a, b = _voice(7), _voice(8)
    embs = [_near(a, 1), _near(a, 2), _near(b, 3), _near(b, 4), _near(b, 5)]
    res = _analyse_samples(embs, [60.0, 3.0, 5.0, 5.0, 5.0], 0.5, [None] * 5)
    assert ANCHOR_DURATION_CAP_S == 10.0
    assert res.core == {2, 3, 4} and res.ambiguous == {0, 1, 2, 3, 4}
    # spec scenario: one 60 s 'voice' sample vs three 5 s 'file' samples -> the file faction is the core, no ambiguity
    res = _analyse_samples([_near(a, 1), *[_near(b, i) for i in (3, 4, 5)]], [60.0, 5.0, 5.0, 5.0], 0.5, ["voice", "file", "file", "file"])
    assert res.core == {1, 2, 3} and res.ambiguous == frozenset()
    # with a trusted anchor the weights stay uncapped and only trusted samples can be the medoid
    res = _analyse_samples([_near(a, 1), _near(a, 2), _near(b, 3)], [60.0, 3.0, 5.0], 0.5, ["file", "file", "voice"])
    assert res.core == {0, 1}
    assert _analyse_samples([], [], 0.5).ref is None and _analyse_samples([a], [1.0], 0.5).core == {0}


def test_enroll_writes_notes_only_after_a_sample_was_added_and_tags_origins(speaker_engine, db, tmp_path):
    short, good = tmp_path / "short.wav", tmp_path / "good.wav"
    sf.write(short, np.zeros(SR // 2, dtype=np.float32), SR)
    sf.write(good, noise(3.0, amp=0.1, seed=42), SR)
    sid = speaker_engine.enroll(db, "备注", [short], notes="第一次")
    assert db.one("SELECT notes FROM speakers WHERE id=?", (sid,))["notes"] is None     # nothing added: notes not written
    assert speaker_engine.enroll(db, "备注", [good], notes="第二次") == sid
    assert db.one("SELECT notes FROM speakers WHERE id=?", (sid,))["notes"] == "第二次"
    # a mismatching file is skipped and leaves the notes alone
    sf.write(short, chirp(3.0), SR)
    assert speaker_engine.enroll(db, "备注", [short], notes="第三次") == sid
    assert db.one("SELECT notes FROM speakers WHERE id=?", (sid,))["notes"] == "第二次"
    # origins of the enrol paths: file upload / label (in-memory clip, segment range)
    assert speaker_engine.enroll_from_audio(db, "备注", noise(2.0, amp=0.1, seed=7), source="s") == sid
    seg = tmp_path / "seg.wav"
    sf.write(seg, noise(3.0, amp=0.1, seed=3), SR)
    assert speaker_engine.enroll_from_segment_turns(db, "备注", seg, [(0.0, 3.0)]) == sid
    assert [r["origin"] for r in db.query("SELECT origin FROM speaker_embeddings WHERE speaker_id=? ORDER BY id", (sid,))] == ["file", "label", "label"]


# --- v3b A2: split_utterance hardening -------------------------------------------------------------------

def test_split_v3b_every_piece_needs_text_and_ids_are_fresh(split_setup):
    s = split_setup
    db = s["db"]

    class PatchyASR(FakeASR):
        def transcribe(self, audio):
            res = super().transcribe(audio)
            return ASRResult("", res.seconds, 1) if len(self.calls) == 2 else res

    proc = _proc(s, FakeSplitEngine(s["a"]), PatchyASR())
    with pytest.raises(ValueError, match="有一段没有转写出文字"):
        proc.split_utterance(s["mixed"], at=[3.0])
    assert db.one("SELECT id FROM utterances WHERE id=?", (s["mixed"],)) is not None and db_count(db) == 2
    # new rows get ids above every id ever used (never the deleted row's rowid)
    top = int(db.one("SELECT MAX(id) AS m FROM utterances")["m"])
    res = _proc(s, FakeSplitEngine(s["a"])).split_utterance(s["mixed"], at=[3.0])
    ids = [p["id"] for p in res["pieces"]]
    assert ids == [top + 1, top + 2]
    res2 = _proc(s, FakeSplitEngine(s["a"])).split_utterance(ids[0], at=[1.0])
    assert [p["id"] for p in res2["pieces"]] == [top + 3, top + 4]                       # the freed id (top+1) is not reused
    assert db.one("SELECT loudness_db FROM utterances WHERE id=?", (top + 3,))["loudness_db"] is not None


def test_split_v3b_single_piece_after_merge_and_concurrent_change(split_setup):
    s = split_setup
    db = s["db"]
    proc = _proc(s, FakeSplitEngine(s["a"]))
    with pytest.raises(ValueError, match="只剩一段"):
        proc.split_utterance(s["mixed"], at=[7.6])                     # (7.6, 8.0) is too short and merges back
    with pytest.raises(ValueError, match="只剩一段"):
        proc.preview_split(s["mixed"], at=[0.4], auto=False)           # the preview reports it too
    assert db_count(db) == 2

    class RacingASR(FakeASR):
        def transcribe(self, audio):
            db.execute("UPDATE utterances SET end_s=7.5 WHERE id=?", (s["mixed"],))      # a relabel / split got there first
            return super().transcribe(audio)

    with pytest.raises(ValueError, match="请刷新"):
        _proc(s, FakeSplitEngine(s["a"]), RacingASR()).split_utterance(s["mixed"], at=[3.0])
    assert db_count(db) == 2 and db.one("SELECT end_s FROM utterances WHERE id=?", (s["mixed"],))["end_s"] == 7.5


# --- v3b API: origin / ambiguity fields, force validation, split under the scheduler lock -----------------

def test_api_samples_carry_origin_and_ambiguity_and_outlier_delete_spares_them(client: TestClient, rt: Runtime):
    db = rt.db
    sid = db.upsert_speaker("谁")
    a, b = _voice(3), _voice(4)
    fa = [db.add_speaker_embedding(sid, _near(a, 50 + i), None, 20.0, origin="voice") for i in range(3)]
    fb = [db.add_speaker_embedding(sid, _near(b, 60 + i), None, 5.0, origin="voice") for i in range(2)]
    lone = db.add_speaker_embedding(sid, _voice(5), "x.wav", 5.0, origin="label")
    rows = {r["id"]: r for r in client.get(f"/api/speakers/{sid}/samples").json()}
    assert rows[lone]["origin"] == "label" and rows[lone]["outlier"] is True and rows[lone]["ambiguous"] is False
    assert all(rows[i]["origin"] == "voice" and rows[i]["ambiguous"] is True and rows[i]["outlier"] is False for i in fa + fb)
    sp = {s["name"]: s for s in client.get("/api/speakers").json()["speakers"]}
    assert sp["谁"]["outlier_samples"] == 1 and sp["谁"]["ambiguous_samples"] == 5
    assert client.delete(f"/api/speakers/{sid}/samples/outliers").json() == {"ok": True, "deleted": 1, "remaining": 5}
    one = db.upsert_speaker("一")
    db.add_speaker_embedding(one, _voice(9), None, 1.0, origin="mic")
    r = client.get(f"/api/speakers/{one}/samples").json()
    assert r[0]["origin"] == "mic" and r[0]["ambiguous"] is False and r[0]["consistency"] is None
    sp = {s["name"]: s for s in client.get("/api/speakers").json()["speakers"]}
    assert sp["一"]["ambiguous_samples"] == 0 and sp["一"]["outlier_samples"] == 0


def test_api_force_must_be_a_json_boolean(client: TestClient, rt: Runtime, monkeypatch):
    db = rt.db
    eng = MismatchEngine()
    monkeypatch.setattr(rt, "engine", lambda: eng)
    seg, _ = _seg(rt.cfg, db, datetime(2026, 10, 30, 9, 0), _dc(6.0, 0.2))
    u1 = _utt(db, seg, DAY, "09:00", "一", None, None, 0.0, 3.0, cluster=0)
    for bad in ("yes", "ok", 2, [True], {"a": 1}, 1.5):
        r = client.post("/api/speakers/label", json={"utterance_id": u1, "name": "甲", "force": bad})
        assert r.status_code == 400 and r.json()["error"] == "force 应为 true/false", (bad, r.text)
        r = client.post("/api/voices/name", json={"members": [[seg, 0]], "name": "甲", "force": bad})
        assert r.status_code == 400 and r.json()["error"] == "force 应为 true/false", (bad, r.text)
    r = client.post("/api/speakers/甲/samples", params={"filename": "a.wav", "force": "maybe"}, content=_wav(_dc(3.0, 0.2)))
    assert r.status_code == 400 and r.json()["error"] == "force 应为 true/false"
    assert eng.calls == []
    for value, expect in ((True, True), (False, False), (1, True), (0, False), ("true", True), ("0", False)):
        r = client.post("/api/speakers/label", json={"utterance_id": u1, "name": f"人{len(eng.calls)}", "force": value})
        assert r.status_code == 200, r.text
        assert eng.calls[-1][1] is expect
    assert rt.wait_jobs(5)


def test_api_split_runs_under_the_scheduler_lock_preview_does_not(client: TestClient, rt: Runtime, monkeypatch):
    import audio_note.speech.asr as asr_mod

    db = rt.db
    a = db.upsert_speaker("甲")
    db.add_speaker_embedding(a, _voice(1), None, 3.0)
    monkeypatch.setattr(rt, "engine", lambda: MismatchEngine(a))
    monkeypatch.setattr(rt, "client", lambda: object())

    class Lock:
        held = 0
        entered = 0

        def __enter__(self):
            Lock.entered += 1
            Lock.held += 1
            return self

        def __exit__(self, *exc):
            Lock.held -= 1
            return False

    class FakeASRClass:
        def __init__(self, cfg, client, vad=None):
            self.inner = FakeASR("拆")

        def transcribe(self, audio):
            assert Lock.held == 1                      # ASR (and the DB write after it) happen under the lock
            return self.inner.transcribe(audio)

    monkeypatch.setattr(asr_mod, "ASR", FakeASRClass)
    monkeypatch.setattr(rt.sched, "lock", Lock())
    seg, _ = _seg(rt.cfg, db, datetime(2026, 10, 30, 9, 0), np.concatenate([_dc(3.0, 0.2), _dc(3.0, -0.2)]))
    mixed = _utt(db, seg, DAY, "09:00", "混合", "甲", a, 0.0, 6.0, 0.6, cluster=0)
    assert client.get(f"/api/utterances/{mixed}/preview-split", params={"at": "3.0"}).status_code == 200
    assert Lock.entered == 0
    r = client.post(f"/api/utterances/{mixed}/split", json={"at": [3.0]})
    assert r.status_code == 200, r.text
    assert Lock.entered == 1 and Lock.held == 0
    r = client.get(f"/api/audio/utterance/{r.json()['pieces'][0]['id']}")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    assert rt.wait_jobs(5)
