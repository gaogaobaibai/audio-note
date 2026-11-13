"""Speakers v4 (docs/SPEC_speakers_v4_negatives.md): negative samples and per-sentence vetoes (A), the review queue (B),
sample influence (C) and the pipeline side of the system-audio share (D.3).

No network, no microphone, no model files: `NumpyEngine` is the real SpeakerEngine identification logic with the
embedding extractor replaced by a table (a clip's mean DC level picks its voice vector), so every decision below runs
through the production code paths.
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
from audio_note.runtime import Runtime
from audio_note.speech import review, voices
from audio_note.speech.asr import ASRResult
from audio_note.speech.pipeline import Processor, channel_pairs, system_share
from audio_note.speech.speakers import SpeakerEngine, Turn, audit_samples
from audio_note.ui.app import create_app

DAY = "2026-11-04"


# --- vectors ------------------------------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _voice(seed: int, dim: int = 192) -> np.ndarray:
    return _unit(np.random.default_rng(seed).standard_normal(dim))


def _at(base: np.ndarray, seed: int, cos: float) -> np.ndarray:
    """A unit vector whose cosine to `base` is exactly `cos`."""
    o = np.random.default_rng(seed).standard_normal(base.shape[0])
    o = o - float(np.dot(o, base)) * base
    o = o / np.linalg.norm(o)
    return _unit(cos * base + np.sqrt(max(0.0, 1.0 - cos * cos)) * o)


A = _voice(1)                 # 甲 (the user, "self")
X = _at(A, 50, 0.55)          # somebody else whose voice passes as 甲 (0.55 >= 0.45): the false positive
Y = _at(A, 60, 0.30)          # an unrelated voice


# --- engine / data helpers --------------------------------------------------------------------------------------

class NumpyEngine(SpeakerEngine):
    """SpeakerEngine without models: `embed` maps a clip's mean level (rounded to 0.05) to a voice from `table`."""

    def __init__(self, cfg, table: dict[float, np.ndarray], turns: list[Turn] | None = None):
        self.cfg = cfg
        self.table = dict(table)
        self.turns = turns or []
        self.dim = 192
        self._diar = None
        self._vad = None
        self.embeds = 0

    @staticmethod
    def key(audio: np.ndarray) -> float:
        return round(round(float(np.mean(audio)) * 20) / 20, 2)

    def embed(self, audio):
        self.embeds += 1
        v = self.table.get(self.key(audio))
        return v if v is not None else _voice(900 + int(abs(self.key(audio)) * 100))

    def speech_only(self, audio, max_s=60.0):
        return np.asarray(audio, dtype=np.float32)[: int(max_s * SR)]

    def diarize(self, audio):
        return [Turn(t.start, t.end, t.cluster) for t in self.turns]


def _dc(seconds: float, level: float) -> np.ndarray:
    return np.full(int(round(seconds * SR)), level, dtype=np.float32)


def _seg(cfg, db: DB, when: datetime, audio: np.ndarray, status: str = "done", write: bool = True) -> tuple[int, Path]:
    p = segment_path(cfg, when, ".wav")
    if write:
        sf.write(p, audio, SR, subtype="PCM_16")
    sid = db.add_segment(p, when, len(audio) / SR, speech_s=len(audio) / SR)
    if status != "new":
        db.set_segment_status(sid, status)
    return sid, p


def _utt(db: DB, seg: int, hhmmss: str, text: str, who: str | None = None, speaker_id: int | None = None,
         start_s: float = 0.0, end_s: float = 1.0, score: float | None = None, cluster: int = 0, source: str | None = None,
         loudness: float | None = None) -> int:
    return db.add_utterances([{
        "segment_id": seg, "start_s": start_s, "end_s": end_s, "abs_start": f"{DAY}T{hhmmss}", "day": DAY,
        "cluster_label": cluster, "speaker_id": speaker_id, "speaker_name": who, "speaker_score": score,
        "text_raw": text, "text": text, "text_seg": text, "speaker_source": source, "loudness_db": loudness,
    }])[0]


def _row(db: DB, uid: int):
    return db.one("SELECT * FROM utterances WHERE id=?", (uid,))


def _channels(pairs: list[tuple[float, float]]) -> bytes:
    return np.asarray([x for p in pairs for x in p], dtype=np.float16).tobytes()


@pytest.fixture
def ncfg(cfg_factory, tmp_path: Path):
    cfg = cfg_factory({"memory": {"retrieval": "fts", "embedding_model": "hash"},
                       "correction": {"enabled": False, "glossary_file": str(tmp_path / "glossary.txt")}})
    yield cfg
    from audio_note.logging_setup import _remove_own_handlers

    _remove_own_handlers(logging.getLogger())


@pytest.fixture
def ndb(ncfg) -> DB:
    return DB(ncfg.db_path)


@pytest.fixture
def jia(ndb: DB) -> int:
    """甲 with two trusted (mic) samples of voice A."""
    sid = ndb.upsert_speaker("甲")
    ndb.add_speaker_embedding(sid, A, "mic1.wav", 10.0, origin="mic")
    ndb.add_speaker_embedding(sid, _at(A, 2, 0.95), "mic2.wav", 10.0, origin="mic")
    return sid


# =========================================================================================================
# A2. match with negative samples
# =========================================================================================================

def test_match_without_negatives_is_unchanged():
    enrolled = [(1, "甲", A), (1, "甲", _at(A, 3, 0.9)), (2, "乙", Y)]
    for probe in (A, X, Y, _voice(77)):
        old_best = max(((sid, float(np.dot(probe, _unit(e)))) for sid, _n, e in enrolled), key=lambda kv: kv[1])
        for negs in (None, []):
            sid, name, score = SpeakerEngine.match(probe, enrolled, 0.45, negs)
            assert score == pytest.approx(old_best[1], abs=1e-6)
            assert sid == (old_best[0] if old_best[1] >= 0.45 else None)
    assert SpeakerEngine.match(A, [], 0.45, [(1, A)]) == (None, None, None)


def test_match_negative_margin_boundary_and_fallback_to_second_candidate():
    e = _at(A, 4, 0.60)                     # p(甲) = 0.60
    neg = _at(e, 5, 0.50)                   # n(甲) = 0.50 -> p - n = 0.10
    b = _at(e, 6, 0.50)                     # 乙's sample: p(乙) = 0.50
    enrolled = [(1, "甲", A), (2, "乙", b)]
    p1 = float(np.dot(e, A))
    assert SpeakerEngine.match(e, enrolled, 0.45, [(1, neg)], margin=0.09)[:2] == (1, "甲")
    sid, name, score = SpeakerEngine.match(e, enrolled, 0.45, [(1, neg)], margin=0.11)
    assert (sid, name) == (2, "乙") and score == pytest.approx(0.50, abs=1e-5)     # downgraded to the next candidate
    sid, name, score = SpeakerEngine.match(e, enrolled, 0.55, [(1, neg)], margin=0.11)
    assert sid is None and score == pytest.approx(p1, abs=1e-5)                   # nobody qualifies: score is still p (best)
    # a negative of another speaker does not matter; a negative of another dimension is ignored
    assert SpeakerEngine.match(e, enrolled, 0.45, [(2, e), (1, np.ones(8, dtype=np.float32))])[:2] == (1, "甲")
    # exclude removes a candidate entirely
    assert SpeakerEngine.match(e, enrolled, 0.45, exclude={1})[:2] == (2, "乙")
    assert SpeakerEngine.match(e, enrolled, 0.45, exclude={1, 2}) == (None, None, None)


def test_identify_turns_respects_negatives_and_turn_exclusions(ncfg, ndb, jia):
    yi = ndb.upsert_speaker("乙")
    ndb.add_speaker_embedding(yi, _at(X, 7, 0.50), None, 5.0)
    audio = np.concatenate([_dc(3.0, 0.2), _dc(3.0, 0.4)])          # 0-3 s voice A, 3-6 s voice X
    eng = NumpyEngine(ncfg, {0.2: A, 0.4: X})
    turns = [Turn(0.0, 3.0, 0), Turn(3.0, 6.0, 1)]
    eng.identify_turns(audio, turns, ndb)
    assert [t.speaker_id for t in turns] == [jia, jia]               # X passes as 甲 ...
    ndb.add_speaker_negative(jia, X, None, 3.0)
    turns = [Turn(0.0, 3.0, 0), Turn(3.0, 6.0, 1)]
    eng.identify_turns(audio, turns, ndb)
    assert [t.speaker_id for t in turns] == [jia, yi]                # ... until X is a negative of 甲: next candidate 乙
    # a turn excluding 甲 (system audio) - also a short one that only has the cluster decision
    ndb.execute("DELETE FROM speaker_negatives")
    turns = [Turn(0.0, 3.0, 0, exclude=frozenset({jia})), Turn(3.0, 6.0, 1)]
    eng.identify_turns(audio, turns, ndb)
    assert turns[0].speaker_id != jia
    turns = [Turn(0.0, 0.8, 0, exclude=frozenset({jia})), Turn(0.8, 3.0, 0)]
    eng.identify_turns(audio, turns, ndb)
    assert turns[0].speaker_id is None and turns[1].speaker_id == jia and turns[0].cluster != turns[1].cluster
    # identify_utterances (relabel) too, cluster ids unchanged
    turns = [Turn(0.0, 0.8, 0, exclude=frozenset({jia})), Turn(0.8, 3.0, 0)]
    eng.identify_utterances(audio, turns, ndb)
    assert turns[0].speaker_id is None and turns[1].speaker_id == jia and turns[0].cluster == turns[1].cluster == 0


# =========================================================================================================
# A1 / A2. DB helpers, vetoes in relabel
# =========================================================================================================

def test_db_negative_and_rejection_helpers_and_cascade(ndb, jia):
    seg = ndb.add_segment("x.wav", datetime(2026, 11, 4, 9), 2.0)
    u1 = _utt(ndb, seg, "09:00:00", "一", "甲", jia, score=0.6)
    u2 = _utt(ndb, seg, "09:00:01", "二", "未知", None)
    nid = ndb.add_speaker_negative(jia, X * 3.0, "x.wav#0.0-1.0", 1.0, utterance_id=u1, segment_id=seg)
    (sid, emb), = ndb.speaker_negatives()
    assert sid == jia and np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5) and float(np.dot(emb, X)) == pytest.approx(1.0, abs=1e-5)
    assert ndb.reject_utterances(jia, [u1, u2, u1], "未知") == 1                  # only the row attributed to 甲 is released
    r = _row(ndb, u1)
    assert (r["speaker_id"], r["speaker_name"], r["speaker_score"], r["speaker_source"]) == (None, "未知", None, None)
    assert ndb.reject_utterances(jia, [u1], "未知") == 0                          # idempotent (INSERT OR IGNORE)
    assert ndb.rejected_speakers([u1, u2, 999]) == {u1: {jia}, u2: {jia}}
    assert ndb.delete_speaker_negative(nid) == 1 and ndb.speaker_negatives() == []
    ndb.add_speaker_negative(jia, X, None, 1.0)
    ndb.delete_speaker(jia, "未知")                                                # FK cascade
    assert ndb.one("SELECT COUNT(*) AS n FROM speaker_negatives")["n"] == 0
    assert ndb.one("SELECT COUNT(*) AS n FROM utterance_rejections")["n"] == 0


def test_relabel_never_assigns_a_vetoed_speaker(ncfg, ndb, jia):
    audio = np.concatenate([_dc(3.0, 0.2), _dc(3.0, 0.2)])
    seg, _p = _seg(ncfg, ndb, datetime(2026, 11, 4, 9), audio)
    u1 = _utt(ndb, seg, "09:00:00", "一", "甲", jia, 0.0, 3.0, 0.9)
    u2 = _utt(ndb, seg, "09:00:03", "二", "甲", jia, 3.0, 6.0, 0.9)
    ndb.reject_utterances(jia, [u1], "未知")
    eng = NumpyEngine(ncfg, {0.2: A})
    Processor(ncfg, ndb, engine=eng).relabel_segment(seg)
    assert _row(ndb, u1)["speaker_id"] is None and _row(ndb, u2)["speaker_id"] == jia

    class Stubborn:
        """Ignores Turn.exclude: says 甲 for everything (the write-back must still enforce the veto)."""

        def identify_utterances(self, audio, turns, db, segment_id=None):
            for t in turns:
                t.speaker_id, t.speaker_name, t.score = jia, "甲", 0.9
            return turns

    Processor(ncfg, ndb, engine=Stubborn()).relabel_segment(seg)
    assert _row(ndb, u1)["speaker_id"] is None and _row(ndb, u1)["speaker_name"] == "未知"
    assert _row(ndb, u2)["speaker_id"] == jia


# =========================================================================================================
# A3. reject_voice / reject_voice_group
# =========================================================================================================

def test_reject_voice_whole_cluster_scope_and_hand_label_protection_without_audio(ncfg, ndb, jia):
    seg, path = _seg(ncfg, ndb, datetime(2026, 11, 4, 9), _dc(10, 0.4), write=False)        # recording expired
    u1 = _utt(ndb, seg, "09:00:00", "一", "甲", jia, 0, 2, 0.5, cluster=0)
    u2 = _utt(ndb, seg, "09:00:02", "二", "甲", jia, 2, 4, 0.5, cluster=0)
    u3 = _utt(ndb, seg, "09:00:04", "三", "甲", jia, 4, 6, None, cluster=0, source="user")   # the user said: 甲
    u4 = _utt(ndb, seg, "09:00:06", "四", "甲", jia, 6, 8, 0.5, cluster=1)
    res = voices.reject_voice(ndb, ncfg, NumpyEngine(ncfg, {0.4: X}), jia, [u1], whole_cluster=True)
    assert res["rejected"] == 2 and res["released"] == 2 and res["negatives_added"] == 0 and res["relabeled_changed"] == 0
    assert "warning" not in res
    assert [_row(ndb, u)["speaker_id"] for u in (u1, u2, u3, u4)] == [None, None, jia, jia]
    assert ndb.rejected_speakers([u1, u2, u3, u4]) == {u1: {jia}, u2: {jia}}
    assert (ncfg.transcripts_dir / f"{DAY}.md").exists()
    with pytest.raises(LookupError):
        voices.reject_voice(ndb, ncfg, None, 999, [u1])
    with pytest.raises(LookupError):
        voices.reject_voice(ndb, ncfg, None, jia, [12345])
    with pytest.raises(ValueError):
        voices.reject_voice(ndb, ncfg, None, jia, [])


def test_reject_voice_stores_negative_and_relabels_the_rest(ncfg, ndb, jia):
    audio = np.concatenate([_dc(3.0, 0.4), _dc(3.0, 0.4), _dc(3.0, 0.2)])
    seg, _p = _seg(ncfg, ndb, datetime(2026, 11, 4, 9), audio)
    u1 = _utt(ndb, seg, "09:00:00", "别人一", "甲", jia, 0, 3, 0.55, cluster=0)
    u2 = _utt(ndb, seg, "09:00:03", "别人二", "甲", jia, 3, 6, 0.55, cluster=0)
    u3 = _utt(ndb, seg, "09:00:06", "我自己", "甲", jia, 6, 9, 0.99, cluster=1)
    eng = NumpyEngine(ncfg, {0.4: X, 0.2: A})
    res = voices.reject_voice(ndb, ncfg, eng, jia, [u1])
    assert res["rejected"] == 1 and res["negatives_added"] == 1 and "warning" not in res
    assert res["relabeled_changed"] == 1                                      # u2 (same voice) is no longer 甲
    (n_sid, n_emb), = ndb.speaker_negatives()
    assert n_sid == jia and float(np.dot(n_emb, X)) == pytest.approx(1.0, abs=1e-4)
    neg = ndb.one("SELECT * FROM speaker_negatives")
    assert neg["utterance_id"] == u1 and neg["segment_id"] == seg and neg["source_path"].endswith("#0.0-3.0")
    assert [_row(ndb, u)["speaker_id"] for u in (u1, u2, u3)] == [None, None, jia]


def test_reject_voice_similar_to_the_reference_warns_and_only_vetoes(ncfg, ndb, jia):
    seg, _p = _seg(ncfg, ndb, datetime(2026, 11, 4, 9), _dc(3.0, 0.2))
    u1 = _utt(ndb, seg, "09:00:00", "其实是我", "甲", jia, 0, 3, 0.95)
    res = voices.reject_voice(ndb, ncfg, NumpyEngine(ncfg, {0.2: A}), jia, [u1])
    assert res["negatives_added"] == 0 and "甲" in res["warning"] and "确定不是 TA" in res["warning"]
    assert ndb.speaker_negatives() == [] and ndb.rejected_speakers([u1]) == {u1: {jia}}
    assert _row(ndb, u1)["speaker_id"] is None                               # the veto still holds after relabel


def test_reject_voice_group_and_closest_hint(ncfg, ndb, jia):
    seg, _p = _seg(ncfg, ndb, datetime(2026, 11, 4, 9), _dc(6.0, 0.4))
    u1 = _utt(ndb, seg, "09:00:00", "一", "未知", None, 0, 3, cluster=0)
    u2 = _utt(ndb, seg, "09:00:03", "二", "未知", None, 3, 6, cluster=0)
    ndb.upsert_cluster_embedding(seg, 0, X, 6.0)
    g = voices.group_unknown_voices(ndb, ncfg)["groups"]
    assert g[0]["closest"]["id"] == jia                                      # "maybe 甲" ...
    res = voices.reject_voice_group(ndb, ncfg, NumpyEngine(ncfg, {0.4: X}), g[0]["members"], jia)
    assert res["rejected"] == 2 and res["negatives_added"] == 1
    assert ndb.rejected_speakers([u1, u2]) == {u1: {jia}, u2: {jia}}
    g = voices.group_unknown_voices(ndb, ncfg)["groups"]
    assert g[0]["closest"] is None                                           # ... not after 「不是 甲」
    with pytest.raises(ValueError):
        voices.reject_voice_group(ndb, ncfg, None, [[seg, 5]], jia)          # no unidentified sentence there
    # the hint also honours negatives alone (no veto on these sentences)
    seg2, _p = _seg(ncfg, ndb, datetime(2026, 11, 4, 10), _dc(3.0, 0.4))
    _utt(ndb, seg2, "10:00:00", "三", "未知", None, 0, 3, cluster=0)
    ndb.upsert_cluster_embedding(seg2, 0, _at(X, 9, 0.98), 3.0)
    groups = voices.group_unknown_voices(ndb, ncfg, threshold=0.999)["groups"]
    assert all(gr["closest"] is None or gr["closest"]["id"] != jia for gr in groups)


# =========================================================================================================
# B. review queue + confirm
# =========================================================================================================

def test_review_queue_order_and_filters(ncfg, ndb, jia):
    seg, _p = _seg(ncfg, ndb, datetime(2026, 11, 4, 9), _dc(30, 0.2))
    gone, _p2 = _seg(ncfg, ndb, datetime(2026, 11, 4, 10), _dc(5, 0.2), write=False)
    s50 = _utt(ndb, seg, "09:00:00", "0.50", "甲", jia, 0, 2, 0.50)
    snull = _utt(ndb, seg, "09:00:02", "null", "甲", jia, 2, 4, None)
    s46_long = _utt(ndb, seg, "09:00:04", "0.46 长", "甲", jia, 4, 8, 0.46)
    s46_short = _utt(ndb, seg, "09:00:08", "0.46 短", "甲", jia, 8, 9, 0.46)
    s90 = _utt(ndb, seg, "09:00:09", "0.90", "甲", jia, 9, 11, 0.90, loudness=-20.0)
    _utt(ndb, seg, "09:00:11", "太短", "甲", jia, 11, 11.5, 0.3)                          # < 0.8 s
    _utt(ndb, seg, "09:00:12", "手标", "甲", jia, 12, 14, None, source="user")             # confirmed already
    _utt(ndb, gone, "10:00:00", "没音频", "甲", jia, 0, 2, 0.1)                           # no audio
    vetoed = _utt(ndb, seg, "09:00:14", "否决过", "甲", jia, 14, 16, 0.2)
    ndb.execute("INSERT INTO utterance_rejections(utterance_id, speaker_id) VALUES (?,?)", (vetoed, jia))
    items = review.review_queue(ndb, ncfg, jia)
    assert [i["utterance_id"] for i in items] == [snull, s46_short, s46_long, s50, s90]
    assert review.review_total(ndb, ncfg, jia) == 5
    last = items[-1]
    # speakers v5 added kind / influence / guess_score to every item (the queue also holds "maybe" sentences now)
    assert last == {"utterance_id": s90, "day": DAY, "time": "09:00:09", "text": "0.90", "seconds": 2.0, "score": 0.9,
                    "loudness_db": -20.0, "system_share": None, "audio": True,
                    "kind": "sure", "influence": 0, "guess_score": None}
    assert [i["utterance_id"] for i in review.review_queue(ndb, ncfg, jia, limit=2, offset=1)] == [s46_short, s46_long]


def test_confirm_marks_user_and_adds_a_sample_only_when_long_and_alike(ncfg, ndb, jia):
    audio = np.concatenate([_dc(4.0, 0.2), _dc(1.0, 0.2), _dc(4.0, 0.3), _dc(3.0, 0.2)])
    seg, _p = _seg(ncfg, ndb, datetime(2026, 11, 4, 9), audio)
    long_ = _utt(ndb, seg, "09:00:00", "长句", "甲", jia, 0, 4, 0.5, cluster=0)
    short = _utt(ndb, seg, "09:00:04", "短句", "甲", jia, 4, 5, 0.5, cluster=1)
    other = _utt(ndb, seg, "09:00:05", "别人", "甲", jia, 5, 9, 0.5, cluster=2)
    same_cluster = _utt(ndb, seg, "09:00:09", "同簇", "甲", jia, 9, 12, 0.5, cluster=0)
    eng = NumpyEngine(ncfg, {0.2: A, 0.3: Y})
    n0 = ndb.one("SELECT COUNT(*) AS n FROM speaker_embeddings WHERE speaker_id=?", (jia,))["n"]
    res = review.confirm_utterance(ndb, ncfg, eng, long_, jia, whole_cluster=True)
    assert res["sample_added"] is True and res["confirmed"] == 2
    assert _row(ndb, long_)["speaker_source"] == "user" and _row(ndb, same_cluster)["speaker_source"] == "user"
    assert _row(ndb, long_)["speaker_score"] == 0.5                                   # attribution (and score) kept
    assert ndb.one("SELECT COUNT(*) AS n FROM speaker_embeddings WHERE speaker_id=?", (jia,))["n"] == n0 + 1
    res = review.confirm_utterance(ndb, ncfg, eng, short, jia)
    assert res["sample_added"] is False and "1.0s" in res["sample_note"] and _row(ndb, short)["speaker_source"] == "user"
    res = review.confirm_utterance(ndb, ncfg, eng, other, jia)                          # voice Y: unlike 甲 -> no sample
    assert res["sample_added"] is False and res["sample_note"] and _row(ndb, other)["speaker_source"] == "user"
    assert ndb.one("SELECT COUNT(*) AS n FROM speaker_embeddings WHERE speaker_id=?", (jia,))["n"] == n0 + 1
    # confirming a vetoed sentence lifts the veto and assigns it
    ndb.reject_utterances(jia, [short], "未知")
    res = review.confirm_utterance(ndb, ncfg, None, short, jia)
    assert _row(ndb, short)["speaker_id"] == jia and ndb.rejected_speakers([short]) == {}
    with pytest.raises(LookupError):
        review.confirm_utterance(ndb, ncfg, None, 99999, jia)


# =========================================================================================================
# C. sample influence
# =========================================================================================================

def test_audit_influence_flags_the_side_door_sample(ndb, jia):
    x_emb = _at(A, 11, 0.37)                              # the side door: a labelled clip of somebody else
    side = ndb.add_speaker_embedding(jia, x_emb, "rec.opus#1.0-3.0", 2.0, origin="label")
    seg = ndb.add_segment("s.wav", datetime(2026, 11, 4, 9), 600.0)
    k = 0
    for cl in range(3):                                   # 3 clusters that only the side-door sample lets through: 8 sentences each
        ndb.upsert_cluster_embedding(seg, cl, _at(x_emb, 20 + cl, 0.9), 10.0)
        for _ in range(8):
            _utt(ndb, seg, f"09:{k // 60:02d}:{k % 60:02d}", "像", "甲", jia, k, k + 1, 0.5, cluster=cl)
            k += 1
    ndb.upsert_cluster_embedding(seg, 9, A, 10.0)        # the real voice: matched by both mic samples
    for _ in range(5):
        _utt(ndb, seg, f"09:{k // 60:02d}:{k % 60:02d}", "我", "甲", jia, k, k + 1, 0.9, cluster=9)
        k += 1
    _utt(ndb, seg, "10:00:00", "手标", "甲", jia, 900, 901, None, cluster=0, source="user")   # hand-labelled: not counted
    rows = {r["id"]: r for r in audit_samples(ndb, jia, 0.5, 0.45)}
    assert rows[side]["influence"] == 24 and rows[side]["outlier"] is True
    assert all(r["influence"] == 0 for i, r in rows.items() if i != side)
    assert audit_samples(ndb, jia, 0.5, 0.95)[2]["influence"] == 0          # nothing reaches a 0.95 threshold


# =========================================================================================================
# D.3 system share
# =========================================================================================================

def test_system_share_computation():
    pairs = channel_pairs(_channels([(0.2, 0.0), (0.2, 0.0), (0.01, 0.2), (0.0, 0.2)]))
    assert pairs.shape == (4, 2)
    assert system_share(pairs, 0.5, 0.0, 1.0) == 0.0
    assert system_share(pairs, 0.5, 1.0, 2.0) == pytest.approx(0.08 / (0.08 + 0.0001), abs=1e-3)
    # half a block of each kind: weighted by overlap
    assert system_share(pairs, 0.5, 0.75, 1.25) == pytest.approx(0.25 * 0.04 / (0.25 * 0.04 + 0.25 * 0.04 + 0.25 * 0.0001), abs=2e-3)
    assert system_share(pairs, 0.5, 5.0, 6.0) is None                        # outside the recorded blocks
    assert system_share(None, 0.5, 0.0, 1.0) is None and system_share(pairs, None, 0.0, 1.0) is None
    assert system_share(channel_pairs(_channels([(0.0, 0.0)])), 0.5, 0.0, 0.5) is None   # silence
    assert channel_pairs(None) is None and channel_pairs(b"") is None


class _ASR:
    def __init__(self):
        self.n = 0

    def transcribe(self, audio):
        self.n += 1
        return ASRResult(f"句子{self.n}", round(len(audio) / SR, 1), 1)


def _meeting_segment(ncfg, ndb, with_channels: bool = True, hour: int = 9) -> int:
    """6 s: 0-3 s the user at the microphone (voice A), 3-6 s the other side of a call whose voice also resembles A."""
    audio = np.concatenate([_dc(3.0, 0.2), _dc(3.0, 0.2)])
    seg, _p = _seg(ncfg, ndb, datetime(2026, 11, 4, hour), audio, status="new")
    if with_channels:
        ch = _channels([(0.2, 0.0)] * 6 + [(0.01, 0.2)] * 6)
        ndb.execute("UPDATE segments SET channel_rms=?, channel_block_s=? WHERE id=?", (ch, 0.5, seg))
    return seg


def test_pipeline_system_turns_are_never_the_self_speaker(ncfg, ndb, jia):
    ndb.set_self_speaker(jia)
    yi = ndb.upsert_speaker("乙")
    ndb.add_speaker_embedding(yi, _at(A, 12, 0.5), None, 5.0)
    seg = _meeting_segment(ncfg, ndb)
    eng = NumpyEngine(ncfg, {0.2: A}, turns=[Turn(0.0, 3.0, 0), Turn(3.0, 6.0, 1)])
    assert Processor(ncfg, ndb, engine=eng, asr=_ASR()).process_segment(seg) == 2
    rows = ndb.query("SELECT * FROM utterances WHERE segment_id=? ORDER BY start_s", (seg,))
    assert [r["speaker_id"] for r in rows] == [jia, yi]                      # the system turn falls to the next candidate
    assert rows[0]["system_share"] == pytest.approx(0.0) and rows[1]["system_share"] > 0.99
    # relabel keeps the rule, and fills system_share
    ndb.execute("UPDATE utterances SET speaker_id=?, speaker_name='甲', system_share=NULL WHERE segment_id=?", (jia, seg))
    Processor(ncfg, ndb, engine=eng).relabel_segment(seg)
    rows = ndb.query("SELECT * FROM utterances WHERE segment_id=? ORDER BY start_s", (seg,))
    assert [r["speaker_id"] for r in rows] == [jia, yi] and rows[1]["system_share"] > 0.99


def test_pipeline_system_share_safety_net_and_no_channels(ncfg, ndb, jia):
    ndb.set_self_speaker(jia)

    class SaysSelf:
        """Ignores Turn.exclude and calls every turn 甲 (one cluster): the pipeline must still keep the system turn off 甲."""

        def diarize(self, audio):
            return [Turn(0.0, 3.0, 0), Turn(3.2, 6.0, 0)]

        def identify_turns(self, audio, turns, db, segment_id=None):
            for t in turns:
                t.speaker_id, t.speaker_name, t.score = jia, "甲", 0.9
            return turns

    seg = _meeting_segment(ncfg, ndb)
    Processor(ncfg, ndb, engine=SaysSelf(), asr=_ASR()).process_segment(seg)
    rows = ndb.query("SELECT * FROM utterances WHERE segment_id=? ORDER BY start_s", (seg,))
    assert [r["speaker_id"] for r in rows] == [jia, None] and rows[1]["speaker_name"] == "未知"
    assert rows[0]["cluster_label"] != rows[1]["cluster_label"]               # not merged back into the user's chunk
    # without channel data (mic only / imported): no restriction, system_share NULL
    seg2 = _meeting_segment(ncfg, ndb, with_channels=False, hour=10)
    Processor(ncfg, ndb, engine=SaysSelf(), asr=_ASR()).process_segment(seg2)
    rows = ndb.query("SELECT * FROM utterances WHERE segment_id=? ORDER BY start_s", (seg2,))
    assert len(rows) == 1 and rows[0]["speaker_id"] == jia and rows[0]["system_share"] is None


# =========================================================================================================
# API
# =========================================================================================================

@pytest.fixture
def api(ncfg):
    rt = Runtime(ncfg)
    with TestClient(create_app(rt), raise_server_exceptions=False) as c:
        yield c, rt
    rt.shutdown()


def test_api_reject_negatives_review_confirm(api, ncfg):
    client, rt = api
    db = rt.db
    jia = db.upsert_speaker("甲")
    db.add_speaker_embedding(jia, A, None, 10.0, origin="mic")
    audio = np.concatenate([_dc(3.0, 0.4), _dc(3.0, 0.4), _dc(4.0, 0.2)])
    seg, _p = _seg(ncfg, db, datetime(2026, 11, 4, 9), audio)
    u1 = _utt(db, seg, "09:00:00", "别人一", "甲", jia, 0, 3, 0.55, cluster=0)
    u2 = _utt(db, seg, "09:00:03", "别人二", "甲", jia, 3, 6, 0.55, cluster=0)
    u3 = _utt(db, seg, "09:00:06", "我", "甲", jia, 6, 10, 0.9, cluster=1)
    nobody = _utt(db, seg, "09:00:10", "没人", "未知", None, 6, 7, cluster=3)
    rt._engine = NumpyEngine(ncfg, {0.4: X, 0.2: A})

    assert client.post("/api/utterances/99999/reject", json={}).status_code == 404
    r = client.post(f"/api/utterances/{nobody}/reject", json={})
    assert r.status_code == 400 and "speaker_id" in r.json()["error"]
    assert client.post(f"/api/utterances/{u1}/reject", json={"speaker_id": 999}).status_code == 404
    assert client.post(f"/api/utterances/{u1}/reject", json={"whole_cluster": "maybe"}).status_code == 400

    review_before = client.get(f"/api/speakers/{jia}/review").json()
    assert review_before["total"] == 3 and [i["utterance_id"] for i in review_before["items"]] == [u1, u2, u3]
    assert client.get("/api/speakers/999/review").status_code == 404

    r = client.post(f"/api/utterances/{u1}/reject", json={"whole_cluster": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] and body["speaker_id"] == jia and body["rejected"] == 2 and body["negatives_added"] == 1
    sp = {s["id"]: s for s in client.get("/api/speakers").json()["speakers"]}[jia]
    assert sp["negatives"] == 1 and sp["rejections"] == 2 and sp["risky_samples"] == 0

    negs = client.get(f"/api/speakers/{jia}/negatives").json()
    assert len(negs) == 1 and negs[0]["audio"] is True and negs[0]["utterance_id"] in (u1, u2) and negs[0]["text"]
    assert set(negs[0]) >= {"id", "duration_s", "source_path", "file", "audio", "utterance_id", "created_at"}
    wav = client.get(f"/api/speakers/negatives/{negs[0]['id']}/audio")
    assert wav.status_code == 200 and wav.headers["content-type"] == "audio/wav"
    assert client.get("/api/speakers/negatives/999/audio").status_code == 404
    assert client.get("/api/speakers/999/negatives").status_code == 404

    tr = client.get(f"/api/transcript/{DAY}").json()
    utts = [u for s in tr["sessions"] for u in s["utterances"]]
    assert all("system_share" in u for u in utts)

    # confirm
    assert client.post(f"/api/utterances/{u3}/confirm", json={}).status_code == 400
    assert client.post("/api/utterances/99999/confirm", json={"speaker_id": jia}).status_code == 404
    r = client.post(f"/api/utterances/{u3}/confirm", json={"speaker_id": jia})
    assert r.status_code == 200 and r.json()["sample_added"] is True
    left = client.get(f"/api/speakers/{jia}/review").json()     # the new sample re-identified the segment:
    assert [i["utterance_id"] for i in left["items"]] == [nobody]  # only the unnamed sentence in 甲's voice is new

    assert client.delete(f"/api/speakers/negatives/{negs[0]['id']}").json()["deleted"] == 1
    assert client.get(f"/api/speakers/{jia}/negatives").json() == []


def test_api_voices_reject_and_samples_influence(api, ncfg):
    client, rt = api
    db = rt.db
    jia = db.upsert_speaker("甲")
    db.add_speaker_embedding(jia, A, None, 10.0, origin="mic")
    db.add_speaker_embedding(jia, _at(A, 2, 0.95), None, 10.0, origin="mic")
    z = _at(A, 13, 0.37)
    side = db.add_speaker_embedding(jia, z, None, 2.0, origin="label")
    seg, _p = _seg(ncfg, db, datetime(2026, 11, 4, 9), _dc(30.0, 0.4))
    db.upsert_cluster_embedding(seg, 0, _at(z, 3, 0.9), 25.0)
    for k in range(25):
        _utt(db, seg, f"09:00:{k:02d}", "侧门", "甲", jia, k, k + 1, 0.5, cluster=0)
    rows = {r["id"]: r for r in client.get(f"/api/speakers/{jia}/samples").json()}
    assert rows[side]["outlier"] is True and rows[side]["influence"] == 25
    assert all(r["influence"] == 0 for i, r in rows.items() if i != side)
    sp = {s["id"]: s for s in client.get("/api/speakers").json()["speakers"]}[jia]
    assert sp["risky_samples"] == 1 and sp["outlier_samples"] == 1
    # voice group 「不是 甲」
    seg2, _p = _seg(ncfg, db, datetime(2026, 11, 4, 10), _dc(4.0, 0.4))
    u = _utt(db, seg2, "10:00:00", "谁", "未知", None, 0, 4, cluster=0)
    db.upsert_cluster_embedding(seg2, 0, X, 4.0)
    rt._engine = NumpyEngine(ncfg, {0.4: X})
    assert client.post("/api/voices/reject", json={"members": [], "speaker_id": jia}).status_code == 400
    assert client.post("/api/voices/reject", json={"members": [[seg2, 0]]}).status_code == 400
    assert client.post("/api/voices/reject", json={"members": [[seg2, 0]], "speaker_id": 999}).status_code == 404
    r = client.post("/api/voices/reject", json={"members": [[seg2, 0]], "speaker_id": jia})
    assert r.status_code == 200 and r.json()["rejected"] == 1
    assert db.rejected_speakers([u]) == {u: {jia}}
    assert client.post("/api/voices/reject", json={"members": [["x", 0]], "speaker_id": jia}).status_code == 400


# =========================================================================================================
# review fixes (vetoes survive split / hand labelling / faint relabel; reject results; engine fallback; faint queue)
# =========================================================================================================

def _split_setup(ncfg, ndb, jia):
    """6 s sentence vetoed for 甲 (and 丙): 0-3 s voice X (passes as 甲 at 0.55), 3-6 s voice Y = 乙."""
    yi = ndb.upsert_speaker("乙")
    ndb.add_speaker_embedding(yi, Y, None, 10.0, origin="mic")
    seg, _p = _seg(ncfg, ndb, datetime(2026, 11, 4, 9), np.concatenate([_dc(3.0, 0.4), _dc(3.0, 0.6)]))
    u = _utt(ndb, seg, "09:00:00", "混在一起", "未知", None, 0.0, 6.0, cluster=0)
    ndb.reject_utterances(jia, [u], "未知")
    other = ndb.upsert_speaker("丙")
    ndb.execute("INSERT INTO utterance_rejections(utterance_id, speaker_id) VALUES (?,?)", (u, other))
    return yi, other, seg, u


@pytest.mark.parametrize("auto", [False, True])
def test_split_keeps_the_veto_for_every_piece(ncfg, ndb, jia, auto):
    yi, other, seg, u = _split_setup(ncfg, ndb, jia)
    eng = NumpyEngine(ncfg, {0.4: X, 0.6: Y}, turns=[Turn(0.0, 3.0, 0), Turn(3.0, 6.0, 1)])
    proc = Processor(ncfg, ndb, engine=eng, asr=_ASR())
    kw = {"auto": True} if auto else {"at": [3.0]}
    prev = proc.preview_split(u, **kw)
    assert [p["speaker_id"] for p in prev["pieces"]] == [None, yi]          # X would pass as 甲, but 甲 is vetoed
    res = proc.split_utterance(u, **kw)
    ids = [p["id"] for p in res["pieces"]]
    assert [p["speaker_id"] for p in res["pieces"]] == [None, yi] and res["pieces"][0]["speaker"] == "未知"
    assert ndb.rejected_speakers(ids) == {i: {jia, other} for i in ids}      # the vetoes moved to the pieces
    assert ndb.rejected_speakers([u]) == {}
    Processor(ncfg, ndb, engine=eng).relabel_segment(seg)
    assert [_row(ndb, i)["speaker_id"] for i in ids] == [None, yi]           # and relabel keeps honouring them


def test_split_enforces_the_veto_even_when_the_engine_ignores_it(ncfg, ndb, jia):
    _yi, _other, _seg_id, u = _split_setup(ncfg, ndb, jia)

    class Stubborn(NumpyEngine):
        """Ignores every exclusion: says 甲 for everything."""

        def match(self, emb, enrolled, thr, **kw):
            return jia, "甲", 0.9

        def identify_turns(self, audio, turns, db, segment_id=None):
            for t in turns:
                t.speaker_id, t.speaker_name, t.score = jia, "甲", 0.9
            return turns

    proc = Processor(ncfg, ndb, engine=Stubborn(ncfg, {}, turns=[Turn(0.0, 3.0, 0), Turn(3.0, 6.0, 1)]), asr=_ASR())
    assert [p["speaker_id"] for p in proc.preview_split(u, at=[3.0])["pieces"]] == [None, None]
    pieces = proc.preview_split(u, auto=True)["pieces"]
    assert len(pieces) == 2 and all(p["speaker_id"] is None and p["speaker"] == "未知" for p in pieces)


def test_name_voice_skips_sentences_vetoed_for_the_person(ncfg, ndb, jia):
    seg, _p = _seg(ncfg, ndb, datetime(2026, 11, 4, 9), _dc(6.0, 0.4), write=False)    # expired: no sample needed
    u1 = _utt(ndb, seg, "09:00:00", "一", "未知", None, 0, 2, cluster=0)
    u2 = _utt(ndb, seg, "09:00:02", "二", "未知", None, 2, 4, cluster=0)
    u3 = _utt(ndb, seg, "09:00:04", "不是甲", "未知", None, 4, 6, cluster=0)
    ndb.reject_utterances(jia, [u3], "未知")
    res = voices.name_voice(ndb, ncfg, None, [[seg, 0]], "甲")
    assert res["utterances_assigned"] == 2 and res["skipped_vetoed"] == 1 and res["skipped_user_labels"] == 0
    assert [_row(ndb, u)["speaker_id"] for u in (u1, u2, u3)] == [jia, jia, None]
    assert ndb.rejected_speakers([u3]) == {u3: {jia}}


def test_relabel_releases_a_faint_sentence_owned_by_a_vetoed_speaker(ncfg, ndb, jia):
    seg, _p = _seg(ncfg, ndb, datetime(2026, 11, 4, 9), _dc(6.0, 0.2))
    faint = _utt(ndb, seg, "09:00:00", "远处", "甲", jia, 0, 2, 0.6, loudness=-55.0)
    faint_ok = _utt(ndb, seg, "09:00:02", "远处二", "甲", jia, 2, 4, 0.6, loudness=-55.0)
    hand = _utt(ndb, seg, "09:00:04", "手标", "甲", jia, 4, 6, None, source="user", loudness=-55.0)
    # vetoes recorded without releasing the rows (an older version / a concurrent write)
    for u in (faint, hand):
        ndb.execute("INSERT INTO utterance_rejections(utterance_id, speaker_id) VALUES (?,?)", (u, jia))
    assert Processor(ncfg, ndb, engine=NumpyEngine(ncfg, {0.2: A})).relabel_segment(seg) == 1
    r = _row(ndb, faint)
    assert (r["speaker_id"], r["speaker_name"], r["speaker_score"], r["speaker_source"]) == (None, "未知", None, None)
    assert _row(ndb, faint_ok)["speaker_id"] == jia and _row(ndb, faint_ok)["speaker_score"] is None   # left as it was
    assert _row(ndb, hand)["speaker_id"] == jia and _row(ndb, hand)["speaker_source"] == "user"


def test_relabel_releases_a_faint_system_audio_sentence_from_the_self_speaker(ncfg, ndb, jia):
    ndb.set_self_speaker(jia)
    seg = _meeting_segment(ncfg, ndb)
    ndb.set_segment_status(seg, "done")
    mine = _utt(ndb, seg, "09:00:00", "我", "甲", jia, 0, 3, 0.9, loudness=-55.0)
    theirs = _utt(ndb, seg, "09:00:03", "对方", "甲", jia, 3, 6, 0.9, loudness=-55.0)
    assert Processor(ncfg, ndb, engine=NumpyEngine(ncfg, {0.2: A})).relabel_segment(seg) == 1
    assert _row(ndb, mine)["speaker_id"] == jia and _row(ndb, theirs)["speaker_id"] is None


def test_reject_results_list_every_vetoed_sentence(ncfg, ndb, jia):
    seg, _p = _seg(ncfg, ndb, datetime(2026, 11, 4, 9), _dc(8, 0.4), write=False)
    u1 = _utt(ndb, seg, "09:00:00", "一", "甲", jia, 0, 2, 0.5, cluster=0)
    u2 = _utt(ndb, seg, "09:00:02", "二", "甲", jia, 2, 4, 0.5, cluster=0)
    u3 = _utt(ndb, seg, "09:00:04", "三", "甲", jia, 4, 6, 0.5, cluster=1)
    assert voices.reject_voice(ndb, ncfg, None, jia, [u1], whole_cluster=True)["utterance_ids"] == [u1, u2]
    assert voices.reject_voice(ndb, ncfg, None, jia, [u3])["utterance_ids"] == [u3]
    n1 = _utt(ndb, seg, "09:00:06", "四", "未知", None, 6, 7, cluster=2)
    n2 = _utt(ndb, seg, "09:00:07", "五", "未知", None, 7, 8, cluster=2)
    assert voices.reject_voice_group(ndb, ncfg, None, [[seg, 2]], jia)["utterance_ids"] == [n1, n2]


def test_review_queue_leaves_out_faint_sentences(ncfg, ndb, jia):
    seg, _p = _seg(ncfg, ndb, datetime(2026, 11, 4, 9), _dc(10, 0.2))
    faint = _utt(ndb, seg, "09:00:00", "远处", "甲", jia, 0, 2, 0.1, loudness=-50.0)
    unknown_level = _utt(ndb, seg, "09:00:02", "没有音量", "甲", jia, 2, 4, 0.3, loudness=None)
    loud = _utt(ndb, seg, "09:00:04", "清楚", "甲", jia, 4, 6, 0.2, loudness=-30.0)
    edge = _utt(ndb, seg, "09:00:06", "刚好", "甲", jia, 6, 8, None, loudness=-40.0)
    assert [i["utterance_id"] for i in review.review_queue(ndb, ncfg, jia)] == [edge, loud, unknown_level]
    assert review.review_total(ndb, ncfg, jia) == 3
    ncfg["audio"]["min_speech_dbfs"] = None                                   # gate off: everything is reviewed
    assert [i["utterance_id"] for i in review.review_queue(ndb, ncfg, jia)] == [edge, faint, loud, unknown_level]
    assert review.review_total(ndb, ncfg, jia) == 4


def test_api_reject_falls_back_to_veto_only_when_the_engine_fails_to_load(api, ncfg, monkeypatch):
    client, rt = api
    db = rt.db
    jia = db.upsert_speaker("甲")
    db.add_speaker_embedding(jia, A, None, 10.0, origin="mic")
    seg, _p = _seg(ncfg, db, datetime(2026, 11, 4, 9), _dc(6.0, 0.4))
    u1 = _utt(db, seg, "09:00:00", "一", "甲", jia, 0, 3, 0.55, cluster=0)
    u2 = _utt(db, seg, "09:00:03", "二", "甲", jia, 3, 6, 0.55, cluster=0)
    monkeypatch.setattr(rt, "speaker_models_missing", lambda: [])

    def broken():
        raise RuntimeError("模型文件损坏")

    monkeypatch.setattr(rt, "engine", broken)
    r = client.post(f"/api/utterances/{u1}/reject", json={"whole_cluster": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rejected"] == 2 and body["negatives_added"] == 0 and body["utterance_ids"] == [u1, u2]
    assert db.rejected_speakers([u1, u2]) == {u1: {jia}, u2: {jia}}
    r = client.post(f"/api/utterances/{u1}/confirm", json={"speaker_id": jia})
    assert r.status_code == 200 and r.json()["sample_added"] is False and "声纹模型不可用" in r.json()["sample_note"]
    assert db.rejected_speakers([u1]) == {}                                  # 「是 TA」 lifts the veto
    # /api/voices/reject reports the vetoed ids too
    seg2, _p = _seg(ncfg, db, datetime(2026, 11, 4, 10), _dc(4.0, 0.4))
    n = _utt(db, seg2, "10:00:00", "谁", "未知", None, 0, 4, cluster=0)
    r = client.post("/api/voices/reject", json={"members": [[seg2, 0]], "speaker_id": jia})
    assert r.status_code == 200 and r.json()["utterance_ids"] == [n]
