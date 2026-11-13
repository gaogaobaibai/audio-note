"""Speakers v5, the scoring layer (audio_note/speech/scoring.py): Gallery.load / score, condition_of, adapt.

Pure numpy on throw-away databases: no model, no audio. Voices are random unit vectors; `_at(base, seed, cos)`
gives a vector with an exact cosine to `base`, so every threshold can be hit on purpose.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from audio_note.db import DB
from audio_note.speech import scoring
from audio_note.speech.scoring import Decision, Gallery, Params, adapt, condition_of
from audio_note.speech.speakers import SpeakerEngine

DIM = 64
DAY = "2026-11-04"


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _voice(seed: int) -> np.ndarray:
    return _unit(np.random.default_rng(seed).standard_normal(DIM))


def _at(base: np.ndarray, seed: int, cos: float) -> np.ndarray:
    o = np.random.default_rng(seed).standard_normal(base.shape[0])
    o = o - float(np.dot(o, base)) * base
    o = o / np.linalg.norm(o)
    return _unit(cos * base + math.sqrt(max(0.0, 1.0 - cos * cos)) * o)


A = _voice(1)      # 甲
B = _voice(2)      # 乙 (independent of A: cosine about 0)


@pytest.fixture
def scfg(cfg_factory):
    return cfg_factory({"memory": {"retrieval": "fts", "embedding_model": "hash"}})


@pytest.fixture
def sdb(scfg) -> DB:
    return DB(scfg.db_path)


def _speaker(db: DB, name: str, vecs: list[np.ndarray], origin: str = "mic", is_self: bool = False) -> int:
    sid = db.upsert_speaker(name)
    for v in vecs:
        db.add_speaker_embedding(sid, v, None, 10.0, origin=origin)
    if is_self:
        db.set_self_speaker(sid)
    return sid


def _cluster(db: DB, k: int, emb: np.ndarray, speech_s: float = 5.0, loudness: float | None = -20.0,
             system_share: float | None = None, speaker_id: int | None = None, source: str | None = None,
             when: datetime | None = None) -> tuple[int, int]:
    """One 'done' segment with one cluster vector and one utterance covering it. Returns (segment id, utterance id)."""
    when = when or (datetime(2026, 11, 4, 9, 0, 0) + timedelta(seconds=20 * k))
    seg = db.add_segment(f"seg_{k}.wav", when, speech_s)
    db.set_segment_status(seg, "done")
    db.upsert_cluster_embedding(seg, 0, emb, speech_s)
    db.execute("UPDATE cluster_embeddings SET loudness_db=?, system_share=? WHERE segment_id=?", (loudness, system_share, seg))
    uid = db.add_utterances([{
        "segment_id": seg, "start_s": 0.0, "end_s": speech_s, "abs_start": when.isoformat(timespec="milliseconds"),
        "day": when.strftime("%Y-%m-%d"), "cluster_label": 0, "speaker_id": speaker_id,
        "speaker_name": None, "speaker_score": None, "text_raw": f"句子{k}", "text": f"句子{k}", "text_seg": f"句子 {k}",
        "speaker_source": source, "loudness_db": loudness, "system_share": system_share}])[0]
    return seg, uid


def _impostors(db: DB, n: int = 40, start: int = 100) -> None:
    """Stored voices of nobody enrolled: what the cohort is made of."""
    for k in range(n):
        _cluster(db, start + k, _voice(1000 + k))


# ================================================================================================ plain pieces
def test_condition_of_near_far_system(scfg):
    assert condition_of(-20.0, None, scfg) == "near" and condition_of(None, None, scfg) == "near"
    assert condition_of(-35.0, 0.1, scfg) == "far" and condition_of(-30.0, None, scfg) == "near"      # the bound itself is near
    assert condition_of(-35.0, 0.6, scfg) == "system" and condition_of(-10.0, 0.9, scfg) == "system"   # system audio wins
    assert condition_of(-35.0, None, Params(far_dbfs=-40.0)) == "near"                                  # Params work as cfg


def test_params_come_from_config_with_safe_fallbacks(scfg):
    p = Params.from_cfg(scfg)
    assert (p.scoring, p.sure_norm, p.guess_norm, p.cohort_min, p.track_link if hasattr(p, "track_link") else None) == \
        ("v5", 0.25, 0.15, 20, None)
    assert p.min_speech_dbfs == -40.0 and p.identify_threshold == 0.45
    scfg["speakers"]["sure_norm"] = "not a number"
    scfg["speakers"]["scoring"] = " LEGACY "
    scfg["speakers"]["adapt"] = "false"
    q = Params.from_cfg(scfg)
    assert q.sure_norm == 0.25 and q.scoring == "legacy" and q.adapt is False
    assert Params.from_cfg(object()).scoring == "v5"                 # a config stand-in without sections
    assert p.key() != q.key()                                        # a changed setting changes the gallery version


def test_decision_states_and_meta():
    sure = Decision(3, "甲", 0.71234, 0.3111, None, None, "centroid:far", 0.02, "far")
    maybe = Decision(None, None, 0.4, 0.2, 3, 0.2, "anchor", None, "near", guess_name="甲")
    assert (sure.state, maybe.state, scoring.NO_DECISION.state) == ("sure", "guess", "unknown")
    assert sure.meta() == {"m": "centroid:far", "raw": 0.712, "norm": 0.31, "next": 0.02, "cond": "far"}
    assert scoring.NO_DECISION.meta() == {"m": "none"}
    assert scoring.model_id({"speakers": {"embedding_model": "sub/dir/voice_model.onnx"}}) == "voice_model"


# ================================================================================================ without a cohort
def test_no_cohort_means_raw_cosine_and_the_old_threshold(sdb, scfg):
    jia = _speaker(sdb, "甲", [A, _at(A, 3, 0.95)])
    g = Gallery.load(sdb, scfg)
    assert g.has_speakers and not g.normalised and g.dim == DIM and g.names == {jia: "甲"}
    anchor = g.speakers[0].anchor
    d = g.score(_at(anchor, 10, 0.60), speech_s=5.0)
    assert (d.speaker_id, d.name, d.method, d.norm, d.guess_id) == (jia, "甲", "anchor", None, None)
    assert d.score == pytest.approx(0.60, abs=1e-3)
    # just under the threshold: not assigned, but worth asking ("maybe 甲")
    d = g.score(_at(anchor, 11, 0.40), speech_s=5.0)
    assert d.speaker_id is None and (d.guess_id, d.guess_name) == (jia, "甲") and d.guess_score == pytest.approx(0.40, abs=1e-3)
    assert d.score == pytest.approx(0.40, abs=1e-3)                  # the raw cosine is still shown
    d = g.score(_at(anchor, 12, 0.30), speech_s=5.0)                # below threshold - guess_raw_margin: unknown
    assert d.state == "unknown" and d.method == "none" and d.score == pytest.approx(0.30, abs=1e-3)
    assert g.score(np.ones(8, dtype=np.float32)).state == "unknown"  # another dimension: nobody to compare with


def test_short_vectors_are_soft_evidence(sdb, scfg):
    jia = _speaker(sdb, "甲", [A])
    g = Gallery.load(sdb, scfg)
    v = _at(A, 20, 0.47)
    assert g.score(v, speech_s=3.0).speaker_id == jia               # long enough: decides on its own
    d = g.score(v, speech_s=1.5)                                     # 1-2 s: needs threshold + short_extra_raw (0.50)
    assert d.speaker_id is None and d.guess_id == jia
    assert g.score(_at(A, 21, 0.52), speech_s=1.5).speaker_id == jia
    d = g.score(_at(A, 22, 0.95), speech_s=0.8)                     # under follow_s: never decides alone, however alike
    assert d.speaker_id is None and d.guess_id == jia
    assert g.score(v).speaker_id == jia                              # duration unknown: treated as long


def test_outlier_samples_no_longer_recognise_on_their_own(sdb, scfg):
    """Rule 1: a sample outside the core does not take part; the legacy mode still lets it through."""
    stray = _at(A, 30, 0.20)                                         # cut from a recording, sounds unlike the others
    jia = _speaker(sdb, "甲", [A, _at(A, 31, 0.96)])
    sdb.add_speaker_embedding(jia, stray, "x.wav#1-4", 3.0, origin="label")
    probe = _at(stray, 32, 0.90)                                     # a voice that only resembles the stray sample
    assert float(probe @ A) < 0.35
    g = Gallery.load(sdb, scfg)
    assert g.speakers[0].outliers == 1 and g.describe()["outlier_samples"] == 1
    assert g.score(probe, speech_s=5.0).speaker_id is None
    scfg["speakers"]["scoring"] = "legacy"
    legacy = Gallery.load(sdb, scfg)
    assert legacy.score(probe, speech_s=5.0).speaker_id == jia and legacy.score(probe, speech_s=0.6).speaker_id == jia
    assert legacy.score(_at(A, 33, 0.40)).state == "unknown"         # the old rule has no "maybe"


def test_legacy_mode_is_speaker_engine_match(sdb, scfg):
    rng = np.random.default_rng(7)
    jia = _speaker(sdb, "甲", [A, _at(A, 40, 0.7)])
    yi = _speaker(sdb, "乙", [B])
    sdb.add_speaker_negative(jia, _at(A, 41, 0.6), None, 2.0)
    scfg["speakers"]["scoring"] = "legacy"
    g = Gallery.load(sdb, scfg)
    enrolled, negs = sdb.speaker_embeddings(), sdb.speaker_negatives()
    for k in range(200):
        base = (A, B, _at(A, 41, 0.6))[k % 3]
        probe = _at(base, 500 + k, float(rng.uniform(0.2, 0.95)))
        exclude = frozenset({jia}) if k % 7 == 0 else frozenset()
        sid, name, score = SpeakerEngine.match(probe, enrolled, 0.45, negs, margin=0.0, exclude=exclude)
        d = g.score(probe, speech_s=float(rng.uniform(0.3, 9.0)), exclude=exclude)
        assert (d.speaker_id, d.name) == (sid, name), k
        if score is not None and d.score is not None:
            assert d.score == pytest.approx(score, abs=1e-5), k
    assert {jia, yi} == set(g.sids)


def test_vetoes_negatives_and_system_audio_keep_priority(sdb, scfg):
    jia = _speaker(sdb, "甲", [A], is_self=True)
    yi = _speaker(sdb, "乙", [B])
    g = Gallery.load(sdb, scfg)
    v = _at(A, 50, 0.8)
    assert g.score(v, speech_s=5.0).speaker_id == jia
    d = g.score(v, speech_s=5.0, exclude=frozenset({jia}))           # vetoed for this sentence
    assert d.speaker_id is None and d.guess_id is None and d.method == "veto"
    d = g.score(v, speech_s=5.0, system_share=0.9)                   # system audio is never the user themselves
    assert d.speaker_id is None and d.guess_id is None and d.condition == "system"
    assert g.score(_at(B, 51, 0.8), speech_s=5.0, system_share=0.9).speaker_id == yi     # but it may be somebody else
    assert g.score(v, speech_s=5.0, exclude=frozenset({jia, yi})).score is None           # nobody left to compare with
    # a negative sample ("not 甲") closer than 甲's own voice vetoes 甲 for the voices that resemble it
    look_alike = _at(A, 52, 0.6)
    sdb.add_speaker_negative(jia, look_alike, None, 2.0)
    g = Gallery.load(sdb, scfg)
    d = g.score(_at(look_alike, 53, 0.95), speech_s=5.0)
    assert d.speaker_id is None and d.guess_id is None and d.method == "veto" and d.score is not None
    assert g.score(_at(A, 54, 0.9), speech_s=5.0).speaker_id == jia  # the real voice is untouched


# ================================================================================================ with a cohort
def test_cohort_normalisation_decides_in_cosine_above_the_impostors(sdb, scfg):
    jia = _speaker(sdb, "甲", [A])
    _impostors(sdb, 40)
    g = Gallery.load(sdb, scfg)
    assert g.normalised and g.describe()["cohort"] == 40
    ev = g.evaluate(_at(A, 60, 0.60), speech_s=5.0)
    d = ev.decision
    assert d.speaker_id == jia and d.norm is not None and 0.25 <= d.norm < d.score      # raw minus what impostors reach
    assert ev.holds(jia) and ev.plausible(jia) and ev.leans(jia) and not ev.rejects(jia) and ev.value(jia) == pytest.approx(d.norm)
    # the unit is "cosine above the closest impostors": every probe is judged by its own normalised score
    states = []
    for k, raw in enumerate((0.30, 0.34, 0.38, 0.42, 0.46, 0.50)):
        w = g.evaluate(_at(A, 610 + k, raw), speech_s=5.0)
        offset = raw - w.value(jia)
        assert 0.05 < offset < 0.30                                  # what impostors reach against 甲 and this probe
        want = "sure" if w.value(jia) >= 0.25 else "guess" if w.value(jia) >= 0.15 else "unknown"
        assert w.decision.state == want, (raw, w.value(jia))
        if want == "guess":
            assert w.decision.guess_id == jia and w.decision.guess_score == pytest.approx(w.value(jia))
        states.append(want)
    assert "guess" in states and states[-1] == "sure" and states == sorted(states, key=("unknown", "guess", "sure").index)
    far_off = g.evaluate(_voice(999), speech_s=5.0)
    assert far_off.decision.state == "unknown" and far_off.rejects(jia) and not far_off.leans(jia)
    assert far_off.rejects(12345) and far_off.value(12345) is None   # a speaker the gallery does not know
    # short vectors: stricter (sure_norm + short_extra_norm), and under follow_s never alone
    v = _at(A, 62, 0.52)
    long_, short, tiny = (g.score(v, speech_s=s) for s in (4.0, 1.5, 0.7))
    assert long_.norm == pytest.approx(short.norm) and long_.speaker_id == jia
    assert tiny.speaker_id is None and tiny.guess_id == jia
    assert short.speaker_id == (jia if short.norm >= 0.30 else None)


def test_fewer_impostors_than_cohort_min_falls_back(sdb, scfg):
    _speaker(sdb, "甲", [A])
    _impostors(sdb, 19)
    assert not Gallery.load(sdb, scfg).normalised
    _cluster(sdb, 400, _voice(4000))
    assert Gallery.load(sdb, scfg).normalised                        # the 20th impostor switches normalisation on
    scfg["speakers"]["cohort_min"] = 21
    assert not Gallery.load(sdb, scfg).normalised
    # vectors under cohort_min_s seconds of speech and voices close to the speaker are no impostors
    scfg["speakers"]["cohort_min"] = 20
    db2 = DB(Path(scfg.db_path).with_name("second.sqlite3"))
    _speaker(db2, "甲", [A])
    for k in range(19):
        _cluster(db2, k, _voice(1000 + k))
    _cluster(db2, 50, _voice(5000), speech_s=2.0)                    # too short to be trusted as "somebody else"
    _cluster(db2, 51, _at(A, 70, 0.5))                               # may be 甲 further away: must not leak in
    assert not Gallery.load(db2, scfg).normalised
    db2.close_all()


def test_the_runner_up_must_be_clearly_behind(sdb, scfg):
    jia = _speaker(sdb, "甲", [A])
    twin_base = _at(A, 80, 0.2)
    yi = _speaker(sdb, "乙", [twin_base])
    _impostors(sdb, 40)
    g = Gallery.load(sdb, scfg)
    assert g.normalised
    between = _unit(A + twin_base)                                   # equally like both
    ev = g.evaluate(between, speech_s=5.0)
    assert abs(ev.value(jia) - ev.value(yi)) < 0.05 and ev.value(jia) >= 0.25
    assert ev.decision.speaker_id is None and ev.decision.guess_id in (jia, yi) and ev.decision.runner_up is not None
    assert g.score(_at(A, 81, 0.7), speech_s=5.0).speaker_id == jia


def test_gallery_is_cached_until_something_it_is_built_from_changes(sdb, scfg):
    jia = _speaker(sdb, "甲", [A])
    g1 = Gallery.load(sdb, scfg)
    assert Gallery.load(sdb, scfg) is g1
    sdb.add_speaker_embedding(jia, _at(A, 90, 0.9), None, 5.0, origin="file")
    g2 = Gallery.load(sdb, scfg)
    assert g2 is not g1 and g2.version != g1.version                 # a new sample: decisions may differ
    sdb.add_speaker_negative(jia, _voice(91), None, 2.0)
    g3 = Gallery.load(sdb, scfg)
    assert g3.version != g2.version
    scfg["speakers"]["sure_norm"] = 0.3
    g4 = Gallery.load(sdb, scfg)
    assert g4.version != g3.version
    _cluster(sdb, 1, _voice(92))                                     # a stored cluster changes the cohort, not the version
    g5 = Gallery.load(sdb, scfg)
    assert g5 is not g4 and g5.version == g4.version
    Gallery.forget(sdb)
    assert Gallery.load(sdb, scfg) is not g5


def test_gallery_tolerates_a_database_stand_in():
    class Fake:                                                      # what older test fakes offer
        def speaker_embeddings(self):
            return [(1, "甲", A), (1, "甲", _at(A, 95, 0.9)), (2, "乙", B)]

    g = Gallery.load(Fake(), {"speakers": {"identify_threshold": 0.5}})
    assert g.sids == [1, 2] and not g.normalised and g.self_id is None
    assert g.score(_at(B, 96, 0.8), speech_s=4.0).name == "乙"
    assert Gallery.load(object(), {}).has_speakers is False and Gallery.load(object(), {}).score(A).state == "unknown"


# ================================================================================================ adapt
def _far(seed: int, drift: np.ndarray, cos_to_drift: float = 0.9) -> np.ndarray:
    return _at(drift, seed, cos_to_drift)


def test_adapt_learns_condition_centroids_from_confident_and_confirmed_speech(sdb, scfg):
    jia = _speaker(sdb, "甲", [A], is_self=True)
    _impostors(sdb, 40)
    wild = _at(A, 200, 0.75)                                         # how 甲 sounds in daily recordings
    far = _at(A, 201, 0.40)                                          # ... and far from the microphone
    for k in range(6):
        _cluster(sdb, 10 + k, _at(wild, 210 + k, 0.92), loudness=-20.0)
    confirmed = [_cluster(sdb, 20 + k, _at(far, 220 + k, 0.92), loudness=-36.0, speaker_id=jia, source="user") for k in range(3)]
    probe_far = _at(far, 230, 0.9)
    before = Gallery.load(sdb, scfg).score(probe_far, speech_s=5.0, loudness_db=-36.0)
    assert before.speaker_id is None                                  # the enrolled voice alone does not reach it
    rep = adapt(sdb, scfg, dry_run=True)
    assert rep["dry_run"] and rep["changed"] and sdb.query("SELECT 1 FROM speaker_centroids") == []
    rep = adapt(sdb, scfg)
    info = rep["speakers"][str(jia)]
    assert info["near"]["kept"] and info["near"]["n"] == 6 and info["far"]["kept"] and info["far"]["user"] == 3
    assert info["system"] == {"n": 0, "speech_s": 0.0, "user": 0, "kept": False, "why": "too_few"}
    assert info["all"]["n"] == 9 and info["far"]["anchor_cos"] >= 0.5
    rows = {r["condition"]: r for r in sdb.query("SELECT * FROM speaker_centroids WHERE speaker_id=?", (jia,))}
    assert set(rows) == {"all", "near", "far"} and rows["far"]["n"] == 3 and rows["far"]["dim"] == DIM
    g = Gallery.load(sdb, scfg)
    after = g.score(probe_far, speech_s=5.0, loudness_db=-36.0)
    assert after.speaker_id == jia and after.method == "centroid:far" and after.condition == "far"
    assert g.score(_at(wild, 231, 0.9), speech_s=5.0, loudness_db=-20.0).method == "centroid:near"
    assert g.describe()["centroids"] == {str(jia): ["all", "far", "near"]}
    # deterministic: nothing changes when nothing changed, and the hand-labelled rows are still the user's
    again = adapt(sdb, scfg)
    assert again["changed"] is False and Gallery.load(sdb, scfg).version == g.version
    assert all(sdb.one("SELECT speaker_source FROM utterances WHERE id=?", (uid,))["speaker_source"] == "user" for _s, uid in confirmed)


def test_adapt_refuses_drift_too_little_evidence_and_what_the_user_rejected(sdb, scfg):
    jia = _speaker(sdb, "甲", [A])
    _impostors(sdb, 40)
    stranger = _at(A, 300, 0.25)                                     # hand-labelled as 甲 by mistake: far from the anchor
    for k in range(8):
        _cluster(sdb, 10 + k, _at(stranger, 310 + k, 0.95), speech_s=20.0, loudness=-36.0, speaker_id=jia, source="user")
    rep = adapt(sdb, scfg)
    far = rep["speakers"][str(jia)]["far"]
    assert far["kept"] is False and far["why"] == "drift" and far["anchor_cos"] < 0.5 and rep["written"] == 0
    # two clusters are not enough for a condition (adapt_min_clusters 3)
    sdb.execute("DELETE FROM utterances WHERE speaker_source='user'")
    sdb.execute("DELETE FROM cluster_embeddings WHERE segment_id IN (SELECT id FROM segments WHERE path LIKE 'seg_1_.wav')")
    good = _at(A, 320, 0.8)
    made = [_cluster(sdb, 30 + k, _at(good, 330 + k, 0.95)) for k in range(2)]
    rep = adapt(sdb, scfg)
    assert rep["speakers"][str(jia)]["near"]["why"] == "too_few"
    # a vetoed cluster and a cluster the user gave to somebody else are never evidence
    yi = _speaker(sdb, "乙", [B])
    seg_v, uid_v = _cluster(sdb, 40, _at(good, 340, 0.95))
    sdb.execute("INSERT INTO utterance_rejections(utterance_id, speaker_id) VALUES (?,?)", (uid_v, jia))
    _cluster(sdb, 41, _at(good, 341, 0.95), speaker_id=yi, source="user")
    rep = adapt(sdb, scfg)
    assert rep["speakers"][str(jia)]["near"]["n"] == 2            # neither the vetoed one nor 乙's counts for 甲
    assert rep["speakers"][str(yi)]["all"] == {"n": 1, "speech_s": 5.0, "user": 1, "kept": False, "why": "too_few"}
    _cluster(sdb, 42, _at(good, 342, 0.95))
    rep = adapt(sdb, scfg)
    assert rep["speakers"][str(jia)]["near"]["kept"] and rep["speakers"][str(jia)]["near"]["n"] == 3
    assert len(made) == 2
    # switching adaptation off removes the centroids again
    scfg["speakers"]["adapt"] = False
    off = adapt(sdb, scfg)
    assert off["enabled"] is False and off["changed"] and sdb.query("SELECT 1 FROM speaker_centroids") == []
    assert Gallery.load(sdb, scfg).describe()["centroids"] == {}


def test_adapt_never_learns_the_self_voice_from_system_audio(sdb, scfg):
    jia = _speaker(sdb, "甲", [A], is_self=True)
    _impostors(sdb, 40)
    for k in range(5):
        _cluster(sdb, 10 + k, _at(A, 400 + k, 0.9), system_share=0.95)          # sounds like 甲, came from the speakers
    rep = adapt(sdb, scfg)
    assert rep["written"] == 0 and rep["speakers"][str(jia)]["all"]["n"] == 0
    assert Gallery.load(sdb, scfg).score(_at(A, 410, 0.9), speech_s=5.0, system_share=0.95).state == "unknown"


def test_outlier_label_samples_seed_the_far_centroid(sdb, scfg):
    """Rule 1, second half: a sample outside the core is only a seed for the far condition."""
    jia = _speaker(sdb, "甲", [A, _at(A, 500, 0.95)])
    far = _at(A, 501, 0.45)
    for k in range(3):
        sdb.add_speaker_embedding(jia, _at(far, 510 + k, 0.95), f"seg.wav#{k}-{k + 4}", 4.0, origin="label")
    g0 = Gallery.load(sdb, scfg)
    assert g0.speakers[0].outliers == 3
    rep = adapt(sdb, scfg)
    info = rep["speakers"][str(jia)]
    assert info["far"]["kept"] and info["far"]["n"] == 3 and info["near"]["n"] == 0
    g = Gallery.load(sdb, scfg)
    probe = _at(far, 520, 0.92)
    assert g0.score(probe, speech_s=5.0, loudness_db=-36.0).speaker_id is None
    assert g.score(probe, speech_s=5.0, loudness_db=-36.0).speaker_id == jia
