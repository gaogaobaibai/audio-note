"""Speakers v5 in the pipeline, the review queue, the API and the command line.

No network, no microphone, no model files: `NumpyEngine` is the real SpeakerEngine with the embedding extractor
replaced by a table (a clip's mean DC level picks its voice vector), so identification, relabel, split and the
after-store hook run through the production code. Constant-level audio also sets the "loudness" of a clip: level 0.2
is near the microphone (-14 dBFS), level 0.02 is far (-34 dBFS, below speakers.far_dbfs -30).
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from audio_note import cli
from audio_note.audio.storage import segment_path
from audio_note.audio.vad import SAMPLE_RATE as SR
from audio_note.db import DB
from audio_note.runtime import Runtime
from audio_note.speech import review, scoring
from audio_note.speech.asr import ASRResult
from audio_note.speech.pipeline import Processor
from audio_note.speech.speakers import SpeakerEngine, Turn
from audio_note.ui.app import create_app

DAY = "2026-11-04"
DIM = 192


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


def _same_voice(base: np.ndarray, direction_seed: int, seed: int, cos_base: float, along: float = 0.85) -> np.ndarray:
    """Vectors that share one direction besides `base` (so they sound alike: cosine about cos_base^2 + along^2
    to each other) and have exactly `cos_base` to `base`: one voice at a fixed distance from the enrolled one."""
    u = np.random.default_rng(direction_seed).standard_normal(base.shape[0])
    u = u - float(u @ base) * base
    u = u / np.linalg.norm(u)
    w = np.random.default_rng(seed).standard_normal(base.shape[0])
    w = w - float(w @ base) * base - float(w @ u) * u
    w = w / np.linalg.norm(w)
    return _unit(cos_base * base + along * u + math.sqrt(max(0.0, 1.0 - cos_base ** 2 - along ** 2)) * w)


A = _voice(1)                  # 甲
NEAR = _at(A, 10, 0.80)        # 甲 near the microphone
MAYBE = _at(A, 11, 0.40)       # a voice that is only "maybe 甲"
OTHER = _voice(12)             # nobody


class NumpyEngine(SpeakerEngine):
    """SpeakerEngine without models: `embed` maps a clip's mean level (rounded to 0.05) to a voice from `table`."""

    def __init__(self, cfg, table: dict[float, np.ndarray], turns: list[Turn] | None = None):
        self.cfg = cfg
        self.table = dict(table)
        self.turns = turns or []
        self.dim = DIM
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


class _ASR:
    def __init__(self):
        self.n = 0

    def transcribe(self, audio):
        self.n += 1
        return ASRResult(f"句子{self.n}", round(len(audio) / SR, 1), 1)


def _dc(seconds: float, level: float) -> np.ndarray:
    return np.full(int(round(seconds * SR)), level, dtype=np.float32)


def _seg(cfg, db: DB, when: datetime, audio: np.ndarray, status: str = "new") -> int:
    p = segment_path(cfg, when, ".wav")
    sf.write(p, audio, SR, subtype="PCM_16")
    sid = db.add_segment(p, when, len(audio) / SR, speech_s=len(audio) / SR)
    if status != "new":
        db.set_segment_status(sid, status)
    return sid


def _rows(db: DB, seg: int):
    return db.query("SELECT * FROM utterances WHERE segment_id=? ORDER BY start_s, id", (seg,))


def _voice_rows(db: DB) -> dict[int, np.ndarray]:
    return {int(r["utterance_id"]): np.frombuffer(r["embedding"], dtype=np.float32)
            for r in db.query("SELECT utterance_id, embedding FROM utterance_voice")}


@pytest.fixture
def pcfg(cfg_factory, tmp_path: Path):
    cfg = cfg_factory({"memory": {"retrieval": "fts", "embedding_model": "hash"},
                       "correction": {"enabled": False, "glossary_file": str(tmp_path / "glossary.txt")}})
    yield cfg
    from audio_note.logging_setup import _remove_own_handlers

    _remove_own_handlers(logging.getLogger())


@pytest.fixture
def pdb(pcfg) -> DB:
    return DB(pcfg.db_path)


@pytest.fixture
def jia(pdb: DB) -> int:
    sid = pdb.upsert_speaker("甲")
    pdb.add_speaker_embedding(sid, A, "mic1.wav", 10.0, origin="mic")
    return sid


# ================================================================================================ first pass
def test_processing_stores_vectors_guesses_and_decision_details(pcfg, pdb, jia):
    audio = np.concatenate([_dc(3.0, 0.2), _dc(3.0, 0.4), _dc(3.0, 0.6), _dc(0.8, 0.2)])
    seg = _seg(pcfg, pdb, datetime(2026, 11, 4, 9), audio)
    eng = NumpyEngine(pcfg, {0.2: NEAR, 0.4: MAYBE, 0.6: OTHER},
                      turns=[Turn(0.0, 3.0, 0), Turn(3.0, 6.0, 1), Turn(6.0, 9.0, 2), Turn(9.0, 9.8, 0)])
    assert Processor(pcfg, pdb, engine=eng, asr=_ASR()).process_segment(seg) == 4
    sure, maybe, nobody, short = _rows(pdb, seg)
    assert (sure["speaker_id"], sure["speaker_name"], sure["speaker_guess_id"]) == (jia, "甲", None)
    assert sure["speaker_score"] == pytest.approx(0.80, abs=1e-3)
    meta = json.loads(sure["speaker_meta"])
    assert meta["m"] == "anchor" and meta["cond"] == "near" and meta["raw"] == pytest.approx(0.8, abs=1e-3)
    # "maybe 甲" is shown and asked, never written as the speaker
    assert (maybe["speaker_id"], maybe["speaker_name"]) == (None, "未知")
    assert maybe["speaker_guess_id"] == jia and maybe["speaker_guess_score"] == pytest.approx(0.40, abs=1e-3)
    assert (nobody["speaker_id"], nobody["speaker_guess_id"]) == (None, None) and json.loads(nobody["speaker_meta"])["m"] == "none"
    # a sentence under a second cannot decide alone: it follows its cluster (here the first sentence's voice)
    assert short["speaker_id"] == jia and short["cluster_label"] == sure["cluster_label"]
    # one voice vector per sentence, L2-normalised, tagged with the model
    vecs = _voice_rows(pdb)
    assert set(vecs) == {r["id"] for r in (sure, maybe, nobody, short)}
    assert float(vecs[sure["id"]] @ NEAR) == pytest.approx(1.0, abs=1e-5) and float(vecs[maybe["id"]] @ MAYBE) == pytest.approx(1.0, abs=1e-5)
    row = pdb.one("SELECT model, dim, speech_s FROM utterance_voice WHERE utterance_id=?", (sure["id"],))
    assert (row["model"], row["dim"], row["speech_s"]) == (scoring.model_id(pcfg), DIM, 3.0)
    # the cluster vectors carry their recording condition
    ce = pdb.one("SELECT loudness_db, system_share, model FROM cluster_embeddings WHERE segment_id=? AND cluster_label=?",
                 (seg, sure["cluster_label"]))
    assert -15.0 < ce["loudness_db"] < -13.0 and ce["system_share"] is None and ce["model"] == scoring.model_id(pcfg)


def test_sentence_vectors_reuse_what_identification_computed(pcfg, pdb, jia):
    """Deliverable 4: the per-sentence vectors must not double the embedding work."""
    audio = np.concatenate([_dc(3.0, 0.2), _dc(3.0, 0.4), _dc(3.0, 0.6)])
    seg = _seg(pcfg, pdb, datetime(2026, 11, 4, 9), audio)
    turns = [Turn(0.0, 3.0, 0), Turn(3.0, 6.0, 1), Turn(6.0, 9.0, 2)]
    eng = NumpyEngine(pcfg, {0.2: NEAR, 0.4: MAYBE, 0.6: OTHER}, turns=turns)
    Processor(pcfg, pdb, engine=eng, asr=_ASR()).process_segment(seg)
    assert eng.embeds == 6                                           # 3 clusters + 3 turn checks, nothing for the vectors
    assert len(_voice_rows(pdb)) == 3
    assert pdb.one("SELECT COUNT(*) AS n FROM usage_local WHERE stage='embed'")["n"] == 0
    # two turns of one cluster merged into one sentence: no single turn vector fits, the cluster vector does
    seg2 = _seg(pcfg, pdb, datetime(2026, 11, 4, 9, 5), np.concatenate([_dc(3.0, 0.2), _dc(3.0, 0.2)]))
    eng2 = NumpyEngine(pcfg, {0.2: NEAR}, turns=[Turn(0.0, 3.0, 0), Turn(3.2, 6.0, 0)])
    Processor(pcfg, pdb, engine=eng2, asr=_ASR()).process_segment(seg2)
    assert len(_rows(pdb, seg2)) == 1 and eng2.embeds == 3           # 1 cluster + 2 turn checks
    # a sentence that is neither one embedded turn nor a whole cluster costs one embedding, and that is accounted
    seg3 = _seg(pcfg, pdb, datetime(2026, 11, 4, 9, 10), np.concatenate([_dc(0.7, 0.2), _dc(2.0, 0.0), _dc(0.7, 0.2)]))
    eng3 = NumpyEngine(pcfg, {0.2: NEAR}, turns=[Turn(0.0, 0.7, 0), Turn(2.7, 3.4, 0)])
    Processor(pcfg, pdb, engine=eng3, asr=_ASR()).process_segment(seg3)
    assert len(_rows(pdb, seg3)) == 2 and eng3.embeds == 1 + 2       # the cluster, then one per short sentence
    use = pdb.one("SELECT n, audio_s FROM usage_local WHERE stage='embed'")
    assert use["n"] == 2 and use["audio_s"] == pytest.approx(1.4, abs=0.05)


def test_engines_that_do_not_score_keep_their_answers_and_get_no_vectors(pcfg, pdb, jia):
    class Fake:
        def diarize(self, audio):
            return [Turn(0.0, 3.0, 0), Turn(3.0, 6.0, 1)]

        def identify_turns(self, audio, turns, db, segment_id=None):
            turns[0].speaker_id, turns[0].speaker_name, turns[0].score = jia, "甲", 0.1      # whatever it says is stored
            if segment_id is not None:
                db.upsert_cluster_embedding(segment_id, 0, OTHER, 3.0)
            return turns

        def embed(self, audio):
            raise AssertionError("a fake's vectors are not stored")

    seg = _seg(pcfg, pdb, datetime(2026, 11, 4, 9), np.concatenate([_dc(3.0, 0.2), _dc(3.0, 0.4)]))
    assert Processor(pcfg, pdb, engine=Fake(), asr=_ASR()).process_segment(seg) == 2
    a, b = _rows(pdb, seg)
    assert (a["speaker_id"], a["speaker_score"], a["speaker_meta"], a["speaker_guess_id"]) == (jia, 0.1, None, None)
    assert b["speaker_id"] is None and _voice_rows(pdb) == {}


def test_the_new_sentence_is_decided_with_its_neighbours_in_time(pcfg, pdb, jia):
    """Tracks at processing time: a short sentence alone in its recording follows the same voice of a minute ago."""
    first = _seg(pcfg, pdb, datetime(2026, 11, 4, 9, 0, 0), _dc(6.0, 0.2))
    Processor(pcfg, pdb, engine=NumpyEngine(pcfg, {0.2: NEAR}, turns=[Turn(0.0, 6.0, 0)]), asr=_ASR()).process_segment(first)
    assert _rows(pdb, first)[0]["speaker_id"] == jia
    second = _seg(pcfg, pdb, datetime(2026, 11, 4, 9, 0, 30), _dc(0.8, 0.2))
    eng = NumpyEngine(pcfg, {0.2: _at(NEAR, 20, 0.9)}, turns=[Turn(0.0, 0.8, 0)])
    Processor(pcfg, pdb, engine=eng, asr=_ASR()).process_segment(second)
    r = _rows(pdb, second)[0]
    assert r["speaker_id"] == jia and json.loads(r["speaker_meta"])["m"] == "track"
    # the same short sentence an hour later has nobody to follow: "maybe 甲"
    third = _seg(pcfg, pdb, datetime(2026, 11, 4, 10, 30, 0), _dc(0.8, 0.2))
    Processor(pcfg, pdb, engine=eng, asr=_ASR()).process_segment(third)
    r = _rows(pdb, third)[0]
    assert r["speaker_id"] is None and r["speaker_guess_id"] == jia


def test_far_voice_is_learned_after_confirmation_and_history_follows(pcfg, pdb, jia):
    """The loop of SPEC 5.1: far-away speech is only "maybe" -> the user confirms -> adapt -> the rest follows."""
    segs = []
    for k in range(4):
        seg = _seg(pcfg, pdb, datetime(2026, 11, 4, 9 + k, 0, 0), _dc(5.0, 0.02))      # level 0.02: -34 dBFS, "far"
        eng = NumpyEngine(pcfg, {0.0: _same_voice(A, 30, 31 + k, 0.42)}, turns=[Turn(0.0, 5.0, 0)])
        Processor(pcfg, pdb, engine=eng, asr=_ASR()).process_segment(seg)
        segs.append(seg)
    rows = [_rows(pdb, s)[0] for s in segs]
    assert all(r["speaker_id"] is None and r["speaker_guess_id"] == jia for r in rows)
    assert all(json.loads(r["speaker_meta"])["cond"] == "far" for r in rows)
    q = review.review_queue(pdb, pcfg, jia)
    assert [i["kind"] for i in q] == ["guess"] * 4 and all(i["influence"] == 3 for i in q) and review.review_total(pdb, pcfg, jia) == 4
    # three confirmations (speakers.adapt_min_clusters) make a far centroid; the fourth sentence follows by itself
    for r in rows[:2]:
        res = review.confirm_utterance(pdb, pcfg, None, r["id"], jia)
        assert res["confirmed"] == 1 and res["sample_added"] is False and res["rescored"] == 0
    res = review.confirm_utterance(pdb, pcfg, None, rows[2]["id"], jia)
    assert res["rescored"] == 1
    last = _rows(pdb, segs[3])[0]
    assert last["speaker_id"] == jia and last["speaker_guess_id"] is None and json.loads(last["speaker_meta"])["m"] == "centroid:far"
    assert {r["condition"] for r in pdb.query("SELECT condition FROM speaker_centroids")} == {"all", "far"}
    confirmed = pdb.one("SELECT speaker_source, speaker_guess_id FROM utterances WHERE id=?", (rows[0]["id"],))
    assert confirmed["speaker_source"] == "user" and confirmed["speaker_guess_id"] is None
    left = review.review_queue(pdb, pcfg, jia)                       # no "maybe" left; the follower can still be checked
    assert [(i["utterance_id"], i["kind"]) for i in left] == [(last["id"], "sure")]
    # ... and the next far sentence is recognised at once
    seg = _seg(pcfg, pdb, datetime(2026, 11, 4, 15, 0, 0), _dc(5.0, 0.02))
    Processor(pcfg, pdb, engine=NumpyEngine(pcfg, {0.0: _same_voice(A, 30, 40, 0.42)}, turns=[Turn(0.0, 5.0, 0)]),
              asr=_ASR()).process_segment(seg)
    assert _rows(pdb, seg)[0]["speaker_id"] == jia


# ================================================================================================ relabel / split
def test_relabel_writes_vectors_and_guesses_and_counts_what_changed(pcfg, pdb, jia):
    audio = np.concatenate([_dc(3.0, 0.2), _dc(3.0, 0.4), _dc(3.0, 0.6)])
    seg = _seg(pcfg, pdb, datetime(2026, 11, 4, 9), audio, status="done")
    ids = pdb.add_utterances([{
        "segment_id": seg, "start_s": 3.0 * k, "end_s": 3.0 * k + 3.0, "abs_start": f"{DAY}T09:00:0{3 * k}", "day": DAY,
        "cluster_label": k, "speaker_id": (jia if k == 2 else None), "speaker_name": ("甲" if k == 2 else "未知"),
        "speaker_score": None, "text_raw": "话", "text": "话", "text_seg": "话"} for k in range(3)])
    eng = NumpyEngine(pcfg, {0.2: NEAR, 0.4: MAYBE, 0.6: OTHER})
    assert Processor(pcfg, pdb, engine=eng).relabel_segment(seg) == 2       # one gained 甲, one lost 甲
    a, b, c = _rows(pdb, seg)
    assert (a["speaker_id"], b["speaker_id"], c["speaker_id"]) == (jia, None, None)
    assert (b["speaker_guess_id"], c["speaker_guess_id"]) == (jia, None) and json.loads(a["speaker_meta"])["m"] == "anchor"
    assert set(_voice_rows(pdb)) == set(ids)
    assert Processor(pcfg, pdb, engine=eng).relabel_segment(seg) == 0       # stable
    # a veto answers the "maybe" as well (db.reject_utterances) and relabel keeps it away
    pdb.reject_utterances(jia, [b["id"]], "未知")
    assert pdb.one("SELECT speaker_guess_id FROM utterances WHERE id=?", (b["id"],))["speaker_guess_id"] is None
    Processor(pcfg, pdb, engine=eng).relabel_segment(seg)
    assert pdb.one("SELECT speaker_guess_id FROM utterances WHERE id=?", (b["id"],))["speaker_guess_id"] is None


def test_split_pieces_get_vectors_and_the_same_rules(pcfg, pdb, jia):
    audio = np.concatenate([_dc(3.0, 0.2), _dc(3.0, 0.4)])
    seg = _seg(pcfg, pdb, datetime(2026, 11, 4, 9), audio, status="done")
    uid = pdb.add_utterances([{"segment_id": seg, "start_s": 0.0, "end_s": 6.0, "abs_start": f"{DAY}T09:00:00", "day": DAY,
                               "cluster_label": 0, "speaker_id": None, "speaker_name": "未知", "speaker_score": None,
                               "text_raw": "两个人", "text": "两个人", "text_seg": "两个人"}])[0]
    eng = NumpyEngine(pcfg, {0.2: NEAR, 0.4: MAYBE})
    proc = Processor(pcfg, pdb, engine=eng, asr=_ASR())
    prev = proc.preview_split(uid, at=[3.0], auto=False)
    assert [p["speaker_id"] for p in prev["pieces"]] == [jia, None]
    res = proc.split_utterance(uid, at=[3.0])
    a, b = _rows(pdb, seg)
    assert [p["id"] for p in res["pieces"]] == [a["id"], b["id"]]
    assert a["speaker_id"] == jia and b["speaker_id"] is None and b["speaker_guess_id"] == jia      # "maybe" for the second piece
    vecs = _voice_rows(pdb)
    assert set(vecs) == {a["id"], b["id"]} and float(vecs[b["id"]] @ MAYBE) == pytest.approx(1.0, abs=1e-5)


# ================================================================================================ review queue
def _utt(db: DB, seg: int, hhmmss: str, start: float, end: float, speaker_id=None, guess=None, guess_score=None, score=None,
         cluster: int = 0, voice: np.ndarray | None = None) -> int:
    uid = db.add_utterances([{
        "segment_id": seg, "start_s": start, "end_s": end, "abs_start": f"{DAY}T{hhmmss}", "day": DAY, "cluster_label": cluster,
        "speaker_id": speaker_id, "speaker_name": "甲" if speaker_id else "未知", "speaker_score": score, "text_raw": hhmmss,
        "text": hhmmss, "text_seg": hhmmss, "speaker_guess_id": guess, "speaker_guess_score": guess_score}])[0]
    if voice is not None:
        db.execute("INSERT INTO utterance_voice(utterance_id, model, dim, embedding, speech_s) VALUES (?,?,?,?,?)",
                   (uid, "m", DIM, np.asarray(voice, dtype=np.float32).tobytes(), end - start))
    return uid


def test_review_queue_asks_from_both_sides_most_influential_guess_first(pcfg, pdb, jia):
    seg = _seg(pcfg, pdb, datetime(2026, 11, 4, 9), _dc(40.0, 0.2), status="done")
    crowd = _voice(50)                                                # three "maybe" sentences in one voice ...
    g1 = _utt(pdb, seg, "09:00:00", 0, 2, guess=jia, guess_score=0.16, voice=_at(crowd, 51, 0.9), cluster=1)
    g2 = _utt(pdb, seg, "09:00:02", 2, 6, guess=jia, guess_score=0.16, voice=_at(crowd, 52, 0.9), cluster=1)
    g3 = _utt(pdb, seg, "09:00:06", 6, 8, guess=jia, guess_score=0.22, voice=_at(crowd, 53, 0.9), cluster=1)
    lone = _utt(pdb, seg, "09:00:08", 8, 10, guess=jia, guess_score=0.24, voice=_voice(54), cluster=2)   # ... and a loner
    sure_low = _utt(pdb, seg, "09:00:10", 10, 12, speaker_id=jia, score=0.46)
    sure_high = _utt(pdb, seg, "09:00:12", 12, 14, speaker_id=jia, score=0.9)
    vetoed = _utt(pdb, seg, "09:00:14", 14, 16, guess=jia, guess_score=0.2, voice=_at(crowd, 55, 0.9))
    pdb.execute("INSERT INTO utterance_rejections(utterance_id, speaker_id) VALUES (?,?)", (vetoed, jia))
    _utt(pdb, seg, "09:00:16", 16, 16.5, guess=jia, guess_score=0.2)                                      # too short to judge by ear
    items = review.review_queue(pdb, pcfg, jia)
    assert [i["utterance_id"] for i in items] == [g3, g2, g1, lone, sure_low, sure_high]
    assert [i["kind"] for i in items] == ["guess"] * 4 + ["sure"] * 2
    assert [i["influence"] for i in items] == [2, 2, 2, 0, 0, 0]
    assert items[0]["guess_score"] == 0.22 and items[0]["score"] is None and items[4]["guess_score"] is None
    assert review.review_counts(pdb, pcfg, jia) == {"guess": 4, "sure": 2} and review.review_total(pdb, pcfg, jia) == 6
    assert [i["utterance_id"] for i in review.review_queue(pdb, pcfg, jia, limit=2, offset=3)] == [lone, sure_low]
    # whole_cluster on a "maybe": the cluster's other guesses become the user's answer too
    res = review.confirm_utterance(pdb, pcfg, None, g1, jia, whole_cluster=True)
    assert res["confirmed"] == 3
    got = {r["id"]: r for r in pdb.query("SELECT id, speaker_id, speaker_source, speaker_guess_id FROM utterances")}
    assert all((got[u]["speaker_id"], got[u]["speaker_source"], got[u]["speaker_guess_id"]) == (jia, "user", None) for u in (g1, g2, g3))
    assert got[lone]["speaker_id"] is None and got[vetoed]["speaker_id"] is None
    assert (pcfg.transcripts_dir / f"{DAY}.md").exists()


# ================================================================================================ API
@pytest.fixture
def api(pcfg):
    rt = Runtime(pcfg)
    with TestClient(create_app(rt), raise_server_exceptions=False) as c:
        yield c, rt
    rt.shutdown()


def test_api_review_scoring_overview_and_events(api, pcfg):
    client, rt = api
    db = rt.db
    jia = db.upsert_speaker("甲")
    keep = db.add_speaker_embedding(jia, A, None, 10.0, origin="mic")
    extra = db.add_speaker_embedding(jia, _at(A, 60, 0.9), None, 8.0, origin="file")
    seg = _seg(pcfg, db, datetime(2026, 11, 4, 9), _dc(20.0, 0.2), status="done")
    db.upsert_cluster_embedding(seg, 0, _at(A, 61, 0.8), 6.0)
    auto = _utt(db, seg, "09:00:00", 0, 3)                                        # the cluster is 甲's voice: not decided yet
    guess = _utt(db, seg, "09:00:03", 3, 6, guess=jia, guess_score=0.4, voice=_at(A, 62, 0.4), cluster=1)

    sc = client.get("/api/speakers/scoring").json()
    assert sc["scoring"] == "v5" and sc["normalised"] is False and sc["cohort"] == 0 and sc["cohort_min"] == 20
    assert sc["thresholds"] == {"sure": 0.45, "guess": pytest.approx(0.37), "unit": "cosine"}
    assert sc["speakers"] == [{"id": jia, "name": "甲", "samples": 2, "outlier_samples": 0, "centroids": []}]
    assert sc["states"] == {"sure": 0, "guess": 1, "unknown": 1, "user": 0} and sc["last_rescore"] is None and sc["stale"] is True

    sp = client.get("/api/speakers").json()["speakers"][0]
    assert sp["guesses"] == 1 and sp["centroids"] == []
    rv = client.get(f"/api/speakers/{jia}/review").json()
    assert (rv["total"], rv["guess"], rv["sure"]) == (1, 1, 0) and rv["items"][0]["kind"] == "guess" and rv["items"][0]["utterance_id"] == guess

    # deleting a sample changes the reference voice: history is brought in step inside the request, and the page is told
    sub = rt.events.subscribe()
    r = client.delete(f"/api/speakers/samples/{extra}")
    assert r.status_code == 200 and r.json() == {"ok": True, "deleted": 1, "rescored": 1}
    assert db.one("SELECT speaker_id FROM utterances WHERE id=?", (auto,))["speaker_id"] == jia
    seen = []
    while (ev := sub.get(timeout=0.2)) is not None:
        seen.append((ev["type"], ev.get("change")))
    assert ("speakers", "rescored") in seen and ("speakers", "sample_deleted") in seen
    sc = client.get("/api/speakers/scoring").json()
    assert sc["stale"] is False and sc["last_rescore"]["summary"]["changed"] == 1 and sc["states"]["sure"] == 1
    assert [r["source"] for r in db.query("SELECT source FROM job_runs WHERE name='rescore'")] == ["ui"]
    assert client.delete(f"/api/speakers/samples/{extra}").json() == {"ok": True, "deleted": 0}      # nothing changed: no key

    # the rescore job: a dry run reports, the real run writes
    r = client.post("/api/jobs/rescore", json={"dry_run": True})
    assert r.status_code == 200 and rt.wait_jobs(30)
    assert rt.jobs["rescore"].error is None and rt.jobs["rescore"].result["dry_run"] is True

    # the speaker goes: nobody can be "maybe" them any more
    assert client.patch(f"/api/speakers/{jia}", json={"is_self": True}).json()["ok"] is True
    assert client.delete(f"/api/speakers/{jia}").status_code == 200
    assert db.one("SELECT speaker_guess_id FROM utterances WHERE id=?", (guess,))["speaker_guess_id"] is None
    assert keep and client.get("/api/speakers/scoring").json()["speakers"] == []


# ================================================================================================ command line
def _run(argv: list[str], capsys) -> tuple[int, str]:
    capsys.readouterr()
    code = cli.main(argv)
    return code, capsys.readouterr().out


def test_cli_rescore_and_scoring(pcfg, pdb, jia, capsys):
    seg = _seg(pcfg, pdb, datetime(2026, 11, 4, 9), _dc(10.0, 0.2), status="done")
    pdb.upsert_cluster_embedding(seg, 0, _at(A, 70, 0.8), 6.0)
    uid = _utt(pdb, seg, "09:00:00", 0, 3)
    code, out = _run(["speakers", "rescore", "--dry-run", "--json"], capsys)
    data = json.loads(out)
    assert code == 0 and data["dry_run"] is True and data["changed"] == 1 and data["after"] == {"sure": 1, "guess": 0, "unknown": 0}
    assert pdb.one("SELECT speaker_id FROM utterances WHERE id=?", (uid,))["speaker_id"] is None
    code, out = _run(["speakers", "rescore"], capsys)
    assert code == 0 and "1 句有变化" in out and "重判说话人" in out
    assert DB(pcfg.db_path).one("SELECT speaker_id FROM utterances WHERE id=?", (uid,))["speaker_id"] == jia
    code, out = _run(["speakers", "rescore", "--day", DAY, "--json"], capsys)
    assert code == 0 and json.loads(out)["changed"] == 0
    code, out = _run(["speakers", "scoring", "--json"], capsys)
    data = json.loads(out)
    assert code == 0 and data["scoring"] == "v5" and data["states"] == {"sure": 1, "guess": 0, "unknown": 0, "user": 0}
    assert data["speakers"][0]["name"] == "甲" and data["stale"] is False and data["thresholds"]["unit"] == "cosine"
    code, out = _run(["speakers", "scoring"], capsys)
    assert code == 0 and "评分方式" in out and "确定 1 句" in out
