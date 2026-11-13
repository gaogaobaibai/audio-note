"""Speakers v5: re-deciding history from stored vectors (scoring.rescore / refresh / run_pending / job_rescore).

No audio, no model: rows, cluster vectors and utterance_voice vectors are written by hand.
"""
from __future__ import annotations

import json
import math
import threading
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from audio_note import chain, events
from audio_note.db import DB
from audio_note.speech import scoring

DIM = 64


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


A, B = _voice(1), _voice(2)
T0 = datetime(2026, 11, 4, 9, 0, 0)


@pytest.fixture
def rcfg(cfg_factory):
    return cfg_factory({"memory": {"retrieval": "fts", "embedding_model": "hash"}})


@pytest.fixture
def rdb(rcfg) -> DB:
    return DB(rcfg.db_path)


class World:
    """Rows written by hand: world.seg(...) makes a segment, world.utt(...) a sentence in it."""

    def __init__(self, db: DB):
        self.db = db
        self.jia = db.upsert_speaker("甲")
        db.add_speaker_embedding(self.jia, A, None, 10.0, origin="mic")
        db.set_self_speaker(self.jia)
        self.n = 0

    def seg(self, at_s: float, clusters: dict[int, np.ndarray] | None = None, speech_s: float = 6.0) -> int:
        when = T0 + timedelta(seconds=at_s)
        seg = self.db.add_segment(f"s{self.n}.wav", when, 30.0)
        self.n += 1
        self.db.set_segment_status(seg, "done")
        for cl, emb in (clusters or {}).items():
            self.db.upsert_cluster_embedding(seg, cl, emb, speech_s)
        return seg

    def utt(self, seg: int, start: float, dur: float, cluster: int = 0, speaker_id: int | None = None, source: str | None = None,
            voice: np.ndarray | None = None, loudness: float | None = -20.0, system_share: float | None = None,
            score: float | None = None) -> int:
        row = self.db.get_segment(seg)
        when = datetime.fromisoformat(row["started_at"]) + timedelta(seconds=start)
        uid = self.db.add_utterances([{
            "segment_id": seg, "start_s": start, "end_s": start + dur, "abs_start": when.isoformat(timespec="milliseconds"),
            "day": when.strftime("%Y-%m-%d"), "cluster_label": cluster, "speaker_id": speaker_id,
            "speaker_name": "甲" if speaker_id == self.jia else "未知", "speaker_score": score, "text_raw": "话", "text": "话",
            "text_seg": "话", "speaker_source": source, "loudness_db": loudness, "system_share": system_share}])[0]
        if voice is not None:
            self.db.execute("INSERT INTO utterance_voice(utterance_id, model, dim, embedding, speech_s) VALUES (?,?,?,?,?)",
                            (uid, scoring.model_id({"speakers": {"embedding_model": "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"}}),
                             DIM, np.asarray(voice, dtype=np.float32).tobytes(), dur))
        return uid

    def row(self, uid: int):
        return self.db.one("SELECT * FROM utterances WHERE id=?", (uid,))


@pytest.fixture
def world(rdb) -> World:
    return World(rdb)


def test_rescore_decides_from_stored_vectors_and_reports_before_and_after(world, rdb, rcfg):
    mine, strange = _at(A, 10, 0.8), _voice(11)
    seg = world.seg(0, {0: mine, 1: strange})
    a = world.utt(seg, 0, 3, cluster=0)                                  # unknown so far, the cluster is 甲's voice
    b = world.utt(seg, 3, 3, cluster=1, speaker_id=world.jia, score=0.5)  # attributed to 甲 by an old rule: not 甲's voice
    c = world.utt(seg, 6, 3, cluster=0, speaker_id=world.jia, source="user")
    d = world.utt(seg, 9, 3, cluster=5)                                  # no vector at all
    dry = scoring.rescore(rdb, rcfg, dry_run=True)
    assert dry["dry_run"] and dry["changed"] == 2 and world.row(a)["speaker_id"] is None      # nothing written
    res = scoring.rescore(rdb, rcfg)
    assert res["utterances"] == 4 and res["scored"] == 2 and res["user"] == 1 and res["no_vector"] == 1 and res["faint"] == 0
    assert res["before"] == {"sure": 1, "guess": 0, "unknown": 2} and res["after"] == {"sure": 1, "guess": 0, "unknown": 2}
    assert res["changed"] == 2 and res["speaker_changed"] == 2 and res["moves"] == {"sure>unknown": 1, "unknown>sure": 1}
    assert res["days"] == ["2026-11-04"] and res["normalised"] is False and res["version"] and res["wall_s"] < 5
    ra, rb, rc_, rd = (world.row(x) for x in (a, b, c, d))
    assert (ra["speaker_id"], ra["speaker_name"], ra["speaker_source"]) == (world.jia, "甲", None)
    assert ra["speaker_score"] == pytest.approx(0.8, abs=1e-3)
    assert json.loads(ra["speaker_meta"]) == {"cond": "near", "m": "anchor", "raw": 0.8, "vec": "cluster"}
    assert (rb["speaker_id"], rb["speaker_name"]) == (None, "未知") and json.loads(rb["speaker_meta"])["m"] == "none"
    assert (rc_["speaker_id"], rc_["speaker_source"], rc_["speaker_meta"]) == (world.jia, "user", None)     # never touched
    assert (rd["speaker_id"], rd["speaker_meta"]) == (None, None)
    again = scoring.rescore(rdb, rcfg)                                   # idempotent
    assert again["changed"] == 0 and again["before"] == again["after"] == res["after"]


def test_a_sentence_vector_wins_over_its_cluster_vector_and_maybe_is_never_a_speaker(world, rdb, rcfg):
    seg = world.seg(0, {0: _at(A, 20, 0.8)})
    own_other = world.utt(seg, 0, 3, voice=_voice(21))                   # the cluster says 甲, its own voice says no
    own_maybe = world.utt(seg, 3, 3, voice=_at(A, 22, 0.40))
    follows = world.utt(seg, 6, 3)                                       # no vector of its own: the cluster's
    scoring.rescore(rdb, rcfg)
    r1, r2, r3 = world.row(own_other), world.row(own_maybe), world.row(follows)
    assert r1["speaker_id"] is None and json.loads(r1["speaker_meta"])["vec"] == "utt"
    assert r2["speaker_id"] is None and r2["speaker_guess_id"] == world.jia and r2["speaker_guess_score"] == pytest.approx(0.40, abs=1e-3)
    assert r2["speaker_name"] == "未知"
    assert r3["speaker_id"] == world.jia and r3["speaker_guess_id"] is None
    # a vector of another voice model is not comparable: ignored, the cluster vector decides
    rdb.execute("UPDATE utterance_voice SET model='some_other_model' WHERE utterance_id=?", (own_other,))
    scoring.rescore(rdb, rcfg)
    assert world.row(own_other)["speaker_id"] == world.jia


def test_vetoes_system_audio_and_faint_sentences(world, rdb, rcfg):
    mine = _at(A, 30, 0.8)
    seg = world.seg(0, {0: mine})
    ok = world.utt(seg, 0, 3)
    vetoed = world.utt(seg, 3, 3)
    system = world.utt(seg, 6, 3, system_share=0.9)
    faint = world.utt(seg, 9, 3, loudness=-55.0, speaker_id=world.jia, score=0.6)
    faint_vetoed = world.utt(seg, 12, 3, loudness=-55.0, speaker_id=world.jia)
    rdb.execute("INSERT INTO utterance_rejections(utterance_id, speaker_id) VALUES (?,?)", (vetoed, world.jia))
    rdb.execute("INSERT INTO utterance_rejections(utterance_id, speaker_id) VALUES (?,?)", (faint_vetoed, world.jia))
    res = scoring.rescore(rdb, rcfg)
    assert world.row(ok)["speaker_id"] == world.jia
    for uid in (vetoed, system):                                         # the cluster is 甲, these two may not be
        r = world.row(uid)
        assert r["speaker_id"] is None and r["speaker_guess_id"] is None and json.loads(r["speaker_meta"])["m"] == "veto"
    assert res["faint"] == 2
    assert (world.row(faint)["speaker_id"], world.row(faint)["speaker_score"]) == (world.jia, 0.6)     # too quiet: kept as it is
    assert world.row(faint_vetoed)["speaker_id"] is None                 # ... except what it may not be


def test_short_sentences_follow_their_neighbours_across_segments(world, rdb, rcfg):
    strong = _at(A, 40, 0.85)
    s1 = world.seg(0, {0: strong}, speech_s=8.0)
    long_ = world.utt(s1, 0, 8)
    s2 = world.seg(20, {0: _at(strong, 41, 0.8)}, speech_s=0.7)           # the next recording: one short "hm" in that voice
    short = world.utt(s2, 0, 0.7)
    s3 = world.seg(4000, {0: _at(strong, 42, 0.8)}, speech_s=0.7)         # an hour later: another conversation
    late = world.utt(s3, 0, 0.7)
    scoring.rescore(rdb, rcfg)
    assert world.row(long_)["speaker_id"] == world.jia
    assert world.row(short)["speaker_id"] == world.jia and json.loads(world.row(short)["speaker_meta"])["m"] == "track"
    assert world.row(late)["speaker_id"] is None and world.row(late)["speaker_guess_id"] == world.jia
    # sessions, once cut, are the conversation boundaries
    rdb.execute("UPDATE utterances SET session_id=1 WHERE id=?", (long_,))
    rdb.execute("UPDATE utterances SET session_id=2 WHERE id=?", (short,))
    scoring.rescore(rdb, rcfg)
    assert world.row(short)["speaker_id"] is None


def test_day_and_segment_filters_use_the_neighbours_as_context_but_write_only_the_selection(world, rdb, rcfg):
    strong = _at(A, 50, 0.85)
    s1 = world.seg(0, {0: strong}, speech_s=8.0)
    ctx = world.utt(s1, 0, 8)                                             # stays unknown: not selected
    s2 = world.seg(20, {0: _at(strong, 51, 0.8)}, speech_s=0.7)
    short = world.utt(s2, 0, 0.7)
    res = scoring.rescore(rdb, rcfg, segment_ids=[s2])
    assert res["utterances"] == 1 and world.row(short)["speaker_id"] == world.jia     # its neighbour vouched for it
    assert world.row(ctx)["speaker_id"] is None and world.row(ctx)["speaker_meta"] is None
    assert scoring.rescore(rdb, rcfg, day="2026-11-05")["utterances"] == 0
    assert scoring.rescore(rdb, rcfg, day="2026-11-04")["utterances"] == 2 and world.row(ctx)["speaker_id"] == world.jia
    assert scoring.rescore(rdb, rcfg, day="2026-11-04", segment_ids=[s1, 99999])["utterances"] == 1


def test_rescore_never_overwrites_a_label_or_veto_that_arrived_meanwhile(world, rdb, rcfg, monkeypatch):
    seg = world.seg(0, {0: _at(A, 60, 0.8), 1: _voice(61)})
    racing = world.utt(seg, 0, 3, cluster=1, speaker_id=world.jia)        # rescore wants to release it ...
    vetoed_now = world.utt(seg, 3, 3, cluster=0)                          # ... and to give this one to 甲
    from audio_note.speech import tracks

    real = tracks.decide

    def decide(obs, g, tp=None, **kw):
        out = real(obs, g, tp, **kw)
        rdb.execute("UPDATE utterances SET speaker_source='user' WHERE id=?", (racing,))           # the user was faster
        rdb.execute("INSERT INTO utterance_rejections(utterance_id, speaker_id) VALUES (?,?)", (vetoed_now, world.jia))
        return out

    monkeypatch.setattr(tracks, "decide", decide)
    scoring.rescore(rdb, rcfg)
    assert (world.row(racing)["speaker_id"], world.row(racing)["speaker_source"]) == (world.jia, "user")
    assert world.row(vetoed_now)["speaker_id"] is None


def test_refresh_adapts_rescores_once_and_leaves_a_job_run(world, rdb, rcfg):
    wild = _at(A, 70, 0.7)
    ids = []
    for k in range(40):                                                   # strangers: the cohort
        seg = world.seg(100 + 40 * k, {0: _voice(700 + k)})
        world.utt(seg, 0, 6)
    for k in range(6):
        seg = world.seg(3000 + 40 * k, {0: _at(wild, 710 + k, 0.93)})
        ids.append(world.utt(seg, 0, 6))
    bus = events.EventBus()
    sub = bus.subscribe()
    out = scoring.refresh(rdb, rcfg, events=bus, source="chain")
    assert out["adapted"] and out["rescored"] and out["rescore"]["changed"] == 6 and out["rescore"]["normalised"]
    assert all(world.row(u)["speaker_id"] == world.jia for u in ids)
    assert {r["condition"] for r in rdb.query("SELECT condition FROM speaker_centroids")} == {"all", "near"}
    runs = rdb.query("SELECT name, source, status, result FROM job_runs")
    assert [(r["name"], r["source"], r["status"]) for r in runs] == [("rescore", "chain", "done")]
    summary = json.loads(runs[0]["result"])
    assert summary["before"] == {"sure": 0, "guess": 0, "unknown": 46} and summary["after"]["sure"] == 6 and summary["changed"] == 6
    ev = sub.get(timeout=1)
    assert ev["type"] == "speakers" and ev["change"] == "rescored" and ev["changed"] == 6 and ev["days"] == ["2026-11-04"]
    kv = rdb.get_kv(scoring.KV_RESCORE)
    assert kv["version"] == out["version"] and kv["summary"]["changed"] == 6
    # nothing changed: neither adapt nor rescore runs again, no second history row
    again = scoring.refresh(rdb, rcfg, events=bus)
    assert again == {"adapted": False, "rescored": False, "version": out["version"]}
    assert len(rdb.query("SELECT 1 FROM job_runs")) == 1
    # a new sample changes the gallery version: history is re-decided (here: nothing moves)
    rdb.add_speaker_embedding(world.jia, _at(A, 720, 0.95), None, 8.0, origin="file")
    third = scoring.refresh(rdb, rcfg, source="ui")
    assert third["rescored"] and third["version"] != out["version"] and third["rescore"]["changed"] == 0
    assert [r["source"] for r in rdb.query("SELECT source FROM job_runs ORDER BY id")] == ["chain", "ui"]
    assert scoring.refresh(rdb, rcfg, force=True)["rescored"] is True
    stop = threading.Event()
    stop.set()
    rdb.add_speaker_negative(world.jia, _voice(721), None, 2.0)
    assert scoring.refresh(rdb, rcfg, stop_event=stop) == {"adapted": False, "rescored": False}     # a stop request: later


def test_run_pending_is_a_chain_step_and_safe_on_an_empty_database(rdb, rcfg):
    assert ("rescore", "audio_note.speech.scoring:run_pending") in chain.STEPS
    assert chain.STEPS[0][0] == "sessions" and chain.STEPS[1][0] == "rescore"       # after the sessions were cut
    assert scoring.run_pending(rcfg, rdb, None, stop_event=None, events=None, limit=3) == {"adapted": False, "rescored": False}
    out = chain.run_chain(rcfg, rdb, lambda: None, only=("rescore",))
    assert out["ran"] == ["rescore"] and out["failed"] == {}
    assert rdb.query("SELECT 1 FROM job_runs") == [] and rdb.get_kv(scoring.KV_RESCORE) is None
    assert scoring.rescore(rdb, rcfg)["utterances"] == 0


def test_job_rescore_takes_the_arguments_of_the_job_endpoint(world, rdb, rcfg):
    seg = world.seg(0, {0: _at(A, 80, 0.8)})
    uid = world.utt(seg, 0, 3)
    rt = SimpleNamespace(db=rdb, cfg=rcfg, events=events.NULL, job_stop_event=lambda: None)
    dry = scoring.job_rescore(rt, dry_run="true", day="2026-11-04")
    assert dry["dry_run"] is True and dry["changed"] == 1 and "adapt" not in dry and world.row(uid)["speaker_id"] is None
    assert rdb.get_kv(scoring.KV_RESCORE) is None
    one = scoring.job_rescore(rt, segment_id=seg)
    assert one["changed"] == 1 and one["adapt"] == {"written": 0, "changed": False} and world.row(uid)["speaker_id"] == world.jia
    assert rdb.get_kv(scoring.KV_RESCORE) is None                       # a partial run does not count as "history is in step"
    full = scoring.job_rescore(rt)
    assert full["changed"] == 0 and rdb.get_kv(scoring.KV_RESCORE)["version"] == full["version"]
    assert scoring.job_rescore(rt, segment_ids="%d, 7" % seg)["utterances"] == 1
    with pytest.raises(ValueError):
        scoring.job_rescore(rt, segment_ids=["x"])
    # after a full job run the chain step has nothing left to do
    assert scoring.refresh(rdb, rcfg)["rescored"] is False


def test_rescore_without_enrolled_speakers_releases_what_nobody_can_vouch_for(rdb, rcfg):
    seg = rdb.add_segment("s.wav", T0, 10.0)
    rdb.set_segment_status(seg, "done")
    gone = rdb.upsert_speaker("走了")                                     # a speaker whose samples were all deleted
    rdb.upsert_cluster_embedding(seg, 0, A, 5.0)
    rows = [{"segment_id": seg, "start_s": s, "end_s": s + 3, "abs_start": (T0 + timedelta(seconds=s)).isoformat(timespec="milliseconds"),
             "day": "2026-11-04", "cluster_label": cl, "speaker_id": gone, "speaker_name": "走了", "speaker_score": 0.6,
             "text_raw": "话", "text": "话", "text_seg": "话"} for s, cl in ((0, 0), (3, 9))]
    with_vec, without = rdb.add_utterances(rows)
    res = scoring.rescore(rdb, rcfg)
    assert res["changed"] == 1 and res["no_vector"] == 1
    assert rdb.one("SELECT speaker_id, speaker_name FROM utterances WHERE id=?", (with_vec,))["speaker_id"] is None
    assert rdb.one("SELECT speaker_id FROM utterances WHERE id=?", (without,))["speaker_id"] == gone     # nothing to judge it by
