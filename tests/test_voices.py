"""audio_note.speech.voices: grouping unidentified clusters (synthetic embeddings, pure numpy), backfill +
naming with the real CAM++ engine on the sample recording, and relabel_segment leaving user labels alone.

No network, no microphone. The real-engine tests are skipped when the sample recording is missing.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from audio_note.audio.storage import segment_path
from audio_note.audio.vad import SAMPLE_RATE as SR
from audio_note.db import DB
from audio_note.speech import voices
from audio_note.speech.pipeline import Processor
from audio_note.speech.speakers import Turn
from audio_note.speech.voices import backfill_cluster_embeddings, cluster_members, group_unknown_voices, name_voice

from .conftest import SAMPLE_WAV

DAY = "2026-10-30"
needs_sample = pytest.mark.skipif(not SAMPLE_WAV.exists(), reason="real sample recording not available")


# --- helpers (same shape as tests/test_ui_api.py) --------------------------------------------------------

def _tone(seconds: float, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _sample_slice(a: float, b: float) -> np.ndarray:
    audio, sr = sf.read(str(SAMPLE_WAV), dtype="float32")
    assert sr == SR
    return audio[int(a * SR):int(b * SR)]


def _seg(cfg, db: DB, when: datetime, audio: np.ndarray | None = None, status: str = "done", with_file: bool = True) -> tuple[int, Path]:
    audio = _tone(5.0) if audio is None else audio
    p = segment_path(cfg, when, ".wav")
    if with_file:
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
        "text_raw": text, "text": text, "speaker_source": source,
    }])[0]


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _voice(seed: int, dim: int = 192) -> np.ndarray:
    return _unit(np.random.default_rng(seed).standard_normal(dim))


def _near(base: np.ndarray, seed: int, noise: float = 0.03) -> np.ndarray:
    """A unit vector close to `base`: per-dimension noise 0.03 in 192-d is ~0.42 in norm -> cosine ~0.92."""
    return _unit(base + noise * np.random.default_rng(seed).standard_normal(base.shape[0]))


# --- cluster_members: pure clustering ------------------------------------------------------------------------

def test_cluster_members_merges_similar_and_keeps_distinct():
    a, b = _voice(1), _voice(2)
    embs = np.stack([_near(a, 10), _near(a, 11), _near(b, 12), _near(a, 13), _near(b, 14)])
    groups = cluster_members(embs, np.ones(5, dtype=np.float32), 0.55)
    assert sorted(sorted(g) for g in groups) == [[0, 1, 3], [2, 4]]
    assert cluster_members(np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), 0.5) == []
    assert cluster_members(embs[:1], np.ones(1, dtype=np.float32), 0.5) == [[0]]


def test_cluster_members_threshold_boundary():
    e = np.zeros((2, 4), dtype=np.float32)
    e[0, 0] = 1.0
    e[1] = _unit(np.array([0.8, 0.6, 0.0, 0.0]))          # cosine exactly 0.8 with e[0]
    w = np.ones(2, dtype=np.float32)
    assert sorted(sorted(g) for g in cluster_members(e, w, 0.8)) == [[0, 1]]        # >= threshold merges
    assert sorted(sorted(g) for g in cluster_members(e, w, 0.8001)) == [[0], [1]]
    assert sorted(sorted(g) for g in cluster_members(e, w, 1.01)) == [[0], [1]]     # nothing merges above 1


def test_cluster_members_centroid_is_speech_weighted():
    # a heavy member dominates its group's centroid: with weights the third row joins, without them it does not
    base = _voice(3)
    far = _unit(base + 0.9 * _voice(4))
    probe = _near(base, 5, noise=0.5)
    embs = np.stack([base, far, probe])
    thr = float(np.dot(probe, _unit(base * 10 + far))) - 1e-3
    assert float(np.dot(probe, _unit(base + far))) < thr                 # equal weights would not merge at thr
    heavy = cluster_members(embs, np.array([10.0, 1.0, 1.0], dtype=np.float32), thr)
    assert any(len(g) == 3 for g in heavy)


# --- group_unknown_voices: synthetic cluster_embeddings ------------------------------------------------------------

def _store(db, seg: int, cluster: int, emb: np.ndarray, speech_s: float) -> None:
    db.upsert_cluster_embedding(seg, cluster, emb, speech_s)


def test_group_unknown_voices_empty_db(cfg, db):
    out = group_unknown_voices(db, cfg)
    assert out == {"groups": [], "missing_segments": 0, "threshold": 0.55, "unknown_utterances": 0}
    assert group_unknown_voices(db, cfg, threshold=0.7)["threshold"] == 0.7
    cfg["speakers"]["voice_group_threshold"] = 0.4
    assert group_unknown_voices(db, cfg)["threshold"] == 0.4


def test_group_unknown_voices_groups_across_segments_and_sorts(cfg, db):
    a, b = _voice(21), _voice(22)
    s1, _ = _seg(cfg, db, datetime(2026, 10, 27, 9, 0))
    s2, _ = _seg(cfg, db, datetime(2026, 10, 29, 9, 0))
    s3, _ = _seg(cfg, db, datetime(2026, 10, 29, 10, 0), with_file=False)      # audio expired
    # voice A: cluster 0 of s1 + cluster 1 of s2 + cluster 0 of s3; voice B: cluster 1 of s1
    _store(db, s1, 0, _near(a, 1), 6.0)
    _store(db, s2, 1, _near(a, 2), 4.0)
    _store(db, s3, 0, _near(a, 3), 3.0)
    _store(db, s1, 1, _near(b, 4), 2.0)
    _store(db, s2, 0, _near(b, 5), 2.0)                 # cluster with no unidentified utterance: must not take part
    ua = [_utt(db, s1, "2026-10-27", "09:00", "我是A第一句比较长", None, None, 0.0, 3.0, cluster=0),
          _utt(db, s1, "2026-10-27", "09:01", "A短", None, None, 3.0, 4.0, cluster=0),
          _utt(db, s2, "2026-10-29", "09:05", "A在第二天", None, None, 0.0, 2.5, cluster=1),
          _utt(db, s3, "2026-10-29", "10:00", "A没有音频但最长最长", None, None, 0.0, 5.0, cluster=0)]
    ub = [_utt(db, s1, "2026-10-27", "09:02", "我是B", None, None, 4.0, 5.0, cluster=1)]
    known = db.upsert_speaker("张三")
    _utt(db, s2, "2026-10-29", "09:06", "已识别的句子", "张三", known, 2.5, 3.5, 0.9, cluster=0)
    _utt(db, s1, "2026-10-27", "09:03", "A的一句已经手标", "张三", known, 5.0, 5.5, cluster=0, source="user")

    out = group_unknown_voices(db, cfg)
    assert out["unknown_utterances"] == 5 and out["missing_segments"] == 0 and len(out["groups"]) == 2
    g_a, g_b = out["groups"]
    assert g_a["members"] == sorted([[s1, 0], [s2, 1], [s3, 0]]) and g_b["members"] == [[s1, 1]]
    assert g_a["utterances"] == 4 and g_a["speech_s"] == pytest.approx(3.0 + 1.0 + 2.5 + 5.0)
    assert g_a["segments"] == 3 and g_a["days"] == ["2026-10-27", "2026-10-29"] and g_a["n_days"] == 2
    assert len(g_a["key"]) == 10 and g_a["key"] != g_b["key"] and g_a["key"] == voices._group_key([(s2, 1), (s1, 0), (s3, 0)])
    # samples: with audio first, longest first; the no-audio one last; at most 5
    assert [s["utterance_id"] for s in g_a["samples"]] == [ua[0], ua[2], ua[1], ua[3]]
    assert g_a["samples"][0] == {"utterance_id": ua[0], "day": "2026-10-27", "time": "09:00", "text": "我是A第一句比较长",
                                 "seconds": 3.0, "audio": True}
    assert g_a["samples"][3]["audio"] is False
    assert g_b["utterances"] == 1 and g_b["samples"][0]["utterance_id"] == ub[0]
    assert g_a["closest"] is None                                           # 张三 has no embedding yet
    # with a strict threshold the three A members stay apart; groups are sorted by speech_s desc
    strict = group_unknown_voices(db, cfg, threshold=0.999)
    assert len(strict["groups"]) == 4 and [g["speech_s"] for g in strict["groups"]] == sorted((g["speech_s"] for g in strict["groups"]), reverse=True)
    # max_members keeps the most recent clusters only
    recent = group_unknown_voices(db, cfg, max_members=2)
    assert {tuple(m) for g in recent["groups"] for m in g["members"]} <= {(s2, 1), (s3, 0)}


def test_group_unknown_voices_closest_hint_and_sample_cap(cfg, db):
    a = _voice(31)
    s1, _ = _seg(cfg, db, datetime(2026, 10, 27, 9, 0))
    _store(db, s1, 0, a, 10.0)
    for i in range(7):
        _utt(db, s1, DAY, f"09:{i:02d}", f"第{i}句", None, None, float(i), float(i) + 0.5 + 0.1 * i, cluster=0)
    zhang = db.upsert_speaker("张三")
    li = db.upsert_speaker("李四")
    db.add_speaker_embedding(zhang, _near(a, 7, noise=0.3), None, 3.0)
    db.add_speaker_embedding(zhang, _voice(99), None, 3.0)                    # a far sample must not lower the max
    db.add_speaker_embedding(li, _voice(98), None, 3.0)
    out = group_unknown_voices(db, cfg)
    g = out["groups"][0]
    assert g["closest"]["id"] == zhang and g["closest"]["name"] == "张三"
    assert g["closest"]["score"] == pytest.approx(float(np.dot(a, _near(a, 7, noise=0.3))), abs=1e-3)
    assert len(g["samples"]) == 5 and [s["seconds"] for s in g["samples"]] == [1.1, 1.0, 0.9, 0.8, 0.7]
    assert g["utterances"] == 7


def test_group_unknown_voices_counts_missing_segments_and_ignores_fully_identified(cfg, db):
    known = db.upsert_speaker("张三")
    done_unknown, _ = _seg(cfg, db, datetime(2026, 10, 27, 9, 0))              # needs a scan
    _utt(db, done_unknown, DAY, "09:00", "x", None, None, 0.0, 2.0)
    done_known, _ = _seg(cfg, db, datetime(2026, 10, 27, 9, 5))                # fully identified: not missing
    _utt(db, done_known, DAY, "09:05", "y", "张三", known, 0.0, 2.0)
    gone, _ = _seg(cfg, db, datetime(2026, 10, 27, 9, 10), with_file=False)   # audio gone: cannot be scanned
    _utt(db, gone, DAY, "09:10", "z", None, None, 0.0, 2.0)
    pending, _ = _seg(cfg, db, datetime(2026, 10, 27, 9, 15), status="new")  # not processed yet
    _utt(db, pending, DAY, "09:15", "w", None, None, 0.0, 2.0)
    tiny, _ = _seg(cfg, db, datetime(2026, 10, 27, 9, 20))                     # only a 0.3 s unknown utterance: nothing to embed
    _utt(db, tiny, DAY, "09:20", "v", None, None, 0.0, 0.3)
    scanned, _ = _seg(cfg, db, datetime(2026, 10, 27, 9, 25))                  # already has cluster_embeddings
    _utt(db, scanned, DAY, "09:25", "u", None, None, 0.0, 2.0)
    _store(db, scanned, 0, _voice(1), 2.0)
    out = group_unknown_voices(db, cfg)
    assert out["missing_segments"] == 1 and out["unknown_utterances"] == 5
    assert [r["id"] for r in voices.missing_segments(db)] == [done_unknown]
    assert len(out["groups"]) == 1 and out["groups"][0]["members"] == [[scanned, 0]]


def test_group_unknown_voices_keeps_the_majority_embedding_dimension(cfg, db):
    s1, _ = _seg(cfg, db, datetime(2026, 10, 27, 9, 0))
    for c in range(3):
        _store(db, s1, c, _voice(c, dim=192), 1.0)
        _utt(db, s1, DAY, f"09:0{c}", "t", None, None, float(c), float(c) + 1, cluster=c)
    _store(db, s1, 3, _voice(9, dim=16), 1.0)                                  # an old model's row
    _utt(db, s1, DAY, "09:05", "old", None, None, 5.0, 6.0, cluster=3)
    out = group_unknown_voices(db, cfg, threshold=0.99)
    assert sum(len(g["members"]) for g in out["groups"]) == 3


# --- backfill + name_voice with the real engine ---------------------------------------------------------------------

@needs_sample
def test_backfill_group_and_name_voice_end_to_end(cfg, db, speaker_engine):
    seg, p = _seg(cfg, db, datetime(2026, 10, 30, 9, 0), audio=_sample_slice(29, 37))
    u1 = _utt(db, seg, DAY, "09:00", "大家好我是雷军", None, None, 0.0, 4.0, cluster=0)
    u2 = _utt(db, seg, DAY, "09:00", "今天发布新产品", None, None, 4.0, 8.0, cluster=0)
    before = group_unknown_voices(db, cfg)
    assert before["groups"] == [] and before["missing_segments"] == 1 and before["unknown_utterances"] == 2

    stats = backfill_cluster_embeddings(db, cfg, speaker_engine)
    assert stats == {"segments": 1, "clusters": 1, "skipped": 0}
    rows = db.cluster_embeddings([seg])
    assert len(rows) == 1 and rows[0][1] == 0 and rows[0][2].shape == (192,) and rows[0][3] == pytest.approx(8.0)
    assert np.linalg.norm(rows[0][2]) == pytest.approx(1.0, abs=1e-3)
    assert backfill_cluster_embeddings(db, cfg, speaker_engine) == {"segments": 0, "clusters": 0, "skipped": 0}   # nothing left

    out = group_unknown_voices(db, cfg)
    assert out["missing_segments"] == 0 and len(out["groups"]) == 1
    g = out["groups"][0]
    assert g["members"] == [[seg, 0]] and g["utterances"] == 2 and g["speech_s"] == 8.0 and g["days"] == [DAY]
    assert {s["utterance_id"] for s in g["samples"]} == {u1, u2} and all(s["audio"] for s in g["samples"])

    res = name_voice(db, cfg, speaker_engine, g["members"], " 雷军 ", is_self=True, relation="导师")
    assert res["name"] == "雷军" and res["samples_added"] == 1 and res["utterances_assigned"] == 2
    assert res["segments"] == [seg] and res["days"] == [DAY]
    sid = res["id"]
    emb = db.query("SELECT speaker_id, source_path, duration_s FROM speaker_embeddings")
    assert len(emb) == 1 and emb[0]["speaker_id"] == sid and 1.0 <= emb[0]["duration_s"] <= 8.0
    # the sample points at the speech-only copy under data/speakers (playable even after the segment expires)
    assert Path(emb[0]["source_path"]).exists() and Path(emb[0]["source_path"]).suffix == ".wav" and "#" not in emb[0]["source_path"]
    utts = db.query("SELECT speaker_id, speaker_name, speaker_score, speaker_source FROM utterances ORDER BY id")
    assert all((u["speaker_id"], u["speaker_name"], u["speaker_score"], u["speaker_source"]) == (sid, "雷军", None, "user") for u in utts)
    me = db.self_speaker()
    assert me["id"] == sid and me["relation"] == "导师" and me["relation_source"] == "user"
    md = (cfg.transcripts_dir / f"{DAY}.md").read_text(encoding="utf-8")
    assert "雷军" in md and "未知" not in md
    after = group_unknown_voices(db, cfg)
    assert after["groups"] == [] and after["unknown_utterances"] == 0

    # naming again with an existing speaker (no new sample needed) and no self / relation keeps the flags
    seg2, _ = _seg(cfg, db, datetime(2026, 10, 30, 10, 0), audio=_sample_slice(30, 36))
    _utt(db, seg2, DAY, "10:00", "再来一段", None, None, 0.0, 6.0, cluster=0)
    backfill_cluster_embeddings(db, cfg, speaker_engine)
    g2 = group_unknown_voices(db, cfg)["groups"][0]
    assert g2["closest"]["id"] == sid and g2["closest"]["score"] > 0.3
    res2 = name_voice(db, cfg, speaker_engine, g2["members"], "雷军", max_samples=0)
    assert res2["id"] == sid and res2["samples_added"] == 0 and res2["utterances_assigned"] == 1
    assert db.self_speaker()["id"] == sid and db.one("SELECT relation FROM speakers WHERE id=?", (sid,))["relation"] == "导师"


@needs_sample
def test_name_voice_too_short_group_leaves_no_empty_speaker(cfg, db, speaker_engine):
    seg, _ = _seg(cfg, db, datetime(2026, 10, 30, 9, 0), audio=_tone(0.8))        # no speech at all, 0.8 s
    uid = _utt(db, seg, DAY, "09:00", "嗯", None, None, 0.0, 0.8, cluster=0)
    assert backfill_cluster_embeddings(db, cfg, speaker_engine)["clusters"] == 1
    with pytest.raises(ValueError, match="语音太短"):
        name_voice(db, cfg, speaker_engine, [[seg, 0]], "新人")
    assert db.one("SELECT COUNT(*) AS n FROM speakers")["n"] == 0 and db.speaker_embeddings() == []
    assert db.one("SELECT speaker_id, speaker_source FROM utterances WHERE id=?", (uid,))["speaker_id"] is None
    # an already enrolled speaker is not deleted: the utterances are assigned without a new sample
    old = db.upsert_speaker("老张")
    db.add_speaker_embedding(old, _voice(5), None, 3.0)
    res = name_voice(db, cfg, speaker_engine, [[seg, 0]], "老张")
    assert res["id"] == old and res["samples_added"] == 0 and res["utterances_assigned"] == 1
    assert db.one("SELECT speaker_name, speaker_source FROM utterances WHERE id=?", (uid,))["speaker_source"] == "user"


def test_name_voice_validates_input(cfg, db):
    engine = object()
    with pytest.raises(ValueError, match="姓名"):
        name_voice(db, cfg, engine, [[1, 0]], "   ")
    with pytest.raises(ValueError, match="members"):
        name_voice(db, cfg, engine, [], "甲")
    with pytest.raises(ValueError, match="members"):
        name_voice(db, cfg, engine, [["x", 0]], "甲")
    assert db.one("SELECT COUNT(*) AS n FROM speakers")["n"] == 0


def test_backfill_respects_stop_event_limit_and_skips_unreadable_files(cfg, db):
    class Engine:
        def __init__(self):
            self.calls = 0

        def cluster_embedding(self, audio, turns):
            self.calls += 1
            return _voice(self.calls)

    s1, p1 = _seg(cfg, db, datetime(2026, 10, 27, 9, 0))
    _utt(db, s1, DAY, "09:00", "a", None, None, 0.0, 2.0, cluster=0)
    _utt(db, s1, DAY, "09:00", "b", None, None, 2.0, 3.0, cluster=1)
    s2, p2 = _seg(cfg, db, datetime(2026, 10, 27, 9, 5))
    _utt(db, s2, DAY, "09:05", "c", None, None, 0.0, 2.0, cluster=None)        # NULL cluster -> index-based turn
    s3, p3 = _seg(cfg, db, datetime(2026, 10, 27, 9, 10))
    _utt(db, s3, DAY, "09:10", "d", None, None, 0.0, 2.0, cluster=0)
    p3.write_bytes(b"not a wav file")

    class Stop:
        def __init__(self, after: int):
            self.after, self.n = after, 0

        def is_set(self) -> bool:
            self.n += 1
            return self.n > self.after

    eng = Engine()
    assert backfill_cluster_embeddings(db, cfg, eng, stop_event=Stop(1)) == {"segments": 1, "clusters": 2, "skipped": 0}
    assert [(r[1], r[3]) for r in db.cluster_embeddings([s1])] == [(0, 2.0), (1, 1.0)]
    assert backfill_cluster_embeddings(db, cfg, eng, limit=1) == {"segments": 1, "clusters": 1, "skipped": 0}
    assert db.cluster_embeddings([s2])[0][1] == 0
    assert backfill_cluster_embeddings(db, cfg, eng) == {"segments": 0, "clusters": 0, "skipped": 1}
    assert db.cluster_embeddings([s3]) == [] and eng.calls == 3


# --- relabel_segment keeps user labels ---------------------------------------------------------------------------

class FakeEngine:
    """identify_turns with the new segment_id keyword: labels every cluster per `names` and stores an embedding."""

    def __init__(self, names: dict[int, tuple[int | None, str | None, float | None]]):
        self.names = names
        self.calls: list[int | None] = []

    def identify_turns(self, audio, turns, db, segment_id=None):
        self.calls.append(segment_id)
        for t in turns:
            t.speaker_id, t.speaker_name, t.score = self.names.get(t.cluster, (None, None, None))
            if segment_id is not None:
                db.upsert_cluster_embedding(segment_id, t.cluster, _voice(t.cluster), t.duration)
        return turns


def test_relabel_segment_skips_user_labelled_utterances_and_counts_real_changes(cfg, db):
    seg, _ = _seg(cfg, db, datetime(2026, 10, 30, 9, 0), audio=_tone(6.0))
    zhang = db.upsert_speaker("张三")
    li = db.upsert_speaker("李四")
    hand = _utt(db, seg, DAY, "09:00", "手标的", "张三", zhang, 0.0, 1.0, cluster=0, source="user")
    auto = _utt(db, seg, DAY, "09:00", "自动的", "未知", None, 1.0, 2.0, cluster=0)
    same = _utt(db, seg, DAY, "09:00", "没变的", "李四", li, 2.0, 3.0, 0.5, cluster=1)
    other = _utt(db, seg, DAY, "09:00", "另一簇", "未知", None, 3.0, 4.0, cluster=2)
    engine = FakeEngine({0: (li, "李四", 0.9), 1: (li, "李四", 0.7), 2: (None, None, 0.2)})
    proc = Processor(cfg, db, engine=engine)
    assert proc.relabel_segment(seg) == 1                                       # only `auto` really changed
    assert engine.calls == [seg]
    rows = {r["id"]: r for r in db.query("SELECT * FROM utterances")}
    assert (rows[hand]["speaker_id"], rows[hand]["speaker_name"], rows[hand]["speaker_source"]) == (zhang, "张三", "user")
    assert (rows[auto]["speaker_id"], rows[auto]["speaker_name"], rows[auto]["speaker_score"]) == (li, "李四", pytest.approx(0.9))
    assert rows[same]["speaker_id"] == li and rows[same]["speaker_score"] == pytest.approx(0.7)   # score refreshed, not counted
    assert rows[other]["speaker_id"] is None and rows[other]["speaker_name"] == "未知"
    assert {r[1] for r in db.cluster_embeddings([seg])} == {0, 1, 2}             # embeddings refreshed on the way
    assert "李四" in (cfg.transcripts_dir / f"{DAY}.md").read_text(encoding="utf-8")
    assert proc.relabel_segment(seg) == 0
    # an engine without the segment_id keyword still works (older fakes / implementations)

    class OldEngine:
        def identify_turns(self, audio, turns, db):
            for t in turns:
                t.speaker_id, t.speaker_name, t.score = None, None, None
            return turns

    assert Processor(cfg, db, engine=OldEngine()).relabel_segment(seg) == 2       # auto + same back to unknown; hand untouched
    assert db.one("SELECT speaker_name FROM utterances WHERE id=?", (hand,))["speaker_name"] == "张三"


def test_process_segment_stores_cluster_embeddings_via_segment_id(cfg, db):
    from audio_note.speech.asr import ASRResult

    seg, _ = _seg(cfg, db, datetime(2026, 10, 30, 9, 0), audio=_tone(4.0), status="new")

    class Diar(FakeEngine):
        def diarize(self, audio):
            return [Turn(0.0, 2.0, 0), Turn(2.0, 4.0, 1)]

    class ASR:
        def transcribe(self, audio):
            return ASRResult("你好", 2.0, 1)

    engine = Diar({0: (None, None, None), 1: (None, None, None)})
    db.upsert_cluster_embedding(seg, 7, _voice(7), 1.0)                            # stale row from an earlier attempt
    p = Processor(cfg, db, client=object(), engine=engine, asr=ASR(), corrector=None)
    p._corrector_tried = True
    assert p.process_segment(seg) == 2
    assert engine.calls == [seg] and {r[1] for r in db.cluster_embeddings([seg])} == {0, 1}
    assert group_unknown_voices(db, cfg)["unknown_utterances"] == 2


# --- review fixes: hand labels vs. relabel races, naming over hand labels, expired audio, scan sentinel -------------

def test_relabel_treats_user_rows_without_speaker_as_automatic_and_skips_rows_labelled_mid_run(cfg, db):
    """Finding 1 + 2: a 'user' row whose speaker was deleted (speaker_id NULL) is re-identified again; a row
    hand-labelled while the engine is embedding is left alone and not counted as changed."""
    seg, _ = _seg(cfg, db, datetime(2026, 10, 30, 9, 0), audio=_tone(6.0))
    li = db.upsert_speaker("李四")
    wang = db.upsert_speaker("王五")
    orphan = _utt(db, seg, DAY, "09:00", "删掉的人说的", "未知", None, 0.0, 1.0, cluster=0, source="user")   # left by a speaker delete
    racing = _utt(db, seg, DAY, "09:00", "正在被手标", "未知", None, 1.0, 2.0, cluster=0)
    plain = _utt(db, seg, DAY, "09:00", "普通的", "未知", None, 2.0, 3.0, cluster=0)

    class RacingEngine(FakeEngine):
        def identify_turns(self, audio, turns, db_, segment_id=None):
            # a request thread labels `racing` to 王五 while this job is "embedding"
            db_.execute("UPDATE utterances SET speaker_id=?, speaker_name='王五', speaker_source='user' WHERE id=?", (wang, racing))
            return super().identify_turns(audio, turns, db_, segment_id)

    proc = Processor(cfg, db, engine=RacingEngine({0: (li, "李四", 0.9)}))
    assert proc.relabel_segment(seg) == 2                                       # orphan + plain; racing skipped
    rows = {r["id"]: r for r in db.query("SELECT * FROM utterances")}
    assert (rows[orphan]["speaker_id"], rows[orphan]["speaker_name"], rows[orphan]["speaker_source"]) == (li, "李四", None)
    assert (rows[plain]["speaker_id"], rows[plain]["speaker_source"]) == (li, None)
    assert (rows[racing]["speaker_id"], rows[racing]["speaker_name"], rows[racing]["speaker_source"]) == (wang, "王五", "user")
    assert proc.relabel_segment(seg) == 0                                       # racing stays hand-labelled


def test_name_voice_keeps_rows_hand_labelled_as_somebody_else(cfg, db):
    """Finding 3: naming a group never overwrites utterances the user labelled to a different speaker;
    rows labelled to the same speaker, automatic rows and orphaned 'user' rows are assigned."""
    class Engine:
        def enroll_from_audio(self, db_, name, audio, source=None, copy_as_source=False):
            sid = db_.upsert_speaker(name)
            db_.add_speaker_embedding(sid, _voice(1), source, 2.0)
            return sid

    seg, _ = _seg(cfg, db, datetime(2026, 10, 30, 9, 0), audio=_tone(6.0))
    zhang = db.upsert_speaker("张三")
    wang = db.upsert_speaker("王五")
    _store(db, seg, 0, _voice(1), 4.0)
    free = _utt(db, seg, DAY, "09:00", "没标的", None, None, 0.0, 1.0, cluster=0)
    theirs = _utt(db, seg, DAY, "09:00", "手标给张三的", "张三", zhang, 1.0, 2.0, cluster=0, source="user")
    mine = _utt(db, seg, DAY, "09:00", "手标给王五的", "王五", wang, 2.0, 3.0, cluster=0, source="user")
    auto = _utt(db, seg, DAY, "09:00", "自动认成张三的", "张三", zhang, 3.0, 4.0, 0.7, cluster=0)
    orphan = _utt(db, seg, DAY, "09:00", "删掉的人", "未知", None, 4.0, 5.0, cluster=0, source="user")
    res = name_voice(db, cfg, Engine(), [[seg, 0]], "王五")
    assert res["id"] == wang and res["utterances_assigned"] == 4 and res["skipped_user_labels"] == 1
    assert res["segments"] == [seg] and res["days"] == [DAY]
    rows = {r["id"]: r for r in db.query("SELECT * FROM utterances")}
    for uid in (free, mine, auto, orphan):
        assert (rows[uid]["speaker_id"], rows[uid]["speaker_name"], rows[uid]["speaker_score"], rows[uid]["speaker_source"]) == (wang, "王五", None, "user")
    assert (rows[theirs]["speaker_id"], rows[theirs]["speaker_name"], rows[theirs]["speaker_source"]) == (zhang, "张三", "user")


def test_name_voice_enrols_stored_cluster_embeddings_when_the_audio_expired(cfg, db):
    """Finding 4: a group whose recordings are gone is named from the cluster embeddings the scan stored;
    with nothing stored either the error says so (and leaves no empty speaker)."""
    class Engine:
        calls = 0

        def enroll_from_audio(self, *a, **k):
            Engine.calls += 1
            raise AssertionError("there is no audio to enrol from")

    a = _voice(41)
    s1, _ = _seg(cfg, db, datetime(2026, 10, 27, 9, 0), with_file=False)
    s2, _ = _seg(cfg, db, datetime(2026, 10, 28, 9, 0), with_file=False)
    _store(db, s1, 0, _near(a, 1), 6.0)
    _store(db, s2, 0, _near(a, 2), 3.0)
    _utt(db, s1, "2026-10-27", "09:00", "a", None, None, 0.0, 6.0, cluster=0)
    _utt(db, s2, "2026-10-28", "09:00", "b", None, None, 0.0, 3.0, cluster=0)
    res = name_voice(db, cfg, Engine(), [[s2, 0], [s1, 0]], "老张", max_samples=1)
    assert Engine.calls == 0
    assert res["samples_added"] == 1 and res["samples_from_clusters"] == 1 and res["utterances_assigned"] == 2
    emb = db.query("SELECT speaker_id, source_path, duration_s, dim FROM speaker_embeddings")
    assert len(emb) == 1 and tuple(emb[0]) == (res["id"], None, 6.0, 192)             # the member with the most speech
    np.testing.assert_allclose(db.speaker_embeddings()[0][2], _near(a, 1), atol=1e-6)
    assert all(r["speaker_source"] == "user" for r in db.query("SELECT speaker_source FROM utterances"))
    assert group_unknown_voices(db, cfg)["groups"] == []
    # audio gone and nothing stored: a clear message, no empty speaker row, nothing assigned
    s3, _ = _seg(cfg, db, datetime(2026, 10, 29, 9, 0), with_file=False)
    u3 = _utt(db, s3, "2026-10-29", "09:00", "c", None, None, 0.0, 2.0, cluster=0)
    with pytest.raises(ValueError, match="录音已过期"):
        name_voice(db, cfg, Engine(), [[s3, 0]], "新人")
    assert db.one("SELECT COUNT(*) AS n FROM speakers WHERE name='新人'")["n"] == 0
    assert db.one("SELECT speaker_id FROM utterances WHERE id=?", (u3,))["speaker_id"] is None
    # an already enrolled speaker is assigned such a member without needing a sample
    res2 = name_voice(db, cfg, Engine(), [[s3, 0]], "老张")
    assert res2["id"] == res["id"] and res2["samples_added"] == 0 and res2["utterances_assigned"] == 1


def test_backfill_marks_unreadable_or_empty_segments_so_they_are_not_rescanned(cfg, db):
    """Finding 5: a scanned segment that yields no embedding gets a sentinel row, so missing_segments
    reaches 0 (no permanent scan banner / re-scan every session) and grouping ignores it."""
    class Engine:
        def __init__(self):
            self.calls = 0

        def cluster_embedding(self, audio, turns):
            self.calls += 1
            return None                                      # e.g. the utterance ranges fall outside the audio

    s1, p1 = _seg(cfg, db, datetime(2026, 10, 27, 9, 0))
    _utt(db, s1, DAY, "09:00", "a", None, None, 0.0, 2.0, cluster=0)
    p1.write_bytes(b"not a wav file")
    s2, _ = _seg(cfg, db, datetime(2026, 10, 27, 9, 5))
    _utt(db, s2, DAY, "09:05", "b", None, None, 0.0, 2.0, cluster=0)
    eng = Engine()
    assert len(voices.missing_segments(db)) == 2
    assert backfill_cluster_embeddings(db, cfg, eng) == {"segments": 0, "clusters": 0, "skipped": 2} and eng.calls == 1
    assert voices.missing_segments(db) == [] and db.cluster_embeddings() == []
    assert [tuple(r) for r in db.query("SELECT segment_id, cluster_label, dim, speech_s, length(embedding) FROM cluster_embeddings"
                                       " ORDER BY segment_id")] == [(s1, -1, 0, 0.0, 0), (s2, -1, 0, 0.0, 0)]
    out = group_unknown_voices(db, cfg)
    assert out["missing_segments"] == 0 and out["groups"] == [] and out["unknown_utterances"] == 2
    assert backfill_cluster_embeddings(db, cfg, eng) == {"segments": 0, "clusters": 0, "skipped": 0} and eng.calls == 1
    # a sentinel is never a nameable member; reprocessing clears it the usual way and the segment is scanned again
    with pytest.raises(ValueError, match="members"):
        name_voice(db, cfg, eng, [[s2, -1]], "甲")
    assert db.delete_cluster_embeddings(s2) == 1
    assert [r["id"] for r in voices.missing_segments(db)] == [s2]
