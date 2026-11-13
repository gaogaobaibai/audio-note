"""scripts/bench/bench_speakers.py: the scorers that replay the 1.0 scoring layer (v5_*, v5_first, v5_cascade, v5_db).

Synthetic trials only: random unit vectors with exact cosines, no model, no audio, no bench root.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "scripts" / "bench"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import bench_speakers as spk  # noqa: E402

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


def _mix(base: np.ndarray, like: np.ndarray, seed: int, cos_base: float, along: float) -> np.ndarray:
    """Cosine `cos_base` to `base`, plus `along` of the part of `like` that is not `base` (the same voice as `like`)."""
    u = like - float(like @ base) * base
    u = u / np.linalg.norm(u)
    w = np.random.default_rng(seed).standard_normal(base.shape[0])
    w = w - float(w @ base) * base - float(w @ u) * u
    w = w / np.linalg.norm(w)
    return _unit(cos_base * base + along * u + math.sqrt(max(0.0, 1.0 - cos_base ** 2 - along ** 2)) * w)


A, B = _voice(1), _voice(2)


def _trial(session: str, line: int, crop: float, emb: np.ndarray, speaker: str, t: float, dur: float, cond: str = "near",
           loud: float = -18.0) -> dict:
    return {"id": f"{session}:{line}:{crop:g}", "emb": emb, "speech_s": dur, "dur": dur, "speaker": speaker, "condition": cond,
            "mixed": False, "session": session, "segment": session, "t": t, "loudness_db": loud, "system_share": None, "crop": crop}


def make() -> tuple[list[dict], spk.Gallery]:
    """Two sessions: 甲 near (strong), 甲 far (weak on its own), a stranger; each full line plus a 1 s crop."""
    wild = _at(A, 5, 0.8)
    trials: list[dict] = []
    for s, base in (("d1s1", 100), ("d1s2", 200)):
        t = 0.0
        for k in range(6):
            trials.append(_trial(s, k, 0.0, _at(wild, base + k, 0.93), "甲", t, 4.0))
            trials.append(_trial(s, k, 1.0, _at(wild, base + 50 + k, 0.85), "甲", t, 1.0))
            t += 5.0
        for k in range(6, 10):
            # far from the microphone: only 0.40 to the enrolled voice, but still the voice of the lines above
            trials.append(_trial(s, k, 0.0, _mix(A, wild, base + k, 0.40, 0.6), "甲", t, 4.0, cond="far", loud=-38.0))
            t += 5.0
        for k in range(10, 28):
            trials.append(_trial(s, k, 0.0, _voice(base * 10 + k), "路人", t, 4.0))
            t += 5.0
    gallery = spk.Gallery({1: "甲", 2: "乙"}, [(1, A, "file", 10.0), (2, B, "file", 10.0)], [], 1, None)
    return trials, gallery


def test_trial_key_reads_line_trials_and_anything_else():
    assert spk._trial_key({"id": "d1s1:7:2"}) == ("d1s1", "7", 2.0)
    assert spk._trial_key({"id": 42, "session": "s"}) == ("s", "42", 0.0)
    assert spk._trial_key({"id": "12:3", "segment": 9}) == ("9", "12:3", 0.0)
    assert spk._trial_key({"id": "a:b:c"}) == ("", "a:b:c", 0.0)


def test_v5_scorers_are_registered_and_follow_the_scorer_contract():
    trials, gallery = make()
    for name in ("v5_centroid", "v5_adapt", "v5_norm", "v5_duration", "v5", "v5_first", "v5_cascade", "v5_db"):
        assert name in spk.SCORERS and spk.load_scorer(name)[0] == name
    res = spk.replay(trials, gallery, [spk.load_scorer(n) for n in ("legacy", "v5_centroid", "v5", "v5_first")], ["甲", "乙"])
    assert set(res) == {"legacy", "v5_centroid", "v5", "v5_first"}
    for name, ev in res.items():
        assert ev["overall"]["n"] == len(trials) and ev["overall"]["far"] == 0.0, name
        assert ev["by_condition"]["far"]["n"] == 8
    # the far lines are out of reach for the enrolled voice alone and recognised once the centroids are learned
    assert res["v5_centroid"]["by_condition"]["far"]["frr"] == 1.0
    assert res["v5"]["by_condition"]["far"]["frr"] == 0.0 and res["v5"]["overall"]["accuracy"] == 1.0
    assert "centroid" in res["v5"]["methods"]
    # a first pass knows less than a rescore of everything: never better on the far lines
    assert res["v5_first"]["by_condition"]["far"]["frr"] >= res["v5"]["by_condition"]["far"]["frr"]
    # "maybe" decisions are reported and never counted as an identification
    ds = spk.SCORERS["v5_centroid"]([{k: v for k, v in t.items() if k != "speaker"} for t in trials], gallery)
    assert len(ds) == len(trials) and all(set(d) >= {"name", "best", "score", "guess", "method"} for d in ds)
    maybe = [d for d in ds if d["name"] is None and d["guess"] is not None]
    assert maybe and all(d["best"] == d["guess"] for d in maybe)
    # speakers.* settings pass through as scorer parameters
    strict = spk.SCORERS["v5"](trials, gallery, sure_norm=0.9, identify_threshold=0.99)
    assert all(d["name"] is None for d in strict)


def test_cascade_asks_the_second_model_only_about_maybes():
    trials, gallery = make()
    for t in trials:
        t["emb2"] = _at(A, 7, 0.9) if t["speaker"] == "甲" else _voice(8)        # the second model hears 甲 clearly
    gallery.samples2 = [(1, A, "file", 10.0), (2, B, "file", 10.0)]
    base = spk.SCORERS["v5_centroid"](trials, gallery)
    n_maybe = sum(1 for d in base if d["name"] is None and d["guess"] is not None)
    # (no adaptation, no cohort, no tracks: the far lines stay "maybe" for the first model)
    off = {"adapt": False, "cohort_min": 10 ** 9, "track_link_cos": 2.0}
    out = spk.scorer_v5_cascade(trials, gallery, **off)
    asked = [d for d in out if d.get("asked2")]
    assert n_maybe == 8 and len(asked) == 8 and all(d["method"] == "cascade" and d["name"] == "甲" for d in asked)
    assert all(d["name"] is None for d, t in zip(out, trials) if t["speaker"] == "路人")
    high = spk.scorer_v5_cascade(trials, gallery, cascade_min=0.95, **off)
    assert all(d["method"] != "cascade" for d in high)                           # the bar of the second model holds
    ev = spk.replay(trials, gallery, [("v5_cascade", spk.scorer_v5_cascade)], ["甲", "乙"])["v5_cascade"]
    assert "second_model" not in ev or ev["second_model"]["asked"] >= ev["second_model"]["accepted"]


def test_v5_db_scores_a_copy_and_leaves_the_database_alone(tmp_path):
    from audio_note.db import DB

    path = tmp_path / "data" / "audio_note.sqlite3"
    path.parent.mkdir()
    db = DB(path)
    jia = db.upsert_speaker("甲")
    db.add_speaker_embedding(jia, A, None, 10.0, origin="mic")
    seg = db.add_segment("s.wav", datetime(2026, 11, 4, 9), 10.0)
    db.set_segment_status(seg, "done")
    db.upsert_cluster_embedding(seg, 0, _at(A, 9, 0.8), 6.0)
    a, b = db.add_utterances([
        {"segment_id": seg, "start_s": 0.0, "end_s": 3.0, "abs_start": "2026-11-04T09:00:00", "day": "2026-11-04", "cluster_label": 0,
         "speaker_id": None, "speaker_name": "未知", "speaker_score": None, "text_raw": "话", "text": "话", "text_seg": "话"},
        {"segment_id": seg, "start_s": 3.0, "end_s": 6.0, "abs_start": "2026-11-04T09:00:03", "day": "2026-11-04", "cluster_label": 0,
         "speaker_id": None, "speaker_name": "未知", "speaker_score": None, "text_raw": "话", "text": "话", "text_seg": "话"}])
    db.execute("INSERT INTO utterance_rejections(utterance_id, speaker_id) VALUES (?,?)", (b, jia))
    db.close_all()
    gallery = spk.load_gallery(path)
    assert gallery.db_path == str(path) and gallery.names == {jia: "甲"}
    out = spk.scorer_v5_db([{"id": a}, {"id": b}, {"id": "not-an-id"}], gallery)
    assert [d["name"] for d in out] == ["甲", None, None] and out[0]["method"] == "anchor"
    assert spk.scorer_v5_db.last_summary["changed"] == 1
    again = DB(path)
    assert again.one("SELECT speaker_id FROM utterances WHERE id=?", (a,))["speaker_id"] is None      # only the copy was scored
    again.close_all()
    res = spk.proxy(path, scorers=[("v5_db", spk.scorer_v5_db), ("spec_gallery", spk.scorer_spec_gallery)])
    assert res["replay"]["v5_db"]["sure"]["n"] == 1 and res["replay"]["v5_db"]["must_not"] == {"n": 1, "violated": 0}
    assert res["replay"]["spec_gallery"]["sure"]["n"] == 2          # the pure scoring layer does not know about the veto
    with pytest.raises(RuntimeError):
        spk.scorer_v5_db([], spk.Gallery({}, [], [], None, None))
