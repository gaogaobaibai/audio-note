"""scripts/bench: the scoring logic of bench_cer / bench_speakers / bench_cost / report on small synthetic inputs
(hand-made truth, utterance dicts and throw-away sqlite files). No audio, no models, no network."""
from __future__ import annotations

import importlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parent.parent / "scripts" / "bench"
PKG = "audio_note_bench"


def bench(module: str):
    if PKG not in sys.modules:
        spec = importlib.util.spec_from_file_location(PKG, BENCH / "__init__.py", submodule_search_locations=[str(BENCH)])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[PKG] = pkg
        spec.loader.exec_module(pkg)
    return importlib.import_module(f"{PKG}.{module}")


cer = bench("bench_cer")
spk = bench("bench_speakers")
cost = bench("bench_cost")
report = bench("report")
common = bench("common")

TERMS = [{"term": "青禾", "group": "name"}, {"term": "灰度", "group": "term"}, {"term": "两千六百九十九", "group": "number"},
         {"term": "T30", "group": "name"}]


def make_truth() -> dict:
    lines = [
        {"i": 0, "session": "s1", "speaker": "我", "text": "今天过一下青禾App的灰度安排。", "t0": 0.6, "t1": 4.0, "condition": "near"},
        {"i": 1, "session": "s1", "speaker": "陈默", "text": "青禾的灰度定在周五。", "t0": 5.0, "t1": 8.0, "condition": "far"},
        {"i": 2, "session": "s1", "speaker": "旁白", "text": "朗月T30售价两千六百九十九。", "t0": 9.0, "t1": 13.0, "condition": "system"},
        {"i": 3, "session": "s1", "speaker": "老秦", "text": "那还行。", "t0": 14.0, "t1": 15.0, "condition": "near"},
    ]
    return {"variant": "mixed", "story": "t", "terms": TERMS, "unknown_label": "未知", "enrolled": ["我", "陈默"],
            "speakers": {"我": {"enrolled": True, "self": True}, "陈默": {"enrolled": True, "self": False},
                         "旁白": {"enrolled": False, "self": False}, "老秦": {"enrolled": False, "self": False}},
            "sessions": [{"id": "s1", "day": "2026-11-02", "start": "2026-11-02T09:30:00", "kind": "meeting", "file": "s1.wav",
                          "duration_s": 16.0, "lines": lines}],
            "lines": lines}


def utt(uid, u0, u1, raw, text=None, name="未知", sid=None, **kw):
    return {"id": uid, "u0": u0, "u1": u1, "text_raw": raw, "text": raw if text is None else text, "speaker_name": name,
            "speaker_id": sid, "segment_id": 1, "cluster_label": uid, **kw}


# --- bench_cer ---------------------------------------------------------------------------------------
def test_cer_perfect_transcript_scores_zero():
    truth = make_truth()
    utts = {"s1": [utt(k + 1, ln["t0"], ln["t1"], ln["text"]) for k, ln in enumerate(truth["lines"])]}
    res, per = cer.score(truth, utts)
    assert res["overall"]["raw"]["cer"] == 0.0 and res["overall"]["corrected"]["cer"] == 0.0
    assert res["overall"]["missed_lines"] == 0 and res["overall"]["covered_lines"] == 4
    assert res["terms"]["overall"] == {"n": 6, "wrong_raw": 0, "wrong_corrected": 0, "rate_raw": 0.0, "rate_corrected": 0.0}
    assert res["edits"]["regions"] == 0 and res["stray_utterances"] == 0
    assert set(res["by_condition"]) == {"near", "far", "system"}
    assert cer.headline(res)["cer_raw"] == 0.0


def test_cer_raw_vs_corrected_terms_and_edit_verdicts():
    truth = make_truth()
    utts = {"s1": [
        # helpful: the corrector turns the mis-heard project name back into 青禾
        utt(1, 0.5, 4.1, "今天过一下清河App的灰度安排。", "今天过一下青禾App的灰度安排。"),
        # harmful: a correct word is "corrected" into a wrong one; 灰度 was mis-heard and stays wrong
        utt(2, 5.0, 8.0, "青禾的灰渡定在周五。", "青禾的灰渡定在周日。"),
        # cosmetic: only punctuation and numeral style change
        utt(3, 9.0, 13.0, "朗月T30售价2699", "朗月T30，售价两千六百九十九。"),
        # neutral: one wrong character becomes another wrong character
        utt(4, 14.0, 15.0, "那还型。", "那还形。"),
    ]}
    res, per = cer.score(truth, utts)
    n = res["overall"]["raw"]["n"]
    assert res["overall"]["raw"]["errors"] == 4            # 清河(2) 灰渡(1) 型(1)
    assert res["overall"]["corrected"]["errors"] == 3      # 渡 + 日 + 形
    assert res["overall"]["raw"]["cer"] == pytest.approx(4 / n, abs=1e-4)
    assert res["by_condition"]["near"]["raw"]["errors"] == 3 and res["by_condition"]["near"]["corrected"]["errors"] == 1
    assert res["by_condition"]["far"]["raw"]["errors"] == 1 and res["by_condition"]["far"]["corrected"]["errors"] == 2
    assert res["by_condition"]["system"]["raw"]["errors"] == 0
    t = res["terms"]
    assert t["by_term"]["青禾"] == {"n": 2, "wrong_raw": 1, "wrong_corrected": 0, "rate_raw": 0.5, "rate_corrected": 0.0}
    assert t["by_term"]["灰度"]["wrong_raw"] == 1 and t["by_term"]["灰度"]["wrong_corrected"] == 1
    assert t["by_term"]["两千六百九十九"]["wrong_raw"] == 0          # "2699" == "两千六百九十九" after normalisation
    assert t["by_group"]["number"]["n"] == 1 and t["by_group"]["name"]["n"] == 3
    assert t["by_day"]["2026-11-02"]["n"] == 6
    e = res["edits"]
    assert (e["helpful"], e["harmful"], e["neutral"]) == (1, 1, 1)
    assert e["changed_utterances"] == 3 and e["cosmetic_utterances"] == 1
    assert e["chars_fixed"] == 2 and e["chars_broken"] == 1
    assert e["utterances_net"] == {"helpful": 1, "harmful": 1, "neutral": 1}
    assert e["by_condition"]["far"] == {"helpful": 0, "harmful": 1, "neutral": 0}
    edits = {x["from"]: x for s in per for x in s["edits"]}
    assert edits["清河"]["to"] == "青禾" and edits["清河"]["truth"] == "青禾" and edits["清河"]["verdict"] == "helpful"
    assert edits["5"]["to"] == "日" and edits["5"]["verdict"] == "harmful"      # 周五 is "周5" after normalisation
    assert edits["清河"]["exact"] and not edits["5"]["exact"] and e["exact"] == 1 and e["wrong_term_injections"] == 0


def test_cer_flags_terms_the_correction_made_up():
    truth = make_truth()
    # the mis-heard word becomes a lexicon term nobody said: fewer wrong characters, still wrong
    utts = {"s1": [utt(2, 5.0, 8.0, "青禾的灰度定在周无五。", "青禾的灰度定在周T30。")]}
    res, per = cer.score(truth, utts)
    e = per[0]["edits"][0]
    assert e["injected_terms"] == ["T30"] and not e["exact"]
    assert res["edits"]["wrong_term_injections"] == 1 and res["edits"]["exact"] == 0
    assert cer.headline(res)["edits_wrong_term_injections"] == 1


def test_cer_missing_lines_and_stray_utterances():
    truth = make_truth()
    utts = {"s1": [utt(1, 0.6, 4.0, truth["lines"][0]["text"]),
                   utt(2, 30.0, 31.0, "谢谢观看")]}                    # hallucination in silence
    res, per = cer.score(truth, utts)
    assert res["overall"]["missed_lines"] == 3 and res["overall"]["covered_lines"] == 1
    assert res["by_condition"]["far"]["raw"]["cer"] == pytest.approx(1.0, abs=0.01) or res["by_condition"]["far"]["raw"]["del"] > 0
    assert res["covered_lines_only"]["lines"] == 1
    assert res["stray_utterances"] == 1 and res["stray_chars"] == 4
    assert res["terms"]["covered_lines_only"]["n"] == 2                 # only the terms of the transcribed line
    # numerals off: the Arabic price no longer equals the spoken one
    res2, _ = cer.score(truth, {"s1": [utt(3, 9.0, 13.0, "朗月T30售价2699")]}, numerals=False)
    assert res2["by_condition"]["system"]["raw"]["errors"] > 0


def test_cer_no_utterances_at_all():
    res, _ = cer.score(make_truth(), {})
    assert res["overall"]["raw"]["cer"] == 1.0 and res["overall"]["missed_lines"] == 4
    assert res["terms"]["overall"]["rate_raw"] == 1.0


def _bench_db(path: Path, truth: dict, rows: list[dict], v1: bool = False) -> None:
    c = sqlite3.connect(path)
    c.executescript("""
        CREATE TABLE segments (id INTEGER PRIMARY KEY, path TEXT, started_at TEXT, duration_s REAL, origin TEXT, status TEXT);
        CREATE TABLE speakers (id INTEGER PRIMARY KEY, name TEXT, is_self INTEGER DEFAULT 0);
        CREATE TABLE speaker_embeddings (id INTEGER PRIMARY KEY, speaker_id INTEGER, embedding BLOB, dim INTEGER, origin TEXT, duration_s REAL);
        CREATE TABLE speaker_negatives (id INTEGER PRIMARY KEY, speaker_id INTEGER, embedding BLOB);
        CREATE TABLE utterance_rejections (utterance_id INTEGER, speaker_id INTEGER);
        CREATE TABLE cluster_embeddings (segment_id INTEGER, cluster_label INTEGER, embedding BLOB, dim INTEGER, speech_s REAL);
        CREATE TABLE utterances (id INTEGER PRIMARY KEY, segment_id INTEGER, start_s REAL, end_s REAL, abs_start TEXT, day TEXT,
            cluster_label INTEGER, speaker_id INTEGER, speaker_name TEXT, speaker_score REAL, text_raw TEXT, text TEXT,
            speaker_source TEXT, loudness_db REAL, system_share REAL);
    """)
    if v1:
        c.executescript("""
            CREATE TABLE inbox_items (id INTEGER PRIMARY KEY, kind TEXT, status TEXT, payload TEXT, utterance_id INTEGER);
            CREATE TABLE utterance_revisions (id INTEGER PRIMARY KEY, utterance_id INTEGER, source TEXT, undone_at TEXT);
            CREATE TABLE utterance_voice (utterance_id INTEGER PRIMARY KEY, model TEXT, dim INTEGER, embedding BLOB, speech_s REAL);
            CREATE TABLE usage_llm (day TEXT, purpose TEXT, model TEXT, calls INTEGER, errors INTEGER, prompt_tokens INTEGER,
                completion_tokens INTEGER, audio_s REAL, wall_s REAL);
        """)
    c.execute("INSERT INTO segments VALUES (1, 'a.opus', '2026-11-02T09:30:00', 16.0, ?, 'done')", (str(Path("x") / "sessions" / "mixed" / "s1.wav"),))
    for r in rows:
        c.execute("INSERT INTO utterances (id, segment_id, start_s, end_s, abs_start, day, cluster_label, speaker_id, speaker_name, speaker_score,"
                  " text_raw, text, speaker_source, loudness_db, system_share) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (r["id"], 1, r["u0"], r["u1"], f"2026-11-02T09:30:{r['u0']:06.3f}", "2026-11-02", r.get("cluster", r["id"]), r.get("sid"),
                   r.get("name", "未知"), r.get("score"), r["raw"], r.get("text", r["raw"]), r.get("source"), r.get("loud"), r.get("share")))
    c.commit()
    c.close()


def test_cer_run_on_a_profile_directory_with_v1_tables(tmp_path):
    truth = make_truth()
    pdir = tmp_path / "profiles" / "p"
    (pdir / "data").mkdir(parents=True)
    common.write_json(pdir / "truth.json", truth)
    rows = [{"id": 1, "u0": 0.6, "u1": 4.0, "raw": "今天过一下清河App的灰度安排。"},
            {"id": 2, "u0": 5.0, "u1": 8.0, "raw": "青禾的灰度定在周五。"}]
    db = pdir / "data" / "audio_note.sqlite3"
    _bench_db(db, truth, rows, v1=True)
    c = sqlite3.connect(db)
    c.execute("INSERT INTO inbox_items VALUES (1, 'term', 'open', ?, 1)", (json.dumps({"span": "清河", "utterance_id": 1}, ensure_ascii=False),))
    c.execute("INSERT INTO inbox_items VALUES (2, 'term', 'open', ?, 2)", (json.dumps({"span": "灰度", "utterance_id": 2}, ensure_ascii=False),))
    c.execute("INSERT INTO inbox_items VALUES (3, 'voice', 'open', '{}', NULL)")
    c.execute("INSERT INTO utterance_revisions VALUES (1, 1, 'lexicon', NULL)")
    c.execute("INSERT INTO usage_llm VALUES ('2026-11-02', 'refine', 'm', 3, 0, 100, 50, 0, 4.2)")
    c.commit()
    c.close()
    res = cer.run(pdir, label="p")
    assert res["kind"] == "cer" and res["label"] == "p" and res["utterances"] == 2
    assert res["by_condition"]["near"]["raw"]["errors"] >= 2
    q = res["questions"]
    assert q["term_questions"] == 2 and q["located"] == 2 and q["pointing_at_errors"] == 1 and q["hit_rate"] == 0.5
    oracle = {o["span"]: o for o in q["oracle"]}
    assert oracle["清河"]["truth"] == "青禾" and oracle["清河"]["really_wrong"] and not oracle["灰度"]["really_wrong"]
    assert res["extras"]["revisions_by_source"] == {"lexicon": 1}
    assert res["extras"]["usage_llm"][0]["purpose"] == "refine" and res["extras"]["usage_llm"][0]["calls"] == 3
    text = cer.format_text(res, {"label": "old", "headline": {**res["headline"], "cer_raw": 0.5}})
    assert "CER raw" in text and "compared with [old]" in text
    cmp_ = cer.compare(res, {"label": "old", "headline": {"cer_raw": 0.5}})
    assert cmp_["headline"]["cer_raw"]["delta"] == pytest.approx(res["headline"]["cer_raw"] - 0.5, abs=1e-4)
    # the pre-1.0 schema simply has no questions
    db2 = tmp_path / "old.sqlite3"
    _bench_db(db2, truth, rows)
    assert cer.question_stats(db2, []) is None and cer.db_extras(db2) == {}
    # utterances are matched to the session by the imported file's name
    assert cer.session_of_segment(r"C:\bench\sessions\mixed\s1.wav", truth["sessions"]) == "s1"
    assert cer.session_of_segment("/x/other.wav", truth["sessions"]) is None


# --- bench_speakers: tallies ---------------------------------------------------------------------------
def test_tally_rates():
    t = spk.Tally()
    for expected, got, sec in [("我", "我", 4.0), ("我", None, 1.0), ("我", "陈默", 1.0), (None, None, 2.0), (None, "我", 2.0)]:
        t.add(expected, got, sec)
    r = t.result()
    assert r["n"] == 5 and r["target"] == 3 and r["nontarget"] == 2
    assert r["accuracy"] == 0.4 and r["frr"] == pytest.approx(2 / 3, abs=1e-4)
    assert r["reject_rate"] == pytest.approx(1 / 3, abs=1e-4) and r["confusion_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert r["far"] == 0.5 and r["unknown_rejection"] == 0.5
    assert r["by_time"]["accuracy"] == 0.6 and r["by_time"]["frr"] == pytest.approx(2 / 6, abs=1e-4)
    assert spk.Tally().result()["accuracy"] is None
    assert [spk.bucket_of(x) for x in (0.4, 1.0, 3.9, 4.0, 8.0, 100.0, None)] == ["<1s", "1-2s", "2-4s", "4-8s", ">=8s", ">=8s", "<1s"]


def test_utterance_trials_and_evaluate():
    truth = make_truth()
    utts = {"s1": [
        utt(1, 0.6, 4.0, "a", name="我", sid=1),
        utt(2, 5.0, 8.0, "b", name="未知"),                # far: the enrolled speaker is rejected
        utt(3, 9.0, 13.0, "c", name="我", sid=1),          # narrator accepted as the owner: false accept
        utt(4, 14.0, 15.0, "d", name="未知"),              # the friend is rejected: right
        utt(5, 3.0, 6.5, "e", name="陈默", sid=2),         # straddles two speakers: mixed
        utt(6, 40.0, 41.0, "f", name="未知"),              # outside every line
    ]}
    trials, facts = spk.utterance_trials(truth, utts)
    assert facts["stray_utterances"] == 1 and len(trials) == 5
    assert facts["line_coverage"]["near"]["lines"] == 2 and facts["line_coverage"]["far"]["missed_lines"] == 0
    assert [t["speaker"] for t in trials] == ["我", "陈默", "旁白", "老秦", "陈默"]
    assert [t["mixed"] for t in trials] == [False, False, False, False, True]
    decided = [None if (t["stored_id"] is None) else t["stored_name"] for t in trials]
    ev = spk.evaluate(trials, decided, truth["enrolled"])
    o = ev["overall"]
    assert o["n"] == 5 and o["target"] == 3 and o["correct"] == 2 and o["rejected"] == 1 and o["accepted"] == 1
    assert o["accuracy"] == 0.6 and o["frr"] == pytest.approx(1 / 3, abs=1e-4) and o["far"] == 0.5
    assert ev["unmixed_only"]["n"] == 4
    assert ev["by_condition"]["far"]["n"] == 2 and ev["by_condition"]["far"]["frr"] == 0.5
    assert ev["by_condition"]["system"]["far"] == 1.0
    assert ev["per_speaker"]["我"] == {"trials": 1, "frr": 0.0, "other_trials": 4, "far": 0.25}
    assert ev["per_speaker"]["陈默"]["frr"] == 0.5 and ev["per_speaker"]["陈默"]["far"] == 0.0
    assert ev["unknown_speakers"] == {"旁白": {"trials": 1, "rejected": 0, "rejection_rate": 0.0},
                                      "老秦": {"trials": 1, "rejected": 1, "rejection_rate": 1.0}}
    assert ev["confusion"]["旁白"] == {"我": 1} and ev["confusion"]["陈默"] == {"(unknown)": 1, "陈默": 1}
    h = spk.headline(ev)
    assert h["accuracy"] == 0.6 and h["far_system"] == 1.0 and h["frr_far"] == 0.5
    assert "FRR" in spk.format_eval("t", ev)


# --- bench_speakers: replay ------------------------------------------------------------------------------
def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def make_gallery():
    return spk.Gallery({1: "我", 2: "陈默"}, [(1, unit(1, 0, 0, 0), "mic", 5.0), (1, unit(0.9, 0.1, 0, 0), "mic", 5.0),
                                           (2, unit(0, 1, 0, 0), "file", 5.0)], [], 1, None)


def make_trials():
    return [
        {"id": 1, "emb": unit(1, 0.1, 0, 0), "speech_s": 4.0, "dur": 4.0, "speaker": "我", "condition": "near", "mixed": False},
        {"id": 2, "emb": unit(0.5, 0, 0.85, 0), "speech_s": 3.0, "dur": 3.0, "speaker": "我", "condition": "far", "mixed": False},
        {"id": 3, "emb": unit(0.1, 1, 0, 0), "speech_s": 1.5, "dur": 1.5, "speaker": "陈默", "condition": "near", "mixed": False},
        {"id": 4, "emb": unit(0.6, 0, 0, 0.8), "speech_s": 5.0, "dur": 5.0, "speaker": "老秦", "condition": "near", "mixed": False},
        {"id": 5, "emb": unit(1, 0, 0.05, 0), "speech_s": 6.0, "dur": 6.0, "speaker": "旁白", "condition": "system", "mixed": False,
         "system_share": 0.95},
        {"id": 6, "emb": unit(0, 0, 0, 1), "speech_s": 2.0, "dur": 2.0, "speaker": "旁白", "condition": "system", "mixed": False,
         "system_share": 0.99},
    ]


def test_legacy_scorer_follows_the_old_rule():
    trials, g = make_trials(), make_gallery()
    d = spk.scorer_legacy(trials, g, threshold=0.45)
    assert [x["name"] for x in d] == ["我", "我", "陈默", "我", None, None]
    assert d[0]["best"] == "我" and d[0]["score"] > 0.99
    assert d[4]["best"] == "陈默" and d[4]["name"] is None       # system audio is never the user: next candidate, below threshold
    assert d[3]["score"] == pytest.approx(0.6, abs=1e-3)           # the friend resembles the owner: a false accept at 0.45
    assert [x["name"] for x in spk.scorer_legacy(trials, g, threshold=0.65)] == ["我", None, "陈默", None, None, None]
    # without the system rule the narrator who sounds like the owner is accepted
    assert spk.scorer_legacy(trials, g, threshold=0.45, system_turn_share=None)[4]["name"] == "我"
    # a negative sample of the owner close to the friend's voice vetoes that match
    g2 = make_gallery()
    g2.negatives.append((1, unit(0.6, 0, 0, 0.8)))
    assert spk.scorer_legacy(trials, g2, threshold=0.45)[3]["name"] is None
    assert spk.scorer_legacy(trials, g2, threshold=0.45)[0]["name"] == "我"


def test_legacy_scorer_agrees_with_speaker_engine_match():
    from audio_note.speech.speakers import SpeakerEngine

    rng = np.random.default_rng(11)
    g = spk.Gallery({1: "a", 2: "b", 3: "c"}, [], [], None, None)
    for sid in (1, 2, 3):
        base = rng.standard_normal(16)
        for _ in range(3):
            g.samples.append((sid, unit(*(base + 0.4 * rng.standard_normal(16))), "file", 3.0))
    g.negatives = [(1, unit(*rng.standard_normal(16))), (2, unit(*rng.standard_normal(16)))]
    trials = [{"id": k, "emb": unit(*rng.standard_normal(16)), "speech_s": 2.0} for k in range(200)]
    trials += [{"id": 500 + k, "emb": unit(*(g.samples[k % 9][1] + 0.5 * rng.standard_normal(16))), "speech_s": 2.0} for k in range(200)]
    enrolled = [(sid, g.names[sid], e) for sid, e, _o, _d in g.samples]
    for thr in (0.3, 0.45, 0.6):
        mine = spk.scorer_legacy(trials, g, threshold=thr, system_turn_share=None)
        for t, d in zip(trials, mine):
            sid, name, score = SpeakerEngine.match(t["emb"], enrolled, thr, g.negatives, margin=0.0)
            assert d["name"] == name
            if name is not None:
                assert d["score"] == pytest.approx(score, abs=1e-5)
            else:
                assert d["raw_top"] == pytest.approx(score, abs=1e-5)  # what match() reports when nobody is accepted


def test_centroid_margin_asnorm_and_replay():
    trials, g = make_trials(), make_gallery()
    c = spk.scorer_centroid(trials, g, threshold=0.45)
    assert c[0]["name"] == "我" and c[2]["name"] == "陈默"
    m = spk.scorer_margin(trials, g, threshold=0.45, margin=0.95)
    assert all(x["name"] is None for x in m[1:2])                  # nothing beats the runner-up by that much there
    a = spk.scorer_asnorm(trials, g)                               # too few cohort vectors: falls back to legacy
    assert [x["name"] for x in a] == [x["name"] for x in spk.scorer_legacy(trials, g)]
    res = spk.replay(trials, g, [("legacy", spk.scorer_legacy), ("centroid", spk.scorer_centroid)], ["我", "陈默"])
    leg = res["legacy"]
    assert leg["overall"]["n"] == 6 and leg["overall"]["correct"] == 3 and leg["overall"]["accepted"] == 1
    assert leg["headline"]["accuracy"] == pytest.approx(5 / 6, abs=1e-4)
    assert leg["unknown_speakers"]["老秦"]["rejected"] == 0 and leg["unknown_speakers"]["旁白"]["rejected"] == 2
    sw = {p["threshold"]: p for p in leg["sweep"]}
    assert sw[0.2]["frr"] == 0.0 and sw[0.8]["far"] == 0.0 and sw[0.8]["frr"] == pytest.approx(1 / 3, abs=1e-4)
    assert leg["equal_error"]["threshold"] in spk.SWEEP
    # a scorer that returns the wrong number of decisions is an error, and scorers never see the truth
    seen = {}

    def nosy(trs, gallery, **kw):
        seen.update(trs[0])
        return [{"name": None, "best": None, "score": None}] * len(trs)

    spk.replay(trials, g, [("nosy", nosy)], ["我", "陈默"])
    assert "speaker" not in seen and "condition" not in seen and "emb" in seen
    with pytest.raises(RuntimeError):
        spk.replay(trials, g, [("bad", lambda trs, gallery, **kw: [])], ["我", "陈默"])


def test_asnorm_with_a_real_cohort_separates_targets():
    rng = np.random.default_rng(5)
    me, boss = rng.standard_normal(32), rng.standard_normal(32)
    g = spk.Gallery({1: "我", 2: "陈默"}, [(1, unit(*me), "mic", 5.0), (2, unit(*boss), "mic", 5.0)], [], 1, None)
    trials = []
    for k in range(30):
        trials.append({"id": f"t{k}", "emb": unit(*(me + 0.8 * rng.standard_normal(32))), "speech_s": 3.0, "dur": 3.0, "speaker": "我",
                       "condition": "near", "mixed": False})
    for k in range(60):
        trials.append({"id": f"n{k}", "emb": unit(*rng.standard_normal(32)), "speech_s": 3.0, "dur": 3.0, "speaker": "旁白",
                       "condition": "near", "mixed": False})
    res = spk.replay(trials, g, [("asnorm", spk.scorer_asnorm)], ["我", "陈默"], {"threshold": 2.0})
    o = res["asnorm"]["overall"]
    assert o["frr"] <= 0.1 and o["far"] <= 0.1
    assert "sweep" not in res["asnorm"]                           # normalised scores are not cosines


def test_load_scorer_from_file_and_builtin(tmp_path):
    f = tmp_path / "my_scorer.py"
    f.write_text("def always(trials, gallery, **kw):\n    return [{'name': '我', 'best': '我', 'score': 0.9} for _ in trials]\n", encoding="utf-8")
    name, fn = spk.load_scorer(f"{f}:always")
    assert name == "my_scorer:always" and fn([{}], None)[0]["name"] == "我"
    assert spk.load_scorer("legacy") == ("legacy", spk.scorer_legacy)
    with pytest.raises(SystemExit):
        spk.load_scorer("nope")


def _blob(v) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def test_gallery_cluster_and_voice_trials_from_a_database(tmp_path):
    truth = make_truth()
    rows = [{"id": 1, "u0": 0.6, "u1": 4.0, "raw": "a", "name": "我", "sid": 1, "score": 0.8, "cluster": 0, "loud": -18.0},
            {"id": 2, "u0": 5.0, "u1": 8.0, "raw": "b", "cluster": 1, "loud": -38.0, "score": 0.41},
            {"id": 3, "u0": 9.0, "u1": 13.0, "raw": "c", "cluster": 2, "share": 0.97, "score": 0.5},
            {"id": 4, "u0": 14.0, "u1": 15.0, "raw": "d", "cluster": 1, "score": 0.2}]
    db = tmp_path / "audio_note.sqlite3"
    _bench_db(db, truth, rows, v1=True)
    c = sqlite3.connect(db)
    c.execute("INSERT INTO speakers VALUES (1, '我', 1)")
    c.execute("INSERT INTO speakers VALUES (2, '陈默', 0)")
    c.execute("INSERT INTO speaker_embeddings VALUES (1, 1, ?, 4, 'mic', 5.0)", (_blob([2, 0, 0, 0]),))
    c.execute("INSERT INTO speaker_embeddings VALUES (2, 2, ?, 4, 'file', 5.0)", (_blob([0, 3, 0, 0]),))
    c.execute("INSERT INTO speaker_negatives VALUES (1, 1, ?)", (_blob([0, 0, 1, 0]),))
    for label, vec, sp in ((0, [1, 0.1, 0, 0], 3.4), (1, [0.2, 1, 0, 0], 4.0), (2, [1, 0, 0.1, 0], 4.0), (9, [0, 0, 0, 1], 1.0)):
        c.execute("INSERT INTO cluster_embeddings VALUES (1, ?, ?, 4, ?)", (label, _blob(vec), sp))
    c.execute("INSERT INTO utterance_voice VALUES (1, 'm', 4, ?, 3.2)", (_blob([1, 0, 0, 0]),))
    c.execute("INSERT INTO utterance_voice VALUES (3, 'm', 4, ?, NULL)", (_blob([1, 0, 0.2, 0]),))
    c.commit()
    c.close()
    g = spk.load_gallery(db)
    assert g.names == {1: "我", 2: "陈默"} and g.self_id == 1 and len(g.samples) == 2 and len(g.negatives) == 1
    assert np.linalg.norm(g.samples[0][1]) == pytest.approx(1.0) and g.samples[0][2] == "mic"
    utts = cer.load_utterances(db, truth)
    assert [u["id"] for u in utts["s1"]] == [1, 2, 3, 4] and utts["s1"][1]["u0"] == pytest.approx(5.0)
    trials, _ = spk.utterance_trials(truth, utts)
    ct, stats = spk.cluster_trials(db, trials)
    assert stats == {"clusters": 4, "without_utterances": 1, "impure": 1}
    by = {t["id"]: t for t in ct}
    assert by["1:0"]["speaker"] == "我" and by["1:0"]["loudness_db"] == -18.0 and not by["1:0"]["mixed"]
    assert by["1:1"]["speaker"] == "陈默" and by["1:1"]["mixed"] and by["1:1"]["purity"] == 0.75      # 3 s of 陈默 + 1 s of 老秦
    assert by["1:2"]["system_share"] == pytest.approx(0.97)
    res = spk.replay(ct, g, [("legacy", spk.scorer_legacy)], truth["enrolled"])
    assert res["legacy"]["overall"]["correct"] == 2 and res["legacy"]["overall"]["accepted"] == 0   # narrator: system audio, not the owner
    vt = spk.voice_trials(db, trials)
    assert [t["id"] for t in vt] == [1, 3] and vt[0]["speech_s"] == 3.2 and vt[1]["speech_s"] == pytest.approx(4.0)
    old = tmp_path / "old.sqlite3"
    _bench_db(old, truth, rows)
    assert spk.voice_trials(old, trials) is None

    # --- profile mode on the same database
    pdir = tmp_path / "profiles" / "p"
    (pdir / "data").mkdir(parents=True)
    common.write_json(pdir / "truth.json", truth)
    (pdir / "data" / "audio_note.sqlite3").write_bytes(db.read_bytes())
    out = spk.run_profile(pdir, label="p")
    assert out["kind"] == "speakers" and out["overall"]["n"] == 4 and out["overall"]["correct"] == 1
    assert out["system_audio"] == {"trials": 1, "labelled_self": 0}
    assert out["near_threshold"]["scored"] == 4 and out["near_threshold"]["share"] == 0.5

    # --- proxy mode: aggregates only
    c = sqlite3.connect(db)
    c.execute("UPDATE utterances SET speaker_source='user', speaker_id=2, speaker_name='陈默' WHERE id=2")
    c.execute("INSERT INTO utterance_rejections VALUES (4, 2)")
    c.commit()
    c.close()
    px = spk.proxy(db, threshold=0.45, band=0.10, scorers=[("legacy", spk.scorer_legacy)])
    assert px["utterances"] == 4 and px["decisions"]["sure"]["n"] == 1 and px["decisions"]["user"]["n"] == 1
    assert px["decisions"]["unknown"]["n"] == 2 and px["near_threshold"] == {"scored": 3, "n": 1, "share": pytest.approx(1 / 3, abs=1e-4)}
    assert px["by_loudness"]["system"] == {"unknown": 1} and px["embedding_source"] == "utterance_voice"
    rp = px["replay"]["legacy"]
    assert px["replay_trials"] == 4 and rp["hand_labels"] == {"n": 1, "contradicted": 0}
    assert rp["must_not"] == {"n": 2, "violated": 1}              # utterance 4 shares the cluster vector of 陈默, whom the user vetoed
    dumped = json.dumps(px, ensure_ascii=False)
    assert "我" not in dumped and "陈默" not in dumped             # no names leave the proxy


def test_spec_gallery_adapter_with_a_stand_in_scoring_module(tmp_path, monkeypatch):
    """The adapter for the 1.0 scoring layer is exercised against a stand-in that follows the SPEC signature."""
    import types

    truth = make_truth()
    pdir = tmp_path / "profiles" / "p"
    (pdir / "data").mkdir(parents=True)
    (pdir / "config.yaml").write_text(f"paths:\n  data_dir: {json.dumps(str(pdir / 'data'))}\n", encoding="utf-8")
    db = pdir / "data" / "audio_note.sqlite3"
    from audio_note.db import DB

    real = DB(db)                                                  # a real, current-schema database (the adapter opens a copy with DB)
    real.close_all()
    calls = []

    class Decision:
        def __init__(self, sid, name, score, guess=None):
            self.speaker_id, self.name, self.score, self.norm = sid, name, score, score * 10
            self.guess_id, self.guess_score, self.method, self.runner_up = guess, None, "anchor", None

    class FakeGallery:
        @classmethod
        def load(cls, db_, cfg):
            calls.append(Path(db_.path))
            return cls()

        def score(self, emb, *, speech_s=None, loudness_db=None, system_share=None, exclude=frozenset()):
            if emb[0] > 0.9:
                return Decision(1, "我", 0.8)
            if emb[1] > 0.9:
                return Decision(None, None, 0.4, guess=2)
            return Decision(None, None, 0.1)

    fake = types.ModuleType("audio_note.speech.scoring")
    fake.Gallery = FakeGallery
    monkeypatch.setitem(sys.modules, "audio_note.speech.scoring", fake)
    import audio_note.speech as speech_pkg

    monkeypatch.setattr(speech_pkg, "scoring", fake, raising=False)
    g = make_gallery()
    g.db_path = str(db)
    before = db.read_bytes()
    out = spk.scorer_spec_gallery(make_trials()[:4], g)
    assert [d["name"] for d in out] == ["我", None, None, None]
    assert out[2]["guess"] == "陈默" and out[2]["best"] == "陈默" and out[0]["score"] == pytest.approx(8.0)
    assert calls and calls[0] != db                                # the scoring layer only ever saw a copy
    assert db.read_bytes() == before
    g.db_path = None
    with pytest.raises(RuntimeError):
        spk.scorer_spec_gallery(make_trials(), g)


# --- bench_cost + report -----------------------------------------------------------------------------
def test_cost_per_audio_second():
    rep = {"audio_s": 200.0, "segments": 10, "utterances": 40, "process_cpu_s": 60.0, "process_wall_s": 30.0,
           "stages": {"diarize": {"n": 10, "wall_s": 20.0, "cpu_s": 40.0}, "embed": {"n": 50, "wall_s": 4.0, "cpu_s": 8.0},
                      "identify": {"n": 60, "wall_s": 0.1, "cpu_s": 0.2}, "vad_gate": {"n": 1, "wall_s": 3.0, "cpu_s": 5.0}}}
    r = cost.per_audio_second(rep)
    assert r["diarize"]["cpu_per_audio_s"] == 0.2 and r["embed"]["wall_per_audio_s"] == 0.02
    assert r["voice_total"]["cpu_per_audio_s"] == pytest.approx(0.241) and r["pipeline"]["cpu_per_audio_s"] == 0.3
    assert r["vad_gate"]["cpu_per_audio_s"] == 0.025
    empty = cost.per_audio_second({"audio_s": 0, "stages": {}})
    assert empty["diarize"]["cpu_per_audio_s"] is None and empty["diarize"]["calls"] == 0
    text = cost.format_text({"label": "x", "variant": "mixed", "audio_s": 200.0, "threads": {"1": r}})
    assert "0.200 / 0.100" in text


def test_report_merges_results_into_markdown(tmp_path):
    truth = make_truth()
    utts = {"s1": [utt(1, 0.5, 4.1, "今天过一下清河App的灰度安排。", "今天过一下青禾App的灰度安排。", name="我", sid=1),
                   utt(2, 5.0, 8.0, "青禾的灰度定在周五。")]}
    res, _ = cer.score(truth, utts)
    a = {"kind": "cer", "label": "baseline", "headline": cer.headline(res), **res, "import": {"code": {"version": None, "commit": "a1f36ac"}}}
    b = {"kind": "cer", "label": "v1", "headline": cer.headline(res), **res}
    trials, facts = spk.utterance_trials(truth, utts)
    ev = spk.evaluate(trials, ["我", None], truth["enrolled"])
    s = {"kind": "speakers", "label": "baseline", **ev, **facts, "near_threshold": {"share": 0.25}}
    rp = {"kind": "speakers_replay", "label": "baseline", "source": "clusters", "trials": 2,
          "scorers": {"legacy": {**ev, "equal_error": {"threshold": 0.5, "frr": 0.1, "far": 0.1}}}}
    c = {"kind": "cost", "label": "baseline", "threads": {"1": cost.per_audio_second({"audio_s": 100.0, "process_cpu_s": 11.0, "process_wall_s": 9.0,
                                                                                    "stages": {"diarize": {"n": 1, "wall_s": 8.0, "cpu_s": 10.0}}})}}
    px = {"kind": "speakers_proxy", "label": "real", "utterances": 100, "decisions": {"sure": {"n": 60, "share": 0.6}, "unknown": {"n": 40, "share": 0.4}},
          "near_threshold": {"share": 0.275}, "replay_trials": 90, "embedding_source": "cluster_embeddings",
          "replay": {"legacy": {"sure": {"share": 0.6}, "maybe": {"share": 0.0}, "unknown": {"share": 0.4}, "near_threshold": {"share": 0.27},
                                "hand_labels": {"n": 3, "contradicted": 1}, "must_not": {"n": 2, "violated": 0},
                                "differs_from_stored": {"share": 0.1}}}}
    md = report.build_report([a, b, s, rp, c, px])
    assert md.startswith("# audio-note 评测汇总")
    for needle in ("## 转写（CER）", "| 指标 | baseline | v1 |", "## 说话人（流水线的实际判定）", "## 说话人（向量回放，不跑模型）", "## 本机开销",
                   "## 真实数据代理指标", "a1f36ac", "1 线程 · 分离", "0.100 / 0.080", "27.5%", "1 / 3"):
        assert needle in md, needle
    files = []
    for k, obj in enumerate([a, s]):
        f = tmp_path / f"r{k}.json"
        common.write_json(f, obj)
        files.append(str(f))
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    out = tmp_path / "report.md"
    assert report.main(files + [str(bad), "--out", str(out)]) == 0
    assert "## 转写（CER）" in out.read_text(encoding="utf-8")
    assert report.pct(None) == "—" and report.pct(0.1234, 2) == "12.34%" and report.table(["a"], []) == ""
