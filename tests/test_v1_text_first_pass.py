"""The first correction pass (SPEC v1 section 6.4) and the closed loop it makes with the inbox and the lexicon:

    a wrong word in today's text -> a question -> the answer -> an alias -> the history fixed through revisions
    (undoable) -> the same mis-hearing in a NEW segment replaced and marked at processing time -> a hot word.

No network, no models: the engine and the ASR are fakes, as in tests/test_v1_pipeline.py. Names are invented.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from audio_note import inbox
from audio_note.db import DB
from audio_note.llm.client import LLMError
from audio_note.speech import correct, lexicon, pinyin, pipeline, revisions
from audio_note.speech.asr import ASRResult
from audio_note.speech.correct import Corrector, LexiconCorrector, SegmentCorrector, correction_mode, default_corrector
from audio_note.speech.pipeline import Processor
from audio_note.speech.speakers import Turn
from audio_note.speech.stage_text import revision_steps

SR = 16000
DAY = "2026-11-02"
needs_pypinyin = pytest.mark.skipif(not pinyin.available(), reason="pypinyin is not installed")


# --------------------------------------------------------------------------- fixtures / doubles
@pytest.fixture
def pcfg(cfg_factory):
    return cfg_factory({"pipeline": {"min_segment_age_s": 0}, "memory": {"retrieval": "fts"}, "mimo": {"asr_concurrency": 1}})


@pytest.fixture
def pdb(pcfg) -> DB:
    return DB(pcfg.db_path)


def make_segment(cfg, db: DB, started: datetime, seconds: float = 6.0, seed: int = 0) -> int:
    rng = np.random.default_rng(seed)
    audio = (rng.standard_normal(int(seconds * SR)) * 0.05).astype(np.float32)
    d = cfg.audio_dir / started.strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    path = d / (started.strftime("%H%M%S_000") + ".wav")
    sf.write(path, audio, SR, format="WAV", subtype="PCM_16")
    return db.add_segment(path, started, seconds, SR, speech_s=seconds)


class Engine:
    """Scripted diarization: one turn per (start, end, cluster); nobody is identified."""

    def __init__(self, *turns: tuple[float, float, int]):
        self.turns = [Turn(a, b, c) for a, b, c in turns]

    def diarize(self, audio):
        return [Turn(t.start, t.end, t.cluster) for t in self.turns]

    def identify_turns(self, audio, turns, db, segment_id=None):
        return turns


class ASR:
    """Returns the scripted texts in call order (the last one repeats)."""

    def __init__(self, *texts: str):
        self.texts = list(texts)
        self.calls = 0
        self._lock = threading.Lock()

    def transcribe(self, audio):
        with self._lock:
            self.calls += 1
            k = self.calls
        return ASRResult(self.texts[min(k, len(self.texts)) - 1], round(len(audio) / SR, 2), 1)


class NoKeyProcessor(Processor):
    """A Processor on a machine without an API key: building the client fails, as MimoClient would."""

    @property
    def client(self):
        raise LLMError("ASR 需要 MiMo API key")


class FakeLLM:
    """client.json double for the legacy corrector: proposes the scripted edits, records what it was shown."""

    def __init__(self, edits: list[dict] | None = None, fail: bool = False):
        self.edits = edits or []
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def json(self, system, user, **kw):
        self.calls.append((system, user))
        if self.fail:
            raise LLMError("boom")
        return {"edits": self.edits}


def proc(cfg, db, asr, engine=None, corrector=None, cls=Processor, client: Any = None) -> Processor:
    return cls(cfg, db, client=client if client is not None else object(), engine=engine or Engine((0.0, 5.5, 0)), asr=asr,
               corrector=corrector)


def utterances(db: DB, seg_id: int | None = None) -> list[Any]:
    if seg_id is None:
        return db.query("SELECT * FROM utterances ORDER BY abs_start, id")
    return db.query("SELECT * FROM utterances WHERE segment_id=? ORDER BY start_s, id", (seg_id,))


def teach(db: DB, term: str, *aliases: str, **kw: Any) -> tuple[int, list[int]]:
    tid = lexicon.upsert_term(db, term, **kw)
    return tid, [lexicon.add_alias(db, tid, a) for a in aliases]


# --------------------------------------------------------------------------- which corrector
def test_correction_mode_and_default_corrector(cfg_factory, pdb):
    cfg = cfg_factory()
    assert cfg.correction.mode == "segment" and correction_mode(cfg) == "segment" and correct.MODES == ("session", "segment", "off")
    assert correction_mode(cfg_factory({"correction": {"mode": "Segment "}}, name="a.yaml")) == "segment"
    assert correction_mode(cfg_factory({"correction": {"mode": "whatever"}}, name="b.yaml")) == "segment"
    assert correction_mode(cfg_factory({"correction": {"enabled": False, "mode": "segment"}}, name="c.yaml")) == "off"
    off = cfg_factory({"correction": {"mode": "off"}}, name="d.yaml")
    seg = cfg_factory({"correction": {"mode": "segment"}}, name="e.yaml")
    sess = cfg_factory({"correction": {"mode": "session"}}, name="f.yaml")
    called: list[int] = []

    def factory():
        called.append(1)
        return FakeLLM()

    c = default_corrector(sess, pdb, factory)
    assert isinstance(c, LexiconCorrector) and c.source == "lexicon" and called == []          # session: the client is never built
    assert default_corrector(off, pdb, factory) is None and default_corrector(sess, None, factory) is None
    s = default_corrector(seg, pdb, factory)
    assert isinstance(s, SegmentCorrector) and isinstance(s, Corrector) and called == [1]
    assert isinstance(default_corrector(seg, pdb, FakeLLM()), SegmentCorrector)                # a client instead of a factory

    def no_key():
        raise LLMError("no key")

    assert isinstance(default_corrector(seg, pdb, no_key), LexiconCorrector)                   # segment mode without a key: rules only


def test_processor_builds_the_corrector_for_the_mode(cfg_factory, pdb):
    session = cfg_factory({"correction": {"mode": "session"}}, name="s.yaml")
    assert isinstance(NoKeyProcessor(session, pdb).corrector, LexiconCorrector)                # no API key needed
    seg = cfg_factory({"correction": {"mode": "segment"}}, name="g.yaml")
    assert isinstance(Processor(seg, pdb, client=FakeLLM()).corrector, SegmentCorrector)
    assert isinstance(NoKeyProcessor(seg, pdb).corrector, LexiconCorrector)
    assert Processor(cfg_factory({"correction": {"mode": "off"}}, name="o.yaml"), pdb, client=object()).corrector is None
    assert Processor(cfg_factory({"correction": {"enabled": False}}, name="x.yaml"), pdb, client=object()).corrector is None
    mine = object()
    assert Processor(session, pdb, client=object(), corrector=mine).corrector is mine           # an injected one always wins


def test_lexicon_corrector_applies_the_rules_and_says_who_did_it(pdb, pcfg):
    tid, (aid,) = teach(pdb, "星河", "兴和")
    c = LexiconCorrector(pcfg, pdb)
    out = c.correct([{"i": 0, "speaker": "未知", "text": "兴和的排期"}, {"i": 1, "speaker": "未知", "text": "没事"}], ["忽略的词表"])
    assert out[0]["text"] == "星河的排期" and out[1] == {"i": 1, "text": "没事", "edits": []}
    assert out[0]["edits"] == [{"from": "兴和", "to": "星河", "reason": "词库", "alias_id": aid, "lexicon_id": tid, "pos": 0, "start": 0,
                                "end": 2, "source": "lexicon"}]
    assert c.correct([], None) == [] and LexiconCorrector(pcfg, None).correct([{"i": 0, "text": "兴和"}]) == [{"i": 0, "text": "兴和", "edits": []}]


def test_segment_corrector_runs_the_rules_first_then_the_llm(cfg_factory, pdb):
    cfg = cfg_factory({"correction": {"mode": "segment"}})
    teach(pdb, "星河", "兴和")
    llm = FakeLLM([{"i": 0, "from": "在说", "to": "再说", "reason": "同音"}])
    c = SegmentCorrector(cfg, llm, pdb)
    (out,) = c.correct([{"i": 0, "speaker": "未知", "text": "兴和的事明天在说"}], ["星河"])
    assert out["text"] == "星河的事明天再说"
    assert [(e["from"], e["to"], e["source"]) for e in out["edits"]] == [("兴和", "星河", "lexicon"), ("在说", "再说", "llm")]
    assert "星河的事明天在说" in llm.calls[0][1] and "兴和" not in llm.calls[0][1]              # the model sees the fixed text
    # the model failing costs only its own edits
    (out,) = SegmentCorrector(cfg, FakeLLM(fail=True), pdb).correct([{"i": 0, "speaker": "未知", "text": "兴和的事"}], [])
    assert out["text"] == "星河的事" and [e["source"] for e in out["edits"]] == ["lexicon"]
    assert c.correct([], []) == []


# --------------------------------------------------------------------------- revision_steps
def test_revision_steps_replays_edits_one_at_a_time():
    edits = [{"i": 0, "from": "兴和", "to": "星河", "source": "lexicon", "alias_id": 4, "pos": 0, "start": 0, "end": 2},
             {"i": 0, "from": "兰姐", "to": "岚姐", "source": "lexicon", "alias_id": 5},
             {"i": 0, "from": "在说", "to": "再说", "reason": "同音"}]
    steps = revision_steps("兴和的事兰姐明天在说", "星河的事岚姐明天再说", edits, "llm")
    assert [(s["old"], s["new"], s["source"]) for s in steps] == [
        ("兴和的事兰姐明天在说", "星河的事兰姐明天在说", "lexicon"),
        ("星河的事兰姐明天在说", "星河的事岚姐明天在说", "lexicon"),
        ("星河的事岚姐明天在说", "星河的事岚姐明天再说", "llm")]
    assert steps[0]["detail"] == {"edits": [{"from": "兴和", "to": "星河", "alias_id": 4, "start": 0, "end": 2}]}
    assert steps[2]["detail"] == {"edits": [{"from": "在说", "to": "再说", "reason": "同音", "start": 8, "end": 10}]}
    # the same wrong word twice: each occurrence is its own step, left to right
    twice = revision_steps("兴和和兴和", "星河和星河", [{"from": "兴和", "to": "星河"}, {"from": "兴和", "to": "星河"}], "lexicon")
    assert [(s["new"], s["detail"]["edits"][0]["start"]) for s in twice] == [("星河和兴和", 0), ("星河和星河", 3)]
    # edits that do not explain the stored text: skipped, and one closing step ends at the stored text
    odd = revision_steps("原文", "成文了", [{"from": "没有", "to": "有"}, {"from": "", "to": "x"}, {"bad": 1}], "llm")
    assert odd == [{"old": "原文", "new": "成文了", "source": "llm", "detail": {"edits": []}}]
    assert revision_steps("一样", "一样", [], "llm") == []


# --------------------------------------------------------------------------- the first pass in the pipeline
def test_session_mode_fixes_known_mis_hearings_at_processing_time(pcfg, pdb):
    tid, (a1,) = teach(pdb, "星河", "兴和", kind="project")
    tid2, (a2,) = teach(pdb, "岚姐", "兰姐", kind="person")
    seg = make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, 0, 0))
    p = proc(pcfg, pdb, ASR("兴和的排期兰姐在跟，兴和下周上线", "这句没有问题"), Engine((0.0, 2.5, 0), (3.0, 5.5, 1)), cls=NoKeyProcessor)
    assert p.process_segment(seg) == 2
    u1, u2 = utterances(pdb, seg)
    assert u1["text_raw"] == "兴和的排期兰姐在跟，兴和下周上线" and u1["text"] == "星河的排期岚姐在跟，星河下周上线"
    assert u1["text_source"] == "lexicon" and u1["edited_at"]
    assert (u2["text"], u2["text_source"], u2["edited_at"]) == ("这句没有问题", None, None)
    # one revision per applied change, a chain from the raw text to the stored text
    h = list(reversed(revisions.history(pdb, u1["id"])))
    assert [x["source"] for x in h] == ["lexicon"] * 3 and h[0]["old_text"] == u1["text_raw"] and h[-1]["new_text"] == u1["text"]
    assert all(h[k]["new_text"] == h[k + 1]["old_text"] for k in range(2))
    assert [(x["detail"]["edits"][0]["from"], x["detail"]["edits"][0]["alias_id"]) for x in h] == [("兴和", a1), ("兰姐", a2), ("兴和", a1)]
    assert revisions.history(pdb, u2["id"]) == []
    # the page shows what was fixed, and every fix can be taken back on its own
    marks = revisions.marks(pdb, [u1, u2])
    assert marks == {u1["id"]: [{"start": 0, "end": 2, "kind": "fixed", "from": "兴和", "source": "lexicon"},
                                {"start": 5, "end": 7, "kind": "fixed", "from": "兰姐", "source": "lexicon"},
                                {"start": 10, "end": 12, "kind": "fixed", "from": "兴和", "source": "lexicon"}]}
    revisions.undo(pdb, pcfg, u1["id"], h[1]["id"])
    assert utterances(pdb, seg)[0]["text"] == "星河的排期兰姐在跟，星河下周上线"
    # the transcript row of the segment keeps the edits too, and the day file has the fixed text
    tr = pdb.one("SELECT * FROM transcripts WHERE segment_id=?", (seg,))
    assert [e["source"] for e in json.loads(tr["correction_json"])] == ["lexicon"] * 3 and "星河的排期岚姐" in tr["text_corrected"]
    assert "星河的排期" in (pcfg.transcripts_dir / f"{DAY}.md").read_text(encoding="utf-8")
    # what was used and what was said counts as hits (hot-word order)
    t = lexicon.get_term(pdb, tid)
    assert t["hits"] == 2 and t["aliases"][0]["hits"] == 2 and lexicon.get_term(pdb, tid2)["hits"] == 1
    assert pdb.get_segment(seg)["status"] == "done"


def test_an_injected_corrector_keeps_working_and_its_changes_get_revisions(pcfg, pdb):
    class Replace:
        def correct(self, items, glossary):
            return [{"i": it["i"], "text": it["text"].replace("世界", "世间"),
                     "edits": [{"from": "世界", "to": "世间", "reason": "test"}] if "世界" in it["text"] else []} for it in items]

    class Rewrite:                                              # changes the text without saying how
        def correct(self, items, glossary):
            return [{"i": it["i"], "text": it["text"] + "。"} for it in items]

    teach(pdb, "星河", "兴和")                                  # the lexicon is NOT consulted: the injected corrector is the pass
    seg = make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, 0, 0))
    assert proc(pcfg, pdb, ASR("你好世界，兴和"), corrector=Replace()).process_segment(seg) == 1
    (u,) = utterances(pdb, seg)
    assert (u["text"], u["text_source"]) == ("你好世间，兴和", "llm")
    (h,) = revisions.history(pdb, u["id"])
    assert (h["old_text"], h["new_text"], h["source"]) == ("你好世界，兴和", "你好世间，兴和", "llm")
    assert h["detail"] == {"edits": [{"from": "世界", "to": "世间", "reason": "test", "start": 2, "end": 4}]}
    assert json.loads(pdb.one("SELECT correction_json FROM transcripts WHERE segment_id=?", (seg,))["correction_json"]) == [
        {"i": 0, "from": "世界", "to": "世间", "reason": "test"}]                               # the seam's own shape is untouched
    seg2 = make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, 5, 0), seed=1)
    assert proc(pcfg, pdb, ASR("没说怎么改"), corrector=Rewrite()).process_segment(seg2) == 1
    (u2,) = utterances(pdb, seg2)
    (h2,) = revisions.history(pdb, u2["id"])
    assert (h2["old_text"], h2["new_text"], h2["source"], h2["detail"]) == ("没说怎么改", "没说怎么改。", "llm", {"edits": []})
    assert u2["text_source"] == "llm"


def test_mode_off_and_a_failing_corrector_store_the_raw_text(cfg_factory, pdb):
    cfg = cfg_factory({"pipeline": {"min_segment_age_s": 0}, "memory": {"retrieval": "fts"}, "correction": {"mode": "off"}})
    teach(pdb, "星河", "兴和")
    seg = make_segment(cfg, pdb, datetime(2026, 11, 2, 9, 0, 0))
    assert proc(cfg, pdb, ASR("兴和的排期")).process_segment(seg) == 1
    (u,) = utterances(pdb, seg)
    assert (u["text"], u["text_source"]) == ("兴和的排期", None) and revisions.history(pdb, u["id"]) == []

    class Boom:
        def correct(self, items, glossary):
            raise RuntimeError("boom")

    seg2 = make_segment(cfg, pdb, datetime(2026, 11, 2, 9, 5, 0), seed=1)
    assert proc(cfg, pdb, ASR("兴和的排期"), corrector=Boom()).process_segment(seg2) == 1
    assert utterances(pdb, seg2)[0]["text_source"] is None


def test_segment_mode_records_both_sources(cfg_factory, pdb):
    cfg = cfg_factory({"pipeline": {"min_segment_age_s": 0}, "memory": {"retrieval": "fts"}, "correction": {"mode": "segment"}})
    teach(pdb, "星河", "兴和")
    llm = FakeLLM([{"i": 0, "from": "在说", "to": "再说", "reason": "同音错别字"}])
    seg = make_segment(cfg, pdb, datetime(2026, 11, 2, 9, 0, 0))
    assert proc(cfg, pdb, ASR("兴和的事明天在说"), client=llm).process_segment(seg) == 1
    (u,) = utterances(pdb, seg)
    assert (u["text"], u["text_source"]) == ("星河的事明天再说", "llm")                        # who made the LAST change
    assert [x["source"] for x in reversed(revisions.history(pdb, u["id"]))] == ["lexicon", "llm"]
    assert [m["source"] for m in revisions.marks(pdb, [u])[u["id"]]] == ["lexicon", "llm"]


def test_split_pieces_get_the_first_pass_and_questions_follow_the_word(pcfg, pdb):
    seg = make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, 0, 0))
    p = proc(pcfg, pdb, ASR("兴和的排期牙测在跟"))
    assert p.process_segment(seg) == 1
    (u,) = utterances(pdb, seg)
    q = inbox.add(pdb, "term", title="「牙测」是哪个词？", cfg=pcfg, payload={
        "span": "牙测", "sentence": u["text"], "utterance_id": u["id"], "start": 5, "end": 7, "candidates": ["压测"], "term_kind": "term"})
    teach(pdb, "星河", "兴和")                                  # learned after the sentence was stored
    p2 = proc(pcfg, pdb, ASR("兴和的排期", "牙测在跟"))
    res = p2.split_utterance(u["id"], at=[2.5])
    a, b = [x["id"] for x in res["pieces"]]
    rows = {r["id"]: r for r in utterances(pdb, seg)}
    assert (rows[a]["text"], rows[a]["text_raw"], rows[a]["text_source"]) == ("星河的排期", "兴和的排期", "lexicon")
    assert (rows[b]["text"], rows[b]["text_source"]) == ("牙测在跟", None)
    assert [h["source"] for h in revisions.history(pdb, a)] == ["lexicon"] and revisions.history(pdb, b) == []
    assert revisions.history(pdb, u["id"]) == []                # the replaced row took its history with it
    it = inbox.get(pdb, q)
    assert it["utterance_id"] == b and it["payload"]["utterance_id"] == b and it["payload"]["sentence"] == "牙测在跟"
    assert "start" not in it["payload"]
    assert revisions.marks(pdb, [rows[b]])[b] == [{"start": 0, "end": 2, "kind": "uncertain", "inbox_id": q, "candidates": ["压测"]}]


def test_transcribe_file_follows_the_mode(pcfg, pdb, tmp_path):
    teach(pdb, "星河", "兴和")
    wav = tmp_path / "clip.wav"
    sf.write(wav, (np.random.default_rng(3).standard_normal(3 * SR) * 0.05).astype(np.float32), SR)
    out = pipeline.transcribe_file(pcfg, object(), wav, db=pdb, engine=Engine((0.0, 3.0, 0)), asr=ASR("兴和的排期"))
    assert [(o["text_raw"], o["text"]) for o in out] == [("兴和的排期", "星河的排期")]
    assert pdb.query("SELECT id FROM utterances") == [] and pdb.query("SELECT id FROM utterance_revisions") == []


def test_the_glossary_seam_gets_what_the_user_taught(pcfg, pdb):
    pcfg.glossary_path.write_text("灰度发布 # 上线方式\n旧词\n", encoding="utf-8")
    pdb.upsert_speaker("王敏")
    assert pipeline.load_glossary(pcfg, pdb) == ["灰度发布", "旧词", "王敏"]
    lexicon.sync_sources(pdb, pcfg)
    lexicon.upsert_term(pdb, "星河", source="answer")
    lexicon.upsert_term(pdb, "岚姐", source="edit")
    lexicon.upsert_term(pdb, "候选", source="auto", status="candidate")
    old = [t for t in lexicon.list_terms(pdb) if t["term"] == "旧词"][0]
    lexicon.update_term(pdb, old["id"], status="disabled")
    assert pipeline.load_glossary(pcfg, pdb) == ["灰度发布", "王敏", "星河", "岚姐"]
    assert pipeline.load_glossary(pcfg) == ["灰度发布", "旧词"]                                 # without a database: the file


# --------------------------------------------------------------------------- the closed loop
@needs_pypinyin
def test_closed_loop_from_a_wrong_word_to_automatic_fixes(pcfg, pdb):
    from audio_note.events import EventBus

    bus = EventBus()
    sub = bus.subscribe(since=0)
    # 1. today's recordings: the recogniser writes the project name 星河 as 兴和; nobody knows better yet
    seg1 = make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, 0, 0))
    seg2 = make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, 30, 0), seed=1)
    p = proc(pcfg, pdb, ASR("今天过一下兴和的排期", "兴和下周要上线"))
    assert p.process_segment(seg1) == 1 and p.process_segment(seg2) == 1
    u1, u2 = utterances(pdb)
    assert (u1["text"], u1["text_source"], u2["text"]) == ("今天过一下兴和的排期", None, "兴和下周要上线")
    assert lexicon.Lexicon.load(pdb).hotwords(50) == [] and revisions.marks(pdb, [u1, u2]) == {}

    # 2. the correction step is not sure about the word and asks (speech.refine does this; here by hand)
    q = inbox.add(pdb, "term", title="「兴和」是哪个词？", cfg=pcfg, events=bus, priority=2.0, payload={
        "span": "兴和", "sentence": u1["text"], "utterance_id": u1["id"], "start": 5, "end": 7, "candidates": ["星河", "新河"],
        "why": "像是一个项目名", "source": "llm", "term_kind": "name"})
    assert inbox.add(pdb, "term", title="「兴和」是哪个词？", cfg=pcfg, payload={
        "span": "兴和", "sentence": u2["text"], "utterance_id": u2["id"], "candidates": ["星河"], "term_kind": "name"}) == q
    assert inbox.counts(pdb, pcfg)["open"] == 1
    marks = revisions.marks(pdb, [u1, u2])
    assert marks[u1["id"]] == [{"start": 5, "end": 7, "kind": "uncertain", "inbox_id": q, "candidates": ["星河", "新河"]}]
    assert marks[u2["id"]][0]["kind"] == "uncertain"

    # 3. the user answers once
    res = inbox.answer(pdb, pcfg, q, choice=0, events=bus)
    assert (res["term"], res["alias"], res["scope"]) == ("星河", "兴和", "auto") and sorted(res["changed"]) == [u1["id"], u2["id"]]

    # 4. the history is fixed through revisions - marked as fixed, and undoable
    u1, u2 = utterances(pdb)
    assert (u1["text"], u1["text_source"], u2["text"]) == ("今天过一下星河的排期", "answer", "星河下周要上线")
    assert revisions.marks(pdb, [u1, u2]) == {
        u1["id"]: [{"start": 5, "end": 7, "kind": "fixed", "from": "兴和", "source": "answer"}],
        u2["id"]: [{"start": 0, "end": 2, "kind": "fixed", "from": "兴和", "source": "answer"}]}
    assert "星河下周要上线" in (pcfg.transcripts_dir / f"{DAY}.md").read_text(encoding="utf-8")
    undone = revisions.undo(pdb, pcfg, u2["id"])
    assert undone["text"] == "兴和下周要上线"
    revisions.undo(pdb, pcfg, u2["id"], undone["revision_id"])                                 # ... and back again
    assert utterances(pdb)[1]["text"] == "星河下周要上线"

    # 5. the word is a hot word for the recogniser from now on
    lex = lexicon.Lexicon.load(pdb)
    assert lex.hotwords(50) == ["星河"] and lex.by_term["星河"].source == "answer"
    assert [a.alias for a in lex.by_term["星河"].aliases] == ["兴和"]

    # 6. a NEW segment with the same mis-hearing: replaced and marked at processing time, without asking again
    seg3 = make_segment(pcfg, pdb, datetime(2026, 11, 3, 10, 0, 0), seed=2)
    p3 = proc(pcfg, pdb, ASR("兴和的灰度定在周五"), cls=NoKeyProcessor)                           # no LLM involved at all
    assert p3.process_segment(seg3) == 1
    (u3,) = utterances(pdb, seg3)
    assert (u3["text_raw"], u3["text"], u3["text_source"]) == ("兴和的灰度定在周五", "星河的灰度定在周五", "lexicon")
    assert revisions.marks(pdb, [u3]) == {u3["id"]: [{"start": 0, "end": 2, "kind": "fixed", "from": "兴和", "source": "lexicon"}]}
    assert [h["source"] for h in revisions.history(pdb, u3["id"])] == ["lexicon"]
    assert inbox.add(pdb, "term", title="「星河」对吗？", cfg=pcfg, payload={"span": "星河", "utterance_id": u3["id"]}) == 0
    assert inbox.counts(pdb, pcfg)["open"] == 0
    assert lexicon.get_term(pdb, res["lexicon_id"])["hits"] >= 3

    # the page was told about every step
    types = [e["type"] for e in iter(lambda: sub.get(timeout=0.01), None)]
    assert types.count("inbox") == 2 and types.count("utterance") == 2 and "lexicon" in types
