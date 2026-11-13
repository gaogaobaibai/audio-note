"""audio_note.inbox: questions for the user, one per thing, answered once and generalised (SPEC v1 section 6.6).

No network, no models. Names and sentences are invented.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

import pytest

from audio_note import inbox
from audio_note.db import DB
from audio_note.events import EventBus
from audio_note.speech import lexicon, pinyin, revisions

DAY = "2026-11-02"
needs_pypinyin = pytest.mark.skipif(not pinyin.available(), reason="pypinyin is not installed")


@pytest.fixture
def icfg(cfg_factory):
    return cfg_factory({"memory": {"retrieval": "fts"}})


@pytest.fixture
def idb(icfg) -> DB:
    return DB(icfg.db_path)


def seed(db: DB, cfg: Any, texts: list[str], day: str = DAY) -> list[int]:
    seg = db.add_segment(cfg.audio_dir / f"{day}-{len(texts)}-{texts[0][:2]}.wav", datetime.fromisoformat(f"{day}T09:00:00"), 60.0,
                         speech_s=30.0)
    db.set_segment_status(seg, "done")
    return db.add_utterances([{
        "segment_id": seg, "start_s": float(i), "end_s": float(i + 1), "abs_start": f"{day}T09:{i:02d}:00", "day": day,
        "cluster_label": 0, "speaker_id": None, "speaker_name": None, "speaker_score": None, "text_raw": t, "text": t,
        "text_seg": " ".join(t)} for i, t in enumerate(texts)])


def texts_of(db: DB) -> list[str]:
    return [r["text"] for r in db.query("SELECT text FROM utterances ORDER BY id")]


def ask(db: DB, cfg: Any, span: str, uid: int | None, sentence: str = "", candidates: list[str] | None = None, **kw: Any) -> int:
    payload = {"span": span, "sentence": sentence, "utterance_id": uid, "candidates": candidates or [], "why": "听起来像",
               "source": "llm", "term_kind": kw.pop("term_kind", "name")}
    payload.update(kw.pop("extra", {}))
    return inbox.add(db, "term", title=f"「{span}」是哪个词？", payload=payload, cfg=cfg, **kw)


def drain(sub: Any) -> list[dict]:
    out = []
    while (ev := sub.get(timeout=0.01)) is not None:
        out.append(ev)
    return out


# ---------------------------------------------------------------------------------------------- asking
def test_add_creates_an_open_question_and_announces_it(idb, icfg):
    (a,) = seed(idb, icfg, ["兴和的排期定了"])
    bus = EventBus()
    sub = bus.subscribe(since=0)
    q = ask(idb, icfg, "兴和", a, "兴和的排期定了", ["星河", "兴和", "星河", ""], priority=2.5, session_id=4, events=bus)
    it = inbox.get(idb, q)
    assert (it["kind"], it["status"], it["title"], it["priority"], it["occurrences"]) == ("term", "open", "「兴和」是哪个词？", 2.5, 1)
    assert (it["utterance_id"], it["session_id"], it["answer"], it["answered_at"]) == (a, 4, None, None)
    assert it["payload"]["candidates"] == ["星河"] and it["payload"]["examples"] == []      # no duplicates, never the span itself
    if pinyin.available():
        assert it["dedupe_key"] == "xing he"
    assert inbox.counts(idb, icfg) == {"open": 1, "by_kind": {"term": 1}, "queued": 0, "snoozed": 0}
    assert [it["id"] for it in inbox.list_items(idb)] == [q] and inbox.list_items(idb, kind="voice") == []
    (ev,) = drain(sub)
    assert (ev["type"], ev["open"], ev["kind"], ev["id"], ev["change"]) == ("inbox", 1, "term", q, "added")
    with pytest.raises(ValueError):
        inbox.add(idb, "term", title="  ", payload={"span": "某词"})
    assert inbox.get(idb, 999) is None


@needs_pypinyin
def test_same_sounding_spellings_are_one_question(idb, icfg):
    ids = seed(idb, icfg, [f"第{i}句提到兴和" for i in range(8)])
    q = ask(idb, icfg, "兴和", ids[0], "第0句提到兴和", ["星河"], priority=1.0)
    assert ask(idb, icfg, "幸和", ids[1], "第1句提到幸和", ["星合"], priority=0.5) == q         # same pinyin, other characters
    assert ask(idb, icfg, "兴和", ids[1], "第1句提到兴和", ["星河"]) == q                       # the same sentence again: not counted twice
    for k in range(2, 8):
        assert ask(idb, icfg, "兴和", ids[k], f"第{k}句提到兴和", priority=0.2) == q
    it = inbox.get(idb, q)
    assert it["occurrences"] == 8 and it["priority"] == pytest.approx(8.0)                    # 1.0 + one step per new sentence
    assert it["payload"]["candidates"] == ["星河", "星合"] and it["payload"]["utterance_id"] == ids[0]
    assert [e["utterance_id"] for e in it["payload"]["examples"]] == ids[1:6]                  # five examples at most
    assert inbox.counts(idb, icfg)["open"] == 1
    # an explicit key is used as given; other kinds without a key are never merged
    v1 = inbox.add(idb, "voice", title="这个声音是谁？", payload={"voice_id": 3}, dedupe_key="voice:3")
    assert inbox.add(idb, "voice", title="这个声音是谁？", payload={"voice_id": 3}, dedupe_key="voice:3") == v1
    c1 = inbox.add(idb, "conflict", title="哪个对？", payload={})
    assert inbox.add(idb, "conflict", title="哪个对？", payload={}) != c1
    assert inbox.counts(idb, icfg)["by_kind"] == {"term": 1, "voice": 1, "conflict": 2}
    assert [it["kind"] for it in inbox.list_items(idb, limit=1)] == ["term"]                   # highest priority first


def test_settled_things_are_not_asked_again(idb, icfg):
    a, b = seed(idb, icfg, ["兴和的排期", "又说兴和"])
    q = ask(idb, icfg, "兴和", a, "兴和的排期", ["星河"])
    inbox.dismiss(idb, q)                                                         # a plain dismissal may be asked again later
    q2 = ask(idb, icfg, "兴和", b, "又说兴和", ["星河"])
    assert q2 and q2 != q
    out = inbox.dismiss(idb, q2, never=True)
    assert out["status"] == "dismissed" and out["payload"]["never"] is True
    assert ask(idb, icfg, "兴和", b, "又说兴和", ["星河"]) == 0                    # "never ask again"
    # a spelling the lexicon knows is right is never asked about
    lexicon.upsert_term(idb, "压测")
    assert ask(idb, icfg, "压测", a, "压测报告") == 0
    assert inbox.counts(idb, icfg)["open"] == 0


def test_daily_allowance_queues_the_rest_by_priority(idb, cfg_factory, monkeypatch):
    cfg = cfg_factory({"inbox": {"daily_max": 2}, "memory": {"retrieval": "fts"}})
    ids = seed(idb, cfg, ["甲乙丙丁戊己"])
    monkeypatch.setattr(inbox, "_today", lambda: "2026-11-02")
    qs = [ask(idb, cfg, w, ids[0], "", priority=p) for w, p in (("星河", 1), ("压测", 1), ("埋点", 0.5), ("灰度", 3), ("岚姐", 2))]
    states = [inbox.get(idb, q)["status"] for q in qs]
    assert states == ["open", "open", "queued", "queued", "queued"]
    assert inbox.counts(idb, cfg) == {"open": 2, "by_kind": {"term": 2}, "queued": 3, "snoozed": 0}
    assert [it["id"] for it in inbox.list_items(idb, cfg=cfg)] == qs[:2] or {it["id"] for it in inbox.list_items(idb, cfg=cfg)} == set(qs[:2])
    # a waiting question still collects occurrences instead of being created twice
    (b,) = seed(idb, cfg, ["又说灰度"], day="2026-11-03")
    assert ask(idb, cfg, "灰度", b, "又说灰度") == qs[3] and inbox.get(idb, qs[3])["occurrences"] == 2
    # other kinds are not rationed
    assert inbox.get(idb, inbox.add(idb, "rule", title="其他 3 处也改吗？", payload={}, cfg=cfg))["status"] == "open"
    # the next day: two more become visible, the most pressing first
    monkeypatch.setattr(inbox, "_today", lambda: "2026-11-03")
    assert inbox.counts(idb, cfg)["open"] == 5 and inbox.counts(idb, cfg)["queued"] == 1
    assert [inbox.get(idb, q)["status"] for q in qs] == ["open", "open", "queued", "open", "open"]
    monkeypatch.setattr(inbox, "_today", lambda: "2026-11-04")
    assert inbox.counts(idb, cfg)["queued"] == 0
    # 0 = no limit
    free = cfg_factory({"inbox": {"daily_max": 0}, "memory": {"retrieval": "fts"}}, name="free.yaml")
    assert all(inbox.get(idb, ask(idb, free, w, ids[0]))["status"] == "open" for w in ("甲词", "乙词", "丙词", "丁词"))


def test_snooze_hides_a_question_until_its_time(idb, icfg):
    (a,) = seed(idb, icfg, ["兴和的排期"])
    q = ask(idb, icfg, "兴和", a, "兴和的排期", ["星河"])
    out = inbox.snooze(idb, q, days=2)
    assert out["status"] == "snoozed" and out["snooze_until"] > (datetime.now() + timedelta(days=1)).isoformat()
    assert inbox.counts(idb, icfg) == {"open": 0, "by_kind": {}, "queued": 0, "snoozed": 1} and inbox.list_items(idb) == []
    assert ask(idb, icfg, "兴和", a, "兴和的排期") == q                           # still the same question while it sleeps
    idb.execute("UPDATE inbox_items SET snooze_until=? WHERE id=?", ((datetime.now() - timedelta(minutes=1)).isoformat(timespec="seconds"), q))
    assert inbox.counts(idb, icfg)["open"] == 1 and inbox.get(idb, q)["status"] == "open" and inbox.get(idb, q)["snooze_until"] is None
    for bad in (0, -1, 400, "x"):
        with pytest.raises(ValueError):
            inbox.snooze(idb, q, days=bad)
    with pytest.raises(LookupError):
        inbox.snooze(idb, 999)


# ---------------------------------------------------------------------------------------------- answering a word question
@needs_pypinyin
def test_answering_a_word_teaches_the_lexicon_and_fixes_the_history(idb, icfg):
    ids = seed(idb, icfg, ["今天过一下兴和的排期", "兴和要上线了", "无关的一句", "兴和和兴和路"])
    lexicon.upsert_term(idb, "兴和路", kind="place")
    q = ask(idb, icfg, "兴和", ids[0], "今天过一下兴和的排期", ["星河", "新河"], extra={"start": 5, "end": 7})
    bus = EventBus()
    sub = bus.subscribe(since=0)
    res = inbox.answer(idb, icfg, q, choice=0, events=bus)
    assert (res["id"], res["kind"], res["status"], res["term"], res["alias"], res["scope"]) == (q, "term", "answered", "星河", "兴和", "auto")
    assert res["as_correct"] is False and res["created_term"] is True and res["created_alias"] is True
    assert sorted(res["changed"]) == [ids[0], ids[1], ids[3]]
    assert texts_of(idb) == ["今天过一下星河的排期", "星河要上线了", "无关的一句", "星河和兴和路"]
    # the lexicon: the entry (a hot word from now on) and the wrong spelling as its alias
    t = lexicon.get_term(idb, res["lexicon_id"])
    assert (t["term"], t["kind"], t["source"], t["hotword"]) == ("星河", "person", "answer", True)
    assert [(a["id"], a["alias"], a["scope"], a["source"]) for a in t["aliases"]] == [(res["alias_id"], "兴和", "auto", "answer")]
    assert "星河" in lexicon.Lexicon.load(idb).hotwords(10)
    # every change is a revision that names the question, and can be undone on its own
    h = revisions.history(idb, ids[1])
    assert len(h) == 1 and h[0]["source"] == "answer" and h[0]["detail"]["inbox_id"] == q
    revisions.undo(idb, icfg, ids[1])
    assert texts_of(idb)[1] == "兴和要上线了"
    # the question is closed and remembers the answer
    it = inbox.get(idb, q)
    assert it["status"] == "answered" and it["answered_at"] and it["lexicon_id"] == res["lexicon_id"]
    assert it["answer"]["choice"] == 0 and it["answer"]["result"]["alias_id"] == res["alias_id"]
    assert inbox.counts(idb, icfg)["open"] == 0
    evs = drain(sub)
    assert [e["type"] for e in evs].count("utterance") == 3 and evs[-1]["type"] == "inbox" and evs[-1]["open"] == 0
    assert {e["type"] for e in evs} == {"lexicon", "utterance", "inbox"}
    # and it holds for what is recorded later: the first pass rewrites the same mis-hearing
    assert lexicon.Lexicon.load(idb).apply("明天兴和发版")[0] == "明天星河发版"
    with pytest.raises(ValueError, match="已经处理过"):
        inbox.answer(idb, icfg, q, choice=0)


@needs_pypinyin
def test_answer_by_typed_text_or_candidate_text_and_only_this_sentence(idb, icfg):
    ids = seed(idb, icfg, ["兰姐说可以", "问一下兰姐", "牙测报告", "牙测排期"])
    q = ask(idb, icfg, "兰姐", ids[0], "兰姐说可以", ["蓝姐"])
    res = inbox.answer(idb, icfg, q, text=" 岚姐 ", apply_history=False)
    assert res["term"] == "岚姐" and res["changed"] == [ids[0]] and texts_of(idb)[:2] == ["岚姐说可以", "问一下兰姐"]
    assert lexicon.get_alias(idb, res["alias_id"])["scope"] == "auto"            # the rule still holds for the future
    q2 = ask(idb, icfg, "牙测", ids[2], "牙测报告", ["压测", "押测"], term_kind="term")
    res2 = inbox.answer(idb, icfg, q2, choice="压测")
    assert res2["term"] == "压测" and sorted(res2["changed"]) == [ids[2], ids[3]]
    assert lexicon.get_term(idb, res2["lexicon_id"])["kind"] == "term"


@needs_pypinyin
def test_a_common_word_alias_only_fixes_the_sentences_that_were_asked_about(idb, icfg):
    ids = seed(idb, icfg, ["新河的排期定了", "新河也要压测", "他住在新河边上"])
    q = ask(idb, icfg, "新河", ids[0], "新河的排期定了", ["星河"])
    assert ask(idb, icfg, "新河", ids[1], "新河也要压测") == q
    res = inbox.answer(idb, icfg, q, choice=0)
    assert res["scope"] == "hint" and sorted(res["changed"]) == ids[:2]
    assert texts_of(idb) == ["星河的排期定了", "星河也要压测", "他住在新河边上"]   # the everyday use of the word is left alone
    assert lexicon.Lexicon.load(idb).apply("新河边上") == ("新河边上", [])           # and it is not rewritten in new recordings


def test_the_text_is_right_makes_it_an_entry_and_is_never_asked_again(idb, icfg):
    a, b = seed(idb, icfg, ["兴和的排期", "又说兴和"])
    q = ask(idb, icfg, "兴和", a, "兴和的排期", ["星河"], term_kind="term")
    res = inbox.answer(idb, icfg, q, as_correct=True)
    assert res["as_correct"] is True and res["changed"] == [] and res["never"] is True and res["created_term"] is True
    assert lexicon.get_term(idb, res["lexicon_id"])["term"] == "兴和" and texts_of(idb) == ["兴和的排期", "又说兴和"]
    assert ask(idb, icfg, "兴和", b, "又说兴和", ["星河"]) == 0
    # choosing the span itself is the same answer
    q2 = ask(idb, icfg, "压测", a, "x", ["压测"])
    idb.execute("UPDATE inbox_items SET payload=? WHERE id=?", (json.dumps({"span": "压测", "candidates": ["压测", "押测"], "utterance_id": a}), q2))
    assert inbox.answer(idb, icfg, q2, choice=0)["as_correct"] is True


def test_numbers_are_corrected_in_place_but_never_become_vocabulary(idb, icfg):
    ids = seed(idb, icfg, ["目标是三千五，不是三千五百", "别处也有三千五"])
    q = ask(idb, icfg, "三千五", ids[0], "目标是三千五，不是三千五百", ["三千二"], term_kind="number", extra={"start": 3, "end": 6})
    res = inbox.answer(idb, icfg, q, choice=0)
    assert (res["lexicon_id"], res["alias_id"], res["scope"], res["changed"]) == (None, None, None, [ids[0]])
    assert texts_of(idb) == ["目标是三千二，不是三千五百", "别处也有三千五"] and lexicon.list_terms(idb) == []
    assert revisions.history(idb, ids[0])[0]["detail"] == {"edits": [{"from": "三千五", "to": "三千二", "reason": "回答", "start": 3, "end": 6}],
                                                          "inbox_id": q}
    # a name whose correction changes a digit is treated the same way (no automatic rule may change numbers)
    (c,) = seed(idb, icfg, ["型号是T三十"], day="2026-11-03")
    q2 = ask(idb, icfg, "T三十", c, "型号是T三十", ["T30"], term_kind="name")
    assert inbox.answer(idb, icfg, q2, choice=0)["alias_id"] is None and texts_of(idb)[2] == "型号是T30"
    # "it is right" for a number: closed for good, still no entry
    (d,) = seed(idb, icfg, ["一共四千二"], day="2026-11-04")
    q3 = ask(idb, icfg, "四千二", d, "一共四千二", ["四千"], term_kind="number")
    assert inbox.answer(idb, icfg, q3, as_correct=True)["lexicon_id"] is None and lexicon.list_terms(idb) == []
    assert ask(idb, icfg, "四千二", d, "一共四千二") == 0


def test_unusable_answers_leave_the_question_open(idb, icfg):
    (a,) = seed(idb, icfg, ["兴和的排期"])
    q = ask(idb, icfg, "兴和", a, "兴和的排期", ["星河"])
    for kw in ({}, {"choice": 5}, {"choice": -1}, {"text": "   "}, {"choice": True}):
        with pytest.raises(ValueError):
            inbox.answer(idb, icfg, q, **kw)
    assert inbox.get(idb, q)["status"] == "open" and lexicon.list_terms(idb) == [] and texts_of(idb) == ["兴和的排期"]
    with pytest.raises(LookupError):
        inbox.answer(idb, icfg, 999, choice=0)
    v = inbox.add(idb, "nobody-handles-this", title="？", payload={})
    with pytest.raises(ValueError, match="还不能在这里回答"):
        inbox.answer(idb, icfg, v, choice=0)
    assert inbox.get(idb, v)["status"] == "open"


def test_other_packages_register_their_own_handlers(idb, icfg, monkeypatch):
    seen: list[tuple[dict, dict]] = []

    def handler(db, cfg, item, answer):
        seen.append((item, answer))
        if answer["text"] == "不行":
            raise ValueError("这个回答不行")
        return {"named": answer["text"], "lexicon_id": None}

    monkeypatch.setitem(inbox._HANDLERS, "voice", handler)
    assert inbox.register_handler("voice", handler) is handler
    bus = EventBus()
    v = inbox.add(idb, "voice", title="这个声音出现了 3 天，是谁？", payload={"voice_id": 7}, dedupe_key="voice:7", speaker_id=None)
    with pytest.raises(ValueError, match="这个回答不行"):
        inbox.answer(idb, icfg, v, text="不行")
    assert inbox.get(idb, v)["status"] == "open"
    res = inbox.answer(idb, icfg, v, text="王敏", apply_history=False, events=bus)
    assert res == {"id": v, "kind": "voice", "status": "answered", "named": "王敏", "lexicon_id": None}
    item, answer = seen[-1]
    assert item["payload"] == {"voice_id": 7} and item["id"] == v
    assert answer == {"choice": None, "text": "王敏", "as_correct": False, "apply_history": False, "events": bus}
    stored = inbox.get(idb, v)["answer"]
    assert stored == {"choice": None, "text": "王敏", "as_correct": False, "apply_history": False, "result": {"named": "王敏", "lexicon_id": None}}


def test_a_handler_is_found_by_importing_its_owner_module(idb, icfg, monkeypatch):
    import sys
    import types

    def register_on_import(name, *a, **k):
        if name == "zz_owner_of_conflicts":
            mod = types.ModuleType(name)
            inbox.register_handler("conflict", lambda db, cfg, item, answer: {"kept": answer["choice"]})
            sys.modules[name] = mod
            return mod
        raise ImportError(name)

    monkeypatch.delitem(inbox._HANDLERS, "conflict", raising=False)
    monkeypatch.setitem(inbox.HANDLER_MODULES, "conflict", ["zz_missing_module", "zz_owner_of_conflicts"])
    monkeypatch.setattr(inbox.importlib, "import_module", register_on_import)
    c = inbox.add(idb, "conflict", title="哪个对？", payload={"a": 1, "b": 2})
    try:
        assert inbox.answer(idb, icfg, c, choice=1)["kept"] == 1                  # nobody had imported the owner before
        assert inbox.handler_for("nobody") is None
    finally:
        inbox._HANDLERS.pop("conflict", None)
        sys.modules.pop("zz_owner_of_conflicts", None)


# ---------------------------------------------------------------------------------------------- rule questions, reopen, expiry
@needs_pypinyin
def test_rule_question_yes_and_no(idb, icfg):
    ids = seed(idb, icfg, ["明天和兰姐开会", "兰姐说可以", "问一下兰姐"])
    revisions.update_text(idb, icfg, ids[0], "明天和岚姐开会", source="user")
    learned = revisions.learn_from_edit(idb, icfg, ids[0], "明天和兰姐开会", "明天和岚姐开会")
    q = learned["inbox_id"]
    with pytest.raises(ValueError):
        inbox.answer(idb, icfg, q, choice=7)
    res = inbox.answer(idb, icfg, q, choice="no")
    assert res["applied"] == 0 and res["changed"] == [] and texts_of(idb)[1:] == ["兰姐说可以", "问一下兰姐"]
    assert inbox.reopen(idb, icfg, q) == {"id": q, "status": "open", "reverted": 0}
    res = inbox.answer(idb, icfg, q, choice="yes")
    assert res["applied"] == 2 and sorted(res["changed"]) == ids[1:] and texts_of(idb) == ["明天和岚姐开会", "岚姐说可以", "问一下岚姐"]
    assert revisions.history(idb, ids[1])[0]["detail"]["inbox_id"] == q and revisions.history(idb, ids[1])[0]["source"] == "answer"
    # taking the answer back restores the sentences; the rule the hand edit made stays
    assert inbox.reopen(idb, icfg, q)["reverted"] == 2 and texts_of(idb)[1:] == ["兰姐说可以", "问一下兰姐"]
    assert lexicon.get_alias(idb, learned["alias_id"]) is not None
    # the rule is gone from the lexicon: the question cannot be carried out any more
    lexicon.delete_alias(idb, learned["alias_id"])
    with pytest.raises(ValueError, match="已经不在词库"):
        inbox.answer(idb, icfg, q)


@needs_pypinyin
def test_reopen_reverts_an_answer_completely(idb, icfg):
    ids = seed(idb, icfg, ["兴和的排期", "兴和要上线了"])
    q = ask(idb, icfg, "兴和", ids[0], "兴和的排期", ["星河"])
    res = inbox.answer(idb, icfg, q, choice=0)
    revisions.update_text(idb, icfg, ids[1], "星河要上线了，下周", source="user")           # edited again since
    bus = EventBus()
    out = inbox.reopen(idb, icfg, q, events=bus)
    assert out == {"id": q, "status": "open", "reverted": 2}
    assert texts_of(idb) == ["兴和的排期", "兴和要上线了，下周"]                            # the later hand edit survives
    assert lexicon.list_terms(idb, status="all") == [] and lexicon.get_alias(idb, res["alias_id"]) is None
    it = inbox.get(idb, q)
    assert (it["status"], it["answer"], it["answered_at"]) == ("open", None, None) and inbox.counts(idb, icfg)["open"] == 1
    # an entry that existed before the answer is kept
    lexicon.upsert_term(idb, "星河", source="glossary")
    inbox.answer(idb, icfg, q, choice=0)
    inbox.reopen(idb, icfg, q)
    assert [t["term"] for t in lexicon.list_terms(idb)] == ["星河"] and lexicon.list_terms(idb)[0]["aliases"] == []
    # a dismissed "never" question can be brought back, and asked again
    inbox.dismiss(idb, q, never=True)
    assert inbox.reopen(idb, icfg, q)["status"] == "open" and "never" not in inbox.get(idb, q)["payload"]
    assert inbox.reopen(idb, icfg, q) == {"id": q, "status": "open", "reverted": 0}
    with pytest.raises(LookupError):
        inbox.reopen(idb, icfg, 999)


def test_questions_follow_deleted_sentences(idb, icfg):
    a, b, c = seed(idb, icfg, ["兴和的排期", "又说兴和", "压测报告"])
    q = ask(idb, icfg, "兴和", a, "兴和的排期", ["星河"])
    ask(idb, icfg, "兴和", b, "又说兴和")
    q2 = ask(idb, icfg, "牙测", c, "牙测报告", ["压测"])
    assert inbox.expire_for_utterances(idb, [a, c]) == 1
    it = inbox.get(idb, q)
    assert it["status"] == "open" and it["utterance_id"] == b and it["payload"]["utterance_id"] == b and it["payload"]["examples"] == []
    assert inbox.get(idb, q2)["status"] == "expired" and inbox.counts(idb, icfg)["open"] == 1
    assert inbox.expire_for_utterances(idb, []) == 0


# ---------------------------------------------------------------------------------------------- the runtime's number
def test_runtime_status_reports_the_same_open_count(cfg_factory, tmp_path):
    from audio_note.runtime import Runtime

    rt = Runtime(cfg_factory({"memory": {"retrieval": "fts", "embedding_model": "hash"}}))
    try:
        assert rt.status()["inbox_open"] == 0 == inbox.counts(rt.db, rt.cfg)["open"]
        (a,) = seed(rt.db, rt.cfg, ["兴和的排期"])
        q = ask(rt.db, rt.cfg, "兴和", a, "兴和的排期", ["星河"])
        v = inbox.add(rt.db, "voice", title="是谁？", payload={})
        assert rt.status()["inbox_open"] == rt.inbox_open() == inbox.counts(rt.db, rt.cfg)["open"] == 2
        inbox.snooze(rt.db, q)
        assert rt.status()["inbox_open"] == 1
        # a question whose snooze ran out is counted again without anybody opening the inbox page first
        rt.db.execute("UPDATE inbox_items SET snooze_until='2020-01-01T00:00:00' WHERE id=?", (q,))
        assert rt.inbox_open() == 2 == inbox.counts(rt.db, rt.cfg)["open"] and rt.status_lite().get("inbox_open", 2) == 2
        inbox.dismiss(rt.db, v)
        assert rt.status()["inbox_open"] == 1
    finally:
        rt.shutdown()
        from audio_note.logging_setup import _remove_own_handlers

        _remove_own_handlers(logging.getLogger())
