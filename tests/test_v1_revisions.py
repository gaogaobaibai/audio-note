"""audio_note.speech.revisions: every text change recorded, reversible, and visible as marks (SPEC v1 section 6.3).

No network, no models. Names and sentences are invented.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
import pytest

from audio_note import inbox
from audio_note.db import DB
from audio_note.events import EventBus
from audio_note.speech import lexicon, pinyin, revisions

DAY = "2026-11-02"
needs_pypinyin = pytest.mark.skipif(not pinyin.available(), reason="pypinyin is not installed")


@pytest.fixture
def rcfg(cfg_factory):
    return cfg_factory({"memory": {"retrieval": "fts"}})


@pytest.fixture
def rdb(rcfg) -> DB:
    return DB(rcfg.db_path)


def seed(db: DB, cfg: Any, texts: list[str], day: str = DAY, raw: list[str] | None = None) -> list[int]:
    seg = db.add_segment(cfg.audio_dir / f"{day}-{len(texts)}-{texts[0][:2]}.wav", datetime.fromisoformat(f"{day}T09:00:00"), 60.0,
                         speech_s=30.0)
    db.set_segment_status(seg, "done")
    return db.add_utterances([{
        "segment_id": seg, "start_s": float(i), "end_s": float(i + 1), "abs_start": f"{day}T09:{i:02d}:00", "day": day,
        "cluster_label": 0, "speaker_id": None, "speaker_name": None, "speaker_score": None,
        "text_raw": (raw[i] if raw else t), "text": t, "text_seg": " ".join(t)} for i, t in enumerate(texts)])


def row(db: DB, uid: int) -> Any:
    return db.one("SELECT * FROM utterances WHERE id=?", (uid,))


def fts(db: DB, word: str) -> list[int]:
    return [int(r[0]) for r in db.query("SELECT rowid FROM utterances_fts WHERE utterances_fts MATCH ?", (f'"{word}"',))]


def drain(sub: Any) -> list[dict]:
    out = []
    while (ev := sub.get(timeout=0.01)) is not None:
        out.append(ev)
    return out


# ---------------------------------------------------------------------------------------------- update_text
def test_update_text_keeps_everything_that_hangs_off_the_text_consistent(rdb, rcfg):
    db, cfg = rdb, rcfg
    a, b = seed(db, cfg, ["兴和的排期定了", "别的话"])
    lexicon.upsert_term(db, "星河")                                             # the tokenizer must keep it whole
    # a session, vectors of both kinds
    db.execute("INSERT INTO sessions(id, day, started_at, ended_at) VALUES (5, ?, ?, ?)", (DAY, f"{DAY}T09:00:00", f"{DAY}T09:02:00"))
    db.execute("UPDATE utterances SET session_id=5")
    before_hashes = db.refresh_session_hashes(5)
    db.add_vectors("m", [(a, np.ones(4)), (b, np.ones(4))])
    for uid in (a, b):
        db.execute("INSERT INTO memory_vectors(kind, ref_id, view, model, dim, vec) VALUES ('utt', ?, 'window', 'm', 4, ?)",
                   (uid, np.ones(4, dtype=np.float32).tobytes()))
    db.execute("INSERT INTO memory_vectors(kind, ref_id, view, model, dim, vec) VALUES ('item', ?, '', 'm', 4, ?)",
               (a, np.ones(4, dtype=np.float32).tobytes()))
    bus = EventBus()
    sub = bus.subscribe(since=0)

    res = revisions.update_text(db, cfg, a, "  星河的排期定了\r\n", source="refine", detail={"edits": [{"from": "兴和", "to": "星河"}]},
                                events=bus)
    assert res == {"utterance_id": a, "revision_id": res["revision_id"], "old_text": "兴和的排期定了", "new_text": "星河的排期定了",
                   "source": "refine", "text_source": "refine", "day": DAY, "session_id": 5}
    u = row(db, a)
    assert (u["text"], u["text_raw"], u["text_source"]) == ("星河的排期定了", "兴和的排期定了", "refine") and u["edited_at"]
    # FTS follows through the triggers: the new word is found as one token, the old one is gone
    assert fts(db, "星河") == [a] and fts(db, "兴") == []
    # vectors of THIS sentence are dropped (re-embedded by the next indexing pass); nothing else is
    assert [int(r[0]) for r in db.query("SELECT utterance_id FROM utterance_vectors")] == [b]
    assert [(r["kind"], int(r["ref_id"])) for r in db.query("SELECT kind, ref_id FROM memory_vectors ORDER BY kind, ref_id")] == [
        ("item", a), ("utt", b)]
    # the session's text fingerprint moved, its raw fingerprint did not
    s = db.one("SELECT raw_hash, text_hash FROM sessions WHERE id=5")
    assert s["raw_hash"] == before_hashes[0] and s["text_hash"] != before_hashes[1] and s["text_hash"] == db.session_hashes(5)[1]
    # the day's Markdown was rewritten, and the page was told
    assert "星河的排期定了" in (cfg.transcripts_dir / f"{DAY}.md").read_text(encoding="utf-8")
    evs = drain(sub)
    assert [(e["type"], e["id"], e["change"], e["day"], e["source"]) for e in evs] == [("utterance", a, "text", DAY, "refine")]
    # the revision row
    (h,) = revisions.history(db, a)
    assert (h["id"], h["old_text"], h["new_text"], h["source"], h["undone_at"]) == (res["revision_id"], "兴和的排期定了", "星河的排期定了",
                                                                                "refine", None)
    assert h["detail"] == {"edits": [{"from": "兴和", "to": "星河"}]} and h["created_at"]


def test_update_text_unchanged_and_errors(rdb, rcfg):
    (a,) = seed(rdb, rcfg, ["一句话"])
    assert revisions.update_text(rdb, rcfg, a, " 一句话 ", source="user") is None       # same text: nothing written
    assert revisions.history(rdb, a) == [] and row(rdb, a)["text_source"] is None
    with pytest.raises(LookupError):
        revisions.update_text(rdb, rcfg, 999, "x", source="user")
    for bad in ("", "   ", None, 5):
        with pytest.raises(ValueError):
            revisions.update_text(rdb, rcfg, a, bad, source="user")
    with pytest.raises(ValueError):
        revisions.update_text(rdb, rcfg, a, "改", source="magic")
    with pytest.raises(ValueError):
        revisions.update_text(rdb, rcfg, a, "长" * (revisions.MAX_TEXT_CHARS + 1), source="user")
    assert row(rdb, a)["text"] == "一句话"


def test_batches_defer_the_markdown_rewrite(rdb, rcfg, monkeypatch):
    from audio_note.speech import pipeline

    a, b = seed(rdb, rcfg, ["第一句", "第二句"])
    (c,) = seed(rdb, rcfg, ["另一天"], day="2026-11-03")
    written: list[str] = []
    real = pipeline.write_day_markdown
    monkeypatch.setattr(pipeline, "write_day_markdown", lambda db, cfg, day: written.append(day) or real(db, cfg, day))
    for uid, t in ((a, "第一句改"), (b, "第二句改"), (c, "另一天改")):
        revisions.update_text(rdb, rcfg, uid, t, source="user", flush=False)
    assert written == [] and rdb.get_kv(revisions.KV_MD_DIRTY) == [DAY, "2026-11-03"]
    assert revisions.flush_markdown(rdb, rcfg) == [DAY, "2026-11-03"] and written == [DAY, "2026-11-03"]
    assert revisions.flush_markdown(rdb, rcfg) == [] and rdb.get_kv(revisions.KV_MD_DIRTY) is None
    # a failing writer never fails the edit: the database is the source of truth
    monkeypatch.setattr(pipeline, "write_day_markdown", lambda db, cfg, day: (_ for _ in ()).throw(OSError("disk full")))
    assert revisions.update_text(rdb, rcfg, a, "第一句再改", source="user")["new_text"] == "第一句再改"
    # without a config nothing can be written; the day stays noted for later
    monkeypatch.setattr(pipeline, "write_day_markdown", real)
    revisions.update_text(rdb, None, a, "第一句三改", source="user")
    assert rdb.get_kv(revisions.KV_MD_DIRTY) == [DAY]


# ---------------------------------------------------------------------------------------------- undo / history
def test_undo_takes_back_the_latest_change_and_is_itself_history(rdb, rcfg):
    (a,) = seed(rdb, rcfg, ["兴和的排期"])
    bus = EventBus()
    sub = bus.subscribe(since=0)
    aid = lexicon.add_alias(rdb, lexicon.upsert_term(rdb, "星河"), "兴和")
    r1 = revisions.update_text(rdb, rcfg, a, "星河的排期", source="lexicon", detail={"edits": [{"from": "兴和", "to": "星河", "alias_id": aid}]})
    r2 = revisions.update_text(rdb, rcfg, a, "星河的排期定了", source="user")
    out = revisions.undo(rdb, rcfg, a, events=bus)
    assert out == {"utterance_id": a, "undone": r2["revision_id"], "revision_id": out["revision_id"], "text": "星河的排期"}
    assert row(rdb, a)["text_source"] == "lexicon"                               # whoever made the change still in effect
    assert [e["source"] for e in drain(sub)] == ["undo"]
    out2 = revisions.undo(rdb, rcfg, a)
    assert out2["undone"] == r1["revision_id"] and row(rdb, a)["text"] == "兴和的排期" and row(rdb, a)["text_source"] is None
    h = revisions.history(rdb, a)
    assert [(x["source"], x["undone_at"] is not None) for x in h] == [("undo", False), ("undo", False), ("user", True), ("lexicon", True)]
    assert h[0]["detail"]["undo_of"] == r1["revision_id"] and h[0]["detail"]["edits"] == [{"from": "星河", "to": "兴和"}]
    with pytest.raises(ValueError, match="没有可以撤销"):
        revisions.undo(rdb, rcfg, a)
    with pytest.raises(ValueError, match="已经撤销"):
        revisions.undo(rdb, rcfg, a, r1["revision_id"])
    with pytest.raises(LookupError):
        revisions.undo(rdb, rcfg, a, 9999)
    with pytest.raises(LookupError):
        revisions.undo(rdb, rcfg, 9999)
    assert revisions.undone_aliases(rdb, [a]) == {a: {aid}}


def test_a_veto_is_about_one_rule_in_one_sentence(rdb, rcfg):
    a, b = seed(rdb, rcfg, ["兴和的排期", "兴和要上线"])
    tid = lexicon.upsert_term(rdb, "星河")
    aid = lexicon.add_alias(rdb, tid, "兴和")
    assert lexicon.recorrect(rdb, rcfg)["changed"] == 2
    ra, rb = revisions.history(rdb, a)[0]["id"], revisions.history(rdb, b)[0]["id"]
    revisions.undo(rdb, rcfg, a, ra)                              # the user: "not in this sentence"
    revisions.undo(rdb, rcfg, b, rb, veto=False)                  # the system taking a whole operation back
    assert revisions.undone_aliases(rdb, [a, b]) == {a: {aid}}
    assert revisions.history(rdb, b)[1]["detail"]["veto"] is False and "veto" not in revisions.history(rdb, a)[1]["detail"]
    assert lexicon.recorrect(rdb, rcfg)["changed"] == 1 and row(rdb, a)["text"] == "兴和的排期" and row(rdb, b)["text"] == "星河要上线"
    # the alias is deleted and another one gets its id (SQLite hands ids out again): the old veto is not about it
    lexicon.delete_alias(rdb, aid)
    other = lexicon.add_alias(rdb, lexicon.upsert_term(rdb, "压测"), "牙测")
    assert other == aid and revisions.undone_aliases(rdb, [a, b]) == {}


def test_an_older_change_can_be_undone_alone_while_its_edit_is_still_there(rdb, rcfg):
    (a,) = seed(rdb, rcfg, ["兴和的排期，兰姐说的"])
    r1 = revisions.update_text(rdb, rcfg, a, "星河的排期，兰姐说的", source="lexicon",
                               detail={"edits": [{"from": "兴和", "to": "星河", "start": 0, "end": 2}]})
    r2 = revisions.update_text(rdb, rcfg, a, "星河的排期，岚姐说的", source="answer",
                               detail={"edits": [{"from": "兰姐", "to": "岚姐", "start": 6, "end": 8}]})
    out = revisions.undo(rdb, rcfg, a, r1["revision_id"])                        # the later change stays
    assert out["text"] == "兴和的排期，岚姐说的" and row(rdb, a)["text_source"] == "answer"
    # undoing the undo puts the first change back in effect (and it can be undone again later)
    redo = revisions.undo(rdb, rcfg, a, out["revision_id"])
    assert redo["text"] == "星河的排期，岚姐说的" and row(rdb, a)["text_source"] == "answer"
    by_id = {h["id"]: h for h in revisions.history(rdb, a)}
    assert by_id[r1["revision_id"]]["undone_at"] is None and by_id[out["revision_id"]]["undone_at"] is not None
    assert revisions.undone_aliases(rdb, [a]) == {}
    # a change whose words were typed over since cannot be undone on its own
    revisions.update_text(rdb, rcfg, a, "排期是岚姐说的", source="user")
    with pytest.raises(ValueError, match="先撤销后面的修改"):
        revisions.undo(rdb, rcfg, a, r1["revision_id"])
    assert r2 and row(rdb, a)["text"] == "排期是岚姐说的"


def test_undo_of_a_change_that_was_typed_back_by_hand_just_closes_it(rdb, rcfg):
    (a,) = seed(rdb, rcfg, ["原文"])
    r1 = revisions.update_text(rdb, rcfg, a, "改文", source="llm", detail={"edits": [{"from": "原", "to": "改"}]})
    rdb.execute("UPDATE utterances SET text='原文' WHERE id=?", (a,))            # reverted behind our back
    with pytest.raises(ValueError):
        revisions.undo(rdb, rcfg, a, r1["revision_id"])                          # its edit is gone from the text
    assert revisions.history(rdb, a)[0]["undone_at"] is None


# ---------------------------------------------------------------------------------------------- marks
def test_marks_show_what_was_fixed_with_the_word_and_who(rdb, rcfg):
    a, b, c, d = seed(rdb, rcfg, ["兴和的排期，兰姐说的", "没改过", "旧的改法。", "标点，变了"],
                      raw=["兴和的排期，兰姐说的", "没改过", "就的改法。", "标点。变了"])
    revisions.update_text(rdb, rcfg, a, "星河的排期，兰姐说的", source="lexicon",
                          detail={"edits": [{"from": "兴和", "to": "星河", "start": 0, "end": 2, "alias_id": 1}]})
    revisions.update_text(rdb, rcfg, a, "星河的排期，岚姐说的", source="user",
                          detail={"edits": [{"from": "兰姐", "to": "岚姐", "start": 6, "end": 8}]})
    rows = rdb.query("SELECT * FROM utterances ORDER BY id")
    m = revisions.marks(rdb, rows)
    assert m[a] == [{"start": 0, "end": 2, "kind": "fixed", "from": "兴和", "source": "lexicon"},
                    {"start": 6, "end": 8, "kind": "fixed", "from": "兰姐", "source": "user"}]     # whole words, not the changed character
    assert b not in m
    # a sentence corrected before revisions existed: the bare character difference, attributed to the old corrector
    assert m[c] == [{"start": 0, "end": 1, "kind": "fixed", "from": "就", "source": "llm"}]
    assert d not in m                                                            # punctuation only: not worth a mark
    # an undone change is no mark any more
    revisions.undo(rdb, rcfg, a)
    m = revisions.marks(rdb, rdb.query("SELECT * FROM utterances ORDER BY id"))
    assert m[a] == [{"start": 0, "end": 2, "kind": "fixed", "from": "兴和", "source": "lexicon"}]
    # rows without text_raw (a narrow SELECT) simply get no fixed marks
    assert revisions.marks(rdb, rdb.query("SELECT id, text FROM utterances")) == {}
    assert revisions.marks(rdb, []) == {}


def test_marks_locate_repeated_words_by_the_recorded_position(rdb, rcfg):
    (a,) = seed(rdb, rcfg, ["星河说兴和和星河"])
    revisions.update_text(rdb, rcfg, a, "星河说星河和星河", source="lexicon", detail={"edits": [{"from": "兴和", "to": "星河", "start": 3, "end": 5}]})
    assert revisions.marks(rdb, rdb.query("SELECT * FROM utterances"))[a] == [
        {"start": 3, "end": 5, "kind": "fixed", "from": "兴和", "source": "lexicon"}]


def test_marks_show_open_questions_and_they_win_over_fixed_marks(rdb, rcfg):
    a, b, c = seed(rdb, rcfg, ["兴和的排期和兴和的预算", "也提到兴和", "三千五的目标"], raw=["兴和的排期和兴何的预算", "也提到兴和", "三千五的目标"])
    q = inbox.add(rdb, "term", title="「兴和」是哪个词？", cfg=rcfg, payload={
        "span": "兴和", "sentence": "兴和的排期和兴和的预算", "utterance_id": a, "start": 6, "end": 8, "candidates": ["星河", "兴和"],
        "term_kind": "name"})
    n = inbox.add(rdb, "term", title="「三千五」对吗？", cfg=rcfg, payload={
        "span": "三千五", "sentence": "三千五的目标", "utterance_id": c, "candidates": ["三千二"], "term_kind": "number"})
    (d,) = seed(rdb, rcfg, ["别处的三千五"], day="2026-11-03")
    m = revisions.marks(rdb, rdb.query("SELECT * FROM utterances ORDER BY id"))
    ask = {"kind": "uncertain", "inbox_id": q, "candidates": ["星河"]}
    # in the evidence sentence the occurrence that was asked about comes first; the legacy "fixed" mark under it gives way
    assert m[a] == [{"start": 0, "end": 2, **ask}, {"start": 6, "end": 8, **ask}]
    assert m[b] == [{"start": 3, "end": 5, **ask}]                               # a name: marked wherever it is spelled like this
    assert m[c] == [{"start": 0, "end": 3, "kind": "uncertain", "inbox_id": n, "candidates": ["三千二"]}]
    assert d not in m                                                            # a number: only where it was asked
    inbox.dismiss(rdb, q)
    m = revisions.marks(rdb, rdb.query("SELECT * FROM utterances ORDER BY id"))
    assert b not in m and m[a] == [{"start": 7, "end": 8, "kind": "fixed", "from": "何", "source": "llm"}]


def test_marks_are_sorted_disjoint_and_capped(rdb, rcfg):
    raw = "".join(f"甲{i % 10}，" for i in range(40))
    new = raw.replace("甲", "乙")
    (a,) = seed(rdb, rcfg, [new], raw=[raw])
    ms = revisions.marks(rdb, rdb.query("SELECT * FROM utterances"))[a]
    assert len(ms) == revisions.MAX_MARKS
    assert all(x["end"] > x["start"] for x in ms) and all(p["end"] <= n["start"] for p, n in zip(ms, ms[1:]))


def test_the_transcript_serializer_emits_the_marks(rdb, rcfg):
    from audio_note.ui.serializers import utt_dicts

    a, b = seed(rdb, rcfg, ["星河的排期", "没改"], raw=["兴和的排期", "没改"])
    out = utt_dicts(rdb, rdb.query("SELECT * FROM utterances ORDER BY id"), "未知")
    assert out[0]["marks"] == [{"start": 0, "end": 2, "kind": "fixed", "from": "兴和", "source": "llm"}] and out[1]["marks"] == []
    assert out[0]["text_raw"] == "兴和的排期" and "text_source" in out[0]


# ---------------------------------------------------------------------------------------------- diff / learning
def test_diff_edits():
    assert revisions.diff_edits("一样", "一样") == []
    assert revisions.diff_edits("兴和的排期", "星河的排期") == [{"from": "兴和", "to": "星河", "pos": 0, "start": 0, "end": 2}]
    assert revisions.diff_edits("明天和兰姐开会", "明天和岚姐开会") == [{"from": "兰", "to": "岚", "pos": 3, "start": 3, "end": 4}]
    two = revisions.diff_edits("甲说乙好", "丙说丁好")
    assert [(e["from"], e["to"], e["pos"], e["start"], e["end"]) for e in two] == [("甲", "丙", 0, 0, 1), ("乙", "丁", 2, 2, 3)]
    assert revisions.diff_edits("好的", "好的。") == [{"from": "", "to": "。", "pos": 2, "start": 2, "end": 3}]
    assert revisions.diff_edits("删掉这里", "删这里") == [{"from": "掉", "to": "", "pos": 1, "start": 1, "end": 1}]
    big_a, big_b = "字" * 2500 + "甲" + "尾" * 10, "字" * 2500 + "乙丙" + "尾" * 10
    assert revisions.diff_edits(big_a, big_b) == [{"from": "甲", "to": "乙丙", "pos": 2500, "start": 2500, "end": 2502}]


@pytest.mark.parametrize("old,new,expected", [
    ("明天和兰姐开会", "明天和岚姐开会", ("兰姐", "岚姐")),            # a form of address is kept whole, the 和 before it is not
    ("牙测报告出来了", "压测报告出来了", ("牙测", "压测")),            # an unknown word: its single characters are joined
    ("兴和计划下周上线", "星河计划下周上线", ("兴和", "星河")),
    ("阿离说周末去", "阿梨说周末去", ("阿离", "阿梨")),
    ("这个cbn的预算", "这个CDN的预算", ("cbn", "CDN")),
    ("我们明天在见", "我们明天再见", ("在见", "再见")),
    ("好的", "好的。", None),                                           # only added something
    ("甲说乙好", "丙说丁好", None),                                     # two places changed: a rewrite
    ("没变", "没变", None),
    ("一二三四五六七八九十十一十二十三", "完全不同的另外一句很长的话在这里呀", None),   # too long to be a word
])
def test_edit_pair_widens_the_change_to_a_word(old, new, expected):
    assert revisions.edit_pair(old, new) == expected


def test_edit_pair_prefers_a_known_lexicon_spelling(rdb):
    lexicon.upsert_term(rdb, "星河计划")
    lex = lexicon.Lexicon.load(rdb)
    assert revisions.edit_pair("兴和计划下周上线", "星河计划下周上线", lex) == ("兴和计划", "星河计划")


@needs_pypinyin
def test_learn_from_edit_makes_a_rule_and_asks_about_the_other_places(rdb, rcfg):
    ids = seed(rdb, rcfg, ["明天和兰姐开会", "兰姐说可以", "问一下兰姐", "无关的话"])
    old, new = "明天和兰姐开会", "明天和岚姐开会"
    revisions.update_text(rdb, rcfg, ids[0], new, source="user")
    bus = EventBus()
    learned = revisions.learn_from_edit(rdb, rcfg, ids[0], old, new, events=bus)
    assert learned["term"] == "岚姐" and learned["alias"] == "兰姐" and learned["scope"] == "auto" and learned["created_term"] is True
    assert learned["other_matches"] == 2 and learned["inbox_id"]
    t = lexicon.get_term(rdb, learned["lexicon_id"])
    assert (t["kind"], t["source"], t["status"]) == ("person", "edit", "active")
    assert [(a["alias"], a["source"], a["kind"]) for a in t["aliases"]] == [("兰姐", "edit", "mishear")]
    # nothing else was rewritten: the question decides about the history
    assert [r["text"] for r in rdb.query("SELECT text FROM utterances ORDER BY id")][1:3] == ["兰姐说可以", "问一下兰姐"]
    item = inbox.get(rdb, learned["inbox_id"])
    assert item["kind"] == "rule" and item["title"] == "其他 2 处「兰姐」也改成「岚姐」吗？" and item["dedupe_key"] == f"alias:{learned['alias_id']}"
    assert item["payload"]["count"] == 2 and {e["utterance_id"] for e in item["payload"]["examples"]} == {ids[1], ids[2]}
    assert item["lexicon_id"] == learned["lexicon_id"] and item["utterance_id"] == ids[0]
    # the same edit elsewhere: the rule exists, no second question
    again = revisions.learn_from_edit(rdb, rcfg, ids[1], "兰姐说可以", "岚姐说可以")
    assert again["alias_id"] == learned["alias_id"] and again["created_term"] is False and again["inbox_id"] == learned["inbox_id"]
    assert len(inbox.list_items(rdb, kind="rule")) == 1


@needs_pypinyin
def test_learn_from_edit_only_learns_what_looks_like_a_mis_heard_name_or_term(rdb, rcfg):
    (a,) = seed(rdb, rcfg, ["占位"])
    learn = lambda old, new, **kw: revisions.learn_from_edit(rdb, rcfg, a, old, new, **kw)       # noqa: E731
    assert learn("我们明天在见", "我们明天再见") is None                         # an everyday word, not vocabulary
    assert learn("今天天气不错", "今天天气很好") is None                         # reworded, does not sound alike
    assert learn("老钱来了", "老秦来了") is None                                 # another person, not a mis-hearing
    assert learn("好的", "好的。") is None
    assert lexicon.list_terms(rdb) == []
    # a known entry replaced by another known entry: the user meant a different word
    lexicon.upsert_term(rdb, "兴和")
    lexicon.upsert_term(rdb, "星河")
    assert learn("兴和计划下周上线", "星河计划下周上线") is None
    # "remember" overrides the judgement (the alias still gets a safe scope)
    got = learn("我们明天在见", "我们明天再见", remember=True)
    assert got["term"] == "再见" and got["alias"] == "在见" and got["other_matches"] == 0 and got["inbox_id"] is None
    got = learn("新河计划下周上线", "星河计划下周上线", remember=True)
    assert got["term"] == "星河" and got["alias"] == "新河" and got["scope"] == "hint" and got["created_term"] is False
    # a dictionary-unknown term that sounds the same is learned without being asked
    got = learn("牙测报告出来了", "压测报告出来了")
    assert (got["term"], got["alias"], got["scope"]) == ("压测", "牙测", "auto")
    assert json.dumps(got, ensure_ascii=False)                                   # plain JSON for the API
