"""Retrieval v2 (SPEC v1 section 7.4): a keyword side that contributes, lexicon aliases, parts of the day,
multi-view sentence vectors, time-aware question search; and the no-LLM levels of scripts/memory_bench.py.
No network, no model files: the embedder is the HashEmbedder."""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from audio_note.db import DB
from audio_note.memory import index, items, sessions, temporal, vectors
from audio_note.memory.embed import HashEmbedder, get_embedder
from audio_note.memory.index import tokenize
from audio_note.memory.retrieval import Retriever
from audio_note.speech import glossary_file

ROOT = Path(__file__).resolve().parent.parent
NOW = datetime(2026, 11, 5, 10, 0)

#: (day, time, speaker, text). Three days; noon and evening talk every day.
UTTS = [
    ("2026-11-02", "09:30:00", "周岚", "白鹭网关上周五合了主干，目前还剩二十三个问题。"),
    ("2026-11-02", "09:30:20", "陈默", "那灰度先定在这周五，比例百分之五。"),
    ("2026-11-02", "12:15:00", None, "我最近在看扫地机，朗月T30你听过没有？"),
    ("2026-11-02", "12:15:10", "我", "听过，好像是两千六百九十九？"),
    ("2026-11-02", "12:15:20", None, "对对对，到手两千六百九十九，还送一套滤网。"),
    ("2026-11-02", "20:40:00", "阿梨", "饭在锅里，自己热一下。"),
    ("2026-11-02", "20:40:10", "我", "好。今晚我就下单那台扫地机。"),
    ("2026-11-03", "09:30:00", "周岚", "昨天修了十六个，还剩七个。"),
    ("2026-11-03", "12:10:00", "周岚", "买了，极影X7，二十七寸，三千二百块。"),
    ("2026-11-03", "12:10:10", "我", "岚姐，那个显示器的支架晃不晃？"),
    ("2026-11-03", "20:30:00", "阿梨", "这个纪录片拍得真好。"),
    ("2026-11-03", "20:30:10", "我", "朗月T30今天到了，我放在门口了。"),
    ("2026-11-04", "09:30:00", "周岚", "今天下午两点压测，目标三千五。"),
    ("2026-11-04", "12:20:00", "我", "朗月出了T30 Pro，三千四百九十九。"),
    ("2026-11-04", "20:30:00", None, "打扰了啊，路过你们家，上来坐坐。"),
    ("2026-11-04", "20:30:10", "阿梨", "快坐，我在切水果。"),
    ("2026-11-04", "20:30:20", "我", "给你看看那台新买的机器，吸力不错。"),
]


@pytest.fixture
def rcfg(cfg_factory):
    return cfg_factory({"memory": {"embedding_model": "hash", "retrieval": "hybrid"}})


@pytest.fixture
def rdb(rcfg):
    d = DB(rcfg.db_path)
    lex = d.execute("INSERT INTO lexicon(term, kind) VALUES ('周岚', 'person')").lastrowid
    d.execute("INSERT INTO lexicon_aliases(lexicon_id, alias, kind) VALUES (?, '岚姐', 'aka')", (lex,))
    d.execute("INSERT INTO lexicon(term, kind) VALUES ('朗月', 'product'), ('极影', 'product'), ('白鹭', 'project')")
    off = d.execute("INSERT INTO lexicon(term, kind, status) VALUES ('停用词条', 'term', 'disabled')").lastrowid
    d.execute("INSERT INTO lexicon_aliases(lexicon_id, alias) VALUES (?, '停用别名')", (off,))
    index.register_from(rcfg, d)
    ids = []
    seg = None
    last = None
    for i, (day, t, spk, text) in enumerate(UTTS):
        if (day, t[:5]) != last:                       # one recording per conversation
            seg = d.add_segment(rcfg.audio_dir / f"{day}_{t[:5].replace(':', '')}.opus", datetime.fromisoformat(f"{day}T{t}"), 60.0)
            last = (day, t[:5])
        ids += d.add_utterances([{"segment_id": seg, "start_s": float(i), "end_s": float(i) + 3.0, "abs_start": f"{day}T{t}.000",
                                  "day": day, "speaker_name": spk, "text_raw": text, "text": text, "text_seg": tokenize(text)}])
    d.ids = ids  # type: ignore[attr-defined]
    return d


def texts(hits) -> list[str]:
    return [h.text for h in hits]


def uid(db, fragment: str) -> int:
    return int(db.one("SELECT id FROM utterances WHERE text LIKE ?", (f"%{fragment}%",))["id"])


# ---------------------------------------------------------------------------------------------- glossary / terms
def test_load_glossary_is_the_shared_parser(tmp_path):
    p = tmp_path / "glossary.txt"
    p.write_text("﻿# 注释\n灰度发布 # 上线方式\nC#\n\n灰度发布\n  朗月  \n", encoding="utf-8")
    assert index.load_glossary(p) == glossary_file.read_glossary(p) == ["灰度发布", "C#", "朗月"]
    assert index.load_glossary(None) == [] and index.load_glossary(tmp_path / "missing.txt") == []


def test_register_from_adds_lexicon_spellings_but_not_disabled_ones(rcfg, rdb):
    assert set(index.lexicon_terms(rdb)) == {"周岚", "岚姐", "朗月", "极影", "白鹭"}
    assert "岚姐" in tokenize("问问岚姐吧").split() and "朗月" in tokenize("朗月T30").split()


def test_query_terms_keep_content_and_drop_question_and_time_words(rdb):
    def terms(q: str) -> list[tuple[str, bool]]:
        return [(t.text, t.strong) for t in index.query_terms(q, rdb)]

    assert terms("朗月T30到手价多少钱？") == [("朗月", True), ("T30", True), ("到手价", False)] or \
        terms("朗月T30到手价多少钱？")[:2] == [("朗月", True), ("T30", True)]
    assert terms("昨天晚上聊了什么") == [] and terms("上周三下午说了啥") == []           # nothing but a time: browse, do not match
    assert terms("的吗呢") == [] and terms("") == []
    got = dict(terms("压测目标是不是三千五"))       # ("压测" is strong only if some earlier test registered it with jieba)
    assert got["三千五"] is True and "压测" in got and "目标" in got and "是不是" not in got
    # a lexicon entry is matched under every spelling it has (the nickname in the question, the name in the archive)
    got = index.query_terms("岚姐买的显示器", rdb)
    assert got[0].text == "岚姐" and got[0].strong and sorted(a[0] for a in got[0].alts) == ["周岚", "岚姐"]
    assert got[0].match() in ('("周岚" OR "岚姐")', '("岚姐" OR "周岚")') and [t.text for t in got[1:]] == ["显示器"]
    assert [t.text for t in index.query_terms("岚姐买的显示器")] == ["岚姐", "显示器"]            # without a db: no widening
    assert index.alias_groups(rdb, "停用别名说了什么") == []
    assert index.is_number_token("T30") and index.is_number_token("两千六百九十九") and not index.is_number_token("一")
    assert index.is_strong_token("QPS") and not index.is_strong_token("显示器")


def test_is_lookup_tells_search_words_from_questions(rdb):
    assert index.is_lookup("朗月T30") and index.is_lookup("极影") and index.is_lookup("压测 目标")
    assert not index.is_lookup("朗月T30到手价多少钱") and not index.is_lookup("昨天晚上聊了什么") and not index.is_lookup("")


def test_keyword_search_needs_two_terms_or_a_rare_one(rdb):
    kw = index.keyword_search
    # "朗月" and "T30" are each in three sentences (not rare: RARE_DF_MIN = 2): a sentence needs both of them,
    # or the one word of the question that IS rare ("到手", said once)
    got = kw(rdb, "朗月T30到手价多少钱", limit=10)
    assert len(got) == 4 and all(("朗月" in h.text and "T30" in h.text) or "到手" in h.text for h in got)
    assert all("朗月" in h.text and "T30" in h.text for h in kw(rdb, "朗月T30怎么样", limit=10))
    # one rare word is enough: "极影" occurs once
    assert texts(kw(rdb, "极影那个牌子怎么样")) == ["买了，极影X7，二十七寸，三千二百块。"]
    # a single everyday word that is in many sentences is not evidence
    rdb.add_utterances([{"segment_id": 1, "start_s": 90.0 + i, "end_s": 91.0 + i, "abs_start": f"2026-11-05T08:00:0{i}.000",
                         "day": "2026-11-05", "text_raw": f"今天第{i}次提到问题", "text": f"今天第{i}次提到问题",
                         "text_seg": tokenize(f"今天第{i}次提到问题")} for i in range(4)])
    assert kw(rdb, "那个问题后来怎么解决") == []
    assert len(kw(rdb, "问题")) == 5                                              # as the whole query it is all there is to match
    # the nickname finds what was said with the full name as speaker text or in the sentence
    assert "岚姐，那个显示器的支架晃不晃？" in texts(kw(rdb, "岚姐的显示器"))
    rdb.add_utterances([{"segment_id": 1, "start_s": 99.0, "end_s": 100.0, "abs_start": "2026-11-05T09:00:00.000", "day": "2026-11-05",
                         "text_raw": "周岚说显示器支架有点晃", "text": "周岚说显示器支架有点晃", "text_seg": tokenize("周岚说显示器支架有点晃")}])
    assert "周岚说显示器支架有点晃" in texts(kw(rdb, "岚姐的显示器"))
    # filters
    assert all(h.day == "2026-11-03" for h in kw(rdb, "朗月T30", day_from="2026-11-03", day_to="2026-11-03"))
    assert texts(kw(rdb, "朗月T30", hours=(18, 24))) == ["朗月T30今天到了，我放在门口了。"]
    assert kw(rdb, "昨天晚上聊了什么") == [] and kw(rdb, "") == []


def test_hour_filter_and_wraparound(rdb):
    sql, params = index._filters(None, None, None, (11, 14))
    rows = rdb.query("SELECT u.abs_start FROM utterances u WHERE 1=1" + sql, params)
    assert rows and all(11 <= int(r["abs_start"][11:13]) < 14 for r in rows)
    sql, params = index._filters("2026-11-02", "2026-11-02", None, (20, 10))          # 20:00 .. 10:00 across midnight
    hours = sorted({int(r["abs_start"][11:13]) for r in rdb.query("SELECT u.abs_start FROM utterances u WHERE 1=1" + sql, params)})
    assert hours == [9, 20]
    assert index._filters(None, None, None, None) == ("", [])
    assert all(h.abs_start[11:13] == "12" for h in index.like_search(rdb, "朗月", hours=(11, 14)))
    assert all(h.abs_start[11:13] == "12" for h in index.search(rdb, "朗月", hours=(11, 14)))


# ---------------------------------------------------------------------------------------------- time
def test_parts_of_the_day():
    assert temporal.parse_hours("昨天晚上聊了什么") == (18.0, 24.0) and temporal.parse_hours("上午开会说了啥") == (6.0, 12.0)
    assert temporal.parse_hours("中午吃的什么") == (11.0, 14.0) and temporal.parse_hours("下午的会") == (12.0, 18.5)
    assert temporal.parse_hours("今早几点到的") == (5.0, 11.0) and temporal.parse_hours("凌晨还在聊") == (0.0, 6.0)
    assert temporal.parse_hours("傍晚出门") == (17.0, 20.5) and temporal.parse_hours("深夜") == (21.0, 24.0)
    assert temporal.parse_hours("从上午到晚上") == (6.0, 12.0)                         # the first one named
    assert temporal.parse_hours("聊聊蛋糕") is None and temporal.parse_hours("") is None
    w = temporal.parse_when("昨天晚上在家聊了什么", NOW)
    assert (w.day_from, w.day_to, w.hours) == ("2026-11-04", "2026-11-04", (18.0, 24.0)) and w.label == "2026-11-04 18:00-24:00"
    assert temporal.parse_when("昨晚说了啥", NOW) == w                                  # 昨晚 = yesterday + evening
    w = temporal.parse_when("这三天中午分别吃的什么", NOW)
    assert (w.day_from, w.day_to, w.hours, w.label) == ("2026-11-03", "2026-11-05", (11.0, 14.0), "2026-11-03~2026-11-05 11:00-14:00")
    w = temporal.parse_when("下午说的那件事", NOW)
    assert (w.day_from, w.day_to, w.hours, w.label) == (None, None, (12.0, 18.5), "12:00-18:30")
    assert temporal.parse_when("上周三", NOW).hours is None and temporal.parse_when("上周三", NOW).label == "2026-10-28"
    assert temporal.parse_when("聊聊蛋糕", NOW) is None
    assert temporal.parse_range("昨天晚上聊了什么", NOW) == ("2026-11-04", "2026-11-04")   # the old function is unchanged


# ---------------------------------------------------------------------------------------------- vectors
def test_views_are_normalised_and_carried_by_the_embedder(rcfg):
    assert vectors.normalise_views(None) == ("plain",) and vectors.normalise_views(["window"]) == ("plain", "window")
    assert vectors.normalise_views("speaker+window") == ("plain", "window", "speaker")
    assert vectors.normalise_views(["PLAIN", "bogus", " Window "]) == ("plain", "window")
    emb = get_embedder(rcfg)
    assert emb.views == ("plain", "window") and emb.index_memory is True and vectors.views_of(emb) == ("plain", "window")
    rcfg["memory"]["vector_views"] = ["plain", "speaker"]
    assert get_embedder(rcfg).views == ("plain", "speaker")
    bare = HashEmbedder(16)
    assert vectors.views_of(bare) == ("plain",) and bare.index_memory is False


def test_view_text_and_neighbour_window(rdb):
    row = rdb.one("SELECT * FROM utterances WHERE id=?", (uid(rdb, "好像是两千六百九十九"),))
    prev, nxt = vectors._neighbours(rdb, row)
    assert prev["text"].startswith("我最近在看扫地机") and nxt["text"].startswith("对对对")
    assert vectors.view_text("plain", row) == "听过，好像是两千六百九十九？"
    assert vectors.view_text("speaker", row) == "我：听过，好像是两千六百九十九？"
    assert vectors.view_text("speaker", prev) == "未知：我最近在看扫地机，朗月T30你听过没有？"
    win = vectors.view_text("window", row, prev, nxt)
    assert win.split("\n") == ["我最近在看扫地机，朗月T30你听过没有？", "听过，好像是两千六百九十九？", "对对对，到手两千六百九十九，还送一套滤网。"]
    assert "我：" not in win                                        # no speaker names: they change when a voice is named
    # neighbours further away than WINDOW_MAX_GAP_S do not belong to the window
    first = rdb.one("SELECT * FROM utterances WHERE id=?", (uid(rdb, "饭在锅里"),))
    assert vectors._neighbours(rdb, first)[0] is None and vectors._neighbours(rdb, first)[1] is not None


def test_index_missing_fills_every_view_and_reports_storage(rdb):
    emb = HashEmbedder(32)
    n = len(UTTS)
    assert vectors.missing_count(rdb, emb.name) == n and vectors.missing_count(rdb, emb.name, ["plain", "window", "speaker"]) == 3 * n
    assert vectors.index_missing(rdb, emb) == n                                # a bare embedder: the plain view only
    assert vectors.missing_count(rdb, emb.name, ["window"]) == n and rdb.one("SELECT COUNT(*) AS c FROM memory_vectors")["c"] == 0
    assert vectors.index_missing(rdb, emb, views=["window", "speaker"]) == 0   # nothing new in plain, the views are filled
    assert vectors.missing_count(rdb, emb.name, ["plain", "window", "speaker"]) == 0
    st = vectors.storage(rdb)
    assert st["plain"] == st["window"] == st["speaker"] == {"rows": n, "bytes": n * 32 * 4}
    # vectors of another model count as missing and are replaced, not mixed
    other = HashEmbedder(16)
    other.name = "hash16"
    assert vectors.missing_count(rdb, other.name, ["window"]) == 2 * n
    vectors.index_missing(rdb, other, views=["window"], limit=5)
    assert vectors.missing_count(rdb, other.name, ["window"]) == 2 * n - 10
    assert vectors.search(rdb, emb, "朗月T30", view="window", limit=50) and len(vectors.search(rdb, other, "朗月T30", view="window", limit=50)) == 5


@pytest.mark.parametrize("use_sql", [True, False])
def test_view_search_paths_and_filters(rdb, monkeypatch, use_sql):
    if use_sql and not DB.vec_available:
        pytest.skip("sqlite-vec not loaded")
    monkeypatch.setattr(DB, "vec_available", use_sql)
    emb = HashEmbedder(128)
    vectors.index_missing(rdb, emb, views=["window", "speaker"])
    for view in ("plain", "window", "speaker"):
        hits = vectors.search(rdb, emb, "朗月T30扫地机", view=view, limit=4)
        assert len(hits) == 4 and all(h.source == "vector" for h in hits) and hits[0].score >= hits[-1].score
        assert all(h.abs_start[11:13] == "20" for h in vectors.search(rdb, emb, "朗月T30", view=view, hours=(18, 24)))
        assert all(h.day == "2026-11-04" for h in vectors.search(rdb, emb, "朗月", "2026-11-04", "2026-11-04", view=view))
    # the window view finds a sentence through what was said right before it
    win = vectors.search(rdb, emb, "朗月T30你听过没有", view="window", limit=3)
    assert "听过，好像是两千六百九十九？" in texts(win)
    qv = emb.embed_query("朗月T30")
    assert texts(vectors.search(rdb, emb, "别的话", view="plain", limit=3, qv=qv)) == texts(vectors.search(rdb, emb, "朗月T30", limit=3))


def test_a_new_or_edited_sentence_refreshes_the_windows_next_to_it(rdb):
    emb = HashEmbedder(64)
    emb.views = ("plain", "window")
    vectors.index_missing(rdb, emb)
    before_id = uid(rdb, "给你看看那台新买的机器")

    def win(u: int) -> bytes:
        return rdb.one("SELECT vec FROM memory_vectors WHERE kind='utt' AND view='window' AND ref_id=?", (u,))["vec"]

    old = win(before_id)
    far = win(uid(rdb, "饭在锅里"))
    new = rdb.add_utterances([{"segment_id": 1, "start_s": 50.0, "end_s": 52.0, "abs_start": "2026-11-04T20:30:30.000", "day": "2026-11-04",
                               "text_raw": "小满吓得躲到沙发底下", "text": "小满吓得躲到沙发底下", "text_seg": tokenize("小满吓得躲到沙发底下")}])[0]
    assert vectors.index_missing(rdb, emb) == 1
    assert win(before_id) != old and win(new) and win(uid(rdb, "饭在锅里")) == far       # only the neighbour was computed again
    # an edit: revisions drop the sentence's own vectors; indexing again also renews its neighbours' windows
    rdb.execute("UPDATE utterances SET text='小满躲到阳台去了' WHERE id=?", (new,))
    rdb.execute("DELETE FROM utterance_vectors WHERE utterance_id=?", (new,))
    rdb.execute("DELETE FROM memory_vectors WHERE kind='utt' AND ref_id=?", (new,))
    mid = win(before_id)
    assert vectors.index_missing(rdb, emb) == 1 and win(before_id) != mid


def test_item_and_session_vectors(rcfg, rdb):
    emb = get_embedder(rcfg)                                   # the profile's embedder: index_missing covers items / sessions too
    a, _ = items.upsert(rdb, rcfg, {"kind": "fact", "title": "朗月T30到手两千六百九十九", "content": "送一套滤网"})
    b, _ = items.upsert(rdb, rcfg, {"kind": "todo", "title": "给猫打疫苗"})
    gone, _ = items.upsert(rdb, rcfg, {"kind": "fact", "title": "删掉的"})
    items.reject(rdb, gone)
    sessions.sync(rcfg, rdb, now=NOW)
    sid = rdb.one("SELECT session_id FROM utterances WHERE id=?", (uid(rdb, "朗月T30你听过没有"),))["session_id"]
    rdb.execute("UPDATE sessions SET title='午饭', summary='聊了朗月T30扫地机的价格' WHERE id=?", (sid,))
    vectors.index_missing(rdb, emb)
    st = vectors.storage(rdb)
    assert st["item"]["rows"] == 2 and st["session"]["rows"] == 1 and st["window"]["rows"] == len(UTTS)
    assert vectors.item_text({"title": "标题", "content": "标题"}) == "标题" and vectors.item_text({"title": "标题", "content": "内容"}) == "标题：内容"
    near = vectors.nearest(rdb, emb, "item", "朗月T30多少钱")
    assert near[0][0] == a and near[0][1] > near[1][1] and {i for i, _ in near} == {a, b}
    assert vectors.nearest(rdb, emb, "item", "朗月T30多少钱", ids=[b]) == [(b, near[1][1])]
    assert vectors.nearest(rdb, emb, "session", "扫地机的价格")[0][0] == sid and vectors.nearest(rdb, emb, "item", "") == []
    assert vectors.index_items(rdb, emb) == 0 and vectors.index_sessions(rdb, emb) == 0
    assert vectors.put(rdb, "item", "m", [(a, np.ones(4))]) == 1 and st["item"]["bytes"] == 2 * 256 * 4


# ---------------------------------------------------------------------------------------------- Retriever
def test_hybrid_fuses_the_keyword_list_with_every_view(rcfg, rdb):
    r = Retriever(rcfg, rdb)
    assert (r.mode, r.keywords, r.views, r.fts_weight) == ("hybrid", "content", ("plain", "window"), 1.0)
    res = r.search("朗月T30到手价多少钱", limit=5)
    assert res.how == "hybrid" and any("fts" in h.source and "vector" in h.source for h in res.hits)
    assert vectors.missing_count(rdb, "hash", r.views) == 0                       # both views were indexed on the way
    assert r.loaded_embedder() is not None and Retriever(rcfg, rdb, mode="fts").loaded_embedder() is None
    assert Retriever(rcfg, rdb).loaded_embedder() is None                         # not loaded until something needs it
    # the keyword list can be changed or switched off; an unknown value falls back to the default
    for mode, has_fts in (("off", False), ("strict", False), ("loose", True), ("content", True)):
        rcfg["memory"]["hybrid_keywords"] = mode
        got = Retriever(rcfg, rdb).search("朗月T30到手价多少钱", limit=5)
        assert any("fts" in h.source for h in got.hits) is has_fts, mode
    rcfg["memory"]["hybrid_keywords"] = "nonsense"
    assert Retriever(rcfg, rdb).keywords == "content"
    rcfg["memory"]["hybrid_fts_strict"] = False                                   # the 0.3 switch in an old config.yaml
    assert Retriever(rcfg, rdb).keywords == "loose"


def test_a_large_view_backlog_is_filled_in_the_background(rcfg, rdb, monkeypatch):
    import audio_note.memory.retrieval as rmod

    monkeypatch.setattr(rmod, "SYNC_VIEW_BACKLOG", 5)
    r = Retriever(rcfg, rdb)
    res = r.search("朗月T30到手价多少钱", limit=5)             # answers at once from the plain view (indexed on the spot)
    assert res.hits and rdb.vector_count("hash") == len(UTTS)
    assert r._backfill is not None
    r._backfill.join(timeout=30)
    assert not r._backfill.is_alive() and vectors.missing_count(rdb, "hash", r.views) == 0
    first = r._backfill
    r.search("朗月T30", limit=5)                               # nothing missing any more: no second thread
    assert r._backfill is first
    # asking for everything (the benchmark, a reindex) never goes to the background
    rdb.execute("DELETE FROM memory_vectors WHERE kind='utt'")
    r2 = Retriever(rcfg, rdb)
    r2.ensure_indexed(limit=None)
    assert r2._backfill is None and vectors.missing_count(rdb, "hash", r2.views) == 0


def test_a_query_of_search_words_puts_the_exact_match_first(rcfg, rdb):
    r = Retriever(rcfg, rdb)
    assert r.search("极影", limit=5).hits[0].text == "买了，极影X7，二十七寸，三千二百块。"
    top = r.search("朗月T30", limit=3).hits
    assert all("朗月" in h.text and "T30" in h.text for h in top) and len(top) == 3
    rcfg["memory"]["hybrid_keywords"] = "off"                                     # the vector lists alone are not that sure
    assert Retriever(rcfg, rdb).search("极影", limit=5).how == "hybrid"
    # filters reach every list
    rcfg["memory"]["hybrid_keywords"] = "content"
    got = Retriever(rcfg, rdb).search("朗月T30", limit=10, hours=(18, 24)).hits
    assert got and all(h.abs_start[11:13] == "20" for h in got)
    assert all(h.speaker_name == "阿梨" for h in Retriever(rcfg, rdb).search("水果", speaker="阿梨").hits)


def test_search_question_reads_the_time(rcfg, rdb):
    r = Retriever(rcfg, rdb)
    plain = r.search_question("朗月T30多少钱", now=NOW, limit=5)
    assert plain.when is None and texts(plain.hits) == texts(r.search("朗月T30多少钱", limit=5).hits)
    # a time and nothing else: the window itself, in order
    res = r.search_question("昨天晚上聊了什么", now=NOW, limit=12)
    assert res.how == "browse" and res.how_label == "按时间浏览" and res.when.label == "2026-11-04 18:00-24:00"
    assert texts(res.hits) == ["打扰了啊，路过你们家，上来坐坐。", "快坐，我在切水果。", "给你看看那台新买的机器，吸力不错。"]
    assert len(r.browse(temporal.When("2026-11-02", "2026-11-04"), limit=5)) == 5          # evenly spread when it does not fit
    assert r.browse(temporal.When("2020-01-01", "2020-01-01")) == []
    # a time and a subject: a window that fits is shown whole and first, the rest fills up
    res = r.search_question("昨天晚上给朋友看了什么机器", now=NOW, limit=6)
    assert res.how == "hybrid" and {h.day for h in res.hits[:3]} == {"2026-11-04"} and all(h.abs_start[11:13] == "20" for h in res.hits[:3])
    assert len(res.hits) == 6 and len({h.id for h in res.hits}) == 6
    # a bigger window gets SCOPED_SHARE of the room; hits from outside hedge against a misread date
    res = r.search_question("这三天聊过扫地机吗", now=NOW, limit=4)
    inside = [h for h in res.hits if "2026-11-03" <= h.day <= "2026-11-05"]
    assert res.when.label == "2026-11-03~2026-11-05" and len(inside) >= 3 and len(res.hits) == 4
    # nothing in the window: the plain result, with the time still reported
    res = r.search_question("上周三聊了扫地机吗", now=NOW, limit=3)
    assert res.when.label == "2026-10-28" and len(res.hits) == 3


def test_prefetch_and_tools_use_it(rcfg, rdb):
    from audio_note.memory.chat import MemoryChat
    from audio_note.memory.tools import run_tool

    chat = MemoryChat(rcfg, rdb, None)
    chat.now = NOW
    items.upsert(rdb, rcfg, {"kind": "fact", "title": "朗月T30到手两千六百九十九", "content": "送一套滤网", "day": "2026-11-02"})
    text = chat.prefetch("前天中午岚姐买了什么")
    assert "问题里的日期按 2026-11-03 11:00-14:00 优先" in text
    assert "极影X7" in "\n".join(text.splitlines()[1:3])                   # the two sentences of that noon come first
    text = chat.prefetch("朗月T30多少钱")
    assert "相关记忆笔记" in text and "[fact] 朗月T30到手两千六百九十九 (2026-11-02): 送一套滤网" in text
    out = run_tool("search_transcripts", {"query": "极影"}, rdb, rcfg, chat.retriever)
    assert out.splitlines()[1].endswith("买了，极影X7，二十七寸，三千二百块。") and "混合检索" in out
    assert "朗月T30到手两千六百九十九" in run_tool("search_notes", {"query": "朗月T30"}, rdb, rcfg, chat.retriever)


def test_pipeline_embedder_fills_the_configured_views(rcfg, rdb):
    """What Processor._embed_new does: get_embedder(cfg) + vectors.index_missing(db, embedder)."""
    emb = get_embedder(rcfg)
    assert vectors.index_missing(rdb, emb) == len(UTTS)
    assert vectors.missing_count(rdb, emb.name, emb.views) == 0 and vectors.storage(rdb)["window"]["rows"] == len(UTTS)
    assert "speaker" not in vectors.storage(rdb)


# ---------------------------------------------------------------------------------------------- memory_bench
@pytest.fixture(scope="module")
def mb():
    spec = importlib.util.spec_from_file_location("memory_bench_v1", ROOT / "scripts" / "memory_bench.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["memory_bench_v1"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def story(mb, tmp_path, monkeypatch):
    """The benchmark story as a text-only profile, with the hash embedder (no model files)."""
    from audio_note.config import load_config

    monkeypatch.delenv("AUDIO_NOTE_CONFIG", raising=False)
    cfg = load_config(mb.build_story(tmp_path / "story"))
    cfg["memory"]["embedding_model"] = "hash"
    return cfg


def test_bench_story_dataset_and_helpers(mb, story, tmp_path):
    db = DB(story.db_path)
    assert db.one("SELECT COUNT(*) AS n FROM utterances")["n"] == 157 and db.one("SELECT COUNT(*) AS n FROM sessions")["n"] == 12
    assert db.one("SELECT COUNT(*) AS n FROM utterances WHERE speaker_name IS NULL")["n"] > 0          # the unenrolled voices
    assert db.one("SELECT alias FROM lexicon_aliases")["alias"] == "岚姐"
    assert mb.build_story(tmp_path / "story") and db.one("SELECT COUNT(*) AS n FROM utterances")["n"] == 157   # built once
    qs = mb._prepare(mb.STORY_QUESTIONS)
    assert len(qs) >= 25 and [q["n"] for q in mb._prepare(mb.STORY_QUESTIONS, {2, 5})] == [2, 5]
    for q in qs:                                                    # every piece of evidence exists in the story
        assert all(mb.resolve_evidence(db, q["evidence"])), q["q"]
    assert mb.first_rank([7, 8, 9], [{9}, {1}]) == 3 and mb.first_rank([7], [{9}]) is None and mb.first_rank([], []) is None
    assert mb.by_tag([{"tags": ["a", "b"], "ok": True}, {"tags": ["a"], "ok": False}, {"tags": [], "ok": True}],
                     lambda x: x["ok"]) == {"a": "1/2", "b": "1/1", "untagged": "1/1"}
    assert len(mb.story_gold_items()) == 67
    index.register_from(story, db)
    res = mb.run_retrieval(Retriever(story, db, mode="fts"), db, qs, [5, 10], mb.STORY_NOW)
    s = res["summary"]
    assert s["questions"] == 27 and 0 < s["q_ok@10"] <= 27 and 0 < s["mrr"] <= 1 and set(s["by_tag"]) >= {"entity", "number", "temporal"}
    look = mb.run_lookup(Retriever(story, db, mode="fts"), db, n=20)
    assert look["words"] == 20 and look["hit@1"] >= 18 and look["hit@1"] <= look["hit@3"] <= look["hit@10"]
    timed = mb.run_retrieval(Retriever(story, db), db, [q for q in qs if "temporal" in q["tags"]], [5, 10], mb.STORY_NOW, temporal=True)
    assert timed["summary"]["questions"] == 3 and timed["summary"]["q_ok@10"] >= 1     # (the hash embedder is a weak ranker)
    out = mb.bench_prefetch(story, db, qs, mb.STORY_NOW, k=12)["configured"]["summary"]
    assert out["groups"].endswith("/36") and int(out["groups"].split("/")[0]) >= 28 and out["avg_chars"] > 0
    db.close_all()


def test_bench_items_forget_and_sessions_levels_leave_the_profile_alone(mb, story, tmp_path):
    before = Path(story.db_path).stat().st_mtime_ns
    it = mb.bench_items(story, tmp_path / "work", mb.story_gold_items())
    assert it["played"] == 67 and it["distinct_items"] == 67
    assert (it["duplicates_after_replays"], it["tombstones_resurrected"], it["user_edits_overwritten"], it["merges_undone"]) == (0, 0, 0, 0)
    assert it["tested"] == {"tombstones": 5, "user_edits": 5, "merges": 4, "replays": 3} and set(it["near_duplicate_pairs"]) == {">=0.8", ">=0.85", ">=0.9"}
    fg = mb.bench_forget(story, tmp_path / "work")
    assert fg["leak_count"] == 0 and fg["leaks"] == {"tables": [], "files": []} and fg["dry_run_matches"] is True
    assert fg["report"]["utterances"] == 3 and fg["other_utterances_untouched"] is True and sum(fg["found_after"].values()) == 0
    assert sum(fg["found_before"].values()) >= 2
    ss = mb.bench_sessions(story, tmp_path / "work")
    assert ss["utterances"] == 157 and ss["sessions"] == 12 and ss["sessions_per_day"] == [4, 4, 4] and ss["unassigned_after"] == 0
    assert ss["ms_per_1000_utterances"] < 1000 and ss["idle_sync_ms_median"] < 50
    assert Path(story.db_path).stat().st_mtime_ns == before                          # everything ran on copies
    db = DB(story.db_path)
    assert db.one("SELECT COUNT(*) AS n FROM memory_items")["n"] == 0 and db.one("SELECT COUNT(*) AS n FROM utterances")["n"] == 157
    db.close_all()


def test_bench_main_refuses_a_profile_without_a_database(mb, tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "nothing.yaml"
    cfg.write_text(f"paths:\n  data_dir: {(tmp_path / 'no_data').as_posix()}\n", encoding="utf-8")
    monkeypatch.setenv("AUDIO_NOTE_CONFIG", str(cfg))
    assert mb.main(["retrieval"]) == 2 and "no database" in capsys.readouterr().err
    assert not (tmp_path / "no_data").exists()                                      # and creates nothing


def test_bench_chat_counts_api_calls_under_the_clients_key(mb, story):
    from types import SimpleNamespace

    class Client:
        def __init__(self):
            self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

        def chat(self, messages, **kw):
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += 100
            self.usage["completion_tokens"] += 10
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="首期十万", tool_calls=None))])

    db = DB(story.db_path)
    qs = mb._prepare(mb.STORY_QUESTIONS, {3})
    out = mb.bench_chat(story, db, Client(), qs, ["rag"], "fts", False, mb.STORY_NOW, False)
    s = out["rag/fts"]["summary"]
    assert (s["pass"], s["questions"], s["llm_calls"], s["api_calls"], s["prompt_tokens"]) == (1, 1, 1, 1, 100)
    assert s["by_tag"] == {"entity": "1/1", "update": "1/1"} and "requests" not in s
    db.close_all()
