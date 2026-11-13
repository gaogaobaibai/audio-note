"""memory.items: memory items with provenance, state and tombstones (SPEC v1 section 7.2), the paths that used
to write / read memory_notes, and the HTTP / CLI surface (memory items, todos, people, sessions, forget)."""
from __future__ import annotations

import io
import json
import logging
from datetime import date, datetime

import pytest
from fastapi.testclient import TestClient
from rich.console import Console

from audio_note import cli
from audio_note.analysis.daily import DailyAnalyzer
from audio_note.analysis.relations import person_notes
from audio_note.db import DB, copy_notes_to_items, memory_fingerprint
from audio_note.memory import index, items, sessions
from audio_note.memory.embed import HashEmbedder
from audio_note.memory.tools import run_tool
from audio_note.runtime import Runtime
from audio_note.ui.app import create_app

DAY = "2026-11-03"


@pytest.fixture
def mcfg(cfg_factory):
    return cfg_factory({"memory": {"retrieval": "fts", "embedding_model": "hash"}})


@pytest.fixture
def mdb(mcfg):
    return DB(mcfg.db_path)


def add(db, cfg, kind, title, **kw):
    source = kw.pop("source", "auto")
    embedder = kw.pop("embedder", None)
    return items.upsert(db, cfg, {"kind": kind, "title": title, **kw}, source=source, embedder=embedder)


def utts(db: DB, rows: list[tuple[str, str | None, str]], day: str = DAY, speaker_ids: dict | None = None) -> list[int]:
    seg = db.add_segment(f"{day}-{len(rows)}-{rows[0][0]}.opus", datetime.fromisoformat(f"{day}T{rows[0][0]}"), 600.0)
    return db.add_utterances([{
        "segment_id": seg, "start_s": float(i), "end_s": float(i) + 2.0, "abs_start": f"{day}T{t}.000", "day": day,
        "speaker_name": spk, "speaker_id": (speaker_ids or {}).get(spk), "text_raw": text, "text": text,
        "text_seg": index.tokenize(text)} for i, (t, spk, text) in enumerate(rows)])


# ---------------------------------------------------------------------------------------------- identity
def test_fingerprint_is_the_migration_fingerprint():
    assert items.fingerprint("todo", "买牛奶") == memory_fingerprint("todo", "买牛奶")
    assert items.fingerprint("todo", " 买 牛奶！") == items.fingerprint("todo", "买牛奶")          # spaces / punctuation
    assert items.fingerprint("fact", "Uses QPS") == items.fingerprint("FACT", "uses  qps.")       # case
    assert items.fingerprint("todo", "买牛奶") != items.fingerprint("decision", "买牛奶")           # the kind is part of it
    assert len(items.fingerprint("todo", "买牛奶")) == 16


def test_upsert_adds_then_merges_by_fingerprint(mcfg, mdb):
    i1, a1 = add(mdb, mcfg, "fact", "网关上周五合入主干", content="还剩二十三个问题", evidence=[1, 2], day="2026-11-02",
                 subject="网关", importance=4, confidence=0.8, session_id=7)
    i2, a2 = add(mdb, mcfg, "fact", "网关 上周五合入主干。", content="还剩七个问题", evidence=[2, 3], day="2026-11-03")
    assert (a1, a2) == ("added", "updated") and i1 == i2
    it = items.get(mdb, i1)
    assert it["title"] == "网关上周五合入主干" and it["content"] == "还剩七个问题"          # the newer wording wins
    assert it["evidence"] == [1, 2, 3] and it["day"] == "2026-11-03" and it["valid_from"] == "2026-11-02"
    assert it["subject"] == "网关" and it["importance"] == 0.75 and it["confidence"] == 0.8 and it["session_id"] == 7
    assert it["status"] == "active" and it["source"] == "auto" and it["pinned"] is False
    assert mdb.one("SELECT COUNT(*) AS n FROM memory_items")["n"] == 1
    # an unknown kind becomes a note; a missing title is an error
    assert items.get(mdb, add(mdb, mcfg, "whatever", "杂项")[0])["kind"] == "note"
    with pytest.raises(ValueError, match="标题"):
        add(mdb, mcfg, "fact", "   ")


def test_tombstones_are_never_resurrected(mcfg, mdb):
    i, _ = add(mdb, mcfg, "todo", "预约会议室", content="周三之前")
    items.reject(mdb, i)
    assert add(mdb, mcfg, "todo", "预约会议室") == (i, "skipped")
    assert add(mdb, mcfg, "todo", "预约 会议室！", source="digest", content="换个写法") == (i, "skipped")
    row = items.get(mdb, i)
    assert row["status"] == "rejected" and row["content"] == "周三之前"                        # untouched
    assert mdb.one("SELECT COUNT(*) AS n FROM memory_items")["n"] == 1
    assert items.search(mdb, mcfg, "会议室") == [] and items.list_items(mdb, status="all") == []
    assert [r["id"] for r in items.list_items(mdb, status="rejected")] == [i]
    # the user may bring it back, by restoring it or by writing it down again
    assert items.restore(mdb, i)["status"] == "active" and [r["id"] for r in items.search(mdb, mcfg, "会议室")] == [i]
    items.reject(mdb, i)
    assert add(mdb, mcfg, "todo", "预约会议室", source="user", content="我自己加回来的") == (i, "updated")
    assert items.get(mdb, i)["status"] == "active" and items.get(mdb, i)["source"] == "user"


def test_user_edits_and_pins_are_protected_from_automatic_writers(mcfg, mdb):
    i, _ = add(mdb, mcfg, "fact", "首期付八万", content="评审会上定的", evidence=[1], day="2026-11-02")
    items.update(mdb, i, content="我核对过：首期八万")
    assert add(mdb, mcfg, "fact", "首期付八万", content="模型的新说法", evidence=[1, 5], day="2026-11-04") == (i, "updated")
    it = items.get(mdb, i)
    assert it["content"] == "我核对过：首期八万" and it["source"] == "user"                     # the text is the user's
    assert it["evidence"] == [1, 5] and it["day"] == "2026-11-04" and it["valid_from"] == "2026-11-02"   # only evidence / last seen
    assert add(mdb, mcfg, "fact", "首期付八万", content="又一个说法", evidence=[5], day="2026-11-04") == (i, "skipped")
    p, _ = add(mdb, mcfg, "preference", "咖啡不加糖", content="原话")
    mdb.execute("UPDATE memory_items SET pinned=1 WHERE id=?", (p,))
    add(mdb, mcfg, "preference", "咖啡不加糖", content="被覆盖？")
    assert items.get(mdb, p)["content"] == "原话"


def test_person_items_keep_one_dated_line_per_day(mcfg, mdb):
    i, _ = add(mdb, mcfg, "person", "周岚", content="2026-11-02：负责网关", day="2026-11-02")
    add(mdb, mcfg, "person", "周岚", content="2026-11-03：买了显示器", day="2026-11-03")
    add(mdb, mcfg, "person", "周岚", content="2026-11-03：买了显示器，在写 OKR", day="2026-11-03")      # the day again: replaced
    assert items.get(mdb, i)["content"].split("\n") == ["2026-11-02：负责网关", "2026-11-03：买了显示器，在写 OKR"]
    for d in range(4, 20):
        add(mdb, mcfg, "person", "周岚", content=f"2026-11-{d:02d}：第{d}天", day=f"2026-11-{d:02d}")
    lines = items.get(mdb, i)["content"].split("\n")
    assert len(lines) == items.PERSON_LINES_MAX and lines[-1] == "2026-11-19：第19天" and lines[0].startswith("2026-11-08")


def test_done_todos_stay_done_and_status_can_be_reported(mcfg, mdb):
    i, _ = add(mdb, mcfg, "todo", "发会议纪要", owner="我", due="2026-11-03")
    items.update(mdb, i, status="done")
    mdb.execute("UPDATE memory_items SET source='auto' WHERE id=?", (i,))          # as if the consolidation closed it
    add(mdb, mcfg, "todo", "发会议纪要", content="今天又提了一次")
    assert items.get(mdb, i)["status"] == "done"
    j, _ = add(mdb, mcfg, "todo", "写周报")
    add(mdb, mcfg, "todo", "写周报", status="done")                                   # a producer saw it being finished
    assert items.get(mdb, j)["status"] == "done"


def test_importance_scale_and_due_dates():
    assert items._importance(None) == 0.5 and items._importance(0.9) == 0.9 and items._importance("x") == 0.5
    assert [items._importance(v) for v in (1, 3, 5, 9)] == [1.0, 0.5, 1.0, 1.0]          # 1 is the top of 0..1; 3 and 5 are the 1-5 scale
    base = "2026-11-02"                                                                    # a Monday
    assert items.parse_due("2026-11-06") == "2026-11-06" and items.parse_due("11月6号", base) == "2026-11-06"
    assert items.parse_due("明天", base) == "2026-11-03" and items.parse_due("今天下班前", base) == "2026-11-02"
    assert items.parse_due("周三", base) == "2026-11-04" and items.parse_due("下周一", base) == "2026-11-09"
    assert items.parse_due("周一", "2026-11-04") == "2026-11-09"                           # already passed this week: the next one
    assert items.parse_due("月底", base) == "2026-11-30" and items.parse_due("1月5号", "2026-12-20") == "2027-01-05"
    assert items.parse_due("尽快") is None and items.parse_due("") is None and items.parse_due(None) is None


# ---------------------------------------------------------------------------------------------- user operations
def test_update_validates_and_reindexes(mcfg, mdb):
    i, _ = add(mdb, mcfg, "fact", "灰度定在周五", content="比例百分之五")
    it = items.update(mdb, i, title="灰度改到下周一", content="比例百分之十", pinned=True, importance=5, due="2026-11-09",
                      subject="青禾", kind="decision")
    assert (it["title"], it["kind"], it["pinned"], it["importance"], it["due"], it["subject"], it["source"]) == \
        ("灰度改到下周一", "decision", True, 1.0, "2026-11-09", "青禾", "user")
    assert it["fingerprint"] == items.fingerprint("decision", "灰度改到下周一")
    assert [r["id"] for r in items.search(mdb, mcfg, "下周一")] == [i] and items.search(mdb, mcfg, "周五") == []
    assert items.update(mdb, i, due="")["due"] is None and items.update(mdb, i, status="archived")["status"] == "archived"
    with pytest.raises(LookupError):
        items.update(mdb, 9999, title="x")
    for bad in ({"title": "  "}, {"kind": "poem"}, {"status": "rejected"}, {"due": "很快"}, {"fingerprint": "x"}, {"id": 3}):
        with pytest.raises(ValueError):
            items.update(mdb, i, **bad)
    other, _ = add(mdb, mcfg, "decision", "另一条")
    with pytest.raises(ValueError, match="同名"):
        items.update(mdb, other, title="灰度改到下周一")
    assert items.touch(mdb, [i, i, other]) == 2 and items.get(mdb, i)["hits"] == 1 and items.get(mdb, i)["last_used_at"]


def test_merge_keeps_one_item_and_later_sightings_do_not_undo_it(mcfg, mdb):
    emb = HashEmbedder(64)
    keep, _ = add(mdb, mcfg, "fact", "新店开在栖霞路", content="下个月十二号开业", evidence=[1, 2], day="2026-11-02", importance=2,
                  embedder=emb)
    drop, _ = add(mdb, mcfg, "fact", "咖啡店新址在栖霞路", content="走路十分钟", evidence=[2, 9], day="2026-11-04", importance=5,
                  embedder=emb)
    out = items.merge(mdb, keep, drop)
    assert out["id"] == keep and out["evidence"] == [1, 2, 9] and out["content"] == "下个月十二号开业\n走路十分钟"
    assert out["day"] == "2026-11-04" and out["valid_from"] == "2026-11-02" and out["importance"] == 1.0
    gone = items.get(mdb, drop)
    assert gone["status"] == "superseded" and gone["superseded_by"] == keep
    assert [r["id"] for r in items.search(mdb, mcfg, "栖霞路", embedder=emb)] == [keep]
    assert mdb.one("SELECT COUNT(*) AS n FROM memory_vectors WHERE kind='item' AND ref_id=?", (drop,))["n"] == 0
    # the next digest phrases it the old way again: nothing new, nothing revived
    assert add(mdb, mcfg, "fact", "咖啡店新址在栖霞路", content="又说了一遍", source="digest") == (keep, "skipped")
    assert items.get(mdb, drop)["status"] == "superseded" and mdb.one("SELECT COUNT(*) AS n FROM memory_items")["n"] == 2
    third, _ = add(mdb, mcfg, "fact", "第三种说法")
    items.merge(mdb, third, keep)                                                    # chains are flattened
    assert items.get(mdb, drop)["superseded_by"] == third
    with pytest.raises(ValueError):
        items.merge(mdb, third, third)
    with pytest.raises(ValueError):
        items.merge(mdb, keep, third)                                                # keep is merged away already
    with pytest.raises(LookupError):
        items.merge(mdb, third, 9999)


# ---------------------------------------------------------------------------------------------- search
def test_search_fuses_keywords_and_vectors_and_filters(mcfg, mdb):
    emb = HashEmbedder(256)
    a, _ = add(mdb, mcfg, "fact", "扫地机到手价两千六百九十九", content="送一套滤网", subject="扫地机", day="2026-11-02")
    b, _ = add(mdb, mcfg, "todo", "今晚下单扫地机", owner="我", day="2026-11-02")
    c, _ = add(mdb, mcfg, "fact", "猫两岁半", content="是一只橘猫", subject="小满", day="2026-11-02")
    d, _ = add(mdb, mcfg, "decision", "压测排在周三下午", day="2026-11-02")
    got = items.search(mdb, mcfg, "扫地机多少钱", embedder=emb, now=date(2026, 11, 5))
    assert {r["id"] for r in got[:2]} == {a, b} and c not in [r["id"] for r in got] and d not in [r["id"] for r in got]
    assert got[0]["match"] == "fts+vector+like" and got[0]["score"] > got[1]["score"] > 0
    assert mdb.one("SELECT COUNT(*) AS n FROM memory_vectors WHERE kind='item'")["n"] == 4      # embedded on the way
    assert [r["id"] for r in items.search(mdb, mcfg, "扫地机", kinds=["todo"])] == [b]
    assert [r["id"] for r in items.search(mdb, mcfg, "扫地机", kinds="fact", subject="扫地机")] == [a]
    assert items.search(mdb, mcfg, "扫地机", subject="小满") == []
    items.update(mdb, b, status="done")
    assert [r["id"] for r in items.search(mdb, mcfg, "扫地机")] == [a]
    assert {r["id"] for r in items.search(mdb, mcfg, "扫地机", status=("active", "done"))} == {a, b}
    # no content word in the question -> substring fallback on what is written; nothing -> nothing
    assert [r["match"] for r in items.search(mdb, mcfg, "猫")] == ["like"] and items.search(mdb, mcfg, "火箭") == []
    # an empty query lists the newest
    assert {r["id"] for r in items.search(mdb, mcfg, "")} == {a, c, d}


def test_search_prefers_important_pinned_and_recent_items(mcfg, mdb):
    old, _ = add(mdb, mcfg, "fact", "发布安排一", content="发布", day="2026-05-01")
    new, _ = add(mdb, mcfg, "fact", "发布安排二", content="发布", day="2026-11-04")
    assert [r["id"] for r in items.search(mdb, mcfg, "发布安排", now=date(2026, 11, 5))][0] == new
    items.update(mdb, old, importance=5, pinned=True)
    assert [r["id"] for r in items.search(mdb, mcfg, "发布安排", now=date(2026, 11, 5))][0] == old


def test_search_expands_lexicon_aliases_and_backfills_legacy_tokens(mcfg, mdb):
    lex = mdb.execute("INSERT INTO lexicon(term, kind) VALUES ('周岚', 'person')").lastrowid
    mdb.execute("INSERT INTO lexicon_aliases(lexicon_id, alias, kind) VALUES (?, '岚姐', 'aka')", (lex,))
    index.register_from(mcfg, mdb)
    # a legacy row as the migration leaves it: no tokens, no vector
    mdb.execute("INSERT INTO memory_notes(kind, title, content, day_from, day_to, source_utterance_ids) VALUES (?,?,?,?,?,?)",
                ("person", "周岚", "2026-11-02：负责压测", "2026-11-02", "2026-11-02", "[4, 5]"))
    with mdb.transaction() as c:
        assert copy_notes_to_items(c) == 1
    assert mdb.one("SELECT COUNT(*) AS n FROM memory_item_tokens")["n"] == 0
    got = items.search(mdb, mcfg, "岚姐在忙什么")                    # the nickname finds the item written with the full name
    assert [r["title"] for r in got] == ["周岚"] and got[0]["source"] == "legacy" and got[0]["evidence"] == [4, 5]
    assert mdb.one("SELECT COUNT(*) AS n FROM memory_item_tokens")["n"] == 1
    emb = HashEmbedder(32)
    assert items.reindex(mdb, mcfg, emb) == {"tokens": 0, "vectors": 1} and items.reindex(mdb, mcfg, emb) == {"tokens": 0, "vectors": 0}
    assert items.reindex(mdb, mcfg, None, force=True) == {"tokens": 1, "vectors": 0}
    assert index.reindex(mdb) == 0 and mdb.one("SELECT COUNT(*) AS n FROM memory_item_tokens")["n"] == 1   # the reindex job covers items


def test_related_facts_for_prompts(mcfg, mdb):
    lex = mdb.execute("INSERT INTO lexicon(term, kind) VALUES ('周岚', 'person')").lastrowid
    mdb.execute("INSERT INTO lexicon_aliases(lexicon_id, alias, kind) VALUES (?, '岚姐', 'aka')", (lex,))
    index.register_from(mcfg, mdb)
    add(mdb, mcfg, "fact", "网关的压测目标 QPS 三千五", subject="网关", day="2026-11-02", importance=5)
    add(mdb, mcfg, "person", "周岚", subject="周岚", content="后端工程师，负责网关", day="2026-11-02")
    add(mdb, mcfg, "todo", "发会议纪要", owner="我", day="2026-11-02")
    add(mdb, mcfg, "fact", "CDN 首期十万", day="2026-11-03")
    done, _ = add(mdb, mcfg, "todo", "网关文档补完", subject="网关")
    items.update(mdb, done, status="done")
    lines = items.related_facts(mdb, mcfg, "岚姐说网关今天下午压测，CDN 的合同也在走")
    assert lines[0] == "[事实] 网关的压测目标 QPS 三千五（2026-11-02）"                         # by subject, most important first
    assert "[人物] 周岚：后端工程师，负责网关（2026-11-02）" in lines                          # reached through the nickname
    assert "[事实] CDN 首期十万（2026-11-03）" in lines                                       # shares a Latin term
    assert not any("会议纪要" in ln or "文档" in ln for ln in lines)                          # unrelated / done
    assert len(items.related_facts(mdb, mcfg, "岚姐说网关今天下午压测，CDN 的合同也在走", limit=2)) == 2
    assert items.related_facts(mdb, mcfg, "今天天气不错") == [] and items.related_facts(mdb, mcfg, "") == []


def test_todos_view(mcfg, mdb):
    a, _ = add(mdb, mcfg, "todo", "交 OKR", due="2026-11-05", owner="我")
    b, _ = add(mdb, mcfg, "todo", "修两个问题", due="2026-11-02", owner="周岚")
    c, _ = add(mdb, mcfg, "todo", "有空看看显示器", importance=1)
    d, _ = add(mdb, mcfg, "todo", "催文档", importance=5)
    add(mdb, mcfg, "fact", "不是待办")
    assert [t["id"] for t in items.todos(mdb)] == [b, a, d, c]                    # deadlines first, then importance
    assert [t["id"] for t in items.todos(mdb, due_before="2026-11-03")] == [b]
    items.update(mdb, b, status="done")
    assert [t["id"] for t in items.todos(mdb, status="done")] == [b] and len(items.todos(mdb, status="all")) == 4
    assert b not in [t["id"] for t in items.todos(mdb)]
    with pytest.raises(ValueError):
        items.todos(mdb, due_before="下周")


# ---------------------------------------------------------------------------------------------- people
def test_people_merge_speakers_lexicon_and_memory_by_alias(mcfg, mdb):
    me = mdb.upsert_speaker("甲方")
    zl = mdb.upsert_speaker("周岚")
    mdb.set_self_speaker(me)
    mdb.set_relation(zl, "同事", "user")
    lex = mdb.execute("INSERT INTO lexicon(term, kind, note, speaker_id) VALUES ('周岚', 'person', '后端', ?)", (zl,)).lastrowid
    mdb.execute("INSERT INTO lexicon_aliases(lexicon_id, alias, kind) VALUES (?, '岚姐', 'aka')", (lex,))
    mdb.execute("INSERT INTO lexicon(term, kind) VALUES ('老秦', 'person'), ('青禾', 'project')")
    ids = utts(mdb, [("09:00:00", "周岚", "我先说后端"), ("09:00:05", "甲方", "岚姐压测排哪天"), ("09:00:10", "周岚", "周三下午")],
               speaker_ids={"周岚": zl, "甲方": me})
    utts(mdb, [("12:00:00", None, "老秦说新店下个月开业")])
    sessions.sync(mcfg, mdb, now=datetime(2026, 11, 5))
    add(mdb, mcfg, "person", "周岚", subject="周岚", content="2026-11-03：负责压测", day=DAY, evidence=ids)
    add(mdb, mcfg, "fact", "岚姐买了显示器", subject="岚姐", day="2026-11-04")
    add(mdb, mcfg, "todo", "压测报告发出来", owner="岚姐", day=DAY)
    add(mdb, mcfg, "todo", "修完问题", owner="周岚", status="done", day=DAY)
    add(mdb, mcfg, "person", "阿梨", content="2026-11-03：晚上在家", day=DAY)
    add(mdb, mcfg, "fact", "项目灰度定在周一", subject="青禾", day=DAY)

    people = items.people(mdb, mcfg)
    assert [p["name"] for p in people] == ["甲方", "周岚", "阿梨", "老秦"]     # the user, then by how much they said / is known
    zhou = people[1]
    assert zhou["aliases"] == ["岚姐"] and zhou["speaker_id"] == zl and zhou["lexicon_id"] == lex and zhou["relation"] == "同事"
    assert (zhou["utterances"], zhou["speech_s"], zhou["facts"], zhou["todos"], zhou["last_seen"]) == (2, 4.0, 2, 1, "2026-11-04")
    assert people[0]["is_self"] is True and people[3]["speaker_id"] is None and people[2]["facts"] == 1
    assert "青禾" not in [p["name"] for p in people]                                  # a project is not a person

    p = items.person(mdb, mcfg, "岚姐")                                               # by nickname
    assert p["name"] == "周岚" and p["note"] == "后端" and p["utterances"] == 2 and p["days"] == 1 and p["first_seen"] == DAY
    assert sorted(f["title"] for f in p["facts"]) == ["周岚", "岚姐买了显示器"]
    assert [(t["title"], t["status"]) for t in p["todos"]] == [("压测报告发出来", "active"), ("修完问题", "done")]
    assert len(p["recent_sessions"]) == 1 and p["recent_sessions"][0]["utterance_count"] == 3 and p["last_seen"] == "2026-11-04"
    q = items.person(mdb, mcfg, "老秦")                                               # never enrolled: found by mention
    assert q["speaker_id"] is None and q["mentions"] == 1 and len(q["recent_sessions"]) == 1 and q["facts"] == []
    with pytest.raises(LookupError, match="没有找到"):
        items.person(mdb, mcfg, "不存在的人")


# ---------------------------------------------------------------------------------------------- former memory_notes paths
class DigestClient:
    """Fake LLM for DailyAnalyzer: one map answer, one reduce answer (both adjustable)."""

    def __init__(self):
        self.reduce = {"overview": "聊了发布", "todos": [{"text": "整理测试用例", "owner": "周岚", "due": "周三"}, "预约会议室"],
                       "decisions": ["周五前完成联调"], "people": [{"name": "周岚", "context": "负责测试"}]}
        self.calls = 0

    def json(self, system, user, **kw):
        self.calls += 1
        return {"title": "晨会", "summary": "聊了发布"} if "概括" in system or "这一段" in system or self.calls % 2 else self.reduce


def test_digest_writes_items_once_and_respects_deletes_and_edits(mcfg, mdb, monkeypatch):
    ids = utts(mdb, [("09:00:00", "周岚", "测试用例我周三前整理好"), ("09:00:05", "我", "好，周五前完成联调")], day="2026-11-02")
    an = DailyAnalyzer(mcfg, mdb, DigestClient())
    monkeypatch.setattr(an, "map_block", lambda day, b, total, people="": {"title": "晨会", "summary": "s", "topics": [], "decisions": [],
                                                                         "todos": [], "people": [], "mood": "", "open_questions": []})
    monkeypatch.setattr(an, "reduce_blocks", lambda day, results, **kw: __import__("audio_note.analysis.daily", fromlist=["x"])
                        .normalize_digest(an.client.reduce))
    an.run("2026-11-02")
    rows = {(r["kind"], r["title"]): r for r in items.list_items(mdb)}
    assert set(rows) == {("todo", "整理测试用例"), ("todo", "预约会议室"), ("decision", "周五前完成联调"), ("person", "周岚")}
    todo = rows[("todo", "整理测试用例")]
    assert todo["owner"] == "周岚" and todo["due"] == "2026-11-04" and todo["source"] == "digest" and todo["day"] == "2026-11-02"
    assert todo["content"] == "整理测试用例（负责人：周岚，截止：周三）"
    assert rows[("person", "周岚")]["evidence"] == [ids[0]] and rows[("person", "周岚")]["subject"] == "周岚"
    assert mdb.one("SELECT COUNT(*) AS n FROM memory_notes")["n"] == 0                 # the old table is not written any more

    items.reject(mdb, rows[("todo", "预约会议室")]["id"])                              # the user deletes one ...
    items.update(mdb, rows[("decision", "周五前完成联调")]["id"], content="改成周四")     # ... and edits another
    an.client.reduce["todos"].append("写周报")
    an.run("2026-11-02", force=True)                                                   # the digest is generated again
    after = {(r["kind"], r["title"]): r for r in items.list_items(mdb)}
    assert ("todo", "预约会议室") not in after and ("todo", "写周报") in after and len(after) == 4
    assert after[("decision", "周五前完成联调")]["content"] == "改成周四"
    assert mdb.one("SELECT COUNT(*) AS n FROM memory_items")["n"] == 5                 # 4 visible + 1 tombstone, no duplicates


def test_relations_tool_and_chat_read_items(mcfg, mdb):
    from audio_note.memory.chat import MemoryChat

    add(mdb, mcfg, "person", "周岚", content="2026-11-02：约了周末打球", day="2026-11-02")
    add(mdb, mcfg, "person", "别人", content="不相关", day="2026-11-02")
    gone, _ = add(mdb, mcfg, "person", "周岚的旧条目", subject="周岚", content="已经删掉的内容")
    items.reject(mdb, gone)
    assert person_notes(mdb, "周岚") == "2026-11-02：约了周末打球" and person_notes(mdb, "没有这个人") == ""
    add(mdb, mcfg, "todo", "确认预算", content="开会前确认预算数字", day="2026-11-02")
    done, _ = add(mdb, mcfg, "todo", "预算表发给财务", day="2026-11-01")
    items.update(mdb, done, status="done")
    out = run_tool("search_notes", {"query": "预算"}, mdb, mcfg)
    assert "[todo] 确认预算 (2026-11-02): 开会前确认预算数字" in out and "预算表发给财务" in out and "（已完成）" in out
    assert "已经删掉的内容" not in run_tool("search_notes", {"query": "周岚"}, mdb, mcfg)
    assert run_tool("search_notes", {"kind": "preference"}, mdb, mcfg).startswith("没有找到")      # the new kinds are accepted
    assert run_tool("search_notes", {"kind": "poem"}, mdb, mcfg).startswith("ERROR:")
    chat = MemoryChat(mcfg, mdb, None)
    assert chat.related_notes("预算确认了吗") == ["[todo] 确认预算 (2026-11-02): 开会前确认预算数字"]
    assert chat.related_notes("的吗") == []


# ---------------------------------------------------------------------------------------------- HTTP
@pytest.fixture
def rt(mcfg):
    r = Runtime(mcfg)
    yield r
    r.shutdown()
    from audio_note.logging_setup import _remove_own_handlers

    _remove_own_handlers(logging.getLogger())


@pytest.fixture
def client(rt):
    with TestClient(create_app(rt), raise_server_exceptions=False) as c:
        yield c


def test_api_items_crud_merge_restore_and_events(client, rt):
    sub = rt.events.subscribe()
    r = client.post("/api/memory/items", json={"kind": "todo", "title": "给猫打疫苗", "due": "2026-11-07", "owner": "我", "pinned": True})
    assert r.status_code == 200 and r.json()["action"] == "added"
    it = r.json()["item"]
    assert (it["kind"], it["title"], it["due"], it["owner"], it["pinned"], it["source"], it["status"]) == \
        ("todo", "给猫打疫苗", "2026-11-07", "我", True, "user", "active")
    other = client.post("/api/memory/items", json={"kind": "fact", "title": "宠物医院约了周六十点", "subject": "小满"}).json()["item"]
    assert client.post("/api/memory/items", json={"kind": "todo", "title": "给猫 打疫苗！"}).json()["action"] == "updated"
    assert client.post("/api/memory/items", json={"title": ""}).status_code == 400
    assert client.post("/api/memory/items", json={"kind": "poem", "title": "x"}).json() == {"error": "kind 应为 " + " / ".join(items.KINDS)}

    body = client.get("/api/memory/items").json()
    assert [x["id"] for x in body["items"]] == [it["id"], other["id"]] and body["counts"]["total"] == 2           # pinned first
    assert body["counts"]["by_kind"] == {"todo": 1, "fact": 1}
    assert [x["id"] for x in client.get("/api/memory/items", params={"kind": "fact"}).json()["items"]] == [other["id"]]
    assert [x["id"] for x in client.get("/api/memory/items", params={"q": "疫苗"}).json()["items"]] == [it["id"]]
    assert [x["id"] for x in client.get("/api/memory/items", params={"subject": "小满"}).json()["items"]] == [other["id"]]
    assert client.get("/api/memory/items", params={"status": "nope"}).status_code == 400
    assert client.get("/api/memory/items", params={"kind": "nope"}).status_code == 400

    r = client.patch(f"/api/memory/items/{it['id']}", json={"status": "done", "content": "周六上午十点"})
    assert r.json()["item"]["status"] == "done" and r.json()["item"]["content"] == "周六上午十点"
    assert client.patch(f"/api/memory/items/{it['id']}", json={"fingerprint": "x"}).status_code == 400
    assert client.patch(f"/api/memory/items/{it['id']}", json={}).status_code == 400
    assert client.patch("/api/memory/items/9999", json={"title": "x"}).status_code == 404
    assert client.patch(f"/api/memory/items/{it['id']}", json={"status": "rejected"}).status_code == 400
    one = client.get(f"/api/memory/items/{other['id']}").json()
    assert one["title"] == "宠物医院约了周六十点" and one["evidence"] == [] and one["evidence_utterances"] == []
    assert [x["id"] for x in client.get("/api/memory/items").json()["items"]] == [other["id"]]      # the done todo left the default list
    assert [x["id"] for x in client.get("/api/memory/items", params={"status": "done"}).json()["items"]] == [it["id"]]
    assert len(client.get("/api/memory/items", params={"status": "active,done"}).json()["items"]) == 2
    assert [t["id"] for t in client.get("/api/memory/todos", params={"status": "all"}).json()["todos"]] == [it["id"]]
    assert client.get("/api/memory/todos").json() == {"todos": []}
    assert client.get("/api/memory/todos", params={"due_before": "soon"}).status_code == 400

    assert client.delete(f"/api/memory/items/{other['id']}").json() == {"ok": True, "deleted": 1}
    assert client.delete(f"/api/memory/items/{other['id']}").json() == {"ok": True, "deleted": 0}
    assert client.delete("/api/memory/items/9999").status_code == 404
    assert client.get("/api/memory/items", params={"status": "all"}).json()["items"][0]["id"] == it["id"]
    assert client.post(f"/api/memory/items/{other['id']}/restore").json()["item"]["status"] == "active"
    assert client.post("/api/memory/items/9999/restore").status_code == 404

    r = client.post(f"/api/memory/items/{other['id']}/merge", json={"into": it["id"]})
    assert r.status_code == 200 and r.json()["item"]["id"] == it["id"] and "宠物医院约了周六十点" in r.json()["item"]["content"]
    assert client.get(f"/api/memory/items/{other['id']}").json()["superseded_by"] == it["id"]
    assert client.post(f"/api/memory/items/{it['id']}/merge", json={}).status_code == 400
    assert client.post(f"/api/memory/items/{it['id']}/merge", json={"into": it["id"]}).status_code == 400
    assert client.post(f"/api/memory/items/{it['id']}/merge", json={"into": 9999}).status_code == 404
    assert client.get("/api/memory/items/9999").json() == {"error": "没有这条记忆"}

    changes = [e["change"] for e in sub.drain() if e["type"] == "memory"]
    assert changes == ["added", "added", "updated", "updated", "deleted", "restored", "merged"]


def test_api_notes_are_a_view_of_items_and_delete_is_a_tombstone(client, rt, mcfg):
    a, _ = add(rt.db, mcfg, "todo", "买菜", content="明天买菜", day="2026-11-02", source="digest")
    b, _ = add(rt.db, mcfg, "person", "周岚", content="同事", valid_from="2026-11-01", day="2026-11-03", source="legacy")
    arch, _ = add(rt.db, mcfg, "fact", "归档的", status="archived")
    rows = client.get("/api/notes").json()
    assert {r["id"] for r in rows} == {a, b}
    row = next(r for r in rows if r["id"] == b)
    assert set(row) == {"id", "kind", "title", "content", "day_from", "day_to", "updated_at"}        # the 0.3 shape
    assert (row["day_from"], row["day_to"]) == ("2026-11-01", "2026-11-03")
    assert client.delete(f"/api/notes/{a}").json() == {"ok": True, "deleted": 1}
    assert client.delete(f"/api/notes/{a}").json() == {"ok": True, "deleted": 0}
    assert items.get(rt.db, a)["status"] == "rejected"                                                # a tombstone, not a hole
    assert add(rt.db, mcfg, "todo", "买菜", source="digest") == (a, "skipped")
    assert [r["id"] for r in client.get("/api/notes").json()] == [b]


def test_api_people_sessions_and_forget(client, rt, mcfg):
    db = rt.db
    zl = db.upsert_speaker("周岚")
    ids = utts(db, [("09:00:00", "周岚", "压测排在周三下午"), ("09:00:05", None, "好的"), ("15:00:00", "周岚", "下午再对一下")],
               speaker_ids={"周岚": zl})
    (rt.cfg.audio_dir).mkdir(parents=True, exist_ok=True)
    sessions.sync(mcfg, db, now=datetime(2026, 11, 5))
    add(db, mcfg, "fact", "压测排在周三下午", subject="周岚", evidence=ids[:1], day=DAY)

    assert [p["name"] for p in client.get("/api/memory/people").json()["people"]] == ["周岚"]
    p = client.get("/api/memory/people/周岚").json()
    assert p["utterances"] == 2 and [f["title"] for f in p["facts"]] == ["压测排在周三下午"] and len(p["recent_sessions"]) == 2
    assert client.get("/api/memory/people/谁").status_code == 404

    day = client.get("/api/sessions", params={"day": DAY}).json()
    assert day["day"] == DAY and [s["utterance_count"] for s in day["sessions"]] == [2, 1]
    assert day["sessions"][0]["speakers"][0]["name"] == "周岚" and day["sessions"][0]["speakers"][1]["name"] == "未知"
    assert client.get("/api/sessions", params={"day": "bad"}).status_code == 400
    assert [s["id"] for s in client.get("/api/sessions").json()["sessions"]] == [day["sessions"][1]["id"], day["sessions"][0]["id"]]
    one = client.get(f"/api/sessions/{day['sessions'][0]['id']}").json()
    assert [u["id"] for u in one["utterances"]] == ids[:2] and one["utterances"][0]["speaker"] == "周岚"
    assert one["utterances"][0]["session_id"] == one["id"] and one["utterances"][1]["speaker"] == "未知"
    assert client.get("/api/sessions/9999").json() == {"error": "没有这个会话"}

    sub = rt.events.subscribe()
    dry = client.post("/api/memory/forget", json={"utterance_ids": ids[:2]}).json()                # dry run by default
    assert dry["dry_run"] is True and dry["utterances"] == 2 and db.one("SELECT COUNT(*) AS n FROM utterances")["n"] == 3
    assert [e for e in sub.drain() if e["type"] == "memory"] == []
    assert client.post("/api/memory/forget", json={}).status_code == 400
    assert client.post("/api/memory/forget", json={"utterance_ids": "1,2"}).status_code == 400
    assert client.post("/api/memory/forget", json={"day": DAY, "dry_run": "maybe"}).status_code == 400
    rep = client.post("/api/memory/forget", json={"session_id": one["id"], "dry_run": False}).json()
    assert rep["dry_run"] is False and rep["utterances"] == 2 and rep["sessions"]["removed"] == 1 and rep["items"]["tombstoned"] == 1
    assert [r["id"] for r in db.query("SELECT id FROM utterances")] == [ids[2]]
    evs = [(e["type"], e.get("change")) for e in sub.drain()]
    assert ("utterances", "deleted") in evs and ("memory", "forgot") in evs
    assert client.get("/api/memory/items", params={"status": "all"}).json()["items"] == []


# ---------------------------------------------------------------------------------------------- CLI
@pytest.fixture
def out(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=buf, width=300, force_terminal=False, color_system=None, highlight=False,
                                                 soft_wrap=True))
    yield buf
    from audio_note.logging_setup import _remove_own_handlers

    _remove_own_handlers(logging.getLogger())


def test_cli_memory_tree(mcfg, mdb, out):
    zl = mdb.upsert_speaker("周岚")
    ids = utts(mdb, [("09:00:00", "周岚", "压测排在周三下午"), ("09:00:05", None, "好的")], speaker_ids={"周岚": zl})
    a, _ = add(mdb, mcfg, "todo", "交压测报告", owner="周岚", due="2026-11-06", day=DAY, evidence=ids[:1])
    add(mdb, mcfg, "fact", "压测排在周三下午", subject="周岚", day=DAY)
    mdb.close_all()

    def run(*argv: str) -> tuple[int, str]:
        out.seek(0)
        out.truncate()
        code = cli.main(list(argv))
        return code, out.getvalue()

    code, text = run("memory", "items")
    assert code == 0 and "交压测报告" in text and "压测排在周三下午" in text and "共显示 2 条" in text
    code, text = run("memory", "items", "--kind", "todo", "--json")
    assert code == 0 and [x["id"] for x in json.loads(text)["items"]] == [a]
    assert [x["title"] for x in json.loads(run("memory", "items", "-q", "周三", "--json")[1])["items"]] == ["压测排在周三下午"]
    assert run("memory", "items", "--kind", "poem")[0] == 2
    assert json.loads(run("memory", "todos", "--json")[1])["todos"][0]["due"] == "2026-11-06"
    assert "交压测报告" in run("memory", "todos")[1] and run("memory", "todos", "--done")[1].strip() == "没有待办。"
    assert run("memory", "todos", "--due-before", "soon")[0] == 2
    people = json.loads(run("memory", "people", "--json")[1])["people"]
    assert people[0]["name"] == "周岚" and people[0]["todos"] == 1 and people[0]["facts"] == 1
    assert "周岚" in run("memory", "people")[1] and json.loads(run("memory", "people", "周岚", "--json")[1])["utterances"] == 1
    assert run("memory", "people", "没有这个人")[0] == 2
    doc = json.loads(run("memory", "sessions", "--day", DAY, "--sync", "--json")[1])
    assert doc["sync"]["assigned"] == 2 and [s["utterance_count"] for s in doc["sessions"]] == [2]
    assert "2 句" in run("memory", "sessions", "--day", DAY)[1] and "没有会话" in run("memory", "sessions", "--day", "2020-01-01")[1]

    code, text = run("memory", "forget", "--day", DAY)                              # a preview unless --yes
    assert code == 0 and "将会忘记 2 句话" in text and "只是预览" in text
    assert json.loads(run("memory", "forget", "--utterance", str(ids[0]), "--dry-run", "--json")[1])["dry_run"] is True
    assert run("memory", "forget")[0] == 2 and run("memory", "forget", "--from", "10:00")[0] == 2
    code, text = run("memory", "forget", "--day", DAY, "--from", "09:00", "--to", "09:00:03", "--yes")
    assert code == 0 and "已经忘记 1 句话" in text
    db = DB(mcfg.db_path)
    assert [r["id"] for r in db.query("SELECT id FROM utterances")] == [ids[1]] and items.get(db, a)["status"] == "rejected"
    db.close_all()
    sub = next(x for x in cli.build_parser()._actions if getattr(x, "choices", None) and "memory" in x.choices)
    names = list(sub.choices)
    assert names.index("memory") > names.index("init")                              # after the commands whose order is pinned
