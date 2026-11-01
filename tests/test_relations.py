"""Tests for audio_note.analysis.relations (LLM relation inference) and its wiring into the prompts,
the daily digest, the memory chat system prompt and the list_speakers tool. Fake client only."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from audio_note.analysis import prompts, relations
from audio_note.analysis.relations import RELATION_CHOICES, infer_relations, maybe_infer_after_daily
from audio_note.db import DB
from audio_note.llm.client import LLMError
from audio_note.memory.tools import run_tool

from .test_daily import FakeClient as DailyFakeClient


# ----------------------------------------------------------------------------------------------
# fixtures / helpers
# ----------------------------------------------------------------------------------------------
@pytest.fixture
def cfg(cfg_factory):
    return cfg_factory({"mimo": {"api_key": "test-key-not-real"}, "speakers": {"relation_min_utterances": 3}})


@pytest.fixture
def db(cfg) -> DB:
    return DB(cfg.db_path)


def _seg(db: DB, started: datetime, n: int = 0) -> int:
    k = db.one("SELECT COUNT(*) AS n FROM segments")["n"]
    return db.add_segment(f"seg_{k}_{n}_{started:%Y%m%d%H%M%S}.opus", started, 3600.0)


def _utts(db: DB, seg: int, items: list[tuple[int | None, str | None, str]], started: datetime, day: str | None = None) -> list[int]:
    """items: (speaker_id, speaker_name, text) placed 5 s apart inside segment `seg`."""
    rows = []
    for i, (sid, name, text) in enumerate(items):
        ts = started + timedelta(seconds=i * 5)
        rows.append({
            "segment_id": seg, "start_s": i * 5.0, "end_s": i * 5.0 + 4.0,
            "abs_start": ts.isoformat(timespec="seconds"), "day": day or ts.strftime("%Y-%m-%d"),
            "cluster_label": i % 2, "speaker_id": sid, "speaker_name": name, "speaker_score": None,
            "text_raw": text, "text": text,
        })
    return db.add_utterances(rows)


def _talk(db: DB, sid: int, name: str, n: int, day: datetime, partner: tuple[int | None, str | None] = (None, None)) -> int:
    """One segment on `day` where `name` says n lines, alternating with `partner`."""
    seg = _seg(db, day, n)
    items: list[tuple[int | None, str | None, str]] = []
    for i in range(n):
        items.append((partner[0], partner[1], f"{partner[1] or '某人'}的第{i}句"))
        items.append((sid, name, f"{name}的第{i}句话"))
    _utts(db, seg, items, day)
    return seg


class FakeJson:
    """client.json() stand-in: answers per speaker name (parsed from the user prompt), records every call."""

    def __init__(self, answers: dict[str, Any] | None = None, fail: set[str] | None = None, default: Any = None):
        self.answers = dict(answers or {})
        self.fail = set(fail or ())
        self.default = default if default is not None else {"relation": "不确定", "confidence": 0.1, "reason": "", "address": ""}
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def _name(user: str) -> str:
        return user.split("说话人：", 1)[1].split("（", 1)[0]

    def json(self, system: str, user: str, **kw: Any) -> Any:
        name = self._name(user)
        self.calls.append({"name": name, "system": system, "user": user, "kw": kw})
        if name in self.fail:
            raise LLMError(f"boom for {name}")
        return json.loads(json.dumps(self.answers.get(name, self.default)))

    def chat(self, *a, **k):
        raise AssertionError("chat() should not be called")

    def text(self, *a, **k):
        raise AssertionError("text() should not be called")


D1 = datetime(2026, 10, 27, 9, 0, 0)
D2 = datetime(2026, 10, 28, 9, 0, 0)
D3 = datetime(2026, 10, 29, 9, 0, 0)


def _by_id(results: list[dict], sid: int) -> dict:
    return next(r for r in results if r["id"] == sid)


# ----------------------------------------------------------------------------------------------
# infer_relations: candidates, cache, thresholds, errors
# ----------------------------------------------------------------------------------------------
def test_candidates_self_user_labelled_and_too_few_are_skipped(cfg, db):
    me = db.upsert_speaker("张伟")
    db.set_self_speaker(me)
    wang = db.upsert_speaker("王敏")
    db.set_relation(wang, "同事", "user", note="同一个组")
    li = db.upsert_speaker("李强")
    xiao = db.upsert_speaker("小雨")
    _talk(db, me, "张伟", 5, D1)
    _talk(db, wang, "王敏", 5, D1, (me, "张伟"))
    _talk(db, li, "李强", 5, D2, (me, "张伟"))
    _talk(db, xiao, "小雨", 2, D3)                     # 2 < relation_min_utterances (3)

    client = FakeJson({"李强": {"relation": "朋友", "confidence": 0.85, "reason": "一起约了周末打球，互称哥们", "address": "老张"}})
    res = infer_relations(cfg, db, client)
    assert [c["name"] for c in client.calls] == ["李强"]
    assert _by_id(res, me)["skipped"] == "本人" and _by_id(res, me)["applied"] is False
    assert "用户已标注" in _by_id(res, wang)["skipped"] and _by_id(res, wang)["relation"] == "同事"
    assert "语句太少" in _by_id(res, xiao)["skipped"]
    r = _by_id(res, li)
    assert r == {"id": li, "name": "李强", "relation": "朋友", "confidence": 0.85, "reason": "一起约了周末打球，互称哥们",
                 "address": "老张", "applied": True, "skipped": None}
    row = db.one("SELECT relation, relation_source, relation_note, relation_updated_at FROM speakers WHERE id=?", (li,))
    assert (row["relation"], row["relation_source"], row["relation_note"]) == ("朋友", "auto", "一起约了周末打球，互称哥们")
    assert row["relation_updated_at"]
    assert db.get_kv(f"relation_inferred_n:{li}") == 5
    # the user label is untouched
    w = db.one("SELECT relation, relation_source, relation_note FROM speakers WHERE id=?", (wang,))
    assert (w["relation"], w["relation_source"], w["relation_note"]) == ("同事", "user", "同一个组")


def test_user_label_is_never_overwritten_even_with_force(cfg, db):
    wang = db.upsert_speaker("王敏")
    db.set_relation(wang, "同事", "user")
    _talk(db, wang, "王敏", 6, D1)
    client = FakeJson({"王敏": {"relation": "领导", "confidence": 0.99, "reason": "x"}})
    res = infer_relations(cfg, db, client, speaker_ids=[wang], force=True)
    assert client.calls == [] and res[0]["skipped"] and res[0]["applied"] is False
    assert db.one("SELECT relation, relation_source FROM speakers WHERE id=?", (wang,))["relation_source"] == "user"


def test_cache_in_kv_skips_until_growth_or_force(cfg, db):
    li = db.upsert_speaker("李强")
    _talk(db, li, "李强", 4, D1)
    client = FakeJson({"李强": {"relation": "朋友", "confidence": 0.9, "reason": "r"}})
    assert infer_relations(cfg, db, client)[0]["applied"] is True
    assert len(client.calls) == 1 and db.get_kv(f"relation_inferred_n:{li}") == 4

    # same count -> cached
    res = infer_relations(cfg, db, client)
    assert len(client.calls) == 1 and "已推断过" in res[0]["skipped"] and res[0]["relation"] == "朋友"
    # 5 < 4 * 1.5 -> still cached
    _talk(db, li, "李强", 1, D2)
    assert len(infer_relations(cfg, db, client)) == 1 and len(client.calls) == 1
    # force -> asked again
    infer_relations(cfg, db, client, force=True)
    assert len(client.calls) == 2
    # 6 >= 4 * 1.5 ... but the cache now holds 5 -> need 8; add until 8
    _talk(db, li, "李强", 3, D3)
    assert db.one("SELECT COUNT(*) AS n FROM utterances WHERE speaker_id=?", (li,))["n"] == 8
    infer_relations(cfg, db, client)
    assert len(client.calls) == 3 and db.get_kv(f"relation_inferred_n:{li}") == 8
    # explicit speaker_ids restricts the run; unknown ids are simply absent
    assert infer_relations(cfg, db, client, speaker_ids=[9999]) == []


def test_confidence_threshold_invalid_relation_and_unsure_are_not_written(cfg, db):
    a = db.upsert_speaker("甲")
    b = db.upsert_speaker("乙")
    c = db.upsert_speaker("丙")
    for sid, name in ((a, "甲"), (b, "乙"), (c, "丙")):
        _talk(db, sid, name, 4, D1)
    client = FakeJson({
        "甲": {"relation": "朋友", "confidence": 0.5, "reason": "不太确定"},          # below 0.6
        "乙": {"relation": "死党", "confidence": 0.95, "reason": "不是候选"},         # not in RELATION_CHOICES
        "丙": {"relation": "不确定", "confidence": "0.9", "reason": "证据不足"},
    })
    res = infer_relations(cfg, db, client)
    assert len(client.calls) == 3
    assert all(r["applied"] is False and r["skipped"] is None for r in res)
    assert _by_id(res, a)["confidence"] == 0.5 and _by_id(res, b)["relation"] == "死党" and _by_id(res, c)["relation"] == "不确定"
    for sid in (a, b, c):
        assert db.one("SELECT relation, relation_source FROM speakers WHERE id=?", (sid,))["relation"] is None
        assert db.get_kv(f"relation_inferred_n:{sid}") == 4        # cached anyway: not asked again every day
    assert infer_relations(cfg, db, client) and len(client.calls) == 3

    # an earlier automatic relation survives a later "不确定"
    db.set_relation(a, "朋友", "auto", note="old")
    client.answers["甲"] = {"relation": "不确定", "confidence": 0.2, "reason": ""}
    infer_relations(cfg, db, client, speaker_ids=[a], force=True)
    assert db.one("SELECT relation FROM speakers WHERE id=?", (a,))["relation"] == "朋友"


def test_llm_error_for_one_speaker_does_not_stop_the_others(cfg, db, caplog):
    a = db.upsert_speaker("甲")
    b = db.upsert_speaker("乙")
    _talk(db, a, "甲", 4, D1)
    _talk(db, b, "乙", 4, D1)
    client = FakeJson({"乙": {"relation": "领导", "confidence": 0.8, "reason": "叫他老板"}}, fail={"甲"})
    with caplog.at_level("WARNING", logger="audio_note.analysis.relations"):
        res = infer_relations(cfg, db, client)
    assert len(client.calls) == 2
    ra, rb = _by_id(res, a), _by_id(res, b)
    assert ra["applied"] is False and "LLM 调用失败" in ra["skipped"] and "boom" in ra["skipped"]
    assert rb["applied"] is True and rb["relation"] == "领导"
    assert db.get_kv(f"relation_inferred_n:{a}") is None       # retried next time
    assert db.get_kv(f"relation_inferred_n:{b}") == 4
    assert any("boom for 甲" in m for m in caplog.messages)


# ----------------------------------------------------------------------------------------------
# evidence + prompt
# ----------------------------------------------------------------------------------------------
def test_evidence_has_context_self_label_notes_and_choices(cfg, db):
    me = db.upsert_speaker("张伟")
    db.set_self_speaker(me)
    li = db.upsert_speaker("李强")
    seg = _seg(db, D1)
    _utts(db, seg, [
        (me, "张伟", "老李，周末打球去不去？"),
        (li, "李强", "去啊，老张，老地方见。"),
        (None, None, "那我也去。"),
        (li, "李强", "带上球拍。"),
    ], D1)
    db.execute("INSERT INTO memory_notes(kind, title, content, day_from, day_to) VALUES (?,?,?,?,?)",
               ("person", "李强", "2026-10-27：约了周末打球", "2026-10-27", "2026-10-27"))
    db.execute("INSERT INTO memory_notes(kind, title, content, day_from, day_to) VALUES (?,?,?,?,?)",
               ("person", "别人", "不相关", "2026-10-27", "2026-10-27"))
    cfg2 = cfg
    cfg2["speakers"]["relation_min_utterances"] = 1
    client = FakeJson({"李强": {"relation": "朋友", "confidence": 0.9, "reason": "r"}})
    infer_relations(cfg2, db, client)
    call = client.calls[0]
    system, user = call["system"], call["user"]
    assert call["kw"]["max_tokens"] == 512
    # system prompt: who is the user, the target name, all candidates, the JSON schema
    assert "「本人」" in system and "张伟" in system and "「李强」" in system and "不确定" in system
    assert "、".join(RELATION_CHOICES) in system and '"relation"' in system and '"confidence"' in system and '"address"' in system
    # evidence: day header, previous/next lines of the same segment, self labelled 本人, unknown label, target marked
    assert "【2026-10-27】" in user
    assert "09:00 本人：老李，周末打球去不去？" in user
    assert ">> 09:00 李强：去啊，老张，老地方见。" in user
    assert f"09:00 {cfg.speakers.unknown_label}：那我也去。" in user
    assert ">> 09:00 李强：带上球拍。" in user
    assert "共 2 条语句" in user and "抽取了 2 条" in user
    # person notes of that name only
    assert "约了周末打球" in user and "不相关" not in user

    # no self speaker: the system prompt says so instead of pointing at a name
    db.set_self_speaker(None)
    infer_relations(cfg2, db, client, force=True)
    assert {c["name"] for c in client.calls[1:]} == {"李强", "张伟"}    # 张伟 is an ordinary candidate now
    s2 = next(c for c in client.calls[1:] if c["name"] == "李强")["system"]
    assert "没有登记" in s2 and "第一人称" in s2 and "张伟" not in s2
    u2 = next(c for c in client.calls[1:] if c["name"] == "李强")["user"]
    assert "09:00 张伟：老李，周末打球去不去？" in u2 and "本人：" not in u2


def test_evidence_sampling_is_spread_across_days(cfg, db):
    li = db.upsert_speaker("李强")
    for i, day in enumerate((D1, D2, D3)):
        _talk(db, li, "李强", 40, day)
    text, total, shown = relations.build_evidence(db, li, unknown="未知")
    assert total == 120 and shown == relations.MAX_EVIDENCE == 60
    assert text.count("【2026-10-") == 3
    for d in ("27", "28", "29"):
        part = text.split(f"【2026-10-{d}】", 1)[1].split("【", 1)[0]
        assert part.count(">> ") == 20
    # a small set is returned whole, in time order
    rows = db.query("SELECT id, day FROM utterances WHERE speaker_id=? ORDER BY abs_start, id", (li,))
    assert relations._sample_across_days(rows, 500) == list(rows)
    assert relations._evenly([1, 2, 3, 4, 5], 3) == [1, 3, 5] and relations._evenly([1, 2, 3], 1) == [2]
    assert relations._evenly([], 3) == [] and relations._evenly([1, 2], 0) == []


def test_parse_result_tolerates_odd_shapes():
    p = relations.parse_result({"relation": " 「朋友」 ", "confidence": "85", "reason": "x" * 300, "address": None})
    assert p["relation"] == "朋友" and p["confidence"] == 0.85 and len(p["reason"]) == 120 and p["address"] == ""
    assert relations.parse_result("garbage") == {"relation": "", "confidence": 0.0, "reason": "", "address": ""}
    assert relations.parse_result({"confidence": "abc"})["confidence"] == 0.0
    assert relations.parse_result({"confidence": 7})["confidence"] == 0.07
    assert relations.parse_result({"confidence": -3})["confidence"] == 0.0


# ----------------------------------------------------------------------------------------------
# maybe_infer_after_daily
# ----------------------------------------------------------------------------------------------
def test_maybe_infer_after_daily_respects_config_key_and_client(cfg_factory):
    cfg = cfg_factory({"mimo": {"api_key": "test-key-not-real"}, "speakers": {"relation_min_utterances": 2, "infer_relations": False}})
    db = DB(cfg.db_path)
    li = db.upsert_speaker("李强")
    _talk(db, li, "李强", 3, D1)
    client = FakeJson({"李强": {"relation": "朋友", "confidence": 0.9, "reason": "r"}})
    assert maybe_infer_after_daily(cfg, db, client) == [] and client.calls == []

    cfg["speakers"]["infer_relations"] = True
    assert maybe_infer_after_daily(cfg, db, None) == []
    cfg["mimo"]["api_key"] = ""
    assert maybe_infer_after_daily(cfg, db, client) == [] and client.calls == []
    cfg["mimo"]["api_key"] = "sk-填这里"                                   # placeholder counts as missing
    assert maybe_infer_after_daily(cfg, db, client) == [] and client.calls == []

    cfg["mimo"]["api_key"] = "test-key-not-real"
    res = maybe_infer_after_daily(cfg, db, client)
    assert len(client.calls) == 1 and res[0]["applied"] is True
    assert db.one("SELECT relation, relation_source FROM speakers WHERE id=?", (li,))["relation"] == "朋友"

    # cached now: a second daily run does not ask again
    assert maybe_infer_after_daily(cfg, db, client)[0]["skipped"] and len(client.calls) == 1

    # an unexpected (non-LLMError) failure inside is swallowed (the digest already succeeded)
    class Broken:
        def json(self, *a, **k):
            raise RuntimeError("exploded")
    db.execute("DELETE FROM kv")
    assert maybe_infer_after_daily(cfg, db, Broken()) == []


# ----------------------------------------------------------------------------------------------
# prompts.people_line + daily wiring
# ----------------------------------------------------------------------------------------------
def test_people_line_variants():
    people = [
        {"id": 1, "name": "张伟", "is_self": True, "relation": None, "relation_source": None, "relation_note": None, "utterances": 9},
        {"id": 2, "name": "王敏", "is_self": False, "relation": "同事", "relation_source": "user", "relation_note": None, "utterances": 5},
        {"id": 3, "name": "李强", "is_self": False, "relation": "朋友", "relation_source": "auto", "relation_note": "r", "utterances": 3},
        {"id": 4, "name": "小雨", "is_self": False, "relation": None, "relation_source": None, "relation_note": None, "utterances": 1},
    ]
    line = prompts.people_line(people, "未知")
    assert line == "人物：我=「张伟」（用户本人）；王敏（同事）；李强（朋友，自动推断）；小雨（关系未标）；未识别的说话人标为「未知」"
    assert "\n" not in line
    # no self speaker
    line = prompts.people_line(people[1:], "未知")
    assert line.startswith("人物：用户本人：未登记（记录里没有标出哪位是本人）；王敏（同事）")
    # nobody at all
    line = prompts.people_line([], "陌生人")
    assert line == "人物：用户本人：未登记（记录里没有标出哪位是本人）；没有登记其他说话人；未识别的说话人标为「陌生人」"
    # only self
    assert prompts.people_line(people[:1], "未知") == "人物：我=「张伟」（用户本人）；没有登记其他说话人；未识别的说话人标为「未知」"


def test_prompts_accept_people_and_keep_old_signature():
    s, u = prompts.map_prompt("2026-01-01", 1, 2, "09:00", "09:30", "09:00 张三：你好")
    assert "人物" not in u and "09:00 张三：你好" in u and "open_loops" not in s
    assert "人物」一行" in s and "第一人称" in s and "不要包含用户本人" in s
    s, u = prompts.map_prompt("2026-01-01", 1, 2, "09:00", "09:30", "09:00 张三：你好", people="人物：我=「张伟」（用户本人）")
    assert "时段：第 1/2 块，09:00–09:30\n人物：我=「张伟」（用户本人）\n\n=== 转写文本开始 ===" in u
    s, u = prompts.reduce_prompt("2026-01-01", [{"title": "t"}], utterance_count=3, session_count=1, speakers=["张三"])
    assert "人物" not in u.split("=== 分块总结 JSON 开始 ===")[0] and "人物」一行" in s and "open_loops" in s
    s, u = prompts.reduce_prompt("2026-01-01", [{"title": "t"}], utterance_count=3, session_count=1, speakers=["张三"],
                                 people="人物：X")
    assert "出现的说话人：张三\n人物：X\n\n=== 分块总结 JSON 开始 ===" in u


def test_daily_run_passes_people_line_and_skips_self_person_note(cfg, db):
    from audio_note.analysis.daily import DailyAnalyzer

    me = db.upsert_speaker("张三")
    db.set_self_speaker(me)
    li = db.upsert_speaker("李四")
    db.set_relation(li, "同事", "user")
    seg = _seg(db, datetime(2026, 10, 30, 9, 0, 0))
    _utts(db, seg, [
        (me, "张三", "今天我们先过一下项目进度。"),
        (li, "李四", "服务器预算还没批下来。"),
        (me, "张三", "测试用例我来整理。"),
    ], datetime(2026, 10, 30, 9, 0, 0))
    client = DailyFakeClient()
    md, js = DailyAnalyzer(cfg, db, client).run("2026-10-30")
    kinds = [c["kind"] for c in client.calls]
    assert kinds == ["map", "reduce"]
    line = "人物：我=「张三」（用户本人）；李四（同事）；未识别的说话人标为「未知」"
    assert line in client.calls[0]["user"] and line in client.calls[1]["user"]
    assert "人物」一行" in client.calls[0]["system"] and "人物」一行" in client.calls[1]["system"]
    assert js["self_name"] == "张三" and js["speakers"] == ["张三", "李四"]
    # reduce result names 张三 and 李四 as people: no person note for the user themselves
    titles = {r["title"] for r in db.query("SELECT title FROM memory_notes WHERE kind='person'")}
    assert titles == {"李四"}
    assert json.loads(db.one("SELECT summary_json FROM daily_summaries WHERE day='2026-10-30'")["summary_json"])["self_name"] == "张三"
    # no utterances: self_name still reported
    _, js0 = DailyAnalyzer(cfg, db, client).run("2026-10-29")
    assert js0["self_name"] == "张三" and js0["utterance_count"] == 0


# ----------------------------------------------------------------------------------------------
# memory chat system prompt + list_speakers tool
# ----------------------------------------------------------------------------------------------
def test_chat_system_prompt_names_self_and_relations(cfg, db):
    from audio_note.memory.chat import MemoryChat

    dummy_retriever = SimpleNamespace(mode="fts", rrf_k=60)
    chat = MemoryChat(cfg, db, FakeJson(), retriever=dummy_retriever)
    p = chat.system_prompt(now=datetime(2026, 10, 30, 21, 0))
    assert "用户本人：未登记" in p and "提问里的“我”指用户本人" in p
    assert p.index("用户本人：未登记") < p.index("工作方式")

    me = db.upsert_speaker("张伟")
    db.set_self_speaker(me)
    li = db.upsert_speaker("李强")
    db.set_relation(li, "朋友", "auto", note="x")
    p = chat.system_prompt(now=datetime(2026, 10, 30, 21, 0), with_tools=False)
    assert "人物：我=「张伟」（用户本人）；李强（朋友，自动推断）" in p
    assert "提问里的“我”指用户本人「张伟」" in p and "2026-10-30 21:00（周五）" in p
    assert p.index("人物：我=「张伟」") < p.index("工作方式")


def test_list_speakers_tool_shows_self_and_relations(cfg, db):
    assert "还没有登记" in run_tool("list_speakers", {}, db, cfg)
    me = db.upsert_speaker("张伟")
    wang = db.upsert_speaker("王敏", notes="隔壁组")
    db.set_relation(wang, "同事", "user")
    li = db.upsert_speaker("李强")
    db.set_relation(li, "朋友", "auto", note="r")
    xiao = db.upsert_speaker("小雨")
    seg = _seg(db, D1)
    _utts(db, seg, [(wang, "王敏", "a"), (wang, "王敏", "b"), (li, "李强", "c"), (me, "张伟", "d"), (None, None, "e")], D1)

    out = run_tool("list_speakers", {}, db, cfg)
    lines = out.splitlines()
    assert lines[0] == "已登记 4 位说话人："
    assert "- 用户本人：未登记" in out
    assert "- 王敏: 2 条记录，0 个登记样本，关系：同事，备注：隔壁组" in out
    assert "- 李强: 1 条记录，0 个登记样本，关系：朋友（自动推断）" in out
    assert "- 小雨: 0 条记录，0 个登记样本" in out
    assert f"- {cfg.speakers.unknown_label}（未识别）: 1 条记录" in out

    db.set_self_speaker(me)
    out = run_tool("list_speakers", {}, db, cfg)
    lines = out.splitlines()
    assert lines[1] == "- 张伟（本人）: 1 条记录，0 个登记样本"      # self first
    assert "用户本人：未登记" not in out and "- 王敏: 2 条记录" in lines[2]


# ----------------------------------------------------------------------------------------------
# review fixes: stale cache, cleared relation, a user label arriving during the LLM call
# ----------------------------------------------------------------------------------------------
def test_cache_above_current_count_is_stale_and_clearing_or_deleting_forgets_it(cfg, db):
    li = db.upsert_speaker("李强")
    _talk(db, li, "李强", 4, D1)
    client = FakeJson({"李强": {"relation": "朋友", "confidence": 0.9, "reason": "r"}})
    infer_relations(cfg, db, client)
    assert len(client.calls) == 1 and db.get_kv(f"relation_inferred_n:{li}") == 4
    # (a) utterances re-identified away: the cached count is above the current one -> stale, asked again
    db.execute("DELETE FROM utterances WHERE id=(SELECT MIN(id) FROM utterances WHERE speaker_id=?)", (li,))
    res = infer_relations(cfg, db, client)
    assert len(client.calls) == 2 and res[0]["skipped"] is None and db.get_kv(f"relation_inferred_n:{li}") == 3
    assert "已推断过" in infer_relations(cfg, db, client)[0]["skipped"] and len(client.calls) == 2
    # (b) the user clears the relation -> the cache row goes with it, so the next manual run asks again
    db.set_relation(li, None, None)
    assert db.get_kv(f"relation_inferred_n:{li}") is None
    assert db.one("SELECT relation, relation_source FROM speakers WHERE id=?", (li,))[:] == (None, None)
    assert infer_relations(cfg, db, client)[0]["applied"] is True
    assert len(client.calls) == 3 and db.get_kv(f"relation_inferred_n:{li}") == 3
    # (c) deleting the speaker drops the cache row too
    db.delete_speaker(li, "未知")
    assert db.one("SELECT 1 FROM kv WHERE key LIKE 'relation_inferred_n:%'") is None


def test_auto_relation_does_not_replace_a_label_set_during_the_llm_call(cfg, db):
    li = db.upsert_speaker("李强")
    _talk(db, li, "李强", 4, D1)

    class LabelsMeanwhile(FakeJson):
        def json(self, system, user, **kw):
            db.set_relation(li, "导师", "user", note="我自己标的")       # the user saves while the model thinks
            return super().json(system, user, **kw)

    client = LabelsMeanwhile({"李强": {"relation": "朋友", "confidence": 0.9, "reason": "r"}})
    res = infer_relations(cfg, db, client, force=True)
    assert len(client.calls) == 1 and res[0]["applied"] is False and res[0]["relation"] == "朋友"
    row = db.one("SELECT relation, relation_source, relation_note FROM speakers WHERE id=?", (li,))
    assert tuple(row) == ("导师", "user", "我自己标的")
    assert db.get_kv(f"relation_inferred_n:{li}") == 4
    assert infer_relations(cfg, db, client)[0]["skipped"] == "用户已标注，不覆盖" and len(client.calls) == 1
