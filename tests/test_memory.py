"""Tests for audio_note.memory (index / tools / chat). No network: the LLM client is a fake."""
from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import pytest
import yaml

from audio_note.config import load_config
from audio_note.db import DB
from audio_note.memory import index
from audio_note.memory.chat import MemoryChat
from audio_note.memory.index import Hit, like_search, register_terms, reindex, search, tokenize, window
from audio_note.memory.tools import MAX_CHARS, TOOL_NAMES, TOOLS, norm_day, run_tool


# --- fixtures -----------------------------------------------------------------
@pytest.fixture
def cfg(tmp_path, monkeypatch):
    data = tmp_path / "data"
    gl = tmp_path / "glossary.txt"
    gl.write_text("王总\n# comment\n\n", encoding="utf-8")
    conf = tmp_path / "config.yaml"
    conf.write_text(yaml.safe_dump({
        "paths": {"data_dir": str(data)},
        "correction": {"glossary_file": str(gl)},
        "memory": {"max_tool_rounds": 3},
    }), encoding="utf-8")
    monkeypatch.setenv("AUDIO_NOTE_CONFIG", str(conf))
    c = load_config()
    c.ensure_dirs()
    return c


UTTS = [
    # (day, time, speaker, text)
    ("2026-10-30", "08:19:10", "张三", "今天我们先过一下项目进度，下周二之前要交付第一版。"),
    ("2026-10-30", "08:19:32", "未知", "好的，预算还要再确认一下。"),
    ("2026-10-30", "09:05:00", "李四", "下午三点和王总开会讨论预算。"),
    ("2026-10-31", "10:00:00", "张三", "昨天说的项目进度文档我已经写好了。"),
    ("2026-10-31", "10:01:00", "李四", "我把会议纪要发到群里了。"),
    ("2026-10-31", "15:30:00", "张三", "机器学习模型的训练明天再跑一次。"),
]


@pytest.fixture
def db(cfg):
    register_terms(["张三", "李四", "王总"])
    d = DB(cfg.db_path)
    sid_z = d.upsert_speaker("张三", "同事")
    sid_l = d.upsert_speaker("李四")
    spk_ids = {"张三": sid_z, "李四": sid_l}
    seg = d.add_segment(cfg.audio_dir / "x.opus", datetime(2026, 10, 30, 8, 19, 0), 600.0)
    rows = []
    for i, (day, t, spk, text) in enumerate(UTTS):
        rows.append({
            "segment_id": seg, "start_s": float(i * 10), "end_s": float(i * 10 + 5),
            "abs_start": f"{day}T{t}", "day": day, "cluster_label": 0,
            "speaker_id": spk_ids.get(spk), "speaker_name": spk, "speaker_score": 0.9,
            "text_raw": text, "text": text, "text_seg": tokenize(text),
        })
    ids = d.add_utterances(rows)
    d.ids = ids  # type: ignore[attr-defined]
    return d


# --- index ----------------------------------------------------------------------
def test_tokenize_basic():
    seg = tokenize("今天我们先过一下项目进度，下周二之前要交付第一版。")
    toks = seg.split()
    assert "项目" in toks and "进度" in toks
    assert "，" not in toks and "。" not in toks
    assert "  " not in seg
    assert tokenize("") == "" and tokenize(None) == ""
    assert tokenize("。。。！！") == ""


def test_register_terms_idempotent():
    n1 = register_terms(["王总", "深度学习平台"])
    n2 = register_terms(["王总", "深度学习平台", ""])
    assert n2 == 0 and n1 >= 0
    assert "王总" in tokenize("和王总开会").split()
    assert "深度学习平台" in tokenize("我们的深度学习平台上线了").split()


def test_search_chinese_phrase(db):
    hits = search(db, "项目进度")
    assert hits, "should find utterances containing 项目进度"
    assert all(isinstance(h, Hit) for h in hits)
    assert all("项目" in h.text and "进度" in h.text for h in hits)
    assert {h.day for h in hits} == {"2026-10-30", "2026-10-31"}
    assert hits[0].speaker_name == "张三"
    assert hits[0].abs_start.startswith("2026-10-")


def test_search_filters(db):
    only31 = search(db, "项目进度", day_from="2026-10-31", day_to="2026-10-31")
    assert len(only31) == 1 and only31[0].day == "2026-10-31"
    lisi = search(db, "预算", speaker="李四")
    assert len(lisi) == 1 and lisi[0].speaker_name == "李四"
    assert search(db, "预算", speaker="张三") == []
    assert search(db, "预算", day_to="2026-10-29") == []
    assert search(db, "预算", limit=1).__len__() == 1


def test_search_or_fallback_and_empty(db):
    # no single utterance has both 预算 and 交付 -> AND yields 0 -> OR picks both
    hits = search(db, "预算 交付")
    assert len(hits) >= 2
    texts = " ".join(h.text for h in hits)
    assert "预算" in texts and "交付" in texts
    assert search(db, "") == []
    assert search(db, "。。") == []
    # stopwords ("的") must not make the OR fallback match everything
    assert search(db, "不存在的词汇xyz") == []
    assert search(db, "我的预算") and all("预算" in h.text for h in search(db, "我的预算"))


def test_like_search(db):
    hits = like_search(db, "王总")
    assert len(hits) == 1 and hits[0].speaker_name == "李四"
    assert like_search(db, "100%") == []
    assert like_search(db, "王总", speaker="张三") == []


def test_window(db):
    ids = db.ids
    rows = window(db, ids[2], before=1, after=1)
    assert [r["id"] for r in rows] == [ids[1], ids[2], ids[3]]
    assert [r["abs_start"] for r in rows] == sorted(r["abs_start"] for r in rows)
    first = window(db, ids[0], before=5, after=1)
    assert [r["id"] for r in first] == [ids[0], ids[1]]
    assert window(db, 999999) == []


def test_reindex(db, cfg):
    seg = db.one("SELECT id FROM segments")["id"]
    (uid,) = db.add_utterances([{
        "segment_id": seg, "start_s": 0.0, "end_s": 1.0, "abs_start": "2026-11-01T09:00:00", "day": "2026-11-01",
        "speaker_name": "张三", "text_raw": "量子计算很有意思", "text": "量子计算很有意思",
    }])
    assert search(db, "量子计算") == []          # no text_seg -> not indexed
    n = reindex(db)
    assert n == len(UTTS) + 1
    assert [h.id for h in search(db, "量子计算")] == [uid]
    assert search(db, "项目进度")                # existing rows still searchable after upsert


# --- tools ------------------------------------------------------------------------
def test_tool_schema():
    assert set(TOOL_NAMES) == {"search_transcripts", "get_transcript", "get_daily_summary", "list_days",
                               "list_speakers", "search_notes", "get_context"}
    for t in TOOLS:
        assert t["type"] == "function"
        assert t["function"]["parameters"]["type"] == "object"


def test_norm_day():
    assert norm_day("2026-10-31") == "2026-10-31"
    assert norm_day("2026/10/3") == "2026-10-03"
    assert norm_day("today") == datetime.now().date().isoformat()
    assert norm_day("") is None and norm_day("abc") is None and norm_day("2026-13-01") is None


def test_tools_search_and_context(db, cfg):
    out = run_tool("search_transcripts", {"query": "项目进度"}, db, cfg)
    assert "找到 2 条" in out and "张三" in out and "2026-10-30 08:19" in out
    # json-string args + speaker/day filters
    out = run_tool("search_transcripts", json.dumps({"query": "预算", "speaker": "李四", "day_from": "2026-10-30"}), db, cfg)
    assert "找到 1 条" in out and "王总" in out
    out = run_tool("search_transcripts", {"query": "没有的词"}, db, cfg)
    assert "没有找到" in out
    # substring fallback: FTS misses but LIKE hits
    out = run_tool("search_transcripts", {"query": "纪要发"}, db, cfg)
    assert "子串匹配" in out and "会议纪要" in out
    uid = db.ids[2]
    out = run_tool("get_context", {"utterance_id": uid, "before": 1, "after": 1}, db, cfg)
    assert f">> [#{uid}]" in out and out.count("\n") == 3
    assert run_tool("get_context", {"utterance_id": 424242}, db, cfg).startswith("ERROR:")


def test_tools_transcript_summary_days_speakers_notes(db, cfg):
    out = run_tool("get_transcript", {"day": "2026-10-30"}, db, cfg)
    assert "共 3 条" in out and "08:19 张三" in out and "2026-10-30 09:05" not in out
    out = run_tool("get_transcript", {"day": "2026-10-30", "hour_from": 9, "hour_to": 10}, db, cfg)
    assert "共 1 条" in out and "王总" in out
    assert "没有记录" in run_tool("get_transcript", {"day": "2026-10-30", "hour_from": 20}, db, cfg)
    assert "没有任何记录" in run_tool("get_transcript", {"day": "2020-01-01"}, db, cfg)
    assert run_tool("get_transcript", {"day": "bad"}, db, cfg).startswith("ERROR:")
    assert run_tool("get_transcript", {}, db, cfg).startswith("ERROR:")

    assert "还没有生成日报" in run_tool("get_daily_summary", {"day": "2026-10-30"}, db, cfg)
    db.execute("INSERT INTO daily_summaries(day, summary_md, model, utterance_count) VALUES (?,?,?,?)",
               ("2026-10-30", "# 概览\n讨论了项目进度和预算。", "mimo-v2.6-flash", 3))
    out = run_tool("get_daily_summary", {"day": "2026-10-30"}, db, cfg)
    assert "讨论了项目进度和预算" in out and "3 条记录" in out

    out = run_tool("list_days", {}, db, cfg)
    lines = out.splitlines()
    assert lines[1].startswith("2026-10-31") and lines[2].startswith("2026-10-30")
    assert "3 条" in lines[2] and "有日报" in lines[2] and "有日报" not in lines[1]

    out = run_tool("list_speakers", {}, db, cfg)
    assert "张三" in out and "李四" in out and "同事" in out and "未知" in out and "1 条" in out

    assert "没有找到" in run_tool("search_notes", {"query": "预算"}, db, cfg)
    db.execute("INSERT INTO memory_notes(kind, title, content, day_from, day_to) VALUES (?,?,?,?,?)",
               ("todo", "确认预算", "和王总开会前确认预算数字", "2026-10-30", "2026-10-30"))
    out = run_tool("search_notes", {"query": "预算", "kind": "todo"}, db, cfg)
    assert "[todo] 确认预算 (2026-10-30)" in out
    assert "没有找到" in run_tool("search_notes", {"query": "预算", "kind": "person"}, db, cfg)
    assert run_tool("search_notes", {"kind": "weird"}, db, cfg).startswith("ERROR:")


def test_tools_never_raise_and_cap_length(db, cfg):
    assert run_tool("nope", {}, db, cfg).startswith("ERROR:")
    assert run_tool("search_transcripts", "{not json", db, cfg).startswith("ERROR:")
    assert run_tool("search_transcripts", {}, db, cfg).startswith("ERROR:")
    assert run_tool("search_transcripts", [1, 2], db, cfg).startswith("ERROR:")  # type: ignore[arg-type]
    seg = db.one("SELECT id FROM segments")["id"]
    long_text = "这是一段很长的发言，" * 200
    db.add_utterances([{
        "segment_id": seg, "start_s": 0.0, "end_s": 1.0, "abs_start": f"2026-11-02T0{i}:00:00", "day": "2026-11-02",
        "speaker_name": "张三", "text_raw": long_text, "text": long_text, "text_seg": tokenize(long_text),
    } for i in range(5)])
    out = run_tool("get_transcript", {"day": "2026-11-02"}, db, cfg)
    assert len(out) <= MAX_CHARS and "截断" in out
    out = run_tool("search_transcripts", {"query": "很长的发言", "limit": 50}, db, cfg)
    assert len(out) <= MAX_CHARS


# --- chat ------------------------------------------------------------------------
def _msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _resp(msg):
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=None)


def _call(cid, name, args):
    return SimpleNamespace(id=cid, type="function", function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)))


class FakeClient:
    """Scripted chat(): returns the queued responses in order and records every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages, **kw):
        self.calls.append({"messages": [dict(m) for m in messages], **kw})
        if self.responses:
            return _resp(self.responses.pop(0))
        return _resp(_msg("fallback answer"))

    def text(self, *a, **k):  # pragma: no cover - interface completeness
        raise AssertionError("not used")

    def json(self, *a, **k):  # pragma: no cover
        raise AssertionError("not used")

    def transcribe_bytes(self, *a, **k):  # pragma: no cover
        raise AssertionError("not used")


def test_chat_tool_loop(db, cfg):
    client = FakeClient([
        _msg(None, [_call("call_1", "search_transcripts", {"query": "项目进度"})]),
        _msg("张三在 10-30 08:19 说要过一下项目进度，下周二前交付第一版。"),
    ])
    chat = MemoryChat(cfg, db, client)
    assert chat.model == cfg.mimo.chat_model
    seen = []
    chat.on_tool = lambda n, a, r: seen.append((n, a, r))
    answer = chat.ask("项目进度说了什么？")
    assert answer.startswith("张三在 10-30 08:19")
    assert len(client.calls) == 2
    first, second = client.calls
    assert first["tools"] == TOOLS and first["model"] == cfg.mimo.chat_model
    assert first["messages"][0]["role"] == "system" and datetime.now().strftime("%Y-%m-%d") in first["messages"][0]["content"]
    assert first["messages"][-1] == {"role": "user", "content": "项目进度说了什么？"}
    # second call carries the assistant tool-call turn and the tool result
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    asst = second["messages"][2]
    assert asst["tool_calls"][0]["id"] == "call_1" and asst["tool_calls"][0]["function"]["name"] == "search_transcripts"
    tool_msg = second["messages"][3]
    assert tool_msg["tool_call_id"] == "call_1" and "项目进度" in tool_msg["content"] and "张三" in tool_msg["content"]
    assert seen and seen[0][0] == "search_transcripts" and seen[0][1] == {"query": "项目进度"}


def test_chat_history_and_direct_answer(db, cfg):
    client = FakeClient([_msg("直接回答")])
    chat = MemoryChat(cfg, db, client)
    history = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]
    assert chat.ask("再见", history) == "直接回答"
    msgs = client.calls[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[1]["content"] == "你好" and msgs[3]["content"] == "再见"
    assert len(history) == 2  # not mutated


def test_chat_budget_exhausted_and_bad_tool(db, cfg):
    looping = [_msg(None, [_call(f"c{i}", "list_days", {}), _call(f"d{i}", "bogus_tool", {})]) for i in range(10)]
    client = FakeClient(looping + [_msg("最终回答")])
    chat = MemoryChat(cfg, db, client)
    chat.max_rounds = 2
    # the scripted 3rd response is still a tool call, but the final call is made without tools -> answered by the 3rd script
    client.responses = looping[:2] + [_msg("最终回答")]
    answer = chat.ask("有哪些天？")
    assert answer == "最终回答"
    assert len(client.calls) == 3
    assert client.calls[0]["tools"] == TOOLS and client.calls[1]["tools"] == TOOLS
    assert client.calls[2]["tools"] is None
    last_msgs = client.calls[2]["messages"]
    assert last_msgs[-1]["role"] == "user" and "工具调用次数已用完" in last_msgs[-1]["content"]
    tool_msgs = [m for m in last_msgs if m["role"] == "tool"]
    assert len(tool_msgs) == 4
    assert any(m["content"].startswith("ERROR:") for m in tool_msgs)


def test_chat_trims_old_tool_results(db, cfg):
    seg = db.one("SELECT id FROM segments")["id"]
    long_text = "这是一段很长的发言，" * 300
    db.add_utterances([{
        "segment_id": seg, "start_s": 0.0, "end_s": 1.0, "abs_start": "2026-11-02T09:00:00", "day": "2026-11-02",
        "speaker_name": "张三", "text_raw": long_text, "text": long_text, "text_seg": tokenize(long_text),
    }])
    cfg["memory"]["max_context_chars"] = 5000  # each get_transcript result is ~3000 chars (attr access returns a copy)
    big = [_msg(None, [_call(f"c{i}", "get_transcript", {"day": "2026-11-02"})]) for i in range(3)]
    client = FakeClient(big + [_msg("ok")])
    chat = MemoryChat(cfg, db, client)
    chat.max_rounds = 5
    assert chat.ask("q") == "ok"
    final_tool_msgs = [m for m in client.calls[-1]["messages"] if m["role"] == "tool"]
    assert len(final_tool_msgs) == 3
    assert "已省略" in final_tool_msgs[0]["content"] and "已省略" in final_tool_msgs[1]["content"]
    assert "已省略" not in final_tool_msgs[2]["content"]  # newest result kept intact
    # the second round already saw the first result untouched (only one result so far, under budget)
    assert "已省略" not in [m for m in client.calls[1]["messages"] if m["role"] == "tool"][0]["content"]
