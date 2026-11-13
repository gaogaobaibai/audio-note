"""Tests for audio_note.analysis.daily with a fake MiMo client (no network)."""
from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from audio_note.analysis import prompts
from audio_note.analysis.daily import DailyAnalyzer, render_markdown
from audio_note.config import load_config
from audio_note.db import DB
from audio_note.llm.client import LLMError

DAY = "2026-10-30"


# ----------------------------------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------------------------------
@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data = tmp_path / "data"
    y = tmp_path / "config.yaml"
    y.write_text(
        f"paths:\n  data_dir: {json.dumps(str(data))}\nmimo:\n  api_key: ''\n  analysis_model: mimo-v2.6-pro\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AUDIO_NOTE_CONFIG", str(y))
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    c = load_config()
    c.ensure_dirs()
    return c


@pytest.fixture
def db(cfg):
    return DB(cfg.db_path)


class FakeClient:
    """Mimics MimoClient.json(); answers depend on which prompt (map or reduce) it is given."""

    def __init__(self, map_result=None, reduce_result=None, fail_reduce: bool = False):
        self.calls: list[dict] = []
        self.map_result = map_result or {
            "title": "项目进度同步", "summary": "讨论了项目进度和下周安排。",
            "topics": ["项目进度", "下周安排"], "decisions": ["周五前完成接口联调"],
            "todos": ["张三整理测试用例"], "people": ["张三", "李四"], "mood": "平静",
            "open_questions": ["服务器预算还没确定"],
        }
        self.reduce_result = reduce_result or {
            "overview": "今天主要和张三同步了项目进度。",
            "topics": ["项目进度", "下周安排"],
            "decisions": ["周五前完成接口联调"],
            "todos": [{"text": "整理测试用例", "owner": "张三", "due": "周三"}, "预约会议室"],
            "people": [{"name": "张三", "context": "负责测试用例"}, {"name": "李四", "context": "提到预算"}],
            "mood": "整体平静，下午略有焦虑",
            "open_loops": ["服务器预算还没确定"],
            "timeline": [{"time": "09:00", "event": "晨会"}, {"time": "14:00", "event": "和李四聊预算"}],
        }
        self.fail_reduce = fail_reduce

    def json(self, system: str, user: str, **kw):
        kind = "reduce" if "open_loops" in system else "map"
        self.calls.append({"kind": kind, "system": system, "user": user, "kw": kw})
        if kind == "reduce":
            if self.fail_reduce:
                raise LLMError("boom")
            return json.loads(json.dumps(self.reduce_result))
        return json.loads(json.dumps(self.map_result))

    # the other MimoClient methods exist so a stray call fails loudly instead of silently
    def chat(self, *a, **k):
        raise AssertionError("chat() should not be called")

    def text(self, *a, **k):
        raise AssertionError("text() should not be called")

    def transcribe_bytes(self, *a, **k):
        raise AssertionError("transcribe_bytes() should not be called")


def _insert(db: DB, items: list[tuple[datetime, str | None, str]], seg_dur: float = 3600.0) -> list[int]:
    """items: (abs_start, speaker_name, text). One segment per call."""
    seg = db.add_segment(f"seg_{len(db.query('SELECT id FROM segments'))}.opus", items[0][0], seg_dur)
    rows = []
    for i, (ts, who, text) in enumerate(items):
        rows.append({
            "segment_id": seg, "start_s": i * 5.0, "end_s": i * 5.0 + 4.0,
            "abs_start": ts.isoformat(timespec="seconds"), "day": ts.strftime("%Y-%m-%d"),
            "cluster_label": 0, "speaker_id": None, "speaker_name": who, "speaker_score": None,
            "text_raw": text, "text": text,
        })
    return db.add_utterances(rows)


def _two_sessions(db: DB) -> list[int]:
    t0 = datetime(2026, 10, 30, 9, 0, 0)
    ids = _insert(db, [
        (t0, "张三", "今天我们先过一下项目进度。"),
        (t0 + timedelta(minutes=1), None, "好的，接口联调周五前做完。"),
        (t0 + timedelta(minutes=10), "张三", "测试用例我来整理。"),
    ])
    t1 = datetime(2026, 10, 30, 14, 0, 0)
    ids += _insert(db, [
        (t1, "李四", "服务器预算还没批下来。"),
        (t1 + timedelta(minutes=2), "张三", "那先等财务回复。"),
    ])
    return ids


# ----------------------------------------------------------------------------------------------
# grouping
# ----------------------------------------------------------------------------------------------
def test_sessions_split_on_gap(cfg, db):
    _two_sessions(db)
    an = DailyAnalyzer(cfg, db, FakeClient())
    s = an.sessions(DAY)
    assert [len(x) for x in s] == [3, 2]
    assert s[0][0]["text"].startswith("今天")
    assert an.sessions("2026-10-29") == []


def test_blocks_respect_char_limit(cfg, db):
    t0 = datetime(2026, 10, 30, 10, 0, 0)
    _insert(db, [(t0 + timedelta(seconds=i * 10), "张三", "很长的一句话" * 200) for i in range(12)])  # 12 x ~1200 chars
    an = DailyAnalyzer(cfg, db, FakeClient())
    blocks = an.blocks(an.sessions(DAY))
    assert len(blocks) >= 3
    assert all(len(b.text) <= 6000 for b in blocks)
    assert sum(len(b.utterance_ids) for b in blocks) == 12
    assert blocks[0].text.startswith("10:00 张三：")
    assert [b.index for b in blocks] == list(range(1, len(blocks) + 1))


# ----------------------------------------------------------------------------------------------
# run()
# ----------------------------------------------------------------------------------------------
def test_run_without_utterances_skips_llm(cfg, db):
    client = FakeClient()
    an = DailyAnalyzer(cfg, db, client)
    md, js = an.run(DAY)
    assert "没有记录" in md and js["utterance_count"] == 0
    assert client.calls == []
    assert db.one("SELECT * FROM daily_summaries WHERE day=?", (DAY,)) is None

    md, js = an.run(DAY, force=True)
    row = db.one("SELECT * FROM daily_summaries WHERE day=?", (DAY,))
    assert row is not None and row["utterance_count"] == 0
    assert (cfg.daily_dir / f"{DAY}.md").read_text(encoding="utf-8") == md
    assert client.calls == []


def test_run_writes_markdown_db_and_notes(cfg, db):
    _two_sessions(db)
    client = FakeClient()
    an = DailyAnalyzer(cfg, db, client)
    md, js = an.run(DAY)

    # LLM usage: one map call per block (2 sessions -> 2 blocks) + one reduce, analysis model used
    kinds = [c["kind"] for c in client.calls]
    assert kinds == ["map", "map", "reduce"]
    assert all(c["kw"].get("model") == "mimo-v2.6-pro" for c in client.calls)
    # map prompt carries times + speaker labels, reduce prompt carries block results
    assert "09:00 张三：今天我们先过一下项目进度。" in client.calls[0]["user"]
    assert "09:01 未知：好的" in client.calls[0]["user"]
    assert "14:00 李四：" in client.calls[1]["user"]
    assert "项目进度同步" in client.calls[2]["user"]

    # markdown sections
    for sec in ("# 2026-10-30 每日记录", "## 概览", "## 主要话题", "## 决定", "## 待办", "## 提到的人", "## 情绪", "## 未闭环", "## 时间线"):
        assert sec in md
    assert "- [ ] 整理测试用例（负责人：张三，截止：周三）" in md
    assert "- [ ] 预约会议室" in md
    assert "**张三**：负责测试用例" in md
    assert "- 09:00 晨会" in md
    assert "共 5 条记录" in md and "2 个时段" in md and "说话人：张三、李四" in md

    # json meta
    assert js["day"] == DAY and js["utterance_count"] == 5 and js["session_count"] == 2 and js["block_count"] == 2
    assert js["speakers"] == ["张三", "李四"]
    assert js["todos"][1] == {"text": "预约会议室"}

    # file + daily_summaries row
    assert (cfg.daily_dir / f"{DAY}.md").read_text(encoding="utf-8") == md
    row = db.one("SELECT * FROM daily_summaries WHERE day=?", (DAY,))
    assert row["utterance_count"] == 5 and row["model"] == "mimo-v2.6-pro" and row["pushed_at"] is None
    assert json.loads(row["summary_json"])["overview"] == js["overview"]

    # memory notes (1.0: written to memory_items; valid_from / day / evidence are the old day_from / day_to / source ids)
    notes = db.query("SELECT kind, title, content, valid_from AS day_from, day AS day_to, evidence AS source_utterance_ids,"
                     " source, status, owner FROM memory_items ORDER BY kind, title")
    by = {(n["kind"], n["title"]): n for n in notes}
    assert {n["source"] for n in notes} == {"digest"} and {n["status"] for n in notes} == {"active"}
    assert by[("todo", "整理测试用例")]["owner"] == "张三"
    assert db.one("SELECT COUNT(*) AS n FROM memory_notes")["n"] == 0          # the 0.3 table is no longer written
    assert set(by) == {("todo", "整理测试用例"), ("todo", "预约会议室"), ("decision", "周五前完成接口联调"),
                       ("person", "张三"), ("person", "李四")}
    assert by[("todo", "整理测试用例")]["content"] == "整理测试用例（负责人：张三，截止：周三）"
    assert by[("person", "张三")]["day_from"] == DAY and by[("person", "张三")]["day_to"] == DAY
    assert len(json.loads(by[("person", "张三")]["source_utterance_ids"])) == 3  # the three 张三 utterances


def test_rerun_uses_cache_unless_forced_and_dedupes_notes(cfg, db):
    _two_sessions(db)
    client = FakeClient()
    an = DailyAnalyzer(cfg, db, client)
    md1, _ = an.run(DAY)
    n_calls = len(client.calls)

    # same utterances, no force -> cached, no LLM
    md2, js2 = an.run(DAY)
    assert md2 == md1 and js2["utterance_count"] == 5 and len(client.calls) == n_calls

    # force -> re-analysed with updated reduce output; notes updated in place, not duplicated
    client.reduce_result["people"][0]["context"] = "负责测试用例和联调"
    client.reduce_result["todos"].append({"text": "写周报"})
    md3, _ = an.run(DAY, force=True)
    assert len(client.calls) == n_calls * 2
    assert "负责测试用例和联调" in md3
    assert db.one("SELECT COUNT(*) AS n FROM memory_items WHERE kind='person' AND title='张三'")["n"] == 1
    assert db.one("SELECT content FROM memory_items WHERE kind='person' AND title='张三'")["content"] == f"{DAY}：负责测试用例和联调"
    assert db.one("SELECT COUNT(*) AS n FROM memory_items")["n"] == 6
    assert db.one("SELECT COUNT(*) AS n FROM daily_summaries")["n"] == 1

    # a later day touching the same person extends day_to, keeps day_from
    _insert(db, [(datetime(2026, 10, 31, 8, 0, 0), "张三", "继续联调。")])
    an.run("2026-10-31")
    p = db.one("SELECT valid_from AS day_from, day AS day_to, content FROM memory_items WHERE kind='person' AND title='张三'")
    assert (p["day_from"], p["day_to"]) == (DAY, "2026-10-31")
    # 1.0: a person item keeps one dated line per day instead of being overwritten by the latest digest
    assert p["content"].split("\n") == [f"{DAY}：负责测试用例和联调", "2026-10-31：负责测试用例和联调"]


def test_reduce_failure_falls_back_to_block_concat(cfg, db):
    _two_sessions(db)
    an = DailyAnalyzer(cfg, db, FakeClient(fail_reduce=True))
    md, js = an.run(DAY)
    assert "讨论了项目进度和下周安排。" in js["overview"]
    assert js["todos"] == [{"text": "张三整理测试用例"}]
    assert "## 时间线" in md and "09:00 项目进度同步" in md


def test_run_without_client_needs_api_key(cfg, db):
    _two_sessions(db)
    from audio_note.llm.client import MissingAPIKey

    with pytest.raises(MissingAPIKey):
        DailyAnalyzer(cfg, db).run(DAY)


# ----------------------------------------------------------------------------------------------
# render_markdown
# ----------------------------------------------------------------------------------------------
def test_render_markdown_tolerates_missing_and_odd_shapes():
    md = render_markdown("2026-01-01", {"overview": "x", "topics": "单个字符串", "todos": [{"text": "a"}, "b"],
                                        "people": ["王五"], "timeline": ["08:00 起床"]})
    assert md.startswith("# 2026-01-01 每日记录")
    assert "- 单个字符串" in md and "- [ ] a" in md and "- [ ] b" in md and "**王五**" in md and "- 08:00 起床" in md
    assert md.count("（无）") == 3  # 决定 / 情绪 / 未闭环 empty
    assert render_markdown("2026-01-01", {}).count("（无）") == 8


def test_prompts_mention_schema_keys():
    s, u = prompts.map_prompt("2026-01-01", 1, 2, "09:00", "09:30", "09:00 张三：你好")
    assert "open_questions" in s and "09:00 张三：你好" in u and "第 1/2 块" in u
    s, u = prompts.reduce_prompt("2026-01-01", [{"title": "t"}], utterance_count=3, session_count=1, speakers=["张三"])
    assert "open_loops" in s and "timeline" in s and '"title": "t"' in u and "张三" in u


# ----------------------------------------------------------------------------------------------
# push()
# ----------------------------------------------------------------------------------------------
def _fake_push_module(monkeypatch: pytest.MonkeyPatch, results: dict[str, str | None]) -> list:
    sent: list = []

    class Message:
        def __init__(self, title: str, body_md: str, url: str | None = None):
            self.title, self.body_md, self.url = title, body_md, url

    class Notifier:
        def __init__(self, cfg):
            self.cfg = cfg

        def send(self, msg):
            sent.append(msg)
            return dict(results)

    mod = types.ModuleType("audio_note.push.base")
    mod.Message, mod.Notifier = Message, Notifier
    monkeypatch.setitem(sys.modules, "audio_note.push.base", mod)
    return sent


def test_push_sets_pushed_at_on_success(cfg, db, monkeypatch):
    _two_sessions(db)
    an = DailyAnalyzer(cfg, db, FakeClient())
    sent = _fake_push_module(monkeypatch, {"toast": None, "feishu": "HTTP 500"})
    res = an.push(DAY)  # no summary yet -> run() first
    assert res == {"toast": None, "feishu": "HTTP 500"}
    assert len(sent) == 1 and sent[0].title == f"每日记录 {DAY}" and "## 概览" in sent[0].body_md
    assert db.one("SELECT pushed_at FROM daily_summaries WHERE day=?", (DAY,))["pushed_at"]


def test_push_leaves_pushed_at_when_all_fail(cfg, db, monkeypatch):
    _two_sessions(db)
    an = DailyAnalyzer(cfg, db, FakeClient())
    an.run(DAY)
    _fake_push_module(monkeypatch, {"toast": "no toast"})
    assert an.push(DAY) == {"toast": "no toast"}
    assert db.one("SELECT pushed_at FROM daily_summaries WHERE day=?", (DAY,))["pushed_at"] is None


# ----------------------------------------------------------------------------------------------
# review fix: the cache also keys on the people side (names / self / relations), not just the count
# ----------------------------------------------------------------------------------------------
def test_rerun_refreshes_when_people_change_and_keeps_legacy_digests(cfg, db):
    _two_sessions(db)
    client = FakeClient()
    an = DailyAnalyzer(cfg, db, client)
    _, js1 = an.run(DAY)
    n_calls = len(client.calls)
    assert isinstance(js1.get("people_fp"), str) and len(js1["people_fp"]) == 40
    assert an.run(DAY)[1]["people_fp"] == js1["people_fp"] and len(client.calls) == n_calls and an.last_run_cached
    # a voice gets named: same utterance count, different (speaker, count) pairs -> regenerated
    db.execute("UPDATE utterances SET speaker_name='王五' WHERE speaker_name IS NULL")
    _, js2 = an.run(DAY)
    assert len(client.calls) == 2 * n_calls and not an.last_run_cached and js2["people_fp"] != js1["people_fp"]
    # the user marks somebody as self -> the 「人物」 line changes -> regenerated again
    sid = db.upsert_speaker("张三")
    db.execute("UPDATE utterances SET speaker_id=? WHERE speaker_name='张三'", (sid,))
    db.set_self_speaker(sid)
    _, js3 = an.run(DAY)
    assert len(client.calls) == 3 * n_calls and js3["people_fp"] != js2["people_fp"] and js3["self_name"] == "张三"
    assert "张三" in client.calls[-1]["user"]
    an.run(DAY)
    assert len(client.calls) == 3 * n_calls and an.last_run_cached
    # a digest stored before the fingerprint existed is still trusted while the count matches
    row = json.loads(db.one("SELECT summary_json FROM daily_summaries WHERE day=?", (DAY,))["summary_json"])
    row.pop("people_fp")
    db.execute("UPDATE daily_summaries SET summary_json=? WHERE day=?", (json.dumps(row, ensure_ascii=False), DAY))
    db.set_relation(sid, "同事", "user")
    _, js4 = an.run(DAY)
    assert js4.get("people_fp") is None and len(client.calls) == 3 * n_calls and an.last_run_cached
    assert an.run(DAY, force=True)[1]["people_fp"] and len(client.calls) == 4 * n_calls
