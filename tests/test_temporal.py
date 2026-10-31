"""Relative-date parsing (memory/temporal.py) and its use by the chat prefetch."""
from __future__ import annotations

from datetime import date, datetime

import pytest

from audio_note.memory.temporal import parse_range

NOW = datetime(2026, 10, 30, 21, 0)   # Friday


@pytest.mark.parametrize("text,expected", [
    ("昨天晚上在家聊了什么", ("2026-10-29", "2026-10-29")),
    ("昨晚聊了什么", ("2026-10-29", "2026-10-29")),
    ("前天中午吃的什么", ("2026-10-28", "2026-10-28")),
    ("大前天的会", ("2026-10-27", "2026-10-27")),
    ("今天开了几个会", ("2026-10-30", "2026-10-30")),
    ("今晚吃什么", ("2026-10-30", "2026-10-30")),
    ("明天几点到公司", ("2026-10-31", "2026-10-31")),
    ("周三早会说了什么", ("2026-10-28", "2026-10-28")),       # most recent Wednesday
    ("周五的安排", ("2026-10-30", "2026-10-30")),             # today counts
    ("周六打算干什么", ("2026-10-24", "2026-10-24")),         # not yet this week -> last Saturday
    ("上周三聊了什么", ("2026-10-21", "2026-10-21")),
    ("这周一定了什么", ("2026-10-26", "2026-10-26")),
    ("本周日有安排吗", ("2026-11-01", "2026-11-01")),
    ("下周一全量上线吗", ("2026-11-02", "2026-11-02")),
    ("上上周五", ("2026-10-16", "2026-10-16")),
    ("上周都干了什么", ("2026-10-19", "2026-10-25")),
    ("这周的bug数", ("2026-10-26", "2026-10-30")),           # capped at today
    ("最近三天的午饭", ("2026-10-28", "2026-10-30")),
    ("过去7天", ("2026-10-24", "2026-10-30")),
    ("最近几天聊了什么", ("2026-10-24", "2026-10-30")),
    ("这几天bug怎么变化的", ("2026-10-24", "2026-10-30")),
    ("最近有什么决定", ("2026-10-24", "2026-10-30")),
    ("这两天", ("2026-10-29", "2026-10-30")),
    ("10月28号聊了什么", ("2026-10-28", "2026-10-28")),
    ("10月28日", ("2026-10-28", "2026-10-28")),
    ("2026-10-27 的会", ("2026-10-27", "2026-10-27")),
    ("2025年10月27日", ("2025-10-27", "2025-10-27")),
    ("12月25号的事", ("2025-12-25", "2025-12-25")),        # in the future -> last year
    ("上个月的开销", ("2026-09-01", "2026-09-30")),
    ("这个月", ("2026-10-01", "2026-10-30")),
    ("昨天和前天", ("2026-10-29", "2026-10-29")),          # first expression wins
    ("扫地机器人买的什么牌子", None),
    ("", None),
    ("13月40号", None),
])
def test_parse_range(text, expected):
    assert parse_range(text, NOW) == expected


def test_parse_range_accepts_date_and_defaults_to_today():
    assert parse_range("昨天", date(2026, 1, 1)) == ("2025-12-31", "2025-12-31")
    today = date.today().isoformat()
    assert parse_range("今天") == (today, today)


def test_prefetch_scopes_to_parsed_day(cfg_factory):
    from datetime import datetime as dt
    from types import SimpleNamespace

    from audio_note.db import DB
    from audio_note.memory.chat import MemoryChat
    from audio_note.memory.index import tokenize

    cfg = cfg_factory({"memory": {"embedding_model": "hash", "retrieval": "hybrid", "prefetch_k": 3}})
    db = DB(cfg.db_path)
    seg = db.add_segment(cfg.audio_dir / "x.opus", dt(2026, 10, 27, 20, 0), 60.0)
    rows = []
    for i, (day, text) in enumerate([
        ("2026-10-27", "今晚聊聊妈的生日礼物，科沃斯。"),
        ("2026-10-28", "今晚聊聊蛋糕，八寸黑森林。"),
        ("2026-10-29", "今晚聊聊海底捞包间。"),
        ("2026-10-29", "扫地机器人我今天拆了，挺好用的。"),
    ]):
        rows.append({"segment_id": seg, "start_s": float(i), "end_s": i + 0.5, "abs_start": f"{day}T20:30:0{i}", "day": day,
                     "speaker_name": "我", "text_raw": text, "text": text, "text_seg": tokenize(text)})
    db.add_utterances(rows)

    class C:
        def chat(self, messages, **kw):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))], usage=None)

    chat = MemoryChat(cfg, db, C())
    chat.now = dt(2026, 10, 30, 21, 0)
    text = chat.prefetch("昨天晚上聊了什么")
    assert "2026-10-29 优先" in text
    first_two = text.splitlines()[1:3]
    assert all("2026-10-29" in line for line in first_two)
    assert "2026-10-29 优先" not in chat.prefetch("聊聊蛋糕")

    # memory notes matching a content token of the question ride along; nothing else -> no notes block
    db.execute("INSERT INTO memory_notes(kind, title, content, day_from, day_to) VALUES (?,?,?,?,?)",
               ("decision", "生日礼物", "妈的生日礼物买科沃斯 T30", "2026-10-27", "2026-10-27"))
    db.execute("INSERT INTO memory_notes(kind, title, content, day_from, day_to) VALUES (?,?,?,?,?)",
               ("todo", "压测", "QPS 目标一千", "2026-10-28", "2026-10-29"))
    text = chat.prefetch("生日礼物买了什么")
    assert "相关记忆笔记" in text and "[decision] 生日礼物 (2026-10-27): 妈的生日礼物买科沃斯 T30" in text
    assert "[todo] 压测" not in text
    assert chat.related_notes("压测目标") == ["[todo] 压测 (2026-10-28~2026-10-29): QPS 目标一千"]
    assert chat.related_notes("的") == [] and chat.related_notes("") == []
    # notes alone (no utterance hits) still produce a block
    db.execute("DELETE FROM utterances")
    assert "相关记忆笔记" in chat.prefetch("压测目标")
