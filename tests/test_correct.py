"""Tests for audio_note.speech.correct: guardrails, batching, failure path, glossary. No network."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from audio_note.config import load_config
from audio_note.llm.client import LLMError
from audio_note.speech.correct import (
    Corrector,
    apply_edits,
    build_system_prompt,
    digit_runs,
    levenshtein,
    load_glossary,
    parse_edits,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Config whose data dir is a temp directory; AUDIO_NOTE_CONFIG points at a temp yaml."""
    y = tmp_path / "config.yaml"
    y.write_text(
        "paths:\n  data_dir: " + json.dumps(str(tmp_path / "data")) + "\n"
        "mimo:\n  api_key: test\n"
        "correction:\n  max_chars_per_call: 10\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AUDIO_NOTE_CONFIG", str(y))
    return load_config()


class FakeClient:
    """Same method names as MimoClient; json() pops canned responses (or raises)."""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages, **kw):  # pragma: no cover - not used here
        raise AssertionError("chat() should not be called")

    def text(self, system, user, **kw):  # pragma: no cover
        raise AssertionError("text() should not be called")

    def transcribe_bytes(self, *a, **kw):  # pragma: no cover
        raise AssertionError("transcribe_bytes() should not be called")

    def json(self, system, user, **kw):
        self.calls.append({"system": system, "user": user, **kw})
        if not self.responses:
            return {"edits": []}
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------
def test_levenshtein():
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("", "abc") == 3
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("在见", "再见") == 1
    assert levenshtein("李思", "李四") == 1


def test_digit_runs():
    assert digit_runs("第3季度2024年") == ["3", "2024"]
    assert digit_runs("三十万") == ["三十万"]  # Chinese numeral runs are protected too


def test_apply_homophone_edit():
    text = "我们明天在见，记得把报告发给李思。"
    new, applied = apply_edits(text, [
        {"i": 0, "from": "在见", "to": "再见", "reason": "同音"},
        {"i": 0, "from": "李思", "to": "李四", "reason": "词表"},
    ])
    assert new == "我们明天再见，记得把报告发给李四。"
    assert [(e["from"], e["to"]) for e in applied] == [("在见", "再见"), ("李思", "李四")]
    assert all(set(e) == {"from", "to", "reason"} for e in applied)


def test_digit_change_dropped():
    text = "预算是30万，2023年完成，还有三个人。"
    new, applied = apply_edits(text, [
        {"from": "30万", "to": "31万"},          # digit value changes
        {"from": "2023年", "to": "2024年"},      # digit value changes
        {"from": "三个人", "to": "3个人"},        # introduces digits
        {"from": "30万", "to": "三十万"},        # removes digits
    ])
    assert new == text
    assert applied == []


def test_digit_kept_edit_allowed():
    # digits present but unchanged -> the edit around them is fine
    text = "第3季度在见。"
    new, applied = apply_edits(text, [{"from": "第3季度在见", "to": "第3季度再见"}])
    assert new == "第3季度再见。"
    assert len(applied) == 1


def test_non_substring_dropped():
    text = "今天天气不错。"
    new, applied = apply_edits(text, [{"from": "明天", "to": "后天"}])
    assert new == text
    assert applied == []


def test_empty_from_and_identity_ignored():
    text = "今天天气不错。"
    new, applied = apply_edits(text, [
        {"from": "", "to": "哈哈"},
        {"from": "天气", "to": "天气"},
        {"from": None, "to": "x"},
        "not a dict",
    ])
    assert new == text
    assert applied == []


def test_over_30pct_drops_all_edits_of_item():
    text = "今天天气不错，我们出去走走吧。"   # 15 chars -> budget 4.5
    edits = [
        {"from": "不错", "to": "很好"},                       # distance 2, fine alone
        {"from": "我们出去走走吧", "to": "大家一起去爬山怎么样"},  # large rewrite
    ]
    new, applied = apply_edits(text, edits)
    assert new == text, "the whole item must be reverted, not just the big edit"
    assert applied == []
    # the small edit alone is accepted
    new, applied = apply_edits(text, edits[:1])
    assert new == "今天天气很好，我们出去走走吧。"
    assert len(applied) == 1


def test_multi_occurrence_replaces_first_unreplaced():
    text = "在见，在见，在见。"
    new, applied = apply_edits(text, [{"from": "在见", "to": "再见"}])
    assert new == "再见，在见，在见。"
    new, applied = apply_edits(text, [{"from": "在见", "to": "再见"}, {"from": "在见", "to": "再见"}])
    assert new == "再见，再见，在见。"
    assert len(applied) == 2


def test_edited_span_not_edited_again():
    # after 在见->再见 the produced "再见" must not be targeted by a later edit
    text = "在见，大家明天都要过来。"
    new, applied = apply_edits(text, [{"from": "在见", "to": "再见"}, {"from": "再见", "to": "再会"}])
    assert new == "再见，大家明天都要过来。"
    assert len(applied) == 1
    # ...but an untouched occurrence elsewhere still can be (text long enough for the 30% budget)
    text = "在见，再见，大家明天都要过来。"
    new, applied = apply_edits(text, [{"from": "在见", "to": "再见"}, {"from": "再见", "to": "再会"}])
    assert new == "再见，再会，大家明天都要过来。"
    assert len(applied) == 2


def test_apply_edits_empty_inputs():
    assert apply_edits("", [{"from": "a", "to": "b"}]) == ("", [])
    assert apply_edits("abc", []) == ("abc", [])


def test_load_glossary(tmp_path: Path):
    p = tmp_path / "glossary.txt"
    p.write_text("# 人名\n李四\n\n  王五  \n# 术语\nMiMo\nsherpa-onnx  # 注释\n李四\n", encoding="utf-8")
    assert load_glossary(p) == ["李四", "王五", "MiMo", "sherpa-onnx"]
    assert load_glossary(tmp_path / "missing.txt") == []
    assert load_glossary(None) == []


def test_parse_edits_tolerant():
    ids = [0, 1]
    r = parse_edits({"edits": [{"i": "1", "from": "a", "to": "b"}, {"i": 0, "from": "c", "to": "d"},
                               {"i": 7, "from": "x", "to": "y"}, "junk"]}, ids)
    assert set(r) == {0, 1}
    assert r[1][0]["from"] == "a" and r[0][0]["from"] == "c"
    # bare list, float index
    r = parse_edits([{"i": 1.0, "from": "a", "to": "b"}], ids)
    assert list(r) == [1]
    # missing i with a single item
    r = parse_edits({"edits": [{"from": "a", "to": "b"}]}, [5])
    assert list(r) == [5]
    # missing i with several items -> dropped
    assert parse_edits({"edits": [{"from": "a", "to": "b"}]}, ids) == {}
    with pytest.raises(ValueError):
        parse_edits({"foo": 1}, ids)
    with pytest.raises(ValueError):
        parse_edits("nope", ids)


# ---------------------------------------------------------------------------
# Corrector with a fake client
# ---------------------------------------------------------------------------
def test_corrector_batches_by_max_chars(cfg):
    # max_chars_per_call = 10 ; items of 6 chars each -> one item per batch
    items = [
        {"i": 0, "speaker": "张三", "text": "我们明天在见。"},
        {"i": 1, "speaker": "未知", "text": "好的好的好的。"},
        {"i": 2, "speaker": "李四", "text": "报告发给李思。"},
    ]
    client = FakeClient([
        {"edits": [{"i": 0, "from": "在见", "to": "再见", "reason": "同音"}]},
        {"edits": []},
        {"edits": [{"i": 2, "from": "李思", "to": "李四", "reason": "词表"}]},
    ])
    corr = Corrector(cfg, client)
    assert [[it["i"] for it in b] for b in corr.batches(items)] == [[0], [1], [2]]
    out = corr.correct(items, ["李四", "MiMo"])
    assert len(client.calls) == 3
    assert [o["i"] for o in out] == [0, 1, 2]
    assert out[0]["text"] == "我们明天再见。"
    assert out[1] == {"i": 1, "text": "好的好的好的。", "edits": []}
    assert out[2]["text"] == "报告发给李四。"
    assert out[2]["edits"][0]["reason"] == "词表"
    # each call used JSON mode via client.json(system, user, max_tokens=...)
    assert all("max_tokens" in c for c in client.calls)
    # the user prompt of the first batch carries only item 0
    assert json.loads(client.calls[0]["user"].split("\n", 1)[1]) == [items[0]]


def test_corrector_batches_group_small_items(cfg):
    cfg["correction"]["max_chars_per_call"] = 20
    corr = Corrector(cfg, FakeClient([]))
    items = [{"i": k, "speaker": "", "text": "一二三四五六"} for k in range(5)]   # 6 chars each
    groups = corr.batches(items)
    assert [[it["i"] for it in b] for b in groups] == [[0, 1, 2], [3, 4]]
    # a single oversize item is its own batch
    items = [{"i": 0, "speaker": "", "text": "x" * 50}, {"i": 1, "speaker": "", "text": "y"}]
    assert [[it["i"] for it in b] for b in corr.batches(items)] == [[0], [1]]


def test_corrector_failure_returns_originals_for_that_batch(cfg):
    items = [
        {"i": 0, "speaker": "张三", "text": "我们明天在见。"},
        {"i": 1, "speaker": "张三", "text": "报告发给李思。"},
        {"i": 2, "speaker": "张三", "text": "下午三点开会。"},
    ]
    client = FakeClient([
        LLMError("boom"),
        {"edits": [{"i": 1, "from": "李思", "to": "李四"}]},
        "not json at all",   # unparsable shape -> originals, no raise
    ])
    out = Corrector(cfg, client).correct(items, [])
    assert out[0] == {"i": 0, "text": "我们明天在见。", "edits": []}
    assert out[1]["text"] == "报告发给李四。"
    assert out[2] == {"i": 2, "text": "下午三点开会。", "edits": []}


def test_corrector_guardrails_applied_to_model_output(cfg):
    cfg["correction"]["max_chars_per_call"] = 3000
    items = [{"i": 3, "speaker": "张三", "text": "预算30万，明天在见。"}]
    client = FakeClient([{"edits": [
        {"i": 3, "from": "30万", "to": "三十万", "reason": "digits -> dropped"},
        {"i": 3, "from": "后天", "to": "明天", "reason": "not substring -> dropped"},
        {"i": 3, "from": "在见", "to": "再见", "reason": "ok"},
    ]}])
    out = Corrector(cfg, client).correct(items, [])
    assert out == [{"i": 3, "text": "预算30万，明天再见。",
                    "edits": [{"from": "在见", "to": "再见", "reason": "ok"}]}]


def test_prompt_contains_glossary_speakers_and_rules(cfg):
    cfg["correction"]["max_chars_per_call"] = 3000
    # speaker name chosen so it does not occur in the few-shot examples of the system prompt
    items = [
        {"i": 0, "speaker": "赵六", "text": "你好。"},
        {"i": 1, "speaker": "未知", "text": "你好。"},
        {"i": 2, "speaker": "赵六", "text": "再见。"},
    ]
    client = FakeClient([{"edits": []}])
    Corrector(cfg, client).correct(items, ["MiMo", "sherpa-onnx"])
    system = client.calls[0]["system"]
    assert "MiMo" in system and "sherpa-onnx" in system
    assert "说话人：赵六" in system
    assert system.count("赵六") == 1           # deduplicated speaker list
    assert "未知" not in system.split("说话人：", 1)[1].split("\n", 1)[0]   # unknown label is not a name
    for kw in ("同音", "数字", "翻译", "不确定", "edits", "from"):
        assert kw in system
    # few-shot example present
    assert "示例" in system
    user = client.calls[0]["user"]
    assert "赵六" in user and "未知" in user and "你好" in user


def test_prompt_without_glossary(cfg):
    s = build_system_prompt([], [])
    assert "词表：（空）" in s
    assert "说话人：" not in s


def test_corrector_skips_llm_for_empty_text(cfg):
    client = FakeClient([])
    out = Corrector(cfg, client).correct([{"i": 0, "speaker": "", "text": ""}, {"i": 1, "speaker": "", "text": "  "}], [])
    assert out == [{"i": 0, "text": "", "edits": []}, {"i": 1, "text": "  ", "edits": []}]
    assert client.calls == []
    assert Corrector(cfg, client).correct([], []) == []


def test_corrector_disabled(cfg):
    cfg["correction"]["enabled"] = False
    client = FakeClient([{"edits": [{"i": 0, "from": "在见", "to": "再见"}]}])
    out = Corrector(cfg, client).correct([{"i": 0, "speaker": "", "text": "在见"}], [])
    assert out == [{"i": 0, "text": "在见", "edits": []}]
    assert client.calls == []


def test_chinese_numeral_change_is_dropped():
    """Seen in practice: glossary term 全开 replaced the number 五百."""
    from audio_note.speech.correct import apply_edits, digit_runs

    assert digit_runs("五百不够，灰度二零零零人至少要撑到一千QPS") == ["五百", "二零零零", "一千"]
    text = "五百不够，灰度二零零零人至少要撑到一千QPS。"
    new, applied = apply_edits(text, [{"from": "五百", "to": "全开", "reason": "x"}])
    assert new == text and applied == []
    # a lone 一 inside ordinary words is not treated as a number, so this homophone fix still applies
    new, applied = apply_edits("我们以下再说。", [{"from": "以下", "to": "一下", "reason": "同音"}])
    assert new == "我们一下再说。" and len(applied) == 1
