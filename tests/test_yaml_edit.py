"""audio_note.ui.yaml_edit: comment-preserving edits of config.yaml.

Every edit is checked twice: the text keeps what it should (comments, ordering, line endings) and
`yaml.safe_load` of the result equals what the settings page asked for.
"""
from __future__ import annotations

import yaml

from audio_note.ui.yaml_edit import EditResult, flatten, render, set_values, split_value_comment

DOC = """# audio_note config
mimo:
  api_key: ""           # fill in
  chat_model: mimo-v2.6-flash   # cheap
  rpm: 90

analysis:
  daily_time: "22:30"   # HH:MM
  push_daily: true
push:
  channels: [toast]     # list
  feishu: {webhook: "", secret: ""}   # inline
  wecom:
    webhook: ""   # paste
ui:
  port: 8765
"""


def _load(text: str) -> dict:
    return yaml.safe_load(text) or {}


# --- scalars ----------------------------------------------------------------------------------

def test_scalar_replace_keeps_trailing_comment():
    res = set_values(DOC, {("mimo", "chat_model"): "mimo-v2.6-pro"})
    assert isinstance(res, EditResult) and not res.comments_lost and res.problems == []
    lines = res.text.split("\n")
    assert "  chat_model: mimo-v2.6-pro # cheap" in lines
    assert "# audio_note config" in lines and '  api_key: "" # fill in' not in lines   # untouched line is verbatim
    assert '  api_key: ""           # fill in' in lines
    data = _load(res.text)
    assert data["mimo"]["chat_model"] == "mimo-v2.6-pro"
    assert data["mimo"]["rpm"] == 90 and data["ui"]["port"] == 8765


def test_time_like_string_stays_a_string():
    res = set_values(DOC, {("analysis", "daily_time"): "23:05"})
    assert not res.comments_lost
    line = next(ln for ln in res.text.split("\n") if "daily_time" in ln)
    assert line.endswith("# HH:MM") and ("'23:05'" in line or '"23:05"' in line)
    assert _load(res.text)["analysis"]["daily_time"] == "23:05"     # not the integer 1385 (sexagesimal)


def test_bool_null_number_and_empty_rendering():
    assert render(True) == "true" and render(False) == "false"
    assert render(None) == "null"
    assert render(8765) == "8765" and render(0.45) == "0.45"
    assert render("") == "''"
    assert render("22:30") in ("'22:30'", '"22:30"')
    assert render(["toast", "feishu"]) == "[toast, feishu]"
    assert render({"webhook": "", "secret": "s"}) == "{webhook: '', secret: s}"
    res = set_values(DOC, {("analysis", "push_daily"): False, ("ui", "port"): 9000,
                           ("mimo", "api_key"): None, ("mimo", "rpm"): 1.5})
    assert not res.comments_lost
    data = _load(res.text)
    assert data["analysis"]["push_daily"] is False and data["ui"]["port"] == 9000
    assert data["mimo"]["api_key"] is None and data["mimo"]["rpm"] == 1.5
    assert "  push_daily: false" in res.text and "  api_key: null # fill in" in res.text


# --- nested paths -----------------------------------------------------------------------------

def test_nested_block_mapping_value():
    res = set_values(DOC, {("push", "wecom", "webhook"): "https://qyapi.weixin.qq.com/x"})
    assert not res.comments_lost
    assert "    webhook: https://qyapi.weixin.qq.com/x # paste" in res.text
    data = _load(res.text)
    assert data["push"]["wecom"] == {"webhook": "https://qyapi.weixin.qq.com/x"}
    assert data["push"]["feishu"] == {"webhook": "", "secret": ""}     # sibling untouched


def test_inline_flow_mapping_updated_and_comment_kept():
    res = set_values(DOC, {("push", "feishu", "webhook"): "https://open.feishu.cn/hook/1"})
    assert not res.comments_lost
    line = next(ln for ln in res.text.split("\n") if ln.startswith("  feishu:"))
    assert line.endswith("# inline") and line.startswith("  feishu: {")
    data = _load(res.text)
    assert data["push"]["feishu"] == {"webhook": "https://open.feishu.cn/hook/1", "secret": ""}
    # a second key of the same flow mapping in the same call
    res2 = set_values(res.text, {("push", "feishu", "secret"): "abc"})
    assert _load(res2.text)["push"]["feishu"] == {"webhook": "https://open.feishu.cn/hook/1", "secret": "abc"}
    assert "# inline" in res2.text


def test_empty_value_with_comment_gets_a_value_without_gluing_the_comment():
    # `device:   # null = default` is the natural way to leave a key empty; the comment must not become
    # part of the new value (which would fail the read-back and drop every comment in the file)
    text = "# keep\naudio:\n  device:   # null = default\n  gain: 1.0\n"
    res = set_values(text, {("audio", "device"): 3})
    assert not res.comments_lost and res.problems == []
    assert "  device: 3 # null = default" in res.text and "# keep" in res.text
    assert _load(res.text)["audio"] == {"device": 3, "gain": 1.0}


def test_scalar_becomes_block_mapping_when_nested_path_needed():
    text = "push:\n  feishu: old   # was a string\n  channels: [toast]\n"
    res = set_values(text, {("push", "feishu", "webhook"): "u"})
    assert not res.comments_lost
    assert _load(res.text) == {"push": {"feishu": {"webhook": "u"}, "channels": ["toast"]}}
    assert "  feishu: # was a string" in res.text and "    webhook: u" in res.text


def test_inline_flow_section_gets_nested_key():
    text = "# top\npush: {channels: [toast]}   # one-liner\nui:\n  port: 1\n"
    res = set_values(text, {("push", "feishu", "webhook"): "u"})
    assert not res.comments_lost
    assert _load(res.text) == {"push": {"channels": ["toast"], "feishu": {"webhook": "u"}}, "ui": {"port": 1}}
    assert "# top" in res.text and "# one-liner" in res.text


# --- missing keys / sections ------------------------------------------------------------------

def test_missing_key_is_appended_inside_its_section():
    res = set_values(DOC, {("mimo", "asr_model"): "mimo-v2.5-asr"})
    assert not res.comments_lost
    lines = res.text.split("\n")
    i_new = lines.index("  asr_model: mimo-v2.5-asr")
    assert lines[i_new - 1] == "  rpm: 90"                   # after the last key of the section
    assert lines[i_new + 1] == ""                             # the blank line before `analysis:` is kept
    assert lines.index("analysis:") > i_new
    data = _load(res.text)
    assert data["mimo"]["asr_model"] == "mimo-v2.5-asr" and data["analysis"]["daily_time"] == "22:30"


def test_missing_nested_key_and_missing_section_are_appended():
    res = set_values(DOC, {("push", "ntfy", "topic"): "notes", ("log", "level"): "DEBUG"})
    assert not res.comments_lost
    assert res.text.endswith("ui:\n  port: 8765\n\nlog:\n  level: DEBUG")
    data = _load(res.text)
    assert data["log"] == {"level": "DEBUG"}
    assert data["push"]["ntfy"] == {"topic": "notes"}
    assert "  ntfy:\n    topic: notes" in res.text
    # the new section lands after the existing content, not in the middle
    assert res.text.index("log:") > res.text.index("ui:")


def test_scalar_section_is_turned_into_a_mapping():
    text = "push: none   # disabled\nui:\n  port: 1\n"
    res = set_values(text, {("push", "channels"): ["toast"]})
    assert not res.comments_lost
    assert _load(res.text) == {"push": {"channels": ["toast"]}, "ui": {"port": 1}}
    assert res.text.startswith("push: # disabled\n  channels: [toast]\n")


# --- lists ------------------------------------------------------------------------------------

def test_list_replaces_inline_list():
    res = set_values(DOC, {("push", "channels"): ["toast", "feishu"]})
    assert not res.comments_lost
    assert "  channels: [toast, feishu] # list" in res.text
    assert _load(res.text)["push"]["channels"] == ["toast", "feishu"]
    res2 = set_values(res.text, {("push", "channels"): []})
    assert _load(res2.text)["push"]["channels"] == [] and "  channels: [] # list" in res2.text


def test_list_replaces_block_item_list():
    text = ("push:\n  channels:   # which\n    - toast\n    - feishu\n"
            "  feishu: {webhook: ''}\nui:\n  port: 1\n")
    res = set_values(text, {("push", "channels"): ["ntfy"]})
    assert not res.comments_lost
    assert "    - toast" not in res.text and "    - feishu" not in res.text
    assert "  channels: [ntfy] # which" in res.text
    assert _load(res.text) == {"push": {"channels": ["ntfy"], "feishu": {"webhook": ""}}, "ui": {"port": 1}}


# --- fallback / verification ------------------------------------------------------------------

def test_parse_back_mismatch_falls_back_to_structured_rewrite():
    # a duplicate key: the line editor changes the first one, YAML keeps the last -> read-back differs
    text = "# header comment\nmimo:\n  chat_model: a   # first\n  chat_model: b\nui:\n  port: 1\n"
    res = set_values(text, {("mimo", "chat_model"): "c"})
    assert res.comments_lost is True
    assert any("读回不一致" in p for p in res.problems)
    assert "#" not in res.text                                   # comments are gone, values are right
    assert _load(res.text) == {"mimo": {"chat_model": "c"}, "ui": {"port": 1}}


def test_broken_flow_section_falls_back_without_raising():
    text = "mimo: {api_key: '',\n  chat_model: a}\nui:\n  port: 1\n"
    res = set_values(text, {("mimo", "chat_model"): "c"})
    assert res.comments_lost is True and res.problems
    assert _load(res.text)["mimo"]["chat_model"] == "c" and _load(res.text)["ui"] == {"port": 1}


def test_short_path_is_reported_and_handled_by_fallback():
    res = set_values(DOC, {("ui",): 5})
    assert res.comments_lost is True
    assert any("路径至少两级" in p for p in res.problems)
    assert _load(res.text)["ui"] == 5


def test_no_updates_returns_text_unchanged():
    res = set_values(DOC, {})
    assert res.text == DOC and not res.comments_lost and res.problems == []


def test_edit_is_idempotent():
    upd = {("mimo", "chat_model"): "x", ("push", "channels"): ["a"], ("push", "feishu", "secret"): "s", ("log", "level"): "DEBUG"}
    once = set_values(DOC, upd)
    twice = set_values(once.text, upd)
    assert once.text == twice.text and not twice.comments_lost


def test_crlf_line_endings_are_preserved():
    text = DOC.replace("\n", "\r\n")
    res = set_values(text, {("mimo", "chat_model"): "x", ("log", "level"): "DEBUG"})
    assert not res.comments_lost
    assert "\r\n" in res.text
    assert res.text.count("\n") == res.text.count("\r\n")      # no bare \n crept in (also for appended lines)
    data = _load(res.text)
    assert data["mimo"]["chat_model"] == "x" and data["log"] == {"level": "DEBUG"}
    # and the fallback keeps CRLF too
    bad = "mimo:\r\n  chat_model: a\r\n  chat_model: b\r\n"
    fb = set_values(bad, {("mimo", "chat_model"): "c"})
    assert fb.comments_lost and "\r\n" in fb.text and fb.text.count("\n") == fb.text.count("\r\n")


# --- helpers ----------------------------------------------------------------------------------

def test_split_value_comment_is_quote_and_brace_aware():
    assert split_value_comment('"a # b" # real') == ('"a # b"', " # real")
    assert split_value_comment("'x#y' #c") == ("'x#y'", " #c")
    assert split_value_comment("[a, b] # c") == ("[a, b]", " # c")
    assert split_value_comment('{x: "#"}') == ('{x: "#"}', "")
    assert split_value_comment("{x: [1, 2]}  # d") == ("{x: [1, 2]}", " # d")
    assert split_value_comment("a#b") == ("a#b", "")               # '#' needs a space before it
    assert split_value_comment("#only") == ("", " #only")          # blank before '#' restored (see _KEY_RE)
    assert split_value_comment("value   ") == ("value", "")
    assert split_value_comment("") == ("", "")


def test_flatten_stops_at_three_levels_and_keeps_lists():
    d = {"a": {"b": 1, "c": {"d": 2}}, "e": [1, 2], "f": {"g": {"h": {"i": 1}}}, "k": None}
    assert flatten(d) == {("a", "b"): 1, ("a", "c", "d"): 2, ("e",): [1, 2], ("f", "g", "h"): {"i": 1}, ("k",): None}
    assert flatten({}) == {} and flatten(None) == {}
    assert flatten({1: {2: 3}}) == {("1", "2"): 3}                 # keys become strings
