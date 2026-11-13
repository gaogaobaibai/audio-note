"""audio_note.speech.pinyin: sound-alike comparison (SPEC v1 section 6.1). No network, no database."""
from __future__ import annotations

import pytest

from audio_note.speech import pinyin


needs_pypinyin = pytest.mark.skipif(not pinyin.available(), reason="pypinyin is not installed")


@needs_pypinyin
def test_key_is_toneless_syllables_and_keeps_latin_runs_lowercase():
    assert pinyin.key("灰度发布") == "hui du fa bu"
    assert pinyin.key("星河App") == "xing he app"
    assert pinyin.key("T30 Pro！") == "t30 pro"
    assert pinyin.key("银行行长") == "yin hang hang zhang"          # the reading in context, not the first in the dictionary
    assert pinyin.key("") == "" and pinyin.key("，。！") == ""
    # same sound, different characters -> same key (what the inbox dedupes on)
    assert pinyin.key("星河") == pinyin.key("兴和") != pinyin.key("新河")


@needs_pypinyin
@pytest.mark.parametrize("a,b,expected", [
    ("星河", "兴和", 1.0),            # homophones
    ("周兰", "周岚", 1.0),
    ("牙测", "压测", 1.0),            # tones do not count
    ("四十", "事实", 0.75),           # s/sh: one near-sound syllable of two = half a step
    ("蓝天", "南天", 0.75),           # n/l
    ("陈总", "程总", 0.75),           # -n / -ng
    ("世界", "世间", 0.5),            # one unrelated syllable of two
    ("灰度发布", "回顾发布", 0.75),   # one unrelated syllable of four
    ("星河", "苹果", 0.0),
])
def test_similarity_counts_syllable_steps(a, b, expected):
    assert pinyin.similarity(a, b) == pytest.approx(expected, abs=1e-4)
    assert pinyin.similarity(b, a) == pytest.approx(expected, abs=1e-4)


@needs_pypinyin
def test_close_uses_the_calibrated_thresholds():
    assert pinyin.DEFAULT_MIN_SIM == 0.75 and pinyin.CANDIDATE_MIN_SIM == 0.66
    assert pinyin.close("星河", "兴和") and pinyin.close("四十", "事实") and pinyin.close("陈总", "程总")
    assert not pinyin.close("世界", "世间")
    # three syllables with one unrelated one: a candidate, not "the same word"
    assert pinyin.similarity("南湖路", "南山路") == pytest.approx(0.6667, abs=1e-3)
    assert not pinyin.close("南湖路", "南山路") and pinyin.close("南湖路", "南山路", pinyin.CANDIDATE_MIN_SIM)
    # an explicit threshold is honoured exactly at the boundary
    assert pinyin.close("世界", "世间", 0.5) and not pinyin.close("世界", "世间", 0.51)


def test_latin_runs_are_compared_by_their_letters():
    assert pinyin.similarity("OKR", "okr") == 1.0                    # case does not matter
    assert pinyin.similarity("CDN", "CBN") == pytest.approx(0.6667, abs=1e-3)
    assert not pinyin.close("CDN", "CBN") and pinyin.close("CDN", "CBN", pinyin.CANDIDATE_MIN_SIM)
    assert pinyin.similarity("T30", "X7") == 0.0


def test_edge_cases_never_raise():
    assert pinyin.similarity("", "") == 0.0 and pinyin.similarity("词", "") == 0.0
    assert pinyin.similarity("。", "，") == 0.0                     # nothing to compare
    assert pinyin.similarity("同一个词", "同一个词") == 1.0
    assert pinyin.close(None, None) is False                         # type: ignore[arg-type]
    assert pinyin.units("") == ()


@needs_pypinyin
def test_units_carry_character_spans_and_stored_keys_can_override_a_reading():
    us = pinyin.units("去星河App吧")
    assert [(u.syl, u.start, u.end, u.latin) for u in us] == [
        ("qu", 0, 1, False), ("xing", 1, 2, False), ("he", 2, 3, False), ("app", 3, 6, True), ("ba", 6, 7, False)]
    alt = pinyin.units_from_key("shan zong okr")
    assert [(u.syl, u.latin) for u in alt] == [("shan", False), ("zong", False), ("okr", True)]
    assert pinyin.similarity_units(alt[:2], pinyin.units("山总")) == 1.0


def test_split_syllable_and_cost_table():
    assert pinyin.split_syllable("zhang") == ("zh", "ang") and pinyin.split_syllable("an") == ("", "an")
    assert pinyin.syllable_cost("zi", "zhi") == pinyin.NEAR_COST == 0.5
    assert pinyin.syllable_cost("lan", "nang") == 0.5               # initial and final both near
    assert pinyin.syllable_cost("ba", "pa") == 1.0                  # aspiration pairs are not near (measured: too many false friends)
    assert pinyin.syllable_cost("ma", "ma") == 0.0


def test_without_pypinyin_the_module_degrades_to_characters(monkeypatch):
    monkeypatch.setattr(pinyin, "_pypinyin", False)
    for fn in (pinyin.units, pinyin._context_syllables, pinyin._char_syllable):
        fn.cache_clear()
    try:
        assert pinyin.available() is False
        assert pinyin.key("星河App") == "星 河 app"
        assert pinyin.similarity("星河", "星河") == 1.0
        assert pinyin.similarity("星河", "兴和") == 0.0             # no sound without the library: characters only
        assert pinyin.similarity("星河湾", "星河弯") == pytest.approx(0.6667, abs=1e-3)
        assert not pinyin.close("星河", "兴和")
    finally:
        for fn in (pinyin.units, pinyin._context_syllables, pinyin._char_syllable):
            fn.cache_clear()
