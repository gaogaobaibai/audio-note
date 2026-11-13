"""scripts/bench/textnorm.py + align.py: normalisation for CER, alignment, per-term error rate, time matching.
Pure functions only: no audio, no models, no network."""
from __future__ import annotations

import importlib
import importlib.util
import random
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "scripts" / "bench"
PKG = "audio_note_bench"


def bench(module: str):
    """scripts/bench is not on sys.path: load it as a package under a private name."""
    if PKG not in sys.modules:
        spec = importlib.util.spec_from_file_location(PKG, BENCH / "__init__.py", submodule_search_locations=[str(BENCH)])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[PKG] = pkg
        spec.loader.exec_module(pkg)
    return importlib.import_module(f"{PKG}.{module}")


textnorm = bench("textnorm")
align = bench("align")
N = textnorm.normalize


# --- normalisation ---------------------------------------------------------------------------------
@pytest.mark.parametrize("raw, want", [
    ("今天，T30 到手价 2,699 元！", "今天t30到手价2699元"),
    ("ＡＢＣ　ｄｅｆ １２３", "abcdef123"),
    ("  Hello,   World…… ", "helloworld"),
    ("“好的”——他说：‘行’。", "好的他说行"),
    ("", ""),
    (None, ""),
    ("。，！？", ""),
])
def test_punctuation_width_case_whitespace(raw, want):
    assert N(raw) == want


@pytest.mark.parametrize("cn, want", [
    ("两千六百九十九", "2699"),
    ("三千四百九十九", "3499"),
    ("十二万五", "125000"),
    ("三千五", "3500"),
    ("四千二", "4200"),
    ("一千八", "1800"),
    ("一百五", "150"),
    ("一百零五", "105"),
    ("两千零二十六", "2026"),
    ("十", "10"),
    ("十一", "11"),
    ("二十", "20"),
    ("二十三", "23"),
    ("十五万", "150000"),
    ("三万五千", "35000"),
    ("一亿二千万", "120000000"),
    ("二零二六", "2026"),
    ("零五", "05"),
    ("两", "2"),
    ("〇", "0"),
])
def test_chinese_numerals_become_digits(cn, want):
    assert textnorm.chinese_to_number(cn) == want
    assert N(cn) == want


@pytest.mark.parametrize("a, b", [
    ("到手价两千六百九十九", "到手价2699"),
    ("到手价两千六百九十九", "到手价2,699"),
    ("报价一年十二万五", "报价一年12.5万"),
    ("报价一年十二万五", "报价一年125000"),
    ("首期八万", "首期8万"),
    ("目标QPS三千五", "目标 qps 3500"),
    ("十一月九号", "11月9号"),
    ("二零二六年十一月", "2026年11月"),
    ("比例百分之五", "比例5%"),
    ("比例百分之十", "比例10％"),
    ("成功率百分之九十九点九", "成功率99.9%"),
    ("版本二点四", "版本2.4"),
    ("版本二点四点一", "版本2.4.1"),
    ("朗月T三十", "朗月T30"),
    ("出了P零", "出了P0"),
    ("还剩二十三个bug", "还剩23个BUG"),
    ("现在是零bug", "现在是0bug"),
    ("周六十点", "周六10点"),
    ("十二点五十", "12点50"),
    ("十二点五万", "12.5万"),
    ("三千块", "3千块"),
])
def test_equivalent_spellings_normalise_to_the_same_string(a, b):
    assert N(a) == N(b), (N(a), N(b))


def test_things_that_must_stay_different():
    assert N("首期八万") != N("首期十万")
    assert N("十一月九号") != N("十一月十八号")
    assert N("二点四") != N("二点四点一")
    assert N("百分之五") != N("百分之十")
    assert N("三点半") == "3点半"                 # a clock time is not a decimal
    assert N("十二点五十") == "12点50"
    assert N("3.14") == "3.14" and N("3.") == "3" and N("a.b") == "ab"
    assert N("100%正确") == "100%正确" and N("%") == ""


def test_numerals_can_be_switched_off():
    assert N("两千六百九十九元", numerals=False) == "两千六百九十九元"
    assert N("T30，２６９９", numerals=False) == "t302699"


def test_normalisation_is_idempotent():
    for s in ["今天，T30 到手价 2,699 元！", "十二万五，百分之十，二点四点一", "周六十点带小满打疫苗", "X7S 和 T30 Pro"]:
        assert N(N(s)) == N(s)


# --- alignment -------------------------------------------------------------------------------------
def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def test_align_basic_ops():
    assert align.align("abc", "abc") == [("=", 0, 0), ("=", 1, 1), ("=", 2, 2)]
    assert align.align("", "ab") == [("i", None, 0), ("i", None, 1)]
    assert align.align("ab", "") == [("d", 0, None), ("d", 1, None)]
    assert align.align("", "") == []
    ops = align.align("青禾app上线", "新核ip上线")
    c = align.counts(ops)
    assert c == {"n": 7, "hits": 3, "sub": 3, "del": 1, "ins": 0, "errors": 4}
    assert align.distance("kitten", "sitting") == 3
    assert align.cer("abcd", "abcd") == 0.0
    assert align.cer("abcd", "") == 1.0
    assert align.cer("", "abc") is None


def test_align_matches_reference_edit_distance_and_is_a_valid_path():
    rnd = random.Random(7)
    for _ in range(400):
        a = "".join(rnd.choice("甲乙丙丁ab") for _ in range(rnd.randint(0, 14)))
        b = "".join(rnd.choice("甲乙丙丁ab") for _ in range(rnd.randint(0, 14)))
        ops = align.align(a, b)
        c = align.counts(ops)
        assert c["errors"] == _lev(a, b) == align.distance(a, b)
        assert c["n"] == len(a) and c["hits"] + c["sub"] + c["ins"] == len(b)
        # every index visited exactly once, in order
        assert [i for _, i, _ in ops if i is not None] == list(range(len(a)))
        assert [j for _, _, j in ops if j is not None] == list(range(len(b)))
        for op, i, j in ops:
            if op == "=":
                assert a[i] == b[j]
            elif op == "s":
                assert a[i] != b[j]


def test_align_accepts_token_sequences():
    assert align.distance(["a", "bc", "d"], ["a", "d"]) == 1
    assert [op for op, _, _ in align.align(["x", "y"], ["x", "z", "y"])] == ["=", "i", "="]


def test_diff_regions_rebuild_the_new_string():
    rnd = random.Random(3)
    assert align.diff_regions("新核ip的上线", "青禾app的上线") == [(0, 3, "青禾ap")]
    assert align.diff_regions("abc", "abXc") == [(2, 2, "X")]
    assert align.diff_regions("abc", "ac") == [(1, 2, "")]
    assert align.diff_regions("same", "same") == []
    for _ in range(300):
        a = "".join(rnd.choice("abcd") for _ in range(rnd.randint(0, 12)))
        b = "".join(rnd.choice("abcd") for _ in range(rnd.randint(0, 12)))
        out, pos = [], 0
        for s, e, r in align.diff_regions(a, b):
            assert s >= pos and e >= s
            out += [a[pos:s], r]
            pos = e
        out.append(a[pos:])
        assert "".join(out) == b


def test_concat_and_attribute_split_errors_by_owner():
    ref, ref_owner = align.concat(["青禾灰度", "周五上线"])          # two truth lines
    hyp, hyp_owner = align.concat(["清河灰度啊", "周五上"])          # two utterances
    assert ref_owner == [0, 0, 0, 0, 1, 1, 1, 1] and len(hyp_owner) == len(hyp)
    ops = align.align(ref, hyp)
    by_ref, by_hyp = align.attribute(ops, ref_owner, hyp_owner)
    assert by_ref[0] == {"n": 4, "hits": 2, "sub": 2, "del": 0, "ins": 1}
    assert by_ref[1] == {"n": 4, "hits": 3, "sub": 0, "del": 1, "ins": 0}
    assert by_hyp[0]["sub"] == 2 and by_hyp[0]["ins"] == 1
    assert by_hyp[1]["del"] == 1 and by_hyp[1]["hits"] == 3
    total = align.counts(ops)
    assert sum(v["sub"] + v["del"] + v["ins"] for v in by_ref.values()) == total["errors"]
    assert sum(v["sub"] + v["del"] + v["ins"] for v in by_hyp.values()) == total["errors"]
    assert sum(v["n"] for v in by_ref.values()) == len(ref)


def test_attribute_with_an_empty_side():
    by_ref, by_hyp = align.attribute(align.align("ab", ""), [0, 1], [])
    assert by_ref == {0: {"n": 1, "hits": 0, "sub": 0, "del": 1, "ins": 0}, 1: {"n": 1, "hits": 0, "sub": 0, "del": 1, "ins": 0}}
    assert by_hyp == {}
    by_ref, by_hyp = align.attribute(align.align("", "xy"), [], [5, 5])
    assert by_ref == {} and by_hyp == {5: {"n": 2, "hits": 0, "sub": 0, "del": 0, "ins": 2}}


def test_ref_window_covers_adjacent_deletions():
    ref, hyp = "青禾app上线", "新核ip上线"
    ops = align.align(ref, hyp)
    h2r = align.hyp_to_ref(ops, len(hyp))
    lo, hi = align.ref_window(h2r, len(ref), 0, 4)        # "新核ip" answers for "青禾app" (one p was dropped)
    assert ref[lo:hi] == "青禾app"
    lo, hi = align.ref_window(h2r, len(ref), 4, 6)
    assert ref[lo:hi] == "上线"
    # an empty hypothesis range sees exactly the characters deleted at that point
    ops = align.align("abXYcd", "abcd")
    h2r = align.hyp_to_ref(ops, 4)
    assert align.ref_window(h2r, 6, 2, 2) == (2, 4)
    assert align.ref_window(h2r, 6, 0, 0) == (0, 0)
    assert align.ref_window(h2r, 6, 4, 4) == (6, 6)


# --- terms -----------------------------------------------------------------------------------------
def test_term_occurrences_respect_number_boundaries():
    assert align.term_occurrences("qps3500和13500", "3500") == [(3, 7)]
    assert align.term_occurrences("版本2.4和2.4.1", "2.4") == [(2, 5)]
    assert align.term_occurrences("10%和110%", "10%") == [(0, 3)]
    assert align.term_occurrences("t30和t30pro和x7s", "t30") == [(0, 3), (4, 7)]
    assert align.term_occurrences("x7s", "x7") == [(0, 2)]
    assert align.term_occurrences("青禾青禾", "青禾") == [(0, 2), (2, 4)]
    assert align.term_occurrences("abc", "") == []


def test_term_stats_counts_wrong_occurrences():
    ref = N("青禾App的灰度定在周五，青禾的DAU目标两万，朗月T30到手两千六百九十九")
    hyp = N("清河App的灰度定在周五，青禾的DAU目标两万，朗月T30到手两千六百九十")
    ops = align.align(ref, hyp)
    terms = [N(t) for t in ["青禾", "灰度", "DAU", "T30", "两千六百九十九", "陈默"]]
    st = align.term_stats(ref, ops, terms)
    assert st["青禾"] == {"n": 2, "wrong": 1}
    assert st["灰度"] == {"n": 1, "wrong": 0}
    assert st["dau"] == {"n": 1, "wrong": 0}
    assert st["t30"] == {"n": 1, "wrong": 0}
    assert st["2699"] == {"n": 1, "wrong": 1}
    assert st["陈默"] == {"n": 0, "wrong": 0}


def test_term_with_an_insertion_inside_is_wrong():
    ref, hyp = "看青禾的", "看青的禾的"
    st = align.term_stats(ref, align.align(ref, hyp), ["青禾"])
    assert st["青禾"] == {"n": 1, "wrong": 1}
    hits = align.term_hits(ref, align.align(ref, ref), ["青禾"], [0, 0, 0, 1])
    assert hits == [{"term": "青禾", "start": 1, "end": 3, "owner": 0, "ok": True}]
    # with owners a term is only searched inside one owner's text
    assert align.term_hits("看青禾的", align.align(ref, ref), ["青禾"], [0, 0, 1, 1]) == []
    two, owner = align.concat(["报价125000", "125000有点高"])
    assert align.term_stats(two, align.align(two, two), ["125000"], owner)["125000"] == {"n": 2, "wrong": 0}
    assert align.term_stats(two, align.align(two, two), ["125000"])["125000"] == {"n": 0, "wrong": 0}


# --- time matching ---------------------------------------------------------------------------------
def test_assign_lines_by_overlap():
    lines = [(0.6, 4.0), (5.0, 9.0), (8.5, 11.0)]
    utts = [(0.5, 4.2), (4.9, 8.8), (9.2, 11.1), (20.0, 21.0), (3.5, 6.0)]
    got = align.assign_lines(utts, lines)
    assert [g["line"] for g in got] == [0, 1, 2, None, 1]
    assert got[0]["share"] == pytest.approx(3.4 / 3.7)
    assert got[1]["others"] == [(2, pytest.approx(0.3))]
    assert got[3] == {"line": None, "overlap": 0.0, "share": 0.0, "others": []}
    assert got[4]["others"] == [(0, pytest.approx(0.5))]


def test_line_coverage_uses_the_union_of_utterances():
    lines = [(0.0, 4.0), (5.0, 9.0), (10.0, 12.0)]
    utts = [(0.0, 2.0), (1.0, 3.0), (5.0, 9.5)]
    cov = align.line_coverage(utts, lines)
    assert cov == [pytest.approx(0.75), pytest.approx(1.0), 0.0]
    assert align.overlap(0, 1, 2, 3) == 0.0 and align.overlap(0, 2, 1, 3) == 1.0
