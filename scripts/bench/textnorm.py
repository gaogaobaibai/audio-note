"""Text normalisation for character error rates (pure functions, no I/O).

    normalize("今天，T30 到手价 2,699 元！")      -> "今天t30到手价2699元"
    normalize("到手两千六百九十九元")              -> "到手2699元"
    normalize("比例百分之五，版本二点四点一")       -> "比例5%版本2.4.1"

What it does, in order:
  1. NFKC (full-width letters / digits / punctuation become half-width), lower-case;
  2. numerals: Chinese numerals become Arabic digits so that "十二万五" == "125000" == "12.5万",
     "二零二六年" == "2026年", "百分之十" == "10%", "二点四" == "2.4", "T三十" == "t30";
  3. whitespace, punctuation and symbols are dropped (a decimal point between digits and a percent sign
     after a digit are kept).
The same function is applied to the reference and to the hypothesis, so a conversion that looks odd in
isolation ("一起" -> "1起") costs nothing; what matters is that two spellings of the same number meet.

Known limits (cheap on purpose): clock times ("三点半" vs "3:30"), "号" vs "日", ordinal / measure words and
traditional characters are not unified; "周六十点" is read as 周六 + 十点.
"""
from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation

_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_SMALL = {"十": 10, "百": 100, "千": 1000}
_BIG = {"万": 10 ** 4, "亿": 10 ** 8}
_D = "零〇一二两三四五六七八九"
_CN = _D + "十百千万亿"

_NUM_RE = re.compile(
    r"(?P<wd>(?:周|星期|礼拜)[一二三四五六])"
    r"|(?P<pct>百分之)?(?:"
    rf"(?P<cn>[{_CN}]+)(?P<frac>(?:点[{_D}]+(?![{_D}十百千]))*)(?P<mul>[万亿])?"
    r"|(?P<ar>\d+(?:\.\d+)*)(?P<armul>[千万亿])?"
    r")"
)
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def chinese_to_number(s: str) -> str:
    """Digits for a run of Chinese numeral characters.

    Without unit characters the run is positional ("二零二六" -> "2026", "零五" -> "05"); with units it is
    read as a quantity ("两千零二十六" -> "2026"), including the colloquial short forms where a trailing
    digit continues the previous unit ("三万五" -> "35000", "一千八" -> "1800", "一百五" -> "150").
    Two digits in a row inside a quantity start a new number ("二十三四" -> "23" + "4")."""
    if not s:
        return ""
    if not any(ch in _SMALL or ch in _BIG for ch in s):
        return "".join(str(_DIGITS[ch]) for ch in s)
    out: list[str] = []
    total = 0
    section = 0
    number: int | None = None
    started = False

    def flush() -> None:
        nonlocal total, section, number, started
        if started:
            out.append(str(total + section + (number or 0)))
        total, section, number, started = 0, 0, None, False

    for ch in s:
        if ch in ("零", "〇"):
            if number is not None:           # "二零" inside a quantity: not a quantity any more, keep the digits
                flush()
                out.append("0")
            else:
                started = True               # placeholder zero: "两千零二十六", "一百零五"
        elif ch in _DIGITS:
            if number is not None:           # two digits in a row: a new number starts
                flush()
            number = _DIGITS[ch]
            started = True
        elif ch in _SMALL:
            section += (1 if number is None else number) * _SMALL[ch]
            number = None
            started = True
        else:                                # 万 / 亿
            section += number or 0
            number = None
            if section == 0 and total == 0:
                section = 1
            if ch == "亿":
                total = (total + section) * _BIG[ch]
            else:
                total += section * _BIG[ch]
            section = 0
            started = True
    # a trailing digit continues the previous unit after 百 / 千 / 万 / 亿 ("三千五" = 3500); after 十 / 零 it is units
    if number is not None and len(s) >= 2 and s[-2] in ("百", "千", "万", "亿"):
        unit = _SMALL.get(s[-2]) or _BIG[s[-2]]
        section += number * (unit // 10)
        number = None
    flush()
    return "".join(out)


def _mul(value: str, unit: str) -> str | None:
    """"12.5" x 万 -> "125000"; None when the product is not an integer or the value is not a plain number."""
    if value.count(".") > 1:
        return None
    try:
        v = Decimal(value) * ({"千": 1000, "万": 10 ** 4, "亿": 10 ** 8}[unit])
    except (InvalidOperation, KeyError):
        return None
    if v != v.to_integral_value():
        return None
    return str(int(v))


def convert_numerals(text: str) -> str:
    """Rewrite every numeral in `text` to Arabic digits (see the module docstring)."""
    text = _THOUSANDS_RE.sub("", text)

    def repl(m: re.Match[str]) -> str:
        if m.group("wd"):
            w = m.group("wd")
            return w[:-1] + str(_DIGITS[w[-1]])
        pct = "%" if m.group("pct") else ""
        if m.group("cn") is not None:
            num = chinese_to_number(m.group("cn"))
            frac = m.group("frac") or ""
            if frac:
                num += "".join("." + chinese_to_number(part) for part in frac.split("点")[1:])
            mul = m.group("mul")
            if mul:
                prod = _mul(num, mul)
                num = prod if prod is not None else num + str(_BIG[mul])
            return num + pct
        num = m.group("ar")
        mul = m.group("armul")
        if mul:
            prod = _mul(num, mul)
            num = prod if prod is not None else num + mul
        return num + pct

    return _NUM_RE.sub(repl, text)


_DOT_RE = re.compile(r"(?<!\d)\.|\.(?!\d)")
_PCT_RE = re.compile(r"(?<!\d)%")


def normalize(text: str | None, *, numerals: bool = True) -> str:
    """Canonical string for character-level comparison (see the module docstring)."""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).lower()
    if numerals:
        s = convert_numerals(s)
    kept = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "N") or ch in ".%":
            kept.append(ch)
    s = "".join(kept)
    # a dot survives only between digits, a percent sign only after a digit (repeat: "1..2" style leftovers)
    prev = None
    while prev != s:
        prev = s
        s = _DOT_RE.sub("", s)
        s = _PCT_RE.sub("", s)
    return s


def normalize_term(term: str) -> str:
    """A lexicon term in the same canonical form as normalised text."""
    return normalize(term)
