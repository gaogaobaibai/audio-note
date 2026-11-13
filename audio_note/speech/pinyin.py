"""Sound-alike comparison of short Chinese / mixed strings (SPEC v1 section 6.1).

    key(text)               "灰度发布" -> "hui du fa bu"      toneless syllables; Latin / digit runs lower-cased as they are
    similarity(a, b)        0..1, 1.0 = sounds the same
    close(a, b, min_sim)    similarity(a, b) >= min_sim
    units(text)             the tokens the comparison works on, with their character spans

Why it exists: the ASR writes a word it does not know with characters that sound the same (a project name
came back with same-sounding characters 10 times out of 10 in the bench recordings). Characters cannot tell "the
same word, mis-heard" from "a different word"; syllables can. The lexicon uses this to find the entries a piece
of text may be a mis-hearing of, the inbox to merge spellings of one word into one question, and the correction
guard to refuse an LLM edit that does not even sound like the original.

How similar
-----------
Both strings become syllable sequences; the distance is an edit distance over syllables in which

* the same syllable costs 0 (tones are ignored; a character is read the way its context suggests),
* a near-sound pair costs NEAR_COST (half a step): the initials or finals differ only by a confusion that
  speakers and recognisers make all the time (z/zh, c/ch, s/sh, n/l, l/r, f/h; -n/-ng),
* anything else, and a missing or extra syllable, costs 1,

and similarity = 1 - distance / max(len(a), len(b)). Latin / digit runs are one token each and are compared by
their characters ("t30" vs "t3o" = 0.67 of a step).

DEFAULT_MIN_SIM, CANDIDATE_MIN_SIM and the near-sound table were chosen on measurements (the bench story's gold
terms and what the baseline recordings turned them into): the numbers are next to the constants. A wider
near-sound table (aspiration pairs, more finals) or a discount for a shared initial / final found a few more
mis-hearings and several times more unrelated dictionary words, so the table stays small.

pypinyin is imported on first use. Without it the module still works, one token per character: only identical
characters match, which is a plain character edit distance - degraded, never an error (`available()` tells).
"""
from __future__ import annotations

import re
import threading
from functools import lru_cache
from typing import Any, NamedTuple

#: cost of substituting a near-sound syllable (z/zh, n/l, -n/-ng ...): "half a step"
NEAR_COST = 0.5

#: Default threshold of close(): "the same word, heard differently". Measured on the bench story (24 gold terms,
#: 47 real mis-hearings in the baseline recordings, 44,894 windows of other text): 0.75 accepts 63.8% of the
#: mis-hearings and none of the windows; a 3-syllable dictionary word then has 0.01 other dictionary words that
#: count as close. Similarities are coarse (two syllables: 1.0, 0.75, 0.5 ...), so 0.75 means: one near-sound
#: syllable in a two- or three-syllable word, one unrelated syllable only from four syllables on.
DEFAULT_MIN_SIM = 0.75
#: Looser threshold for SUGGESTING candidates (Lexicon.near), never for deciding: it also accepts one unrelated
#: syllable in three and one wrong letter in a three-letter acronym. Same bench: 74.5% of the mis-hearings, still
#: none of the windows - but a 3-syllable dictionary word has 1.1 close dictionary words instead of 0.01.
CANDIDATE_MIN_SIM = 0.66

_INITIALS = ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s",
             "y", "w")

#: initials that are confused all the time (flat / retroflex, n / l, l / r, f / h)
_NEAR_INITIALS = frozenset(frozenset(p) for p in (
    ("z", "zh"), ("c", "ch"), ("s", "sh"), ("n", "l"), ("l", "r"), ("f", "h"),
))
#: finals that are confused all the time (front / back nasal)
_NEAR_FINALS = frozenset(frozenset(p) for p in (
    ("an", "ang"), ("en", "eng"), ("in", "ing"), ("ian", "iang"), ("uan", "uang"), ("un", "ong"),
))

_CJK = re.compile(r"[㐀-䶿一-鿿]")
_TOKEN = re.compile(r"[㐀-䶿一-鿿]|[A-Za-z0-9]+")

_lock = threading.Lock()
_pypinyin: Any = None          # the module, False when it cannot be imported, None = not tried yet


class Unit(NamedTuple):
    """One comparison token: a Chinese character (as its syllable) or a Latin / digit run."""
    syl: str                       # toneless syllable; a Latin run: its lower-cased text
    start: int                     # character span in the source text
    end: int
    latin: bool


def _load() -> Any:
    global _pypinyin
    if _pypinyin is None:
        with _lock:
            if _pypinyin is None:
                try:
                    import pypinyin

                    _pypinyin = pypinyin
                except Exception:  # noqa: BLE001 - not installed / broken install: degrade to characters
                    _pypinyin = False
    return _pypinyin


def available() -> bool:
    """True when pypinyin is importable (False: comparisons fall back to characters)."""
    return bool(_load())


@lru_cache(maxsize=8192)
def _char_syllable(ch: str) -> str:
    """The usual toneless reading of one Chinese character; the character itself when pypinyin is missing or
    does not know it."""
    pp = _load()
    if not pp:
        return ch
    try:
        got = pp.lazy_pinyin(ch, style=pp.Style.NORMAL, errors="ignore")
    except Exception:  # noqa: BLE001
        return ch
    r = str(got[0]).strip().lower().replace("ü", "v") if got else ""
    return r or ch


@lru_cache(maxsize=4096)
def _context_syllables(text: str) -> tuple[str, ...]:
    """The reading pypinyin picks for each Chinese character of `text` in its context (its phrase dictionary
    knows 银行 is "yin hang"), one entry per Chinese character, in order. Empty when pypinyin is missing."""
    pp = _load()
    if not pp:
        return ()
    chars = "".join(_CJK.findall(text))
    if not chars:
        return ()
    try:
        got = pp.lazy_pinyin(chars, style=pp.Style.NORMAL, errors="ignore")
    except Exception:  # noqa: BLE001
        return ()
    got = [str(r).strip().lower().replace("ü", "v") for r in got]
    return tuple(got) if len(got) == len(chars) else ()


@lru_cache(maxsize=4096)
def units(text: str) -> tuple[Unit, ...]:
    """`text` as comparison tokens (punctuation and spaces are dropped). Cached: texts are short and repeat.

    Only the reading in context is used. pypinyin also lists every other reading a character ever had, and
    counting those as matches made unrelated common words "sound the same" as a name (measured: twice as many
    confusable dictionary words, no mis-hearing gained)."""
    text = text or ""
    ctx = _context_syllables(text)
    out: list[Unit] = []
    k = 0
    for m in _TOKEN.finditer(text):
        tok = m.group(0)
        if _CJK.match(tok):
            syl = ctx[k] if k < len(ctx) and ctx[k] else _char_syllable(tok)
            k += 1
            out.append(Unit(syl, m.start(), m.end(), False))
        else:
            out.append(Unit(tok.lower(), m.start(), m.end(), True))
    return tuple(out)


def units_from_key(pinyin_key: str) -> tuple[Unit, ...]:
    """Tokens from a stored key ("shan zong"): lets a lexicon entry carry a reading pypinyin would not pick
    (a surname, a made-up word). Spans are token indexes, not character offsets."""
    out: list[Unit] = []
    for i, tok in enumerate((pinyin_key or "").split()):
        tok = tok.strip().lower()
        if tok:
            out.append(Unit(tok, i, i + 1, bool(re.fullmatch(r"[a-z0-9]+", tok)) and not _is_syllable(tok)))
    return tuple(out)


_FINALS = frozenset("a o e i u v ai ei ui ao ou iu ie ve ue er an en in un vn ang eng ing ong ia iao ian iang iong ua uo uai "
                    "uan uang ueng van".split())


def _is_syllable(tok: str) -> bool:
    """Whether a lower-case Latin token is a pinyin syllable (so a stored key can mix syllables and acronyms)."""
    ini, fin = split_syllable(tok)
    return fin in _FINALS


def key(text: str) -> str:
    """Toneless pinyin key: syllables joined by spaces, Latin / digit runs lower-cased as they are
    ("青禾App" -> "qing he app"). Two spellings with the same key sound the same. Without pypinyin the
    characters themselves are the tokens."""
    return " ".join(u.syl for u in units(text or ""))


def split_syllable(s: str) -> tuple[str, str]:
    """("zh", "ang") for "zhang"; a syllable without an initial has "" ("an" -> ("", "an"))."""
    for ini in _INITIALS:
        if s.startswith(ini) and len(s) > len(ini):
            return ini, s[len(ini):]
    return "", s


@lru_cache(maxsize=65536)
def syllable_cost(a: str, b: str) -> float:
    """Substitution cost between two toneless syllables: 0 same, NEAR_COST near-sound, 1 otherwise."""
    if a == b:
        return 0.0
    ia, fa = split_syllable(a)
    ib, fb = split_syllable(b)
    ini_ok = ia == ib or frozenset((ia, ib)) in _NEAR_INITIALS
    fin_ok = fa == fb or frozenset((fa, fb)) in _NEAR_FINALS
    return NEAR_COST if (ini_ok and fin_ok) else 1.0


def _latin_cost(a: str, b: str) -> float:
    """Latin / digit runs: character edit distance relative to the longer one (0 same .. 1 unrelated)."""
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i]
        for j in range(1, lb + 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1])))
        prev = cur
    return min(1.0, prev[-1] / max(la, lb))


def unit_cost(a: Unit, b: Unit) -> float:
    """Substitution cost between two tokens (see the module docstring)."""
    if a.latin != b.latin:
        return 1.0
    return _latin_cost(a.syl, b.syl) if a.latin else syllable_cost(a.syl, b.syl)


def distance_units(ua: tuple[Unit, ...] | list[Unit], ub: tuple[Unit, ...] | list[Unit]) -> float:
    """Weighted edit distance between two token sequences (insertion / deletion = 1)."""
    la, lb = len(ua), len(ub)
    if not la:
        return float(lb)
    if not lb:
        return float(la)
    prev = [float(j) for j in range(lb + 1)]
    for i in range(1, la + 1):
        cur = [float(i)]
        a = ua[i - 1]
        for j in range(1, lb + 1):
            cur.append(min(prev[j] + 1.0, cur[j - 1] + 1.0, prev[j - 1] + unit_cost(a, ub[j - 1])))
        prev = cur
    return prev[-1]


def similarity_units(ua: tuple[Unit, ...] | list[Unit], ub: tuple[Unit, ...] | list[Unit]) -> float:
    n = max(len(ua), len(ub))
    if n == 0:
        return 0.0
    return max(0.0, 1.0 - distance_units(ua, ub) / n)


def similarity(a: str, b: str) -> float:
    """How alike two strings sound, 0..1 (1.0 = the same syllables). Identical strings are 1.0; a string
    without any letter or character (only punctuation) is similar to nothing but itself."""
    a, b = a or "", b or ""
    if a == b:
        return 1.0 if a else 0.0
    return round(similarity_units(units(a), units(b)), 4)


def close(a: str, b: str, min_sim: float = DEFAULT_MIN_SIM) -> bool:
    """True when `a` and `b` sound alike enough to be the same word heard differently."""
    return similarity(a, b) >= float(min_sim) - 1e-9


__all__ = ["CANDIDATE_MIN_SIM", "DEFAULT_MIN_SIM", "NEAR_COST", "Unit", "available", "close", "distance_units", "key", "similarity",
           "similarity_units", "split_syllable", "syllable_cost", "unit_cost", "units", "units_from_key"]
