"""Memory index: jieba tokenization, SQLite FTS5 search and context windows over `utterances`.

The FTS5 table `utterances_fts` (external content on `utterance_tokens.text_seg`, unicode61) is
kept in sync by triggers in db.py. Every utterance is stored with `text_seg = tokenize(text)`
(jieba words joined by spaces) so that the ASCII-oriented unicode61 tokenizer sees Chinese words
as separate tokens. Queries are tokenized the same way, each token double-quoted and combined
with AND (strict) then OR (loose) and ranked by bm25().

keyword_search() is the keyword side of hybrid retrieval: the question is reduced to its content words
(query_terms: no function / question words; names, numbers and Latin tokens always kept), every term is
widened with the lexicon's aliases of it, and a sentence only counts when it contains at least two of the
question's terms, or one that is rare in the whole archive.

Speaker names and glossary terms should be registered with `register_terms()` before tokenizing
(both at index time and at query time) so that e.g. "张三说" is cut as "张三 / 说".
"""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import jieba

from ..db import DB

log = logging.getLogger(__name__)
jieba.setLogLevel(logging.WARNING)  # silence "Building prefix dict ..." on first use

_PUNCT_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)
_registered: set[str] = set()

# function words that carry no retrieval signal; dropped from the loose (OR) query only
STOPWORDS: frozenset[str] = frozenset(
    "的 了 在 是 我 你 他 她 它 们 这 那 和 与 跟 就 也 都 还 又 很 吗 吧 呢 啊 呀 哦 嗯 把 被 给 对 从 到 去 来 有 没 不 要 "
    "会 能 可以 可能 什么 怎么 怎样 为什么 哪 哪些 哪个 谁 一下 一个 这个 那个 这些 那些 我们 你们 他们 自己 "
    "没有 有没有 的话 时候 然后 但是 所以 因为 如果 还是 或者 以及 关于 about the a an of to in on and or is are was were "
    "be been it this that these those what which who how why when where".split()
)
#: words of a QUESTION that say nothing about its subject ("多少钱", "是谁", "有没有说过"): on top of STOPWORDS
#: when a question is turned into keyword terms (query_terms). Not applied to the loose OR query above.
QUESTION_WORDS: frozenset[str] = frozenset(
    "多少 多久 多大 多长 几个 几点 几号 几天 哪天 哪里 哪儿 哪家 哪种 哪位 是谁 谁的 什么样 什么时候 怎么样 如何 为何 是否 是不是 "
    "对不对 能不能 会不会 要不要 可不可以 有无 多少钱 请问 告诉 知道 记得 记不记得 帮 帮忙 一共 总共 分别 到底 究竟 最后 后来 "
    "当时 那天 上次 这次 之前 之后 以前 以后 现在 目前 已经 还有 另外 还 再 说 说过 提到 提过 聊 聊过 聊到 讲 讲过 问 问过 "
    "东西 事情 事 情况 问题 内容 时候 时间 一些 那些 这种 那种 哪 了吗 了没 没 吗 呢 吧".split()
)
_NUMERAL_CHARS = frozenset("零〇一二两三四五六七八九十百千万亿点半")
#: a token that only says WHEN ("昨天晚上", "上周三", "中午"): the time filter handles it (memory.temporal), as a
#: keyword it would match every sentence in which somebody mentions a time of day
_TIME_TOKEN = re.compile(
    r"^(?:大前天|前天|昨天|今天|明天|后天|昨|今|明|当天|那天|这天)?(?:凌晨|清晨|早晨|早上|上午|中午|下午|傍晚|晚上|夜里|深夜|半夜|晚|早|天)?$"
    r"|^(?:上上|上|这|本|下)?(?:周|星期|礼拜)[一二三四五六日天]?$|^(?:上个?|这个?|本|下个?)月$|^(?:最近|这几天|这两天|这三天|近期)$")
_LATIN_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_+#.\-]*$")


def content_tokens(tokens: list[str]) -> list[str]:
    """Tokens worth matching loosely: drop stopwords and single characters (unless nothing else is left)."""
    strong = [t for t in tokens if t.lower() not in STOPWORDS and len(t) > 1]
    if strong:
        return strong
    return [t for t in tokens if t.lower() not in STOPWORDS] or tokens


# --- tokenization ---------------------------------------------------------
def tokenize(text: str | None) -> str:
    """jieba.lcut -> drop whitespace/punctuation-only tokens -> space-joined string."""
    if not text:
        return ""
    out: list[str] = []
    for tok in jieba.lcut(text):
        tok = tok.strip()
        if not tok or _PUNCT_ONLY.match(tok):
            continue
        out.append(tok)
    return " ".join(out)


def register_terms(words: Iterable[str]) -> int:
    """Add words (speaker names, glossary terms) to jieba's dictionary. Idempotent; returns #added."""
    n = 0
    for w in words:
        w = (w or "").strip()
        if not w or w in _registered:
            continue
        jieba.add_word(w)
        _registered.add(w)
        n += 1
    if n:
        log.debug("registered %d terms with jieba", n)
    return n


def load_glossary(path: Path | str | None) -> list[str]:
    """Terms of glossary.txt (one per line, `#` comments, inline ` # ...` remarks dropped). Missing file -> [].
    A thin wrapper: speech.glossary_file is the one parser of that file."""
    from ..speech.glossary_file import read_glossary

    return read_glossary(path)


def lexicon_terms(db: DB) -> list[str]:
    """Canonical spellings and aliases of the lexicon that are in use (status not 'disabled'). Read straight
    from the tables (their DDL is the contract), so this works whether or not the lexicon module is installed;
    [] on a database without those tables."""
    try:
        rows = db.query("SELECT term FROM lexicon WHERE status <> 'disabled'"
                        " UNION SELECT a.alias FROM lexicon_aliases a JOIN lexicon l ON l.id = a.lexicon_id"
                        " WHERE l.status <> 'disabled'")
    except sqlite3.Error:
        return []
    return [str(r[0]) for r in rows if r[0]]


def register_from(cfg, db: DB) -> int:
    """Register all enrolled speaker names + glossary terms + lexicon spellings (call once at startup)."""
    names = [r["name"] for r in db.query("SELECT name FROM speakers")]
    terms = load_glossary(getattr(cfg, "glossary_path", None))
    return register_terms(names + terms + lexicon_terms(db))


# --- search ---------------------------------------------------------------
@dataclass
class Hit:
    id: int
    segment_id: int
    day: str
    abs_start: str
    start_s: float
    end_s: float
    speaker_name: str | None
    text: str
    score: float = 0.0        # fts: bm25 (lower is better); vector: cosine similarity; fused: RRF score
    source: str = "fts"       # fts | vector | like | 'fts+vector' after fusion

    @classmethod
    def from_row(cls, r: sqlite3.Row, score: float = 0.0, source: str = "fts") -> "Hit":
        return cls(
            id=int(r["id"]), segment_id=int(r["segment_id"]), day=r["day"], abs_start=r["abs_start"],
            start_s=float(r["start_s"]), end_s=float(r["end_s"]), speaker_name=r["speaker_name"],
            text=r["text"], score=float(score), source=source,
        )


def _fts_token(tok: str) -> str:
    return '"' + tok.replace('"', '""') + '"'


def build_match(tokens: list[str], op: str) -> str:
    return f" {op} ".join(_fts_token(t) for t in tokens)


#: hour of the day of an utterance as a number (9.5 = 09:30), from the ISO text of abs_start
_HOUR_SQL = "(CAST(substr(u.abs_start, 12, 2) AS REAL) + CAST(substr(u.abs_start, 15, 2) AS REAL) / 60.0)"


def _filters(day_from: str | None, day_to: str | None, speaker: str | None,
             hours: tuple[float, float] | None = None) -> tuple[str, list]:
    """SQL conditions (on the alias `u`) + parameters. `hours` = (from, to) on the clock, e.g. (18, 24) for
    "in the evening"; from > to wraps past midnight ((22, 4) = 22:00-04:00)."""
    sql, params = "", []
    if day_from:
        sql += " AND u.day >= ?"
        params.append(day_from)
    if day_to:
        sql += " AND u.day <= ?"
        params.append(day_to)
    if speaker:
        sql += " AND u.speaker_name = ?"
        params.append(speaker)
    if hours:
        h0, h1 = float(hours[0]), float(hours[1])
        if h0 <= h1:
            sql += f" AND {_HOUR_SQL} >= ? AND {_HOUR_SQL} < ?"
        else:
            sql += f" AND ({_HOUR_SQL} >= ? OR {_HOUR_SQL} < ?)"
        params += [h0, h1]
    return sql, params


def _fts_query(db: DB, match: str, day_from, day_to, speaker, limit: int,
               hours: tuple[float, float] | None = None) -> list[Hit]:
    fsql, fparams = _filters(day_from, day_to, speaker, hours)
    sql = (
        "SELECT u.*, bm25(utterances_fts) AS score FROM utterances_fts"
        " JOIN utterances u ON u.id = utterances_fts.rowid"
        " WHERE utterances_fts MATCH ?" + fsql +
        " ORDER BY bm25(utterances_fts), u.abs_start LIMIT ?"
    )
    try:
        rows = db.query(sql, [match, *fparams, limit])
    except sqlite3.OperationalError as e:  # malformed MATCH expression
        log.warning("FTS query failed for %r: %s", match, e)
        return []
    return [Hit.from_row(r, r["score"]) for r in rows]


def search(db: DB, query: str, day_from: str | None = None, day_to: str | None = None,
           speaker: str | None = None, limit: int = 20, strict: bool = False,
           hours: tuple[float, float] | None = None) -> list[Hit]:
    """FTS5 search. All tokens ANDed first; if fewer than 3 hits, retry with OR over the content
    tokens (stopwords / single characters dropped). Ranked by bm25. strict=True skips the OR
    retry (memory.hybrid_keywords: strict)."""
    tokens = tokenize(query).split()
    if not tokens:
        return []
    hits = _fts_query(db, build_match(tokens, "AND"), day_from, day_to, speaker, limit, hours)
    if not strict and len(hits) < 3 and len(tokens) > 1:
        loose_tokens = content_tokens(tokens)
        if loose_tokens != tokens or len(loose_tokens) > 1:
            loose = _fts_query(db, build_match(loose_tokens, "OR"), day_from, day_to, speaker, limit, hours)
            if len(loose) > len(hits):
                hits = loose
    return hits


# --- keyword side of hybrid retrieval -------------------------------------------
@dataclass
class QueryTerm:
    """One thing a question asks about. `alts` are token sequences that all mean it (the word itself plus the
    lexicon's other spellings of it); `strong` marks names, numbers and Latin tokens (kept even when short;
    memory.items.related_facts looks facts up by them)."""
    text: str
    alts: list[list[str]]
    strong: bool = False

    def match(self) -> str:
        """FTS5 expression matching any alternative (a multi-token alternative is a phrase)."""
        parts = ['"' + " ".join(t.replace('"', '""') for t in alt) + '"' for alt in self.alts if alt]
        return parts[0] if len(parts) == 1 else "(" + " OR ".join(parts) + ")"

    def found_in(self, tokens: set[str], seg: str) -> bool:
        for alt in self.alts:
            if len(alt) == 1:
                if alt[0].lower() in tokens:
                    return True
            elif alt and " ".join(alt).lower() in seg:
                return True
        return False


def is_number_token(tok: str) -> bool:
    """'2899', 'T30', '三十五万', '两千八百九十九': a figure the transcript must contain literally."""
    if any(ch.isdigit() for ch in tok):
        return True
    return len(tok) >= 2 and all(ch in _NUMERAL_CHARS for ch in tok)


def is_strong_token(tok: str) -> bool:
    """A name, a registered term, a number or a Latin word: rare in speech, so one match means something."""
    if tok in _registered:
        return True
    if is_number_token(tok):
        return True
    return len(tok) >= 2 and bool(_LATIN_TOKEN.match(tok))


def alias_groups(db: DB, text: str) -> list[tuple[str, list[str]]]:
    """Lexicon entries the text mentions, by canonical spelling or by any alias: [(matched text, [canonical,
    alias, ...])]. Longer matches win; nothing when the lexicon is empty or its tables are missing."""
    try:
        rows = db.query("SELECT l.id, l.term, a.alias FROM lexicon l LEFT JOIN lexicon_aliases a ON a.lexicon_id = l.id"
                        " WHERE l.status <> 'disabled'")
    except sqlite3.Error:
        return []
    by_id: dict[int, list[str]] = {}
    for r in rows:
        names = by_id.setdefault(int(r["id"]), [str(r["term"])])
        if r["alias"] and r["alias"] not in names:
            names.append(str(r["alias"]))
    found: list[tuple[int, int, str, list[str]]] = []
    low = text.lower()
    for names in by_id.values():
        if len(names) < 2:
            continue                       # nothing to widen with
        for n in sorted(names, key=len, reverse=True):
            i = low.find(n.lower())
            if i >= 0 and len(n) >= 2:
                found.append((i, len(n), n, names))
                break
    out: list[tuple[str, list[str]]] = []
    taken: list[tuple[int, int]] = []
    for i, ln, n, names in sorted(found, key=lambda f: (-f[1], f[0])):
        if any(i < b and a < i + ln for a, b in taken):
            continue
        taken.append((i, i + ln))
        out.append((n, names))
    return out


def query_terms(query: str, db: DB | None = None) -> list[QueryTerm]:
    """The content of a question as keyword terms: stopwords, question words and single characters go, names /
    numbers / Latin tokens stay (marked strong). With `db`, a term the lexicon knows under several spellings
    matches every one of them (a nickname also finds the full name)."""
    out: list[QueryTerm] = []
    seen: set[str] = set()
    rest = query or ""
    if db is not None:
        low = rest.lower()
        for matched, names in alias_groups(db, rest):
            alts = [tokenize(n).split() for n in names]
            out.append(QueryTerm(matched, [a for a in alts if a], strong=True))
            seen.update(t.lower() for a in alts for t in a)
            i = low.find(matched.lower())
            if i >= 0:                      # the term is handled: keep its pieces out of the plain tokens
                rest = rest[:i] + " " + rest[i + len(matched):]
                low = rest.lower()
    for tok in tokenize(rest).split():
        low_tok = tok.lower()
        if low_tok in seen:
            continue
        strong = is_strong_token(tok)
        if not strong and (len(tok) < 2 or low_tok in STOPWORDS or low_tok in QUESTION_WORDS or _TIME_TOKEN.match(tok)):
            continue
        if strong and len(tok) < 2 and not tok.isdigit():
            continue
        seen.add(low_tok)
        out.append(QueryTerm(tok, [[tok]], strong=strong))
    if not out:
        # a query of one to three words that are all "question words" ("问题", "情况"): somebody typed exactly
        # that into a search box, so the words are what to look for
        tokens = tokenize(rest).split()
        if len(tokens) <= 3:
            out = [QueryTerm(t, [[t]]) for t in dict.fromkeys(tokens)
                   if len(t) >= 2 and t.lower() not in STOPWORDS and not _TIME_TOKEN.match(t)]
    return out


def is_lookup(query: str, terms: list[QueryTerm] | None = None) -> bool:
    """True for a query that is just a few search words ("朗月T30", "周岚 压测", an order number) rather than a
    question: at most three tokens and every one of them a content term. Such a query wants the sentences that
    contain the words first; a sentence embedding of a bare code or name says next to nothing."""
    tokens = tokenize(query).split()
    if not tokens or len(tokens) > 3:
        return False
    terms = query_terms(query) if terms is None else terms
    return sum(len(t.alts[0]) for t in terms if t.alts) >= len(tokens)


#: candidates read from the index before the coverage rule is applied, as a multiple of the wanted hits
KEYWORD_OVERSAMPLE = 6
#: a sentence that contains only ONE term of the question counts when that term is rare: it occurs in at most
#: RARE_SHARE of all sentences, clamped to [RARE_DF_MIN, RARE_DF_MAX] sentences. Measured on the two benchmark
#: sets (about 150 sentences each): "two terms, or one term found in <= 2 sentences" was the best keyword list
#: next to the vectors; <= 3 already cost a question in the top 5, and "any name / number / Latin term" cost
#: three (a product name occurs in ten sentences, the answer is in one that does not even contain it).
RARE_SHARE = 0.01
RARE_DF_MIN = 2
RARE_DF_MAX = 20


def keyword_search(db: DB, query: str, day_from: str | None = None, day_to: str | None = None,
                   speaker: str | None = None, limit: int = 20, hours: tuple[float, float] | None = None,
                   min_cover: int = 2, terms: list[QueryTerm] | None = None) -> list[Hit]:
    """Sentences that share real content with the question, best first (bm25, so rare words weigh most).

    One OR query over the question's terms (query_terms), then a coverage rule: a sentence stays when it
    contains at least `min_cover` different terms, or a single term that is rare in the whole archive. A loose
    OR list without that rule mostly adds sentences that share one everyday word with the question, which is
    what made the old keyword side useless next to the vector list."""
    terms = query_terms(query, db) if terms is None else terms
    if not terms:
        return []
    fsql, fparams = _filters(day_from, day_to, speaker, hours)
    match = " OR ".join(t.match() for t in terms)
    sql = ("SELECT u.*, t.text_seg AS _seg, bm25(utterances_fts) AS score FROM utterances_fts"
           " JOIN utterances u ON u.id = utterances_fts.rowid"
           " JOIN utterance_tokens t ON t.utterance_id = u.id"
           " WHERE utterances_fts MATCH ?" + fsql + " ORDER BY bm25(utterances_fts), u.abs_start LIMIT ?")
    try:
        rows = db.query(sql, [match, *fparams, max(limit * KEYWORD_OVERSAMPLE, 60)])
    except sqlite3.OperationalError as e:
        log.warning("keyword query failed for %r: %s", match, e)
        return []
    need = min(max(1, int(min_cover)), len(terms))
    rare_df: int | None = None
    df: dict[int, int] = {}

    def is_rare(i: int) -> bool:
        """Document frequency of term i against the rarity limit (looked up only for sentences that need it)."""
        nonlocal rare_df
        if rare_df is None:
            total = int(db.one("SELECT COUNT(*) AS n FROM utterance_tokens")["n"])
            rare_df = max(RARE_DF_MIN, min(RARE_DF_MAX, int(total * RARE_SHARE)))
        if i not in df:
            try:
                df[i] = int(db.one("SELECT COUNT(*) AS n FROM utterances_fts WHERE utterances_fts MATCH ?", (terms[i].match(),))["n"])
            except sqlite3.OperationalError:
                df[i] = 10 ** 9
        return df[i] <= rare_df

    out: list[Hit] = []
    for r in rows:
        seg = str(r["_seg"] or "").lower()
        toks = set(seg.split())
        hit = [i for i, t in enumerate(terms) if t.found_in(toks, seg)]
        if len(hit) >= need or any(is_rare(i) for i in hit):
            out.append(Hit.from_row(r, r["score"]))
            if len(out) >= limit:
                break
    return out


def search_terms(db: DB, terms: list[str], day_from: str | None = None, day_to: str | None = None,
                 speaker: str | None = None, limit: int = 20) -> list[Hit]:
    """OR query over the content tokens of several phrasings (query expansion), ranked by bm25."""
    tokens: list[str] = []
    for t in terms:
        for tok in content_tokens(tokenize(t).split()):
            if tok not in tokens:
                tokens.append(tok)
    if not tokens:
        return []
    return _fts_query(db, build_match(tokens, "OR"), day_from, day_to, speaker, limit)


def like_search(db: DB, query: str, day_from: str | None = None, day_to: str | None = None,
                speaker: str | None = None, limit: int = 20, hours: tuple[float, float] | None = None) -> list[Hit]:
    """Substring fallback on utterances.text (newest first). Useful when jieba cuts a term differently."""
    q = (query or "").strip()
    if not q:
        return []
    pat = "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    fsql, fparams = _filters(day_from, day_to, speaker, hours)
    rows = db.query(
        "SELECT u.* FROM utterances u WHERE u.text LIKE ? ESCAPE '\\'" + fsql +
        " ORDER BY u.abs_start DESC LIMIT ?",
        [pat, *fparams, limit],
    )
    return [Hit.from_row(r, source="like") for r in rows]


# --- context --------------------------------------------------------------
def window(db: DB, utterance_id: int, before: int = 5, after: int = 5) -> list[sqlite3.Row]:
    """The utterance plus `before` earlier and `after` later utterances (any segment), ordered by abs_start."""
    target = db.one("SELECT * FROM utterances WHERE id=?", (utterance_id,))
    if target is None:
        return []
    ts, tid = target["abs_start"], target["id"]
    prev = db.query(
        "SELECT * FROM utterances WHERE abs_start < ? OR (abs_start = ? AND id < ?)"
        " ORDER BY abs_start DESC, id DESC LIMIT ?",
        (ts, ts, tid, max(0, int(before))),
    )
    nxt = db.query(
        "SELECT * FROM utterances WHERE abs_start > ? OR (abs_start = ? AND id > ?)"
        " ORDER BY abs_start, id LIMIT ?",
        (ts, ts, tid, max(0, int(after))),
    )
    return list(reversed(prev)) + [target] + list(nxt)


# --- maintenance ----------------------------------------------------------
def reindex(db: DB, batch: int = 500) -> int:
    """Recompute text_seg for every utterance (e.g. after registering new glossary terms) and for every memory
    item (items copied from the old notes table have none yet). Returns the number of utterances."""
    try:
        from . import items as _items

        _items.reindex_tokens(db, force=True)
    except Exception as e:  # noqa: BLE001 - the sentence index must not depend on the item index
        log.warning("memory item tokens not rebuilt: %s", e)
    n = 0
    last = 0
    while True:
        rows = db.query("SELECT id, text FROM utterances WHERE id > ? ORDER BY id LIMIT ?", (last, batch))
        if not rows:
            break
        with db.transaction() as c:
            for r in rows:
                c.execute(
                    "INSERT INTO utterance_tokens(utterance_id, text_seg) VALUES (?,?)"
                    " ON CONFLICT(utterance_id) DO UPDATE SET text_seg=excluded.text_seg",
                    (r["id"], tokenize(r["text"])),
                )
        last = int(rows[-1]["id"])
        n += len(rows)
    log.info("reindexed %d utterances", n)
    return n


# --- formatting helpers (shared by tools / cli) -----------------------------
def fmt_time(abs_start: str, with_date: bool = True) -> str:
    """Minute resolution: seconds made the model misquote times ('09:30:05' -> '09:05')."""
    s = (abs_start or "").replace("T", " ")
    return s[:16] if with_date else s[11:16]


def fmt_row(r: sqlite3.Row | Hit, with_date: bool = True, with_id: bool = True) -> str:
    """'[#12] 2026-10-31 08:19 张三: text' (one line)."""
    if isinstance(r, Hit):
        uid, abs_start, spk, text = r.id, r.abs_start, r.speaker_name, r.text
    else:
        uid, abs_start, spk, text = r["id"], r["abs_start"], r["speaker_name"], r["text"]
    head = f"[#{uid}] " if with_id else ""
    return f"{head}{fmt_time(abs_start, with_date)} {spk or '?'}: {' '.join((text or '').split())}"
