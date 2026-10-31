"""Memory index: jieba tokenization, SQLite FTS5 search and context windows over `utterances`.

The FTS5 table `utterances_fts` (external content on `utterance_tokens.text_seg`, unicode61) is
kept in sync by triggers in db.py. Every utterance is stored with `text_seg = tokenize(text)`
(jieba words joined by spaces) so that the ASCII-oriented unicode61 tokenizer sees Chinese words
as separate tokens. Queries are tokenized the same way, each token double-quoted and combined
with AND (strict) then OR (loose) and ranked by bm25().

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
    """One term per line; blank lines and '#' comments ignored. Missing file -> []."""
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    terms: list[str] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                terms.append(s)
    return terms


def register_from(cfg, db: DB) -> int:
    """Register all enrolled speaker names + glossary terms (call once at startup)."""
    names = [r["name"] for r in db.query("SELECT name FROM speakers")]
    terms = load_glossary(getattr(cfg, "glossary_path", None))
    return register_terms(names + terms)


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
    score: float = 0.0        # bm25 (lower is better) for FTS hits, 0 for substring hits

    @classmethod
    def from_row(cls, r: sqlite3.Row, score: float = 0.0) -> "Hit":
        return cls(
            id=int(r["id"]), segment_id=int(r["segment_id"]), day=r["day"], abs_start=r["abs_start"],
            start_s=float(r["start_s"]), end_s=float(r["end_s"]), speaker_name=r["speaker_name"],
            text=r["text"], score=float(score),
        )


def _fts_token(tok: str) -> str:
    return '"' + tok.replace('"', '""') + '"'


def build_match(tokens: list[str], op: str) -> str:
    return f" {op} ".join(_fts_token(t) for t in tokens)


def _filters(day_from: str | None, day_to: str | None, speaker: str | None) -> tuple[str, list]:
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
    return sql, params


def _fts_query(db: DB, match: str, day_from, day_to, speaker, limit: int) -> list[Hit]:
    fsql, fparams = _filters(day_from, day_to, speaker)
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
           speaker: str | None = None, limit: int = 20) -> list[Hit]:
    """FTS5 search. All tokens ANDed first; if fewer than 3 hits, retry with OR over the content
    tokens (stopwords / single characters dropped). Ranked by bm25."""
    tokens = tokenize(query).split()
    if not tokens:
        return []
    hits = _fts_query(db, build_match(tokens, "AND"), day_from, day_to, speaker, limit)
    if len(hits) < 3 and len(tokens) > 1:
        loose_tokens = content_tokens(tokens)
        if loose_tokens != tokens or len(loose_tokens) > 1:
            loose = _fts_query(db, build_match(loose_tokens, "OR"), day_from, day_to, speaker, limit)
            if len(loose) > len(hits):
                hits = loose
    return hits


def like_search(db: DB, query: str, day_from: str | None = None, day_to: str | None = None,
                speaker: str | None = None, limit: int = 20) -> list[Hit]:
    """Substring fallback on utterances.text (newest first). Useful when jieba cuts a term differently."""
    q = (query or "").strip()
    if not q:
        return []
    pat = "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    fsql, fparams = _filters(day_from, day_to, speaker)
    rows = db.query(
        "SELECT u.* FROM utterances u WHERE u.text LIKE ? ESCAPE '\\'" + fsql +
        " ORDER BY u.abs_start DESC LIMIT ?",
        [pat, *fparams, limit],
    )
    return [Hit.from_row(r) for r in rows]


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
    """Recompute text_seg for every utterance (e.g. after registering new glossary terms). Returns #rows."""
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
