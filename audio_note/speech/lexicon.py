"""The lexicon: the words this user's recordings are about, and how the recogniser tends to get them wrong.

One table of canonical spellings (`lexicon`: people, projects, products, terms ...) and one of aliases
(`lexicon_aliases`): a *mishear* alias is a wrong spelling the ASR produces for the term, an *aka* alias is another
name for the same thing (a nickname). It is the ONE vocabulary of the program (SPEC v1 section 6.1):

    first pass        Lexicon.apply(text): every scope='auto' mishear alias is rewritten to its term, at once,
                      without an LLM (correct.LexiconCorrector); the edits become revisions and UI marks
    ASR hot words     Lexicon.hotwords(limit, context)
    correction prompt Lexicon.near(text) / find_in(text) / prompt_rows(terms)
    search            Lexicon.expand(query), jieba_words(db)  (aliases as query expansion, terms as single tokens)
    people / entities Lexicon.resolve(name): alias -> canonical entry

Where entries come from: the user (lexicon page, CLI), an answered inbox question (source 'answer'), a hand
edit of a sentence (revisions.learn_from_edit, source 'edit'), glossary.txt and the enrolled speakers
(sync_sources, sources 'glossary' / 'speaker'). glossary.txt is only an import / export format now.

auto or hint
------------
An alias is rewritten automatically only when that is safe (scope 'auto'). add_alias() decides when the caller
does not: a single character, a common dictionary word (jieba frequency above correction.alias_auto_max_freq),
the spelling of another entry, an alias whose rewrite would change a number, or an alias that already points at
a different entry is only a *hint* - it is
shown to the correcting model / the user, never replaced blindly. The frequency limit is measured: of the 24
distinct wrong spellings the bench recordings produced for their gold terms, 22 have a dictionary frequency of
5 or less, two (114 and 173, together 14 of 47 mis-hearings) are rare proper-noun-like words that must be fixed
automatically, and one (2313) is an everyday word that must not.

Reading and writing
-------------------
`Lexicon.load(db)` returns an immutable snapshot, cached per database file. Every write goes through the
functions below; they change the kv value `lexicon_version` ("<token of this database>:<counter>") in the same
transaction, and load() compares that value and the tables' row counts (one tiny query), so a snapshot is never
stale - not even when another process wrote, when rows were inserted with plain SQL, or when the file was
deleted and a new database created under the same path.

Deleting: an entry that glossary.txt or an enrolled speaker would bring back is switched to status 'disabled'
(a tombstone sync_sources respects) instead of being removed.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, NamedTuple

from ..db import now_iso
from . import glossary_file, pinyin

log = logging.getLogger(__name__)

KV_VERSION = "lexicon_version"
KV_GLOSSARY_SIG = "lexicon_glossary_sig"

TERM_KINDS = ("person", "term", "project", "org", "place", "product", "other")
TERM_STATUSES = ("active", "candidate", "disabled")
TERM_SOURCES = ("user", "answer", "edit", "glossary", "speaker", "auto")
ALIAS_KINDS = ("mishear", "aka")
ALIAS_SCOPES = ("auto", "hint")
ALIAS_SOURCES = ("user", "answer", "edit", "llm")

KIND_LABELS = {"person": "人名", "term": "术语", "project": "项目", "org": "机构", "place": "地点", "product": "产品", "other": "其他"}

#: an alias that is a dictionary word more frequent than this is a common word: scope 'hint' (see module docstring)
DEFAULT_ALIAS_AUTO_MAX_FREQ = 500
MAX_TERM_CHARS = 40
#: sync_sources_throttled: glossary.txt and the speakers table are looked at this often at most
SYNC_EVERY_S = 30.0

_ASCII_WORD = re.compile(r"[A-Za-z0-9]")
_HAS_WORD = re.compile(r"\w", re.UNICODE)


# ============================================================================================== data
@dataclass
class Alias:
    id: int
    lexicon_id: int
    alias: str
    kind: str = "mishear"          # mishear | aka
    scope: str = "auto"            # auto | hint
    source: str = "user"
    hits: int = 0
    pinyin: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "lexicon_id": self.lexicon_id, "alias": self.alias, "kind": self.kind, "scope": self.scope,
                "source": self.source, "hits": self.hits}


@dataclass
class Term:
    id: int
    term: str
    kind: str = "term"
    note: str | None = None
    aliases: list[Alias] = field(default_factory=list)
    hotword: bool = True
    status: str = "active"
    source: str = "user"
    hits: int = 0
    speaker_id: int | None = None
    pinyin: str | None = None
    last_hit_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "term": self.term, "kind": self.kind, "note": self.note, "pinyin": self.pinyin,
                "speaker_id": self.speaker_id, "source": self.source, "status": self.status, "hotword": bool(self.hotword),
                "hits": self.hits, "last_hit_at": self.last_hit_at, "created_at": self.created_at,
                "updated_at": self.updated_at, "aliases": [a.as_dict() for a in self.aliases]}

    def mishears(self) -> list[str]:
        return [a.alias for a in self.aliases if a.kind == "mishear"]


class Hit(NamedTuple):
    """An entry found in a text: its canonical spelling (alias is None) or one of its aliases."""
    start: int
    end: int
    text: str
    term: Term
    alias: Alias | None


class NearHit(NamedTuple):
    """A stretch of text that sounds like an entry without being spelled like it."""
    term: Term
    start: int
    end: int
    text: str
    similarity: float


def _pattern(surface: str) -> str:
    """Regex for one spelling: literal, and a Latin / digit edge must not continue a longer word or number
    ("SDK" is not found inside "MSDKX", "T30" not inside "T300")."""
    pat = re.escape(surface)
    if _ASCII_WORD.match(surface[0]):
        pat = r"(?<![A-Za-z0-9])" + pat
    if _ASCII_WORD.match(surface[-1]):
        pat = pat + r"(?![A-Za-z0-9])"
    return pat


# ============================================================================================== snapshot
class Lexicon:
    """Immutable snapshot of the lexicon (see the module docstring). Build with Lexicon.load(db)."""

    _cache: dict[str, "Lexicon"] = {}
    _cache_lock = threading.Lock()

    def __init__(self, terms: Iterable[Term] = (), version: Any = 0):
        self.version = version               # what _version(db) returned when this snapshot was read
        self.terms: list[Term] = list(terms)
        self.active: list[Term] = [t for t in self.terms if t.status == "active"]
        self.by_id: dict[int, Term] = {t.id: t for t in self.terms}
        self.by_term: dict[str, Term] = {t.term: t for t in self.terms}
        self.alias_by_id: dict[int, tuple[Term, Alias]] = {a.id: (t, a) for t in self.terms for a in t.aliases}
        # surface (lower-cased) -> what it stands for. A canonical spelling always wins over an alias of another
        # entry; an alias that points at several entries is ambiguous and is never rewritten.
        self._surface: dict[str, tuple[Term, Alias | None]] = {}
        self._ambiguous: set[str] = set()
        for t in self.active:
            self._surface[t.term.lower()] = (t, None)
        for t in self.active:
            for a in t.aliases:
                k = a.alias.lower()
                cur = self._surface.get(k)
                if cur is None:
                    self._surface[k] = (t, a)
                elif cur[1] is not None and cur[0].id != t.id and a.kind == "mishear" and cur[1].kind == "mishear":
                    self._ambiguous.add(k)
        self._regex: re.Pattern[str] | None = None
        self._units: dict[int, tuple[pinyin.Unit, ...]] = {}

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, db: Any) -> "Lexicon":
        """The current lexicon of `db` (cached per database file; a write anywhere invalidates it)."""
        key = str(getattr(db, "path", id(db)))
        version = _version(db)
        with cls._cache_lock:
            cached = cls._cache.get(key)
            if cached is not None and cached.version == version:
                return cached
        lex = cls(_read_terms(db), version)
        with cls._cache_lock:
            cls._cache[key] = lex
        return lex

    @classmethod
    def forget(cls, db: Any = None) -> None:
        """Drop the cached snapshot of `db` (all of them without an argument). Only needed by tests that swap
        database files under one path; writes invalidate on their own."""
        with cls._cache_lock:
            if db is None:
                cls._cache.clear()
            else:
                cls._cache.pop(str(getattr(db, "path", id(db))), None)

    # ------------------------------------------------------------------ exact matches
    def _matcher(self) -> re.Pattern[str] | None:
        if self._regex is None and self._surface:
            # longest spelling first: the alternation then prefers "青禾App" over "青禾" at the same position
            surfaces = sorted(self._surface, key=lambda s: (-len(s), s))
            self._regex = re.compile("|".join(_pattern(s) for s in surfaces), re.IGNORECASE)
        return self._regex

    def find_in(self, text: str) -> list[Hit]:
        """Every entry spelled out in `text`, canonical or by alias: non-overlapping, left to right, the longest
        spelling at each position. Latin letters match in any case."""
        rx = self._matcher()
        if not text or rx is None:
            return []
        out: list[Hit] = []
        for m in rx.finditer(text):
            got = self._surface.get(m.group(0).lower())
            if got is not None:
                out.append(Hit(m.start(), m.end(), m.group(0), got[0], got[1]))
        return out

    def resolve(self, name: str) -> Term | None:
        """The active entry `name` is a spelling of (canonical or any alias), or None."""
        got = self._surface.get((name or "").strip().lower())
        return got[0] if got else None

    def is_term(self, text: str) -> bool:
        """True when `text` is the canonical spelling of an active entry."""
        got = self._surface.get((text or "").strip().lower())
        return bool(got and got[1] is None)

    # ------------------------------------------------------------------ deterministic first pass
    def apply(self, text: str, *, only_alias_ids: Iterable[int] | None = None,
              only_lexicon_ids: Iterable[int] | None = None, force: bool = False,
              skip_alias_ids: Iterable[int] | None = None) -> tuple[str, list[dict[str, Any]]]:
        """Rewrite the mishear aliases found in `text` to their canonical spelling.

        Returns (new_text, edits). An edit is {"from", "to", "reason", "alias_id", "lexicon_id", "pos", "start",
        "end"}: `pos` is where `from` stood in the input, [start, end) is where `to` stands in the result.
        Only scope='auto' aliases that point at exactly one entry are applied; `force=True` (an explicit request
        of the user for the aliases named in only_alias_ids) also applies scope='hint' ones. A spelling inside a
        longer known spelling is left alone (find_in is longest-match)."""
        only_a = None if only_alias_ids is None else {int(x) for x in only_alias_ids}
        only_t = None if only_lexicon_ids is None else {int(x) for x in only_lexicon_ids}
        skip = set() if skip_alias_ids is None else {int(x) for x in skip_alias_ids}
        edits: list[dict[str, Any]] = []
        out: list[str] = []
        last = 0
        shift = 0
        for h in self.find_in(text):
            a = h.alias
            if a is None or a.kind != "mishear" or a.id in skip:
                continue
            if only_a is not None and a.id not in only_a:
                continue
            if only_t is not None and h.term.id not in only_t:
                continue
            if not force and (a.scope != "auto" or a.alias.lower() in self._ambiguous):
                continue
            to = h.term.term
            if h.text == to:
                continue
            out.append(text[last:h.start])
            out.append(to)
            start = h.start + shift
            edits.append({"from": h.text, "to": to, "reason": "词库", "alias_id": a.id, "lexicon_id": h.term.id,
                          "pos": h.start, "start": start, "end": start + len(to)})
            shift += len(to) - (h.end - h.start)
            last = h.end
        if not edits:
            return text, []
        out.append(text[last:])
        return "".join(out), edits

    # ------------------------------------------------------------------ sound-alike candidates
    def _term_units(self, t: Term) -> tuple[pinyin.Unit, ...]:
        u = self._units.get(t.id)
        if u is None:
            # a stored key wins (it may carry a reading pypinyin would not choose); the key column of an entry
            # created without pypinyin holds characters and is recomputed here
            stored = (t.pinyin or "").strip()
            fresh = pinyin.units(t.term)
            u = fresh
            if stored and pinyin.available() and stored != " ".join(x.syl for x in fresh):
                alt = pinyin.units_from_key(stored)
                if len(alt) == len(fresh):
                    u = alt
            self._units[t.id] = u
        return u

    def near_spans(self, text: str, limit: int = 20, min_sim: float = pinyin.CANDIDATE_MIN_SIM) -> list[NearHit]:
        """Stretches of `text` that sound like an active entry but are not spelled like it (and are not one of
        its known aliases - find_in reports those): candidates for "did they say <term> here?". Best first.
        Windows have as many syllables as the entry; entries shorter than two syllables are skipped - a single
        syllable sounds like half the language (a Latin acronym of three letters or more is compared with the
        Latin runs of the text instead)."""
        if not text or not self.active:
            return []
        tu = pinyin.units(text)
        if not tu:
            return []
        pos_by_syl: dict[str, list[int]] = {}
        for i, u in enumerate(tu):
            pos_by_syl.setdefault(_sound_class(u), []).append(i)
        exact = [(h.start, h.end, h.term.id) for h in self.find_in(text)]
        found: dict[tuple[int, int], NearHit] = {}
        for t in self.active:
            us = self._term_units(t)
            n = len(us)
            if n < 2 and not (n == 1 and us[0].latin and len(us[0].syl) >= 3):
                continue
            starts: set[int] = set()
            for k, u in enumerate(us):
                for p in pos_by_syl.get(_sound_class(u), ()):
                    s = p - k
                    if 0 <= s <= len(tu) - n:
                        starts.add(s)
            if n == 1:      # an acronym: compare with every Latin run of the text
                starts = {i for i, u in enumerate(tu) if u.latin}
            for s in starts:
                window = tu[s:s + n]
                sim = pinyin.similarity_units(us, window)
                if sim < float(min_sim) - 1e-9:
                    continue
                a, b = window[0].start, window[-1].end
                piece = text[a:b]
                # not a candidate: the entry itself, a stretch inside a spelling that is already known (part of
                # a longer entry, or one of this entry's aliases - find_in reports those)
                if piece.lower() == t.term.lower() or any((st <= a and b <= e) or (tid == t.id and a < e and st < b)
                                                          for st, e, tid in exact):
                    continue
                cur = found.get((t.id, a))
                if cur is None or sim > cur.similarity:
                    found[(t.id, a)] = NearHit(t, a, b, piece, round(sim, 4))
        hits = sorted(found.values(), key=lambda h: (-h.similarity, -h.term.hits, h.start, h.term.id))
        return hits[:max(0, int(limit))] if limit else hits

    def near(self, text: str, limit: int = 20) -> list[Term]:
        """Active entries that sound like some stretch of `text` (see near_spans), best match first."""
        out: list[Term] = []
        seen: set[int] = set()
        for h in self.near_spans(text, limit=0):
            if h.term.id not in seen:
                seen.add(h.term.id)
                out.append(h.term)
                if limit and len(out) >= int(limit):
                    break
        return out

    # ------------------------------------------------------------------ hot words / prompts / search
    def hotwords(self, limit: int = 50, context: str = "") -> list[str]:
        """Canonical spellings to hand to the ASR as soft hot words, most useful first: entries found in (or
        sounding like something in) `context`, then enrolled speakers, then what the user confirmed (lexicon
        page, answers, hand edits), then glossary terms; inside a group the most and most recently hit first.
        Only active entries with the hotword flag, and nothing shorter than two characters."""
        cands = [t for t in self.active if t.hotword and len(t.term.strip()) >= 2]
        if not cands or limit is not None and int(limit) <= 0:
            return []
        in_ctx: set[int] = set()
        if context:
            in_ctx = {h.term.id for h in self.find_in(context)} | {h.term.id for h in self.near_spans(context, limit=0)}

        def rank(t: Term) -> tuple[Any, ...]:
            if t.speaker_id is not None or t.source == "speaker":
                tier = 0
            elif t.source in ("user", "answer", "edit"):
                tier = 1
            elif t.source == "glossary":
                tier = 2
            else:
                tier = 3
            return (0 if t.id in in_ctx else 1, tier, -int(t.hits or 0), _neg_time(t.last_hit_at), t.id)

        return [t.term for t in sorted(cands, key=rank)[:int(limit)]]

    @staticmethod
    def prompt_rows(terms: Iterable[Term]) -> str:
        """Table rows for a correction prompt, one entry per line: 写法｜说明｜常见误识别."""
        rows = []
        for t in terms:
            note = (t.note or "").strip() or KIND_LABELS.get(t.kind, "")
            rows.append(f"{t.term}｜{note}｜{'、'.join(t.mishears()[:6])}")
        return "\n".join(rows)

    def expand(self, text: str) -> list[str]:
        """Other spellings worth searching for when `text` (a query) names an entry: the canonical spelling and
        every alias of each entry found in it, without the spellings already present."""
        out: list[str] = []
        low = (text or "").lower()
        for h in self.find_in(text or ""):
            for s in [h.term.term] + [a.alias for a in h.term.aliases]:
                if s.lower() not in low and s not in out:
                    out.append(s)
        return out

    def jieba_words(self) -> list[str]:
        """Words the tokenizer should keep whole: canonical spellings and nicknames of active entries."""
        out = [t.term for t in self.active]
        out += [a.alias for t in self.active for a in t.aliases if a.kind == "aka"]
        return out


def _sound_class(u: pinyin.Unit) -> str:
    """Bucket of a token for candidate search: syllables that may cost less than a full step share a bucket."""
    if u.latin:
        return "@" + u.syl[:1]
    ini, fin = pinyin.split_syllable(u.syl)
    ini = {"zh": "z", "ch": "c", "sh": "s", "l": "n", "r": "n", "h": "f"}.get(ini, ini)
    fin = {"ang": "an", "eng": "en", "ing": "in", "iang": "ian", "uang": "uan", "ong": "un"}.get(fin, fin)
    return ini + fin


def _neg_time(ts: str | None) -> float:
    """Sort key: the more recent the timestamp, the smaller (None last)."""
    if not ts:
        return 0.0
    try:
        from datetime import datetime

        return -datetime.fromisoformat(str(ts).replace(" ", "T")).timestamp()
    except ValueError:
        return 0.0


# ============================================================================================== reading rows
def _version(db: Any) -> Any:
    """What a snapshot is valid for: the kv value every write of this module changes, plus the row counts and
    highest ids of the two tables - so rows another module inserted or deleted with plain SQL are noticed too
    (an in-place UPDATE that bypasses this module is not: go through update_term)."""
    try:
        r = db.one("SELECT (SELECT value FROM kv WHERE key=?), (SELECT COUNT(*) FROM lexicon), (SELECT COALESCE(MAX(id),0) FROM lexicon),"
                   " (SELECT COUNT(*) FROM lexicon_aliases), (SELECT COALESCE(MAX(id),0) FROM lexicon_aliases)", (KV_VERSION,))
        return tuple(r)
    except Exception:  # noqa: BLE001 - a database without the tables / a closed one: never equal, never cached
        return object()


def _bump(con: Any) -> None:
    """Invalidate every snapshot, in the writer's transaction (other processes see it with the data). The value
    is "<token>:<counter>": the token is drawn when the database gets its first lexicon write, so two database
    files that lived under one path at different times never share a version."""
    row = con.execute("SELECT value FROM kv WHERE key=?", (KV_VERSION,)).fetchone()
    token, n = "", 0
    try:
        cur = json.loads(row[0]) if row else None
        if isinstance(cur, str) and ":" in cur:
            token, tail = cur.rsplit(":", 1)
            n = int(tail)
    except (TypeError, ValueError):
        token, n = "", 0
    if not token:
        token = uuid.uuid4().hex[:8]
    con.execute("INSERT INTO kv(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (KV_VERSION, json.dumps(f"{token}:{n + 1}")))


def _read_terms(db: Any) -> list[Term]:
    terms: dict[int, Term] = {}
    for r in db.query("SELECT * FROM lexicon ORDER BY id"):
        terms[int(r["id"])] = Term(id=int(r["id"]), term=r["term"], kind=r["kind"] or "term", note=r["note"],
                                   hotword=bool(r["hotword"]), status=r["status"] or "active", source=r["source"] or "user",
                                   hits=int(r["hits"] or 0), speaker_id=r["speaker_id"], pinyin=r["pinyin"],
                                   last_hit_at=r["last_hit_at"], created_at=r["created_at"], updated_at=r["updated_at"])
    for r in db.query("SELECT * FROM lexicon_aliases ORDER BY id"):
        t = terms.get(int(r["lexicon_id"]))
        if t is not None:
            t.aliases.append(Alias(id=int(r["id"]), lexicon_id=t.id, alias=r["alias"], kind=r["kind"] or "mishear",
                                   scope=r["scope"] or "auto", source=r["source"] or "user", hits=int(r["hits"] or 0),
                                   pinyin=r["pinyin"]))
    return list(terms.values())


def get_term(db: Any, term_id: int) -> dict[str, Any] | None:
    t = Lexicon.load(db).by_id.get(int(term_id))
    return t.as_dict() if t else None


def get_alias(db: Any, alias_id: int) -> dict[str, Any] | None:
    """{"id", "alias", ..., "term": <term dict>} or None."""
    got = Lexicon.load(db).alias_by_id.get(int(alias_id))
    if got is None:
        return None
    return {**got[1].as_dict(), "term": got[0].term}


def list_terms(db: Any, *, q: str | None = None, kind: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
    """Entries as dicts (with their aliases). status: None = everything but tombstones (active + candidate),
    "all", or one status. q: substring of the spelling, the note or an alias, or the start of its pinyin."""
    lex = Lexicon.load(db)
    q = (q or "").strip().lower()
    qkey = pinyin.key(q) if q else ""
    out = []
    for t in lex.terms:
        if kind and t.kind != kind:
            continue
        if status in (None, ""):
            if t.status == "disabled":
                continue
        elif status != "all" and t.status != status:
            continue
        if q:
            hay = [t.term.lower(), (t.note or "").lower()] + [a.alias.lower() for a in t.aliases]
            if not any(q in h for h in hay) and not (qkey and (t.pinyin or "").startswith(qkey)):
                continue
        out.append(t)
    out.sort(key=lambda t: (t.status != "active", -int(t.hits or 0), t.term))
    return [t.as_dict() for t in out]


def counts(db: Any) -> dict[str, Any]:
    lex = Lexicon.load(db)
    by_kind: dict[str, int] = {}
    for t in lex.active:
        by_kind[t.kind] = by_kind.get(t.kind, 0) + 1
    return {"active": len(lex.active), "candidate": sum(1 for t in lex.terms if t.status == "candidate"),
            "disabled": sum(1 for t in lex.terms if t.status == "disabled"),
            "aliases": sum(len(t.aliases) for t in lex.active), "by_kind": by_kind}


# ============================================================================================== writing
def _clean(text: Any, what: str) -> str:
    s = " ".join(str(text or "").split())
    if not s or not _HAS_WORD.search(s):
        raise ValueError(f"{what}不能为空")
    if len(s) > MAX_TERM_CHARS:
        raise ValueError(f"{what}太长了（最多 {MAX_TERM_CHARS} 个字）")
    return s


def _check(value: Any, allowed: tuple[str, ...], what: str) -> str:
    v = str(value or "").strip()
    if v not in allowed:
        raise ValueError(f"{what}只能是 {' / '.join(allowed)}")
    return v


def _publish(events: Any, change: str, **data: Any) -> None:
    if events is None:
        return
    try:
        events.publish("lexicon", change=change, **data)
    except Exception:  # noqa: BLE001 - telling the page is never worth failing a write
        log.debug("lexicon event not published", exc_info=True)


def _find_row(con: Any, term: str) -> Any:
    row = con.execute("SELECT * FROM lexicon WHERE term=?", (term,)).fetchone()
    if row is None and term.isascii():
        # "okr" and "OKR" are one entry: Latin spellings match in any case everywhere else too
        row = con.execute("SELECT * FROM lexicon WHERE lower(term)=lower(?) ORDER BY id LIMIT 1", (term,)).fetchone()
    return row


def upsert_term(db: Any, term: str, *, kind: str = "term", note: str | None = None, source: str = "user",
                status: str = "active", speaker_id: int | None = None, hotword: bool | None = None,
                events: Any = None) -> int:
    """Create the entry `term`, or update the one that exists. Returns its id.

    On an existing entry: a more specific kind replaces the generic 'term', a given note / speaker / hotword
    flag is stored, and status 'active' switches a candidate or a disabled entry back on (this is the explicit
    path - sync_sources never does that). The original source is kept unless it was 'auto'."""
    term = _clean(term, "词条")
    kind = _check(kind or "term", TERM_KINDS, "类别")
    source = _check(source or "user", TERM_SOURCES, "来源")
    status = _check(status or "active", TERM_STATUSES, "状态")
    now = now_iso()
    with db.transaction() as con:
        row = _find_row(con, term)
        if row is None:
            hw = 1 if (hotword is None or hotword) else 0
            cur = con.execute(
                "INSERT INTO lexicon(term, kind, note, pinyin, speaker_id, source, status, hotword, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (term, kind, (note or None), pinyin.key(term), speaker_id, source, status, hw, now, now))
            _bump(con)
            tid = int(cur.lastrowid)
            created = True
        else:
            tid = int(row["id"])
            sets: dict[str, Any] = {}
            if kind != "term" and row["kind"] in ("term", "other") and kind != row["kind"]:
                sets["kind"] = kind
            if note is not None and (note or None) != row["note"]:
                sets["note"] = note or None
            if speaker_id is not None and row["speaker_id"] != speaker_id:
                sets["speaker_id"] = int(speaker_id)
            if hotword is not None and bool(row["hotword"]) != bool(hotword):
                sets["hotword"] = 1 if hotword else 0
            if status == "active" and row["status"] != "active":
                sets["status"] = "active"
            if row["source"] == "auto" and source != "auto":
                sets["source"] = source
            if sets:
                sets["updated_at"] = now
                con.execute(f"UPDATE lexicon SET {', '.join(k + '=?' for k in sets)} WHERE id=?", [*sets.values(), tid])
                _bump(con)
            created = False
    if created and status == "active":
        reindex_spelling(db, term)
    _publish(events, "term", id=tid, created=created)
    return tid


#: words rarer than this are not kept in the frequency table (they count as 0: "not a dictionary word to speak of")
_FREQ_FLOOR = 20
_base_freq: dict[str, int] | None = None
_base_freq_lock = threading.Lock()


def word_frequency(word: str) -> int:
    """Frequency of `word` in jieba's dictionary FILE (0 = rarer than _FREQ_FLOOR, unknown, or no jieba).

    The file, not the tokenizer's live table: registering a word with the tokenizer (jieba.add_word, which
    happens to every lexicon spelling) rewrites its live frequency - a made-up alias of two common characters
    comes out above 500 and would suddenly count as an everyday word. The table (about 80,000 words) is read
    once per process, on the first question."""
    global _base_freq
    if _base_freq is None:
        with _base_freq_lock:
            if _base_freq is None:
                table: dict[str, int] = {}
                try:
                    import jieba

                    with jieba.get_dict_file() as f:
                        for raw in f:
                            parts = raw.split(b" ", 2)
                            if len(parts) >= 2 and parts[1].isdigit() and int(parts[1]) >= _FREQ_FLOOR:
                                table[parts[0].decode("utf-8")] = int(parts[1])
                except Exception as e:  # noqa: BLE001 - no jieba: every alias is "rare", the other scope rules still hold
                    log.debug("word frequencies unavailable: %s", e)
                _base_freq = table
    return _base_freq.get(word, 0)


def alias_auto_max_freq(cfg: Any = None) -> int:
    try:
        return int(cfg.correction.get("alias_auto_max_freq", DEFAULT_ALIAS_AUTO_MAX_FREQ)) if cfg is not None \
            else DEFAULT_ALIAS_AUTO_MAX_FREQ
    except (AttributeError, TypeError, ValueError):
        return DEFAULT_ALIAS_AUTO_MAX_FREQ


def auto_scope(db: Any, lexicon_id: int, alias: str, cfg: Any = None) -> tuple[str, str]:
    """(scope, why) for a mishear alias nobody chose a scope for: 'hint' when rewriting it blindly could
    damage correct text - why is one of single / term / number / ambiguous / common - else ('auto', 'ok')."""
    alias = (alias or "").strip()
    if len(alias) <= 1:
        return "hint", "single"
    lex = Lexicon.load(db)
    if lex.is_term(alias):
        return "hint", "term"
    target = lex.by_id.get(int(lexicon_id))
    if target is not None:
        from .correct import digit_runs

        if digit_runs(alias) != digit_runs(target.term):
            return "hint", "number"            # no automatic rule may change a number (the old corrector's hard rule)
    for t in lex.terms:
        if t.id != int(lexicon_id) and any(a.alias.lower() == alias.lower() and a.kind == "mishear" for a in t.aliases):
            return "hint", "ambiguous"
    if word_frequency(alias) > alias_auto_max_freq(cfg):
        return "hint", "common"
    return "auto", "ok"


def add_alias(db: Any, lexicon_id: int, alias: str, *, kind: str = "mishear", scope: str | None = None,
              source: str = "user", cfg: Any = None, events: Any = None) -> int:
    """Attach `alias` to an entry (or update the pair when it exists). Returns the alias id.

    kind 'mishear': a wrong spelling to rewrite; 'aka': another name, never rewritten. scope None = decide
    here (auto_scope). When the alias already points at another entry as a mis-hearing nobody can tell which
    one is meant: this one becomes a hint and the older one is downgraded to a hint too."""
    alias = _clean(alias, "别名")
    kind = _check(kind or "mishear", ALIAS_KINDS, "别名类型")
    source = _check(source or "user", ALIAS_SOURCES, "来源")
    if scope is not None:
        scope = _check(scope, ALIAS_SCOPES, "范围")
    lexicon_id = int(lexicon_id)
    term = Lexicon.load(db).by_id.get(lexicon_id)
    if term is None:
        raise LookupError("没有这个词条")
    if alias.lower() == term.term.lower():
        raise ValueError("别名不能和词条本身一样")
    decided = scope
    if kind == "aka":
        decided = "hint"                      # stored for completeness; an aka is never rewritten
    elif decided is None:
        decided, _why = auto_scope(db, lexicon_id, alias, cfg)
    with db.transaction() as con:
        row = con.execute("SELECT id, kind, scope FROM lexicon_aliases WHERE alias=? AND lexicon_id=?", (alias, lexicon_id)).fetchone()
        if row is not None:
            aid = int(row["id"])
            if row["kind"] != kind or (scope is not None and row["scope"] != decided) or (kind == "aka" and row["scope"] != "hint"):
                con.execute("UPDATE lexicon_aliases SET kind=?, scope=? WHERE id=?",
                            (kind, decided if (scope is not None or kind == "aka") else row["scope"], aid))
                _bump(con)
        else:
            cur = con.execute("INSERT INTO lexicon_aliases(lexicon_id, alias, kind, scope, pinyin, source, created_at)"
                              " VALUES (?,?,?,?,?,?,?)", (lexicon_id, alias, kind, decided, pinyin.key(alias), source, now_iso()))
            aid = int(cur.lastrowid)
            if kind == "mishear":
                con.execute("UPDATE lexicon_aliases SET scope='hint' WHERE lower(alias)=lower(?) AND lexicon_id<>? AND kind='mishear'"
                            " AND scope='auto'", (alias, lexicon_id))
            con.execute("UPDATE lexicon SET updated_at=? WHERE id=?", (now_iso(), lexicon_id))
            _bump(con)
    _publish(events, "alias", id=aid, lexicon_id=lexicon_id)
    return aid


def update_term(db: Any, term_id: int, *, term: str | None = None, kind: str | None = None, note: str | None = None,
                hotword: bool | None = None, status: str | None = None, pinyin_key: str | None = None,
                events: Any = None) -> dict[str, Any]:
    """Edit an entry (only the given fields). Renaming keeps the aliases; the old spelling is NOT turned into
    an alias (the caller decides). Raises LookupError / ValueError with a message for the user."""
    tid = int(term_id)
    with db.transaction() as con:
        row = con.execute("SELECT * FROM lexicon WHERE id=?", (tid,)).fetchone()
        if row is None:
            raise LookupError("没有这个词条")
        sets: dict[str, Any] = {}
        if term is not None:
            new = _clean(term, "词条")
            if new != row["term"]:
                if row["speaker_id"] is not None and row["source"] == "speaker":
                    raise ValueError("这是说话人的名字：请到「说话人」里改名")
                other = _find_row(con, new)
                if other is not None and int(other["id"]) != tid:
                    raise ValueError("已经有这个词条了")
                sets["term"] = new
                sets["pinyin"] = pinyin.key(new)
        if kind is not None:
            sets["kind"] = _check(kind, TERM_KINDS, "类别")
        if note is not None:
            sets["note"] = str(note).strip()[:200] or None
        if hotword is not None:
            sets["hotword"] = 1 if hotword else 0
        if status is not None:
            sets["status"] = _check(status, TERM_STATUSES, "状态")
        if pinyin_key is not None:
            sets["pinyin"] = " ".join(str(pinyin_key).lower().split()) or pinyin.key(sets.get("term", row["term"]))
        if sets:
            sets["updated_at"] = now_iso()
            con.execute(f"UPDATE lexicon SET {', '.join(k + '=?' for k in sets)} WHERE id=?", [*sets.values(), tid])
            _bump(con)
    _publish(events, "term", id=tid)
    return get_term(db, tid) or {}


def delete_term(db: Any, cfg: Any, term_id: int, *, events: Any = None) -> dict[str, Any]:
    """Remove an entry and its aliases. An entry that would come straight back - the name of an enrolled
    speaker, a line of glossary.txt - is switched off instead (status 'disabled': a tombstone).
    Returns {"deleted": bool, "disabled": bool}."""
    tid = int(term_id)
    with db.transaction() as con:
        row = con.execute("SELECT * FROM lexicon WHERE id=?", (tid,)).fetchone()
        if row is None:
            raise LookupError("没有这个词条")
        keep = False
        if row["speaker_id"] is not None and con.execute("SELECT 1 FROM speakers WHERE id=?", (row["speaker_id"],)).fetchone():
            keep = True
        if not keep and cfg is not None:
            try:
                keep = row["term"] in glossary_file.read_glossary(cfg.glossary_path)
            except Exception:  # noqa: BLE001 - an unreadable file cannot bring anything back
                keep = False
        if keep:
            con.execute("UPDATE lexicon SET status='disabled', updated_at=? WHERE id=?", (now_iso(), tid))
        else:
            con.execute("DELETE FROM lexicon_aliases WHERE lexicon_id=?", (tid,))
            con.execute("DELETE FROM lexicon WHERE id=?", (tid,))
        _bump(con)
    _publish(events, "term", id=tid, deleted=True)
    return {"deleted": not keep, "disabled": keep}


def delete_alias(db: Any, alias_id: int, *, events: Any = None) -> bool:
    with db.transaction() as con:
        cur = con.execute("DELETE FROM lexicon_aliases WHERE id=?", (int(alias_id),))
        if cur.rowcount:
            _bump(con)
    if cur.rowcount:
        _publish(events, "alias", id=int(alias_id), deleted=True)
    return bool(cur.rowcount)


def touch(db: Any, *, lexicon_ids: Iterable[int] = (), alias_ids: Iterable[int] = ()) -> None:
    """Count hits (an id listed twice counts twice): what was just said or fixed moves up in hotwords().
    Best effort - statistics never fail the caller."""
    lids = [int(x) for x in lexicon_ids]
    aids = [int(x) for x in alias_ids]
    if not lids and not aids:
        return
    try:
        now = now_iso()
        with db.transaction() as con:
            for lid in set(lids):
                con.execute("UPDATE lexicon SET hits=hits+?, last_hit_at=? WHERE id=?", (lids.count(lid), now, lid))
            for aid in set(aids):
                con.execute("UPDATE lexicon_aliases SET hits=hits+? WHERE id=?", (aids.count(aid), aid))
            _bump(con)
    except Exception as e:  # noqa: BLE001
        log.debug("lexicon hits not recorded: %s", e)


# ============================================================================================== glossary.txt / speakers
def _glossary_sig(cfg: Any) -> tuple[str, list[str] | None]:
    """(fingerprint of glossary.txt, its terms). ('', []) without a file; (sig, None) when it cannot be read."""
    path = getattr(cfg, "glossary_path", None)
    try:
        if not path or not path.is_file():
            return "", []
        data = path.read_bytes()
    except OSError as e:
        log.warning("cannot read glossary: %s", e)
        return "?", None
    sig = hashlib.sha1(data).hexdigest()[:16]
    try:
        return sig, glossary_file.parse_glossary(data.decode("utf-8-sig"))
    except UnicodeDecodeError as e:
        log.warning("cannot read glossary: %s", e)
        return sig, None


def sync_sources(db: Any, cfg: Any) -> dict[str, int]:
    """Bring the terms of glossary.txt and the names of the enrolled speakers into the lexicon. Idempotent and
    cheap; safe to call often.

    glossary.txt is imported when its content changed since the last import (first run included): a term
    without an entry gets one (source 'glossary'); existing entries are left as they are, tombstones included.
    Speakers: every speaker has an entry of kind 'person' linked by speaker_id (source 'speaker'); a renamed
    speaker renames its entry. A one-character name is no hot word."""
    out = {"glossary_added": 0, "speakers_added": 0, "speakers_renamed": 0}
    sig, terms = _glossary_sig(cfg)
    try:
        known_sig = db.get_kv(KV_GLOSSARY_SIG, None)
    except Exception:  # noqa: BLE001
        known_sig = None
    now = now_iso()
    with db.transaction() as con:
        changed = False
        if terms is not None and sig != known_sig:
            for term in terms:
                try:
                    term = _clean(term, "词条")
                except ValueError:
                    continue
                if _find_row(con, term) is None:
                    con.execute("INSERT INTO lexicon(term, kind, pinyin, source, status, hotword, created_at, updated_at)"
                                " VALUES (?,?,?,?,?,?,?,?)", (term, "term", pinyin.key(term), "glossary", "active", 1, now, now))
                    out["glossary_added"] += 1
                    changed = True
            con.execute("INSERT INTO kv(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (KV_GLOSSARY_SIG, '"' + sig + '"'))
        linked = {int(r["speaker_id"]): r for r in con.execute("SELECT * FROM lexicon WHERE speaker_id IS NOT NULL").fetchall()}
        for sp in con.execute("SELECT id, name FROM speakers ORDER BY id").fetchall():
            sid, name = int(sp["id"]), " ".join(str(sp["name"] or "").split())
            if not name or len(name) > MAX_TERM_CHARS:
                continue
            row = linked.get(sid)
            if row is not None and row["term"] == name:
                continue
            same = _find_row(con, name)
            if row is not None:
                if same is None and row["source"] == "speaker":
                    con.execute("UPDATE lexicon SET term=?, pinyin=?, updated_at=? WHERE id=?", (name, pinyin.key(name), now, int(row["id"])))
                    out["speakers_renamed"] += 1
                    changed = True
                    continue
                con.execute("UPDATE lexicon SET speaker_id=NULL, updated_at=? WHERE id=?", (now, int(row["id"])))
                changed = True
            if same is not None:
                con.execute("UPDATE lexicon SET speaker_id=?, kind=CASE WHEN kind IN ('term','other') THEN 'person' ELSE kind END,"
                            " updated_at=? WHERE id=?", (sid, now, int(same["id"])))
            else:
                con.execute("INSERT INTO lexicon(term, kind, pinyin, speaker_id, source, status, hotword, created_at, updated_at)"
                            " VALUES (?,?,?,?,?,?,?,?,?)", (name, "person", pinyin.key(name), sid, "speaker", "active",
                                                           1 if len(name) >= 2 else 0, now, now))
                out["speakers_added"] += 1
            changed = True
        if changed:
            _bump(con)
    return out


_last_sync: dict[str, float] = {}


def sync_sources_throttled(db: Any, cfg: Any, every_s: float = SYNC_EVERY_S) -> None:
    """sync_sources at most every `every_s` seconds per database, never raising: for callers on a hot path
    (one processed segment, one page load)."""
    key = str(getattr(db, "path", id(db)))
    now = time.monotonic()
    if now - _last_sync.get(key, -1e9) < float(every_s):
        return
    _last_sync[key] = now
    try:
        sync_sources(db, cfg)
    except Exception as e:  # noqa: BLE001
        log.warning("lexicon sync skipped: %s: %s", type(e).__name__, e)


def import_terms(db: Any, terms: Iterable[str], *, source: str = "glossary", replace: bool = False,
                 events: Any = None) -> dict[str, int]:
    """Bulk import of canonical spellings (glossary text box, `lexicon import`).

    Every term gets an active entry (an existing one is switched back on). With `replace` the list is the whole
    vocabulary the user wants: active entries that are not in it and do not belong to an enrolled speaker are
    switched off (status 'disabled' - their aliases and history stay, the lexicon page can bring them back)."""
    source = _check(source, TERM_SOURCES, "来源")
    wanted: list[str] = []
    for t in terms:
        try:
            t = _clean(t, "词条")
        except ValueError:
            continue
        if t not in wanted:
            wanted.append(t)
    out = {"added": 0, "enabled": 0, "disabled": 0, "total": len(wanted)}
    now = now_iso()
    with db.transaction() as con:
        keep_ids: set[int] = set()
        for t in wanted:
            row = _find_row(con, t)
            if row is None:
                cur = con.execute("INSERT INTO lexicon(term, kind, pinyin, source, status, hotword, created_at, updated_at)"
                                  " VALUES (?,?,?,?,?,?,?,?)", (t, "term", pinyin.key(t), source, "active", 1, now, now))
                keep_ids.add(int(cur.lastrowid))
                out["added"] += 1
            else:
                keep_ids.add(int(row["id"]))
                if row["status"] != "active":
                    con.execute("UPDATE lexicon SET status='active', updated_at=? WHERE id=?", (now, int(row["id"])))
                    out["enabled"] += 1
        if replace:
            for row in con.execute("SELECT id FROM lexicon WHERE status='active' AND speaker_id IS NULL").fetchall():
                if int(row["id"]) not in keep_ids:
                    con.execute("UPDATE lexicon SET status='disabled', updated_at=? WHERE id=?", (now, int(row["id"])))
                    out["disabled"] += 1
        if out["added"] or out["enabled"] or out["disabled"]:
            _bump(con)
    _publish(events, "import", **out)
    return out


def export_terms(db: Any, *, include_speakers: bool = False) -> list[str]:
    """Canonical spellings of the active entries, oldest first. Speaker names are left out by default: they
    live in the speakers table and would otherwise be written into glossary.txt on the next save."""
    return [t.term for t in Lexicon.load(db).active if include_speakers or t.speaker_id is None]


def export_glossary(db: Any) -> str:
    """The lexicon as glossary text: one canonical spelling per line (speaker names excluded)."""
    terms = export_terms(db)
    return "\n".join(terms) + ("\n" if terms else "")


def glossary_text(db: Any, cfg: Any) -> str:
    """What GET /api/glossary shows: glossary.txt as the user wrote it (comments and order kept) without the
    lines of switched-off entries, followed by the active entries the file does not have yet. Saving this text
    back (import_terms(replace=True)) therefore changes nothing."""
    path = getattr(cfg, "glossary_path", None)
    text = ""
    try:
        if path and path.exists():
            text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        log.warning("cannot read glossary: %s", e)
    lex = Lexicon.load(db)
    off = {t.term for t in lex.terms if t.status == "disabled"}
    kept: list[str] = []
    present: set[str] = set()
    dropped = False
    for line in text.splitlines(keepends=True):
        term = glossary_file.parse_line(line)
        if term is not None and term in off:
            dropped = True
            continue
        if term is not None:
            present.add(term)
        kept.append(line)
    body = "".join(kept) if dropped else text
    low = {p.lower() for p in present}
    extra = [t for t in export_terms(db) if t.lower() not in low]
    if extra:
        if body and not body.endswith("\n"):
            body += "\n"
        body += "\n".join(extra) + "\n"
    return body


# ============================================================================================== tokenizer
_jieba_seen: dict[str, Any] = {}


def jieba_words(db: Any) -> list[str]:
    """Words the FTS tokenizer must keep whole (for memory.index.register_from)."""
    return Lexicon.load(db).jieba_words()


def reindex_spelling(db: Any, spelling: str) -> int:
    """Re-tokenize the stored sentences that contain `spelling`, after it became a word of its own (a new
    entry): the keyword index had cut it into pieces, and a search for the whole word would miss those
    sentences. Returns how many were re-tokenized. Best effort: search quality never fails a lexicon write
    (bulk imports rely on the `reindex` job instead)."""
    try:
        rows = db.query("SELECT id, text FROM utterances WHERE instr(text, ?) > 0", (spelling,))
        if not rows:
            return 0
        ensure_jieba(db)
        index = importlib.import_module("audio_note.memory.index")
        with db.transaction() as con:
            for r in rows:
                con.execute("INSERT INTO utterance_tokens(utterance_id, text_seg) VALUES (?,?)"
                            " ON CONFLICT(utterance_id) DO UPDATE SET text_seg=excluded.text_seg",
                            (int(r["id"]), index.tokenize(r["text"] or "")))
        return len(rows)
    except Exception as e:  # noqa: BLE001
        log.debug("sentences containing a new entry not re-tokenized: %s", e)
        return 0


def ensure_jieba(db: Any) -> None:
    """Register the lexicon's words with the tokenizer of this process, once per lexicon version. Called
    before text is re-tokenized (revisions.update_text) so a term learned a minute ago is one token."""
    key = str(getattr(db, "path", id(db)))
    lex = Lexicon.load(db)
    if _jieba_seen.get(key) == lex.version:
        return
    try:
        index = importlib.import_module("audio_note.memory.index")
        index.register_terms(lex.jieba_words())
        _jieba_seen[key] = lex.version
    except Exception as e:  # noqa: BLE001 - the memory package is optional for the lexicon
        log.debug("jieba registration skipped: %s", e)


# ============================================================================================== history
def _candidates(db: Any, needle: str, *, day: str | None = None, utterance_ids: Iterable[int] | None = None) -> list[Any]:
    """Utterances whose text contains `needle`, newest first (Latin letters in any case)."""
    where, params = [], []
    if needle.isascii():
        esc = needle.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        where.append("lower(text) LIKE ? ESCAPE '\\'")
        params.append(f"%{esc}%")
    else:
        where.append("instr(text, ?) > 0")
        params.append(needle)
    if day:
        where.append("day=?")
        params.append(day)
    ids = None if utterance_ids is None else [int(x) for x in utterance_ids]
    if ids is not None:
        if not ids:
            return []
        where.append(f"id IN ({','.join('?' * len(ids))})")
        params += ids
    return db.query(f"SELECT id, day, abs_start, text FROM utterances WHERE {' AND '.join(where)} ORDER BY abs_start DESC, id DESC",
                    params)


def count_alias_matches(db: Any, alias_id: int, *, exclude_utterance_id: int | None = None) -> int:
    """How many stored sentences apply_alias_to_history would change (optionally not counting one of them)."""
    ids = apply_alias_to_history(db, None, alias_id, dry_run=True, max_listed=0)["match_ids"]
    return sum(1 for i in ids if i != exclude_utterance_id)


def apply_alias_to_history(db: Any, cfg: Any, alias_id: int, *, dry_run: bool = False, limit: int | None = None,
                           source: str = "lexicon", detail: dict[str, Any] | None = None, events: Any = None,
                           day: str | None = None, utterance_ids: Iterable[int] | None = None,
                           max_listed: int = 200, respect_undone: bool = True) -> dict[str, Any]:
    """Rewrite one alias to its term in the stored sentences, newest first, each through revisions.update_text
    (undoable; `source` and `detail` go into the revision).

    This is an explicit request, so the alias's scope does not matter; what does: a spelling inside a longer
    known spelling is left alone, and a sentence where the user undid this very rewrite is skipped
    (respect_undone=False lifts that: the user is answering about exactly these sentences).
    Returns {"alias_id", "alias", "term", "lexicon_id", "matches": [{utterance_id, day, time, before, after}],
    "match_ids", "total", "applied", "changed", "dry_run"}: `matches` describes at most max_listed sentences,
    `match_ids` / `total` cover all of them, `changed` are the ids rewritten now, `limit` caps how many are."""
    lex = Lexicon.load(db)
    got = lex.alias_by_id.get(int(alias_id))
    if got is None:
        raise LookupError("没有这个别名")
    term, alias = got
    if alias.kind != "mishear":
        raise ValueError("这是别称，不是误识别：不会改写文字")
    from . import revisions

    rows = _candidates(db, alias.alias, day=day, utterance_ids=utterance_ids)
    vetoed = revisions.undone_aliases(db, [int(r["id"]) for r in rows]) if rows and respect_undone else {}
    matches: list[dict[str, Any]] = []
    total = applied = 0
    changed_ids: list[int] = []
    match_ids: list[int] = []
    for r in rows:
        uid = int(r["id"])
        if alias.id in vetoed.get(uid, ()):
            continue
        new_text, edits = lex.apply(r["text"] or "", only_alias_ids=[alias.id], force=True)
        if not edits:
            continue
        total += 1
        match_ids.append(uid)
        if len(matches) < max_listed:
            matches.append({"utterance_id": uid, "day": r["day"], "time": (r["abs_start"] or "")[11:19],
                            "before": r["text"], "after": new_text})
        if dry_run or (limit is not None and applied >= int(limit)):
            continue
        d = {"edits": [{k: e[k] for k in ("from", "to", "reason", "alias_id", "lexicon_id", "start", "end")} for e in edits]}
        d.update(detail or {})
        if revisions.update_text(db, cfg, uid, new_text, source=source, detail=d, events=events, flush=False) is not None:
            applied += 1
            changed_ids.append(uid)
    if applied:
        revisions.flush_markdown(db, cfg)
        touch(db, lexicon_ids=[term.id] * applied, alias_ids=[alias.id] * applied)
    return {"alias_id": alias.id, "alias": alias.alias, "term": term.term, "lexicon_id": term.id, "matches": matches,
            "match_ids": match_ids, "total": total, "applied": applied, "changed": changed_ids, "dry_run": bool(dry_run)}


def recorrect(db: Any, cfg: Any, *, day: str | None = None, lexicon_id: int | None = None, alias_id: int | None = None,
              dry_run: bool = False, events: Any = None, stop_event: Any = None, batch: int = 500) -> dict[str, Any]:
    """Run the deterministic first pass over stored sentences (all of them, one day, one entry or one alias):
    what processing would do today with the lexicon as it is now. Through revisions (source 'lexicon');
    sentences where the user undid a rewrite keep their text. Returns {"scanned", "changed", "edits"}."""
    from . import revisions

    lex = Lexicon.load(db)
    only_a = [int(alias_id)] if alias_id is not None else None
    only_t = [int(lexicon_id)] if lexicon_id is not None else None
    out = {"scanned": 0, "changed": 0, "edits": 0, "dry_run": bool(dry_run)}
    if not any(a.kind == "mishear" and a.scope == "auto" for t in lex.active for a in t.aliases):
        return out
    last = 0
    while True:
        if stop_event is not None and stop_event.is_set():
            break
        rows = db.query("SELECT id, text FROM utterances WHERE id > ?" + (" AND day=?" if day else "") + " ORDER BY id LIMIT ?",
                        (last, day, batch) if day else (last, batch))
        if not rows:
            break
        last = int(rows[-1]["id"])
        out["scanned"] += len(rows)
        todo = []
        for r in rows:
            new_text, edits = lex.apply(r["text"] or "", only_alias_ids=only_a, only_lexicon_ids=only_t)
            if edits:
                todo.append((int(r["id"]), r["text"] or "", edits))
        if not todo:
            continue
        vetoed = revisions.undone_aliases(db, [uid for uid, _, _ in todo])
        for uid, text, edits in todo:
            skip = vetoed.get(uid)
            if skip:
                new_text, edits = lex.apply(text, only_alias_ids=only_a, only_lexicon_ids=only_t, skip_alias_ids=skip)
                if not edits:
                    continue
            else:
                new_text, _ = lex.apply(text, only_alias_ids=only_a, only_lexicon_ids=only_t)
            out["edits"] += len(edits)
            if dry_run:
                out["changed"] += 1
                continue
            d = {"edits": [{k: e[k] for k in ("from", "to", "reason", "alias_id", "lexicon_id", "start", "end")} for e in edits]}
            if revisions.update_text(db, cfg, uid, new_text, source="lexicon", detail=d, events=events, flush=False) is not None:
                out["changed"] += 1
                touch(db, lexicon_ids=[e["lexicon_id"] for e in edits], alias_ids=[e["alias_id"] for e in edits])
    if out["changed"] and not dry_run:
        revisions.flush_markdown(db, cfg)
    return out


def job_recorrect(rt: Any, *, day: str | None = None, lexicon_id: int | None = None, alias_id: int | None = None,
                  dry_run: bool = False) -> dict[str, Any]:
    """Runtime job `recorrect` (按词库重校历史; POST /api/jobs/recorrect {day?, lexicon_id?, alias_id?, dry_run?}):
    sync the sources, then recorrect(). Ids may arrive as text from a form."""
    def as_id(v: Any, name: str) -> int | None:
        if v in (None, ""):
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            raise ValueError(f"{name} 应为整数") from None

    sync_sources(rt.db, rt.cfg)
    return recorrect(rt.db, rt.cfg, day=day or None, lexicon_id=as_id(lexicon_id, "lexicon_id"),
                     alias_id=as_id(alias_id, "alias_id"), dry_run=bool(dry_run), events=getattr(rt, "events", None))


__all__ = ["ALIAS_KINDS", "ALIAS_SCOPES", "Alias", "DEFAULT_ALIAS_AUTO_MAX_FREQ", "Hit", "KIND_LABELS", "Lexicon", "NearHit",
           "TERM_KINDS", "TERM_STATUSES", "Term", "add_alias", "apply_alias_to_history", "auto_scope", "count_alias_matches",
           "counts", "delete_alias", "delete_term", "ensure_jieba", "export_glossary", "export_terms", "get_alias", "get_term",
           "glossary_text", "import_terms", "jieba_words", "job_recorrect", "list_terms", "recorrect", "reindex_spelling", "sync_sources",
           "sync_sources_throttled", "touch", "update_term", "upsert_term", "word_frequency"]
