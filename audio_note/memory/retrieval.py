"""Retriever: one entry point for keyword (FTS5), vector (embeddings) and hybrid (RRF-fused) search,
with optional LLM query expansion. Used by the search_transcripts tool, the prefetch/rag chat
strategies and scripts/memory_bench.py.

Modes (memory.retrieval):
  fts     jieba + FTS5 bm25 (AND, then OR over content tokens)                  no model needed
  vector  cosine over sentence embeddings (memory.embedding_model)                 90 MB local model
  hybrid  keyword list + one vector list per sentence view, fused with reciprocal rank fusion
          (score = Σ w/(k+rank))                                                   default

Hybrid in detail:
  keyword list   memory.hybrid_keywords: `content` (index.keyword_search: content words of the question, names /
                 numbers / Latin tokens, lexicon aliases; a sentence needs two of the terms or a rare one) |
                 `strict` (every token must match) | `loose` (any content token) | `off`.
                 Weight memory.fts_weight; for a query that is only search words the keyword list leads.
  vector lists   one per view of memory.vector_views (plain / window / speaker, see memory.vectors)

search_question() adds the time a question names: "昨天晚上…" is searched inside yesterday 18:00-24:00 first
and globally second; a question that is nothing but a time ("昨晚聊了什么") browses that window in order.

Every mode falls back to a LIKE substring search when nothing matches, and vector/hybrid degrade
to fts when the embedding model cannot be loaded.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..db import DB
from . import index, temporal, vectors
from .embed import BaseEmbedder, get_embedder
from .expand import QueryExpander
from .index import Hit

log = logging.getLogger(__name__)

MODES = ("fts", "vector", "hybrid")
#: search_question: share of the result list reserved for hits inside the time window the question names
SCOPED_SHARE = 0.75
#: extra-view vectors a search computes on the spot; a bigger backlog (the first search after the upgrade, a view
#: switched on later) is filled by a background thread while searches use the views that are already there
SYNC_VIEW_BACKLOG = 300
KEYWORD_MODES = ("content", "strict", "loose", "off")
HOW = {"fts": "全文检索", "vector": "向量检索", "hybrid": "混合检索", "like": "子串匹配", "browse": "按时间浏览"}


@dataclass
class Retrieved:
    hits: list[Hit]
    how: str                      # key of HOW
    expansions: list[str] = field(default_factory=list)
    when: temporal.When | None = None      # the time the question named (search_question only)

    @property
    def how_label(self) -> str:
        return HOW.get(self.how, self.how)


def rrf_fuse(lists: list[list[Hit]], k: int = 60, weights: list[float] | None = None) -> list[Hit]:
    """Reciprocal rank fusion. Each list is ranked best-first; a hit's fused score is
    Σ w_i / (k + rank_i) over the lists it appears in (rank starting at 1). The first Hit object
    seen is kept, with score replaced by the fused score and source joined ('fts+vector')."""
    weights = list(weights) if weights else [1.0] * len(lists)
    score: dict[int, float] = {}
    keep: dict[int, Hit] = {}
    sources: dict[int, list[str]] = {}
    for w, hits in zip(weights, lists):
        for rank, h in enumerate(hits, start=1):
            score[h.id] = score.get(h.id, 0.0) + w / (k + rank)
            if h.id not in keep:
                keep[h.id] = h
                sources[h.id] = []
            if h.source not in sources[h.id]:
                sources[h.id].append(h.source)
    out: list[Hit] = []
    for uid, sc in sorted(score.items(), key=lambda kv: (-kv[1], kv[0])):
        h = keep[uid]
        h.score = sc
        h.source = "+".join(sources[uid])
        out.append(h)
    return out


class Retriever:
    def __init__(self, cfg, db: DB, client: Any = None, embedder: BaseEmbedder | None = None,
                 mode: str | None = None, expand: bool | None = None):
        self.cfg = cfg
        self.db = db
        self.mode = (mode or str(cfg.memory.retrieval or "fts")).lower()
        if self.mode not in MODES:
            log.warning("unknown memory.retrieval %r, using fts", self.mode)
            self.mode = "fts"
        self.rrf_k = max(1, int(cfg.memory.get("rrf_k", 60)))
        self.fts_weight = float(cfg.memory.get("fts_weight", 1.0))   # keyword list weight in hybrid fusion
        self.keywords = str(cfg.memory.get("hybrid_keywords", "content") or "content").lower()
        if self.keywords not in KEYWORD_MODES:
            log.warning("unknown memory.hybrid_keywords %r, using content", self.keywords)
            self.keywords = "content"
        if cfg.memory.get("hybrid_fts_strict") is False:
            self.keywords = "loose"       # the 0.3 switch, still honoured when an old config.yaml carries it
        self.views = vectors.normalise_views(cfg.memory.get("vector_views", vectors.DEFAULT_VIEWS))
        self._embedder = embedder
        self._embedder_failed = False
        self._backfill: threading.Thread | None = None
        want_expand = bool(cfg.memory.get("query_expansion", False)) if expand is None else bool(expand)
        self.expander: QueryExpander | None = None
        if want_expand and client is not None:
            self.expander = QueryExpander(client, cfg.memory.chat_model or cfg.mimo.chat_model)

    # --- embedding model ----------------------------------------------------------
    def embedder(self) -> BaseEmbedder | None:
        """The embedder, loaded on first use; None in fts mode or when the model cannot be loaded."""
        if self._embedder_failed or self.mode == "fts":
            return None
        if self._embedder is None:
            try:
                emb = get_embedder(self.cfg)
                emb.embed_query("预热")   # triggers the load (and the download on first use)
                self._embedder = emb
            except Exception as e:  # noqa: BLE001
                log.warning("embedding model unavailable (%s); falling back to keyword search", e)
                self._embedder_failed = True
                return None
        return self._embedder

    def loaded_embedder(self) -> BaseEmbedder | None:
        """The embedder only if it is already in memory (never triggers the model load)."""
        return self._embedder if not self._embedder_failed and self.mode != "fts" else None

    def ensure_indexed(self, limit: int | None = 5000) -> int:
        """Embed utterances that have no vector yet (old databases, pipeline runs with vectors off). The plain
        view is filled right here, as it always was. The extra views (memory.vector_views) too when little is
        missing or the caller asks for everything (`limit=None`); a large backlog - about a minute of CPU for
        2000 sentences - is left to a background thread, and until it is done the searches fuse what exists.
        Returns the number of utterances that received their plain vector."""
        emb = self.embedder()
        if emb is None:
            return 0
        try:
            plain_missing = vectors.missing_count(self.db, emb.name)
            done = vectors.index_missing(self.db, emb, limit=limit, views=[vectors.PLAIN]) if plain_missing else 0
            backlog = vectors.missing_count(self.db, emb.name, self.views) - vectors.missing_count(self.db, emb.name)
            if backlog > 0:
                if limit is None or backlog <= SYNC_VIEW_BACKLOG:
                    vectors.index_missing(self.db, emb, limit=limit, views=self.views)
                else:
                    self._backfill_views(emb)
            return done
        except Exception as e:  # noqa: BLE001
            log.warning("vector indexing failed: %s", e)
            return 0

    def _backfill_views(self, emb: BaseEmbedder) -> None:
        """Fill the missing extra views on a daemon thread (one at a time per Retriever)."""
        if self._backfill is not None and self._backfill.is_alive():
            return

        def work() -> None:
            try:
                vectors.index_missing(self.db, emb, views=self.views)
            except Exception as e:  # noqa: BLE001 - e.g. the database was closed by a profile switch
                log.warning("background vector indexing stopped: %s", e)

        self._backfill = threading.Thread(target=work, name="memory-view-index", daemon=True)
        self._backfill.start()

    # --- search -----------------------------------------------------------------------
    def _keyword_list(self, query: str, day_from, day_to, speaker, cand: int, hours) -> list[Hit]:
        """The keyword list of hybrid mode, per memory.hybrid_keywords."""
        if self.keywords == "off":
            return []
        if self.keywords == "content":
            return index.keyword_search(self.db, query, day_from, day_to, speaker, cand, hours)
        return index.search(self.db, query, day_from, day_to, speaker, cand, strict=self.keywords == "strict", hours=hours)

    def search(self, query: str, day_from: str | None = None, day_to: str | None = None,
               speaker: str | None = None, limit: int = 20, *, hours: tuple[float, float] | None = None) -> Retrieved:
        query = (query or "").strip()
        if not query:
            return Retrieved([], "fts")
        limit = max(1, int(limit))
        cand = 2 * limit
        expansions = self.expander.expand(query) if self.expander else []

        lists: list[list[Hit]] = []
        weights: list[float] = []
        how = self.mode
        if self.mode == "fts":
            lists.append(index.search(self.db, query, day_from, day_to, speaker, cand, hours=hours))
            weights.append(1.0)
        elif self.mode == "hybrid":
            kw = self._keyword_list(query, day_from, day_to, speaker, cand, hours)
            if kw or self.keywords != "off":
                lists.append(kw)
                # a query that is only search words (a name, a code): exact matches lead, the vector lists - one
                # per view, and nearly blind to a bare code - follow. Otherwise they would out-vote the one
                # sentence that actually contains the word (measured: found at rank 1 instead of not in 20).
                lookup = self.keywords == "content" and index.is_lookup(query)
                weights.append(max(self.fts_weight, float(len(self.views) + 1)) if lookup else self.fts_weight)
        if self.mode in ("fts", "hybrid") and expansions:
            w = self.fts_weight if self.mode == "hybrid" else 1.0
            lists.append(index.search_terms(self.db, [query, *expansions], day_from, day_to, speaker, cand))
            weights.append(0.7 * w)
        if self.mode in ("vector", "hybrid"):
            emb = self.embedder()
            if emb is not None:
                self.ensure_indexed()
                qv = emb.embed_query(query)
                for view in self.views:
                    lists.append(vectors.search(self.db, emb, query, day_from, day_to, speaker, cand, view=view,
                                                hours=hours, qv=qv))
                    weights.append(1.0)
                if expansions and self.mode == "vector":
                    lists.append(vectors.search(self.db, emb, " ".join([query, *expansions]), day_from, day_to, speaker, cand,
                                                hours=hours))
                    weights.append(0.7)
            else:
                how = "fts"
                if self.mode == "vector":      # model missing: behave like fts
                    lists.append(index.search(self.db, query, day_from, day_to, speaker, cand, hours=hours))
                    weights.append(1.0)
                elif self.keywords != "content" or not any(lists):
                    # hybrid without its vector half: the plain keyword search is the better stand-alone list
                    lists = [index.search(self.db, query, day_from, day_to, speaker, cand, hours=hours)]
                    weights = [1.0]

        lists_w = [(lst, w) for lst, w in zip(lists, weights) if lst]
        if len(lists_w) == 1:
            hits = lists_w[0][0][:limit]
        else:
            hits = rrf_fuse([lst for lst, _ in lists_w], self.rrf_k, [w for _, w in lists_w])[:limit]
        if not hits:
            hits = index.like_search(self.db, query, day_from, day_to, speaker, limit, hours)
            if hits:
                how = "like"
        return Retrieved(hits, how, expansions)

    def browse(self, when: temporal.When, limit: int = 20, speaker: str | None = None) -> list[Hit]:
        """Up to `limit` sentences of a time window in chronological order, spread evenly over the window
        (for a question that names a time and nothing else: there is nothing to rank by)."""
        fsql, fparams = index._filters(when.day_from, when.day_to, speaker, when.hours)
        ids = [int(r["id"]) for r in self.db.query(
            "SELECT u.id FROM utterances u WHERE 1=1" + fsql + " ORDER BY u.abs_start, u.id", fparams)]
        if not ids:
            return []
        limit = max(1, int(limit))
        if len(ids) > limit:
            step = (len(ids) - 1) / (limit - 1) if limit > 1 else 0
            ids = sorted({ids[round(i * step)] for i in range(limit)})
        rows = self.db.query(f"SELECT * FROM utterances WHERE id IN ({','.join('?' * len(ids))}) ORDER BY abs_start, id", ids)
        return [Hit.from_row(r, 0.0, source="browse") for r in rows]

    def search_question(self, question: str, now: datetime | None = None, limit: int = 20,
                        speaker: str | None = None) -> Retrieved:
        """search() for a natural question, aware of the time it names (temporal.parse_when):

        * no time in the question: the plain search;
        * a day range and / or a part of the day: the hits inside that window come first and take up to
          SCOPED_SHARE of the list, hits from outside fill the rest, so a misread date still finds the content;
        * a time and no content at all ("昨天晚上聊了什么"): the window itself, in order (how = "browse")."""
        limit = max(1, int(limit))
        when = temporal.parse_when(question, now or datetime.now())
        if when is None:
            return self.search(question, speaker=speaker, limit=limit)
        if not index.query_terms(question):
            hits = self.browse(when, limit, speaker)
            if hits:
                return Retrieved(hits, "browse", [], when)
        res = self.search(question, speaker=speaker, limit=limit)
        scoped = self.search(question, when.day_from, when.day_to, speaker, limit, hours=when.hours)
        if not scoped.hits:
            return Retrieved(list(res.hits), res.how, res.expansions, when)
        # outside the window, in order of trust: the same hours on any day (the day may be off by one: "这三天"
        # asked a day later), then anything. Fused, so a sentence both lists like comes first.
        rest_lists, rest_weights = [list(res.hits)], [1.0]
        if when.hours and (when.day_from or when.day_to):
            rest_lists.insert(0, self.search(question, speaker=speaker, limit=limit, hours=when.hours).hits)
            rest_weights.insert(0, 1.5)
        rest = rrf_fuse([lst for lst in rest_lists if lst], self.rrf_k, [w for lst, w in zip(rest_lists, rest_weights) if lst])
        # the window the question names gets most of the room (SCOPED_SHARE), the rest hedges against a
        # misread date; what the window cannot fill goes to the rest
        fsql, fparams = index._filters(when.day_from, when.day_to, speaker, when.hours)
        in_window = int(self.db.one("SELECT COUNT(*) AS n FROM utterances u WHERE 1=1" + fsql, fparams)["n"])
        # a window small enough to be shown whole is shown whole: "what did we talk about last night" is answered
        # by the evening itself, not by its best-matching three quarters
        room = limit if in_window <= limit else max(1, round(limit * SCOPED_SHARE))
        hits = scoped.hits[:room]
        seen = {h.id for h in hits}
        for h in rest:
            if len(hits) >= limit:
                break
            if h.id not in seen:
                seen.add(h.id)
                hits.append(h)
        for h in scoped.hits[room:]:
            if len(hits) >= limit:
                break
            if h.id not in seen:
                seen.add(h.id)
                hits.append(h)
        return Retrieved(hits, res.how, res.expansions, when)
