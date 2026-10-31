"""Retriever: one entry point for keyword (FTS5), vector (embeddings) and hybrid (RRF-fused) search,
with optional LLM query expansion. Used by the search_transcripts tool, the prefetch/rag chat
strategies and scripts/memory_bench.py.

Modes (memory.retrieval):
  fts     jieba + FTS5 bm25 (AND, then OR over content tokens)                  no model needed
  vector  cosine over sentence embeddings (memory.embedding_model)                 90 MB local model
  hybrid  both lists fused with reciprocal rank fusion (score = Σ w/(k+rank))    default

Every mode falls back to a LIKE substring search when nothing matches, and vector/hybrid degrade
to fts when the embedding model cannot be loaded.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..db import DB
from . import index, vectors
from .embed import BaseEmbedder, get_embedder
from .expand import QueryExpander
from .index import Hit

log = logging.getLogger(__name__)

MODES = ("fts", "vector", "hybrid")
HOW = {"fts": "全文检索", "vector": "向量检索", "hybrid": "混合检索", "like": "子串匹配"}


@dataclass
class Retrieved:
    hits: list[Hit]
    how: str                      # key of HOW
    expansions: list[str] = field(default_factory=list)

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
        self._embedder = embedder
        self._embedder_failed = False
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

    def ensure_indexed(self, limit: int | None = 5000) -> int:
        """Embed utterances that have no vector yet (old databases, pipeline runs with vectors off)."""
        emb = self.embedder()
        if emb is None:
            return 0
        try:
            if vectors.missing_count(self.db, emb.name) == 0:
                return 0
            return vectors.index_missing(self.db, emb, limit=limit)
        except Exception as e:  # noqa: BLE001
            log.warning("vector indexing failed: %s", e)
            return 0

    # --- search -----------------------------------------------------------------------
    def search(self, query: str, day_from: str | None = None, day_to: str | None = None,
               speaker: str | None = None, limit: int = 20) -> Retrieved:
        query = (query or "").strip()
        if not query:
            return Retrieved([], "fts")
        limit = max(1, int(limit))
        cand = 2 * limit
        expansions = self.expander.expand(query) if self.expander else []

        lists: list[list[Hit]] = []
        weights: list[float] = []
        how = self.mode
        if self.mode in ("fts", "hybrid"):
            w = self.fts_weight if self.mode == "hybrid" else 1.0
            strict = self.mode == "hybrid" and bool(self.cfg.memory.get("hybrid_fts_strict", True))
            lists.append(index.search(self.db, query, day_from, day_to, speaker, cand, strict=strict))
            weights.append(w)
            if expansions:
                lists.append(index.search_terms(self.db, [query, *expansions], day_from, day_to, speaker, cand))
                weights.append(0.7 * w)
        if self.mode in ("vector", "hybrid"):
            emb = self.embedder()
            if emb is not None:
                self.ensure_indexed()
                lists.append(vectors.search(self.db, emb, query, day_from, day_to, speaker, cand))
                weights.append(1.0)
                if expansions and self.mode == "vector":
                    lists.append(vectors.search(self.db, emb, " ".join([query, *expansions]), day_from, day_to, speaker, cand))
                    weights.append(0.7)
            else:
                how = "fts"
                if self.mode == "vector":      # model missing: behave like fts
                    lists.append(index.search(self.db, query, day_from, day_to, speaker, cand))
                    weights.append(1.0)

        hits = lists[0][:limit] if len(lists) == 1 else rrf_fuse(lists, self.rrf_k, weights)[:limit]
        if not hits:
            hits = index.like_search(self.db, query, day_from, day_to, speaker, limit)
            if hits:
                how = "like"
        return Retrieved(hits, how, expansions)
