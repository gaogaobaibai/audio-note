"""Vector store over `utterance_vectors`: embed missing utterances, cosine search with day/speaker filters.

Search runs in SQL through sqlite-vec (`vec_distance_cosine`) when the extension loaded, otherwise the
candidate vectors are pulled into numpy. Either way the result is a list of index.Hit with
score = cosine similarity (higher is better) and source = "vector".
"""
from __future__ import annotations

import logging

import numpy as np

from ..db import DB
from . import index
from .embed import BaseEmbedder
from .index import Hit

log = logging.getLogger(__name__)


def index_missing(db: DB, embedder: BaseEmbedder, batch: int = 256, limit: int | None = None) -> int:
    """Embed every utterance that has no vector for embedder.name yet. Returns the number embedded."""
    done = 0
    while True:
        take = batch if limit is None else min(batch, limit - done)
        if take <= 0:
            break
        rows = db.utterances_without_vectors(embedder.name, take)
        if not rows:
            break
        vecs = embedder.embed_passages([r["text"] or "" for r in rows])
        db.add_vectors(embedder.name, zip((int(r["id"]) for r in rows), vecs))
        done += len(rows)
        if len(rows) < take:
            break
    if done:
        log.info("embedded %d utterances with %s", done, embedder.name)
    return done


def missing_count(db: DB, model: str) -> int:
    row = db.one(
        "SELECT COUNT(*) AS n FROM utterances u LEFT JOIN utterance_vectors v ON v.utterance_id = u.id"
        " WHERE v.utterance_id IS NULL OR v.model != ?",
        (model,),
    )
    return int(row["n"]) if row else 0


def search(db: DB, embedder: BaseEmbedder, query: str, day_from: str | None = None, day_to: str | None = None,
           speaker: str | None = None, limit: int = 20, min_score: float | None = None) -> list[Hit]:
    q = (query or "").strip()
    if not q or limit <= 0:
        return []
    qv = np.asarray(embedder.embed_query(q), dtype=np.float32)
    fsql, fparams = index._filters(day_from, day_to, speaker)
    if DB.vec_available:
        rows = db.query(
            "SELECT u.*, vec_distance_cosine(v.vec, ?) AS dist FROM utterance_vectors v"
            " JOIN utterances u ON u.id = v.utterance_id WHERE v.model = ? AND v.dim = ?" + fsql +
            " ORDER BY dist, u.abs_start LIMIT ?",
            [qv.tobytes(), embedder.name, int(qv.shape[0]), *fparams, int(limit)],
        )
        hits = [Hit.from_row(r, 1.0 - float(r["dist"]), source="vector") for r in rows]
        return hits if min_score is None else [h for h in hits if h.score >= min_score]
    # numpy fallback: scan the candidates
    rows = db.query(
        "SELECT u.*, v.vec FROM utterance_vectors v JOIN utterances u ON u.id = v.utterance_id"
        " WHERE v.model = ? AND v.dim = ?" + fsql,
        [embedder.name, int(qv.shape[0]), *fparams],
    )
    if not rows:
        return []
    mat = np.stack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
    sims = mat @ qv
    order = np.argsort(-sims, kind="stable")[:limit]
    return [Hit.from_row(rows[i], float(sims[i]), source="vector") for i in order
            if min_score is None or sims[i] >= min_score]
