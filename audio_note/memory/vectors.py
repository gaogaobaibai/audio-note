"""Vector store: embed what has no vector yet, cosine search with day / speaker / time-of-day filters.

Three kinds of things carry vectors:

    sentences   the plain text of every utterance                      -> utterance_vectors (one row per sentence)
                extra VIEWS of the same sentence                        -> memory_vectors(kind='utt', view=...)
    items       memory items (title + content)                          -> memory_vectors(kind='item')
    sessions    session summaries, once the episode step wrote them    -> memory_vectors(kind='session')

Views (memory.vector_views; "plain" always exists). A short sentence on its own often says too little to be
found ("那就这么定了"), so the same sentence is embedded again in other clothes and the result lists are fused:

    window    the sentence with the sentence before and after it (when they are close in time)
    speaker   "说话人：text" (optional: it goes stale when a voice is named later, until the next reindex)

Search runs in SQL through sqlite-vec (`vec_distance_cosine`) when the extension loaded, otherwise the candidate
vectors are pulled into numpy. Either way a sentence search returns index.Hit objects with score = cosine
similarity (higher is better) and source = "vector".
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

import numpy as np

from ..db import DB
from . import index
from .embed import BaseEmbedder
from .index import Hit

log = logging.getLogger(__name__)

PLAIN = "plain"
#: every view a sentence can be embedded as; "plain" lives in utterance_vectors, the others in memory_vectors
VIEWS = (PLAIN, "window", "speaker")
DEFAULT_VIEWS = (PLAIN, "window")
#: a neighbouring sentence belongs to the window when it starts within this many seconds
WINDOW_MAX_GAP_S = 120.0
#: characters of a neighbour that go into a window (the sentence itself is never cut)
WINDOW_NEIGHBOUR_CHARS = 120
#: characters of a memory item / session summary that are embedded
ITEM_TEXT_CHARS = 400


def normalise_views(value: Any) -> tuple[str, ...]:
    """memory.vector_views as a tuple of known views in canonical order; "plain" is always part of it."""
    if isinstance(value, str):
        value = [v for v in value.replace("+", ",").split(",")]
    wanted = {str(v).strip().lower() for v in (value or [])}
    return tuple(v for v in VIEWS if v == PLAIN or v in wanted)


def views_of(embedder: Any) -> tuple[str, ...]:
    """The views an embedder was configured to index (embed.get_embedder sets `.views` from the config);
    a bare embedder (tests, old callers) means the plain view only."""
    return normalise_views(getattr(embedder, "views", None))


def _norm_blob(v: np.ndarray) -> tuple[bytes, int]:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    if n > 0:
        v = v / n
    return v.tobytes(), int(v.shape[0])


def put(db: DB, kind: str, model: str, pairs: Iterable[tuple[int, np.ndarray]], view: str = "") -> int:
    """Upsert (ref_id, vector) pairs into memory_vectors (L2-normalised float32). Returns the number written."""
    n = 0
    with db.transaction() as c:
        for ref_id, v in pairs:
            blob, dim = _norm_blob(v)
            c.execute(
                "INSERT INTO memory_vectors(kind, ref_id, view, model, dim, vec) VALUES (?,?,?,?,?,?)"
                " ON CONFLICT(kind, ref_id, view) DO UPDATE SET model=excluded.model, dim=excluded.dim, vec=excluded.vec",
                (str(kind), int(ref_id), str(view), str(model), dim, blob))
            n += 1
    return n


# --- sentence views ----------------------------------------------------------------------------------
def _speaker(row: Any) -> str:
    return str(row["speaker_name"] or "未知")


def _secs(a: str, b: str) -> float:
    """Seconds between two abs_start texts (inf when one cannot be read)."""
    from datetime import datetime

    try:
        return abs((datetime.fromisoformat(str(b).replace(" ", "T")) - datetime.fromisoformat(str(a).replace(" ", "T")))
                   .total_seconds())
    except ValueError:
        return float("inf")


def _neighbours(db: DB, row: Any) -> tuple[Any | None, Any | None]:
    """The sentence right before and right after `row` in time, each only when it is within WINDOW_MAX_GAP_S."""
    ts, uid = row["abs_start"], int(row["id"])
    prev = db.one("SELECT id, abs_start, speaker_name, text FROM utterances WHERE abs_start < ? OR (abs_start = ? AND id < ?)"
                  " ORDER BY abs_start DESC, id DESC LIMIT 1", (ts, ts, uid))
    nxt = db.one("SELECT id, abs_start, speaker_name, text FROM utterances WHERE abs_start > ? OR (abs_start = ? AND id > ?)"
                 " ORDER BY abs_start, id LIMIT 1", (ts, ts, uid))
    if prev is not None and _secs(prev["abs_start"], ts) > WINDOW_MAX_GAP_S:
        prev = None
    if nxt is not None and _secs(ts, nxt["abs_start"]) > WINDOW_MAX_GAP_S:
        nxt = None
    return prev, nxt


def view_text(view: str, row: Any, prev: Any | None = None, nxt: Any | None = None) -> str:
    """What is embedded for one sentence in one view."""
    text = " ".join(str(row["text"] or "").split())
    if view == "speaker":
        return f"{_speaker(row)}：{text}"
    if view == "window":
        # no speaker names in here on purpose: they change when a voice gets a name, and on both benchmark sets
        # the window found the same evidence with and without them (35/43/49 of 51 questions at 5/10/20)
        parts = []
        if prev is not None:
            parts.append(" ".join(str(prev["text"] or "").split())[-WINDOW_NEIGHBOUR_CHARS:])
        parts.append(text)
        if nxt is not None:
            parts.append(" ".join(str(nxt["text"] or "").split())[:WINDOW_NEIGHBOUR_CHARS])
        return "\n".join(parts)
    return text


def _missing_view_rows(db: DB, model: str, view: str, limit: int) -> list[Any]:
    return db.query(
        "SELECT u.id, u.abs_start, u.speaker_name, u.text FROM utterances u LEFT JOIN memory_vectors v"
        " ON v.kind='utt' AND v.ref_id = u.id AND v.view = ? WHERE v.ref_id IS NULL OR v.model != ? ORDER BY u.id LIMIT ?",
        (view, model, int(limit)))


def _index_view(db: DB, embedder: BaseEmbedder, view: str, batch: int, limit: int | None) -> int:
    done = 0
    while True:
        take = batch if limit is None else min(batch, limit - done)
        if take <= 0:
            break
        rows = _missing_view_rows(db, embedder.name, view, take)
        if not rows:
            break
        if view == "window":
            texts = [view_text(view, r, *_neighbours(db, r)) for r in rows]
        else:
            texts = [view_text(view, r) for r in rows]
        put(db, "utt", embedder.name, zip((int(r["id"]) for r in rows), embedder.embed_passages(texts)), view=view)
        done += len(rows)
        if len(rows) < take:
            break
    return done


def _stale_windows(db: DB, model: str) -> int:
    """A sentence that is about to get its first plain vector is new (or its text was edited: revisions drop the
    vectors of an edited sentence). The windows of its neighbours contain its text, so they are computed again."""
    rows = db.utterances_without_vectors(model, 2000)
    if not rows or len(rows) >= 2000:          # a bulk (re)index: everything is rebuilt anyway
        return 0
    stale: set[int] = set()
    for r in rows:
        full = db.one("SELECT id, abs_start FROM utterances WHERE id=?", (int(r["id"]),))
        if full is None:
            continue
        for nb in _neighbours(db, full):
            if nb is not None:
                stale.add(int(nb["id"]))
    stale -= {int(r["id"]) for r in rows}
    if not stale:
        return 0
    ids = sorted(stale)
    db.execute(f"DELETE FROM memory_vectors WHERE kind='utt' AND view='window' AND ref_id IN ({','.join('?' * len(ids))})", ids)
    return len(ids)


def index_missing(db: DB, embedder: BaseEmbedder, batch: int = 256, limit: int | None = None,
                  views: Sequence[str] | None = None) -> int:
    """Embed every utterance that has no vector for embedder.name yet, in every wanted view (`views`, default:
    the views the embedder was configured with, see views_of). Returns the number of utterances that received
    their plain vector; the extra views are filled in the same call."""
    wanted = normalise_views(views) if views is not None else views_of(embedder)
    if "window" in wanted:
        try:
            _stale_windows(db, embedder.name)
        except Exception as e:  # noqa: BLE001 - a stale window only costs a little recall
            log.debug("window refresh skipped: %s", e)
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
    extra = {v: _index_view(db, embedder, v, batch, limit) for v in wanted if v != PLAIN}
    if getattr(embedder, "index_memory", False):
        # the profile's own embedder: whoever keeps the sentence vectors current (the pipeline after every
        # segment, the reindex job) keeps the item / session vectors current in the same breath
        try:
            extra["item"] = index_items(db, embedder)
            extra["session"] = index_sessions(db, embedder)
        except Exception as e:  # noqa: BLE001 - search fills them in later
            log.debug("memory item / session vectors skipped: %s", e)
    if done or any(extra.values()):
        log.info("embedded %d utterances with %s%s", done, embedder.name,
                 "".join(f", {n} {v} vectors" for v, n in extra.items() if n))
    return done


def missing_count(db: DB, model: str, views: Sequence[str] | None = None) -> int:
    """Vectors still to compute for `model`: utterances without a plain vector, plus (with `views`) the missing
    rows of every extra view."""
    row = db.one(
        "SELECT COUNT(*) AS n FROM utterances u LEFT JOIN utterance_vectors v ON v.utterance_id = u.id"
        " WHERE v.utterance_id IS NULL OR v.model != ?",
        (model,),
    )
    n = int(row["n"]) if row else 0
    for view in normalise_views(views) if views is not None else ():
        if view == PLAIN:
            continue
        r = db.one("SELECT COUNT(*) AS n FROM utterances u LEFT JOIN memory_vectors v ON v.kind='utt' AND v.ref_id = u.id"
                   " AND v.view = ? WHERE v.ref_id IS NULL OR v.model != ?", (view, model))
        n += int(r["n"]) if r else 0
    return n


def search(db: DB, embedder: BaseEmbedder, query: str, day_from: str | None = None, day_to: str | None = None,
           speaker: str | None = None, limit: int = 20, min_score: float | None = None, *, view: str = PLAIN,
           hours: tuple[float, float] | None = None, qv: np.ndarray | None = None) -> list[Hit]:
    """Sentences nearest to the query in one view. `qv` is the already embedded query (several views share it)."""
    q = (query or "").strip()
    if not q or limit <= 0:
        return []
    qv = np.asarray(embedder.embed_query(q) if qv is None else qv, dtype=np.float32)
    fsql, fparams = index._filters(day_from, day_to, speaker, hours)
    if view == PLAIN:
        table, join, cond, vparams = "utterance_vectors v", "u.id = v.utterance_id", "", []
    else:
        table, join, cond, vparams = "memory_vectors v", "u.id = v.ref_id", " AND v.kind='utt' AND v.view = ?", [view]
    source = "vector"          # whatever the view: after fusion a hit is "fts+vector", not a list of views
    if DB.vec_available:
        rows = db.query(
            f"SELECT u.*, vec_distance_cosine(v.vec, ?) AS dist FROM {table}"
            f" JOIN utterances u ON {join} WHERE v.model = ? AND v.dim = ?" + cond + fsql +
            " ORDER BY dist, u.abs_start LIMIT ?",
            [qv.tobytes(), embedder.name, int(qv.shape[0]), *vparams, *fparams, int(limit)],
        )
        hits = [Hit.from_row(r, 1.0 - float(r["dist"]), source=source) for r in rows]
        return hits if min_score is None else [h for h in hits if h.score >= min_score]
    # numpy fallback: scan the candidates
    rows = db.query(
        f"SELECT u.*, v.vec FROM {table} JOIN utterances u ON {join}"
        " WHERE v.model = ? AND v.dim = ?" + cond + fsql,
        [embedder.name, int(qv.shape[0]), *vparams, *fparams],
    )
    if not rows:
        return []
    mat = np.stack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
    sims = mat @ qv
    order = np.argsort(-sims, kind="stable")[:limit]
    return [Hit.from_row(rows[i], float(sims[i]), source=source) for i in order
            if min_score is None or sims[i] >= min_score]


# --- memory items and session summaries ---------------------------------------------------------------
def item_text(row: Any) -> str:
    """What is embedded for a memory item: title, then the content when it adds something."""
    title = " ".join(str(row["title"] or "").split())
    content = " ".join(str(row["content"] or "").split())
    text = title if not content or content == title else f"{title}：{content}"
    return text[:ITEM_TEXT_CHARS]


def index_items(db: DB, embedder: BaseEmbedder, batch: int = 256, limit: int | None = None) -> int:
    """Embed the memory items that have no vector for embedder.name (deleted and merged-away items never get one)."""
    done = 0
    while True:
        take = batch if limit is None else min(batch, limit - done)
        if take <= 0:
            break
        rows = db.query(
            "SELECT i.id, i.title, i.content FROM memory_items i LEFT JOIN memory_vectors v"
            " ON v.kind='item' AND v.ref_id = i.id AND v.view = ''"
            " WHERE i.status NOT IN ('rejected', 'superseded') AND (v.ref_id IS NULL OR v.model != ?) ORDER BY i.id LIMIT ?",
            (embedder.name, take))
        if not rows:
            break
        put(db, "item", embedder.name, zip((int(r["id"]) for r in rows), embedder.embed_passages([item_text(r) for r in rows])))
        done += len(rows)
        if len(rows) < take:
            break
    return done


def index_sessions(db: DB, embedder: BaseEmbedder, batch: int = 128) -> int:
    """Embed "title：summary" of the sessions that have a summary and no (current) vector."""
    done = 0
    while True:
        rows = db.query(
            "SELECT s.id, s.title, s.summary AS content FROM sessions s LEFT JOIN memory_vectors v"
            " ON v.kind='session' AND v.ref_id = s.id AND v.view = ''"
            " WHERE COALESCE(s.summary, '') <> '' AND (v.ref_id IS NULL OR v.model != ?) ORDER BY s.id LIMIT ?",
            (embedder.name, batch))
        if not rows:
            break
        put(db, "session", embedder.name,
            zip((int(r["id"]) for r in rows), embedder.embed_passages([item_text(r) for r in rows])))
        done += len(rows)
        if len(rows) < batch:
            break
    return done


def nearest(db: DB, embedder: BaseEmbedder, kind: str, query: str, limit: int = 20, *, ids: Iterable[int] | None = None,
            qv: np.ndarray | None = None) -> list[tuple[int, float]]:
    """[(ref_id, cosine)] of the `kind` ('item' | 'session') vectors nearest to the query, best first.
    `ids` restricts the candidates (the caller's own filters)."""
    q = (query or "").strip()
    if (not q and qv is None) or limit <= 0:
        return []
    qv = np.asarray(embedder.embed_query(q) if qv is None else qv, dtype=np.float32)
    rows = db.query("SELECT ref_id, vec FROM memory_vectors WHERE kind = ? AND view = '' AND model = ? AND dim = ?",
                    (str(kind), embedder.name, int(qv.shape[0])))
    if ids is not None:
        allowed = {int(i) for i in ids}
        rows = [r for r in rows if int(r["ref_id"]) in allowed]
    if not rows:
        return []
    sims = np.stack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows]) @ qv
    order = np.argsort(-sims, kind="stable")[:limit]
    return [(int(rows[i]["ref_id"]), float(sims[i])) for i in order]


def storage(db: DB) -> dict[str, dict[str, int]]:
    """Rows and vector bytes per kind / view: {"plain": {"rows", "bytes"}, "window": ..., "item": ..., "session": ...}."""
    out: dict[str, dict[str, int]] = {}
    r = db.one("SELECT COUNT(*) AS n, COALESCE(SUM(LENGTH(vec)), 0) AS b FROM utterance_vectors")
    out[PLAIN] = {"rows": int(r["n"]), "bytes": int(r["b"])}
    for row in db.query("SELECT kind, view, COUNT(*) AS n, COALESCE(SUM(LENGTH(vec)), 0) AS b FROM memory_vectors"
                        " GROUP BY kind, view"):
        key = str(row["view"]) if row["kind"] == "utt" else str(row["kind"])
        out[key] = {"rows": int(row["n"]), "bytes": int(row["b"])}
    return out


__all__ = ["PLAIN", "VIEWS", "DEFAULT_VIEWS", "normalise_views", "views_of", "put", "view_text", "index_missing",
           "missing_count", "search", "item_text", "index_items", "index_sessions", "nearest", "storage"]
