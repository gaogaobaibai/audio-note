"""Speaker operations shared by the speakers / enroll / voices / utterances routers (not a router itself).

    audit(rt, sid)                    consistency audit of one speaker's samples
    speakers_overview(rt)             the rows of GET /api/speakers
    engine_or_none(rt)                the speaker engine, or None when its model is missing / broken
    assign_utterances(db, ...)        hand-label sentences (speaker_source='user')
    apply_self_relation(db, ...)      "this is me" / relation given together with a label
    after_enroll(rt, seg_id=None)     refresh the keyword index after a name appeared, re-identify one segment
    rewrite_markdown(rt, days, why)   regenerate the day files after speaker names changed
    refresh_scores(rt)                the reference voices changed: bring centroids + stored decisions in step
    notify(rt, change, **data)        publish a `speakers` event (the pages refresh from it)

These took `rt` from the enclosing closure in the old single-file app; now the caller passes the Runtime of its
request. `rt.engine` / `rt.client` and the speech / memory modules are looked up at call time (tests replace them).
"""
from __future__ import annotations

import logging
from typing import Any, Iterable

log = logging.getLogger(__name__)

#: an outlier sample that alone lets through at least this many automatic sentences is "risky" (GET /api/speakers)
RISKY_INFLUENCE = 20
#: up to this many sentences the history is re-decided inside the request (about 0.05 s per 1000); above, as a job
RESCORE_INLINE_MAX = 30000


def notify(rt: Any, change: str, **data: Any) -> None:
    """Publish a `speakers` event {change, ...}; a runtime stand-in without a bus is fine."""
    try:
        rt.events.publish("speakers", change=change, **data)
    except Exception as e:  # noqa: BLE001 - an event is a convenience
        log.debug("speakers event not published: %s", e)


def refresh_scores(rt: Any) -> dict[str, Any]:
    """Samples / negatives / labels changed: rebuild the centroids and re-decide the stored sentences from their
    stored vectors (speech.scoring.refresh; pure numpy, a no-op when nothing changed). A very long history is
    handed to the `rescore` job instead. Returns {} (nothing changed) | {"rescored": changed sentences} |
    {"rescore_started": True}; never raises."""
    try:
        from ...speech import scoring

        n = rt.db.one("SELECT COUNT(*) AS n FROM utterances")
        if n and int(n["n"]) > RESCORE_INLINE_MAX:
            try:
                rt.submit("rescore")
                return {"rescore_started": True}
            except Exception as e:  # noqa: BLE001 - already running / read-only: the chain step catches up later
                log.info("rescore not started: %s", e)
                return {}
        res = scoring.refresh(rt.db, rt.cfg, events=getattr(rt, "events", None), source="ui")
        changed = int((res.get("rescore") or {}).get("changed") or 0) if res.get("rescored") else 0
        if changed:
            return {"rescored": changed}
    except Exception as e:  # noqa: BLE001 - the request did its job; the chain step brings the rest in step
        log.warning("decisions not refreshed: %s", e)
    return {}


def audit(rt: Any, sid: int) -> list[dict[str, Any]]:
    from ...speech.speakers import audit_samples, sample_consistency_min

    return audit_samples(rt.db, int(sid), sample_consistency_min(rt.cfg), float(rt.cfg.speakers.identify_threshold))


def speakers_overview(rt: Any) -> list[dict[str, Any]]:
    db = rt.db
    rows = db.query(
        "SELECT s.id, s.name, s.notes, s.created_at, s.is_self, s.relation, s.relation_source, s.relation_note,"
        " COUNT(e.id) AS samples, COALESCE(SUM(e.duration_s),0) AS dur"
        " FROM speakers s LEFT JOIN speaker_embeddings e ON e.speaker_id=s.id GROUP BY s.id ORDER BY s.name")
    counts = {r["speaker_id"]: int(r["n"]) for r in db.query(
        "SELECT speaker_id, COUNT(*) AS n FROM utterances WHERE speaker_id IS NOT NULL GROUP BY speaker_id")}
    guesses = {int(r["g"]): int(r["n"]) for r in db.query(
        "SELECT speaker_guess_id AS g, COUNT(*) AS n FROM utterances WHERE speaker_id IS NULL AND speaker_guess_id IS NOT NULL"
        " GROUP BY speaker_guess_id")}
    cents: dict[int, list[str]] = {}
    for r in db.query("SELECT speaker_id, condition FROM speaker_centroids ORDER BY speaker_id, condition"):
        cents.setdefault(int(r["speaker_id"]), []).append(str(r["condition"]))
    negs = {int(r["speaker_id"]): int(r["n"]) for r in db.query(
        "SELECT speaker_id, COUNT(*) AS n FROM speaker_negatives GROUP BY speaker_id")}
    rejs = {int(r["speaker_id"]): int(r["n"]) for r in db.query(
        "SELECT speaker_id, COUNT(*) AS n FROM utterance_rejections GROUP BY speaker_id")}
    out = []
    for r in rows:
        a = audit(rt, int(r["id"])) if int(r["samples"]) > 1 else []
        out.append({"id": int(r["id"]), "name": r["name"], "notes": r["notes"] or "", "created_at": r["created_at"],
                    "samples": int(r["samples"]), "duration_s": round(float(r["dur"]), 1), "utterances": counts.get(r["id"], 0),
                    "is_self": bool(r["is_self"]), "relation": r["relation"], "relation_source": r["relation_source"],
                    "relation_note": r["relation_note"],
                    "outlier_samples": sum(1 for x in a if x["outlier"]),
                    "ambiguous_samples": sum(1 for x in a if x["ambiguous"]),
                    "negatives": negs.get(int(r["id"]), 0), "rejections": rejs.get(int(r["id"]), 0),
                    # v5: sentences that MAY be this person (review queue) and the conditions a centroid was learned for
                    "guesses": guesses.get(int(r["id"]), 0), "centroids": cents.get(int(r["id"]), []),
                    "risky_samples": sum(1 for x in a if x["outlier"] and int(x.get("influence") or 0) >= RISKY_INFLUENCE)})
    return out


def engine_or_none(rt: Any) -> Any:
    """The speaker engine, or None when its model files are missing (a veto still works without it)."""
    if rt.speaker_models_missing():
        return None
    try:
        return rt.engine()
    except Exception as e:  # noqa: BLE001 - e.g. a corrupt model file: veto only, no negative / positive sample
        log.warning("speaker engine unavailable (%s): veto only, no voice sample", e)
        return None


def assign_utterances(db: Any, sid: int, name: str, seg_id: int, ids: list[int], whole_cluster: bool) -> tuple[int, int]:
    """Hand-label utterances (speaker_source='user': relabel leaves them alone). With whole_cluster every
    utterance of the same segment that shares a cluster_label with one of `ids` is included - except the
    ones the user already labelled by hand as somebody else and the ones the user vetoed for this speaker
    (utterance_rejections), which are kept. The sentences in `ids` were picked explicitly: they are assigned
    and their veto for this speaker (if any) is lifted. Returns (assigned, skipped)."""
    if not ids:
        return 0, 0
    marks = ",".join("?" * len(ids))
    db.execute(f"DELETE FROM utterance_rejections WHERE speaker_id=? AND utterance_id IN ({marks})", [int(sid), *ids])
    where, params = f"id IN ({marks})", [*ids]
    skipped = 0
    if whole_cluster:
        labels = [r["cluster_label"] for r in db.query(f"SELECT DISTINCT cluster_label FROM utterances WHERE id IN ({marks})", ids)
                  if r["cluster_label"] is not None]
        if labels:
            lmarks = ",".join("?" * len(labels))
            guard = ("(speaker_source IS NULL OR speaker_source<>'user' OR speaker_id IS NULL OR speaker_id=?)"
                     " AND NOT EXISTS (SELECT 1 FROM utterance_rejections r WHERE r.utterance_id=utterances.id AND r.speaker_id=?)")
            other = db.one(f"SELECT COUNT(*) AS n FROM utterances WHERE segment_id=? AND cluster_label IN ({lmarks})"
                           f" AND id NOT IN ({marks}) AND NOT ({guard})", [int(seg_id), *labels, *ids, int(sid), int(sid)])
            skipped = int(other["n"]) if other else 0
            where += f" OR (segment_id=? AND cluster_label IN ({lmarks}) AND {guard})"
            params += [int(seg_id), *labels, int(sid), int(sid)]
    cur = db.execute(f"UPDATE utterances SET speaker_id=?, speaker_name=?, speaker_score=NULL, speaker_source='user'"
                     f" WHERE {where}", [int(sid), name, *params])
    return int(cur.rowcount or 0), skipped


def apply_self_relation(db: Any, sid: int, is_self: bool, relation: Any) -> None:
    if is_self:
        db.set_self_speaker(int(sid))
    if relation is not None and str(relation).strip():
        db.set_relation(int(sid), str(relation).strip(), "user")


def after_enroll(rt: Any, seg_id: int | None = None) -> dict[str, Any]:
    """Register the (new) name with jieba, refresh text_seg, re-identify one segment if given."""
    from ...memory import index as mem_index

    out: dict[str, Any] = {}
    try:
        mem_index.register_from(rt.cfg, rt.db)
        n = rt.db.one("SELECT COUNT(*) AS n FROM utterances")
        if n and n["n"]:
            out["reindexed"] = mem_index.reindex(rt.db)
    except Exception as e:  # noqa: BLE001
        log.warning("reindex after enroll failed: %s", e)
    if seg_id is not None:
        try:
            from ...speech.pipeline import Processor

            out["relabeled_changed"] = Processor(rt.cfg, rt.db, engine=rt.engine()).relabel_segment(seg_id)
        except Exception as e:  # noqa: BLE001
            log.warning("relabel after label failed: %s", e)
            out["relabel_error"] = str(e)
    out.update(refresh_scores(rt))          # a new sample / label is new evidence for everything already stored
    return out


def rewrite_markdown(rt: Any, days: Iterable[str], why: str) -> None:
    """Regenerate the Markdown transcript of each day (speaker names changed). Never fails the request."""
    try:
        from ...speech.pipeline import write_day_markdown

        for d in days:
            write_day_markdown(rt.db, rt.cfg, d)
    except Exception as e:  # noqa: BLE001
        log.warning("rewrite markdown after %s failed: %s", why, e)
