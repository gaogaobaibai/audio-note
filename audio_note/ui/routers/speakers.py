"""Enrolled speakers: list, edit, delete, calibrate, their samples / negative samples, the review queue.

    GET    /api/speakers
    GET    /api/speakers/scoring                     how voices are being decided right now (v5 scoring layer)
    POST   /api/speakers/calibrate
    GET    /api/speakers/samples/{emb_id}/audio      DELETE /api/speakers/samples/{emb_id}
    GET    /api/speakers/negatives/{neg_id}/audio    DELETE /api/speakers/negatives/{neg_id}
    GET    /api/speakers/{sid}/samples               DELETE /api/speakers/{sid}/samples/outliers
    GET    /api/speakers/{sid}/negatives             GET    /api/speakers/{sid}/review?limit=20&offset=0
    PATCH  /api/speakers/{sid} {notes?, is_self?, relation?, relation_note?, name?}
    DELETE /api/speakers/{sid}

Static paths (`samples/...`, `negatives/...`, `calibrate`, `scoring`) are registered before the `{sid}` ones: a new
`GET /api/speakers/<word>` must go above them too, or `{sid}: int` swallows it and answers 422.
(POST /api/speakers/{name}/samples and POST /api/speakers/label live in the enroll router.)

Everything that changes the reference voices (a sample or a negative sample deleted, a speaker removed, "this is
me" moved) publishes a `speakers` event and brings the stored decisions in step (_speaker_ops.refresh_scores).
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Response

from ..deps import RT, bool_arg, require_speaker, rewriting, unknown_label
from ..serializers import wav_bytes
from ._speaker_ops import after_enroll, audit, notify, refresh_scores, rewrite_markdown, speakers_overview

router = APIRouter(prefix="/api")


def source_audio(src: str) -> bytes:
    """WAV bytes of a sample's source_path: a file, or "<file>#a-b" (a slice of a recording)."""
    from ...audio.importer import decode_any

    p = Path(src.split("#")[0])
    if not p.exists():
        raise HTTPException(404, "样本文件已不存在")
    audio = decode_any(p)
    if "#" in src:
        from ...audio.vad import SAMPLE_RATE

        try:
            a, b = src.split("#")[1].split("-")
            audio = audio[int(float(a) * SAMPLE_RATE):int(float(b) * SAMPLE_RATE)]
        except ValueError:
            pass
    return wav_bytes(audio)


@router.get("/speakers")
def speakers_list(rt: RT) -> dict[str, Any]:
    from ...analysis.relations import RELATION_CHOICES
    from ...audio.loudness import min_speech_dbfs
    from ...speech.voices import unknown_utterance_count

    me = rt.db.self_speaker()
    return {"speakers": speakers_overview(rt), "unknown_utterances": unknown_utterance_count(rt.db, min_speech_dbfs(rt.cfg)),
            "identify_threshold": float(rt.cfg.speakers.identify_threshold), "unknown_label": unknown_label(rt),
            "self_id": int(me["id"]) if me else None, "relation_choices": list(RELATION_CHOICES)}


@router.get("/speakers/scoring")
def speakers_scoring(rt: RT) -> dict[str, Any]:
    """State of the scoring layer: {scoring, version, normalised, cohort, thresholds: {sure, guess, unit},
    speakers: [{id, name, samples, outlier_samples, centroids: [{condition, n, speech_s, updated_at}]}],
    states: {sure, guess, unknown, user}, last_rescore: {version, at, summary} | null, stale: bool}."""
    from ...speech import scoring

    db = rt.db
    g = scoring.Gallery.load(db, rt.cfg)
    p = g.params
    cents: dict[int, list[dict[str, Any]]] = {}
    for r in db.query("SELECT speaker_id, condition, n, speech_s, updated_at FROM speaker_centroids ORDER BY speaker_id, condition"):
        cents.setdefault(int(r["speaker_id"]), []).append({"condition": r["condition"], "n": int(r["n"] or 0),
                                                           "speech_s": round(float(r["speech_s"] or 0.0), 1),
                                                           "updated_at": r["updated_at"]})
    st = db.one("SELECT SUM(speaker_source='user' AND speaker_id IS NOT NULL) AS user,"
                " SUM(NOT (speaker_source IS 'user' AND speaker_id IS NOT NULL) AND speaker_id IS NOT NULL) AS sure,"
                " SUM(speaker_id IS NULL AND speaker_guess_id IS NOT NULL) AS guess,"
                " SUM(speaker_id IS NULL AND speaker_guess_id IS NULL) AS unknown FROM utterances")
    last = db.get_kv(scoring.KV_RESCORE)
    return {"scoring": p.scoring, "version": g.version, "normalised": bool(g.normalised),
            "cohort": int(len(g.cohort)) if g.cohort is not None else 0, "cohort_min": int(p.cohort_min),
            "thresholds": ({"sure": p.sure_norm, "guess": p.guess_norm, "unit": "norm"} if g.normalised else
                           {"sure": p.identify_threshold, "guess": p.identify_threshold - p.guess_raw_margin, "unit": "cosine"}),
            "speakers": [{"id": s.sid, "name": s.name, "samples": int(len(s.every)), "outlier_samples": int(s.outliers),
                          "centroids": cents.get(s.sid, [])} for s in g.speakers],
            "states": {k: int(st[k] or 0) for k in ("sure", "guess", "unknown", "user")} if st else {},
            "last_rescore": last or None, "stale": bool(g.has_speakers and (last or {}).get("version") != g.version)}


def _calibrate_threshold_fn() -> Any:
    """calibrate_threshold from its home (speech.calibrate), or from the CLI on a tree where it still lives there."""
    try:
        from ...speech.calibrate import calibrate_threshold
    except ImportError:
        from ...cli import calibrate_threshold
    return calibrate_threshold


@router.post("/speakers/calibrate")
def speakers_calibrate(rt: RT) -> dict[str, Any]:
    embs = rt.db.speaker_embeddings()
    if len(embs) < 2:
        raise HTTPException(400, "至少需要 2 个注册样本才能校准（建议每人 2 个以上、至少 2 个人）")
    r = _calibrate_threshold_fn()(embs)
    r["current"] = float(rt.cfg.speakers.identify_threshold)
    return r


# ---------------------------------------------------------------------------------------------- samples
@router.get("/speakers/samples/{emb_id}/audio")
def speaker_sample_audio(emb_id: int, rt: RT) -> Response:
    r = rt.db.one("SELECT source_path FROM speaker_embeddings WHERE id=?", (int(emb_id),))
    if r is None or not r["source_path"]:
        raise HTTPException(404, "样本不存在")
    return Response(source_audio(str(r["source_path"])), media_type="audio/wav")


@router.delete("/speakers/samples/{emb_id}")
def speaker_sample_delete(emb_id: int, rt: RT) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    with rewriting(rt):
        cur = rt.db.execute("DELETE FROM speaker_embeddings WHERE id=?", (int(emb_id),))
        deleted = int(cur.rowcount or 0)
        if deleted:
            extra = refresh_scores(rt)         # what this sample alone let through is decided again
    if deleted:
        notify(rt, "sample_deleted", sample_id=int(emb_id))
    return {"ok": True, "deleted": deleted, **extra}


@router.get("/speakers/{sid}/samples")
def speaker_samples(sid: int, rt: RT) -> list[dict[str, Any]]:
    rows = rt.db.query("SELECT id, source_path, duration_s, created_at, origin FROM speaker_embeddings WHERE speaker_id=? ORDER BY id",
                       (int(sid),))
    audits = {a["id"]: a for a in audit(rt, int(sid))} if rows else {}
    out = []
    for r in rows:
        d = dict(r)
        d["file"] = Path(str(d["source_path"]).split("#")[0]).name if d["source_path"] else ""
        d["audio"] = bool(d["source_path"]) and Path(str(d["source_path"]).split("#")[0]).exists()
        a = audits.get(int(r["id"])) or {}
        d["consistency"] = a.get("consistency")
        d["outlier"] = bool(a.get("outlier", False))
        d["ambiguous"] = bool(a.get("ambiguous", False))
        d["influence"] = int(a.get("influence") or 0)
        out.append(d)
    return out


@router.delete("/speakers/{sid}/samples/outliers")
def speaker_delete_outlier_samples(sid: int, rt: RT) -> dict[str, Any]:
    """Remove every sample of the speaker that does not resemble the others (speakers.sample_consistency_min).
    The speaker row stays even when no sample is left."""
    db = rt.db
    require_speaker(db, sid)
    bad = [a["id"] for a in audit(rt, int(sid)) if a["outlier"]]
    deleted = 0
    extra: dict[str, Any] = {}
    if bad:
        with rewriting(rt):
            cur = db.execute(f"DELETE FROM speaker_embeddings WHERE speaker_id=? AND id IN ({','.join('?' * len(bad))})", [int(sid), *bad])
            deleted = int(cur.rowcount or 0)
            if deleted:
                extra = refresh_scores(rt)
    left = db.one("SELECT COUNT(*) AS n FROM speaker_embeddings WHERE speaker_id=?", (int(sid),))
    if deleted:
        notify(rt, "samples_deleted", speaker_id=int(sid), deleted=deleted)
    return {"ok": True, "deleted": deleted, "remaining": int(left["n"]) if left else 0, **extra}


# ---------------------------------------------------------------------------------------------- negatives / review
@router.delete("/speakers/negatives/{neg_id}")
def speaker_negative_delete(neg_id: int, rt: RT) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    with rewriting(rt):
        deleted = rt.db.delete_speaker_negative(int(neg_id))
        if deleted:
            extra = refresh_scores(rt)         # voices this "not X" kept away from X may be X again
    if deleted:
        notify(rt, "negative_deleted", negative_id=int(neg_id))
    return {"ok": True, "deleted": deleted, **extra}


@router.get("/speakers/negatives/{neg_id}/audio")
def speaker_negative_audio(neg_id: int, rt: RT) -> Response:
    r = rt.db.one("SELECT source_path FROM speaker_negatives WHERE id=?", (int(neg_id),))
    if r is None or not r["source_path"]:
        raise HTTPException(404, "负样本不存在")
    return Response(source_audio(str(r["source_path"])), media_type="audio/wav")


@router.get("/speakers/{sid}/negatives")
def speaker_negatives(sid: int, rt: RT) -> list[dict[str, Any]]:
    """Negative samples ("not this person") of a speaker, oldest first."""
    db = rt.db
    require_speaker(db, sid)
    rows = db.query("SELECT n.id, n.duration_s, n.source_path, n.utterance_id, n.segment_id, n.created_at, u.text"
                    " FROM speaker_negatives n LEFT JOIN utterances u ON u.id = n.utterance_id WHERE n.speaker_id=? ORDER BY n.id",
                    (int(sid),))
    out = []
    for r in rows:
        src = r["source_path"]
        f = Path(str(src).split("#")[0]) if src else None
        d: dict[str, Any] = {"id": int(r["id"]), "duration_s": r["duration_s"], "source_path": src,
                             "file": f.name if f else "", "audio": bool(f and f.exists()),
                             "utterance_id": r["utterance_id"], "segment_id": r["segment_id"], "created_at": r["created_at"]}
        if r["text"]:
            d["text"] = r["text"]
        out.append(d)
    return out


@router.get("/speakers/{sid}/review")
def speaker_review(sid: int, rt: RT, limit: int = 20, offset: int = 0) -> dict[str, Any]:
    """Review queue, asked from both sides: first the sentences that MAY be this speaker (kind "guess", the most
    influential first), then the ones automatically attributed to them, least certain first (kind "sure").
    {items: [{utterance_id, day, time, text, seconds, score, loudness_db, system_share, audio, kind, influence,
    guess_score}], total, guess, sure, limit, offset}. Answer with POST /api/utterances/{id}/confirm {speaker_id}
    or /reject {speaker_id}."""
    from ...speech import review

    require_speaker(rt.db, sid)
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    counts = review.review_counts(rt.db, rt.cfg, int(sid))
    return {"items": review.review_queue(rt.db, rt.cfg, int(sid), limit, offset),
            "total": counts["guess"] + counts["sure"], "guess": counts["guess"], "sure": counts["sure"],
            "limit": limit, "offset": offset}


# ---------------------------------------------------------------------------------------------- edit / delete
@router.patch("/speakers/{sid}")
def speaker_update(sid: int, rt: RT, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    db = rt.db
    row = require_speaker(db, sid, "id, name")
    is_self = bool_arg(body["is_self"], "is_self") if "is_self" in body else None
    new_name = str(body.get("name") or "").strip()
    renaming = bool(new_name) and new_name != row["name"]
    if renaming and db.one("SELECT id FROM speakers WHERE name=? AND id != ?", (new_name, int(sid))):
        raise HTTPException(400, f"已经有叫 {new_name} 的说话人")        # checked before anything is written
    renamed_days: list[str] = []
    # a rename rewrites utterances.speaker_name and the keyword index: through the gate; notes / relation
    # only touch the speakers row
    with rewriting(rt) if renaming else nullcontext():
        if "notes" in body:
            db.execute("UPDATE speakers SET notes=? WHERE id=?", (str(body["notes"] or "") or None, int(sid)))
        if is_self is not None:
            if is_self:
                db.set_self_speaker(int(sid))                       # the previous "self" (if any) is cleared
            else:
                me = db.self_speaker()
                if me is not None and int(me["id"]) == int(sid):
                    db.set_self_speaker(None)
        if "relation" in body:
            # "" clears relation / source / note and the inference cache (the next 自动推断关系 asks again);
            # otherwise it is the user's own label (an inferred one is replaced)
            rel = str(body.get("relation") or "").strip()
            note = str(body.get("relation_note") or "").strip() or None
            db.set_relation(int(sid), rel or None, "user" if rel else None, note)
        elif "relation_note" in body:
            db.execute("UPDATE speakers SET relation_note=? WHERE id=?", (str(body["relation_note"] or "").strip() or None, int(sid)))
        if renaming:
            db.execute("UPDATE speakers SET name=? WHERE id=?", (new_name, int(sid)))
            db.execute("UPDATE utterances SET speaker_name=? WHERE speaker_id=?", (new_name, int(sid)))
            renamed_days = [r["day"] for r in db.query("SELECT DISTINCT day FROM utterances WHERE speaker_id=?", (int(sid),))]
            after_enroll(rt)
            rewrite_markdown(rt, renamed_days, "rename")
    extra: dict[str, Any] = {}
    if is_self is not None:
        with rewriting(rt):                    # who "me" is decides what system audio may be
            extra = refresh_scores(rt)
    notify(rt, "updated", speaker_id=int(sid))
    return {"ok": True, "renamed_days": len(renamed_days), **extra}


@router.delete("/speakers/{sid}")
def speaker_delete(sid: int, rt: RT) -> dict[str, Any]:
    db = rt.db
    require_speaker(db, sid, "id, name")
    unk = unknown_label(rt)
    with rewriting(rt):
        days = [r["day"] for r in db.query("SELECT DISTINCT day FROM utterances WHERE speaker_id=?", (int(sid),))]
        # the utterances become unidentified again (no 'user' mark left behind, so relabel may re-identify
        # them) and the relation inference cache of the id is dropped
        db.delete_speaker(int(sid), unk)
        rewrite_markdown(rt, days, "delete")
        try:
            # nobody can be "maybe" a speaker who is gone
            db.execute("UPDATE utterances SET speaker_guess_id=NULL, speaker_guess_score=NULL WHERE speaker_guess_id=?", (int(sid),))
        except Exception:  # noqa: BLE001 - a database without the 1.0 columns
            pass
        extra = refresh_scores(rt)
    notify(rt, "deleted", speaker_id=int(sid))
    return {"ok": True, "unlabeled_days": len(days), **extra}
