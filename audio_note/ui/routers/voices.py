"""Not-yet-identified voices, grouped across recordings: list them, give a group a name, or say who it is not.

    GET  /api/voices
    POST /api/voices/name   {members: [[segment_id, cluster_label], ...], name, is_self?, relation?, auto_relabel?=true, force?}
    POST /api/voices/reject {members: [[segment_id, cluster_label], ...], speaker_id}
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from ...runtime import VOICE_MODELS_MISSING_MSG, JobError, ReadOnly
from ..deps import RT, bool_arg, force_arg, require_speaker, rewriting, speaker_id_arg, value_error_http
from ._speaker_ops import after_enroll, engine_or_none

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@router.get("/voices")
def voices_list(rt: RT) -> dict[str, Any]:
    """Not-yet-identified voices grouped across recordings (see speech.voices.group_unknown_voices)."""
    from ...speech import voices

    out = voices.group_unknown_voices(rt.db, rt.cfg)
    out["scanning"] = bool(rt.jobs["voices"].running)
    return out


@router.post("/voices/name")
def voices_name(rt: RT, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Give a voice group a name: {members: [[segment_id, cluster_label], ...], name, is_self?, relation?,
    auto_relabel?: true}. Enrolls samples from the group, assigns its utterances (speaker_source='user'),
    then re-identifies the other recordings in the background when unidentified utterances remain."""
    from ...speech import voices          # looked up per call: tests replace voices.name_voice

    name = str(body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "姓名不能为空")
    members = body.get("members")
    if not isinstance(members, list) or not members:
        raise HTTPException(400, "members 不能为空")
    is_self = bool_arg(body.get("is_self"), "is_self")
    relation = body.get("relation")
    relation = str(relation).strip() if relation is not None else None
    force = force_arg(body.get("force"))
    auto_relabel = bool_arg(body.get("auto_relabel"), "auto_relabel", default=True)
    if rt.speaker_models_missing():
        raise HTTPException(400, VOICE_MODELS_MISSING_MSG)
    db = rt.db
    with rewriting(rt):                  # a profile switch waits (JobError -> 409) instead of closing the DB under us
        try:
            res = voices.name_voice(db, rt.cfg, rt.engine(), members, name,
                                    is_self=is_self, relation=relation or None, force=force)
        except ValueError as e:
            raise value_error_http(e)
        extra = after_enroll(rt)
    relabel_started = False
    if auto_relabel:
        left = db.one("SELECT COUNT(*) AS n FROM utterances WHERE speaker_id IS NULL")
        if left and int(left["n"]) and db.one("SELECT 1 FROM speaker_embeddings LIMIT 1"):
            try:
                rt.submit("relabel")
                relabel_started = True
            except (JobError, ReadOnly) as e:
                log.info("relabel after naming a voice not started: %s", e)
    return {"ok": True, **res, "relabel_started": relabel_started, **extra}


@router.post("/voices/reject")
def voices_reject(rt: RT, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """"Not X" on a voice group: {members: [[segment_id, cluster_label], ...], speaker_id}."""
    from ...speech import voices

    members = body.get("members")
    if not isinstance(members, list) or not members:
        raise HTTPException(400, "members 不能为空")
    sid = speaker_id_arg(body.get("speaker_id"))
    if sid is None:
        raise HTTPException(400, "需要 speaker_id")
    db = rt.db
    require_speaker(db, sid)
    with rewriting(rt):
        engine = engine_or_none(rt)
        try:
            res = voices.reject_voice_group(db, rt.cfg, engine, members, sid)
        except LookupError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
    return {"ok": True, **res}
