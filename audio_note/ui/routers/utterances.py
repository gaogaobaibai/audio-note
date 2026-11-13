"""Acting on one sentence: "not this person", "yes, this person", splitting a sentence two people share, and
correcting its text.

    POST  /api/utterances/{uid}/reject  {speaker_id?, whole_cluster?=false}
    POST  /api/utterances/{uid}/confirm {speaker_id, whole_cluster?=false}
    GET   /api/utterances/{uid}/preview-split?auto=true&at=1.8,4.2
    POST  /api/utterances/{uid}/split   {auto?: bool, at?: [seconds, ...]}
    PATCH /api/utterances/{uid}         {text, remember?=false}   -> {ok, utterance, changed, revision_id, learned}
    GET   /api/utterances/{uid}/revisions                         -> {id, text, text_raw, text_source, revisions}
    POST  /api/utterances/{uid}/undo    {revision_id?}            -> {ok, utterance, undone, revision_id}

Everything that changes rows runs inside deps.rewriting(rt); the split additionally holds the scheduler lock.
Text changes go through speech.revisions (one revision row per change, undoable); a hand edit may also teach the
lexicon a rule (`learned`, see revisions.learn_from_edit).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from ...runtime import VOICE_MODELS_MISSING_MSG
from ..deps import RT, bool_arg, int_arg, require_speaker, require_utterance, rewriting, speaker_id_arg, unknown_label
from ._speaker_ops import after_enroll, engine_or_none

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------------------------- review
@router.post("/utterances/{uid}/reject")
def utterance_reject(uid: int, rt: RT, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """"Not this person": {speaker_id?: the sentence's current speaker by default, whole_cluster?: false}. Vetoes
    the speaker for the sentence(s), takes negative samples, re-identifies the segment."""
    from ...speech import voices

    db = rt.db
    u = require_utterance(db, uid, "id, speaker_id")
    sid = speaker_id_arg(body.get("speaker_id"))
    if sid is None:
        sid = int(u["speaker_id"]) if u["speaker_id"] is not None else None
    if sid is None:
        raise HTTPException(400, "这句还没有说话人：请指定 speaker_id")
    require_speaker(db, sid)
    whole = bool_arg(body.get("whole_cluster"), "whole_cluster")
    with rewriting(rt):
        engine = engine_or_none(rt)
        try:
            res = voices.reject_voice(db, rt.cfg, engine, sid, [int(uid)], whole_cluster=whole)
        except LookupError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
    return {"ok": True, **res}


@router.post("/utterances/{uid}/confirm")
def utterance_confirm(uid: int, rt: RT, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """"Yes, this person": {speaker_id, whole_cluster?: false}. Marks the sentence(s) as the user's own label and
    adds a voice sample when the sentence is long enough and resembles the person (sample_added: false otherwise)."""
    from ...speech import review

    db = rt.db
    require_utterance(db, uid)
    sid = speaker_id_arg(body.get("speaker_id"))
    if sid is None:
        raise HTTPException(400, "需要 speaker_id")
    require_speaker(db, sid)
    whole = bool_arg(body.get("whole_cluster"), "whole_cluster")
    with rewriting(rt):
        engine = engine_or_none(rt)
        try:
            res = review.confirm_utterance(db, rt.cfg, engine, int(uid), sid, whole_cluster=whole)
        except LookupError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        if res.get("sample_added"):
            res.update(after_enroll(rt, int(res["segment_id"])))
    return {"ok": True, **res}


# ---------------------------------------------------------------------------------------------- mixed utterances
def _split_args(body: dict[str, Any]) -> tuple[list[float] | None, bool]:
    auto = bool_arg(body.get("auto"), "auto")
    at = body.get("at")
    if at is None or at == "":
        return None, auto
    if not isinstance(at, list):
        raise HTTPException(400, "at 应为秒数列表")
    pts: list[float] = []
    for x in at:
        if isinstance(x, bool) or not isinstance(x, (int, float, str)):
            raise HTTPException(400, "at 应为秒数列表")
        try:
            pts.append(float(x))
        except ValueError:
            raise HTTPException(400, "at 应为秒数列表")
    if not pts and not auto:
        raise HTTPException(400, "需要给出拆分点（at）或选择自动拆分（auto）")
    return (pts or None), auto


def _split_processor(rt: Any, with_asr: bool) -> Any:
    from ...speech.pipeline import Processor

    if rt.speaker_models_missing():
        raise HTTPException(400, VOICE_MODELS_MISSING_MSG)
    client = rt.client() if with_asr else None          # JobError (409) without an API key, like /api/chat
    return Processor(rt.cfg, rt.db, client=client, engine=rt.engine())


@router.get("/utterances/{uid}/preview-split")
def utterance_preview_split(uid: int, rt: RT, auto: bool = True, at: str | None = None) -> dict[str, Any]:
    """Pieces an auto split would make (bounds + who was recognised in each), no ASR and no DB change.
    `at=1.8,4.2` previews a manual split instead."""
    require_utterance(rt.db, uid)
    pts: list[float] | None = None
    if at:
        try:
            pts = [float(x) for x in str(at).replace("，", ",").split(",") if x.strip()]
        except ValueError:
            raise HTTPException(400, "at 应为秒数列表")
    proc = _split_processor(rt, with_asr=False)
    with rt.busy():                      # changes nothing: no rewriting() gate, but a profile switch still waits
        try:
            return proc.preview_split(int(uid), at=pts, auto=auto if pts is None else False)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))


@router.post("/utterances/{uid}/split")
def utterance_split(uid: int, rt: RT, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Split a mixed utterance: {"auto": true} (by voice) or {"at": [1.8, 4.2]} (seconds from its start).
    The pieces are transcribed again (needs the API key: 409 without one) and replace the row."""
    from ...llm.client import LLMError

    require_utterance(rt.db, uid)
    at, auto = _split_args(body)
    if not auto and not at:
        raise HTTPException(400, "需要给出拆分点（at）或选择自动拆分（auto）")
    proc = _split_processor(rt, with_asr=True)
    sched_lock = rt.sched.lock
    with rewriting(rt):
        try:
            with sched_lock:                 # never interleave with a running job (process / relabel) on the same rows
                res = proc.split_utterance(int(uid), at=at, auto=auto)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        except LLMError as e:
            raise HTTPException(502, f"重新转写失败，没有修改：{e}")
    return {"ok": True, **res}


# ---------------------------------------------------------------------------------------------- text
def _utterance_json(rt: Any, uid: int) -> dict[str, Any]:
    """The sentence as the transcript shows it (with its marks and whether its audio still exists)."""
    from ..serializers import utt_dicts

    db = rt.db
    row = require_utterance(db, uid, "*")

    def audio_ok(seg_id: int) -> bool:
        seg = db.get_segment(seg_id)
        return bool(seg and Path(seg["path"]).exists())

    return utt_dicts(db, [row], unknown_label(rt), audio_ok)[0]


@router.patch("/utterances/{uid}")
def utterance_edit(uid: int, rt: RT, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Correct the text by hand: {text, remember?: false}.

    The change is a revision with source 'user' (undo: POST .../undo). When the edit replaces one word by a
    same-sounding name / term - or `remember` is true - the lexicon learns it: `learned` = {lexicon_id,
    alias_id, term, alias, scope, created_term, other_matches, inbox_id} (null otherwise). From then on the
    wrong spelling is fixed automatically in new recordings (scope 'auto'); other_matches > 0 means older
    sentences contain it too: the page may ask "change those as well?" and call POST
    /api/lexicon/aliases/{alias_id}/apply, or leave the waiting inbox question (inbox_id) for later.
    Typing an automatic change back exactly is taken as an undo of that change."""
    from ...speech import revisions

    text = body.get("text")
    if not isinstance(text, str):
        raise HTTPException(400, "text 应为文字")
    remember = bool_arg(body.get("remember"), "remember")
    db = rt.db
    row = require_utterance(db, uid, "id, text")
    old = row["text"] or ""
    new = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    learned = None
    res: dict[str, Any] | None = None
    with rewriting(rt):
        try:
            last = db.one("SELECT id, old_text, source FROM utterance_revisions WHERE utterance_id=? AND undone_at IS NULL"
                          " AND source<>'undo' ORDER BY id DESC LIMIT 1", (int(uid),))
            if last is not None and new != old and new == (last["old_text"] or "") and last["source"] != "user":
                # the user typed an automatic change away: record it as taking that change back, so the rule
                # that made it leaves this sentence alone from now on
                out = revisions.undo(db, rt.cfg, int(uid), int(last["id"]), events=rt.events)
                res = {"revision_id": out["revision_id"]}
            else:
                res = revisions.update_text(db, rt.cfg, int(uid), new, source="user",
                                            detail={"edits": [{k: e[k] for k in ("from", "to", "start", "end")}
                                                              for e in revisions.diff_edits(old, new)]},
                                            events=rt.events)
                if res is not None:
                    learned = revisions.learn_from_edit(db, rt.cfg, int(uid), old, new, remember=remember, events=rt.events)
        except LookupError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
    return {"ok": True, "utterance": _utterance_json(rt, int(uid)), "changed": res is not None,
            "revision_id": res["revision_id"] if res else None, "learned": learned}


@router.get("/utterances/{uid}/revisions")
def utterance_revisions(uid: int, rt: RT) -> dict[str, Any]:
    """The history of a sentence's text, newest first: revisions = [{id, old_text, new_text, source, detail,
    created_at, undone_at}]; source: lexicon | llm | refine | answer | user | guard | undo."""
    from ...speech import revisions

    row = require_utterance(rt.db, uid, "id, text, text_raw, text_source, edited_at")
    return {"id": int(uid), "text": row["text"], "text_raw": row["text_raw"], "text_source": row["text_source"],
            "edited_at": row["edited_at"], "revisions": revisions.history(rt.db, int(uid))}


@router.post("/utterances/{uid}/undo")
def utterance_undo(uid: int, rt: RT, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Take back the latest change of the text still in effect, or the one named by {revision_id}."""
    from ...speech import revisions

    require_utterance(rt.db, uid)
    rev = int_arg(body.get("revision_id"), "revision_id", 1)
    with rewriting(rt):
        try:
            out = revisions.undo(rt.db, rt.cfg, int(uid), rev, events=rt.events)
        except LookupError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
    return {"ok": True, "utterance": _utterance_json(rt, int(uid)), "undone": out["undone"], "revision_id": out["revision_id"]}
