"""The two enrolment flows: a voice sample uploaded / recorded on the page, and "this sentence is X".

    POST /api/speakers/{name}/samples?filename=sample.webm&notes=&force=      (raw audio body)
    POST /api/speakers/label {name, utterance_id | (segment_id, from, to), whole_cluster?=true, force?, is_self?, relation?}

Both rewrite utterances (the keyword index, the labels of a segment), so both run inside deps.rewriting(rt).
Included before the speakers router: its static POST paths must be tried before any `/speakers/{...}` route.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request

from ..deps import RT, bool_arg, force_arg, force_kwargs, int_arg, rewriting, to_thread, value_error_http
from ._speaker_ops import after_enroll, apply_self_relation, assign_utterances, notify, rewrite_markdown

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

SAMPLE_EXTENSIONS = (".wav", ".webm", ".ogg", ".opus", ".mp3", ".m4a", ".flac", ".aac", ".mp4", ".wma", ".amr")


def _mismatch_of_upload(rt: Any, name: str, tmp: Path) -> HTTPException | None:
    """enroll() skips a mismatching file silently: re-check it so the page can offer "仍然加入"."""
    db = rt.db
    engine = rt.engine()
    try:
        from ...audio.importer import decode_any
        from ...audio.vad import SAMPLE_RATE
        from ...speech.speakers import MIN_ENROLL_SPEECH_S, SampleMismatch, sample_consistency_min

        row = db.one("SELECT id FROM speakers WHERE name=?", (name,))
        if row is None:
            return None
        speech = engine.speech_only(decode_any(tmp))
        if len(speech) / SAMPLE_RATE < MIN_ENROLL_SPEECH_S:
            return None
        score, ok = engine.check_sample(db, int(row["id"]), engine.embed(speech))
        if ok or score is None:
            return None
        return value_error_http(SampleMismatch(name, float(score), sample_consistency_min(rt.cfg)))
    except Exception as e:  # noqa: BLE001 - diagnostics only
        log.debug("mismatch re-check failed: %s", e)
        return None


def _enroll_upload(rt: Any, name: str, data: bytes, ext: str, notes: str | None, force: bool) -> dict[str, Any]:
    """Blocking part of POST /api/speakers/{name}/samples (runs on the thread pool)."""
    db = rt.db
    with rewriting(rt):
        tmp_dir = rt.cfg.data_dir / "uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = tmp_dir / f"enroll_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{ext}"
        existed = db.one("SELECT id FROM speakers WHERE name=?", (name,)) is not None
        try:
            tmp.write_bytes(data)
            before = db.one("SELECT COUNT(*) AS n FROM speaker_embeddings e JOIN speakers s ON s.id=e.speaker_id WHERE s.name=?", (name,))
            n0 = int(before["n"]) if before else 0
            sid = rt.engine().enroll(db, name, [tmp], notes, **force_kwargs(force))
            after = db.one("SELECT COUNT(*) AS n, COALESCE(MAX(duration_s),0) AS last FROM speaker_embeddings WHERE speaker_id=?", (sid,))
            n1, last = (int(after["n"]), float(after["last"])) if after else (0, 0.0)
            exc = None
            if n1 <= n0 and n0 > 0 and not force:
                exc = _mismatch_of_upload(rt, name, tmp)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
        if n1 <= n0:
            if exc is not None:
                raise exc
            if n1 == 0 and not existed:
                # a brand-new speaker row without samples is useless and confusing: remove it again
                # (a pre-existing speaker keeps its row: utterances still carry its id and name)
                db.execute("DELETE FROM speakers WHERE id=?", (sid,))
            raise HTTPException(400, "样本里检测到的语音太少（至少 1 秒，建议 10 秒以上），请重录")
        extra = after_enroll(rt)
    notify(rt, "enrolled", speaker_id=int(sid), name=name)
    return {"ok": True, "id": sid, "name": name, "samples": n1, "speech_s": round(last, 1), **extra}


@router.post("/speakers/{name}/samples")
async def speaker_add_sample(name: str, request: Request, rt: RT, filename: str = "sample.webm", notes: str | None = None,
                             force: str | None = None) -> dict[str, Any]:
    name = name.strip()
    if not name:
        raise HTTPException(400, "姓名不能为空")
    force_flag = force_arg(force)
    data = await request.body()
    if len(data) < 1000:
        raise HTTPException(400, "没有收到音频数据")
    ext = Path(filename).suffix.lower() or ".webm"
    if ext not in SAMPLE_EXTENSIONS:
        raise HTTPException(400, f"不支持的音频格式 {ext}")
    # async only to read the raw body; everything blocking (decode, CAM++, database) runs on the thread pool
    return await to_thread(_enroll_upload, rt, name, data, ext, notes, force_flag)


@router.post("/speakers/label")
def speaker_label(rt: RT, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    name = str(body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "姓名不能为空")
    db = rt.db
    whole_cluster = bool_arg(body.get("whole_cluster"), "whole_cluster", default=True)
    is_self = bool_arg(body.get("is_self"), "is_self")
    utt_ids: list[int]
    uid = int_arg(body.get("utterance_id"), "utterance_id")
    if uid is not None:
        u = db.one("SELECT id, segment_id, start_s, end_s FROM utterances WHERE id=?", (uid,))
        if u is None:
            raise HTTPException(404, "没有这条记录")
        seg_id, a, b = int(u["segment_id"]), float(u["start_s"]), float(u["end_s"])
        utt_ids = [int(u["id"])]
    else:
        try:
            seg_id, a, b = int(body["segment_id"]), float(body["from"]), float(body["to"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, "需要 utterance_id，或 segment_id + from + to")
        # the utterances the range covers (overlapping it) are the ones labelled by hand
        utt_ids = [int(r["id"]) for r in db.query(
            "SELECT id FROM utterances WHERE segment_id=? AND start_s < ? AND end_s > ?", (seg_id, b, a))]
    if b <= a:
        raise HTTPException(400, "结束时间必须大于开始时间")
    seg = db.get_segment(seg_id)
    if seg is None or not Path(seg["path"]).exists():
        raise HTTPException(404, "这段音频已不存在（超过保留期），无法取声纹")
    force_kw = force_kwargs(force_arg(body.get("force")))
    sample_note: str | None = None
    with rewriting(rt):
        try:
            from ...speech.speakers import TooShortSample, min_label_sample_s

            seg_path = Path(seg["path"])
            min_s = min_label_sample_s(rt.cfg)
            sid = rt.engine().enroll_from_segment_turns(db, name, seg_path, [(a, b)], min_speech_s=min_s, **force_kw)
        except TooShortSample as e:
            # the user said who spoke: assign the sentence anyway, just without a voice sample
            sample_note = str(e)
            sid = db.upsert_speaker(name)
        except ValueError as e:
            raise value_error_http(e)
        assigned, skipped = assign_utterances(db, sid, name, seg_id, utt_ids, whole_cluster)
        extra = after_enroll(rt, seg_id)       # relabel_segment leaves the hand-labelled ones alone
        apply_self_relation(db, sid, is_self, body.get("relation"))
        if assigned:
            rewrite_markdown(rt, [d["day"] for d in db.query(
                "SELECT DISTINCT day FROM utterances WHERE segment_id=? AND speaker_id=?", (seg_id, sid))], "label")
    n = db.one("SELECT COUNT(*) AS n FROM speaker_embeddings WHERE speaker_id=?", (sid,))
    notify(rt, "labelled", speaker_id=int(sid), name=name, segment_id=int(seg_id))
    return {"ok": True, "id": sid, "name": name, "samples": int(n["n"]) if n else 0, "segment_id": seg_id,
            "assigned": assigned, "skipped_user_labels": skipped, "sample_note": sample_note, **extra}
