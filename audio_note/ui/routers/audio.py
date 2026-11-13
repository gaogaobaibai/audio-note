"""Audio clips for the page, and importing an existing recording.

    GET  /api/audio/utterance/{uid}?pad=0.3
    GET  /api/audio/segment/{seg_id}?from=&to=
    POST /api/import?filename=import.wav&start=<ISO>      (raw audio body)
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response

from ..deps import RT, to_thread
from ..serializers import wav_bytes

router = APIRouter(prefix="/api")


def segment_audio(rt: Any, seg_id: int, a: float, b: float) -> bytes:
    """WAV bytes of seconds a..b of a recorded segment (b <= a: until the end). 404 when the row or file is gone."""
    from ...audio.storage import read_segment
    from ...audio.vad import SAMPLE_RATE

    seg = rt.db.get_segment(seg_id)
    if seg is None:
        raise HTTPException(404, "录音段不存在")
    p = Path(seg["path"])
    if not p.exists():
        raise HTTPException(404, "音频文件已删除（超过保留期）")
    audio = read_segment(p)
    i, j = int(max(0.0, a) * SAMPLE_RATE), int(max(0.0, b) * SAMPLE_RATE)
    if j <= i:
        j = len(audio)
    return wav_bytes(audio[i:j])


@router.get("/audio/utterance/{uid}")
def audio_utterance(uid: int, rt: RT, pad: float = 0.3) -> Response:
    u = rt.db.one("SELECT segment_id, start_s, end_s FROM utterances WHERE id=?", (int(uid),))
    if u is None:
        raise HTTPException(404, "没有这条记录")
    pad = max(0.0, min(float(pad), 5.0))
    data = segment_audio(rt, int(u["segment_id"]), float(u["start_s"]) - pad, float(u["end_s"]) + pad)
    # utterance ids are reused by splits / re-processing: never let the browser play a stale clip
    return Response(data, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@router.get("/audio/segment/{seg_id}")
def audio_segment(seg_id: int, rt: RT, start: float = Query(0.0, alias="from"), end: float = Query(0.0, alias="to")) -> Response:
    data = segment_audio(rt, int(seg_id), float(start), float(end))
    return Response(data, media_type="audio/wav", headers={"Cache-Control": "private, max-age=3600"})


@router.post("/import")
async def import_audio(request: Request, rt: RT, filename: str = "import.wav", start: str | None = None) -> dict[str, Any]:
    data = await request.body()
    if len(data) < 1000:
        raise HTTPException(400, "没有收到音频数据")
    start_dt = None
    if start:
        try:
            start_dt = datetime.fromisoformat(start)
        except ValueError:
            raise HTTPException(400, "开始时间格式应为 YYYY-MM-DD HH:MM")
    safe = re.sub(r"[^\w.\-一-鿿]+", "_", Path(filename).name) or "import.wav"

    def work() -> dict[str, Any]:
        # everything blocking (file write, VAD over the whole file, database) runs off the event loop;
        # busy(): a profile switch waits for the import instead of closing the database under it
        from ...audio.importer import import_file
        from ...audio.vad import SileroVAD

        with rt.busy():
            d = rt.cfg.data_dir / "imports"
            d.mkdir(parents=True, exist_ok=True)
            dest = d / safe
            k = 0
            while dest.exists():
                k += 1
                dest = d / f"{Path(safe).stem}_{k}{Path(safe).suffix}"
            dest.write_bytes(data)
            vad = SileroVAD(rt.cfg.model_path("vad_model"), threads=2)
            ids = import_file(rt.cfg, rt.db, dest, start_time=start_dt, vad=vad)
            return {"ok": True, "file": dest.name, "segments": len(ids)}

    return await to_thread(work)
