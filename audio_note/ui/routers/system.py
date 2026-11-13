"""Status, recording / service control, background jobs, log tail.

    GET  /api/status
    POST /api/recording/start | /api/recording/stop
    POST /api/service/start {record?} | /api/service/stop
    GET  /api/jobs                 POST /api/jobs/{name} {...job arguments}
    GET  /api/logs?lines=200&which=service|cli
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from ..deps import RT, bool_arg, day_arg, int_arg, speaker_id_arg

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------------------------- status / service
@router.get("/status")
def status(rt: RT) -> dict[str, Any]:
    return rt.status()


@router.post("/recording/start")
def recording_start(rt: RT) -> dict[str, Any]:
    if rt.status()["models_missing"]:
        raise HTTPException(400, "模型文件缺失，无法录音（运行 scripts/download_models.ps1）")
    if not rt.service_running():
        rt.start_service(record=True)
    else:
        rt.start_recording()
    time.sleep(0.3)   # let the thread hit the mic so a device error shows up in this response
    st = rt.status()["service"]
    return {"ok": True, "recording": st["recording"], "last_error": st["last_error"]}


@router.post("/recording/stop")
def recording_stop(rt: RT) -> dict[str, Any]:
    rt.stop_recording()
    return {"ok": True, "recording": rt.recording()}


@router.post("/service/start")
def service_start(rt: RT, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    rt.start_service(record=bool_arg(body.get("record"), "record"))
    return {"ok": True, "service": rt.status()["service"]}


@router.post("/service/stop")
def service_stop(rt: RT) -> dict[str, Any]:
    rt.stop_service()
    return {"ok": True, "service": rt.status()["service"]}


# ---------------------------------------------------------------------------------------------- jobs
def job_names(rt: Any) -> set[str]:
    """Job names the runtime knows at this moment: runtime.JOBS (read at call time, other packages extend it)
    plus whatever this Runtime instance registered."""
    from ... import runtime as runtime_mod

    return set(runtime_mod.JOBS) | set(getattr(rt, "jobs", None) or ())


def job_kwargs(name: str, body: dict[str, Any]) -> dict[str, Any]:
    """Validated keyword arguments for Runtime.submit(name, **kwargs).

    The original jobs take exactly the arguments below (unknown body keys are ignored, as before). Any other
    job known to the runtime gets the body's simple values as keywords: `day` is checked as a date, and the
    texts "true" / "false" become booleans (a form value "false" must never arrive as a truthy string). A job
    that does not accept one of the keywords fails with that message in its own state; nothing is half-applied.
    """
    if name == "process":
        return {"retry_failed": bool_arg(body.get("retry_failed"), "retry_failed"),
                "limit": int_arg(body.get("limit"), "limit", minimum=0)}
    if name == "daily":
        return {"day": day_arg(body.get("day") or datetime.now().strftime("%Y-%m-%d")),
                "force": bool_arg(body.get("force"), "force"), "push": bool_arg(body.get("push"), "push")}
    if name == "push":
        return {"day": day_arg(body.get("day") or "")}
    if name == "relabel":
        return {"day": day_arg(body["day"]) if body.get("day") else None,
                "segment_id": int_arg(body.get("segment_id"), "segment_id", minimum=1)}
    if name == "reindex":
        return {"vectors": bool_arg(body.get("vectors"), "vectors", default=True)}
    if name == "relations":
        return {"speaker_id": speaker_id_arg(body.get("speaker_id")), "force": bool_arg(body.get("force"), "force")}
    if name in ("cleanup", "voices", "loudness"):
        return {}
    kwargs: dict[str, Any] = {}
    for k, v in body.items():
        if not isinstance(k, str) or not k.isidentifier() or k.startswith("_"):
            raise HTTPException(400, f"任务参数名不合法: {k!r}")
        if isinstance(v, dict) or (isinstance(v, list) and any(isinstance(x, (dict, list)) for x in v)):
            raise HTTPException(400, f"任务参数 {k} 应为简单值或简单值的列表")
        if k == "day":
            v = day_arg(v) if v not in (None, "") else None
        elif isinstance(v, str) and v.strip().lower() in ("true", "false"):
            v = v.strip().lower() == "true"
        kwargs[k] = v
    return kwargs


@router.get("/jobs")
def jobs(rt: RT) -> dict[str, Any]:
    return {n: j.as_dict() for n, j in rt.jobs.items()}


@router.post("/jobs/{name}")
def job_start(name: str, rt: RT, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    if name not in job_names(rt):
        raise HTTPException(404, f"未知任务 {name}")
    kwargs = job_kwargs(name, body)        # argument validation happens before anything is queued
    st = rt.submit(name, **kwargs)
    return {"ok": True, "job": st.as_dict()}


# ---------------------------------------------------------------------------------------------- logs
@router.get("/logs")
def logs(rt: RT, lines: int = 200, which: str = "service") -> dict[str, Any]:
    from ...logging_setup import log_file_name

    role = "cli" if which == "cli" else "service"
    p = rt.cfg.data_dir / log_file_name(rt.cfg, role)
    if not p.exists():
        return {"path": str(p), "lines": [], "size": 0}
    n = max(10, min(int(lines), 5000))
    with open(p, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        chunk = min(size, 256 * 1024 if n <= 2000 else 2 * 1024 * 1024)
        f.seek(size - chunk)
        data = f.read().decode("utf-8", errors="replace")
    out = data.splitlines()[-n:]
    return {"path": str(p), "lines": out, "size": size}
