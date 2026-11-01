"""HTTP API + static files for the local web UI.

    app = create_app(runtime)          # runtime: audio_note.runtime.Runtime

Everything is JSON under /api; the single-page app in ui/static is served at /. The API is local
only (no auth): ui.host must stay 127.0.0.1.

Endpoints (see the handlers for parameters)
  status / service        GET /api/status   POST /api/recording/{start,stop}   POST /api/service/{start,stop}
  jobs                    GET /api/jobs     POST /api/jobs/{name}   (process, daily, cleanup, reindex, relabel, push, voices, relations, loudness)
  transcripts             GET /api/days  /api/transcript/{day}  /api/search  /api/context/{id}  /api/segments
  audio                   GET /api/audio/utterance/{id}  /api/audio/segment/{id}?from=&to=
  daily digests / notes   GET /api/daily  /api/daily/{day}   GET,DELETE /api/notes
  speakers                GET /api/speakers  POST /api/speakers/{name}/samples?force=  POST /api/speakers/label
                          PATCH,DELETE /api/speakers/{id}  GET /api/speakers/{id}/samples  DELETE /api/speakers/samples/{id}
                          DELETE /api/speakers/{id}/samples/outliers  POST /api/speakers/calibrate
  unknown voices          GET /api/voices   POST /api/voices/name   (name a group of not-yet-identified voices)
  mixed utterances        POST /api/utterances/{id}/split {auto | at: [s, ...]}   GET /api/utterances/{id}/preview-split?auto=1
  memory chat             POST /api/chat   (NDJSON stream: tool / answer / error / ping events)
  settings                GET,PUT /api/config   GET,PUT /api/glossary   GET /api/devices   POST /api/config/test-api
                          POST /api/push/test   GET /api/logs   POST /api/import
"""
from __future__ import annotations

import copy
import io
import json
import logging
import os
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..config import (DEFAULT_PROFILE, DEFAULTS, canonical_profile_name, create_profile, delete_profile, is_placeholder_key,
                      list_profiles, load_config, profile_config_path, profile_exists, valid_profile_name)
from ..runtime import JOBS, VOICE_MODELS_MISSING_MSG, JobError, ReadOnly, Runtime, RuntimeHolder, profile_label
from ..service import AlreadyRunning
from . import yaml_edit

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: config sections / keys the settings page may change (anything else is rejected)
EDITABLE: dict[str, set[str]] = {
    "mimo": {"api_key", "base_url", "chat_model", "analysis_model", "asr_model", "asr_language", "asr_concurrency"},
    "audio": {"device", "vad_threshold", "min_silence_s", "min_speech_s", "max_segment_s", "gain", "codec",
              "system_audio", "system_device", "system_gain", "min_speech_dbfs"},
    "storage": {"cleanup_enabled", "keep_audio_days", "max_disk_gb"},
    "speakers": {"identify_threshold", "cluster_threshold", "unknown_label", "num_threads",
                 "voice_group_threshold", "infer_relations", "relation_min_utterances", "relation_min_confidence",
                 "sample_consistency_min", "turn_check_min_s", "turn_reject_margin", "min_label_sample_s"},
    "correction": {"enabled"},
    "pipeline": {"process_interval_min", "min_segment_age_s"},
    "memory": {"strategy", "retrieval", "query_expansion", "prefetch_k", "chat_model", "max_tool_rounds", "embedding_model"},
    "analysis": {"daily_time", "push_daily", "session_gap_min", "catchup_days"},
    "push": {"channels", "feishu", "wecom", "serverchan", "pushplus", "wxpusher", "ntfy"},
    "ui": {"port", "open_browser"},
    "log": {"level"},
}
SECRET_KEYS = {("mimo", "api_key"), ("push", "feishu", "webhook"), ("push", "feishu", "secret"), ("push", "wecom", "webhook"),
               ("push", "serverchan", "sendkey"), ("push", "pushplus", "token"), ("push", "wxpusher", "spt"),
               ("push", "wxpusher", "app_token")}


_CHOICES: dict[tuple[str, ...], tuple[str, ...]] = {
    ("mimo", "asr_language"): ("auto", "zh", "en"),
    ("audio", "codec"): ("opus", "flac", "wav"),
    ("memory", "strategy"): ("agentic", "prefetch", "rag"),
    ("memory", "retrieval"): ("fts", "vector", "hybrid"),
    ("log", "level"): ("DEBUG", "INFO", "WARNING", "ERROR"),
}
_RANGES: dict[tuple[str, ...], tuple[float, float]] = {
    ("audio", "vad_threshold"): (0.0, 1.0), ("audio", "min_silence_s"): (0.2, 60.0), ("audio", "min_speech_s"): (0.0, 10.0),
    ("audio", "max_segment_s"): (10.0, 3600.0), ("audio", "gain"): (0.05, 50.0), ("audio", "system_gain"): (0.05, 50.0),
    ("audio", "min_speech_dbfs"): (-90.0, 0.0),
    ("storage", "keep_audio_days"): (0, 100000), ("storage", "max_disk_gb"): (0, 100000),
    ("speakers", "identify_threshold"): (0.0, 1.0), ("speakers", "cluster_threshold"): (0.0, 1.0), ("speakers", "num_threads"): (1, 64),
    ("speakers", "voice_group_threshold"): (0.0, 1.0), ("speakers", "relation_min_utterances"): (1, 1000),
    ("speakers", "relation_min_confidence"): (0.0, 1.0), ("speakers", "sample_consistency_min"): (0.0, 1.0),
    ("speakers", "turn_check_min_s"): (0.0, 60.0), ("speakers", "turn_reject_margin"): (0.0, 1.0),
    ("speakers", "min_label_sample_s"): (1.0, 60.0),
    ("pipeline", "process_interval_min"): (1, 1440), ("pipeline", "min_segment_age_s"): (0, 3600),
    ("mimo", "asr_concurrency"): (1, 32), ("memory", "prefetch_k"): (1, 100), ("memory", "max_tool_rounds"): (0, 50),
    ("analysis", "session_gap_min"): (1, 1440), ("analysis", "catchup_days"): (1, 365), ("ui", "port"): (1024, 65535),
}
_PUSH_CHANNELS = ("toast", "feishu", "wecom", "serverchan", "pushplus", "wxpusher", "ntfy")
#: numeric settings that may also be null (= feature off); an empty form field means null
_NULLABLE = {("audio", "min_speech_dbfs")}


def _default_for(path: tuple[str, ...]) -> Any:
    node: Any = DEFAULTS
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return _NO_DEFAULT
        node = node[k]
    return node


_NO_DEFAULT = object()


def coerce_setting(path: tuple[str, ...], value: Any) -> Any:
    """Validate / coerce one config leaf against the type of its DEFAULTS entry (form values arrive as
    strings or JSON scalars). Raises HTTPException(400) with a readable message."""
    dotted = ".".join(path)
    default = _default_for(path)
    if default is _NO_DEFAULT:
        raise HTTPException(400, f"不允许修改 {dotted}")
    if isinstance(default, dict):
        raise HTTPException(400, f"{dotted} 是一组设置，不能整个替换")
    if path == ("push", "channels"):
        if isinstance(value, str):
            value = [v.strip() for v in value.replace("，", ",").split(",") if v.strip()]
        if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
            raise HTTPException(400, "push.channels 必须是渠道名列表")
        bad = [v for v in value if v not in _PUSH_CHANNELS]
        if bad:
            raise HTTPException(400, f"未知推送渠道: {', '.join(bad)}（可选 {', '.join(_PUSH_CHANNELS)}）")
        return list(dict.fromkeys(value))
    if isinstance(default, list):
        if isinstance(value, str):
            value = [v.strip() for v in value.replace("，", ",").split(",") if v.strip()]
        if not isinstance(value, list):
            raise HTTPException(400, f"{dotted} 必须是列表")
        return [str(v) for v in value]
    if path in _NULLABLE and (value is None or (isinstance(value, str) and not value.strip())):
        return None
    if path in (("audio", "device"), ("audio", "system_device")):
        if value is None or value == "":
            return None
        if isinstance(value, bool) or isinstance(value, (dict, list)):
            raise HTTPException(400, f"{dotted} 应为设备序号或名称")
        if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
            return int(value)
        return str(value)
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false", "1", "0", "yes", "no", "on", "off"):
            return value.strip().lower() in ("true", "1", "yes", "on")
        raise HTTPException(400, f"{dotted} 应为 true/false")
    if isinstance(default, int) and not isinstance(default, bool):
        if isinstance(value, bool) or isinstance(value, (dict, list)):
            raise HTTPException(400, f"{dotted} 应为整数")
        try:
            iv = int(float(value)) if not isinstance(value, int) else value
        except (TypeError, ValueError):
            raise HTTPException(400, f"{dotted} 应为整数")
        if isinstance(value, float) and abs(value - iv) > 1e-9:
            raise HTTPException(400, f"{dotted} 应为整数")
        value = iv
    elif isinstance(default, float):
        if isinstance(value, bool) or isinstance(value, (dict, list)):
            raise HTTPException(400, f"{dotted} 应为数字")
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise HTTPException(400, f"{dotted} 应为数字")
    elif isinstance(default, str) or default is None:
        if value is None and default is None:
            return None
        if isinstance(value, (dict, list, bool)):
            raise HTTPException(400, f"{dotted} 应为文本")
        value = str(value).strip() if not isinstance(value, str) else value.strip()
        if path in _CHOICES and value not in _CHOICES[path]:
            raise HTTPException(400, f"{dotted} 只能是 {' / '.join(_CHOICES[path])}")
        if path in (("mimo", "base_url"), ("push", "ntfy", "server")) and value and not value.startswith(("http://", "https://")):
            raise HTTPException(400, f"{dotted} 应以 http:// 或 https:// 开头")
        return value
    if path in _RANGES:
        lo, hi = _RANGES[path]
        if not (lo <= value <= hi):
            raise HTTPException(400, f"{dotted} 应在 {lo:g}–{hi:g} 之间")
    return value


def _effective(cfg: Any, path: tuple[str, ...]) -> Any:
    node: Any = cfg
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return _NO_DEFAULT
        node = node[k]
    return node


def _day(s: str) -> str:
    if not DAY_RE.match(s or ""):
        raise HTTPException(400, f"日期格式应为 YYYY-MM-DD: {s!r}")
    return s


def _err(status: int, msg: str) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=status)


def _value_error_http(e: Exception, status: int = 400) -> HTTPException:
    """ValueError -> HTTPException(400, message); a SampleMismatch carries "mismatch": true plus the score so
    the page can offer "仍然加入" (the handler passes a dict detail through as the JSON body)."""
    from ..speech.speakers import SampleMismatch

    if isinstance(e, SampleMismatch):
        return HTTPException(status, {"error": str(e), "mismatch": True, "score": round(e.score, 3),
                                      "threshold": e.threshold, "name": e.name})
    return HTTPException(status, str(e))


def _force_kwargs(force: Any) -> dict[str, Any]:
    """`force=True` only when asked, so engines without the keyword (older fakes) keep working."""
    return {"force": True} if force else {}


def _force_arg(v: Any) -> bool:
    """A `force` flag from a body / query: JSON true / false / 0 / 1 (or those as text); anything else is 400."""
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in ("true", "false", "0", "1"):
        return v.strip().lower() in ("true", "1")
    raise HTTPException(400, "force 应为 true/false")


def _loudness_of(r: Any) -> float | None:
    """utterances.loudness_db of a row, rounded for display (None when not computed / column absent)."""
    try:
        v = r["loudness_db"]
    except (IndexError, KeyError):
        return None
    return round(float(v), 1) if v is not None else None


def _hint(secret: str | None) -> str:
    s = str(secret or "")
    if not s:
        return ""
    return s[:3] + "…" + s[-4:] if len(s) > 10 else "…" + s[-2:]


def _utt_dict(r: Any, unknown: str, audio_ok: bool | None = None) -> dict[str, Any]:
    text = r["text"] or ""
    raw = r["text_raw"] or ""
    d = {
        "id": int(r["id"]), "segment_id": int(r["segment_id"]), "day": r["day"], "abs_start": r["abs_start"],
        "time": (r["abs_start"] or "")[11:19], "start_s": float(r["start_s"]), "end_s": float(r["end_s"]),
        "speaker": r["speaker_name"] or unknown, "speaker_id": r["speaker_id"],
        "score": round(float(r["speaker_score"]), 3) if r["speaker_score"] is not None else None,
        "text": text, "cluster": r["cluster_label"], "loudness_db": _loudness_of(r),
    }
    if raw and raw != text:
        d["text_raw"] = raw
    if audio_ok is not None:
        d["audio"] = audio_ok
    return d


def _speaker_id_arg(v: Any) -> int | None:
    """Optional speaker id from a request body: None / "" mean "all", anything else must be a positive integer."""
    if v is None or v == "":
        return None
    if isinstance(v, bool) or isinstance(v, (dict, list)):
        raise HTTPException(400, "speaker_id 应为说话人 id")
    try:
        sid = int(v)
    except (TypeError, ValueError):
        raise HTTPException(400, "speaker_id 应为说话人 id")
    if sid <= 0:
        raise HTTPException(400, "speaker_id 应为说话人 id")
    return sid


def _wav_bytes(audio: Any) -> bytes:
    import numpy as np
    import soundfile as sf

    from ..audio.vad import SAMPLE_RATE

    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class _RuntimeProxy:
    """Every handler reads `rt.<attr>` through this proxy, so a profile switch (holder.rt replaced)
    is picked up by the next request without rebuilding the app."""

    def __init__(self, holder: RuntimeHolder):
        object.__setattr__(self, "_holder", holder)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._holder.rt, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._holder.rt, name, value)


def _profile_counts(cfg: Any) -> dict[str, Any] | None:
    """Row counts of a profile's database read-only (None when it has no database yet)."""
    import sqlite3

    if not cfg.db_path.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{cfg.db_path.as_posix()}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return None
    try:
        row = con.execute("SELECT COUNT(*), COUNT(DISTINCT day), MAX(day) FROM utterances").fetchone()
        spk = con.execute("SELECT COUNT(*) FROM speakers").fetchone()
        return {"utterances": int(row[0] or 0), "days": int(row[1] or 0), "last_day": row[2], "speakers": int(spk[0] or 0)}
    except sqlite3.Error:
        return None
    finally:
        con.close()


def create_app(rt_or_holder: Runtime | RuntimeHolder) -> FastAPI:
    app = FastAPI(title="audio-note", docs_url=None, redoc_url=None, openapi_url=None)
    holder = rt_or_holder if isinstance(rt_or_holder, RuntimeHolder) else RuntimeHolder(rt_or_holder)
    app.state.holder = holder
    rt: Any = _RuntimeProxy(holder)
    unknown = lambda: str(rt.cfg.speakers.unknown_label or "未知")  # noqa: E731

    @app.exception_handler(JobError)
    async def _job_error(request: Request, exc: JobError):
        return _err(409, str(exc))

    @app.exception_handler(ReadOnly)
    async def _read_only(request: Request, exc: ReadOnly):
        return _err(409, str(exc))

    @app.exception_handler(AlreadyRunning)
    async def _already(request: Request, exc: AlreadyRunning):
        return _err(409, str(exc))

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException):
        if isinstance(exc.detail, dict):
            body = dict(exc.detail)
            body["error"] = str(body.get("error") or "")
            return JSONResponse(body, status_code=exc.status_code)
        return _err(exc.status_code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _any_error(request: Request, exc: Exception):
        log.exception("api %s %s failed: %s", request.method, request.url.path, exc)
        return _err(500, f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ status / service
    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return rt.status()

    @app.post("/api/recording/start")
    def recording_start() -> dict[str, Any]:
        if rt.status()["models_missing"]:
            raise HTTPException(400, "模型文件缺失，无法录音（运行 scripts/download_models.ps1）")
        if not rt.service_running():
            rt.start_service(record=True)
        else:
            rt.start_recording()
        time.sleep(0.3)   # let the thread hit the mic so a device error shows up in this response
        st = rt.status()["service"]
        return {"ok": True, "recording": st["recording"], "last_error": st["last_error"]}

    @app.post("/api/recording/stop")
    def recording_stop() -> dict[str, Any]:
        rt.stop_recording()
        return {"ok": True, "recording": rt.recording()}

    @app.post("/api/service/start")
    def service_start(body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        rt.start_service(record=bool(body.get("record", False)))
        return {"ok": True, "service": rt.status()["service"]}

    @app.post("/api/service/stop")
    def service_stop() -> dict[str, Any]:
        rt.stop_service()
        return {"ok": True, "service": rt.status()["service"]}

    # ------------------------------------------------------------------ profiles
    def _profile_info(name: str) -> dict[str, Any]:
        d: dict[str, Any] = {"name": name, "label": profile_label(name), "current": rt.profile == name,
                             "config_path": str(profile_config_path(name)), "default": name == DEFAULT_PROFILE}
        try:
            cfg = load_config(profile=name)
            d["data_dir"] = str(cfg.data_dir)
            d["stats"] = _profile_counts(cfg)
            from ..service import running_pid

            pid = running_pid(cfg)
            d["running_pid"] = pid if pid and pid != os.getpid() else None
        except Exception as e:  # noqa: BLE001
            d["error"] = str(e)
        return d

    @app.get("/api/profiles")
    def profiles_list() -> dict[str, Any]:
        return {"current": rt.profile, "current_label": profile_label(rt.profile),
                "custom_config": str(rt.config_path) if rt.profile is None else None,
                "profiles": [_profile_info(n) for n in list_profiles()]}

    @app.post("/api/profiles")
    def profiles_create(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        name = str(body.get("name") or "").strip()
        if not valid_profile_name(name):
            raise HTTPException(400, "档案名只能用中英文、数字、下划线、短横线（1–32 字），且不能叫 default / current")
        src = None
        copy_from = body.get("copy_from")
        if copy_from:
            if not profile_exists(str(copy_from)):
                raise HTTPException(404, f"档案不存在: {copy_from}")
            src = profile_config_path(str(copy_from))
        try:
            path = create_profile(name, copy_from=src)
        except FileExistsError as e:
            raise HTTPException(409, str(e))
        except (ValueError, OSError) as e:
            raise HTTPException(400, str(e))
        out: dict[str, Any] = {"ok": True, "name": name, "config_path": str(path)}
        if body.get("switch"):
            holder.switch(name)
            out["switched"] = True
        return out

    @app.post("/api/profiles/{name}/switch")
    def profiles_switch(name: str) -> dict[str, Any]:
        name = canonical_profile_name(name) or name
        if not profile_exists(name):
            raise HTTPException(404, f"档案不存在: {name}")
        try:
            holder.switch(name)          # JobError (a job is running) -> 409 via the handler above
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except OSError as e:
            raise HTTPException(500, f"无法打开档案 {name}：{e}")
        st = rt.status()
        return {"ok": True, "profile": st["profile"], "service": st["service"]}

    @app.delete("/api/profiles/{name}")
    def profiles_delete(name: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        name = canonical_profile_name(name) or name
        if name == DEFAULT_PROFILE:
            raise HTTPException(400, "默认档案不能删除")
        if not profile_exists(name):
            raise HTTPException(404, f"档案不存在: {name}")
        if rt.profile == name:
            raise HTTPException(409, "不能删除正在使用的档案，先切换到别的档案")
        if str(body.get("confirm") or "") != name:
            raise HTTPException(400, "需要在 confirm 里重复档案名以确认删除")
        info = _profile_info(name)
        if info.get("running_pid"):
            raise HTTPException(409, f"这个档案的服务正在另一个进程运行 (pid {info['running_pid']})，先停止它")
        try:
            delete_profile(name)
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))
        except OSError as e:          # ProfileInUse: nothing was deleted
            raise HTTPException(409, str(e))
        return {"ok": True, "deleted": name}

    # ------------------------------------------------------------------ jobs
    @app.get("/api/jobs")
    def jobs() -> dict[str, Any]:
        return {n: j.as_dict() for n, j in rt.jobs.items()}

    @app.post("/api/jobs/{name}")
    def job_start(name: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        if name not in JOBS:
            raise HTTPException(404, f"未知任务 {name}")
        kwargs: dict[str, Any] = {}
        if name == "process":
            kwargs = {"retry_failed": bool(body.get("retry_failed", False)), "limit": body.get("limit")}
        elif name == "daily":
            kwargs = {"day": _day(body.get("day") or datetime.now().strftime("%Y-%m-%d")),
                      "force": bool(body.get("force", False)), "push": bool(body.get("push", False))}
        elif name == "push":
            kwargs = {"day": _day(body.get("day") or "")}
        elif name == "relabel":
            kwargs = {"day": _day(body["day"]) if body.get("day") else None, "segment_id": body.get("segment_id")}
        elif name == "reindex":
            kwargs = {"vectors": bool(body.get("vectors", True))}
        elif name == "relations":
            kwargs = {"speaker_id": _speaker_id_arg(body.get("speaker_id")), "force": bool(body.get("force", False))}
        st = rt.submit(name, **kwargs)
        return {"ok": True, "job": st.as_dict()}

    # ------------------------------------------------------------------ transcripts
    @app.get("/api/days")
    def days() -> list[dict[str, Any]]:
        db = rt.db
        out: dict[str, dict[str, Any]] = {}
        for r in db.query("SELECT day, COUNT(*) AS n, MIN(abs_start) AS first, MAX(abs_start) AS last,"
                          " COALESCE(SUM(end_s-start_s),0) AS speech FROM utterances GROUP BY day"):
            out[r["day"]] = {"day": r["day"], "utterances": int(r["n"]), "first": (r["first"] or "")[11:16],
                             "last": (r["last"] or "")[11:16], "speech_min": round(float(r["speech"]) / 60.0, 1),
                             "segments": 0, "pending": 0, "failed": 0, "daily": None, "speakers": []}
        for r in db.query("SELECT substr(started_at,1,10) AS d, status, COUNT(*) AS n FROM segments"
                          " WHERE status != 'deleted' GROUP BY d, status"):
            d = out.setdefault(r["d"], {"day": r["d"], "utterances": 0, "first": "", "last": "", "speech_min": 0.0,
                                        "segments": 0, "pending": 0, "failed": 0, "daily": None, "speakers": []})
            d["segments"] += int(r["n"])
            if r["status"] in ("new", "processing"):
                d["pending"] += int(r["n"])
            elif r["status"] == "failed":
                d["failed"] += int(r["n"])
        for r in db.query("SELECT day, created_at, pushed_at, utterance_count FROM daily_summaries"):
            if r["day"] in out:
                out[r["day"]]["daily"] = {"created_at": r["created_at"], "pushed_at": r["pushed_at"],
                                          "utterance_count": r["utterance_count"]}
        for r in db.query("SELECT day, speaker_name, COUNT(*) AS n FROM utterances GROUP BY day, speaker_name ORDER BY n DESC"):
            if r["day"] in out:
                out[r["day"]]["speakers"].append({"name": r["speaker_name"] or unknown(), "n": int(r["n"])})
        return sorted(out.values(), key=lambda d: d["day"], reverse=True)

    @app.get("/api/transcript/{day}")
    def transcript(day: str, speaker: str | None = None) -> dict[str, Any]:
        from ..analysis.daily import DailyAnalyzer

        _day(day)
        db = rt.db
        unk = unknown()
        sessions = DailyAnalyzer(rt.cfg, db, None).sessions(day)
        seg_ok: dict[int, bool] = {}

        def audio_ok(seg_id: int) -> bool:
            if seg_id not in seg_ok:
                row = db.get_segment(seg_id)
                seg_ok[seg_id] = bool(row and Path(row["path"]).exists())
            return seg_ok[seg_id]

        out_sessions = []
        for i, rows in enumerate(sessions, start=1):
            if speaker:
                rows = [r for r in rows if (r["speaker_name"] or unk) == speaker]
                if not rows:
                    continue
            utts = [_utt_dict(r, unk, audio_ok(int(r["segment_id"]))) for r in rows]
            out_sessions.append({"index": i, "start": utts[0]["time"][:5], "end": utts[-1]["time"][:5], "utterances": utts})
        segs = [dict(r) for r in db.query(
            "SELECT id, started_at, duration_s, speech_s, status, error, source FROM segments"
            " WHERE started_at >= ? AND started_at < ? AND status != 'deleted' ORDER BY started_at",
            (f"{day}T00:00:00", f"{day}T23:59:59.999"))]
        counts: dict[str, int] = {}
        for r in db.query("SELECT speaker_name, COUNT(*) AS n FROM utterances WHERE day=? GROUP BY speaker_name", (day,)):
            counts[r["speaker_name"] or unk] = int(r["n"])
        md = rt.cfg.transcripts_dir / f"{day}.md"
        from ..audio.loudness import min_speech_dbfs

        return {"day": day, "sessions": out_sessions, "segments": segs, "speakers": counts,
                "markdown_path": str(md) if md.exists() else None, "unknown_label": unk,
                "min_speech_dbfs": min_speech_dbfs(rt.cfg)}

    @app.get("/api/segments")
    def segments(status: str | None = None, day: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        where, params = ["status != 'deleted'"], []
        if status:
            where.append("status=?")
            params.append(status)
        if day:
            where.append("started_at >= ? AND started_at < ?")
            params += [f"{_day(day)}T00:00:00", f"{day}T23:59:59.999"]
        rows = rt.db.query(f"SELECT id, path, started_at, duration_s, speech_s, status, error, source, updated_at FROM segments"
                           f" WHERE {' AND '.join(where)} ORDER BY started_at DESC LIMIT ?", [*params, max(1, min(int(limit), 2000))])
        out = []
        for r in rows:
            d = dict(r)
            d["audio"] = Path(d["path"]).exists()
            d["name"] = Path(d.pop("path")).name
            out.append(d)
        return out

    @app.get("/api/search")
    def search(q: str, day_from: str | None = None, day_to: str | None = None, speaker: str | None = None,
               limit: int = 30, mode: str | None = None) -> dict[str, Any]:
        q = (q or "").strip()
        if not q:
            return {"hits": [], "how": ""}
        if day_from:
            _day(day_from)
        if day_to:
            _day(day_to)
        retr = rt.retriever(mode, False) if mode else rt.retriever(None, False)
        res = retr.search(q, day_from=day_from, day_to=day_to, speaker=speaker, limit=max(1, min(int(limit), 200)))
        hits = [{"id": h.id, "segment_id": h.segment_id, "day": h.day, "abs_start": h.abs_start, "time": h.abs_start[11:16],
                 "speaker": h.speaker_name or unknown(), "text": h.text, "score": round(float(h.score), 4), "source": h.source}
                for h in res.hits]
        return {"hits": hits, "how": res.how_label, "expansions": res.expansions}

    @app.get("/api/context/{uid}")
    def context(uid: int, before: int = 6, after: int = 6) -> dict[str, Any]:
        from ..memory import index

        rows = index.window(rt.db, int(uid), max(0, int(before)), max(0, int(after)))
        if not rows:
            raise HTTPException(404, "没有这条记录")
        return {"id": uid, "utterances": [_utt_dict(r, unknown()) for r in rows]}

    # ------------------------------------------------------------------ audio
    def _segment_audio(seg_id: int, a: float, b: float) -> bytes:
        from ..audio.storage import read_segment
        from ..audio.vad import SAMPLE_RATE

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
        return _wav_bytes(audio[i:j])

    @app.get("/api/audio/utterance/{uid}")
    def audio_utterance(uid: int, pad: float = 0.3) -> Response:
        u = rt.db.one("SELECT segment_id, start_s, end_s FROM utterances WHERE id=?", (int(uid),))
        if u is None:
            raise HTTPException(404, "没有这条记录")
        pad = max(0.0, min(float(pad), 5.0))
        data = _segment_audio(int(u["segment_id"]), float(u["start_s"]) - pad, float(u["end_s"]) + pad)
        # utterance ids are reused by splits / re-processing: never let the browser play a stale clip
        return Response(data, media_type="audio/wav", headers={"Cache-Control": "no-store"})

    @app.get("/api/audio/segment/{seg_id}")
    def audio_segment(seg_id: int, start: float = Query(0.0, alias="from"), end: float = Query(0.0, alias="to")) -> Response:
        data = _segment_audio(int(seg_id), float(start), float(end))
        return Response(data, media_type="audio/wav", headers={"Cache-Control": "private, max-age=3600"})

    # ------------------------------------------------------------------ daily digests / notes
    @app.get("/api/daily")
    def daily_list() -> list[dict[str, Any]]:
        rows = rt.db.query("SELECT d.day, d.created_at, d.pushed_at, d.utterance_count, d.model, d.summary_json,"
                           " (SELECT COUNT(*) FROM utterances u WHERE u.day=d.day) AS now_count"
                           " FROM daily_summaries d ORDER BY d.day DESC")
        out = []
        for r in rows:
            partial = False
            overview = ""
            try:
                js = json.loads(r["summary_json"] or "{}")
                partial = bool(js.get("partial"))
                overview = str(js.get("overview") or "")[:160]
            except ValueError:
                pass
            out.append({"day": r["day"], "created_at": r["created_at"], "pushed_at": r["pushed_at"], "model": r["model"],
                        "utterance_count": r["utterance_count"], "now_count": int(r["now_count"]), "partial": partial,
                        "stale": int(r["now_count"]) != int(r["utterance_count"] or 0), "overview": overview})
        return out

    @app.get("/api/daily/{day}")
    def daily_get(day: str) -> dict[str, Any]:
        _day(day)
        r = rt.db.one("SELECT * FROM daily_summaries WHERE day=?", (day,))
        now = rt.db.one("SELECT COUNT(*) AS n FROM utterances WHERE day=?", (day,))
        n_now = int(now["n"]) if now else 0
        if r is None:
            return {"day": day, "exists": False, "now_count": n_now}
        try:
            js = json.loads(r["summary_json"] or "{}")
        except ValueError:
            js = {}
        js.pop("blocks", None)
        return {"day": day, "exists": True, "markdown": r["summary_md"], "json": js, "created_at": r["created_at"],
                "pushed_at": r["pushed_at"], "model": r["model"], "utterance_count": r["utterance_count"],
                "now_count": n_now, "stale": n_now != int(r["utterance_count"] or 0)}

    @app.get("/api/notes")
    def notes(kind: str | None = None, q: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        where, params = [], []
        if kind:
            where.append("kind=?")
            params.append(kind)
        if q:
            where.append("(title LIKE ? OR content LIKE ?)")
            pat = f"%{q.strip()}%"
            params += [pat, pat]
        sql = "SELECT id, kind, title, content, day_from, day_to, updated_at FROM memory_notes"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY updated_at DESC, id DESC LIMIT ?"
        return [dict(r) for r in rt.db.query(sql, [*params, max(1, min(int(limit), 2000))])]

    @app.delete("/api/notes/{note_id}")
    def note_delete(note_id: int) -> dict[str, Any]:
        cur = rt.db.execute("DELETE FROM memory_notes WHERE id=?", (int(note_id),))
        return {"ok": True, "deleted": int(cur.rowcount or 0)}

    # ------------------------------------------------------------------ speakers
    def _speakers() -> list[dict[str, Any]]:
        rows = rt.db.query(
            "SELECT s.id, s.name, s.notes, s.created_at, s.is_self, s.relation, s.relation_source, s.relation_note,"
            " COUNT(e.id) AS samples, COALESCE(SUM(e.duration_s),0) AS dur"
            " FROM speakers s LEFT JOIN speaker_embeddings e ON e.speaker_id=s.id GROUP BY s.id ORDER BY s.name")
        counts = {r["speaker_id"]: int(r["n"]) for r in rt.db.query(
            "SELECT speaker_id, COUNT(*) AS n FROM utterances WHERE speaker_id IS NOT NULL GROUP BY speaker_id")}
        out = []
        for r in rows:
            audit = _audit(int(r["id"])) if int(r["samples"]) > 1 else []
            out.append({"id": int(r["id"]), "name": r["name"], "notes": r["notes"] or "", "created_at": r["created_at"],
                        "samples": int(r["samples"]), "duration_s": round(float(r["dur"]), 1), "utterances": counts.get(r["id"], 0),
                        "is_self": bool(r["is_self"]), "relation": r["relation"], "relation_source": r["relation_source"],
                        "relation_note": r["relation_note"],
                        "outlier_samples": sum(1 for a in audit if a["outlier"]),
                        "ambiguous_samples": sum(1 for a in audit if a["ambiguous"])})
        return out

    def _audit(sid: int) -> list[dict[str, Any]]:
        from ..speech.speakers import audit_samples, sample_consistency_min

        return audit_samples(rt.db, int(sid), sample_consistency_min(rt.cfg))

    def _assign_utterances(sid: int, name: str, seg_id: int, ids: list[int], whole_cluster: bool) -> tuple[int, int]:
        """Hand-label utterances (speaker_source='user': relabel leaves them alone). With whole_cluster every
        utterance of the same segment that shares a cluster_label with one of `ids` is included - except the
        ones the user already labelled by hand as somebody else, which are kept. Returns (assigned, skipped)."""
        db = rt.db
        if not ids:
            return 0, 0
        marks = ",".join("?" * len(ids))
        where, params = f"id IN ({marks})", [*ids]
        skipped = 0
        if whole_cluster:
            labels = [r["cluster_label"] for r in db.query(f"SELECT DISTINCT cluster_label FROM utterances WHERE id IN ({marks})", ids)
                      if r["cluster_label"] is not None]
            if labels:
                lmarks = ",".join("?" * len(labels))
                guard = "(speaker_source IS NULL OR speaker_source<>'user' OR speaker_id IS NULL OR speaker_id=?)"
                other = db.one(f"SELECT COUNT(*) AS n FROM utterances WHERE segment_id=? AND cluster_label IN ({lmarks})"
                               f" AND id NOT IN ({marks}) AND NOT {guard}", [int(seg_id), *labels, *ids, int(sid)])
                skipped = int(other["n"]) if other else 0
                where += f" OR (segment_id=? AND cluster_label IN ({lmarks}) AND {guard})"
                params += [int(seg_id), *labels, int(sid)]
        cur = db.execute(f"UPDATE utterances SET speaker_id=?, speaker_name=?, speaker_score=NULL, speaker_source='user'"
                         f" WHERE {where}", [int(sid), name, *params])
        return int(cur.rowcount or 0), skipped

    def _apply_self_relation(sid: int, is_self: Any, relation: Any) -> None:
        if is_self:
            rt.db.set_self_speaker(int(sid))
        if relation is not None and str(relation).strip():
            rt.db.set_relation(int(sid), str(relation).strip(), "user")

    def _after_enroll(seg_id: int | None = None) -> dict[str, Any]:
        """Register the (new) name with jieba, refresh text_seg, re-identify one segment if given."""
        from ..memory import index as mem_index

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
                from ..speech.pipeline import Processor

                out["relabeled_changed"] = Processor(rt.cfg, rt.db, engine=rt.engine()).relabel_segment(seg_id)
            except Exception as e:  # noqa: BLE001
                log.warning("relabel after label failed: %s", e)
                out["relabel_error"] = str(e)
        return out

    @app.get("/api/speakers")
    def speakers_list() -> dict[str, Any]:
        from ..analysis.relations import RELATION_CHOICES
        from ..audio.loudness import min_speech_dbfs
        from ..speech.voices import unknown_utterance_count

        me = rt.db.self_speaker()
        return {"speakers": _speakers(), "unknown_utterances": unknown_utterance_count(rt.db, min_speech_dbfs(rt.cfg)),
                "identify_threshold": float(rt.cfg.speakers.identify_threshold), "unknown_label": unknown(),
                "self_id": int(me["id"]) if me else None, "relation_choices": list(RELATION_CHOICES)}

    # ------------------------------------------------------------------ unknown voices
    @app.get("/api/voices")
    def voices_list() -> dict[str, Any]:
        """Not-yet-identified voices grouped across recordings (see speech.voices.group_unknown_voices)."""
        from ..speech import voices

        out = voices.group_unknown_voices(rt.db, rt.cfg)
        out["scanning"] = bool(rt.jobs["voices"].running)
        return out

    @app.post("/api/voices/name")
    async def voices_name(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Give a voice group a name: {members: [[segment_id, cluster_label], ...], name, is_self?, relation?,
        auto_relabel?: true}. Enrolls samples from the group, assigns its utterances (speaker_source='user'),
        then re-identifies the other recordings in the background when unidentified utterances remain."""
        from ..speech import voices

        name = str(body.get("name") or "").strip()
        if not name:
            raise HTTPException(400, "姓名不能为空")
        members = body.get("members")
        if not isinstance(members, list) or not members:
            raise HTTPException(400, "members 不能为空")
        is_self = bool(body.get("is_self", False))
        relation = body.get("relation")
        relation = str(relation).strip() if relation is not None else None
        force = _force_arg(body.get("force"))
        if rt.speaker_models_missing():
            raise HTTPException(400, VOICE_MODELS_MISSING_MSG)
        db = rt.db
        with rt.busy():                      # a profile switch waits (JobError -> 409) instead of closing the DB under us
            try:
                res = await _to_thread(lambda: voices.name_voice(db, rt.cfg, rt.engine(), members, name,
                                                                 is_self=is_self, relation=relation or None, force=force))
            except ValueError as e:
                raise _value_error_http(e)
            extra = await _to_thread(_after_enroll)
        relabel_started = False
        if body.get("auto_relabel", True):
            left = db.one("SELECT COUNT(*) AS n FROM utterances WHERE speaker_id IS NULL")
            if left and int(left["n"]) and db.one("SELECT 1 FROM speaker_embeddings LIMIT 1"):
                try:
                    rt.submit("relabel")
                    relabel_started = True
                except (JobError, ReadOnly) as e:
                    log.info("relabel after naming a voice not started: %s", e)
        return {"ok": True, **res, "relabel_started": relabel_started, **extra}

    @app.get("/api/speakers/{sid}/samples")
    def speaker_samples(sid: int) -> list[dict[str, Any]]:
        rows = rt.db.query("SELECT id, source_path, duration_s, created_at, origin FROM speaker_embeddings WHERE speaker_id=? ORDER BY id",
                           (int(sid),))
        audit = {a["id"]: a for a in _audit(int(sid))} if rows else {}
        out = []
        for r in rows:
            d = dict(r)
            d["file"] = Path(str(d["source_path"]).split("#")[0]).name if d["source_path"] else ""
            d["audio"] = bool(d["source_path"]) and Path(str(d["source_path"]).split("#")[0]).exists()
            a = audit.get(int(r["id"])) or {}
            d["consistency"] = a.get("consistency")
            d["outlier"] = bool(a.get("outlier", False))
            d["ambiguous"] = bool(a.get("ambiguous", False))
            out.append(d)
        return out

    @app.delete("/api/speakers/{sid}/samples/outliers")
    def speaker_delete_outlier_samples(sid: int) -> dict[str, Any]:
        """Remove every sample of the speaker that does not resemble the others (speakers.sample_consistency_min).
        The speaker row stays even when no sample is left."""
        db = rt.db
        if db.one("SELECT id FROM speakers WHERE id=?", (int(sid),)) is None:
            raise HTTPException(404, "没有这个说话人")
        bad = [a["id"] for a in _audit(int(sid)) if a["outlier"]]
        deleted = 0
        if bad:
            cur = db.execute(f"DELETE FROM speaker_embeddings WHERE speaker_id=? AND id IN ({','.join('?' * len(bad))})", [int(sid), *bad])
            deleted = int(cur.rowcount or 0)
        left = db.one("SELECT COUNT(*) AS n FROM speaker_embeddings WHERE speaker_id=?", (int(sid),))
        return {"ok": True, "deleted": deleted, "remaining": int(left["n"]) if left else 0}

    @app.get("/api/speakers/samples/{emb_id}/audio")
    def speaker_sample_audio(emb_id: int) -> Response:
        from ..audio.importer import decode_any

        r = rt.db.one("SELECT source_path FROM speaker_embeddings WHERE id=?", (int(emb_id),))
        if r is None or not r["source_path"]:
            raise HTTPException(404, "样本不存在")
        src = str(r["source_path"])
        p = Path(src.split("#")[0])
        if not p.exists():
            raise HTTPException(404, "样本文件已不存在")
        audio = decode_any(p)
        if "#" in src:
            from ..audio.vad import SAMPLE_RATE

            try:
                a, b = src.split("#")[1].split("-")
                audio = audio[int(float(a) * SAMPLE_RATE):int(float(b) * SAMPLE_RATE)]
            except ValueError:
                pass
        return Response(_wav_bytes(audio), media_type="audio/wav")

    @app.delete("/api/speakers/samples/{emb_id}")
    def speaker_sample_delete(emb_id: int) -> dict[str, Any]:
        cur = rt.db.execute("DELETE FROM speaker_embeddings WHERE id=?", (int(emb_id),))
        return {"ok": True, "deleted": int(cur.rowcount or 0)}

    @app.post("/api/speakers/{name}/samples")
    async def speaker_add_sample(name: str, request: Request, filename: str = "sample.webm", notes: str | None = None,
                                 force: str | None = None) -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise HTTPException(400, "姓名不能为空")
        force = _force_arg(force)
        data = await request.body()
        if len(data) < 1000:
            raise HTTPException(400, "没有收到音频数据")
        ext = Path(filename).suffix.lower() or ".webm"
        if ext not in (".wav", ".webm", ".ogg", ".opus", ".mp3", ".m4a", ".flac", ".aac", ".mp4", ".wma", ".amr"):
            raise HTTPException(400, f"不支持的音频格式 {ext}")
        tmp_dir = rt.cfg.data_dir / "uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = tmp_dir / f"enroll_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{ext}"
        existed = rt.db.one("SELECT id FROM speakers WHERE name=?", (name,)) is not None
        db = rt.db

        def enroll() -> tuple[int, int, int, float]:
            tmp.write_bytes(data)
            before = db.one("SELECT COUNT(*) AS n FROM speaker_embeddings e JOIN speakers s ON s.id=e.speaker_id WHERE s.name=?", (name,))
            n0 = int(before["n"]) if before else 0
            sid = rt.engine().enroll(db, name, [tmp], notes, **_force_kwargs(force))
            after = db.one("SELECT COUNT(*) AS n, COALESCE(MAX(duration_s),0) AS last FROM speaker_embeddings WHERE speaker_id=?", (sid,))
            return sid, n0, int(after["n"]) if after else 0, float(after["last"]) if after else 0.0

        def mismatch_of_upload() -> HTTPException | None:
            """enroll() skips a mismatching file silently: re-check it so the page can offer "仍然加入"."""
            engine = rt.engine()
            try:
                from ..audio.importer import decode_any
                from ..speech.speakers import MIN_ENROLL_SPEECH_S, SampleMismatch, sample_consistency_min
                from ..audio.vad import SAMPLE_RATE

                row = db.one("SELECT id FROM speakers WHERE name=?", (name,))
                if row is None:
                    return None
                speech = engine.speech_only(decode_any(tmp))
                if len(speech) / SAMPLE_RATE < MIN_ENROLL_SPEECH_S:
                    return None
                score, ok = engine.check_sample(db, int(row["id"]), engine.embed(speech))
                if ok or score is None:
                    return None
                return _value_error_http(SampleMismatch(name, float(score), sample_consistency_min(rt.cfg)))
            except Exception as e:  # noqa: BLE001 - diagnostics only
                log.debug("mismatch re-check failed: %s", e)
                return None

        with rt.busy():
            try:
                sid, n0, n1, last = await _to_thread(enroll)
                exc = None
                if n1 <= n0 and n0 > 0 and not force:
                    exc = await _to_thread(mismatch_of_upload)
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
                    rt.db.execute("DELETE FROM speakers WHERE id=?", (sid,))
                raise HTTPException(400, "样本里检测到的语音太少（至少 1 秒，建议 10 秒以上），请重录")
            extra = await _to_thread(_after_enroll)
        return {"ok": True, "id": sid, "name": name, "samples": n1, "speech_s": round(last, 1), **extra}

    @app.post("/api/speakers/label")
    async def speaker_label(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        name = str(body.get("name") or "").strip()
        if not name:
            raise HTTPException(400, "姓名不能为空")
        db = rt.db
        whole_cluster = bool(body.get("whole_cluster", True))
        utt_ids: list[int]
        if body.get("utterance_id") is not None:
            u = db.one("SELECT id, segment_id, start_s, end_s FROM utterances WHERE id=?", (int(body["utterance_id"]),))
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
        force_kw = _force_kwargs(_force_arg(body.get("force")))
        sample_note: str | None = None
        with rt.busy():
            try:
                from ..speech.speakers import TooShortSample, min_label_sample_s

                seg_path = Path(seg["path"])
                min_s = min_label_sample_s(rt.cfg)
                sid = await _to_thread(lambda: rt.engine().enroll_from_segment_turns(db, name, seg_path, [(a, b)],
                                                                                     min_speech_s=min_s, **force_kw))
            except TooShortSample as e:
                # the user said who spoke: assign the sentence anyway, just without a voice sample
                sample_note = str(e)
                sid = db.upsert_speaker(name)
            except ValueError as e:
                raise _value_error_http(e)
            assigned, skipped = _assign_utterances(sid, name, seg_id, utt_ids, whole_cluster)
            extra = await _to_thread(_after_enroll, seg_id)       # relabel_segment leaves the hand-labelled ones alone
            _apply_self_relation(sid, body.get("is_self"), body.get("relation"))
            if assigned:
                try:
                    from ..speech.pipeline import write_day_markdown

                    for d in db.query("SELECT DISTINCT day FROM utterances WHERE segment_id=? AND speaker_id=?", (seg_id, sid)):
                        write_day_markdown(db, rt.cfg, d["day"])
                except Exception as e:  # noqa: BLE001
                    log.warning("rewrite markdown after label failed: %s", e)
        n = db.one("SELECT COUNT(*) AS n FROM speaker_embeddings WHERE speaker_id=?", (sid,))
        return {"ok": True, "id": sid, "name": name, "samples": int(n["n"]) if n else 0, "segment_id": seg_id,
                "assigned": assigned, "skipped_user_labels": skipped, "sample_note": sample_note, **extra}

    @app.patch("/api/speakers/{sid}")
    def speaker_update(sid: int, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        db = rt.db
        row = db.one("SELECT id, name FROM speakers WHERE id=?", (int(sid),))
        if row is None:
            raise HTTPException(404, "没有这个说话人")
        if "notes" in body:
            db.execute("UPDATE speakers SET notes=? WHERE id=?", (str(body["notes"] or "") or None, int(sid)))
        if "is_self" in body:
            if body["is_self"]:
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
        new_name = str(body.get("name") or "").strip()
        renamed_days: list[str] = []
        if new_name and new_name != row["name"]:
            if db.one("SELECT id FROM speakers WHERE name=? AND id != ?", (new_name, int(sid))):
                raise HTTPException(400, f"已经有叫 {new_name} 的说话人")
            db.execute("UPDATE speakers SET name=? WHERE id=?", (new_name, int(sid)))
            db.execute("UPDATE utterances SET speaker_name=? WHERE speaker_id=?", (new_name, int(sid)))
            renamed_days = [r["day"] for r in db.query("SELECT DISTINCT day FROM utterances WHERE speaker_id=?", (int(sid),))]
            _after_enroll()
            try:
                from ..speech.pipeline import write_day_markdown

                for d in renamed_days:
                    write_day_markdown(db, rt.cfg, d)
            except Exception as e:  # noqa: BLE001
                log.warning("rewrite markdown after rename failed: %s", e)
        return {"ok": True, "renamed_days": len(renamed_days)}

    @app.delete("/api/speakers/{sid}")
    def speaker_delete(sid: int) -> dict[str, Any]:
        db = rt.db
        row = db.one("SELECT id, name FROM speakers WHERE id=?", (int(sid),))
        if row is None:
            raise HTTPException(404, "没有这个说话人")
        unk = unknown()
        days = [r["day"] for r in db.query("SELECT DISTINCT day FROM utterances WHERE speaker_id=?", (int(sid),))]
        # the utterances become unidentified again (no 'user' mark left behind, so relabel may re-identify
        # them) and the relation inference cache of the id is dropped
        db.delete_speaker(int(sid), unk)
        try:
            from ..speech.pipeline import write_day_markdown

            for d in days:
                write_day_markdown(db, rt.cfg, d)
        except Exception as e:  # noqa: BLE001
            log.warning("rewrite markdown after delete failed: %s", e)
        return {"ok": True, "unlabeled_days": len(days)}

    @app.post("/api/speakers/calibrate")
    def speakers_calibrate() -> dict[str, Any]:
        from ..cli import calibrate_threshold

        embs = rt.db.speaker_embeddings()
        if len(embs) < 2:
            raise HTTPException(400, "至少需要 2 个注册样本才能校准（建议每人 2 个以上、至少 2 个人）")
        r = calibrate_threshold(embs)
        r["current"] = float(rt.cfg.speakers.identify_threshold)
        return r

    # ------------------------------------------------------------------ mixed utterances
    def _split_args(body: dict[str, Any]) -> tuple[list[float] | None, bool]:
        auto = bool(body.get("auto", False))
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

    def _split_processor(with_asr: bool) -> Any:
        from ..speech.pipeline import Processor

        if rt.speaker_models_missing():
            raise HTTPException(400, VOICE_MODELS_MISSING_MSG)
        client = rt.client() if with_asr else None          # JobError (409) without an API key, like /api/chat
        return Processor(rt.cfg, rt.db, client=client, engine=rt.engine())

    @app.get("/api/utterances/{uid}/preview-split")
    def utterance_preview_split(uid: int, auto: bool = True, at: str | None = None) -> dict[str, Any]:
        """Pieces an auto split would make (bounds + who was recognised in each), no ASR and no DB change.
        `at=1.8,4.2` previews a manual split instead."""
        if rt.db.one("SELECT id FROM utterances WHERE id=?", (int(uid),)) is None:
            raise HTTPException(404, "没有这条记录")
        pts: list[float] | None = None
        if at:
            try:
                pts = [float(x) for x in str(at).replace("，", ",").split(",") if x.strip()]
            except ValueError:
                raise HTTPException(400, "at 应为秒数列表")
        proc = _split_processor(with_asr=False)
        with rt.busy():
            try:
                return proc.preview_split(int(uid), at=pts, auto=auto if pts is None else False)
            except FileNotFoundError as e:
                raise HTTPException(404, str(e))
            except ValueError as e:
                raise HTTPException(400, str(e))

    @app.post("/api/utterances/{uid}/split")
    async def utterance_split(uid: int, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Split a mixed utterance: {"auto": true} (by voice) or {"at": [1.8, 4.2]} (seconds from its start).
        The pieces are transcribed again (needs the API key: 409 without one) and replace the row."""
        from ..llm.client import LLMError

        if rt.db.one("SELECT id FROM utterances WHERE id=?", (int(uid),)) is None:
            raise HTTPException(404, "没有这条记录")
        at, auto = _split_args(body)
        if not auto and not at:
            raise HTTPException(400, "需要给出拆分点（at）或选择自动拆分（auto）")
        proc = _split_processor(with_asr=True)
        sched_lock = rt.sched.lock

        def work() -> dict[str, Any]:
            with sched_lock:                     # never interleave with a running job (process / relabel) on the same rows
                return proc.split_utterance(int(uid), at=at, auto=auto)

        with rt.busy():
            try:
                res = await _to_thread(work)
            except FileNotFoundError as e:
                raise HTTPException(404, str(e))
            except ValueError as e:
                raise HTTPException(400, str(e))
            except LLMError as e:
                raise HTTPException(502, f"重新转写失败，没有修改：{e}")
        return {"ok": True, **res}

    # ------------------------------------------------------------------ memory chat
    @app.post("/api/chat")
    def chat(body: dict[str, Any] = Body(...)) -> StreamingResponse:
        question = str(body.get("question") or "").strip()
        if not question:
            raise HTTPException(400, "问题不能为空")
        history = [{"role": m["role"], "content": str(m.get("content") or "")}
                   for m in (body.get("history") or []) if m.get("role") in ("user", "assistant")][-20:]
        strategy = body.get("strategy") or None
        retrieval = body.get("retrieval") or None
        expand = body.get("expand")
        chat_obj = rt.chat(strategy, retrieval, None if expand is None else bool(expand))   # raises JobError without a key
        q: queue.Queue = queue.Queue()

        def on_tool(name: str, args: dict, result: str) -> None:
            q.put({"type": "tool", "name": name, "args": args, "result": (result or "")[:1500],
                   "lines": (result or "").count("\n") + 1})

        chat_obj.on_tool = on_tool

        def work() -> None:
            t0 = time.monotonic()
            try:
                answer = chat_obj.ask(question, history)
                stats = dict(chat_obj.last_stats)
                stats["seconds"] = round(time.monotonic() - t0, 1)
                q.put({"type": "answer", "text": answer, "stats": stats})
            except Exception as e:  # noqa: BLE001
                log.exception("chat failed: %s", e)
                q.put({"type": "error", "message": str(e)})
            finally:
                q.put(None)

        threading.Thread(target=work, name="chat", daemon=True).start()

        def gen():
            yield json.dumps({"type": "start", "strategy": chat_obj.strategy, "retrieval": chat_obj.retriever.mode,
                              "model": chat_obj.model}, ensure_ascii=False) + "\n"
            while True:
                try:
                    item = q.get(timeout=15.0)
                except queue.Empty:
                    yield json.dumps({"type": "ping"}) + "\n"
                    continue
                if item is None:
                    break
                yield json.dumps(item, ensure_ascii=False) + "\n"

        return StreamingResponse(gen(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------------ settings
    @app.get("/api/devices")
    def devices() -> list[dict[str, Any]]:
        try:
            from ..audio.capture import list_input_devices

            return [{"index": d.index, "name": d.name, "hostapi": d.hostapi, "channels": d.channels, "default_sr": d.default_sr}
                    for d in list_input_devices()]
        except Exception as e:  # noqa: BLE001
            log.warning("list devices failed: %s", e)
            return []

    @app.get("/api/devices/system")
    def system_devices() -> dict[str, Any]:
        try:
            from ..audio.loopback import available, list_loopback_devices

            ok = available()
            devs = list_loopback_devices() if ok else []
            return {"available": ok, "devices": [{"index": d.index, "name": d.name, "channels": d.channels,
                                                   "default_sr": d.default_sr, "is_default": d.is_default} for d in devs]}
        except Exception as e:  # noqa: BLE001
            log.warning("list system devices failed: %s", e)
            return {"available": False, "devices": [], "error": str(e)}

    @app.get("/api/config")
    def config_get() -> dict[str, Any]:
        cfg = rt.cfg
        values: dict[str, Any] = {}
        for section, keys in EDITABLE.items():
            sec = cfg.get(section) or {}
            # deep copy: the nested push.* dicts are the live config's own objects and get their secrets blanked below
            values[section] = copy.deepcopy({k: sec.get(k, DEFAULTS.get(section, {}).get(k)) for k in sorted(keys)})
        values["mimo"]["api_key"] = ""
        secrets: dict[str, Any] = {}
        for path in SECRET_KEYS:
            cur: Any = cfg
            for k in path:
                cur = (cur or {}).get(k) if isinstance(cur, dict) else None
            secrets[".".join(path)] = {"set": bool(cur), "hint": _hint(cur) if cur else ""}
            # never ship the secret itself to the page
            node = values
            for k in path[:-1]:
                node = node.setdefault(k, {}) if isinstance(node.get(k), dict) or k not in node else node[k]
            if isinstance(node, dict):
                node[path[-1]] = ""
        return {"path": str(rt.config_path), "exists": rt.config_path.exists(), "values": values, "secrets": secrets,
                "profile": rt.profile, "profile_label": profile_label(rt.profile), "base_config_path": cfg.get("_base_config_path"),
                "env_api_key": bool(os.environ.get("MIMO_API_KEY")), "defaults": {s: {k: DEFAULTS[s].get(k) for k in keys} for s, keys in EDITABLE.items()},
                "glossary_path": str(cfg.glossary_path), "data_dir": str(cfg.data_dir), "models_dir": str(cfg.models_dir)}

    @app.put("/api/config")
    def config_put(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        updates: dict[tuple[str, ...], Any] = {}
        if not isinstance(body, dict) or any(not isinstance(v, dict) for v in body.values()):
            raise HTTPException(400, "请求体应为 {小节: {键: 值}}")
        leaves: list[tuple[tuple[str, ...], Any]] = []
        for path, value in yaml_edit.flatten(body).items():
            if len(path) < 2:
                raise HTTPException(400, f"不允许修改 {'.'.join(path)}")
            section, key = path[0], path[1]
            if section not in EDITABLE or key not in EDITABLE[section]:
                raise HTTPException(400, f"不允许修改 {'.'.join(path)}")
            if isinstance(value, dict):
                if not isinstance(_default_for(path), dict):
                    raise HTTPException(400, f"{'.'.join(path)} 不是一组设置")
                for k2, v2 in value.items():
                    leaves.append((path + (str(k2),), v2))
            else:
                leaves.append((path, value))
        for path, value in leaves:
            if path in SECRET_KEYS:
                if value is None or str(value).strip() == "":
                    continue   # empty secret field = keep what is there
                if path == ("mimo", "api_key") and is_placeholder_key(value):
                    raise HTTPException(400, "API key 看起来是占位符")
                if isinstance(value, (dict, list, bool)):
                    raise HTTPException(400, f"{'.'.join(path)} 应为文本")
                updates[path] = str(value).strip()
                continue
            value = coerce_setting(path, value)
            if path == ("analysis", "daily_time"):
                from ..scheduler import parse_hhmm

                try:
                    parse_hhmm(str(value))
                except ValueError:
                    raise HTTPException(400, "日报时间格式应为 HH:MM")
            # unchanged values are not written: on a profile that would pin inherited settings into
            # profiles/<name>/config.yaml and silently cut them off from the shared root config.yaml
            if _effective(rt.cfg, path) == value:
                continue
            updates[path] = value
        if not updates:
            return {"ok": True, "changed": [], "notes": ["没有需要保存的修改"], "saved": []}
        p = rt.config_path
        text = p.read_text(encoding="utf-8") if p.exists() else "# audio_note 配置（由网页界面生成）\n"
        res = yaml_edit.set_values(text, updates)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(res.text, encoding="utf-8")
        os.replace(tmp, p)
        info = rt.reload_config()
        notes = list(info["notes"])
        if res.comments_lost:
            notes.append("config.yaml 里的注释无法保留，文件已按结构重写（值都在）")
        return {"ok": True, "changed": info["changed"], "restarted_recorder": info["restarted_recorder"], "notes": notes,
                "saved": sorted(".".join(k) for k in updates)}

    @app.post("/api/config/test-api")
    def config_test_api() -> dict[str, Any]:
        client = rt.client()
        t0 = time.monotonic()
        resp = client.chat([{"role": "user", "content": "回复“ok”两个字母即可。"}], max_tokens=8)
        text = (resp.choices[0].message.content or "").strip()
        return {"ok": True, "model": client.chat_model, "reply": text[:50], "seconds": round(time.monotonic() - t0, 2)}

    @app.post("/api/push/test")
    def push_test(body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        from ..push.base import Message, Notifier

        text = str(body.get("text") or "这是一条来自 audio-note 的测试消息。")
        n = Notifier(rt.cfg)
        if not n.channels:
            raise HTTPException(400, "没有配置可用的推送渠道（检查 push.channels 和对应的密钥）")
        results = n.send(Message("audio-note 测试", text))
        return {"ok": any(v is None for v in results.values()), "channels": results}

    @app.get("/api/glossary")
    def glossary_get() -> dict[str, Any]:
        p = rt.cfg.glossary_path
        return {"path": str(p), "text": p.read_text(encoding="utf-8") if p.exists() else ""}

    @app.put("/api/glossary")
    def glossary_put(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        p = rt.cfg.glossary_path
        text = str(body.get("text") or "")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
        terms = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        extra: dict[str, Any] = {}
        if body.get("reindex"):
            try:
                st = rt.submit("reindex", vectors=False)
                extra["job"] = st.as_dict()
            except (JobError, ReadOnly) as e:
                extra["job_error"] = str(e)
        return {"ok": True, "terms": len(terms), **extra}

    @app.get("/api/logs")
    def logs(lines: int = 200, which: str = "service") -> dict[str, Any]:
        from ..logging_setup import log_file_name

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

    @app.post("/api/import")
    async def import_audio(request: Request, filename: str = "import.wav", start: str | None = None) -> dict[str, Any]:
        from ..audio.importer import import_file
        from ..audio.vad import SileroVAD

        data = await request.body()
        if len(data) < 1000:
            raise HTTPException(400, "没有收到音频数据")
        safe = re.sub(r"[^\w.\-一-鿿]+", "_", Path(filename).name) or "import.wav"
        d = rt.cfg.data_dir / "imports"
        d.mkdir(parents=True, exist_ok=True)
        dest = d / safe
        k = 0
        while dest.exists():
            k += 1
            dest = d / f"{Path(safe).stem}_{k}{Path(safe).suffix}"
        await _to_thread(dest.write_bytes, data)
        start_dt = None
        if start:
            try:
                start_dt = datetime.fromisoformat(start)
            except ValueError:
                raise HTTPException(400, "开始时间格式应为 YYYY-MM-DD HH:MM")

        def run() -> list[int]:
            vad = SileroVAD(rt.cfg.model_path("vad_model"), threads=2)
            return import_file(rt.cfg, rt.db, dest, start_time=start_dt, vad=vad)

        ids = await _to_thread(run)
        return {"ok": True, "file": dest.name, "segments": len(ids)}

    # ------------------------------------------------------------------ static
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.middleware("http")
        async def _no_cache_static(request: Request, call_next):
            # the page is tiny and local: always revalidate so an upgrade never shows a stale app.js
            response = await call_next(request)
            if request.url.path.startswith("/static/"):
                response.headers["Cache-Control"] = "no-cache"
            return response

        @app.get("/")
        def index() -> Response:
            # the asset URLs carry the package version + file mtime, so an upgraded app.js / style.css is
            # never served from a browser cache that still holds the previous one
            try:
                from importlib.metadata import version as _pkg_version

                ver = _pkg_version("audio-note")
            except Exception:  # noqa: BLE001
                ver = "0"
            stamp = f"{ver}-{int(max((STATIC_DIR / n).stat().st_mtime for n in ('app.js', 'style.css')))}"
            html = (STATIC_DIR / "index.html").read_text(encoding="utf-8").replace("__V__", stamp)
            return Response(html, media_type="text/html", headers={"Cache-Control": "no-cache"})

    return app


async def _to_thread(fn, *args):
    import anyio

    return await anyio.to_thread.run_sync(fn, *args)
