"""Application runtime: the one object that owns the recorder, the scheduler, the background job
queue and the shared heavy components (speaker engine, MiMo client, retrievers). Both entry points
build on it:

    audio-note run   -> Runtime.start_service(record=True)  + web UI (service.run_service)
    audio-note ui    -> Runtime.start_service(record=False) + web UI, recording toggled from the page

"Service" = the single-instance lock + the scheduler loop (process / cleanup / daily), with the
recorder thread optional and switchable at run time. Only one process may run the service against
a data directory (service.lock); a second `audio-note ui` still works for browsing and memory chat
but cannot record or run jobs ("read-only mode").

Jobs (process / daily / cleanup / reindex / relabel / push / voices / relations) run one at a time on a worker thread
and under Scheduler.lock, so a job started from the UI never overlaps a scheduler tick. Their state
(running / last result / last error) is kept per job name for the UI to poll.
"""
from __future__ import annotations

import logging
import os
import queue
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .config import (DEFAULT_PROFILE, Config, canonical_profile_name, find_config_path, list_profiles, load_config,
                     profile_exists, set_current_profile)
from .db import DB
from .scheduler import KV_DAILY_AT, KV_DAILY_DAY, KV_PROCESS, Scheduler, next_daily_due
from .service import AlreadyRunning, _recorder_loop, acquire_lock, release_lock, running_pid

log = logging.getLogger(__name__)

JOBS = ("process", "daily", "cleanup", "reindex", "relabel", "push", "voices", "relations")
JOB_LABELS = {"process": "转写处理", "daily": "生成日报", "cleanup": "清理音频", "reindex": "重建索引",
              "relabel": "重新识别说话人", "push": "推送日报", "voices": "扫描未识别的声音", "relations": "推断关系"}
JOIN_TIMEOUT_S = 15.0
#: message for a speaker-embedding request / job when the CAM++ model file is absent
VOICE_MODELS_MISSING_MSG = "模型文件缺失，无法提取声纹（运行 scripts/download_models.ps1）"
BUSY_REQUEST_LABEL = "说话人请求（登记声纹 / 标记）"


class JobError(RuntimeError):
    """A job could not be started or failed with a user-facing reason."""


class ReadOnly(RuntimeError):
    """The service runs in another process: this one may read and chat but not record or run jobs."""

    def __init__(self, pid: int):
        super().__init__(f"服务正在另一个进程运行 (pid {pid})；请在那个进程的界面里操作，或先停止它")
        self.pid = pid


@dataclass
class JobState:
    name: str
    running: bool = False
    started_at: str | None = None
    finished_at: str | None = None
    result: Any = None
    error: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    runs: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "label": JOB_LABELS.get(self.name, self.name), "running": self.running,
                "started_at": self.started_at, "finished_at": self.finished_at, "result": self.result,
                "error": self.error, "args": self.args, "runs": self.runs}


class Runtime:
    def __init__(self, cfg: Config, db: DB | None = None, config_path: str | os.PathLike | None = None):
        self.cfg = cfg
        self.config_path = Path(config_path or cfg.get("_config_path") or find_config_path())
        cfg.ensure_dirs()
        self.db = db or DB(cfg.db_path)
        self.started_at = datetime.now()
        self.ui_url: str | None = None
        self.alive = True            # False after shutdown(): the object must not be used any more

        self._svc_stop = threading.Event()
        self._rec_stop = threading.Event()
        self._sched_thread: threading.Thread | None = None
        self._rec_thread: threading.Thread | None = None
        self._rec_ref: dict[str, Any] = {}
        self._lock_held = False
        self._state_lock = threading.RLock()
        # request threads doing heavy DB work (enrolment, naming a voice, labelling): see busy()
        self._busy_lock = threading.Lock()
        self._busy_n = 0
        self.switching = False       # True once RuntimeHolder.switch decided to tear this runtime down

        self.sched = Scheduler(cfg, self.db, self._svc_stop, engine_factory=self.engine)

        self._engine: Any = None
        self._client: Any = None
        self._retrievers: dict[tuple[str, bool], Any] = {}
        # re-entrant: retriever() calls client() while holding it (a plain Lock deadlocked with query expansion on)
        self._heavy_lock = threading.RLock()

        self.jobs: dict[str, JobState] = {n: JobState(n) for n in JOBS}
        self._job_q: queue.Queue = queue.Queue()
        self._job_thread = threading.Thread(target=self._job_loop, name="jobs", daemon=True)
        self._job_thread.start()

    # ------------------------------------------------------------------ service / recorder
    @property
    def recorder(self) -> Any | None:
        return self._rec_ref.get("recorder")

    def recording(self) -> bool:
        return self._rec_thread is not None and self._rec_thread.is_alive()

    def scheduler_running(self) -> bool:
        return self._sched_thread is not None and self._sched_thread.is_alive() and not self._svc_stop.is_set()

    def job_running(self) -> str | None:
        """Label of the UI job currently running, or None."""
        for j in self.jobs.values():
            if j.running:
                return j.label if hasattr(j, "label") else JOB_LABELS.get(j.name, j.name)
        return None

    def service_running(self) -> bool:
        return self._lock_held

    # ------------------------------------------------------------------ request-busy counter
    @contextmanager
    def busy(self):
        """Marks a request thread doing heavy work on this runtime's database (enrolment, naming a
        voice, labelling a sentence): RuntimeHolder.switch refuses while one is active, so the DB is
        not closed under it. Raises JobError once a switch has claimed the runtime."""
        with self._busy_lock:
            if self.switching or not self.alive:
                raise JobError("正在切换档案，请稍后再试")
            self._busy_n += 1
        try:
            yield self
        finally:
            with self._busy_lock:
                self._busy_n -= 1

    def request_busy(self) -> int:
        """Number of request threads currently inside busy()."""
        with self._busy_lock:
            return self._busy_n

    def begin_switch(self) -> None:
        """Claim the runtime for a profile switch: raises JobError while a request is busy, and makes
        later busy() calls fail instead of starting work on a database about to be closed."""
        with self._busy_lock:
            if self._busy_n > 0:
                raise JobError(f"{BUSY_REQUEST_LABEL}正在处理，等它结束再切换档案")
            self.switching = True

    def end_switch(self) -> None:
        with self._busy_lock:
            self.switching = False

    def busy_reason(self) -> str | None:
        """What blocks a profile switch right now: the running UI job's label, the busy request, or None."""
        running = self.job_running()
        if running:
            return running
        return BUSY_REQUEST_LABEL if self.request_busy() > 0 else None

    def speaker_models_missing(self) -> list[str]:
        """Model files the speaker engine needs that are absent ([] once the engine is loaded)."""
        if self._engine is not None:
            return []
        return [k for k in ("embedding_model",) if not self.cfg.model_path(k).exists()]

    def other_pid(self) -> int | None:
        """pid of a service running in another process (None when it is us or nobody)."""
        pid = running_pid(self.cfg)
        return pid if pid and pid != os.getpid() else None

    def start_service(self, record: bool = True, schedule: bool = True) -> None:
        """Acquire the single-instance lock and start the scheduler (+ recorder). Raises AlreadyRunning."""
        with self._state_lock:
            if not self._lock_held:
                acquire_lock(self.cfg)
                self._lock_held = True
            if schedule and not self.scheduler_running():
                if self._sched_thread is not None and self._sched_thread.is_alive():
                    raise JobError("后台处理还在停止中（上一轮任务没结束），稍后再试")
                if self._svc_stop.is_set():
                    self._svc_stop = threading.Event()
                    self.sched.reset_stop_event(self._svc_stop)
                self._sched_thread = threading.Thread(target=self.sched.run, name="scheduler", daemon=True)
                self._sched_thread.start()
            if record:
                self.start_recording()
            log.info("service started (pid %d, record=%s, schedule=%s)", os.getpid(), record, schedule)

    def start_recording(self) -> None:
        with self._state_lock:
            if self.recording():
                return
            if not self._lock_held:
                other = self.other_pid()
                if other:
                    raise ReadOnly(other)
                acquire_lock(self.cfg)
                self._lock_held = True
            self._rec_stop = threading.Event()
            self._rec_ref = {}
            self._rec_thread = threading.Thread(target=_recorder_loop, args=(self.cfg, self.db, self._rec_stop, self._rec_ref),
                                                name="recorder", daemon=True)
            self._rec_thread.start()
            log.info("recorder started")

    def stop_recording(self, timeout: float = JOIN_TIMEOUT_S) -> None:
        with self._state_lock:
            t = self._rec_thread
            self._rec_stop.set()
            if t is not None and t.is_alive():
                t.join(timeout)
                if t.is_alive():
                    log.warning("recorder thread did not stop in %.0fs", timeout)
            self._rec_thread = None
            log.info("recorder stopped")

    def stop_service(self, timeout: float = JOIN_TIMEOUT_S) -> None:
        with self._state_lock:
            self.stop_recording(timeout)
            self._svc_stop.set()
            t = self._sched_thread
            if t is not None and t.is_alive():
                t.join(timeout)
                if t.is_alive():
                    # keep the handle: start_service() must not start a second loop next to this one
                    log.warning("scheduler thread still finishing its tick after %.0fs", timeout)
            if t is None or not t.is_alive():
                self._sched_thread = None
            if self._lock_held:
                release_lock(self.cfg)
                self._lock_held = False
            log.info("service stopped")

    def shutdown(self, close_db: bool = True, timeout: float = JOIN_TIMEOUT_S) -> None:
        """Stop everything this runtime owns. With close_db the database connections of every thread are
        closed too (needed on Windows before the profile's files can be deleted or replaced). A job that
        is still running when `timeout` expires loses its connection and fails (the profile switch refuses
        to run while a job is active for that reason; Ctrl+C accepts it)."""
        self.stop_service(timeout)
        self._job_q.put(None)
        if self._job_thread.is_alive():
            self._job_thread.join(timeout)
        self.alive = False
        if close_db:
            n = self.db.close_all()
            log.debug("closed %d db connection(s)", n)

    def threads(self) -> list[threading.Thread]:
        return [t for t in (self._rec_thread, self._sched_thread) if t is not None]

    # ------------------------------------------------------------------ shared components
    def engine(self) -> Any:
        """Shared SpeakerEngine (CAM++ + pyannote models, ~1 s to load)."""
        with self._heavy_lock:
            if self._engine is None:
                from .speech.speakers import SpeakerEngine

                self._engine = SpeakerEngine(self.cfg)
            return self._engine

    def client(self) -> Any:
        """Shared MiMo client; raises JobError with a clear message when no key is configured."""
        with self._heavy_lock:
            if self._client is None:
                from .llm.client import MimoClient, MissingAPIKey

                try:
                    self._client = MimoClient(self.cfg, rpm=int(self.cfg.mimo.rpm))
                except MissingAPIKey as e:
                    raise JobError("没有配置 MiMo API key：在「设置」里填写，或设置环境变量 MIMO_API_KEY") from e
            return self._client

    def retriever(self, mode: str | None = None, expand: bool | None = None) -> Any:
        from .memory.retrieval import Retriever

        key = ((mode or str(self.cfg.memory.retrieval)).lower(), bool(self.cfg.memory.get("query_expansion", False)) if expand is None else bool(expand))
        with self._heavy_lock:
            r = self._retrievers.get(key)
            if r is None:
                client = None
                if key[1]:
                    client = self.client()
                r = Retriever(self.cfg, self.db, client, mode=key[0], expand=key[1])
                self._retrievers[key] = r
            return r

    def chat(self, strategy: str | None = None, retrieval: str | None = None, expand: bool | None = None) -> Any:
        from .memory.chat import MemoryChat

        client = self.client()
        return MemoryChat(self.cfg, self.db, client, retriever=self.retriever(retrieval, expand), strategy=strategy)

    # ------------------------------------------------------------------ config
    @property
    def profile(self) -> str | None:
        """Profile name this runtime was loaded for (None = a custom --config / AUDIO_NOTE_CONFIG file)."""
        return self.cfg.get("_profile")

    def reload_config(self) -> dict[str, Any]:
        """Re-read config.yaml and hand the new values to every component. Returns what changed."""
        old = self.cfg
        cfg = load_config(self.config_path, base=old.get("_base_config_path"))
        cfg["_profile"] = old.get("_profile")
        changed = sorted(k for k in set(old) | set(cfg) if not str(k).startswith("_") and old.get(k) != cfg.get(k))
        cfg.ensure_dirs()
        self.cfg = cfg
        with self._heavy_lock:
            self._client = None
            self._retrievers.clear()
            if self._engine is not None:
                self._engine.cfg = cfg
        self.sched.reload_config(cfg)
        notes: list[str] = []
        restarted = False
        if "audio" in changed and self.recording():
            self.stop_recording()
            self.start_recording()
            restarted = True
            notes.append("录音已按新的音频设置重启")
        if "paths" in changed:
            notes.append("数据目录的修改需要重启程序才会生效")
        if "ui" in changed:
            notes.append("界面地址/端口的修改需要重启程序才会生效")
        if "log" in changed:
            notes.append("日志设置的修改需要重启程序才会生效")
        log.info("config reloaded: changed sections %s", changed)
        return {"changed": changed, "restarted_recorder": restarted, "notes": notes}

    # ------------------------------------------------------------------ jobs
    def submit(self, name: str, **kwargs: Any) -> JobState:
        """Queue a job; raises JobError when it is unknown / already running / another process owns the service."""
        if name not in self.jobs:
            raise JobError(f"未知任务 {name!r}")
        other = self.other_pid()
        if other:
            raise ReadOnly(other)
        st = self.jobs[name]
        with self._state_lock:
            if st.running:
                raise JobError(f"{JOB_LABELS.get(name, name)}已在运行中")
            st.running = True
            st.started_at = datetime.now().isoformat(timespec="seconds")
            st.finished_at = None
            st.error = None
            st.result = None
            st.args = {k: v for k, v in kwargs.items() if v is not None}
        self._job_q.put((name, kwargs))
        return st

    def run_job(self, name: str, **kwargs: Any) -> Any:
        """Run a job synchronously on the calling thread (CLI / tests). Raises on failure."""
        fn = self._job_fn(name)
        with self.sched.lock:
            return fn(**kwargs)

    def wait_jobs(self, timeout: float = 60.0) -> bool:
        """Block until no job is running (tests). Returns False on timeout."""
        deadline = time.monotonic() + timeout
        while any(j.running for j in self.jobs.values()) or not self._job_q.empty():
            if time.monotonic() > deadline:
                return False
            time.sleep(0.02)
        return True

    def _job_loop(self) -> None:
        while True:
            item = self._job_q.get()
            if item is None:
                break
            name, kwargs = item
            st = self.jobs[name]
            t0 = time.monotonic()
            try:
                fn = self._job_fn(name)
                with self.sched.lock:
                    res = fn(**kwargs)
                st.result = _jsonable(res)
                st.runs += 1
                log.info("job %s (ui) done in %.1fs: %s", name, time.monotonic() - t0, st.result)
            except Exception as e:  # noqa: BLE001
                st.error = str(e) if isinstance(e, (JobError, ReadOnly)) else f"{type(e).__name__}: {e}"
                log.exception("job %s (ui) failed: %s", name, e)
            finally:
                st.running = False
                st.finished_at = datetime.now().isoformat(timespec="seconds")

    def _job_fn(self, name: str) -> Callable[..., Any]:
        fn = getattr(self, f"_job_{name}", None)
        if fn is None:
            raise JobError(f"未知任务 {name!r}")
        return fn

    def _job_process(self, retry_failed: bool = False, limit: int | None = None) -> dict[str, Any]:
        proc = self.sched.processor()
        if proc is None:
            raise JobError("没有配置 MiMo API key，无法转写：在「设置」里填写后再试")
        stats = proc.process_pending(limit=limit, retry_failed=bool(retry_failed))
        self.db.set_kv(KV_PROCESS, datetime.now().isoformat(timespec="seconds"))
        return dict(stats)

    def _job_daily(self, day: str | None = None, force: bool = False, push: bool = False) -> dict[str, Any]:
        from .analysis.daily import DailyAnalyzer

        day = day or datetime.now().strftime("%Y-%m-%d")
        analyzer = DailyAnalyzer(self.cfg, self.db, self.client())
        md, js = analyzer.run(day, force=bool(force))
        out: dict[str, Any] = {"day": day, "utterances": int(js.get("utterance_count") or 0),
                               "cached": analyzer.last_run_cached, "partial": analyzer.last_run_partial, "chars": len(md)}
        try:
            from .analysis import relations

            out["relations"] = relations.maybe_infer_after_daily(self.cfg, self.db, self.client())
        except Exception as e:  # noqa: BLE001 - relation inference never fails the digest
            log.warning("relation inference after daily %s failed: %s", day, e)
            out["relations"] = []
        if push:
            out["push"] = analyzer.push(day)
        return out

    def _job_voices(self) -> dict[str, Any]:
        """Compute the missing cluster embeddings so the speakers page can list unidentified voices."""
        from .speech import voices

        if self.speaker_models_missing():
            raise JobError(VOICE_MODELS_MISSING_MSG)
        stop = self._svc_stop if self.scheduler_running() else None
        return dict(voices.backfill_cluster_embeddings(self.db, self.cfg, self.engine(), stop_event=stop))

    def _job_relations(self, speaker_id: int | None = None, force: bool = False) -> dict[str, Any]:
        """Ask the LLM how the enrolled speakers relate to the user (one speaker, or every candidate)."""
        from .analysis import relations

        ids = [int(speaker_id)] if speaker_id else None
        results = list(relations.infer_relations(self.cfg, self.db, self.client(), speaker_ids=ids, force=bool(force)))
        return {"speakers": len(results), "applied": sum(1 for r in results if r.get("applied")), "results": results}

    def _job_push(self, day: str) -> dict[str, Any]:
        from .analysis.daily import DailyAnalyzer

        row = self.db.one("SELECT day FROM daily_summaries WHERE day=?", (day,))
        if row is None:
            raise JobError(f"{day} 还没有日报，先生成再推送")
        results = DailyAnalyzer(self.cfg, self.db, None).push(day)
        if not results:
            raise JobError("没有配置任何推送渠道（设置 → 推送）")
        if all(v for v in results.values()):
            raise JobError("所有渠道推送失败：" + "；".join(f"{k}: {v}" for k, v in results.items()))
        return {"day": day, "channels": results}

    def _job_cleanup(self) -> dict[str, Any]:
        return dict(self.sched._default_cleanup() or {})

    def _job_reindex(self, vectors: bool = True) -> dict[str, Any]:
        from .memory import index as mem_index

        mem_index.register_from(self.cfg, self.db)
        n = mem_index.reindex(self.db)
        out: dict[str, Any] = {"keywords": n}
        if vectors and self.cfg.memory.retrieval != "fts":
            from .memory import vectors as mem_vectors
            from .memory.embed import get_embedder

            emb = get_embedder(self.cfg)
            out["vectors_added"] = mem_vectors.index_missing(self.db, emb)
            out["vectors"] = self.db.vector_count(emb.name)
        return out

    def _job_relabel(self, day: str | None = None, segment_id: int | None = None) -> dict[str, Any]:
        from .speech.pipeline import Processor

        if not self.db.speaker_embeddings():
            raise JobError("还没有注册任何说话人")
        proc = Processor(self.cfg, self.db, engine=self.engine())
        return dict(proc.relabel(day=day, segment_id=segment_id))

    # ------------------------------------------------------------------ status
    def status(self) -> dict[str, Any]:
        from .audio.storage import dir_size_bytes

        cfg, db = self.cfg, self.db
        today = datetime.now().strftime("%Y-%m-%d")
        rec = self.recorder
        other = self.other_pid()
        seg_rows = db.query("SELECT status, COUNT(*) AS n, COALESCE(SUM(duration_s),0) AS dur FROM segments GROUP BY status")
        segments = {r["status"]: {"n": int(r["n"]), "minutes": round(float(r["dur"]) / 60.0, 1)} for r in seg_rows}
        utt_today = db.one("SELECT COUNT(*) AS n, COALESCE(SUM(end_s-start_s),0) AS s FROM utterances WHERE day=?", (today,))
        seg_today = db.one("SELECT COUNT(*) AS n, COALESCE(SUM(speech_s),0) AS s FROM segments WHERE started_at >= ? AND status != 'deleted'",
                           (f"{today}T00:00:00",))
        utt_all = db.one("SELECT COUNT(*) AS n, COUNT(DISTINCT day) AS days FROM utterances")
        last = db.one("SELECT day, created_at, pushed_at, utterance_count FROM daily_summaries ORDER BY day DESC LIMIT 1")
        audio_bytes = dir_size_bytes(cfg.audio_dir) if cfg.audio_dir.exists() else 0
        try:
            free = shutil.disk_usage(cfg.data_dir).free
        except OSError:
            free = None
        try:
            next_daily = next_daily_due(datetime.now(), str(cfg.analysis.daily_time)).isoformat(timespec="minutes")
        except ValueError:
            next_daily = None
        models_missing = [k for k in ("vad_model", "embedding_model", "segmentation_model") if not cfg.model_path(k).exists()]
        me = db.self_speaker()
        unknown = db.one("SELECT COUNT(*) AS n FROM utterances WHERE speaker_id IS NULL")
        return {
            "pid": os.getpid(),
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "now": datetime.now().isoformat(timespec="seconds"),
            "config_path": str(self.config_path),
            "profile": {"name": self.profile, "label": profile_label(self.profile),
                        "base_config_path": cfg.get("_base_config_path"), "profiles": list_profiles()},
            "data_dir": str(cfg.data_dir),
            "api_key": bool(cfg.mimo.api_key),
            "models_missing": models_missing,
            "ui_url": self.ui_url,
            "service": {
                "running": self.service_running(),
                "read_only": bool(other),
                "other_pid": other,
                "scheduler": self.scheduler_running(),
                "recording": self.recording(),
                "is_speaking": bool(getattr(rec, "is_speaking", False)) if rec else False,
                "segments_written": int(getattr(rec, "segments_written", 0)) if rec else 0,
                "speech_total_s": round(float(getattr(rec, "speech_total_s", 0.0)), 1) if rec else 0.0,
                "recorder_started_at": rec.started_at.isoformat(timespec="seconds") if rec and getattr(rec, "started_at", None) else None,
                "last_error": self._rec_ref.get("last_error"),
                "device": cfg.audio.device,
                "system_audio": bool(cfg.audio.get("system_audio", False)),
                "system_device": cfg.audio.get("system_device"),
                "system_active": bool(getattr(getattr(rec, "capture", None), "system_active", False)) if rec else False,
                "system_name": getattr(getattr(rec, "capture", None), "system_name", None) if rec else None,
                "system_error": getattr(getattr(rec, "capture", None), "system_error", None) if rec else None,
            },
            "segments": segments,
            "pending": segments.get("new", {}).get("n", 0) + segments.get("processing", {}).get("n", 0),
            "failed": segments.get("failed", {}).get("n", 0),
            "today": {"day": today, "utterances": int(utt_today["n"]) if utt_today else 0,
                      "speech_min": round(float(utt_today["s"]) / 60.0, 1) if utt_today else 0.0,
                      "segments": int(seg_today["n"]) if seg_today else 0,
                      "recorded_speech_min": round(float(seg_today["s"]) / 60.0, 1) if seg_today else 0.0},
            "total": {"utterances": int(utt_all["n"]) if utt_all else 0, "days": int(utt_all["days"]) if utt_all else 0},
            "self_name": me["name"] if me else None,
            "unknown_utterances": int(unknown["n"]) if unknown else 0,
            "disk": {"audio_bytes": int(audio_bytes), "cap_gb": float(cfg.storage.max_disk_gb or 0), "free_bytes": free},
            "last_daily": dict(last) if last else None,
            "next_daily": next_daily,
            "daily_time": str(cfg.analysis.daily_time),
            "process_interval_min": cfg.pipeline.process_interval_min,
            "last_process_at": db.get_kv(KV_PROCESS),
            "last_daily_at": db.get_kv(KV_DAILY_AT),
            "last_daily_day": db.get_kv(KV_DAILY_DAY),
            "scheduler_runs": dict(self.sched.runs),
            "scheduler_errors": dict(self.sched.errors),
            "jobs": {n: j.as_dict() for n, j in self.jobs.items()},
        }


def profile_label(name: str | None) -> str:
    if name is None:
        return "自定义配置"
    return "默认" if name == DEFAULT_PROFILE else name


class RuntimeHolder:
    """Mutable pointer to the active Runtime, so the web UI can switch profiles without restarting the
    process: the old runtime is shut down (recorder, scheduler, lock, job thread), a new one is built
    for the other profile's config/data and started the same way (recording resumes if it was on)."""

    def __init__(self, rt: Runtime):
        self.rt = rt
        self.lock = threading.Lock()

    def switch(self, name: str) -> Runtime:
        """Switch the live runtime to profile `name`. The new Runtime (dirs, database, job thread) is
        built first; only once that succeeded is the old one torn down, so a profile whose data dir
        cannot be created or whose database cannot be opened leaves the current one untouched.
        Refuses while a UI job is running or a request thread is inside Runtime.busy() (their database
        connection would be closed under them)."""
        with self.lock:
            name = canonical_profile_name(name) or name
            if not profile_exists(name):
                raise FileNotFoundError(f"档案不存在: {name}")
            old = self.rt
            if old.profile == name and old.alive:
                return old
            running = old.job_running() if old.alive else None
            if running:
                raise JobError(f"{running}正在运行，等它结束再切换档案")
            if old.alive:
                old.begin_switch()      # JobError while a request is busy; new requests are refused from here on
            try:
                was_recording = old.recording()
                cfg = load_config(profile=name)
                new = Runtime(cfg)          # may raise: the old runtime is still intact
            except BaseException:
                old.end_switch()
                raise
            new.ui_url = old.ui_url
            old.shutdown()
            self.rt = new
            try:
                set_current_profile(name)
            except Exception as e:  # noqa: BLE001
                log.warning("cannot remember profile %s: %s", name, e)
            try:
                from .logging_setup import setup_logging

                setup_logging(cfg, role="service")
            except Exception as e:  # noqa: BLE001
                log.warning("logging re-setup for profile %s failed: %s", name, e)
            try:
                new.start_service(record=was_recording)
            except AlreadyRunning as e:
                log.warning("profile %s: %s (read-only here)", name, e)
            except Exception as e:  # noqa: BLE001  (e.g. microphone gone): the runtime still serves the UI
                log.exception("profile %s: service did not start: %s", name, e)
            log.info("switched to profile %s (data %s)", name, cfg.data_dir)
            return new


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, Path):
        return str(v)
    return str(v)


__all__ = ["Runtime", "RuntimeHolder", "JobState", "JobError", "ReadOnly", "AlreadyRunning", "JOBS", "JOB_LABELS",
           "profile_label", "VOICE_MODELS_MISSING_MSG", "BUSY_REQUEST_LABEL"]
