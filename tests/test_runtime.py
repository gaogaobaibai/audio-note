"""audio_note.runtime.Runtime: service lock, job queue, shared components, config reload; plus the
Scheduler additions it relies on (lock, processor(), reset_stop_event(), reload_config()).

No network (mimo.api_key is "" so client() raises JobError; MimoClient is replaced by a fake where
a client object is needed), no microphone (the recorder loop is replaced by a fake thread body).
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
import types
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from audio_note import runtime as rt_mod
from audio_note import scheduler as sch
from audio_note import service
from audio_note.db import DB
from audio_note.runtime import JOBS, AlreadyRunning, JobError, JobState, ReadOnly, Runtime


class FakeClient:
    """Stands in for MimoClient: any call means a test tried to use the network."""

    chat_model = "fake-model"
    usage = {"calls": 0, "audio_seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0}

    def __init__(self, cfg=None, rpm=None):
        self.cfg, self.rpm = cfg, rpm

    def chat(self, *a, **k): raise AssertionError("no network")
    def text(self, *a, **k): raise AssertionError("no network")
    def json(self, *a, **k): raise AssertionError("no network")
    def transcribe_bytes(self, *a, **k): raise AssertionError("no network")


@pytest.fixture
def rt_factory(cfg_factory):
    """Runtime over a fresh temp config; every instance is shut down afterwards."""
    made: list[Runtime] = []

    def make(overrides: dict | None = None) -> Runtime:
        cfg = cfg_factory({"memory": {"retrieval": "fts", "embedding_model": "hash"}, **(overrides or {})})
        r = Runtime(cfg)
        made.append(r)
        return r

    yield make
    for r in made:
        r.shutdown()
    from audio_note.logging_setup import _remove_own_handlers

    _remove_own_handlers(logging.getLogger())


@pytest.fixture
def rt(rt_factory) -> Runtime:
    return rt_factory()


def _fake_recorder_loop(cfg, db, stop_event, ref):
    ref["recorder"] = SimpleNamespace(is_speaking=False, segments_written=1, speech_total_s=2.0, started_at=datetime.now())
    stop_event.wait(30)


def _rewrite_config(rt: Runtime, **sections: dict) -> None:
    """Edit the yaml the fixture wrote (keeps paths/data_dir), the way the settings page would."""
    data = yaml.safe_load(rt.config_path.read_text(encoding="utf-8")) or {}
    for k, v in sections.items():
        data.setdefault(k, {}).update(v)
    rt.config_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")


# --- status ---------------------------------------------------------------------------------------

def test_status_shape_on_empty_db(rt: Runtime):
    st = rt.status()
    assert st["pid"] == os.getpid() and st["config_path"] == str(rt.config_path)
    assert st["data_dir"] == str(rt.cfg.data_dir) and st["api_key"] is False and st["ui_url"] is None
    assert st["models_missing"] == []                                    # the real models/ dir is configured
    svc = st["service"]
    assert svc == {"running": False, "read_only": False, "other_pid": None, "scheduler": False, "recording": False,
                   "is_speaking": False, "segments_written": 0, "speech_total_s": 0.0, "recorder_started_at": None,
                   "last_error": None, "device": None, "system_audio": False, "system_device": None,
                   "system_active": False, "system_name": None, "system_error": None}
    assert st["segments"] == {} and st["pending"] == 0 and st["failed"] == 0
    assert st["today"] == {"day": datetime.now().strftime("%Y-%m-%d"), "utterances": 0, "speech_min": 0.0,
                           "segments": 0, "recorded_speech_min": 0.0}
    assert st["total"] == {"utterances": 0, "days": 0}
    assert st["disk"]["audio_bytes"] == 0 and st["disk"]["cap_gb"] == 20.0 and st["disk"]["free_bytes"] > 0
    assert st["last_daily"] is None and st["daily_time"] == "22:30" and st["next_daily"]
    assert st["last_process_at"] is None and st["last_daily_at"] is None and st["last_daily_day"] is None
    assert st["scheduler_runs"] == {"process": 0, "cleanup": 0, "daily": 0} and st["scheduler_errors"] == st["scheduler_runs"]
    assert set(st["jobs"]) == set(JOBS)
    assert st["jobs"]["daily"] == {"name": "daily", "label": "生成日报", "running": False, "started_at": None, "finished_at": None,
                                   "result": None, "error": None, "args": {}, "runs": 0}
    datetime.fromisoformat(st["started_at"]); datetime.fromisoformat(st["now"])


def test_status_counts_segments_utterances_and_daily(rt: Runtime):
    db = rt.db
    today = datetime.now().replace(microsecond=0)
    done = db.add_segment(rt.cfg.audio_dir / "a.opus", today, 10.0, speech_s=6.0)
    db.set_segment_status(done, "done")
    db.add_segment(rt.cfg.audio_dir / "b.opus", today, 4.0, speech_s=3.0)                     # new -> pending
    failed = db.add_segment(rt.cfg.audio_dir / "c.opus", today, 4.0, speech_s=1.0)
    db.set_segment_status(failed, "failed", "boom")
    day = today.strftime("%Y-%m-%d")
    db.add_utterances([
        {"segment_id": done, "start_s": 0, "end_s": 30, "abs_start": today.isoformat(), "day": day, "text_raw": "a", "text": "a"},
        {"segment_id": done, "start_s": 0, "end_s": 1, "abs_start": "2026-10-01T10:00:00", "day": "2026-10-01", "text_raw": "b", "text": "b"},
    ])
    db.execute("INSERT INTO daily_summaries(day, summary_md, created_at, utterance_count) VALUES (?,?,?,?)", ("2026-10-01", "# x", "2026-10-01 22:30:00", 1))
    db.set_kv(sch.KV_PROCESS, "2026-10-31T10:00:00")
    (rt.cfg.audio_dir / "blob.bin").write_bytes(b"\0" * 4096)
    st = rt.status()
    assert st["segments"] == {"done": {"n": 1, "minutes": 0.2}, "new": {"n": 1, "minutes": 0.1}, "failed": {"n": 1, "minutes": 0.1}}
    assert st["pending"] == 1 and st["failed"] == 1
    assert st["today"]["utterances"] == 1 and st["today"]["speech_min"] == 0.5
    assert st["today"]["segments"] == 3 and st["today"]["recorded_speech_min"] == round(10.0 / 60, 1)
    assert st["total"] == {"utterances": 2, "days": 2}
    assert st["last_daily"]["day"] == "2026-10-01" and st["last_daily"]["utterance_count"] == 1
    assert st["disk"]["audio_bytes"] == 4096 and st["last_process_at"] == "2026-10-31T10:00:00"


# --- service / lock ---------------------------------------------------------------------------------

def test_start_and_stop_service_acquire_and_release_the_lock(rt: Runtime):
    assert service.running_pid(rt.cfg) is None and not rt.service_running()
    rt.start_service(record=False)
    assert rt.service_running() and rt.scheduler_running() and not rt.recording()
    assert service.running_pid(rt.cfg) == os.getpid() and service.lock_path(rt.cfg).exists()
    assert rt.status()["service"]["running"] is True and rt.status()["service"]["scheduler"] is True
    assert rt.other_pid() is None                                        # the lock holder is us
    rt.start_service(record=False)                                       # idempotent
    rt.stop_service()
    assert not rt.service_running() and not rt.scheduler_running()
    assert service.running_pid(rt.cfg) is None and not service.lock_path(rt.cfg).exists()
    assert rt.threads() == []
    rt.stop_service()                                                    # a second stop is a no-op


def test_restart_in_same_process_uses_a_fresh_stop_event(rt: Runtime):
    rt.start_service(record=False)
    first_event = rt._svc_stop
    rt.sched._processor = SimpleNamespace(stop_event=first_event)
    rt.stop_service()
    assert first_event.is_set()
    rt.start_service(record=False)
    assert rt.scheduler_running()
    assert rt._svc_stop is not first_event and not rt._svc_stop.is_set()
    assert rt.sched.stop_event is rt._svc_stop                            # reset_stop_event() was called ...
    assert rt.sched._processor is None                                    # ... and the old processor (old event) was dropped


def test_other_process_holding_the_lock_means_read_only(rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    fake_pid = 999_999
    assert fake_pid != os.getpid()
    service.lock_path(rt.cfg).write_text(f"{fake_pid}\n2026-10-31T09:00:00\n1.0\n", encoding="utf-8")
    monkeypatch.setattr(service, "_same_process", lambda pid, ctime: pid == fake_pid)
    assert rt.other_pid() == fake_pid
    st = rt.status()["service"]
    assert st["read_only"] is True and st["other_pid"] == fake_pid and st["running"] is False
    with pytest.raises(AlreadyRunning) as ex:
        rt.start_service(record=False)
    assert ex.value.pid == fake_pid and not rt.service_running() and not rt.scheduler_running()
    with pytest.raises(ReadOnly) as ex:
        rt.submit("cleanup")
    assert ex.value.pid == fake_pid and str(fake_pid) in str(ex.value)
    with pytest.raises(ReadOnly):
        rt.start_recording()
    assert not rt.recording()
    # the lock file of the other process is untouched
    assert service.read_lock(rt.cfg) == fake_pid


def test_start_recording_alone_takes_the_lock_and_stop_releases_the_thread(rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rt_mod, "_recorder_loop", _fake_recorder_loop)
    rt.start_recording()
    assert rt.recording() and rt.service_running() and not rt.scheduler_running()
    assert service.running_pid(rt.cfg) == os.getpid()
    time.sleep(0.05)
    st = rt.status()["service"]
    assert st["recording"] is True and st["segments_written"] == 1 and st["speech_total_s"] == 2.0
    assert st["recorder_started_at"] and st["is_speaking"] is False
    rt.start_recording()                                                 # already running: no second thread
    assert len([t for t in threading.enumerate() if t.name == "recorder"]) == 1
    t0 = time.monotonic()
    rt.stop_recording()
    assert time.monotonic() - t0 < 5.0 and not rt.recording()
    assert rt.service_running()                                          # the lock stays until stop_service()
    rt.stop_service()
    assert service.running_pid(rt.cfg) is None


# --- jobs -----------------------------------------------------------------------------------------

def test_submit_rejects_unknown_job(rt: Runtime):
    with pytest.raises(JobError, match="未知任务"):
        rt.submit("frobnicate")
    with pytest.raises(JobError, match="未知任务"):
        rt.run_job("frobnicate")


def test_submit_rejects_a_job_that_is_already_running(rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    release = threading.Event()
    started = threading.Event()

    def slow():
        started.set()
        release.wait(10)
        return {"expired": 0}

    monkeypatch.setattr(rt, "_job_cleanup", slow)
    st = rt.submit("cleanup")
    assert isinstance(st, JobState) and st.running and st.started_at and st.finished_at is None
    assert started.wait(5)
    with pytest.raises(JobError, match="清理音频已在运行中"):
        rt.submit("cleanup")
    assert rt.wait_jobs(timeout=0.2) is False                            # still blocked
    release.set()
    assert rt.wait_jobs(timeout=5) is True
    assert not st.running and st.result == {"expired": 0} and st.error is None and st.runs == 1
    assert st.finished_at and rt.jobs["cleanup"] is st
    # finished: it may be submitted again
    release.set()
    assert rt.submit("cleanup").running and rt.wait_jobs(5) and st.runs == 2


def test_job_results_and_errors_are_recorded(rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    seen: list = []

    def relabel(day=None, segment_id=None):
        seen.append((day, segment_id))
        return {"segments": 2, "path": Path("x/y.opus"), "days": {"2026-10-30"}, "when": datetime(2026, 10, 30)}

    monkeypatch.setattr(rt, "_job_relabel", relabel)
    st = rt.submit("relabel", day="2026-10-30", segment_id=None)
    assert st.args == {"day": "2026-10-30"}                               # None arguments are not shown
    assert rt.wait_jobs(5)
    assert seen == [("2026-10-30", None)]
    assert st.result == {"segments": 2, "path": str(Path("x/y.opus")), "days": ["2026-10-30"], "when": "2026-10-30 00:00:00"}
    assert st.as_dict()["label"] == "重新识别说话人" and st.as_dict()["runs"] == 1

    monkeypatch.setattr(rt, "_job_push", lambda day: (_ for _ in ()).throw(JobError("没有日报")))
    st = rt.submit("push", day="2026-10-30")
    assert rt.wait_jobs(5)
    assert st.error == "没有日报" and st.result is None and st.runs == 0 and not st.running

    def boom(**kw):
        raise ValueError("kaboom")

    monkeypatch.setattr(rt, "_job_reindex", boom)
    st = rt.submit("reindex", vectors=False)
    assert rt.wait_jobs(5)
    assert st.error == "ValueError: kaboom" and st.finished_at
    # a later success clears the error
    monkeypatch.setattr(rt, "_job_reindex", lambda vectors=True: {"keywords": 0})
    st = rt.submit("reindex")
    assert rt.wait_jobs(5) and st.error is None and st.result == {"keywords": 0} and st.runs == 1


def test_run_job_cleanup_is_synchronous_and_holds_the_scheduler_lock(rt: Runtime):
    res = rt.run_job("cleanup")                                          # real cleanup over the empty data dir (real VAD)
    assert res == {"rechecked": 0, "no_speech": 0, "expired": 0, "disk_cap": 0}
    assert rt.jobs["cleanup"].runs == 0                                  # run_job bypasses the JobState bookkeeping
    assert not rt.sched.lock._is_owned()

    def check_lock():
        assert rt.sched.lock._is_owned()
        return {"ok": True}

    rt._job_cleanup = check_lock
    assert rt.run_job("cleanup") == {"ok": True}


def test_scheduler_lock_is_held_during_run_once_and_by_queued_jobs(rt: Runtime):
    owned: list[bool] = []
    s = sch.Scheduler(rt.cfg, rt.db, threading.Event(), jobs={
        "process": lambda: owned.append(s.lock._is_owned()),
        "cleanup": lambda: owned.append(s.lock._is_owned()),
        "daily": lambda day, push: None, "push": lambda day: {}})
    assert s.run_once(datetime(2026, 10, 31, 10, 0)) == ["process", "cleanup"]
    assert owned == [True, True] and not s.lock._is_owned()
    # the Runtime's worker thread takes the same lock: a tick cannot overlap a UI-triggered job
    acquired = threading.Event()
    release = threading.Event()

    def hold():
        acquired.set()
        release.wait(10)
        return {}

    rt._job_cleanup = hold
    rt.submit("cleanup")
    assert acquired.wait(5)
    assert rt.sched.lock.acquire(timeout=0.2) is False
    release.set()
    assert rt.wait_jobs(5)
    assert rt.sched.lock.acquire(timeout=1.0) is True
    rt.sched.lock.release()


def test_process_job_needs_a_key(rt: Runtime):
    with pytest.raises(JobError, match="MiMo API key"):
        rt.run_job("process")
    assert rt.db.get_kv(sch.KV_PROCESS) is None


def test_process_job_runs_the_processor_and_stamps_kv(rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    calls: list = []

    class Proc:
        def process_pending(self, limit=None, retry_failed=False):
            calls.append((limit, retry_failed))
            return {"processed": 1, "done": 1}

    monkeypatch.setattr(rt.sched, "processor", lambda: Proc())
    assert rt.run_job("process", retry_failed=True, limit=5) == {"processed": 1, "done": 1}
    assert calls == [(5, True)]
    assert rt.db.get_kv(sch.KV_PROCESS)


def test_push_job_errors_and_success(rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(JobError, match="还没有日报"):
        rt.run_job("push", day="2026-10-30")
    rt.db.execute("INSERT INTO daily_summaries(day, summary_md, utterance_count) VALUES (?,?,?)", ("2026-10-30", "# 日报", 3))
    results: dict = {}
    sent: list = []

    class Message:
        def __init__(self, title, body_md, url=None):
            self.title, self.body_md = title, body_md

    class Notifier:
        def __init__(self, cfg): ...

        def send(self, msg):
            sent.append(msg)
            return dict(results)

    mod = types.ModuleType("audio_note.push.base")
    mod.Message, mod.Notifier = Message, Notifier
    monkeypatch.setitem(sys.modules, "audio_note.push.base", mod)
    with pytest.raises(JobError, match="没有配置任何推送渠道"):
        rt.run_job("push", day="2026-10-30")
    results.update({"toast": "no toast", "feishu": "HTTP 500"})
    with pytest.raises(JobError, match="所有渠道推送失败.*feishu: HTTP 500"):
        rt.run_job("push", day="2026-10-30")
    assert rt.db.one("SELECT pushed_at FROM daily_summaries")["pushed_at"] is None
    results["toast"] = None
    assert rt.run_job("push", day="2026-10-30") == {"day": "2026-10-30", "channels": {"toast": None, "feishu": "HTTP 500"}}
    assert sent[-1].body_md == "# 日报" and rt.db.one("SELECT pushed_at FROM daily_summaries")["pushed_at"]
    assert rt.cfg.mimo.api_key == ""                                     # no LLM client was needed


def test_daily_job_with_fake_analyzer(rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    import audio_note.analysis.daily as daily_mod

    runs: list = []
    pushes: list = []

    class FakeAnalyzer:
        def __init__(self, cfg, db, client):
            assert isinstance(client, FakeClient)
            self.last_run_cached = False
            self.last_run_partial = False

        def run(self, day, force=False):
            runs.append((day, force))
            self.last_run_cached = not force
            return "# md", {"utterance_count": 3}

        def push(self, day):
            pushes.append(day)
            return {"toast": None}

    monkeypatch.setattr(daily_mod, "DailyAnalyzer", FakeAnalyzer)
    with pytest.raises(JobError, match="MiMo API key"):
        rt.run_job("daily", day="2026-10-30")
    rt._client = FakeClient()
    assert rt.run_job("daily", day="2026-10-30") == {"day": "2026-10-30", "utterances": 3, "cached": True, "partial": False, "chars": 4}
    out = rt.run_job("daily", day="2026-10-30", force=True, push=True)
    assert out["cached"] is False and out["push"] == {"toast": None} and pushes == ["2026-10-30"]
    assert runs == [("2026-10-30", False), ("2026-10-30", True)]
    assert rt.run_job("daily")["day"] == datetime.now().strftime("%Y-%m-%d")   # default: today


def test_relabel_job_requires_enrolled_speakers(rt: Runtime):
    with pytest.raises(JobError, match="还没有注册任何说话人"):
        rt.run_job("relabel")


def test_reindex_job_in_fts_mode_skips_vectors(rt: Runtime):
    seg = rt.db.add_segment(rt.cfg.audio_dir / "x.opus", datetime(2026, 10, 30, 9, 0), 3.0)
    rt.db.add_utterances([{"segment_id": seg, "start_s": 0, "end_s": 1, "abs_start": "2026-10-30T09:00:00", "day": "2026-10-30",
                           "text_raw": "测试文本", "text": "测试文本"}])
    assert rt.run_job("reindex") == {"keywords": 1}
    assert rt.db.one("SELECT COUNT(*) AS n FROM utterance_tokens")["n"] == 1


def test_shutdown_stops_the_job_thread(rt_factory):
    r = rt_factory()
    assert r._job_thread.is_alive()
    r.shutdown()
    r._job_thread.join(5)
    assert not r._job_thread.is_alive() and not r.service_running()


# --- shared components ------------------------------------------------------------------------------

def test_client_requires_a_key_and_is_cached(rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(JobError) as ex:
        rt.client()
    assert "MIMO_API_KEY" in str(ex.value)
    import audio_note.llm.client as client_mod

    monkeypatch.setattr(client_mod, "MimoClient", FakeClient)
    rt.cfg["mimo"]["api_key"] = "sk-not-a-real-key-just-for-the-fake"
    c = rt.client()
    assert isinstance(c, FakeClient) and c.rpm == 90 and rt.client() is c


def test_retriever_is_cached_per_mode_and_expansion(rt: Runtime):
    r = rt.retriever()
    assert r.mode == "fts" and rt.retriever() is r and rt.retriever("fts", False) is r
    assert rt.retriever("fts", False) is rt.retriever("FTS")
    # query expansion needs the LLM: client() is called while the component lock is held, so this must
    # raise JobError (no key) rather than deadlock (it did with a non-reentrant lock)
    with pytest.raises(JobError):
        rt.retriever(expand=True)
    assert set(rt._retrievers) == {("fts", False)}


def test_chat_needs_a_key(rt: Runtime):
    with pytest.raises(JobError):
        rt.chat()


# --- reload_config ----------------------------------------------------------------------------------

def test_reload_config_updates_components_and_restarts_recorder(rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    rt._client = object()
    rt._retrievers[("fts", False)] = object()
    old_cfg = rt.cfg
    rt._engine = SimpleNamespace(cfg=old_cfg)
    rt.sched._processor = object()
    rt.sched._client_warned = True
    calls: list[str] = []
    monkeypatch.setattr(rt, "recording", lambda: True)
    monkeypatch.setattr(rt, "stop_recording", lambda timeout=None: calls.append("stop"))
    monkeypatch.setattr(rt, "start_recording", lambda: calls.append("start"))
    _rewrite_config(rt, mimo={"chat_model": "mimo-v2.6-pro"}, audio={"gain": 2.0})
    info = rt.reload_config()
    assert info == {"changed": ["audio", "mimo"], "restarted_recorder": True, "notes": ["录音已按新的音频设置重启"]}
    assert calls == ["stop", "start"]
    assert rt.cfg is not old_cfg and rt.cfg.mimo.chat_model == "mimo-v2.6-pro" and rt.cfg.audio.gain == 2.0
    assert rt.cfg.mimo.api_key == "" and rt.cfg.data_dir == old_cfg.data_dir
    assert rt._client is None and rt._retrievers == {}
    assert rt._engine.cfg is rt.cfg
    assert rt.sched.cfg is rt.cfg and rt.sched._processor is None and rt.sched._client_warned is False


def test_reload_config_notes_for_restart_only_sections_and_no_recorder_restart(rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []
    monkeypatch.setattr(rt, "stop_recording", lambda timeout=None: calls.append("stop"))
    monkeypatch.setattr(rt, "start_recording", lambda: calls.append("start"))
    _rewrite_config(rt, ui={"port": 9999}, log={"level": "DEBUG"}, audio={"gain": 3.0})
    info = rt.reload_config()                                            # not recording: no restart
    assert info["changed"] == ["audio", "log", "ui"] and info["restarted_recorder"] is False
    assert calls == []
    assert "界面地址/端口的修改需要重启程序才会生效" in info["notes"] and "日志设置的修改需要重启程序才会生效" in info["notes"]
    assert not any("录音" in n for n in info["notes"])
    assert rt.reload_config() == {"changed": [], "restarted_recorder": False, "notes": []}   # nothing changed


# --- Scheduler additions ----------------------------------------------------------------------------

def test_scheduler_processor_is_none_without_a_key(rt: Runtime):
    s = rt.sched
    assert s.processor() is None and s._client_warned is True
    assert s.processor() is None                                         # cached "no client" is re-checked, not an error
    assert s._default_process() == "skipped (no api key)"


def test_scheduler_processor_uses_engine_factory_and_is_rebuilt_after_reload(rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    s = rt.sched
    client = FakeClient()
    monkeypatch.setattr(s, "_client", lambda: client)
    engine = object()
    s.engine_factory = lambda: engine
    proc = s.processor()
    assert proc is not None and proc._engine is engine and proc._client is client
    assert proc.stop_event is s.stop_event and s.processor() is proc
    s.reload_config(rt.cfg)
    assert s._processor is None and s.processor() is not proc
    # a failing factory is tolerated: the Processor builds its own engine lazily
    s.reload_config(rt.cfg)
    s.engine_factory = lambda: (_ for _ in ()).throw(RuntimeError("no model"))
    assert s.processor()._engine is None


def test_reset_stop_event_propagates_to_processor(rt: Runtime):
    s = rt.sched
    old = s.stop_event
    s.reset_stop_event(threading.Event())                                # no processor yet: no error
    fake = SimpleNamespace(stop_event=old)
    s._processor = fake
    ev = threading.Event()
    s.reset_stop_event(ev)
    assert s.stop_event is ev
    assert fake.stop_event is old and s._processor is None    # a tick still using `fake` keeps its (set) event
