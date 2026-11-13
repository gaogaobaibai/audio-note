"""Runtime / Scheduler 1.0: the job table (lazy "module:function" targets), job events + job_runs, the realtime
wake-up from the recorder, the post-processing chain / consolidate / brief hooks, deferral, "stopped is not
failed", the status() additions and the holder-owned event bus.

No network, no microphone, no models: processors, recorders and job functions are small fakes.
"""
from __future__ import annotations

import audio_note
import json
import logging
import socket
import sys
import threading
import time
import types
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from audio_note import chain, config, governor, service
from audio_note import runtime as rt_mod
from audio_note import scheduler as sch
from audio_note.db import DB, STOPPED_ERROR
from audio_note.events import NULL, EventBus
from audio_note.runtime import JOB_LABELS, JOB_TARGETS, JOBS, JobError, Runtime, RuntimeHolder

from .conftest import MODELS_DIR, SAMPLE_WAV

OLD_JOBS = ("process", "daily", "cleanup", "reindex", "relabel", "push", "voices", "relations", "loudness")
NEW_JOBS = ("chain", "refine", "episodes", "consolidate", "recorrect", "rescore", "brief", "maintenance")
MISSING = "audio_note.not_written_yet"


@pytest.fixture
def rt(cfg_factory) -> Runtime:
    r = Runtime(cfg_factory({"memory": {"retrieval": "fts", "embedding_model": "hash"}}))
    yield r
    r.shutdown()
    from audio_note.logging_setup import _remove_own_handlers

    _remove_own_handlers(logging.getLogger())


def _events(sub, kind: str | None = None) -> list[dict]:
    return [e for e in sub.drain() if kind is None or e["type"] == kind]


def _quiet(s: sch.Scheduler, now: datetime | None = None) -> None:
    """Stamp the kv so that nothing periodic is due on the next tick (only wake-ups / explicit work run)."""
    now = now or datetime.now()
    s.db.set_kv(sch.KV_PROCESS, now.isoformat(timespec="seconds"))
    s.db.set_kv(sch.KV_CLEANUP, now.isoformat(timespec="seconds"))
    s.db.set_kv(sch.KV_DAILY_DAY, now.date().isoformat())
    s.db.set_kv(sch.KV_DAILY_RECHECK, now.date().isoformat())


def _utt(db: DB, seg: int, text: str = "一句话", day: str | None = None, **extra: Any) -> int:
    day = day or datetime.now().strftime("%Y-%m-%d")
    return db.add_utterances([{"segment_id": seg, "start_s": 0.0, "end_s": 1.0, "abs_start": f"{day}T09:00:00", "day": day,
                               "text_raw": text, "text": text, **extra}])[0]


class FakeProcessor:
    """Stands in for speech.pipeline.Processor: 'transcribes' a segment into one sentence."""

    def __init__(self, db: DB, fail: dict[int, Exception] | None = None, stop_on: set[int] | None = None):
        self.db, self.fail, self.stop_on = db, fail or {}, stop_on or set()
        self.segments: list[int] = []
        self.pending_calls = 0

    def process_segment(self, seg_id: int) -> int:
        self.segments.append(seg_id)
        self.db.set_segment_status(seg_id, "processing")
        if seg_id in self.stop_on:                                           # what the real pipeline does on a stop request
            self.db.set_segment_status(seg_id, "failed", "RuntimeError: stopped")
            raise RuntimeError("stopped")
        if seg_id in self.fail:
            self.db.set_segment_status(seg_id, "failed", f"{type(self.fail[seg_id]).__name__}: {self.fail[seg_id]}")
            raise self.fail[seg_id]
        _utt(self.db, seg_id)
        self.db.set_segment_status(seg_id, "done")
        return 1

    def process_pending(self, limit=None, retry_failed=False):
        self.pending_calls += 1
        stats = {"processed": 0, "done": 0, "failed": 0, "utterances": 0}
        for row in self.db.pending_segments(0, 100):
            stats["processed"] += 1
            try:
                stats["utterances"] += self.process_segment(int(row["id"]))
                stats["done"] += 1
            except Exception:  # noqa: BLE001
                stats["failed"] += 1
        return stats


# --- job table ---------------------------------------------------------------------------------------------------

def test_job_names_labels_and_targets(rt: Runtime):
    assert JOBS[:len(OLD_JOBS)] == OLD_JOBS and set(JOBS[len(OLD_JOBS):]) == set(NEW_JOBS)
    assert set(JOB_LABELS) == set(JOBS) == set(rt.jobs)
    assert {n: JOB_LABELS[n] for n in NEW_JOBS} == {
        "chain": "整理新内容", "refine": "校对会话", "episodes": "会话摘要", "consolidate": "整理记忆",
        "recorrect": "按词库重校历史", "rescore": "重判说话人", "brief": "晨报", "maintenance": "数据库维护"}
    assert JOB_TARGETS["recorrect"] == "audio_note.speech.lexicon:job_recorrect"
    assert JOB_TARGETS["rescore"] == "audio_note.speech.scoring:job_rescore"
    for name, target in JOB_TARGETS.items():
        assert name in JOBS and target.count(":") == 1 and target.startswith("audio_note.")
    # every job is either a method or a target
    assert all(hasattr(rt, f"_job_{n}") or n in JOB_TARGETS for n in JOBS)
    assert rt.status()["jobs"]["maintenance"]["label"] == "数据库维护"


def test_target_whose_module_is_missing_fails_that_job_only(rt: Runtime, monkeypatch):
    monkeypatch.setitem(JOB_TARGETS, "rescore", f"{MISSING}.scoring:job_rescore")
    st = rt.submit("rescore")
    assert rt.wait_jobs(5)
    assert st.error and "重判说话人" in st.error and "还不可用" in st.error and st.runs == 0 and st.result is None
    with pytest.raises(JobError, match="重判说话人.*还不可用"):
        rt.run_job("rescore")
    # a function that is missing from an existing module is reported the same way
    monkeypatch.setitem(JOB_TARGETS, "rescore", "audio_note.chain:no_such_function")
    with pytest.raises(JobError, match="还不可用"):
        rt.run_job("rescore")
    # the worker thread is alive and takes the next job
    assert rt._job_thread.is_alive()
    monkeypatch.setattr(rt, "_job_cleanup", lambda: {"expired": 0})
    done = rt.submit("cleanup")
    assert rt.wait_jobs(5) and done.result == {"expired": 0} and done.error is None


def test_target_is_imported_lazily_and_called_with_the_runtime(rt: Runtime, monkeypatch):
    seen: list = []
    mod = types.ModuleType("v1rt_jobs")
    mod.job = lambda runtime, **kw: seen.append((runtime, kw)) or {"changed": 2, "when": datetime(2026, 11, 1)}
    monkeypatch.setitem(sys.modules, "v1rt_jobs", mod)
    monkeypatch.setitem(JOB_TARGETS, "recorrect", "v1rt_jobs:job")
    st = rt.submit("recorrect", day="2026-11-01", lexicon_id=None)
    assert rt.wait_jobs(5)
    assert seen == [(rt, {"day": "2026-11-01", "lexicon_id": None})]
    assert st.result == {"changed": 2, "when": "2026-11-01 00:00:00"} and st.args == {"day": "2026-11-01"} and st.runs == 1
    assert rt.run_job("recorrect", alias_id=3) == {"changed": 2, "when": datetime(2026, 11, 1)}
    # a broken module (its own import fails) is a JobError too, with the cause
    broken = types.ModuleType("v1rt_broken")
    monkeypatch.setitem(sys.modules, "v1rt_broken", broken)
    monkeypatch.setitem(JOB_TARGETS, "rescore", "v1rt_broken:job_rescore")
    with pytest.raises(JobError, match="重判说话人"):
        rt.run_job("rescore")
    # a wrong keyword fails the job, not the worker
    st = rt.submit("recorrect", nonsense=1)
    mod.job = lambda runtime, day=None: {}
    assert rt.wait_jobs(5) and st.error and "nonsense" in st.error


def test_job_method_takes_precedence_over_a_target(rt: Runtime, monkeypatch):
    monkeypatch.setitem(JOB_TARGETS, "chain", f"{MISSING}.chain:job")
    monkeypatch.setattr(rt, "_job_chain", lambda **kw: {"method": True}, raising=False)
    assert rt.run_job("chain") == {"method": True}
    with pytest.raises(JobError, match="未知任务"):
        rt.run_job("frobnicate")


def test_chain_family_jobs(rt: Runtime, monkeypatch):
    calls: list = []
    mod = types.ModuleType("v1rt_steps")
    mod.sync = lambda cfg, db, client, *, stop_event=None, events=None, limit=None: calls.append(("sessions", client, events, limit)) or {"sessions": 1}
    mod.refine = lambda cfg, db, client, *, stop_event=None, events=None, limit=None: calls.append(("refine", client, events, limit)) or {"refined": 1}
    monkeypatch.setitem(sys.modules, "v1rt_steps", mod)
    monkeypatch.setattr(chain, "STEPS", (("sessions", "v1rt_steps:sync"), ("refine", "v1rt_steps:refine"),
                                         ("episodes", f"{MISSING}.episodes:run_pending")))
    monkeypatch.setattr(chain, "CONSOLIDATE", f"{MISSING}.consolidate:run")
    out = rt.run_job("chain")
    assert out["ran"] == ["sessions", "refine"] and out["skipped"] == ["episodes"] and out["failed"] == {}
    assert [(c[0], c[1], c[2]) for c in calls] == [("sessions", None, rt.events), ("refine", None, rt.events)]   # no API key: client None
    assert rt.run_job("refine", limit=2)["ran"] == ["sessions", "refine"] and calls[-1][3] == 2
    with pytest.raises(JobError, match="会话摘要功能在这个版本里还不可用"):
        rt.run_job("episodes")
    st = rt.submit("consolidate")
    assert rt.wait_jobs(5) and "记忆整理功能在这个版本里还不可用" in st.error


def test_maintenance_job(rt: Runtime):
    rt.db.add_job_run("process", "scheduler", "done", "2020-01-01T00:00:00")
    out = rt.run_job("maintenance")
    assert out["job_runs_pruned"] == 1 and out["vacuum"] is False and out["wal_checkpoint"] in ("ok", "busy")


# --- job events + history -------------------------------------------------------------------------------------------

def test_ui_jobs_publish_events_and_are_written_to_job_runs(rt: Runtime, monkeypatch):
    sub = rt.events.subscribe()
    monkeypatch.setattr(rt, "_job_cleanup", lambda: {"expired": 3, "path": Path("x/y")})
    st = rt.submit("cleanup")
    assert rt.wait_jobs(5) and st.result == {"expired": 3, "path": str(Path("x/y"))}
    time.sleep(0.05)
    evs = _events(sub, "job")
    assert [(e["name"], e["state"], e["source"]) for e in evs] == [("cleanup", "queued", "ui"), ("cleanup", "running", "ui"), ("cleanup", "done", "ui")]
    assert evs[2]["result"] == st.result and "error" not in evs[2]
    row = rt.db.one("SELECT * FROM job_runs")
    assert (row["name"], row["source"], row["status"], row["error"]) == ("cleanup", "ui", "done", None)
    assert json.loads(row["result"]) == st.result and row["started_at"] <= row["finished_at"]

    monkeypatch.setattr(rt, "_job_push", lambda day: (_ for _ in ()).throw(JobError("没有日报")))
    rt.submit("push", day="2026-11-01")
    assert rt.wait_jobs(5)
    time.sleep(0.05)
    last = _events(sub, "job")[-1]
    assert (last["name"], last["state"], last["error"]) == ("push", "failed", "没有日报") and "result" not in last
    row = rt.db.one("SELECT * FROM job_runs ORDER BY id DESC LIMIT 1")
    assert (row["name"], row["status"], row["error"], row["result"]) == ("push", "failed", "没有日报", None)


def test_job_running_reports_the_label(rt: Runtime, monkeypatch):
    release, started = threading.Event(), threading.Event()
    monkeypatch.setattr(rt, "_job_maintenance", lambda **kw: (started.set(), release.wait(10), {})[2])
    assert rt.job_running() is None
    rt.submit("maintenance")
    assert started.wait(5) and rt.job_running() == "数据库维护" and rt.busy_reason() == "数据库维护"
    assert rt.status_lite()["jobs_running"] == ["maintenance"]
    release.set()
    assert rt.wait_jobs(5) and rt.job_running() is None


def test_orphan_job_runs_are_closed_when_the_service_starts(rt: Runtime):
    rid = rt.db.start_job_run("daily", "ui")
    rt.start_service(record=False)
    try:
        row = rt.db.one("SELECT status, error FROM job_runs WHERE id=?", (rid,))
        assert (row["status"], row["error"]) == ("failed", "interrupted")
    finally:
        rt.stop_service()


# --- status ------------------------------------------------------------------------------------------------------------

def test_status_additions(rt: Runtime, monkeypatch):
    st = rt.status()
    assert st["version"] == audio_note.__version__ and st["schema_version"] == 2 and st["inbox_open"] == 0
    assert st["min_speech_dbfs"] == -40.0 and st["unknown_label"] == "未知" and st["system_turn_share"] == 0.6
    assert st["deferred"] is None and st["realtime"] is True and st["events"] is True
    assert st["performance"] == {"mode": "balanced", "threads": 2}
    assert st["scheduler_extra"] == {"runs": {"realtime": 0, "chain": 0, "consolidate": 0, "brief": 0},
                                     "errors": {"realtime": 0, "chain": 0, "consolidate": 0, "brief": 0}}
    assert st["scheduler_runs"] == {"process": 0, "cleanup": 0, "daily": 0}                      # the old shape is untouched
    rt.db.execute("INSERT INTO inbox_items(kind, title) VALUES ('term', '问一')")
    rt.db.execute("INSERT INTO inbox_items(kind, title, status) VALUES ('term', '答过了', 'answered')")
    rt.cfg["audio"]["min_speech_dbfs"] = None
    rt.cfg["pipeline"]["realtime"] = False
    monkeypatch.setattr(governor, "defer_reason", lambda cfg: "正在用电池供电")
    st = rt.status()
    assert st["inbox_open"] == 1 and st["min_speech_dbfs"] is None and st["realtime"] is False and st["deferred"] == "正在用电池供电"
    lite = rt.status_lite()
    assert lite["inbox_open"] == 1 and lite["deferred"] == "正在用电池供电" and lite["version"] == audio_note.__version__ and lite["pending"] == 0
    json.dumps(st), json.dumps(lite)


def test_audio_directory_size_is_cached_until_something_changes_it(rt: Runtime, monkeypatch):
    from audio_note.audio import storage

    walks: list[int] = []
    real = storage.dir_size_bytes
    monkeypatch.setattr(storage, "dir_size_bytes", lambda d: walks.append(1) or real(d))
    (rt.cfg.audio_dir / "a.bin").write_bytes(b"\0" * 1000)
    assert rt.status()["disk"]["audio_bytes"] == 1000 and len(walks) == 1
    (rt.cfg.audio_dir / "b.bin").write_bytes(b"\0" * 500)
    assert rt.status()["disk"]["audio_bytes"] == 1000 and rt.status()["disk"]["audio_bytes"] == 1000 and len(walks) == 1
    rt._on_segment(None)                                                   # the recorder stored a segment
    assert rt.status()["disk"]["audio_bytes"] == 1500 and len(walks) == 2
    (rt.cfg.audio_dir / "b.bin").unlink()
    rt.sched.runs["cleanup"] += 1                                          # a scheduler cleanup ran
    assert rt.status()["disk"]["audio_bytes"] == 1000 and len(walks) == 3
    monkeypatch.setattr(rt_mod, "DISK_CACHE_S", 0.0)                       # ... and it expires by itself
    assert rt.status()["disk"]["audio_bytes"] == 1000 and len(walks) == 4


# --- event bus ownership ---------------------------------------------------------------------------------------------

def test_runtime_scheduler_and_holder_share_one_bus(rt: Runtime):
    assert isinstance(rt.events, EventBus) and rt.sched.events is rt.events
    h = RuntimeHolder(rt)
    assert h.events is rt.events
    mine = EventBus()
    assert sch.Scheduler(rt.cfg, rt.db, events=mine).events is mine and sch.Scheduler(rt.cfg, rt.db).events is NULL
    other = Runtime(rt.cfg, db=rt.db, events=mine)
    try:
        assert other.events is mine and other.sched.events is mine
    finally:
        other.shutdown(close_db=False)


def test_profile_switch_keeps_the_bus_and_announces_the_profile(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AUDIO_NOTE_CONFIG", raising=False)
    monkeypatch.delenv("AUDIO_NOTE_PROFILE", raising=False)
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(config, "CURRENT_PROFILE_FILE", tmp_path / "profiles" / "current")
    (tmp_path / "config.yaml").write_text(
        f"paths:\n  data_dir: data\n  models_dir: {MODELS_DIR.as_posix()}\nmemory:\n  retrieval: fts\n  embedding_model: hash\n",
        encoding="utf-8")
    config.create_profile("demo")
    h = RuntimeHolder(Runtime(config.load_config()))
    try:
        bus, old = h.events, h.rt
        sub = bus.subscribe()
        new = h.switch("demo")
        assert new is h.rt and new is not old and new.events is bus and new.sched.events is bus and h.events is bus
        assert not sub.closed                                              # an open event stream survives the switch
        assert [e["name"] for e in _events(sub, "profile")] == ["demo"]
        new.events.publish("config")
        assert sub.get(1)["type"] == "config"
    finally:
        h.rt.shutdown()
        from audio_note.logging_setup import _remove_own_handlers

        _remove_own_handlers(logging.getLogger())


def test_config_reload_and_recording_publish_events(rt: Runtime, monkeypatch):
    sub = rt.events.subscribe()
    monkeypatch.setattr(rt_mod, "_recorder_loop", lambda cfg, db, ev, ref: ev.wait(30))
    rt.start_recording()
    try:
        assert rt._rec_ref["events"] is rt.events and rt._rec_ref["on_segment"] == rt._on_segment
    finally:
        rt.stop_recording()
        rt.stop_service()
    assert [(e["on"], e["speaking"]) for e in _events(sub, "recording")] == [(True, False), (False, False)]
    rt.stop_recording()                                                    # not recording: no second "off"
    assert _events(sub, "recording") == []
    info = rt.reload_config()
    assert [e["changed"] for e in _events(sub, "config")] == [info["changed"]]


def test_changing_the_thread_count_rebuilds_the_speaker_engine_on_next_use(rt: Runtime):
    import yaml

    def rewrite(**sections):
        data = yaml.safe_load(rt.config_path.read_text(encoding="utf-8")) or {}
        for k, v in sections.items():
            data.setdefault(k, {}).update(v)
        rt.config_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    engine = SimpleNamespace(cfg=rt.cfg)
    rt._engine = engine
    rewrite(mimo={"temperature": 0.5})                                     # nothing about threads: the engine stays
    assert rt.reload_config()["notes"] == [] and rt._engine is engine and engine.cfg is rt.cfg
    rewrite(performance={"mode": "eco"})
    info = rt.reload_config()
    assert info["changed"] == ["performance"] and info["notes"] == ["本机模型的线程数已更新，下一次处理时生效"]
    assert rt._engine is None and governor.threads(rt.cfg) == 1 and rt.status()["performance"] == {"mode": "eco", "threads": 1}
    rt._engine = engine
    rewrite(speakers={"num_threads": 1})                                   # same number by another route: nothing to rebuild
    assert rt.reload_config()["notes"] == [] and rt._engine is engine


# --- realtime wake-up ---------------------------------------------------------------------------------------------------

def test_notify_segment_processes_exactly_that_segment_at_once(rt: Runtime, monkeypatch):
    db, s = rt.db, rt.sched
    now = datetime.now()
    assert int(rt.cfg.pipeline.min_segment_age_s) == 20 and rt.cfg.pipeline.realtime is True
    a = db.add_segment("a.opus", now, 5.0)                                  # just recorded: far younger than min_segment_age_s
    b = db.add_segment("b.opus", now, 5.0)
    proc = FakeProcessor(db)
    monkeypatch.setattr(s, "processor", lambda: proc)
    _quiet(s, now)
    sub = rt.events.subscribe()
    assert s.run_once(now) == [] and proc.segments == []                   # nothing is due, nobody woke us
    s.notify_segment(a)
    s.notify_segment(a)                                                    # reported twice: processed once
    assert s.wake.is_set()
    assert s.run_once(now) == []                                           # the realtime path is not one of the periodic jobs
    assert proc.segments == [a] and proc.pending_calls == 0 and not s.wake.is_set()
    assert db.get_segment(a)["status"] == "done" and db.get_segment(a)["realtime"] == 1
    assert db.get_segment(b)["status"] == "new" and db.get_segment(b)["realtime"] is None
    assert s.extra_runs["realtime"] == 1 and s.runs["process"] == 0
    seg_events = [(e["id"], e["status"]) for e in _events(rt.events.subscribe(since=0), "segment")]
    assert seg_events == [(a, "processing"), (a, "done")]
    utt = _events(sub, "utterances")
    assert len(utt) == 1 and utt[0]["segment_id"] == a and utt[0]["day"] == now.strftime("%Y-%m-%d") and len(utt[0]["ids"]) == 1
    # a segment that is no longer 'new' (the periodic run got there first) is left alone
    s.notify_segment(a)
    s.run_once(now)
    assert proc.segments == [a] and s.extra_runs["realtime"] == 1


def test_realtime_can_be_switched_off_and_needs_a_key(rt: Runtime, monkeypatch):
    db, s = rt.db, rt.sched
    now = datetime.now()
    a = db.add_segment("a.opus", now, 5.0)
    _quiet(s, now)
    s.notify_segment(a)
    assert s.run_once(now) == [] and db.get_segment(a)["status"] == "new"  # no API key: processor() is None, nothing breaks
    proc = FakeProcessor(db)
    monkeypatch.setattr(s, "processor", lambda: proc)
    rt.cfg["pipeline"]["realtime"] = False
    s.notify_segment(a)
    assert s.run_once(now) == [] and proc.segments == [] and not s.wake.is_set()
    s.notify_segment(None)                                                 # an explicit "process now" still works
    assert s.run_once(now) == ["process"] and proc.pending_calls == 1 and db.get_segment(a)["status"] == "done"


def test_wake_with_an_injected_process_job_runs_that_job(cfg):
    db = DB(cfg.db_path)
    calls: list[str] = []
    s = sch.Scheduler(cfg, db, threading.Event(), jobs={"process": lambda: calls.append("process"), "cleanup": lambda: None,
                                                        "daily": lambda day, push: None, "push": lambda day: {}})
    now = datetime.now()
    _quiet(s, now)
    assert s.run_once(now) == []
    s.notify_segment(7)
    assert s.run_once(now) == ["process"] and calls == ["process"]


def test_loop_wakes_immediately_and_stops_promptly(rt: Runtime, monkeypatch):
    db = rt.db
    proc = FakeProcessor(db)
    stop = threading.Event()
    s = sch.Scheduler(rt.cfg, db, stop, tick_s=60.0, events=rt.events)      # a tick so long that only a wake-up can explain progress
    monkeypatch.setattr(s, "processor", lambda: proc)
    _quiet(s)
    t = threading.Thread(target=s.run, daemon=True)
    t.start()
    time.sleep(0.3)                                                        # first tick done, the loop sleeps
    a = db.add_segment("a.opus", datetime.now(), 5.0)
    t0 = time.monotonic()
    s.notify_segment(a)
    while db.get_segment(a)["status"] != "done" and time.monotonic() - t0 < 5:
        time.sleep(0.01)
    assert db.get_segment(a)["status"] == "done" and time.monotonic() - t0 < 2.0
    t0 = time.monotonic()
    stop.set()                                                             # nobody touches `wake`: the poll notices the stop
    t.join(5)
    assert not t.is_alive() and time.monotonic() - t0 < 1.5


def test_recorder_announces_a_stored_segment(monkeypatch):
    from audio_note.audio.recorder import Recorder

    rec = object.__new__(Recorder)                                         # no model, no device: only the announcement
    bus, woken = EventBus(), []
    rec.events, rec.on_segment = bus, woken.append
    seg = SimpleNamespace(audio=[0.0] * 32000, speech_s=1.234)
    rec._announce(12, datetime(2026, 11, 11, 21, 30, 5), seg)
    ev = bus.recent()[0]
    assert (ev["type"], ev["id"], ev["status"], ev["day"], ev["duration_s"], ev["speech_s"]) == ("segment", 12, "new", "2026-11-11", 2.0, 1.23)
    assert woken == [12]
    rec.events, rec.on_segment = NULL, lambda sid: (_ for _ in ()).throw(RuntimeError("scheduler gone"))
    rec._announce(13, datetime(2026, 11, 11, 21, 31), seg)                 # a failing callback never reaches the recorder
    rec.on_segment = None
    rec._announce(14, datetime(2026, 11, 11, 21, 32), seg)


def test_recorder_thread_priority_is_raised_only_while_recording(monkeypatch, cfg):
    from audio_note.audio.recorder import Recorder

    calls: list[str] = []
    monkeypatch.setattr(governor, "boost_current_thread", lambda: calls.append("boost"))
    monkeypatch.setattr(governor, "restore_current_thread", lambda: calls.append("restore"))
    rec = object.__new__(Recorder)
    rec.cfg = cfg
    monkeypatch.setattr(Recorder, "_run", lambda self: calls.append("run"))
    rec.run()
    assert calls == ["boost", "run", "restore"]
    calls.clear()
    monkeypatch.setattr(Recorder, "_run", lambda self: (_ for _ in ()).throw(RuntimeError("input stream stalled")))
    with pytest.raises(RuntimeError, match="stalled"):
        rec.run()
    assert calls == ["boost", "restore"]
    calls.clear()
    cfg["performance"]["background_priority"] = False
    monkeypatch.setattr(Recorder, "_run", lambda self: calls.append("run"))
    rec.run()
    assert calls == ["run"]


def test_recorder_loop_hands_the_bus_and_the_callback_to_each_recorder(cfg, monkeypatch):
    import audio_note.audio.recorder as recorder_mod

    made: list = []

    class FakeRecorder:
        events, on_segment = "default-bus", None

        def __init__(self, cfg, db, stop_event):
            self.stop_event = stop_event
            made.append(self)

        def run(self):
            self.on_segment(41)
            self.stop_event.set()

    monkeypatch.setattr(recorder_mod, "Recorder", FakeRecorder)
    db = DB(cfg.db_path)
    woken: list = []
    ref = {"events": "the-bus", "on_segment": woken.append}
    service._recorder_loop(cfg, db, threading.Event(), ref)
    assert made[0].events == "the-bus" and woken == [41] and ref["recorder"] is made[0]
    # without the two keys (CLI `record`, old callers) the recorder keeps its defaults
    monkeypatch.setattr(FakeRecorder, "run", lambda self: self.stop_event.set())
    service._recorder_loop(cfg, db, threading.Event(), {})
    assert made[1].events == "default-bus" and made[1].on_segment is None


# --- "stopped" is not "failed" -------------------------------------------------------------------------------------------

def test_a_segment_interrupted_by_a_stop_goes_back_to_new(rt: Runtime, monkeypatch):
    db, s = rt.db, rt.sched
    now = datetime.now()
    a = db.add_segment("a.opus", now, 5.0)
    b = db.add_segment("b.opus", now, 5.0)
    proc = FakeProcessor(db, fail={b: ValueError("bad audio")}, stop_on={a})
    monkeypatch.setattr(s, "processor", lambda: proc)
    c = db.add_segment("c.opus", now, 5.0)
    # realtime path: b fails for real, a is cut short by the stop, c is not started any more
    _quiet(s, now)
    for sid in (b, a, c):
        s.notify_segment(sid)
    s.run_once(now)
    assert proc.segments == [b, a]
    assert (db.get_segment(a)["status"], db.get_segment(a)["error"]) == ("new", None)
    assert (db.get_segment(b)["status"], db.get_segment(b)["error"]) == ("failed", "ValueError: bad audio")       # a real failure stays one
    assert db.get_segment(c)["status"] == "new" and s.extra_errors["realtime"] == 1                              # the stop is not an error
    db.set_segment_status(c, "done")
    # periodic path (scheduler) and the UI job
    db.set_kv(sch.KV_PROCESS, None)
    assert s.run_once(now) == ["process"] and db.get_segment(a)["status"] == "new"
    assert rt.run_job("process")["processed"] == 1 and db.get_segment(a)["status"] == "new"
    # a leftover from a process that died right after the stop is picked up by the next run
    proc.stop_on.clear()
    db.set_segment_status(a, "failed", STOPPED_ERROR)
    assert rt.run_job("process")["done"] == 1 and db.get_segment(a)["status"] == "done"


# --- deferral ----------------------------------------------------------------------------------------------------------------

def test_local_processing_is_deferred_on_battery_and_resumes(rt: Runtime, monkeypatch, caplog):
    db, s = rt.db, rt.sched
    now = datetime.now()
    a = db.add_segment("a.opus", now, 5.0)
    proc = FakeProcessor(db)
    monkeypatch.setattr(s, "processor", lambda: proc)
    monkeypatch.setattr(s, "_default_cleanup", lambda: {})
    s.jobs["cleanup"] = lambda: {}
    reason: list[str | None] = ["正在用电池供电，本机处理已推迟"]
    monkeypatch.setattr(governor, "defer_reason", lambda cfg: reason[0])
    db.set_kv(sch.KV_DAILY_DAY, now.date().isoformat())
    db.set_kv(sch.KV_DAILY_RECHECK, now.date().isoformat())
    s.notify_segment(a)
    with caplog.at_level(logging.INFO, logger="audio_note.scheduler"):
        assert s.run_once(now) == ["cleanup"]                              # process (periodic AND realtime) waits; the rest goes on
    assert proc.segments == [] and s.deferred == reason[0] and db.get_kv(sch.KV_PROCESS) is None
    assert sum("deferred" in r.getMessage() for r in caplog.records) == 1
    assert s.run_once(now) == [] and sum("deferred" in r.getMessage() for r in caplog.records) == 1     # said once, not every tick
    reason[0] = None                                                       # plugged in again
    assert s.run_once(now) == ["process"] and proc.segments == [a] and s.deferred is None


# --- chain / consolidate / brief ----------------------------------------------------------------------------------------------

def _recording_scheduler(cfg, db, order: list[str], **extra_jobs) -> sch.Scheduler:
    jobs = {"process": lambda: order.append("process") or {"processed": 1},
            "cleanup": lambda: order.append("cleanup"),
            "daily": lambda day, push: order.append(f"daily:{day}"),
            "push": lambda day: {},
            "chain": lambda: order.append("chain") or {"ran": ["sessions"], "skipped": [], "failed": {}, "results": {}},
            "consolidate": lambda: order.append("consolidate") or {"merged": 0},
            "brief": lambda: order.append("brief") or {"pushed": True}}
    jobs.update(extra_jobs)
    return sch.Scheduler(cfg, db, threading.Event(), jobs=jobs)


def test_chain_runs_after_process_and_before_daily_then_consolidate(cfg):
    db = DB(cfg.db_path)
    now = datetime(2026, 11, 11, 23, 0)                                    # past daily_time 22:30
    seg = db.add_segment("a.opus", now, 5.0)
    db.set_segment_status(seg, "done")
    _utt(db, seg, day="2026-11-11", session_id=1)
    db.set_kv(sch.KV_DAILY_DAY, "2026-11-10")
    order: list[str] = []
    s = _recording_scheduler(cfg, db, order)
    bus = EventBus()
    s.events = bus
    assert s.run_once(now) == ["process", "cleanup", "daily"]              # the new work is invisible in the return value
    assert order == ["process", "chain", "cleanup", "daily:2026-11-11", "consolidate"]
    assert s.extra_runs == {"realtime": 0, "chain": 1, "consolidate": 1, "brief": 0}
    states = [(e["name"], e["state"]) for e in bus.recent() if e["type"] == "job"]
    assert states == [("process", "running"), ("process", "done"), ("chain", "done"), ("cleanup", "running"), ("cleanup", "done"),
                      ("daily", "running"), ("daily", "done"), ("consolidate", "done")]
    assert [e["day"] for e in bus.recent() if e["type"] == "daily"] == ["2026-11-11"]
    assert all(e["source"] == "scheduler" for e in bus.recent() if e["type"] == "job")
    rows = [(r["name"], r["source"], r["status"]) for r in db.query("SELECT * FROM job_runs ORDER BY id")]
    assert rows == [("process", "scheduler", "done"), ("chain", "scheduler", "done"), ("daily", "scheduler", "done"),
                    ("consolidate", "scheduler", "done")]                  # the cleanup that did nothing is not history
    # process was not due on this tick, daily is: process is forced first, and the chain follows it
    order.clear()
    db.set_kv(sch.KV_DAILY_DAY, "2026-11-10")
    db.set_kv(sch.KV_CLEANUP, now.isoformat(timespec="seconds"))
    assert s.run_once(now) == ["process", "daily"] and order == ["process", "chain", "daily:2026-11-11", "consolidate"]


def test_failing_chain_consolidate_or_brief_never_stop_the_tick(cfg, caplog):
    db = DB(cfg.db_path)
    now = datetime(2026, 11, 11, 23, 0)
    seg = db.add_segment("a.opus", now, 5.0)
    db.set_segment_status(seg, "done")
    _utt(db, seg, day="2026-11-11", session_id=1)
    db.set_kv(sch.KV_DAILY_DAY, "2026-11-10")
    cfg["analysis"]["brief_time"] = "22:00"
    order: list[str] = []

    def boom(name):
        def fn():
            order.append(name)
            raise RuntimeError(f"{name} exploded")
        return fn

    def unavailable():
        raise chain.StepUnavailable("模块不存在")

    s = _recording_scheduler(cfg, db, order, chain=boom("chain"), consolidate=unavailable, brief=boom("brief"))
    with caplog.at_level(logging.ERROR, logger="audio_note.scheduler"):
        assert s.run_once(now) == ["process", "cleanup", "daily"]
    assert order == ["process", "chain", "cleanup", "daily:2026-11-11", "brief"]
    assert s.extra_errors == {"realtime": 0, "chain": 1, "consolidate": 0, "brief": 1} and s.extra_runs["consolidate"] == 0
    failed = {r["name"]: r["error"] for r in db.query("SELECT name, error FROM job_runs WHERE status='failed'")}
    assert failed == {"chain": "RuntimeError: chain exploded", "brief": "RuntimeError: brief exploded"}


def test_chain_also_runs_for_sessions_waiting_and_backs_off_without_progress(cfg, monkeypatch):
    db = DB(cfg.db_path)
    now = datetime.now()
    order: list[str] = []
    s = _recording_scheduler(cfg, db, order)
    _quiet(s, now)
    assert s.run_once(now) == [] and order == []                           # nothing waiting
    seg = db.add_segment("a.opus", now, 5.0)
    db.set_segment_status(seg, "done")
    u = _utt(db, seg)                                                      # a sentence without a session
    assert s._chain_signature(now) == (1, 0, 0)
    assert s.run_once(now) == [] and order == ["chain"]
    assert s.run_once(now) == [] and order == ["chain"]                    # same picture: no progress, back off
    db.execute("UPDATE utterances SET session_id=1 WHERE id=?", (u,))
    db.execute("INSERT INTO sessions(id, day, started_at, ended_at, status, raw_hash, text_hash) VALUES (1, ?, ?, ?, 'open', 'r', 't')",
               (now.strftime("%Y-%m-%d"), (now - timedelta(minutes=30)).isoformat(), (now - timedelta(minutes=20)).isoformat()))
    assert s._chain_signature(now) == (0, 1, 0)                            # an open session whose gap has passed
    assert s.run_once(now) == [] and order == ["chain", "chain"]
    db.execute("UPDATE sessions SET status='closed', refined_hash='r', digested_hash='t'")
    assert s._chain_signature(now) is None and s.run_once(now) == [] and order == ["chain", "chain"]
    db.execute("UPDATE sessions SET text_hash='changed'")                  # the text was edited: digest again
    assert s._chain_signature(now) == (0, 0, 1) and s.run_once(now) == [] and order == ["chain", "chain", "chain"]
    monkeypatch.setattr(sch, "CHAIN_RETRY_S", 0.0)                         # the back-off is over: try the stuck work again
    s._chain_retry_at = 0.0
    assert s.run_once(now) == [] and order[-1] == "chain" and len(order) == 4


def test_default_chain_is_skipped_while_no_step_is_installed(cfg, monkeypatch):
    db = DB(cfg.db_path)
    now = datetime.now()
    seg = db.add_segment("a.opus", now, 5.0)
    _utt(db, seg)
    monkeypatch.setattr(chain, "STEPS", (("sessions", f"{MISSING}.sessions:sync"),))
    ran: list = []
    monkeypatch.setattr(chain, "run_chain", lambda *a, **k: ran.append(1) or {})
    s = sch.Scheduler(cfg, db, threading.Event(), jobs={"process": lambda: {"processed": 1}, "cleanup": lambda: None,
                                                        "daily": lambda day, push: None, "push": lambda day: {}})
    db.set_kv(sch.KV_DAILY_DAY, now.date().isoformat())
    db.set_kv(sch.KV_DAILY_RECHECK, now.date().isoformat())
    assert s.run_once(now) == ["process", "cleanup"] and ran == [] and s.extra_runs["chain"] == 0
    mod = types.ModuleType("v1rt_sessions")
    mod.sync = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "v1rt_sessions", mod)
    monkeypatch.setattr(chain, "STEPS", (("sessions", "v1rt_sessions:sync"),))
    s._chain_probe = (0.0, False)                                          # the probe is cached for a minute
    db.set_kv(sch.KV_PROCESS, None)
    assert s.run_once(now) == ["process"] and ran == [1] and s.extra_runs["chain"] == 1


def test_brief_runs_once_a_day_from_brief_time(cfg):
    db = DB(cfg.db_path)
    order: list[str] = []
    s = _recording_scheduler(cfg, db, order)
    day = datetime(2026, 11, 11)
    briefs = lambda: order.count("brief")  # noqa: E731 - the periodic jobs run too; only the brief matters here
    assert not cfg["analysis"].get("brief_time")
    s.run_once(day.replace(hour=8))
    assert briefs() == 0                                                   # no key / empty string: switched off
    cfg["analysis"]["brief_time"] = "08:30"
    s.run_once(day.replace(hour=8, minute=29))
    assert briefs() == 0 and db.get_kv(sch.KV_BRIEF_DAY) is None
    s.run_once(day.replace(hour=8, minute=30))
    assert briefs() == 1 and order[-1] == "brief" and db.get_kv(sch.KV_BRIEF_DAY) == "2026-11-11" and s.extra_runs["brief"] == 1
    s.run_once(day.replace(hour=12))
    assert briefs() == 1                                                   # once per day
    nxt = day + timedelta(days=1)
    s.run_once(nxt.replace(hour=20))                                       # the service was off all morning: too late for a "morning" brief
    assert briefs() == 1 and db.get_kv(sch.KV_BRIEF_DAY) == "2026-11-12"
    cfg["analysis"]["brief_time"] = "25:99"
    s.run_once((nxt + timedelta(days=1)).replace(hour=9))
    assert briefs() == 1                                                   # an invalid time never raises


# --- clients -------------------------------------------------------------------------------------------------------------------

class FakeMimo:
    made: list = []

    def __init__(self, cfg=None, rpm=None):
        from audio_note.llm.client import MissingAPIKey

        if not cfg.mimo.api_key:
            raise MissingAPIKey("mimo.api_key is empty")
        self.cfg, self.rpm, self.on_usage = cfg, rpm, None
        FakeMimo.made.append(self)


def test_scheduler_and_runtime_clients_are_cached_and_report_usage(rt: Runtime, monkeypatch):
    import audio_note.llm.client as client_mod

    FakeMimo.made = []
    monkeypatch.setattr(client_mod, "MimoClient", FakeMimo)
    assert rt.sched._client() is None                                      # no key yet
    rt.cfg["mimo"]["api_key"] = "sk-not-a-real-key-just-for-the-fake"
    c = rt.sched._client()
    assert isinstance(c, FakeMimo) and c.rpm == 90 and rt.sched._client() is c and rt.sched._client_quiet() is c
    assert len(FakeMimo.made) == 1 and callable(c.on_usage)
    c.on_usage({"purpose": "asr", "model": "m", "ok": True, "prompt_tokens": 3, "completion_tokens": 1, "audio_s": 2.0, "wall_s": 0.1})
    mine = rt.client()
    assert isinstance(mine, FakeMimo) and mine is not c and callable(mine.on_usage)
    mine.on_usage({"purpose": "chat", "model": "m", "ok": False})
    rows = {r["purpose"]: (r["calls"], r["errors"]) for r in rt.db.query("SELECT * FROM usage_llm")}
    assert rows == {"asr": (1, 0), "chat": (1, 1)}
    rt.sched.reload_config(rt.cfg)                                         # a config change builds a new one
    assert rt.sched._client() is not c and len(FakeMimo.made) == 3


def test_a_missing_key_is_warned_about_once_and_not_by_the_chain(rt: Runtime, caplog):
    with caplog.at_level(logging.WARNING, logger="audio_note.scheduler"):
        assert rt.sched._client_quiet() is None and caplog.records == [] and rt.sched._client_warned is False
        assert rt.sched._client() is None and rt.sched._client() is None
    assert len(caplog.records) == 1 and rt.sched._client_warned is True


def test_processor_gets_the_event_bus_only_when_it_takes_one(rt: Runtime, monkeypatch):
    import audio_note.speech.pipeline as pipeline_mod

    made: list = []

    class OldProcessor:
        def __init__(self, cfg, db, client=None, engine=None, stop_event=None):
            made.append(("old", None))

    class NewProcessor:
        def __init__(self, cfg, db, client=None, engine=None, stop_event=None, events=None):
            made.append(("new", events))

    monkeypatch.setattr(rt.sched, "_client", lambda: object())
    monkeypatch.setattr(rt.sched, "engine_factory", None)
    monkeypatch.setattr(pipeline_mod, "Processor", OldProcessor)
    rt.sched._processor = None
    rt.sched.processor()
    assert made == [("old", None)] and rt.sched._proc_events is False
    monkeypatch.setattr(pipeline_mod, "Processor", NewProcessor)
    rt.sched.reload_config(rt.cfg)
    rt.sched.processor()
    assert made[-1] == ("new", rt.events) and rt.sched._proc_events is True
    # a Processor that publishes by itself is not announced a second time
    sub = rt.events.subscribe()
    seg = rt.db.add_segment("a.opus", datetime.now(), 5.0)
    _utt(rt.db, seg)
    rt.sched.announce_new(0, [seg])
    assert sub.drain() == []
    rt.sched._proc_events = False
    rt.sched.announce_new(0, [seg])
    assert [e["type"] for e in sub.drain()] == ["segment", "utterances"]


# --- web server shutdown -----------------------------------------------------------------------------------------------------

def test_ui_server_stop_ends_open_event_streams(rt: Runtime):
    import httpx
    from fastapi import FastAPI

    from audio_note.ui.routers import events as events_router
    from audio_note.ui.server import GRACEFUL_SHUTDOWN_S, UIServer

    app = FastAPI()
    app.state.holder = RuntimeHolder(rt)
    app.include_router(events_router.router)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    assert port != 8765
    srv = UIServer(app, "127.0.0.1", port)
    url = srv.start()
    got: list[str] = []
    done = threading.Event()

    def listen() -> None:
        try:
            with httpx.stream("GET", url + "/api/events", timeout=20) as r:
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        got.append(json.loads(line[6:])["type"])
        except Exception as e:  # noqa: BLE001
            got.append(f"error: {type(e).__name__}")
        finally:
            done.set()

    t = threading.Thread(target=listen, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 10
        while rt.events.subscribers == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert rt.events.subscribers == 1
        rt.events.publish("job", name="process", state="running")
        time.sleep(0.3)
        t0 = time.monotonic()
    finally:
        srv.stop(timeout=GRACEFUL_SHUTDOWN_S + 5)
    assert done.wait(5) and time.monotonic() - t0 < GRACEFUL_SHUTDOWN_S + 3
    assert got == ["hello", "job"] and rt.events.closed and srv._thread is None
    # starting again on the same holder accepts listeners again
    srv2 = UIServer(app, "127.0.0.1", port)
    srv2.start()
    try:
        assert not rt.events.closed
        with httpx.stream("GET", url + "/api/events?once=1", timeout=10) as r:
            assert r.status_code == 200 and "hello" in "".join(r.iter_text())
    finally:
        srv2.stop()
    UIServer(None, "127.0.0.1", 1).stop()                                  # an app without a holder: nothing to close


# --- realtime with the real pipeline (real models, fake MiMo) ---------------------------------------------------------------

@pytest.mark.skipif(not SAMPLE_WAV.exists(), reason="real sample recording not available")
def test_realtime_wake_runs_the_real_processor_on_a_fresh_segment(cfg_factory, silero, speaker_engine, monkeypatch, tmp_path):
    """The whole path with speech.pipeline.Processor: a segment younger than pipeline.min_segment_age_s is invisible
    to the periodic run, the wake-up processes it at once and the page hears about it."""
    from audio_note.audio.importer import import_file

    class FakeMimo:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "audio_seconds": 0.0}
        asr_calls = 0

        def transcribe_bytes(self, wav, mime="audio/wav", language="auto", model=None):
            FakeMimo.asr_calls += 1
            return "这是一段测试文本。", 3.0

        def json(self, system, user, **kw):
            return {"edits": []}

        def text(self, system, user, **kw):
            return "{}"

        def chat(self, messages, **kw):
            raise AssertionError("no chat in this test")

    cfg = cfg_factory({"audio": {"min_silence_s": 1.0}, "speakers": {"num_threads": 2}})
    assert int(cfg.pipeline.min_segment_age_s) == 20
    db = DB(cfg.db_path)
    now = datetime.now()
    import soundfile as sf

    audio, sr = sf.read(SAMPLE_WAV, dtype="float32")                       # silent for the first 29 s: take 45 s of speech
    clip = tmp_path / "clip.wav"
    sf.write(clip, audio[28 * sr:73 * sr], sr, subtype="PCM_16")
    seg_ids = import_file(cfg, db, clip, start_time=now - timedelta(minutes=10), vad=silero)
    assert len(seg_ids) >= 2
    bus = EventBus()
    s = sch.Scheduler(cfg, db, threading.Event(), engine_factory=lambda: speaker_engine, events=bus)
    monkeypatch.setattr(s, "_client", lambda: FakeMimo())
    s.jobs["cleanup"] = lambda: {}
    db.set_kv(sch.KV_DAILY_DAY, now.date().isoformat())
    db.set_kv(sch.KV_DAILY_RECHECK, now.date().isoformat())
    # the periodic run does not touch files this young ...
    assert s.run_once(now) == ["process", "cleanup"]
    assert db.one("SELECT COUNT(*) AS n FROM segments WHERE status='new'")["n"] == len(seg_ids) and FakeMimo.asr_calls == 0
    # ... the recorder's wake-up does, for exactly the segment it reports
    target = seg_ids[0]
    s.notify_segment(target)
    assert s.run_once(now) == []
    seg = db.get_segment(target)
    assert seg["status"] in ("done", "no_speech") and seg["realtime"] == 1 and s.extra_runs["realtime"] == 1
    assert db.one("SELECT COUNT(*) AS n FROM segments WHERE status='new'")["n"] == len(seg_ids) - 1
    n_utt = db.one("SELECT COUNT(*) AS n FROM utterances WHERE segment_id=?", (target,))["n"]
    seg_events = [(e["id"], e["status"]) for e in bus.recent() if e["type"] == "segment"]
    assert (target, seg["status"]) in seg_events
    if seg["status"] == "done":
        assert n_utt >= 1 and FakeMimo.asr_calls >= 1
        ids = [i for e in bus.recent() if e["type"] == "utterances" and e["segment_id"] == target for i in e["ids"]]
        assert len(ids) == n_utt
