"""Behaviour the split of ui/app.py tightened on purpose (SPEC v1 sections 8 and 11, package A2):

* one Runtime per request (dependency) instead of a proxy that resolved every attribute separately;
* strict booleans: the text "false" is never truthy;
* POST /api/jobs/{name} accepts whatever the runtime knows at call time;
* every handler that rewrites utterances goes through deps.rewriting (read-only check + Runtime.busy());
* nothing blocking runs on the event loop.

No network, no microphone, no models: engines / jobs / importers are fakes.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from datetime import datetime
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from audio_note import runtime as rt_mod
from audio_note.runtime import JobError, JobState, ReadOnly, Runtime, RuntimeHolder
from audio_note.ui import deps
from audio_note.ui.app import create_app

DAY = "2026-10-30"


def _isolated(tmp_path, **extra) -> dict:
    """Config overrides for a Runtime under test: no vector model, and the glossary inside the temp dir
    (its default location is <project root>/glossary.txt, which a test must never write)."""
    return {"memory": {"retrieval": "fts", "embedding_model": "hash"},
            "correction": {"glossary_file": str(tmp_path / "glossary.txt")}, **extra}


@pytest.fixture
def rt(cfg_factory, tmp_path) -> Runtime:
    r = Runtime(cfg_factory(_isolated(tmp_path)))
    yield r
    r.shutdown()
    from audio_note.logging_setup import _remove_own_handlers

    _remove_own_handlers(logging.getLogger())


@pytest.fixture
def client(rt: Runtime) -> TestClient:
    with TestClient(create_app(rt), raise_server_exceptions=False) as c:
        yield c


def _seed(rt: Runtime, who: str | None = "张三") -> dict[str, Any]:
    """One segment (no audio file) with two sentences of `who` on DAY."""
    db = rt.db
    seg = db.add_segment(rt.cfg.audio_dir / "x.wav", datetime(2026, 10, 30, 9, 0), 10.0, speech_s=10.0)
    db.set_segment_status(seg, "done")
    sid = db.upsert_speaker(who) if who else None
    ids = db.add_utterances([{
        "segment_id": seg, "start_s": float(i), "end_s": float(i + 1), "abs_start": f"{DAY}T09:0{i}:00", "day": DAY,
        "cluster_label": 0, "speaker_id": sid, "speaker_name": who, "speaker_score": 0.7 if sid else None,
        "text_raw": t, "text": t} for i, t in enumerate(["今天讨论了预算问题", "明天去北京出差"])])
    return {"seg": seg, "speaker": sid, "ids": ids}


def _own_the_service_elsewhere(rt: Runtime, monkeypatch: pytest.MonkeyPatch, pid: int = 424242) -> None:
    """Make this process "read-only": the service lock belongs to another (fake) live process."""
    from audio_note import service

    service.lock_path(rt.cfg).write_text(f"{pid}\n2026-10-31T09:00:00\n1.0\n", encoding="utf-8")
    monkeypatch.setattr(service, "_same_process", lambda p, ctime: p == pid)


# --- one Runtime per request ------------------------------------------------------------------------------------------

class _CountingHolder(RuntimeHolder):
    """RuntimeHolder whose `.rt` counts how often it is read."""

    def __init__(self, rt: Runtime):
        self.reads = 0
        super().__init__(rt)

    @property
    def rt(self) -> Runtime:
        self.reads += 1
        return self._rt

    @rt.setter
    def rt(self, value: Runtime) -> None:
        self._rt = value


def test_a_request_resolves_the_runtime_once_however_much_the_handler_uses_it(rt: Runtime):
    _seed(rt)
    holder = _CountingHolder(rt)
    with TestClient(create_app(holder), raise_server_exceptions=False) as c:
        for url in ("/api/days", f"/api/transcript/{DAY}", "/api/speakers", "/api/config"):
            holder.reads = 0
            assert c.get(url).status_code == 200
            # one read by the dependency + one by the guard (ui.host); the old proxy read it per attribute
            assert holder.reads == 2, url
        holder.reads = 0
        assert c.get("/api/devices/system").status_code == 200          # a handler that needs no runtime
        assert holder.reads == 1


def test_get_rt_and_get_holder_dependencies(rt: Runtime):
    holder = RuntimeHolder(rt)
    app = create_app(holder)
    assert app.state.holder is holder

    class Req:
        pass

    req = Req()
    req.app = app
    assert asyncio.run(deps.get_holder(req)) is holder and asyncio.run(deps.get_rt(req)) is rt
    other = object()
    holder.rt = other                                                     # what a profile switch does
    assert asyncio.run(deps.get_rt(req)) is other
    holder.rt = rt
    assert inspect.iscoroutinefunction(deps.get_rt) and inspect.iscoroutinefunction(deps.get_holder)   # no pool hop


# --- strict booleans ----------------------------------------------------------------------------------------------------

def test_bool_arg_is_strict():
    b = deps.bool_arg
    assert b(None, "x") is False and b(None, "x", default=True) is True
    for v, expect in ((True, True), (False, False), (1, True), (0, False), ("true", True), ("FALSE", False),
                      (" 1 ", True), ("0", False)):
        assert b(v, "x") is expect and b(v, "x", default=True) is expect
    for bad in ("yes", "no", "", "maybe", 2, -1, 1.0, [], {}, [True]):
        with pytest.raises(HTTPException) as ex:
            b(bad, "flag")
        assert ex.value.status_code == 400 and ex.value.detail == "flag 应为 true/false"
    assert deps.force_arg("true") is True and deps.force_kwargs(True) == {"force": True} and deps.force_kwargs(False) == {}
    with pytest.raises(HTTPException) as ex:
        deps.force_arg("sure")
    assert ex.value.detail == "force 应为 true/false"


def test_int_arg_day_arg_and_speaker_id_arg():
    assert deps.int_arg(None, "n") is None and deps.int_arg("", "n") is None
    assert deps.int_arg(3, "n") == 3 and deps.int_arg("12", "n") == 12 and deps.int_arg(4.0, "n") == 4
    for bad in (True, 1.5, "abc", [1], {"a": 1}):
        with pytest.raises(HTTPException) as ex:
            deps.int_arg(bad, "limit")
        assert ex.value.status_code == 400 and ex.value.detail == "limit 应为整数"
    with pytest.raises(HTTPException) as ex:
        deps.int_arg(0, "segment_id", minimum=1)
    assert ex.value.detail == "segment_id 不能小于 1"
    assert deps.day_arg("2026-10-30") == "2026-10-30"
    for bad in ("2026-1-1", "", None, 20261030, "2026-10-30x"):
        with pytest.raises(HTTPException) as ex:
            deps.day_arg(bad)
        assert ex.value.status_code == 400 and "YYYY-MM-DD" in ex.value.detail
    assert deps.speaker_id_arg(None) is None and deps.speaker_id_arg("") is None and deps.speaker_id_arg("7") == 7
    for bad in (0, -2, True, "x", [1]):
        with pytest.raises(HTTPException):
            deps.speaker_id_arg(bad)


def test_service_start_does_not_treat_the_text_false_as_true(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    seen: list[bool] = []
    monkeypatch.setattr(rt, "start_service", lambda record=True, schedule=True: seen.append(record))
    assert client.post("/api/service/start", json={"record": "false"}).status_code == 200
    assert client.post("/api/service/start", json={"record": "true"}).status_code == 200
    assert client.post("/api/service/start", json={}).status_code == 200
    assert client.post("/api/service/start").status_code == 200
    assert seen == [False, True, False, False]
    r = client.post("/api/service/start", json={"record": "maybe"})
    assert r.status_code == 400 and r.json() == {"error": "record 应为 true/false"} and len(seen) == 4


def test_job_flags_are_strict(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    seen: list[tuple[str, dict]] = []
    monkeypatch.setattr(rt, "submit", lambda name, **kw: seen.append((name, kw)) or JobState(name))
    assert client.post("/api/jobs/process", json={"retry_failed": "false", "limit": "5"}).status_code == 200
    assert client.post("/api/jobs/reindex", json={"vectors": "false"}).status_code == 200
    assert client.post("/api/jobs/reindex", json={}).status_code == 200
    assert client.post("/api/jobs/daily", json={"day": DAY, "force": "0", "push": 1}).status_code == 200
    assert client.post("/api/jobs/relations", json={"speaker_id": "3", "force": "true"}).status_code == 200
    assert client.post("/api/jobs/relabel", json={"day": DAY, "segment_id": "4"}).status_code == 200
    assert client.post("/api/jobs/cleanup", json={"anything": "ignored"}).status_code == 200
    assert seen == [("process", {"retry_failed": False, "limit": 5}), ("reindex", {"vectors": False}), ("reindex", {"vectors": True}),
                    ("daily", {"day": DAY, "force": False, "push": True}), ("relations", {"speaker_id": 3, "force": True}),
                    ("relabel", {"day": DAY, "segment_id": 4}), ("cleanup", {})]
    for name, body, msg in (("process", {"retry_failed": "nope"}, "retry_failed 应为 true/false"),
                            ("process", {"limit": "many"}, "limit 应为整数"),
                            ("daily", {"force": "x"}, "force 应为 true/false"),
                            ("reindex", {"vectors": []}, "vectors 应为 true/false"),
                            ("relabel", {"segment_id": True}, "segment_id 应为整数")):
        r = client.post(f"/api/jobs/{name}", json=body)
        assert r.status_code == 400 and r.json() == {"error": msg}, (name, body)
    assert len(seen) == 7


def test_speaker_patch_glossary_and_chat_flags_are_strict(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    sid = rt.db.upsert_speaker("张三")
    assert client.patch(f"/api/speakers/{sid}", json={"is_self": "false"}).status_code == 200
    assert rt.db.self_speaker() is None
    assert client.patch(f"/api/speakers/{sid}", json={"is_self": "true"}).status_code == 200
    assert int(rt.db.self_speaker()["id"]) == sid
    r = client.patch(f"/api/speakers/{sid}", json={"is_self": "x", "notes": "不该写进去"})
    assert r.status_code == 400 and r.json() == {"error": "is_self 应为 true/false"}
    assert rt.db.one("SELECT notes FROM speakers WHERE id=?", (sid,))["notes"] is None     # validated before any write
    # glossary: "false" does not start a reindex job
    r = client.put("/api/glossary", json={"text": "星河\n", "reindex": "false"})
    assert r.status_code == 200 and r.json() == {"ok": True, "terms": 1}
    assert client.put("/api/glossary", json={"text": "", "reindex": "nah"}).status_code == 400
    # chat: expand
    seen: list[Any] = []

    def chat(strategy=None, retrieval=None, expand=None):
        seen.append(expand)
        raise JobError("没有配置 MiMo API key")

    monkeypatch.setattr(rt, "chat", chat)
    for body_expand in ("false", "true", None):
        body = {"question": "在吗", "history": ["not a dict", {"role": "user", "content": "之前的问题"}]}
        if body_expand is not None:
            body["expand"] = body_expand
        assert client.post("/api/chat", json=body).status_code == 409
    assert seen == [False, True, None]
    assert client.post("/api/chat", json={"question": "在吗", "expand": "perhaps"}).status_code == 400


# --- jobs: whatever the runtime knows at call time --------------------------------------------------------------------------

def test_jobs_endpoint_accepts_job_names_registered_after_import(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    assert client.post("/api/jobs/tidy").status_code == 404
    got: list[dict] = []
    monkeypatch.setattr(rt_mod, "JOBS", (*rt_mod.JOBS, "tidy"))
    rt.jobs["tidy"] = JobState("tidy")
    rt._job_tidy = lambda **kw: got.append(kw) or {"tidied": len(kw)}              # type: ignore[attr-defined]
    r = client.post("/api/jobs/tidy", json={"day": DAY, "limit": 5, "dry_run": "false", "ids": [1, 2], "note": None})
    assert r.status_code == 200, r.text
    assert r.json()["job"]["args"] == {"day": DAY, "limit": 5, "dry_run": False, "ids": [1, 2]}
    assert rt.wait_jobs(5) and got == [{"day": DAY, "limit": 5, "dry_run": False, "ids": [1, 2], "note": None}]
    assert client.get("/api/jobs").json()["tidy"]["result"] == {"tidied": 5}
    # argument validation happens before anything is queued
    for body, msg in (({"day": "tomorrow"}, "日期格式应为 YYYY-MM-DD"), ({"opts": {"deep": True}}, "任务参数 opts"),
                      ({"rows": [[1], [2]]}, "任务参数 rows"), ({"_private": 1}, "任务参数名不合法"), ({"not-a-name": 1}, "任务参数名不合法")):
        r = client.post("/api/jobs/tidy", json=body)
        assert r.status_code == 400 and msg in r.json()["error"], body
    assert client.post("/api/jobs/tidy", json={"day": ""}).status_code == 200      # "" = not given
    assert rt.wait_jobs(5) and got[-1] == {"day": None}


def test_jobs_endpoint_also_accepts_names_only_the_runtime_instance_knows(client: TestClient, rt: Runtime):
    rt.jobs["extra"] = JobState("extra")
    rt._job_extra = lambda: {"ok": 1}                                            # type: ignore[attr-defined]
    assert "extra" not in rt_mod.JOBS
    assert client.post("/api/jobs/extra").status_code == 200
    assert rt.wait_jobs(5) and rt.jobs["extra"].result == {"ok": 1}


def test_a_listed_job_the_runtime_cannot_run_is_a_409_not_a_500(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rt_mod, "JOBS", (*rt_mod.JOBS, "ghost"))
    r = client.post("/api/jobs/ghost")
    assert r.status_code == 409 and "ghost" in r.json()["error"]
    assert client.post("/api/jobs/frobnicate").json() == {"error": "未知任务 frobnicate"}


# --- the rewrite gate --------------------------------------------------------------------------------------------------------

def test_rewriting_counts_the_request_and_refuses_in_a_read_only_process(rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    assert rt.request_busy() == 0
    with deps.rewriting(rt) as inside:
        assert inside is rt and rt.request_busy() == 1
    assert rt.request_busy() == 0
    with pytest.raises(ValueError):
        with deps.rewriting(rt):
            raise ValueError("handler failed")
    assert rt.request_busy() == 0                                                # released on errors too
    rt.switching = True
    with pytest.raises(JobError, match="切换档案"):
        with deps.rewriting(rt):
            pass
    rt.switching = False
    _own_the_service_elsewhere(rt, monkeypatch)
    with pytest.raises(ReadOnly, match="424242"):
        with deps.rewriting(rt):
            raise AssertionError("must not run")
    assert rt.request_busy() == 0


def test_rename_and_delete_hold_the_busy_counter(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    """They rewrite utterances.speaker_name but used to run outside Runtime.busy()."""
    from audio_note.memory import index as mem_index

    seed = _seed(rt)
    seen: list[int] = []
    real = mem_index.register_from
    monkeypatch.setattr(mem_index, "register_from", lambda cfg, db: (seen.append(rt.request_busy()), real(cfg, db))[1])
    r = client.patch(f"/api/speakers/{seed['speaker']}", json={"name": "李雷"})
    assert r.status_code == 200 and r.json() == {"ok": True, "renamed_days": 1}
    assert seen == [1] and rt.request_busy() == 0
    assert {u["speaker_name"] for u in rt.db.query("SELECT speaker_name FROM utterances")} == {"李雷"}
    orig = rt.db.delete_speaker
    monkeypatch.setattr(rt.db, "delete_speaker", lambda sid, unk: (seen.append(rt.request_busy()), orig(sid, unk))[1])
    assert client.delete(f"/api/speakers/{seed['speaker']}").json() == {"ok": True, "unlabeled_days": 1}
    assert seen == [1, 1] and rt.request_busy() == 0


def test_rewrites_are_refused_while_a_profile_switch_is_in_flight(client: TestClient, rt: Runtime):
    seed = _seed(rt)
    rt.switching = True
    try:
        r = client.patch(f"/api/speakers/{seed['speaker']}", json={"name": "李雷"})
        assert r.status_code == 409 and "切换档案" in r.json()["error"]
        assert client.delete(f"/api/speakers/{seed['speaker']}").status_code == 409
        assert client.post(f"/api/utterances/{seed['ids'][0]}/confirm", json={"speaker_id": seed["speaker"]}).status_code == 409
        assert client.post(f"/api/utterances/{seed['ids'][0]}/reject", json={}).status_code == 409
    finally:
        rt.switching = False
    assert rt.db.one("SELECT name FROM speakers WHERE id=?", (seed["speaker"],))["name"] == "张三"
    assert rt.db.one("SELECT COUNT(*) AS n FROM utterances WHERE speaker_id IS NOT NULL")["n"] == 2


def test_read_only_process_refuses_everything_that_rewrites_utterances(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    seed = _seed(rt)
    sid, (u1, u2) = seed["speaker"], seed["ids"]
    monkeypatch.setattr(rt, "speaker_models_missing", lambda: [])
    monkeypatch.setattr(rt, "client", lambda: object())
    monkeypatch.setattr(rt, "engine", lambda: (_ for _ in ()).throw(AssertionError("the engine must not be touched")))
    _own_the_service_elsewhere(rt, monkeypatch)
    members = [[seed["seg"], 0]]
    calls = [
        ("PATCH", f"/api/speakers/{sid}", {"json": {"name": "李雷"}}),
        ("DELETE", f"/api/speakers/{sid}", {}),
        ("POST", f"/api/utterances/{u1}/reject", {"json": {}}),
        ("POST", f"/api/utterances/{u1}/confirm", {"json": {"speaker_id": sid}}),
        ("POST", "/api/voices/name", {"json": {"members": members, "name": "王敏"}}),
        ("POST", "/api/voices/reject", {"json": {"members": members, "speaker_id": sid}}),
        ("POST", "/api/speakers/王敏/samples", {"content": b"\0" * 2000, "params": {"filename": "a.wav"}}),
    ]
    for method, url, kw in calls:
        r = client.request(method, url, **kw)
        assert r.status_code == 409 and "424242" in r.json()["error"], (method, url, r.text)
    rows = rt.db.query("SELECT speaker_id, speaker_name, speaker_source FROM utterances ORDER BY id")
    assert [(r["speaker_id"], r["speaker_name"], r["speaker_source"]) for r in rows] == [(sid, "张三", None)] * 2
    assert [r["name"] for r in rt.db.query("SELECT name FROM speakers")] == ["张三"]
    assert rt.db.one("SELECT COUNT(*) AS n FROM utterance_rejections")["n"] == 0
    assert not (rt.cfg.data_dir / "uploads").exists()
    # reading, and edits that only touch the speakers row, still work
    assert client.get(f"/api/transcript/{DAY}").status_code == 200
    assert client.patch(f"/api/speakers/{sid}", json={"notes": "同事", "relation": "同事"}).status_code == 200
    assert rt.db.one("SELECT notes, relation FROM speakers WHERE id=?", (sid,))["notes"] == "同事"
    assert client.patch(f"/api/speakers/{sid}", json={"name": "张三"}).json() == {"ok": True, "renamed_days": 0}   # not a rename


def test_label_and_split_are_refused_in_a_read_only_process(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch, tmp_path):
    seed = _seed(rt)
    wav = tmp_path / "seg.wav"
    wav.write_bytes(b"RIFF")
    rt.db.execute("UPDATE segments SET path=? WHERE id=?", (str(wav), seed["seg"]))

    class NoProcessor:
        def __init__(self, *a, **k):
            pass

        def split_utterance(self, *a, **k):
            raise AssertionError("must not run")

    import audio_note.speech.pipeline as pipeline

    monkeypatch.setattr(pipeline, "Processor", NoProcessor)
    monkeypatch.setattr(rt, "speaker_models_missing", lambda: [])
    monkeypatch.setattr(rt, "client", lambda: object())
    monkeypatch.setattr(rt, "engine", lambda: object())
    _own_the_service_elsewhere(rt, monkeypatch)
    r = client.post("/api/speakers/label", json={"name": "王敏", "utterance_id": seed["ids"][0]})
    assert r.status_code == 409 and "424242" in r.json()["error"]
    r = client.post(f"/api/utterances/{seed['ids'][0]}/split", json={"at": [0.5]})
    assert r.status_code == 409 and "424242" in r.json()["error"]
    assert rt.db.one("SELECT COUNT(*) AS n FROM speakers")["n"] == 1 and rt.request_busy() == 0


def test_label_validates_its_arguments_before_touching_anything(client: TestClient, rt: Runtime):
    seed = _seed(rt, who=None)
    uid = seed["ids"][0]
    for body, msg in (({"name": "王敏", "utterance_id": "abc"}, "utterance_id 应为整数"),
                      ({"name": "王敏", "utterance_id": uid, "whole_cluster": "sometimes"}, "whole_cluster 应为 true/false"),
                      ({"name": "王敏", "utterance_id": uid, "is_self": "me"}, "is_self 应为 true/false"),
                      ({"name": "王敏", "utterance_id": ""}, "需要 utterance_id，或 segment_id + from + to")):
        r = client.post("/api/speakers/label", json=body)
        assert r.status_code == 400 and r.json() == {"error": msg}, body
    assert rt.db.one("SELECT COUNT(*) AS n FROM speakers")["n"] == 0


# --- nothing blocking on the event loop ---------------------------------------------------------------------------------------

def _api_endpoints(app) -> dict[str, Any]:
    try:
        from fastapi.routing import iter_route_contexts

        routes = list(iter_route_contexts(app.routes))
    except ImportError:
        routes = list(app.routes)
    return {r.name: r.endpoint for r in routes if getattr(r, "endpoint", None) is not None and str(getattr(r, "path", "")).startswith("/api/")}


def test_only_the_two_raw_body_handlers_are_coroutines(rt: Runtime):
    """Of the original 54 endpoints, the sync ones run on the thread pool as a whole; the two that must be async
    (they read a raw upload) hand everything else to a thread - see the next tests. Endpoints added later are
    not constrained here."""
    from .test_v1_routes import BASELINE

    eps = _api_endpoints(create_app(rt))
    original = {name for _, path, name in BASELINE if path.startswith("/api/")}
    assert original <= set(eps)
    assert {n for n in original if inspect.iscoroutinefunction(eps[n])} == {"speaker_add_sample", "import_audio"}


def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def test_sample_upload_does_its_work_off_the_event_loop(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    from audio_note.ui.routers import enroll

    seen: list[tuple] = []

    def fake(rt_, name, data, ext, notes, force):
        seen.append((_on_event_loop(), rt_ is rt, name, len(data), ext, notes, force))
        return {"ok": True, "id": 1, "name": name, "samples": 1, "speech_s": 3.0}

    monkeypatch.setattr(enroll, "_enroll_upload", fake)
    r = client.post("/api/speakers/王敏/samples", params={"filename": "a.WAV", "notes": "备注", "force": "1"}, content=b"\1" * 1500)
    assert r.status_code == 200 and r.json()["name"] == "王敏"
    assert seen == [(False, True, "王敏", 1500, ".wav", "备注", True)]
    assert client.post("/api/speakers/王敏/samples", params={"force": "perhaps"}, content=b"\1" * 1500).status_code == 400
    assert len(seen) == 1


def test_import_runs_in_a_thread_inside_busy_and_leaves_no_file_on_a_bad_request(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    import audio_note.audio.importer as importer
    import audio_note.audio.vad as vad_mod

    seen: list[tuple] = []

    class FakeVAD:
        def __init__(self, path, threads=1):
            self.threads = threads

    def fake_import(cfg, db, dest, start_time=None, vad=None):
        seen.append((_on_event_loop(), rt.request_busy(), dest.name, dest.read_bytes()[:4], start_time, vad.threads))
        return [11, 12]

    monkeypatch.setattr(vad_mod, "SileroVAD", FakeVAD)
    monkeypatch.setattr(importer, "import_file", fake_import)
    r = client.post("/api/import", params={"filename": "../会议 录音.wav", "start": "2026-10-30 09:00"}, content=b"RIFF" + b"\0" * 2000)
    assert r.status_code == 200 and r.json() == {"ok": True, "file": "会议_录音.wav", "segments": 2}
    assert seen == [(False, 1, "会议_录音.wav", b"RIFF", datetime(2026, 10, 30, 9, 0), 2)] and rt.request_busy() == 0
    r = client.post("/api/import", params={"filename": "会议 录音.wav"}, content=b"RIFF" + b"\0" * 2000)
    assert r.json()["file"] == "会议_录音_1.wav"                                     # same name again: numbered
    # a bad start time is rejected before the upload is written anywhere
    before = sorted(p.name for p in (rt.cfg.data_dir / "imports").iterdir())
    r = client.post("/api/import", params={"filename": "b.wav", "start": "昨天"}, content=b"RIFF" + b"\0" * 2000)
    assert r.status_code == 400 and "开始时间格式" in r.json()["error"]
    assert sorted(p.name for p in (rt.cfg.data_dir / "imports").iterdir()) == before and len(seen) == 2
    # while a profile switch is in flight the import is refused instead of running on a database about to close
    rt.switching = True
    try:
        assert client.post("/api/import", params={"filename": "c.wav"}, content=b"RIFF" + b"\0" * 2000).status_code == 409
    finally:
        rt.switching = False
    assert len(seen) == 2


def test_concurrent_requests_from_several_event_loops_share_one_app(rt: Runtime):
    """Nothing in the app is bound to one event loop (several TestClients = several loops, as in the profile tests)."""
    _seed(rt)
    app = create_app(rt)
    results: list[int] = []

    def go() -> None:
        with TestClient(app, raise_server_exceptions=False) as c:
            for url in ("/api/status", f"/api/transcript/{DAY}", "/", "/api/nope"):
                results.append(c.get(url).status_code)

    threads = [threading.Thread(target=go) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert sorted(results) == sorted([200, 200, 200, 404] * 4)


# --- calibrate_threshold: found before and after it moved out of the CLI ----------------------------------------------------

def test_calibrate_threshold_comes_from_speech_calibrate_or_falls_back_to_the_cli(monkeypatch: pytest.MonkeyPatch):
    import sys
    import types

    import audio_note
    import audio_note.speech as speech_pkg
    from audio_note.ui.routers import speakers

    new_home = types.ModuleType("audio_note.speech.calibrate")
    new_home.calibrate_threshold = lambda embs: {"from": "speech.calibrate"}           # type: ignore[attr-defined]
    old_home = types.ModuleType("audio_note.cli")
    old_home.calibrate_threshold = lambda embs: {"from": "cli"}                        # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "audio_note.cli", old_home)
    monkeypatch.setattr(audio_note, "cli", old_home, raising=False)
    monkeypatch.setitem(sys.modules, "audio_note.speech.calibrate", new_home)
    monkeypatch.setattr(speech_pkg, "calibrate", new_home, raising=False)
    assert speakers._calibrate_threshold_fn()([]) == {"from": "speech.calibrate"}      # its home wins
    monkeypatch.setitem(sys.modules, "audio_note.speech.calibrate", None)              # a tree where it has not moved yet
    monkeypatch.delattr(speech_pkg, "calibrate", raising=False)
    assert speakers._calibrate_threshold_fn()([]) == {"from": "cli"}


def test_calibrate_endpoint_reports_the_current_threshold(client: TestClient, rt: Runtime):
    import numpy as np

    a, b = rt.db.upsert_speaker("张三"), rt.db.upsert_speaker("李四")
    rng = np.random.default_rng(7)
    for sid in (a, a, b, b):
        rt.db.add_speaker_embedding(sid, rng.normal(size=16).astype(np.float32), None, 5.0)
    r = client.post("/api/speakers/calibrate")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_embeddings"] == 4 and body["n_speakers"] == 2 and body["current"] == float(rt.cfg.speakers.identify_threshold)
    assert {"same", "cross", "suggested", "overlap"} <= set(body)
