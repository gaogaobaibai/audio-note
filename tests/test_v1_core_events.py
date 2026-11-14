"""audio_note.events (the in-process bus) and audio_note.ui.routers.events (GET /api/events, GET /api/metrics).

The stream is exercised through fastapi.testclient on a bare app that holds only this router: TestClient returns
when a streaming response ends, so every test makes the stream end (`?once=1`, or the bus is closed).
"""
from __future__ import annotations

import audio_note
import json
import logging
import threading
import time
from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audio_note import events as ev_mod
from audio_note.events import NULL, EventBus
from audio_note.runtime import Runtime, RuntimeHolder
from audio_note.ui.routers import events as router_mod


# --- the bus -------------------------------------------------------------------------------------------------

def test_publish_returns_increasing_ids_and_delivers_to_every_subscriber():
    bus = EventBus()
    a, b = bus.subscribe(), bus.subscribe()
    n1 = bus.publish("job", name="process", state="running")
    n2 = bus.publish("segment", id=42, status="new")
    assert (n1, n2) == (1, 2) and bus.last_seq == 2 and bus.subscribers == 2
    e1, e2 = a.get(1), a.get(1)
    assert e1["type"] == "job" and e1["seq"] == 1 and e1["id"] == 1 and e1["name"] == "process" and e1["state"] == "running"
    assert datetime.fromisoformat(e1["ts"])
    # `id` is the publisher's entity id when one is given; the sequence number is always in `seq`
    assert e2["type"] == "segment" and e2["seq"] == 2 and e2["id"] == 42 and e2["status"] == "new"
    assert [e["seq"] for e in b.drain()] == [1, 2] and b.drain() == []
    assert a.get(0.01) is None                                            # timeout
    a.close()
    assert bus.subscribers == 1 and bus.publish("config") == 3 and a.get(0.01) is None
    assert [e["type"] for e in b.drain()] == ["config"]


def test_envelope_fields_cannot_be_overwritten_by_data():
    bus = EventBus()
    sub = bus.subscribe()
    bus.publish("utterance", seq=999, change="text", id=7)
    ev = sub.get(1)
    assert ev["seq"] == 1 and ev["type"] == "utterance" and ev["id"] == 7 and ev["change"] == "text"


def test_ring_keeps_the_last_500_and_since_replays_what_was_missed():
    bus = EventBus()
    for i in range(520):
        bus.publish("inbox", open=i)
    assert bus.last_seq == 520 and bus.oldest_seq == 21 and len(bus.recent()) == 500 == ev_mod.KEEP_EVENTS
    sub = bus.subscribe(since=515)
    assert sub.pending == 5 and [e["seq"] for e in sub.drain()] == [516, 517, 518, 519, 520]
    bus.publish("inbox", open=0)
    assert sub.get(1)["seq"] == 521
    assert [e["seq"] for e in bus.recent(since=519)] == [520, 521]
    assert bus.subscribe().drain() == []                                   # no `since`: only what comes next


def test_get_blocks_until_an_event_arrives_from_another_thread():
    bus = EventBus()
    sub = bus.subscribe()
    got: list = []
    t = threading.Thread(target=lambda: got.append(sub.get(5)))
    t.start()
    time.sleep(0.05)
    assert got == []
    bus.publish("memory", change="added")
    t.join(5)
    assert got and got[0]["type"] == "memory"


def test_many_threads_publish_without_losing_or_duplicating():
    bus = EventBus(keep=10_000)
    sub = bus.subscribe(maxsize=10_000)
    notified: list[int] = []
    sub.on_event = lambda: notified.append(1)

    def worker(k: int) -> None:
        for i in range(200):
            bus.publish("job", name=f"w{k}", n=i)

    ts = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    [t.start() for t in ts]
    [t.join(10) for t in ts]
    evs = sub.drain()
    assert len(evs) == 1600 and sorted(e["seq"] for e in evs) == list(range(1, 1601)) and len(notified) == 1600
    for k in range(8):                                                     # per publisher the order is kept
        assert [e["n"] for e in evs if e["name"] == f"w{k}"] == list(range(200))


def test_a_slow_subscriber_drops_the_oldest_events_instead_of_growing():
    bus = EventBus()
    sub = bus.subscribe(maxsize=5)
    for i in range(12):
        bus.publish("inbox", open=i)
    assert sub.dropped == 7 and [e["open"] for e in sub.drain()] == [7, 8, 9, 10, 11]


def test_close_wakes_waiters_and_refuses_new_subscriptions_until_reopened():
    bus = EventBus()
    sub = bus.subscribe()
    woke: list[str] = []
    sub.on_event = lambda: woke.append("x")
    got: list = ["unset"]
    t = threading.Thread(target=lambda: got.__setitem__(0, sub.get(10)))
    t.start()
    time.sleep(0.05)
    bus.close()
    t.join(5)
    assert not t.is_alive() and got[0] is None and sub.closed and woke == ["x"] and bus.closed
    assert bus.publish("config") == 1                                      # still recorded, nobody listens
    late = bus.subscribe()
    assert late.closed and late.get(0.01) is None and bus.subscribers == 0
    bus.reopen()
    live = bus.subscribe()
    assert not live.closed and bus.publish("config") == 2 and live.get(1)["seq"] == 2
    with bus.subscribe() as ctx:
        assert bus.subscribers == 2
    assert ctx.closed and bus.subscribers == 1


def test_a_failing_wake_up_callback_never_reaches_the_publisher(caplog):
    bus = EventBus()
    sub = bus.subscribe()
    sub.on_event = lambda: (_ for _ in ()).throw(RuntimeError("loop is closed"))
    with caplog.at_level(logging.DEBUG):
        assert bus.publish("config") == 1
    assert sub.get(1)["type"] == "config"


def test_null_bus_accepts_everything_and_delivers_nothing():
    assert NULL.publish("segment", id=1, status="new") == 0
    sub = NULL.subscribe(since=5)
    assert sub.closed and sub.get(0.01) is None and sub.drain() == [] and NULL.recent() == []
    NULL.close()
    NULL.reopen()
    assert NULL.last_seq == 0 and NULL.oldest_seq == 0


# --- GET /api/events ---------------------------------------------------------------------------------------------

@pytest.fixture
def rt(cfg_factory) -> Runtime:
    r = Runtime(cfg_factory({"memory": {"retrieval": "fts", "embedding_model": "hash"}}))
    yield r
    r.shutdown()
    from audio_note.logging_setup import _remove_own_handlers

    _remove_own_handlers(logging.getLogger())


@pytest.fixture
def app(rt: Runtime) -> FastAPI:
    a = FastAPI()
    a.state.holder = RuntimeHolder(rt)
    a.include_router(router_mod.router)
    return a


def _frames(body: str) -> list[dict]:
    """Parse an SSE body into [{"id": ..., "event": ..., "data": {...}} | {"comment": ...}]."""
    out: list[dict] = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        if block.startswith(":"):
            out.append({"comment": block[1:].strip()})
            continue
        f: dict = {}
        for line in block.split("\n"):
            k, _, v = line.partition(": ")
            f[k] = json.loads(v) if k == "data" else v
        out.append(f)
    return out


def test_router_contract():
    assert router_mod.router.prefix == "/api"
    paths = {r.path for r in router_mod.router.routes}
    assert {"/api/events", "/api/metrics"} <= paths


def test_events_once_sends_hello_with_a_status_snapshot_and_ends(app: FastAPI, rt: Runtime):
    rt.db.execute("INSERT INTO inbox_items(kind, title) VALUES ('term', '是「灰度」吗')")
    rt.db.add_segment(rt.cfg.audio_dir / "a.opus", datetime.now(), 3.0)
    rt.events.publish("config")
    with TestClient(app) as c:
        r = c.get("/api/events?once=1")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert "no-cache" in r.headers["cache-control"]
    frames = _frames(r.text)
    assert len(frames) == 1 and "id" not in frames[0] and "event" not in frames[0]
    hello = frames[0]["data"]
    assert hello["type"] == "hello" and hello["seq"] == 1 and hello["inbox"] == {"open": 1} and hello["gap"] is False and hello["replay"] == 0
    st = hello["status"]
    assert st["version"] == audio_note.__version__ and st["schema_version"] == 2 and st["pending"] == 1 and st["failed"] == 0 and st["inbox_open"] == 1
    assert st["service"] == {"running": False, "read_only": False, "scheduler": False, "recording": False, "is_speaking": False,
                            "paused_for_playback": False}
    assert st["jobs_running"] == [] and st["deferred"] is None and st["api_key"] is False
    assert rt.events.subscribers == 0                                      # the subscription was released


def test_events_replay_with_since_and_last_event_id(app: FastAPI, rt: Runtime):
    for i in range(4):
        rt.events.publish("segment", id=100 + i, status="new")
    with TestClient(app) as c:
        r = c.get("/api/events?once=1&since=2")
        frames = _frames(r.text)
        assert frames[0]["data"]["type"] == "hello" and frames[0]["data"]["replay"] == 2 and frames[0]["data"]["gap"] is False
        assert [(f["id"], f["data"]["seq"], f["data"]["id"], f["data"]["type"]) for f in frames[1:]] == [
            ("3", 3, 102, "segment"), ("4", 4, 103, "segment")]
        # a reconnecting EventSource sends the header instead; ?named=1 adds the event name
        r = c.get("/api/events?once=1&named=1", headers={"Last-Event-ID": "3"})
        frames = _frames(r.text)
        assert frames[0]["event"] == "hello" and [(f["id"], f["event"]) for f in frames[1:]] == [("4", "segment")]
        # ids from another process (server restarted) or older than the ring: the page must reload
        assert _frames(c.get("/api/events?once=1&since=900").text)[0]["data"]["gap"] is True
        assert _frames(c.get("/api/events?once=1&since=4").text)[0]["data"]["gap"] is False
        assert _frames(c.get("/api/events?once=1&since=junk").text)[0]["data"]["replay"] == 0
    small = EventBus(keep=2)
    for i in range(5):
        small.publish("inbox", open=i)
    app.state.holder.events = small
    with TestClient(app) as c:
        hello = _frames(c.get("/api/events?once=1&since=1").text)[0]["data"]
    assert hello["gap"] is True and hello["replay"] == 2


def test_events_stream_delivers_live_events_and_ends_when_the_bus_closes(app: FastAPI, rt: Runtime):
    bus = app.state.holder.events
    assert bus is rt.events

    def later() -> None:
        deadline = time.monotonic() + 5
        while bus.subscribers == 0 and time.monotonic() < deadline:        # wait for the request to subscribe
            time.sleep(0.01)
        bus.publish("job", name="daily", state="running", source="ui")
        bus.publish("utterances", day="2026-11-11", segment_id=3, ids=[7, 8], note="含中文")
        time.sleep(0.2)
        bus.close()

    t = threading.Thread(target=later)
    t.start()
    t0 = time.monotonic()
    with TestClient(app) as c:
        r = c.get("/api/events")
    t.join(5)
    assert r.status_code == 200 and time.monotonic() - t0 < 5
    frames = _frames(r.text)
    assert [f["data"]["type"] for f in frames] == ["hello", "job", "utterances"]
    assert frames[1]["id"] == "1" and frames[1]["data"]["state"] == "running"
    assert frames[2]["data"]["ids"] == [7, 8] and frames[2]["data"]["note"] == "含中文" and "含中文" in r.text    # not \\u-escaped
    assert bus.subscribers == 0


def test_events_heartbeat_comment_while_idle(app: FastAPI, rt: Runtime, monkeypatch):
    monkeypatch.setattr(router_mod, "HEARTBEAT_S", 0.1)
    bus = app.state.holder.events

    def later() -> None:
        time.sleep(0.6)
        bus.close()

    t = threading.Thread(target=later)
    t.start()
    with TestClient(app) as c:
        r = c.get("/api/events")
    t.join(5)
    frames = _frames(r.text)
    assert frames[0]["data"]["type"] == "hello"
    assert len([f for f in frames if f.get("comment") == "keep-alive"]) >= 2


def test_events_disabled_by_config_answers_404(app: FastAPI, rt: Runtime):
    rt.cfg["ui"]["events"] = False
    with TestClient(app) as c:
        r = c.get("/api/events?once=1")
    assert r.status_code == 404 and "ui.events" in r.json()["error"]
    assert rt.status()["events"] is False


def test_one_app_serves_event_requests_from_several_threads_and_loops(app: FastAPI, rt: Runtime):
    """Nothing in the router may bind to one event loop: every TestClient below runs its own."""
    rt.events.publish("config")
    out: list = []

    def go() -> None:
        try:
            with TestClient(app) as c:
                for _ in range(3):
                    r = c.get("/api/events?once=1&since=0")
                    out.append([f["data"]["type"] for f in _frames(r.text)])
        except Exception as e:  # noqa: BLE001
            out.append(e)

    ts = [threading.Thread(target=go) for _ in range(3)]
    [t.start() for t in ts]
    [t.join(20) for t in ts]
    assert out == [["hello", "config"]] * 9


def test_events_are_served_by_the_real_app(rt: Runtime):
    from audio_note.ui.app import create_app

    with TestClient(create_app(rt), raise_server_exceptions=False) as c:
        r = c.get("/api/events?once=1")
        assert r.status_code == 200 and _frames(r.text)[0]["data"]["type"] == "hello"
        assert c.get("/api/metrics").status_code == 200


# --- GET /api/metrics ----------------------------------------------------------------------------------------------

def test_metrics_endpoint(app: FastAPI, rt: Runtime):
    rt.db.add_usage_local("diarize", audio_s=60.0, wall_s=4.0, cpu_s=6.0)
    rt.db.add_usage_local("diarize", audio_s=30.0, wall_s=2.0, cpu_s=3.0, day="2020-01-01")       # outside every window
    rt.db.add_usage_llm("asr", "mimo-v2.5-asr", prompt_tokens=40, completion_tokens=20, audio_s=55.0, wall_s=3.0)
    with TestClient(app) as c:
        body = c.get("/api/metrics?days=7").json()
        assert body["days"] == 7 and [r["stage"] for r in body["local"]] == ["diarize"] and body["local"][0]["cpu_s"] == 6.0
        assert body["llm"][0]["purpose"] == "asr" and body["totals"]["llm"]["calls"] == 1
        assert body["totals"]["local"] == {"n": 1, "audio_s": 60.0, "wall_s": 4.0, "cpu_s": 6.0}
        assert c.get("/api/metrics").json()["days"] == 7
        for bad in ("0", "-3", "abc", "9999"):
            r = c.get(f"/api/metrics?days={bad}")
            assert r.status_code == 400 and "days" in r.json()["error"]


# --- a real server: disconnects -----------------------------------------------------------------------------------

def test_a_client_that_goes_away_releases_its_subscription(app: FastAPI, rt: Runtime):
    import socket

    import httpx

    from audio_note.ui.server import UIServer

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    srv = UIServer(app, "127.0.0.1", port)
    url = srv.start()
    try:
        for _ in range(5):                                                 # tabs opened and closed
            with httpx.stream("GET", url + "/api/events", timeout=10) as r:
                lines = r.iter_lines()            # keep the iterator: dropping it closes the connection at once
                assert json.loads(next(lines)[6:])["type"] == "hello"
                assert rt.events.subscribers == 1
                del lines
            deadline = time.monotonic() + 5
            while rt.events.subscribers and time.monotonic() < deadline:
                time.sleep(0.02)
            assert rt.events.subscribers == 0
    finally:
        srv.stop()
