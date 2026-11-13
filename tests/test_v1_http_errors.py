"""SPEC v1 section 8: every API failure is `{"error": "<Chinese message>"}`, and the local-only guard.

* Starlette's own 404 / 405 and FastAPI's request validation (422) used to answer `{"detail": ...}`, which the
  page showed as "422 Unprocessable Entity"; they keep their status codes and gain an "error" text.
* deps.LocalGuard: requests whose Host is not this machine (DNS rebinding) and state-changing requests sent by
  another site's page are refused with 403 before any handler runs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import socket

import pytest
from fastapi.testclient import TestClient

from audio_note.runtime import Runtime
from audio_note.ui import deps, errors
from audio_note.ui.app import create_app


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


# --- error normalisation ------------------------------------------------------------------------------------------

def test_unknown_route_is_404_with_an_error_text(client: TestClient):
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.json() == {"error": "没有这个接口或页面"}
    assert client.get("/nope").json() == {"error": "没有这个接口或页面"}


def test_wrong_method_is_405_with_an_error_text_and_the_allow_header(client: TestClient):
    r = client.delete("/api/status")
    assert r.status_code == 405 and r.json() == {"error": "这个接口不支持该请求方法"}
    assert r.headers["allow"] == "GET"


@pytest.mark.parametrize("method,url,kwargs,expect", [
    ("GET", "/api/context/abc", {}, "uid 应为整数"),
    ("GET", "/api/search", {}, "缺少参数 q"),
    ("GET", "/api/utterances/1/preview-split", {"params": {"auto": "maybe"}}, "auto 应为 true/false"),
    ("GET", "/api/audio/utterance/1", {"params": {"pad": "wide"}}, "pad 应为数字"),
    ("POST", "/api/chat", {"json": [1, 2]}, "请求体 应为 JSON 对象"),
    ("POST", "/api/chat", {}, "缺少请求体"),
    ("POST", "/api/chat", {"content": b"{nope", "headers": {"content-type": "application/json"}}, "请求体不是合法的 JSON"),
    ("DELETE", "/api/speakers/samples", {}, "sid 应为整数"),
])
def test_request_validation_is_422_with_a_chinese_error(client: TestClient, method: str, url: str, kwargs: dict, expect: str):
    r = client.request(method, url, **kwargs)
    assert r.status_code == 422, r.text
    body = r.json()
    assert body["error"] == f"请求参数不正确：{expect}"
    assert isinstance(body["detail"], list) and body["detail"]
    assert all(set(d) == {"type", "loc", "msg"} for d in body["detail"])        # the offending input is not echoed back


def test_handler_errors_keep_their_shapes(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    assert client.post("/api/chat", json={"question": " "}).json() == {"error": "问题不能为空"}
    r = client.get("/api/context/999")
    assert r.status_code == 404 and r.json() == {"error": "没有这条记录"}
    r = client.post("/api/chat", json={"question": "在吗"})                     # no API key: JobError
    assert r.status_code == 409 and set(r.json()) == {"error"}
    monkeypatch.setattr(rt, "status", lambda: (_ for _ in ()).throw(ValueError("坏了")))
    r = client.get("/api/status")
    assert r.status_code == 500 and r.json() == {"error": "ValueError: 坏了"}


def test_dict_detail_passes_through_with_error_coerced_to_text():
    from fastapi import FastAPI, HTTPException

    app = FastAPI()
    errors.install_error_handlers(app)

    @app.get("/mismatch")
    def mismatch():
        raise HTTPException(400, {"error": ValueError("不像同一个人"), "mismatch": True, "score": 0.31})

    @app.get("/bare")
    def bare():
        raise HTTPException(403)

    @app.get("/teapot")
    def teapot():
        raise HTTPException(418)

    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/mismatch")
        assert r.status_code == 400 and r.json() == {"error": "不像同一个人", "mismatch": True, "score": 0.31}
        assert c.get("/bare").json() == {"error": "拒绝访问"}
        r = c.get("/teapot")                                                     # no Chinese text for it: the phrase
        assert r.status_code == 418 and r.json() == {"error": "I'm a Teapot"}


def test_validation_message_lists_each_problem_once_and_caps_the_list():
    errs = [{"type": "missing", "loc": ("query", "q"), "msg": "Field required"},
            {"type": "missing", "loc": ("query", "q"), "msg": "Field required"},
            {"type": "float_parsing", "loc": ("query", "pad"), "msg": "x"},
            {"type": "string_type", "loc": ("body", "name"), "msg": "x"},
            {"type": "list_type", "loc": ("body", "at"), "msg": "x"},
            {"type": "weird_new_type", "loc": ("header", "x-thing"), "msg": "is odd"}]
    assert errors.validation_message(errs) == ("请求参数不正确：缺少参数 q；pad 应为数字；请求体.name 应为文本；请求体.at 应为列表；"
                                               "x-thing 不合法（is odd）")
    many = [{"type": "int_parsing", "loc": ("query", f"p{i}"), "msg": "x"} for i in range(8)]
    msg = errors.validation_message(many)
    assert msg.count("应为整数") == 5 and msg.endswith("等 8 处")
    assert errors.validation_message([{"type": "missing", "loc": (), "msg": ""}]) == "请求参数不正确：缺少参数 请求"


# --- local guard: Host ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("host", ["127.0.0.1:8765", "localhost:8765", "LOCALHOST", "localhost.", "[::1]:8765", "testserver", "127.0.0.1"])
def test_local_hosts_are_served(client: TestClient, host: str):
    assert client.get("/api/status", headers={"host": host}).status_code == 200


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8765", "127.0.0.1.evil.example", "localhost.evil.example:8765",
                                  "192.168.1.20:8765", "[2001:db8::1]:8765", "0.0.0.0:8765", ""])
@pytest.mark.parametrize("url", ["/api/status", "/", "/static/app.js", "/api/audio/utterance/1"])
def test_foreign_host_is_refused_everywhere(client: TestClient, host: str, url: str):
    r = client.get(url, headers={"host": host})
    assert r.status_code == 403 and set(r.json()) == {"error"} and "本机" in r.json()["error"]


def test_the_configured_ui_host_is_accepted(cfg_factory, tmp_path):
    r = Runtime(cfg_factory(_isolated(tmp_path, ui={"host": "192.168.7.7"})))
    try:
        with TestClient(create_app(r), raise_server_exceptions=False) as c:
            assert c.get("/api/status", headers={"host": "192.168.7.7:8765"}).status_code == 200
            assert c.get("/api/status", headers={"host": "192.168.7.8:8765"}).status_code == 403
            assert c.get("/api/status", headers={"host": "127.0.0.1:8765"}).status_code == 200
            # the guard follows the live config (a reload / profile switch), it holds no copy
            r.cfg["ui"]["host"] = "127.0.0.1"
            assert c.get("/api/status", headers={"host": "192.168.7.7:8765"}).status_code == 403
    finally:
        r.shutdown()
        from audio_note.logging_setup import _remove_own_handlers

        _remove_own_handlers(logging.getLogger())


def test_host_allowed_rules():
    ok = deps.host_allowed
    assert ok("127.0.0.1:8765") and ok("localhost") and ok("[::1]:1") and ok("Localhost:80")
    # "testserver" is the TestClient's name: fine in-process, a foreign name over a real connection
    assert ok("testserver", test_client=True) and not ok("testserver") and not ok("testserver", "127.0.0.1")
    assert not ok("evil.example", test_client=True)
    assert not ok(None) and not ok("") and not ok("evil.example") and not ok("[::1") and not ok("10.0.0.5:8765")
    assert not ok("evil.example", "127.0.0.1") and not ok("localhost.evil.example", "localhost")
    assert ok("box.lan:8765", "box.lan") and ok("BOX.lan.", "box.lan") and not ok("other.lan", "box.lan")
    assert ok("[fd00::7]:8765", "fd00::7") and ok("[fd00::7]:8765", "[fd00::7]")
    # listening on every interface: reached by address or by this machine's name - never by a foreign domain
    me = socket.gethostname().lower()
    for bind in ("0.0.0.0", "::"):
        assert ok("192.168.1.20:8765", bind) and ok("[fd00::7]:8765", bind) and ok("127.0.0.1", bind)
        assert not ok("evil.example:8765", bind) and not ok("192.168.1.20.evil.example", bind)
        if me:
            assert ok(f"{me}:8765", bind) and ok(f"{me}.local", bind)
    assert not ok(f"{me or 'thisbox'}:8765", "127.0.0.1") or me in deps.LOCAL_HOSTS
    assert deps.host_name(" Example.COM.:443 ") == "example.com" and deps.host_name("[::1]:8765") == "::1"
    assert deps.host_name("::1") == "::1" and deps.host_name(None) == ""


def _asgi_call(app, scope: dict) -> list[dict]:
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    return sent


def test_guard_is_a_plain_asgi_middleware_in_front_of_everything(client: TestClient):
    from starlette.middleware.base import BaseHTTPMiddleware

    classes = [m.cls for m in client.app.user_middleware]
    assert classes == [deps.LocalGuard]
    assert not any(issubclass(c, BaseHTTPMiddleware) for c in classes)          # streams are never wrapped


def test_guard_refuses_a_request_without_a_host_header_and_never_calls_the_app():
    calls: list[str] = []

    async def inner(scope, receive, send):
        calls.append(scope["type"])

    guard = deps.LocalGuard(inner)
    sent = _asgi_call(guard, {"type": "http", "method": "GET", "path": "/api/status", "scheme": "http", "headers": []})
    assert calls == [] and sent[0]["status"] == 403
    assert "error" in json.loads(sent[1]["body"].decode("utf-8"))
    # websocket handshakes are held to the same Host rule (closed, never accepted)
    sent = _asgi_call(guard, {"type": "websocket", "path": "/ws", "scheme": "ws", "headers": [(b"host", b"evil.example")]})
    assert calls == [] and sent[0]["type"] == "websocket.close"
    # "testserver" over a real connection (the peer is an address, not the in-process test client) is foreign too
    for peer in (("127.0.0.1", 50123), ("192.168.1.9", 4000), None):
        scope = {"type": "http", "method": "GET", "path": "/", "scheme": "http", "client": peer, "headers": [(b"host", b"testserver")]}
        assert _asgi_call(guard, scope)[0]["status"] == 403
    assert calls == [] and guard.rejected == 5
    # other scope types (lifespan), local requests and the TestClient pass through untouched
    _asgi_call(guard, {"type": "lifespan"})
    _asgi_call(guard, {"type": "http", "method": "GET", "path": "/", "scheme": "http", "headers": [(b"host", b"127.0.0.1:8765")]})
    _asgi_call(guard, {"type": "http", "method": "GET", "path": "/", "scheme": "http", "client": ("testclient", 50000),
                       "headers": [(b"host", b"testserver")]})
    assert calls == ["lifespan", "http", "http"] and guard.rejected == 5


def test_guard_survives_a_broken_config_lookup():
    async def inner(scope, receive, send):
        from starlette.responses import PlainTextResponse

        await PlainTextResponse("ok")(scope, receive, send)

    def broken():
        raise RuntimeError("runtime is being switched")

    guard = deps.LocalGuard(inner, ui_host=broken)
    ok = _asgi_call(guard, {"type": "http", "method": "GET", "path": "/", "scheme": "http", "headers": [(b"host", b"localhost:1")]})
    assert ok[0]["status"] == 200
    bad = _asgi_call(guard, {"type": "http", "method": "GET", "path": "/", "scheme": "http", "headers": [(b"host", b"box.lan")]})
    assert bad[0]["status"] == 403


# --- local guard: cross-site state-changing requests ---------------------------------------------------------------------

@pytest.mark.parametrize("headers", [
    {"origin": "http://evil.example"},
    {"origin": "https://testserver.evil.example"},
    {"origin": "http://testserver:8000"},            # same host, another port = another origin
    {"origin": "null"},                              # sandboxed frame / file://
    {"origin": "http://evil.example", "sec-fetch-site": "cross-site"},
    {"sec-fetch-site": "cross-site"},
    {"origin": "garbage"},
])
@pytest.mark.parametrize("method,url", [("POST", "/api/recording/stop"), ("POST", "/api/service/stop"), ("POST", "/api/jobs/cleanup"),
                                         ("PUT", "/api/glossary"), ("DELETE", "/api/notes/1"), ("PATCH", "/api/speakers/1")])
def test_cross_site_state_changing_requests_never_reach_a_handler(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch,
                                                                  headers: dict, method: str, url: str):
    touched: list[str] = []
    monkeypatch.setattr(rt, "stop_recording", lambda *a, **k: touched.append("stop_recording"))
    monkeypatch.setattr(rt, "stop_service", lambda *a, **k: touched.append("stop_service"))
    monkeypatch.setattr(rt, "submit", lambda *a, **k: touched.append("submit"))
    r = client.request(method, url, headers=headers, json={"text": "x", "name": "y"})
    assert r.status_code == 403 and r.json() == {"error": "拒绝访问：不接受来自其他网站的修改请求"}
    assert touched == [] and not rt.cfg.glossary_path.exists()


def test_same_origin_and_origin_less_requests_pass(client: TestClient):
    assert client.post("/api/recording/stop").status_code == 200                                        # curl / scripts / tests
    assert client.post("/api/recording/stop", headers={"origin": "http://testserver"}).status_code == 200
    assert client.post("/api/recording/stop", headers={"origin": "http://testserver", "sec-fetch-site": "same-origin"}).status_code == 200
    r = client.post("/api/recording/stop", headers={"host": "127.0.0.1:8765", "origin": "http://127.0.0.1:8765"})
    assert r.status_code == 200
    r = client.post("/api/recording/stop", headers={"host": "localhost", "origin": "http://localhost:80"})   # default port spelled out
    assert r.status_code == 200
    r = client.post("/api/recording/stop", headers={"host": "[::1]:8765", "origin": "http://[::1]:8765"})
    assert r.status_code == 200


def test_reading_is_not_subject_to_the_origin_check(client: TestClient):
    """Safe methods change nothing and their responses are unreadable cross-site (no CORS headers are ever sent)."""
    r = client.get("/api/status", headers={"origin": "http://evil.example", "sec-fetch-site": "cross-site"})
    assert r.status_code == 200 and "access-control-allow-origin" not in r.headers
    r = client.options("/api/status", headers={"origin": "http://evil.example", "access-control-request-method": "POST"})
    assert r.status_code == 405 and "access-control-allow-origin" not in r.headers      # no preflight is ever approved


def test_cross_site_reason_rules():
    why = deps.cross_site_reason
    assert why("GET", "http", "testserver", "http://evil.example", "cross-site") is None
    assert why("HEAD", "http", "testserver", "null", None) is None
    assert why("POST", "http", "127.0.0.1:8765", None, None) is None
    assert why("POST", "http", "127.0.0.1:8765", "http://127.0.0.1:8765", "same-origin") is None
    assert why("post", "http", "LOCALHOST:8765", "http://localhost:8765", None) is None
    assert why("POST", "https", "box.lan", "https://box.lan:443", None) is None
    assert why("POST", "http", "127.0.0.1:8765", "http://127.0.0.1:9999", None)
    assert why("POST", "http", "127.0.0.1:8765", "http://localhost:8765", None)        # a different name is a different origin
    assert why("DELETE", "http", "127.0.0.1:8765", "null", None)
    assert why("PUT", "http", "127.0.0.1:8765", "", None)
    assert why("PATCH", "http", "127.0.0.1:8765", None, "Cross-Site")
    assert why("POST", "http", "127.0.0.1:8765", None, "same-site") is None            # no Origin to compare: left to the Origin rule
    assert why("POST", "http", "127.0.0.1:notaport", "http://127.0.0.1:8765", None)
    assert why("POST", "http", None, "http://127.0.0.1:8765", None)


def test_rejections_are_logged_but_do_not_flood(caplog: pytest.LogCaptureFixture):
    async def inner(scope, receive, send):
        raise AssertionError("must not be called")

    guard = deps.LocalGuard(inner)
    scope = {"type": "http", "method": "GET", "path": "/api/days", "scheme": "http", "headers": [(b"host", b"evil.example")]}
    with caplog.at_level(logging.WARNING, logger="audio_note.ui.deps"):
        for _ in range(30):
            _asgi_call(guard, dict(scope))
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 20 and "evil.example" in warnings[0].getMessage() and guard.rejected == 30
    # what the outside sent cannot forge a log line: newlines arrive escaped, long values are cut
    caplog.clear()
    guard = deps.LocalGuard(inner)
    hostile = {"type": "http", "method": "GET", "path": "/x\n2026-11-11 00:00:00 INFO fake line" + "A" * 500, "scheme": "http",
               "headers": [(b"host", b"evil.example\r\nINFO forged" + b"B" * 500)]}
    with caplog.at_level(logging.WARNING, logger="audio_note.ui.deps"):
        _asgi_call(guard, hostile)
    msg = caplog.records[0].getMessage()
    assert "\n" not in msg and "\r" not in msg and len(msg) < 400


# --- over a real socket -----------------------------------------------------------------------------------------------------

def test_guard_over_a_real_connection(rt: Runtime):
    """Through uvicorn (UIServer) the peer is a real address: local names pass, "testserver" and foreign names do not."""
    import httpx

    from audio_note.ui.server import UIServer

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = UIServer(create_app(rt), "127.0.0.1", port)
    url = srv.start()
    try:
        with httpx.Client(base_url=url, timeout=10) as c:
            assert c.get("/api/status").status_code == 200                                  # Host: 127.0.0.1:<port>
            assert c.get("/api/status", headers={"host": f"localhost:{port}"}).status_code == 200
            for host in ("testserver", f"testserver:{port}", "evil.example", f"127.0.0.1.evil.example:{port}"):
                r = c.get("/api/status", headers={"host": host})
                assert r.status_code == 403 and set(r.json()) == {"error"}, host
            r = c.get("/", headers={"host": "evil.example"})
            assert r.status_code == 403 and "audio-note" not in r.text
            # the page's own requests carry its origin; another site's do not
            same = {"origin": url}
            assert c.post("/api/recording/stop", headers=same).status_code == 200
            r = c.post("/api/recording/stop", headers={"origin": "http://evil.example"})
            assert r.status_code == 403 and r.json() == {"error": "拒绝访问：不接受来自其他网站的修改请求"}
            r = c.post("/api/recording/stop", headers={"origin": f"http://localhost:{port}"})  # another name = another origin
            assert r.status_code == 403
            assert c.post("/api/recording/stop", headers={"host": f"localhost:{port}", "origin": f"http://localhost:{port}"}).status_code == 200
    finally:
        srv.stop()
