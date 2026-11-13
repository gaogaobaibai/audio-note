"""Request-scoped plumbing shared by every router of the web API.

    RT / HolderDep      FastAPI dependencies: the Runtime of THIS request / the RuntimeHolder
    rewriting(rt)       the one gate every handler that rewrites utterances goes through
    *_arg(...)          strict parsers for values taken from a JSON body or a query string (400 + Chinese message)
    require_*(...)      "row exists or 404"
    LocalGuard          pure-ASGI admission check: local Host only, no cross-site state-changing request

One Runtime per request: `get_rt` reads `holder.rt` exactly once, so a handler never mixes the config of one
profile with the database of another while a profile switch is in flight (the old attribute-level proxy could).
Handlers that must see the runtime AFTER a switch they triggered read `holder.rt` again themselves.

Nothing here imports numpy / the models: `import audio_note.ui.app` stays cheap.
"""
from __future__ import annotations

import ipaddress
import logging
import re
import socket
from contextlib import contextmanager
from typing import Annotated, Any, Callable, Iterator
from urllib.parse import urlsplit

from fastapi import Depends, HTTPException, Request

from ..runtime import ReadOnly, Runtime, RuntimeHolder

log = logging.getLogger(__name__)

DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------------------------- runtime of a request
async def get_holder(request: Request) -> RuntimeHolder:
    return request.app.state.holder


async def get_rt(request: Request) -> Runtime:
    """The live Runtime, resolved once per request (async: no thread-pool hop for sync handlers)."""
    return request.app.state.holder.rt


HolderDep = Annotated[RuntimeHolder, Depends(get_holder)]
RT = Annotated[Runtime, Depends(get_rt)]


def unknown_label(rt: Any) -> str:
    """Display name of an unidentified speaker (speakers.unknown_label)."""
    return str(rt.cfg.speakers.unknown_label or "未知")


async def to_thread(fn: Callable[..., Any], *args: Any) -> Any:
    """Run blocking work of an async handler on the thread pool (the event loop also serves the SSE streams)."""
    import anyio

    return await anyio.to_thread.run_sync(fn, *args)


@contextmanager
def rewriting(rt: Any) -> Iterator[Any]:
    """Gate for every handler that rewrites utterances (label / confirm / reject / name a voice / split /
    rename or delete a speaker / enrol a sample):

    * read-only process (the service runs in another process, which owns the jobs working on the same rows):
      refused with ReadOnly -> 409, like Runtime.submit;
    * otherwise the request is counted in Runtime.busy(), so a profile switch waits for it instead of closing
      the database under it (JobError -> 409 once a switch has claimed the runtime).

    The counter is released when the block ends, i.e. before the response is sent: keep the `with` inside the
    handler body, never in a yield-dependency (that would end after the response).
    """
    other = rt.other_pid()
    if other:
        raise ReadOnly(other)
    with rt.busy():
        yield rt


# ---------------------------------------------------------------------------------------------- argument parsers
def day_arg(s: Any) -> str:
    if not isinstance(s, str) or not DAY_RE.match(s):
        raise HTTPException(400, f"日期格式应为 YYYY-MM-DD: {s!r}")
    return s


def bool_arg(v: Any, name: str, default: bool = False) -> bool:
    """A boolean flag from a JSON body / query: true / false / 0 / 1 (or those as text); a missing value
    (None) is `default`; anything else is 400 - the text "false" is never truthy."""
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in ("true", "false", "0", "1"):
        return v.strip().lower() in ("true", "1")
    raise HTTPException(400, f"{name} 应为 true/false")


def force_arg(v: Any) -> bool:
    """A `force` flag from a body / query (see bool_arg)."""
    return bool_arg(v, "force")


def force_kwargs(force: Any) -> dict[str, Any]:
    """`force=True` only when asked, so engines without the keyword (older fakes) keep working."""
    return {"force": True} if force else {}


def int_arg(v: Any, name: str, minimum: int | None = None) -> int | None:
    """Optional integer from a JSON body: None / "" mean "not given"; a JSON integer or its text form is
    accepted; anything else (true, 1.5, "abc", a list) is 400."""
    if v is None or v == "":
        return None
    if isinstance(v, bool) or isinstance(v, (dict, list)):
        raise HTTPException(400, f"{name} 应为整数")
    if isinstance(v, float):
        if v != int(v):
            raise HTTPException(400, f"{name} 应为整数")
        v = int(v)
    try:
        n = int(v)
    except (TypeError, ValueError):
        raise HTTPException(400, f"{name} 应为整数")
    if minimum is not None and n < minimum:
        raise HTTPException(400, f"{name} 不能小于 {minimum}")
    return n


def speaker_id_arg(v: Any) -> int | None:
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


def value_error_http(e: Exception, status: int = 400) -> HTTPException:
    """ValueError -> HTTPException(400, message); a SampleMismatch carries "mismatch": true plus the score so
    the page can offer "仍然加入" (the error handler passes a dict detail through as the JSON body)."""
    from ..speech.speakers import SampleMismatch

    if isinstance(e, SampleMismatch):
        return HTTPException(status, {"error": str(e), "mismatch": True, "score": round(e.score, 3),
                                      "threshold": e.threshold, "name": e.name})
    return HTTPException(status, str(e))


def require_speaker(db: Any, sid: Any, cols: str = "id") -> Any:
    """The speakers row (columns `cols`), or 404."""
    row = db.one(f"SELECT {cols} FROM speakers WHERE id=?", (int(sid),))
    if row is None:
        raise HTTPException(404, "没有这个说话人")
    return row


def require_utterance(db: Any, uid: Any, cols: str = "id") -> Any:
    """The utterances row (columns `cols`), or 404."""
    row = db.one(f"SELECT {cols} FROM utterances WHERE id=?", (int(uid),))
    if row is None:
        raise HTTPException(404, "没有这条记录")
    return row


# ---------------------------------------------------------------------------------------------- local-only guard
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
#: Host names that always mean "this machine"
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
#: what Starlette's TestClient puts in Host / in the ASGI scope's client address. The name is accepted only
#: together with that in-process client: over a real socket "testserver" is just another foreign name (and a
#: single-label one, which a hostile LAN can make resolve to anything).
TEST_HOST = "testserver"
TEST_CLIENT = "testclient"
_WILDCARD_BINDS = frozenset({"0.0.0.0", "::"})
_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}


def host_name(value: str | None) -> str:
    """Bare lower-case host of a Host header / config value: no port, no IPv6 brackets, no trailing dot."""
    h = str(value or "").strip().lower()
    if h.startswith("["):
        end = h.find("]")
        return h[1:end] if end > 0 else ""
    if h.count(":") == 1:
        h = h.split(":", 1)[0]
    return h.rstrip(".")


def _host_port(netloc: str, scheme: str) -> tuple[str, int | None]:
    """(host, port) of "host[:port]" with the scheme's default port filled in; ("", None) when malformed."""
    try:
        parts = urlsplit(f"{scheme}://{netloc}")
        host, port = (parts.hostname or "").rstrip("."), parts.port
    except ValueError:
        return "", None
    return host, port if port is not None else _DEFAULT_PORTS.get(scheme)


def _is_ip(name: str) -> bool:
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return False


_own_names_cache: frozenset[str] | None = None


def _own_names() -> frozenset[str]:
    """This machine's own host name (and its mDNS form); only consulted when the UI listens on every interface."""
    global _own_names_cache
    if _own_names_cache is None:
        try:
            n = socket.gethostname().strip().lower().rstrip(".")
        except OSError:
            n = ""
        _own_names_cache = frozenset({n, f"{n}.local"}) if n else frozenset()
    return _own_names_cache


def host_allowed(host_header: str | None, ui_host: str | None = None, test_client: bool = False) -> bool:
    """Is the request's Host header one of this machine's own names?

    A page on some other site can make the browser resolve ITS domain to 127.0.0.1 (DNS rebinding) and then read
    this API "same-origin"; such requests still carry the attacker's domain in Host, so anything that is not a
    local name is refused. Allowed: 127.0.0.1, localhost, [::1], the configured ui.host, and "testserver" when
    the request comes from the in-process TestClient (`test_client`). When ui.host is a wildcard bind
    (0.0.0.0 / ::) the server is reached by address or machine name: IP literals (which no domain can
    impersonate) and this machine's host name are accepted as well.
    """
    name = host_name(host_header)
    if not name:
        return False
    if name in LOCAL_HOSTS or (test_client and name == TEST_HOST):
        return True
    cfg_host = host_name(ui_host)
    if cfg_host in _WILDCARD_BINDS:
        return _is_ip(name) or name in _own_names()
    return bool(cfg_host) and name == cfg_host


def cross_site_reason(method: str, scheme: str, host_header: str | None, origin: str | None,
                      fetch_site: str | None) -> str | None:
    """Why a state-changing request must be refused as cross-site, or None. Safe methods always pass; so do
    requests without an Origin header (curl, scripts, tests) unless the browser itself labels them cross-site."""
    if method.upper() in SAFE_METHODS:
        return None
    if origin is not None:
        o = origin.strip()
        try:
            parts = urlsplit(o)
            o_host, o_port = (parts.hostname or "").rstrip("."), parts.port
            o_scheme = parts.scheme.lower()
        except ValueError:
            o_host, o_port, o_scheme = "", None, ""
        if o_port is None:
            o_port = _DEFAULT_PORTS.get(o_scheme)
        if not o_host or (o_host, o_port) != _host_port(str(host_header or "").strip().lower(), scheme):
            return f"Origin {o[:80]!r}"
    if (fetch_site or "").strip().lower() == "cross-site":
        return "Sec-Fetch-Site: cross-site"
    return None


class LocalGuard:
    """Pure-ASGI middleware (no BaseHTTPMiddleware: streams are not wrapped), config-free and always on.

    The API has no authentication because it only listens on this machine; this guard closes the two ways a web
    page elsewhere could still reach the private transcripts through the user's own browser:

    * a Host header that is not local (DNS rebinding)                         -> 403 {"error": ...}
    * a state-changing request (anything but GET / HEAD / OPTIONS) sent by a
      page of another origin (cross-site POST: needs no preflight)            -> 403 {"error": ...}
    """

    def __init__(self, app: Any, ui_host: Callable[[], Any] | None = None):
        self.app = app
        self._ui_host = ui_host
        self.rejected = 0

    def _configured_host(self) -> str | None:
        if self._ui_host is None:
            return None
        try:
            v = self._ui_host()
        except Exception:  # noqa: BLE001 - a runtime being switched: fall back to the built-in local names
            return None
        return str(v) if v else None

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or []}
        host = headers.get("host")
        peer = scope.get("client")
        in_process_test = bool(peer) and peer[0] == TEST_CLIENT      # never the peer of a real connection
        reason: str | None = None
        message = ""
        if not host_allowed(host, self._configured_host(), test_client=in_process_test):
            reason = f"Host {str(host or '')[:80]!r}"
            message = "拒绝访问：这个界面只接受发给本机地址的请求（请用 http://127.0.0.1:端口 打开）"
        elif scope["type"] == "http":
            reason = cross_site_reason(scope.get("method", "GET"), scope.get("scheme", "http"), host,
                                       headers.get("origin"), headers.get("sec-fetch-site"))
            if reason:
                message = "拒绝访问：不接受来自其他网站的修改请求"
        if reason is None:
            await self.app(scope, receive, send)
            return
        self.rejected += 1
        # the first few are worth a warning; a page hammering the port must not flood the log. Path and header
        # values come from the outside: logged as repr and cut short, so they cannot fake log lines.
        (log.warning if self.rejected <= 20 else log.debug)(
            "rejected %s %r (%s)", scope.get("method", scope["type"]), str(scope.get("path", ""))[:200], reason)
        if scope["type"] == "websocket":
            from starlette.websockets import WebSocketClose

            await WebSocketClose(code=1008)(scope, receive, send)
            return
        from starlette.responses import JSONResponse

        await JSONResponse({"error": message}, status_code=403)(scope, receive, send)


def install_local_guard(app: Any, holder: RuntimeHolder) -> None:
    """Put LocalGuard in front of the whole app (API, static files and the page)."""

    def ui_host() -> Any:
        return holder.rt.cfg.ui.host

    app.add_middleware(LocalGuard, ui_host=ui_host)
