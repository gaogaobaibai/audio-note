"""Live updates and usage figures.

    GET /api/events          server-sent events: what the recorder, the scheduler and the jobs publish on the event bus
    GET /api/metrics?days=7  local CPU / LLM usage per day (metrics.summary)

The stream (`text/event-stream`):

    data: {"type":"hello","seq":41,"status":{...},"inbox":{"open":2},"replay":0,"gap":false}     <- always first

    id: 42
    data: {"seq":42,"id":7,"type":"segment","ts":"2026-11-11T21:30:05.123","status":"new"}

    : keep-alive                                                                                 <- every 15 s of silence

* Frames carry no `event:` line: read them with `EventSource.onmessage` and dispatch on `data.type`
  (`?named=1` adds `event: <type>` for `addEventListener` clients).
* `id:` is the bus sequence number, so a reconnecting EventSource sends `Last-Event-ID` and gets the events it
  missed (`?since=N` does the same by hand); `hello.gap` is true when some of them were already forgotten
  (or the server restarted): reload everything then. A page should refresh its state on every `hello`.
* `?once=1` answers with the hello (plus a replay, if asked for) and ends: for tests and for a quick probe.
* With `ui.events: false` the endpoint answers 404 and the page falls back to polling.

Nothing in this module is bound to an event loop: every request creates its own wake-up primitive inside its
own loop, and the bus calls back from whatever thread published (the app may be driven from several loops).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ... import events as events_mod
from ... import metrics

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

HEARTBEAT_S = 15.0      # a comment line this often while nothing happens (keeps proxies and the browser from giving up)
POLL_S = 2.0            # upper bound of one wait: a lost wake-up costs at most this much delay
MAX_METRIC_DAYS = 366
SSE_HEADERS = {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"}


def _bus(holder: Any) -> Any:
    bus = getattr(holder, "events", None)
    if bus is None:
        bus = getattr(getattr(holder, "rt", None), "events", None)
    return bus if bus is not None else events_mod.NULL


def _frame(ev: dict[str, Any], named: bool = False, with_id: bool = True) -> str:
    lines: list[str] = []
    if with_id and ev.get("seq"):
        lines.append(f"id: {int(ev['seq'])}")
    if named:
        lines.append(f"event: {ev.get('type') or 'message'}")
    lines.append("data: " + json.dumps(ev, ensure_ascii=False, default=str, separators=(",", ":")))
    return "\n".join(lines) + "\n\n"


def _last_seen(request: Request, since: str | None) -> int | None:
    """Sequence number the client already has: ?since= or the Last-Event-ID header of a reconnecting EventSource."""
    for raw in (since, request.headers.get("last-event-id")):
        if raw is None or str(raw).strip() == "":
            continue
        try:
            return max(0, int(str(raw).strip()))
        except ValueError:
            continue
    return None


def _hello(holder: Any, bus: Any, last: int | None, seq_now: int, oldest: int, replay: int) -> dict[str, Any]:
    """The first frame: a small status snapshot so the page is right before the first real event arrives."""
    rt = holder.rt
    try:
        status = rt.status_lite()
    except Exception as e:  # noqa: BLE001 - e.g. a profile switch closed the database under us: the page asks again
        log.debug("hello: status snapshot failed: %s", e)
        status = {}
    gap = False
    if last is not None:
        # events after `last` that the bus no longer holds, or ids from a previous process
        gap = last > seq_now or (seq_now > last and (oldest == 0 or oldest > last + 1))
    return {"type": "hello", "seq": seq_now, "ts": datetime.now().isoformat(timespec="milliseconds"),
            "status": status, "inbox": {"open": int(status.get("inbox_open") or 0)}, "replay": int(replay), "gap": bool(gap)}


async def _stream(sub: Any, hello: dict[str, Any], once: bool, named: bool) -> AsyncIterator[str]:
    try:
        yield _frame(hello, named, with_id=False)
        if once:
            for ev in sub.drain():
                yield _frame(ev, named)
            return
        loop = asyncio.get_running_loop()
        wake = asyncio.Event()          # created inside this request's loop; never shared

        def notify() -> None:           # runs in the publishing thread
            try:
                loop.call_soon_threadsafe(wake.set)
            except RuntimeError:        # the loop is gone (client went away, server stopping)
                pass

        sub.on_event = notify
        last_beat = time.monotonic()
        while True:
            batch = sub.drain()
            for ev in batch:
                yield _frame(ev, named)
            if batch:
                last_beat = time.monotonic()
            if sub.closed:
                break
            idle = time.monotonic() - last_beat
            if idle >= HEARTBEAT_S:
                yield ": keep-alive\n\n"
                last_beat = time.monotonic()
                idle = 0.0
            try:
                await asyncio.wait_for(wake.wait(), timeout=max(0.05, min(POLL_S, HEARTBEAT_S - idle)))
            except asyncio.TimeoutError:
                pass
            wake.clear()
    finally:
        sub.on_event = None
        sub.close()


@router.get("/events")
async def event_stream(request: Request, once: int = 0, named: int = 0, since: str | None = None) -> Any:
    holder = request.app.state.holder
    try:
        enabled = bool(holder.rt.cfg.ui.get("events", True))
    except Exception:  # noqa: BLE001
        enabled = True
    if not enabled:
        return JSONResponse({"error": "事件推送已关闭（设置 ui.events），页面改用定时刷新"}, status_code=404)
    bus = _bus(holder)
    last = _last_seen(request, since)
    seq_now, oldest = int(bus.last_seq), int(bus.oldest_seq)
    sub = bus.subscribe(since=last)
    try:
        import anyio

        replay = int(getattr(sub, "pending", 0) or 0)
        hello = await anyio.to_thread.run_sync(_hello, holder, bus, last, seq_now, oldest, replay)
    except BaseException:
        sub.close()
        raise
    return StreamingResponse(_stream(sub, hello, bool(once), bool(named)), media_type="text/event-stream", headers=SSE_HEADERS)


@router.get("/metrics")
def usage_metrics(request: Request, days: str | None = None) -> Any:
    try:
        n = int(days) if days not in (None, "") else 7
    except (TypeError, ValueError):
        n = 0
    if not 1 <= n <= MAX_METRIC_DAYS:
        return JSONResponse({"error": f"days 应为 1–{MAX_METRIC_DAYS} 的整数"}, status_code=400)
    return metrics.summary(request.app.state.holder.rt.db, n)
