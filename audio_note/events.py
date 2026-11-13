"""In-process event bus: the recorder, the scheduler, jobs and request handlers publish what happened;
the web UI listens through GET /api/events (server-sent events) instead of polling.

    bus = EventBus()
    n = bus.publish("segment", id=12, status="new")      # any thread; never blocks, never raises
    sub = bus.subscribe()                                 # or subscribe(since=n) to replay what was missed
    ev = sub.get(timeout=15)                              # dict or None (timeout / closed)
    sub.close()
    bus.close()                                           # wakes and ends every subscription

An event is a plain dict:  {"seq": 17, "id": ..., "type": "segment", "ts": "2026-11-11T21:30:05.123", ...data}
`seq` is the bus-wide sequence number (the SSE `id:` field, used for Last-Event-ID replay). `id` is whatever the
publisher passed as `id=` (the segment / utterance / session the event is about); without one it equals `seq`.

Event types (docs/SPEC_v1.md section 4.3):
    recording {on, speaking?}      segment {id, status}            utterances {day, segment_id, ids}
    utterance {id, change}         job {name, state, result?, error?, source?}
    session {id, change}           inbox {open, kind?}             memory {change}       daily {day}
    speakers {change}              lexicon {change}                config {}             profile {name}

The bus belongs to the RuntimeHolder and is handed to every Runtime (a profile switch keeps the same bus, so
open event streams survive it). Code that has no bus uses `events.NULL`, which accepts and drops everything.
Nothing here is tied to an asyncio loop: subscriptions are thread-safe queues with an optional wake-up callback.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime
from typing import Any, Callable

log = logging.getLogger(__name__)

#: events kept for replay (Last-Event-ID / subscribe(since=...))
KEEP_EVENTS = 500
#: events a subscription buffers before the oldest are dropped (a stalled browser tab must not grow memory)
SUBSCRIPTION_BUFFER = 1000

EVENT_TYPES = ("recording", "segment", "utterances", "utterance", "job", "session", "inbox", "memory", "daily",
               "speakers", "lexicon", "config", "profile")


class Subscription:
    """One listener's queue. get() blocks a thread; an asyncio consumer sets `on_event` to a callable that
    schedules its own wake-up (it is called from the publishing thread, after an event was queued and when
    the subscription closes) and reads with drain()."""

    def __init__(self, bus: "EventBus | None" = None, maxsize: int = SUBSCRIPTION_BUFFER):
        self._bus = bus
        self._q: deque[dict[str, Any]] = deque(maxlen=max(1, int(maxsize)))
        self._cond = threading.Condition()
        self._closed = False
        self.dropped = 0                                   # events lost because the consumer was too slow
        self.on_event: Callable[[], None] | None = None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pending(self) -> int:
        """Events queued and not read yet."""
        with self._cond:
            return len(self._q)

    def _push(self, ev: dict[str, Any]) -> None:
        with self._cond:
            if self._closed:
                return
            if len(self._q) == self._q.maxlen:
                self.dropped += 1
            self._q.append(ev)
            self._cond.notify_all()
        self._notify()

    def _notify(self) -> None:
        cb = self.on_event
        if cb is None:
            return
        try:
            cb()
        except Exception as e:  # noqa: BLE001 - e.g. the consumer's event loop is gone
            log.debug("event subscriber wake-up failed: %s", e)

    def get(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Next event, waiting up to `timeout` seconds (None = until one arrives or the subscription closes).
        Returns None on timeout and once the subscription is closed and empty."""
        with self._cond:
            if not self._q and not self._closed:
                self._cond.wait(timeout)
            return self._q.popleft() if self._q else None

    def drain(self) -> list[dict[str, Any]]:
        """Everything queued right now, without waiting."""
        with self._cond:
            out = list(self._q)
            self._q.clear()
            return out

    def close(self) -> None:
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._cond.notify_all()
        bus = self._bus
        if bus is not None:
            bus._forget(self)
        self._notify()

    def __enter__(self) -> "Subscription":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class EventBus:
    def __init__(self, keep: int = KEEP_EVENTS):
        self._lock = threading.Lock()
        self._seq = 0
        self._ring: deque[dict[str, Any]] = deque(maxlen=max(1, int(keep)))
        self._subs: list[Subscription] = []
        self._closed = False

    # --- publishing ---------------------------------------------------------------
    def publish(self, type: str, **data: Any) -> int:  # noqa: A002 - "type" is the event vocabulary of the SPEC
        """Record an event and hand it to every subscriber. Thread-safe, never blocks on a consumer and
        never raises. Returns the event's sequence number."""
        try:
            with self._lock:
                self._seq += 1
                seq = self._seq
                ev: dict[str, Any] = {"seq": seq, "id": seq, "type": str(type),
                                      "ts": datetime.now().isoformat(timespec="milliseconds")}
                ev.update(data)
                ev["seq"], ev["type"] = seq, str(type)       # data must not overwrite the envelope
                self._ring.append(ev)
                subs = list(self._subs)
            for s in subs:
                s._push(ev)
            return seq
        except Exception as e:  # noqa: BLE001 - a broken event must never break the work that reported it
            log.debug("event %s dropped: %s", type, e)
            return 0

    # --- listening ----------------------------------------------------------------
    def subscribe(self, since: int | None = None, maxsize: int = SUBSCRIPTION_BUFFER) -> Subscription:
        """A new subscription. With `since` (a sequence number the listener already saw) the retained
        events after it are queued first; older ones are gone (see oldest_seq). On a closed bus the
        subscription is returned closed."""
        sub = Subscription(self, maxsize)
        with self._lock:
            if self._closed:
                sub._bus = None
                sub._closed = True
                return sub
            if since is not None:
                for ev in self._ring:
                    if ev["seq"] > int(since):
                        sub._q.append(ev)
            self._subs.append(sub)
        return sub

    def _forget(self, sub: Subscription) -> None:
        with self._lock:
            try:
                self._subs.remove(sub)
            except ValueError:
                pass

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._seq

    @property
    def oldest_seq(self) -> int:
        """Sequence number of the oldest event still kept (0 when none was published yet)."""
        with self._lock:
            return int(self._ring[0]["seq"]) if self._ring else 0

    @property
    def subscribers(self) -> int:
        with self._lock:
            return len(self._subs)

    @property
    def closed(self) -> bool:
        return self._closed

    def recent(self, since: int | None = None) -> list[dict[str, Any]]:
        """Retained events (after `since`), oldest first."""
        with self._lock:
            return [ev for ev in self._ring if since is None or ev["seq"] > int(since)]

    # --- lifecycle ------------------------------------------------------------------
    def close(self) -> None:
        """End every subscription (their get() returns None, their on_event fires) and refuse new ones.
        UIServer.stop() calls this first so open event streams finish and the server can shut down."""
        with self._lock:
            self._closed = True
            subs, self._subs = self._subs, []
        for s in subs:
            s._bus = None
            s.close()

    def reopen(self) -> None:
        """Accept subscriptions again after close() (a web server started again on the same holder)."""
        with self._lock:
            self._closed = False


class _NullBus:
    """Stand-in where no bus was handed over: publishing is a no-op, subscriptions are born closed."""

    closed = True
    last_seq = 0
    oldest_seq = 0
    subscribers = 0

    def publish(self, type: str, **data: Any) -> int:  # noqa: A002
        return 0

    def subscribe(self, since: int | None = None, maxsize: int = SUBSCRIPTION_BUFFER) -> Subscription:
        sub = Subscription(None, 1)
        sub._closed = True
        return sub

    def recent(self, since: int | None = None) -> list[dict[str, Any]]:
        return []

    def close(self) -> None:
        return None

    def reopen(self) -> None:
        return None


NULL = _NullBus()

__all__ = ["EventBus", "Subscription", "NULL", "EVENT_TYPES", "KEEP_EVENTS"]
