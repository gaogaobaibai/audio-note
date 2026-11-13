"""Background scheduler for the always-on service.

Jobs (all optional, injectable for tests):
  process()           every cfg.pipeline.process_interval_min   -> speech.pipeline.Processor
  cleanup()           every cfg.storage.cleanup_interval_min    -> audio.storage.cleanup
  daily(day, push)    once a day at cfg.analysis.daily_time      -> analysis.daily.DailyAnalyzer
                      plus a catch-up run for yesterday when kv.last_daily_day is older, and a
                      once-a-day re-check of yesterday: when utterances were added after its report
                      (speech after daily_time, segments processed later) it is regenerated + pushed.
                      `process` always runs right before `daily` in the same tick, and a day with
                      zero utterances is only marked done when none of its segments is still pending.

Last-run times live in the kv table (last_process_at, last_cleanup_at, last_daily_day,
last_daily_at, last_daily_recheck_day) so restarts do not re-run everything. Due-time logic is in pure functions
(parse_hhmm / next_daily_due / is_due / daily_days_due) so it can be unit-tested without
threads or a clock.

1.0 additions (docs/SPEC_v1.md section 4.6):
  realtime            the recorder calls notify_segment(id) when a segment file is closed; the loop wakes up at once
                      and processes exactly those segments (pipeline.realtime; min_segment_age_s does not apply,
                      the file is complete). The interval-driven `process` stays as the safety net.
  chain()             post-processing (sessions -> refine -> episodes, chain.run_chain) after every process run, on a
                      tick that finds finished-but-untreated sessions, and right before a daily digest.
  consolidate()       memory consolidation right after a daily digest; brief() once a day at analysis.brief_time.
  deferral            governor.defer_reason(cfg) (on battery with performance.defer_on_battery) postpones the local
                      heavy work (`process`); `deferred` holds the reason for status().
  events / history    job state changes are published on the event bus and written to job_runs.
None of the new work shows up in run_once()'s return value: it lists process / cleanup / daily / push as before.
"""
from __future__ import annotations

import inspect
import logging
import sqlite3
import threading
import time
from collections import deque
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

from . import events as events_mod
from .config import Config
from .db import DB, now_iso

log = logging.getLogger(__name__)

KV_PROCESS = "last_process_at"
KV_CLEANUP = "last_cleanup_at"
KV_DAILY_DAY = "last_daily_day"
KV_DAILY_AT = "last_daily_at"
KV_DAILY_RECHECK = "last_daily_recheck_day"   # yesterday already re-checked for late content
KV_FAILED_REQUEUE_AT = "last_failed_requeue_at"   # last time transient 'failed' segments were put back to 'new'
KV_FAILED_ATTEMPTS = "failed_retry_attempts"      # {segment_id: re-queue count}
KV_PUSH_PENDING = "daily_push_pending"            # {day: {"attempts": n, "last": iso}} digests whose push failed
KV_DAILY_PARTIAL = "daily_partial_attempts"       # {day: n} how often a day's digest came back partial

DAILY_RETRY_MIN = 30.0   # after a failed daily run, wait this long before trying again
REQUEUE_INTERVAL_MIN = 60.0          # transient failures are re-queued at most this often
DEFAULT_MAX_FAILED_RETRIES = 24      # per segment (cfg.pipeline.max_failed_retries); ~1 day of hourly retries
PUSH_RETRY_MIN = 30.0                # a digest whose push failed is re-sent at most this often
MAX_PUSH_RETRIES = 8
MAX_PARTIAL_RETRIES = 6              # regenerate a partial digest at most this often before accepting it
DEFAULT_CATCHUP_DAYS = 7             # cfg.analysis.catchup_days: how many missed days get a report after downtime

KV_BRIEF_DAY = "last_brief_day"      # the day whose morning brief was already attempted
WAKE_POLL_S = 0.25                   # while idle the loop looks at its stop event this often (a wake-up is immediate)
CHAIN_LIMIT = 3                      # sessions a chain step may treat per tick (new audio is never kept waiting long)
CHAIN_RETRY_S = 300.0                # pending chain work that made no progress is tried again after this long
CHAIN_PROBE_S = 60.0                 # how often to look whether any chain step is installed at all
DEFAULT_SESSION_GAP_MIN = 5.0        # cfg.memory.session_gap_min until package B3 adds the key
BRIEF_WINDOW_H = 6.0                 # a morning brief more than this late is skipped for the day


def _jsonable(v: Any) -> Any:
    """A job result reduced to what JSON can carry (paths / dates / sets become strings / lists)."""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, Path):
        return str(v)
    return str(v)


def _is_noop_result(name: str, result: Any) -> bool:
    """True for the outcome of a periodic run that found nothing to do (kept out of the job history)."""
    if name == "process":
        return not (isinstance(result, dict) and int(result.get("processed") or 0) > 0)
    if name == "cleanup":
        return not (isinstance(result, dict) and any(isinstance(v, (int, float)) and v for v in result.values()))
    if name == "chain":
        return not (isinstance(result, dict) and (result.get("ran") or result.get("failed")))
    return False


# --- pure helpers ---------------------------------------------------------

def parse_hhmm(hhmm: str) -> tuple[int, int]:
    """'22:30' -> (22, 30). Raises ValueError on bad input."""
    s = str(hhmm).strip()
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"bad time {hhmm!r}, expected HH:MM")
    h, m = int(parts[0]), int(parts[1])
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"bad time {hhmm!r}")
    return h, m


def next_daily_due(now: datetime, hhmm: str) -> datetime:
    """Next wall-clock moment at hh:mm that is strictly after `now` (today or tomorrow)."""
    h, m = parse_hhmm(hhmm)
    due = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if due <= now:
        due += timedelta(days=1)
    return due


def _to_dt(v: datetime | str | None) -> datetime | None:
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v))
    except ValueError:
        return None


def is_due(last: datetime | str | None, interval_min: float, now: datetime | None = None) -> bool:
    """True when `last` is unknown/unparseable or at least `interval_min` minutes before `now`.
    A non-positive interval disables the job (never due)."""
    if interval_min is None or float(interval_min) <= 0:
        return False
    now = now or datetime.now()
    last_dt = _to_dt(last)
    if last_dt is None:
        return True
    if last_dt > now:               # clock went backwards: treat as due
        return True
    return (now - last_dt) >= timedelta(minutes=float(interval_min))


def daily_days_due(now: datetime, hhmm: str, last_daily_day: str | None, max_catchup_days: int = 1) -> list[str]:
    """Which days need a daily report right now, oldest first.

    * catch-up:  every complete day after `last_daily_day` up to yesterday, at most `max_catchup_days`
                 of them (the most recent ones); with no previous report only yesterday. Runs regardless
                 of the time of day (those days are complete).
    * today:     once `now` is past hh:mm and today's report has not been made yet.
    """
    today: date = now.date()
    yesterday = today - timedelta(days=1)
    last = (last_daily_day or "").strip()
    out: list[str] = []
    span = max(1, int(max_catchup_days or 1))
    if not last:
        out.append(yesterday.isoformat())
    elif last < yesterday.isoformat():
        try:
            first = date.fromisoformat(last) + timedelta(days=1)
        except ValueError:
            first = yesterday
        first = max(first, yesterday - timedelta(days=span - 1))
        d = first
        while d <= yesterday:
            out.append(d.isoformat())
            d += timedelta(days=1)
    h, m = parse_hhmm(hhmm)
    if now >= now.replace(hour=h, minute=m, second=0, microsecond=0) and (not last or last < today.isoformat()):
        out.append(today.isoformat())
    return out


def requeue_transient_failures(db: DB, max_attempts: int = DEFAULT_MAX_FAILED_RETRIES) -> int:
    """Put 'failed' segments whose error looks transient (API outage, timeout, 5xx/429) back to 'new'
    so the next process run retries them; each segment at most `max_attempts` times (count kept in
    kv[KV_FAILED_ATTEMPTS]). Returns the number re-queued. Errors that look deterministic
    (missing file, bad audio, HTTP 400/413) are left for a manual `process --retry-failed`."""
    from .audio.storage import is_transient_error_text

    rows = db.query("SELECT id, error FROM segments WHERE status='failed'")
    raw = db.get_kv(KV_FAILED_ATTEMPTS) or {}
    attempts: dict[str, int] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                attempts[str(k)] = int(v)
            except (TypeError, ValueError):
                continue
    # forget counters of segments that are finished (done / no_speech / deleted)
    active = {str(r["id"]) for r in db.query("SELECT id FROM segments WHERE status IN ('new','processing','failed')")}
    attempts = {k: v for k, v in attempts.items() if k in active}
    n = 0
    exhausted = 0
    for r in rows:
        key = str(r["id"])
        if not is_transient_error_text(r["error"]):
            continue
        if attempts.get(key, 0) >= max(1, int(max_attempts)):
            exhausted += 1
            continue
        attempts[key] = attempts.get(key, 0) + 1
        db.execute("UPDATE segments SET status='new', updated_at=? WHERE id=? AND status='failed'", (now_iso(), int(r["id"])))
        n += 1
    db.set_kv(KV_FAILED_ATTEMPTS, attempts)
    if n or exhausted:
        log.info("re-queued %d failed segment(s) with transient errors (%d exhausted their %d retries)", n, exhausted, max_attempts)
    return n


def _result_is_partial(res: Any) -> bool:
    """True when a daily job returned a digest flagged partial ((md, js) tuple or js dict)."""
    js = res[1] if isinstance(res, tuple) and len(res) == 2 else res
    return bool(isinstance(js, dict) and js.get("partial"))


# --- scheduler --------------------------------------------------------------

class Scheduler:
    """Thread loop running the periodic jobs. Jobs are plain callables so tests can inject fakes:

        jobs = {"process": fn(), "cleanup": fn(), "daily": fn(day: str, push: bool)}
    Missing entries fall back to the real implementations, imported lazily.
    """

    def __init__(self, cfg: Config, db: DB, stop_event: threading.Event | None = None,
                 jobs: dict[str, Callable] | None = None, tick_s: float = 5.0,
                 engine_factory: Callable[[], Any] | None = None, events: Any = None):
        self.cfg = cfg
        self.db = db
        self.stop_event = stop_event or threading.Event()
        self.tick_s = float(tick_s)
        self.events = events if events is not None else events_mod.NULL
        #: set by notify_segment() (recorder thread) and by whoever wants the loop to look right now
        self.wake = threading.Event()
        self._wake_ids: deque[int] = deque()
        self._wake_all = False
        self._wake_lock = threading.Lock()
        #: held while a tick or a manually triggered job runs, so the web UI's "process now" and the
        #: periodic tick never work on the same segments at the same time
        self.lock = threading.RLock()
        self.engine_factory = engine_factory   # shared SpeakerEngine (the UI's label/relabel uses the same one)
        jobs = jobs or {}
        self.jobs: dict[str, Callable] = {
            "process": jobs.get("process") or self._default_process,
            "cleanup": jobs.get("cleanup") or self._default_cleanup,
            "daily": jobs.get("daily") or self._default_daily,
            "push": jobs.get("push") or self._default_push,     # fn(day) -> {channel: error|None}
            "chain": jobs.get("chain") or self._default_chain,                  # fn() -> chain.run_chain result
            "consolidate": jobs.get("consolidate") or self._default_consolidate,   # fn() -> dict
            "brief": jobs.get("brief") or self._default_brief,                  # fn() -> dict
        }
        self._process_injected = jobs.get("process") is not None
        self.runs: dict[str, int] = {"process": 0, "cleanup": 0, "daily": 0}
        self.errors: dict[str, int] = {"process": 0, "cleanup": 0, "daily": 0}
        #: counters of the 1.0 work (kept apart from runs / errors, whose shape the status page relies on)
        self.extra_runs: dict[str, int] = {"realtime": 0, "chain": 0, "consolidate": 0, "brief": 0}
        self.extra_errors: dict[str, int] = {"realtime": 0, "chain": 0, "consolidate": 0, "brief": 0}
        #: why the local heavy work is postponed right now (governor.defer_reason), or None
        self.deferred: str | None = None
        self._daily_retry_after: datetime | None = None
        self._processor = None
        self._proc_events = False        # the cached Processor publishes segment / utterances events itself
        self._llm = None                 # the MiMo client shared by processor / daily / chain (rebuilt after a reload)
        self._vad = None
        self._client_warned = False
        self._chain_sig: tuple[int, int, int] | None = None     # pending-work signature after the last chain run
        self._chain_retry_at = 0.0
        self._chain_probe: tuple[float, bool] = (0.0, False)
        self._brief_warned = False

    # --- loop ---------------------------------------------------------------
    def run(self) -> None:
        log.info("scheduler started: process every %s min, cleanup every %s min, daily at %s",
                 self.cfg.pipeline.process_interval_min, self.cfg.storage.cleanup_interval_min, self.cfg.analysis.daily_time)
        # the loop owns the event it was started with: a later reset_stop_event() (service restarted in the
        # same process while this tick was still finishing) must not revive this thread
        stop_event = self.stop_event
        while not stop_event.is_set():
            try:
                self.run_once()
            except Exception as e:  # noqa: BLE001  (never let the loop die)
                log.exception("scheduler tick failed: %s", e)
            self._sleep(stop_event)
        log.info("scheduler stopped")

    def _sleep(self, stop_event: threading.Event) -> None:
        """Wait for the next tick: returns at once when `wake` is set (a segment was just recorded), after
        tick_s otherwise, and within WAKE_POLL_S of the stop event."""
        deadline = time.monotonic() + self.tick_s
        while not stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0 or self.wake.wait(min(remaining, WAKE_POLL_S)):
                return

    def notify_segment(self, seg_id: int | None = None) -> None:
        """Called (from any thread; the recorder's, typically) when a segment was stored: the loop wakes up
        and, with pipeline.realtime on, processes it right away. Without an id: "run `process` now"."""
        with self._wake_lock:
            if seg_id is None:
                self._wake_all = True
            else:
                self._wake_ids.append(int(seg_id))
        self.wake.set()

    def _take_wake(self) -> tuple[list[int], bool]:
        with self._wake_lock:
            self.wake.clear()
            ids, self._wake_ids = list(dict.fromkeys(self._wake_ids)), deque()
            every, self._wake_all = self._wake_all, False
        return ids, every

    def run_once(self, now: datetime | None = None) -> list[str]:
        """Run every job that is due at `now`. Returns the names of the jobs that ran."""
        with self.lock:
            return self._run_once(now)

    def reset_stop_event(self, stop_event: threading.Event) -> None:
        """Use a fresh stop event (the service was stopped and started again in the same process). The
        cached Processor is dropped rather than re-pointed: a tick that is still finishing keeps the
        processor whose stop event is set, the next tick builds a new one on the new event."""
        self.stop_event = stop_event
        self._processor = None

    def reload_config(self, cfg: Config) -> None:
        """Pick up an edited config: the cached Processor (and its MiMo client) is rebuilt on next use."""
        self.cfg = cfg
        self._processor = None
        self._llm = None
        self._client_warned = False
        self._brief_warned = False
        self._chain_sig = None

    def processor(self) -> Any | None:
        """The shared speech Processor, built on first use; None while no MiMo API key is configured."""
        if self._processor is None:
            client = self._client()
            if client is None:
                return None
            from .speech.pipeline import Processor

            engine = None
            if self.engine_factory is not None:
                try:
                    engine = self.engine_factory()
                except Exception as e:  # noqa: BLE001 - the Processor builds its own when needed
                    log.warning("shared speaker engine unavailable (%s); processor builds its own", e)
            extra: dict[str, Any] = {}
            try:
                # the split pipeline (package A4) may take the event bus; older Processors (and test doubles) do not
                if "events" in inspect.signature(Processor).parameters:
                    extra["events"] = self.events
            except (TypeError, ValueError):
                pass
            self._proc_events = "events" in extra
            self._processor = Processor(self.cfg, self.db, client, engine=engine, stop_event=self.stop_event, **extra)
        return self._processor

    def _run_once(self, now: datetime | None = None) -> list[str]:
        now = now or datetime.now()
        ran: list[str] = []
        ids, every = self._take_wake()
        deferred = self._defer()
        fresh = False        # new sentences may have been written during this tick: the chain should look
        chained = False
        realtime = bool(self.cfg.pipeline.get("realtime", True))
        if (every or (ids and realtime)) and not deferred and not self.stop_event.is_set():
            if every or self._process_injected:
                self._run_job("process", now)
                ran.append("process")
                fresh = True
            else:
                fresh = self._run_realtime(ids)
        if "process" not in ran and not deferred \
                and is_due(self.db.get_kv(KV_PROCESS), float(self.cfg.pipeline.process_interval_min), now):
            self._requeue_failed(now)
            self._run_job("process", now)
            ran.append("process")
            fresh = True
        if self.stop_event.is_set():
            return ran
        if fresh or self._chain_pending(now):
            self._run_chain(now)
            chained = True
        if self.stop_event.is_set():
            return ran
        if bool(self.cfg.storage.cleanup_enabled) and is_due(self.db.get_kv(KV_CLEANUP), float(self.cfg.storage.cleanup_interval_min), now):
            self._run_job("cleanup", now)
            ran.append("cleanup")
        if self.stop_event.is_set():
            return ran
        dailies = 0
        if self._daily_due(now):
            days = self._days_due(now)
            if days and "process" not in ran and not deferred:
                # transcribe what was recorded in the last minutes before summarising the day
                self._run_job("process", now)
                ran.append("process")
                chained = False
            if days and not chained and not self.stop_event.is_set():
                self._run_chain(now)       # the digest reuses session summaries: bring them up to date first
            for day in days:
                if self.stop_event.is_set():
                    break
                if self._run_daily(day, now):
                    ran.append("daily")
                    dailies += 1
        if not self.stop_event.is_set() and self._recheck_yesterday(now):
            ran.append("daily")
            dailies += 1
        if not self.stop_event.is_set() and self._rebuild_forgotten(now):
            ran.append("daily")
            dailies += 1
        if dailies and not self.stop_event.is_set():
            self._run_extra("consolidate")
        if not self.stop_event.is_set() and self._retry_pushes(now):
            ran.append("push")
        if not self.stop_event.is_set() and self._brief_due(now):
            self._run_extra("brief")
        return ran

    # --- 1.0: deferral, realtime, chain, brief ------------------------------------------
    def _defer(self) -> str | None:
        """Ask the governor whether local heavy work should wait (on battery ...); remembers the answer for status()."""
        try:
            from . import governor

            reason = governor.defer_reason(self.cfg)
        except Exception as e:  # noqa: BLE001 - never let a power query stop the loop
            log.debug("defer_reason failed: %s", e)
            reason = None
        if reason != self.deferred:
            if reason:
                log.info("local processing deferred: %s", reason)
            else:
                log.info("local processing resumed")
            self.deferred = reason
        return reason

    def _max_utterance_id(self) -> int:
        try:
            row = self.db.one("SELECT COALESCE(MAX(id), 0) AS n FROM utterances")
            return int(row["n"]) if row else 0
        except sqlite3.Error:
            return 0

    def announce_new(self, since_id: int, seg_ids: Iterable[int] = ()) -> None:
        """Publish what a process run produced, unless the Processor publishes its own events: one `segment`
        event per segment touched and one `utterances` event per (day, segment) with the new sentence ids."""
        if self._proc_events or self.events is events_mod.NULL:
            return
        try:
            rows = self.db.query("SELECT id, day, segment_id FROM utterances WHERE id > ? ORDER BY id", (int(since_id),))
            groups: dict[tuple[str, int], list[int]] = {}
            for r in rows:
                groups.setdefault((r["day"], int(r["segment_id"])), []).append(int(r["id"]))
            touched = {int(i) for i in seg_ids} | {seg for _day, seg in groups}
            for sid in sorted(touched):
                seg = self.db.get_segment(sid)
                if seg is not None:
                    self.events.publish("segment", id=sid, status=seg["status"])
            for (day, seg), utt_ids in groups.items():
                self.events.publish("utterances", day=day, segment_id=seg, ids=utt_ids)
        except Exception as e:  # noqa: BLE001 - events are best effort
            log.debug("announce_new failed: %s", e)

    def _requeue_stopped(self) -> int:
        """Segments a stop request interrupted (stored as failed with db.STOPPED_ERROR) go back to 'new'."""
        try:
            n = self.db.requeue_stopped()
        except Exception as e:  # noqa: BLE001 - e.g. the database is being closed
            log.debug("requeue_stopped failed: %s", e)
            return 0
        if n:
            log.info("%d segment(s) interrupted by a stop are queued again", n)
        return n

    def _run_realtime(self, ids: list[int]) -> bool:
        """Process the segments the recorder just reported, without waiting for process_interval_min or
        min_segment_age_s. Returns True when at least one segment went through the pipeline."""
        proc = self.processor()
        if proc is None:
            return False                       # no API key: the periodic path reports that
        fn = getattr(proc, "process_segment", None)
        if not callable(fn):
            return False
        self._requeue_stopped()
        since = self._max_utterance_id()
        done: list[int] = []
        t0 = time.monotonic()
        for sid in ids:
            if self.stop_event.is_set():
                break
            seg = self.db.get_segment(sid)
            if seg is None or seg["status"] != "new":
                continue                       # already handled (periodic run, manual job) or gone
            done.append(sid)
            try:
                self.db.execute("UPDATE segments SET realtime=1 WHERE id=?", (sid,))
                if not self._proc_events:
                    self.events.publish("segment", id=sid, status="processing")
                fn(sid)
            except Exception as e:  # noqa: BLE001 - process_segment stored the error and logged the traceback
                if self.stop_event.is_set() or str(e) == "stopped":
                    # a stop request cut it short (pipeline: Stopped / RuntimeError("stopped")): not a failure,
                    # the segment is (or is about to be put) back to 'new'
                    log.info("segment #%d: stop requested during realtime processing", sid)
                    break
                self.extra_errors["realtime"] += 1
                log.error("segment #%d failed (realtime): %s", sid, e)
        self._requeue_stopped()
        if not done:
            return False
        self.extra_runs["realtime"] += 1
        self.announce_new(since, done)
        log.info("realtime: %d segment(s) processed in %.1fs", len(done), time.monotonic() - t0)
        return True

    def _session_gap_min(self) -> float:
        try:
            return float(self.cfg.memory.get("session_gap_min", DEFAULT_SESSION_GAP_MIN))
        except (TypeError, ValueError, AttributeError):
            return DEFAULT_SESSION_GAP_MIN

    def _chain_installed(self) -> bool:
        t, ok = self._chain_probe
        if time.monotonic() - t < CHAIN_PROBE_S and t:
            return ok
        try:
            from . import chain

            ok = chain.has_steps()
        except Exception:  # noqa: BLE001
            ok = False
        self._chain_probe = (time.monotonic(), ok)
        return ok

    def _chain_signature(self, now: datetime) -> tuple[int, int, int] | None:
        """(sentences without a session, open sessions whose gap has passed, closed sessions still to refine or
        digest), or None when nothing is waiting."""
        try:
            cutoff = (now - timedelta(minutes=self._session_gap_min())).isoformat(timespec="seconds")
            a = int(self.db.one("SELECT COUNT(*) AS n FROM utterances WHERE session_id IS NULL")["n"])
            b = int(self.db.one("SELECT COUNT(*) AS n FROM sessions WHERE status='open' AND replace(ended_at, ' ', 'T') <= ?",
                                (cutoff,))["n"])
            c = int(self.db.one("SELECT COUNT(*) AS n FROM sessions WHERE status='closed' AND ("
                                " COALESCE(refined_hash, '') <> COALESCE(raw_hash, '')"
                                " OR COALESCE(digested_hash, '') <> COALESCE(text_hash, ''))")["n"])
        except sqlite3.Error as e:
            log.debug("chain signature unavailable: %s", e)
            return None
        return (a, b, c) if (a or b or c) else None

    def _chain_pending(self, now: datetime) -> bool:
        """True when the chain has something to do and is not backing off: work that did not shrink since the
        last run (no API key, a step that keeps failing) is only tried again after CHAIN_RETRY_S."""
        if self.jobs["chain"] == self._default_chain and not self._chain_installed():
            return False
        sig = self._chain_signature(now)
        if sig is None:
            self._chain_sig = None
            return False
        if sig == self._chain_sig and time.monotonic() < self._chain_retry_at:
            return False
        return True

    def _run_chain(self, now: datetime) -> Any:
        """Run the post-processing chain once (never raises)."""
        if self.jobs["chain"] == self._default_chain and not self._chain_installed():
            return None
        started = now_iso()
        t0 = time.monotonic()
        try:
            res = self.jobs["chain"]()
        except Exception as e:  # noqa: BLE001
            self.extra_errors["chain"] += 1
            log.exception("chain failed: %s", e)
            self._job_event("chain", "failed", started, error=f"{type(e).__name__}: {e}")
            res = None
        else:
            self.extra_runs["chain"] += 1
            if not _is_noop_result("chain", res):
                log.info("chain done in %.1fs: %s", time.monotonic() - t0, res)
                self._job_event("chain", "done", started, result=res)
        # remember what is still waiting: the same picture next tick means "no progress", so back off
        self._chain_sig = self._chain_signature(now)
        self._chain_retry_at = time.monotonic() + CHAIN_RETRY_S
        return res

    def _run_extra(self, name: str) -> Any:
        """Run consolidate / brief (never raises; a feature that is not installed is skipped quietly)."""
        from . import chain

        started = now_iso()
        t0 = time.monotonic()
        try:
            res = self.jobs[name]()
        except chain.StepUnavailable as e:
            log.debug("%s skipped: %s", name, e)
            return None
        except Exception as e:  # noqa: BLE001
            self.extra_errors[name] += 1
            log.exception("%s failed: %s", name, e)
            self._job_event(name, "failed", started, error=f"{type(e).__name__}: {e}")
            return None
        self.extra_runs[name] += 1
        log.info("%s done in %.1fs: %s", name, time.monotonic() - t0, res)
        self._job_event(name, "done", started, result=res)
        return res

    def _brief_due(self, now: datetime) -> bool:
        """Once a day, from analysis.brief_time on ('' = off; the key arrives with the memory package). The day
        is stamped before the brief runs, so a failing brief is not retried on every tick."""
        try:
            hhmm = str(self.cfg.analysis.get("brief_time", "") or "").strip()
        except AttributeError:
            return False
        if not hhmm:
            return False
        try:
            h, m = parse_hhmm(hhmm)
        except ValueError as e:
            if not self._brief_warned:
                log.error("analysis.brief_time invalid: %s", e)
                self._brief_warned = True
            return False
        today = now.date().isoformat()
        if str(self.db.get_kv(KV_BRIEF_DAY) or "") >= today:
            return False
        due = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if now < due:
            return False
        self.db.set_kv(KV_BRIEF_DAY, today)
        if now - due > timedelta(hours=BRIEF_WINDOW_H):
            log.info("morning brief for %s skipped: more than %.0f h late", today, BRIEF_WINDOW_H)
            return False
        return True

    def _job_event(self, name: str, state: str, started_at: str | None = None, result: Any = None,
                   error: str | None = None, source: str = "scheduler") -> None:
        """Publish a `job` event and, for a finished run worth keeping, write the job_runs row. Never raises."""
        try:
            data: dict[str, Any] = {"name": name, "state": state, "source": source}
            if result is not None:
                data["result"] = _jsonable(result)
            if error is not None:
                data["error"] = error
            self.events.publish("job", **data)
            if state in ("done", "failed") and started_at and not (state == "done" and _is_noop_result(name, result)):
                self.db.add_job_run(name, source, state, started_at, _jsonable(result) if result is not None else None, error)
        except Exception as e:  # noqa: BLE001 - history / events never break a job
            log.debug("job event %s/%s not recorded: %s", name, state, e)

    # --- due-time helpers -----------------------------------------------------
    def _catchup_days(self) -> int:
        try:
            return max(1, int(self.cfg.analysis.get("catchup_days", DEFAULT_CATCHUP_DAYS)))
        except (TypeError, ValueError):
            return DEFAULT_CATCHUP_DAYS

    def _days_due(self, now: datetime) -> list[str]:
        return daily_days_due(now, str(self.cfg.analysis.daily_time), self.db.get_kv(KV_DAILY_DAY), self._catchup_days())

    def _recheck_grace_s(self) -> float:
        """How long after midnight yesterday's recheck must wait: a segment that was still being spoken at
        00:00 is not in the DB yet (it is only inserted when it closes, up to max_segment_s later), then
        it has to age min_segment_age_s and be picked up by the next process tick."""
        try:
            a, p = self.cfg.audio, self.cfg.pipeline
            return float(a.get("max_segment_s", 300)) + float(p.get("min_segment_age_s", 20)) \
                + float(p.get("process_interval_min", 10)) * 60.0 + 60.0
        except (TypeError, ValueError, AttributeError):
            return 980.0

    def _recheck_yesterday(self, now: datetime) -> bool:
        """Once per day: yesterday's report was made at daily_time, but speech after that (and
        segments still pending at that moment) landed later. If the utterance count changed since the
        report was written (or there was no report because nothing was processed yet), regenerate and
        push it. Waits while yesterday still has unprocessed segments, and never runs in the first
        minutes after midnight (a segment open at 00:00 is invisible until it closes)."""
        if self._daily_retry_after is not None and now < self._daily_retry_after:
            return False
        yesterday = (now.date() - timedelta(days=1)).isoformat()
        if (self.db.get_kv(KV_DAILY_RECHECK) or "") >= yesterday:
            return False
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if now < midnight + timedelta(seconds=self._recheck_grace_s()):
            return False
        last = str(self.db.get_kv(KV_DAILY_DAY) or "")
        if last < yesterday:
            return False   # the normal catch-up path in run_once handles it
        if self.db.unfinished_segments_for_day(yesterday):
            return False   # try again on a later tick, after process has caught up
        n = self.db.one("SELECT COUNT(*) AS n FROM utterances WHERE day=?", (yesterday,))
        count = int(n["n"]) if n else 0
        row = self.db.one("SELECT utterance_count FROM daily_summaries WHERE day=?", (yesterday,))
        reported = int(row["utterance_count"] or 0) if row else 0
        if count == 0 or count == reported:
            self.db.set_kv(KV_DAILY_RECHECK, yesterday)
            return False
        log.info("daily %s: %d utterances now vs %d when reported; regenerating", yesterday, count, reported)
        ok = self._run_daily(yesterday, now)
        if ok:
            self.db.set_kv(KV_DAILY_RECHECK, yesterday)
        return ok

    # --- job runners --------------------------------------------------------
    def _run_job(self, name: str, now: datetime) -> None:
        key = KV_PROCESS if name == "process" else KV_CLEANUP
        # record the attempt first so a crashing job still backs off to the interval
        self.db.set_kv(key, now.isoformat(timespec="seconds"))
        t0 = time.monotonic()
        started = now_iso()
        since = 0
        if name == "process":
            self._requeue_stopped()
            since = self._max_utterance_id()
        self._job_event(name, "running")
        try:
            result = self.jobs[name]()
            self.runs[name] += 1
            log.info("job %s done in %.1fs%s", name, time.monotonic() - t0, f": {result}" if result is not None else "")
            self._job_event(name, "done", started, result=result)
        except Exception as e:  # noqa: BLE001
            self.errors[name] += 1
            log.exception("job %s failed: %s", name, e)
            self._job_event(name, "failed", started, error=f"{type(e).__name__}: {e}")
        finally:
            if name == "process":
                # a stop request in the middle of a segment leaves it 'failed: stopped': that is not a failure
                self._requeue_stopped()
                self.announce_new(since)

    def _daily_due(self, now: datetime) -> bool:
        if self._daily_retry_after is not None and now < self._daily_retry_after:
            return False
        try:
            return bool(daily_days_due(now, str(self.cfg.analysis.daily_time), self.db.get_kv(KV_DAILY_DAY)))
        except ValueError as e:
            log.error("analysis.daily_time invalid: %s", e)
            self._daily_retry_after = now + timedelta(hours=1)
            return False

    def _rebuild_forgotten(self, now: datetime) -> bool:
        """memory.forget removes the digest of every day it touched (it may quote what was forgotten) and lists the
        day under KV_REBUILD_DAYS: write that digest again from what is left, one day per tick, never pushed.
        A day with nothing left just leaves the list; a failure waits like any daily retry."""
        if self._daily_retry_after is not None and now < self._daily_retry_after:
            return False
        try:
            from .memory.forget import KV_REBUILD_DAYS
        except ImportError:
            return False
        days = [str(d) for d in (self.db.get_kv(KV_REBUILD_DAYS, []) or [])]
        if not days:
            return False
        day, rest = days[0], days[1:]
        if self.db.one("SELECT 1 FROM utterances WHERE day=? LIMIT 1", (day,)) is None:
            self.db.set_kv(KV_REBUILD_DAYS, rest)
            return False
        started = now_iso()
        self._job_event("daily", "running")
        try:
            self.jobs["daily"](day, False)
        except Exception as e:  # noqa: BLE001
            self.errors["daily"] += 1
            self._daily_retry_after = now + timedelta(minutes=DAILY_RETRY_MIN)
            log.warning("digest of %s (rebuilt after forgetting) failed, retry in %.0f min: %s", day, DAILY_RETRY_MIN, e)
            self._job_event("daily", "failed", started, error=f"{type(e).__name__}: {e}")
            return False
        self.db.set_kv(KV_REBUILD_DAYS, [d for d in (self.db.get_kv(KV_REBUILD_DAYS, []) or []) if str(d) != day])
        self._job_event("daily", "done", started, result={"day": day, "rebuilt": True})
        self.events.publish("daily", day=day)
        log.info("digest of %s written again after forgetting part of it", day)
        return True

    def _run_daily(self, day: str, now: datetime) -> bool:
        push = bool(self.cfg.analysis.push_daily)
        n = self.db.one("SELECT COUNT(*) AS n FROM utterances WHERE day=?", (day,))
        count = int(n["n"]) if n else 0
        if count == 0:
            pending = self.db.unfinished_segments_for_day(day)
            if pending:
                # nothing transcribed yet (no key / API outage / backlog): do NOT mark the day done,
                # the catch-up logic retries once the segments are processed
                log.info("daily %s: no utterances yet but %d segment(s) unprocessed; retry in %.0f min", day, pending, DAILY_RETRY_MIN)
                self._daily_retry_after = now + timedelta(minutes=DAILY_RETRY_MIN)
                return False
            log.info("daily %s: no utterances, skipping report", day)
            self._mark_daily(day, now)
            return False
        t0 = time.monotonic()
        started = now_iso()
        self._job_event("daily", "running")
        try:
            res = self.jobs["daily"](day, push)
        except Exception as e:  # noqa: BLE001
            self.errors["daily"] += 1
            self._daily_retry_after = now + timedelta(minutes=DAILY_RETRY_MIN)
            log.exception("daily %s failed (retry in %.0f min): %s", day, DAILY_RETRY_MIN, e)
            self._job_event("daily", "failed", started, error=f"{type(e).__name__}: {e}")
            return False
        self._job_event("daily", "done", started, result={"day": day, "partial": _result_is_partial(res)})
        self.events.publish("daily", day=day)
        if _result_is_partial(res):
            # some blocks failed in the map stage: the digest is on disk (flagged) but the day is not
            # done; regenerate after a pause, a bounded number of times
            n = self._bump_partial_attempts(day)
            if n < MAX_PARTIAL_RETRIES:
                self._daily_retry_after = now + timedelta(minutes=DAILY_RETRY_MIN)
                log.warning("daily %s: digest is partial (attempt %d/%d); regenerating in %.0f min",
                            day, n, MAX_PARTIAL_RETRIES, DAILY_RETRY_MIN)
                return False
            log.error("daily %s: still partial after %d attempts; keeping the partial digest", day, n)
        self.runs["daily"] += 1
        self._mark_daily(day, now)
        log.info("daily %s done in %.1fs (push=%s)", day, time.monotonic() - t0, push)
        return True

    def _bump_partial_attempts(self, day: str) -> int:
        raw = self.db.get_kv(KV_DAILY_PARTIAL) or {}
        attempts = {str(k): int(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
        n = attempts.get(day, 0) + 1
        attempts[day] = n
        keep_from = (date.fromisoformat(day) - timedelta(days=30)).isoformat()
        self.db.set_kv(KV_DAILY_PARTIAL, {k: v for k, v in attempts.items() if k >= keep_from})
        return n

    def _partial_attempts(self, day: str) -> int:
        raw = self.db.get_kv(KV_DAILY_PARTIAL) or {}
        try:
            return int(raw.get(day, 0)) if isinstance(raw, dict) else 0
        except (TypeError, ValueError):
            return 0

    # --- transient failures / unpushed digests ----------------------------------
    def _max_failed_retries(self) -> int:
        try:
            return max(1, int(self.cfg.pipeline.get("max_failed_retries", DEFAULT_MAX_FAILED_RETRIES)))
        except (TypeError, ValueError):
            return DEFAULT_MAX_FAILED_RETRIES

    def _requeue_failed(self, now: datetime) -> int:
        """At most once per REQUEUE_INTERVAL_MIN: transient 'failed' segments -> 'new' (see requeue_transient_failures)."""
        if not is_due(self.db.get_kv(KV_FAILED_REQUEUE_AT), REQUEUE_INTERVAL_MIN, now):
            return 0
        self.db.set_kv(KV_FAILED_REQUEUE_AT, now.isoformat(timespec="seconds"))
        try:
            return requeue_transient_failures(self.db, self._max_failed_retries())
        except Exception as e:  # noqa: BLE001
            log.exception("re-queue of failed segments failed: %s", e)
            return 0

    def _pending_pushes(self) -> dict[str, dict[str, Any]]:
        raw = self.db.get_kv(KV_PUSH_PENDING) or {}
        return {str(k): dict(v) for k, v in raw.items() if isinstance(v, dict)} if isinstance(raw, dict) else {}

    def _note_push_result(self, day: str, results: dict[str, str | None] | None, now: datetime | None = None) -> bool:
        """Record the outcome of a digest push; a day whose push failed on every channel is retried later.
        Returns True when at least one channel succeeded."""
        now = now or datetime.now()
        pending = self._pending_pushes()
        ok = any(v is None for v in (results or {}).values())
        if ok or not results:
            # success, or no channel configured at all (nothing a retry could change)
            changed = pending.pop(day, None) is not None
        else:
            prev = pending.get(day, {})
            pending[day] = {"attempts": int(prev.get("attempts", 0)), "last": now.isoformat(timespec="seconds")}
            changed = True
        if changed:
            self.db.set_kv(KV_PUSH_PENDING, pending)
        return ok

    def _retry_pushes(self, now: datetime) -> bool:
        """Re-send digests whose push failed on every channel (network down at daily_time), every
        PUSH_RETRY_MIN, at most MAX_PUSH_RETRIES times per day. Returns True when a push was attempted."""
        pending = self._pending_pushes()
        if not pending:
            return False
        attempted = False
        for day, info in sorted(pending.items()):
            attempts = int(info.get("attempts", 0) or 0)
            if attempts >= MAX_PUSH_RETRIES:
                log.error("daily %s: giving up on push after %d retries", day, attempts)
                pending.pop(day, None)
                continue
            last = _to_dt(info.get("last"))
            if last is not None and now - last < timedelta(minutes=PUSH_RETRY_MIN):
                continue
            attempted = True
            try:
                results = self.jobs["push"](day)
            except Exception as e:  # noqa: BLE001
                log.warning("daily %s: push retry failed: %s", day, e)
                results = {"push": f"{type(e).__name__}: {e}"}
            if any(v is None for v in (results or {}).values()):
                log.info("daily %s: push retry succeeded", day)
                pending.pop(day, None)
            else:
                pending[day] = {"attempts": attempts + 1, "last": now.isoformat(timespec="seconds")}
        self.db.set_kv(KV_PUSH_PENDING, pending)
        return attempted

    def _mark_daily(self, day: str, now: datetime) -> None:
        prev = self.db.get_kv(KV_DAILY_DAY) or ""
        if day > prev:
            self.db.set_kv(KV_DAILY_DAY, day)
        self.db.set_kv(KV_DAILY_AT, now.isoformat(timespec="seconds"))

    # --- default jobs (lazy imports: sibling modules may not exist yet) ------
    def _client(self):
        """The scheduler's MiMo client (one per configuration, with usage accounting); None without an API key."""
        if self._llm is not None:
            return self._llm
        from .llm.client import MimoClient, MissingAPIKey

        try:
            client = MimoClient(self.cfg, rpm=int(self.cfg.mimo.rpm))
        except MissingAPIKey as e:
            if not self._client_warned:
                log.warning("%s -- processing/daily jobs are skipped until a key is configured", e)
                self._client_warned = True
            return None
        from . import metrics

        metrics.install(client, self.db)
        self._llm = client
        return client

    def _client_quiet(self):
        """_client() for optional work (the chain): a missing key is not worth a warning of its own."""
        warned, self._client_warned = self._client_warned, True
        try:
            return self._client()
        finally:
            self._client_warned = warned

    def _chain_stop(self) -> Any:
        """What the chain's steps see as their stop event. With realtime processing on, a segment arriving
        from the recorder also ends the chain early (between two items of a step): the chain holds the
        scheduler lock, and a session being refined must not keep a fresh recording waiting. The next tick
        processes the segment and then runs the chain again."""
        if bool(self.cfg.pipeline.get("realtime", True)) and not self._process_injected and self._defer() is None:
            return _EitherEvent(self.stop_event, self.wake)
        return self.stop_event

    def _default_chain(self):
        from . import chain

        return chain.run_chain(self.cfg, self.db, self._client_quiet, stop_event=self._chain_stop(), events=self.events,
                               limit=CHAIN_LIMIT)

    def _default_consolidate(self):
        from . import chain

        return chain.call_step(chain.CONSOLIDATE, self.cfg, self.db, self._client_quiet(), name="consolidate",
                               stop_event=self.stop_event, events=self.events)

    def _default_brief(self):
        from . import chain

        return chain.call_step(chain.BRIEF, self.cfg, self.db, self._client_quiet(), name="brief",
                               stop_event=self.stop_event, events=self.events)

    def _default_process(self):
        proc = self.processor()
        if proc is None:
            return "skipped (no api key)"
        for name in ("run", "run_once", "process_pending", "process_all"):
            fn = getattr(proc, name, None)
            if callable(fn):
                return fn()
        raise RuntimeError("Processor has no run()/run_once()/process_pending() method")

    def _default_cleanup(self):
        from .audio.storage import cleanup

        if self._vad is None:
            try:
                from .audio.vad import SileroVAD

                self._vad = SileroVAD(self.cfg.model_path("vad_model"), threads=1)
            except Exception as e:  # noqa: BLE001
                log.warning("VAD unavailable for cleanup recheck: %s", e)
        return cleanup(self.cfg, self.db, self._vad)

    def _already_pushed(self, day: str) -> bool:
        row = self.db.one("SELECT pushed_at FROM daily_summaries WHERE day=?", (day,))
        return bool(row and row["pushed_at"])

    def _default_daily(self, day: str, push: bool):
        client = self._client()
        if client is None:
            raise RuntimeError("no MiMo API key")
        from .analysis.daily import DailyAnalyzer

        analyzer = DailyAnalyzer(self.cfg, self.db, _scoped(client, "daily"))
        result = analyzer.run(day)
        if push:
            if analyzer.last_run_cached and self._already_pushed(day):
                log.info("daily %s: digest unchanged and already pushed; not pushing again", day)
            elif analyzer.last_run_partial and self._partial_attempts(day) >= 1:
                log.info("daily %s: partial digest already pushed once; waiting for a complete one", day)
            else:
                try:
                    self._note_push_result(day, analyzer.push(day))
                except Exception as e:  # noqa: BLE001  (a failed push must not re-run the LLM)
                    log.exception("daily push %s failed: %s", day, e)
                    self._note_push_result(day, {"push": f"{type(e).__name__}: {e}"})
        try:
            from .analysis.relations import maybe_infer_after_daily

            maybe_infer_after_daily(self.cfg, self.db, _scoped(client, "relations"))
        except Exception as e:  # noqa: BLE001  (relation inference is best effort)
            log.warning("relation inference after daily %s failed: %s", day, e)
        return result

    def _default_push(self, day: str) -> dict[str, str | None]:
        """Push an already generated digest (no LLM call needed when the row exists)."""
        from .analysis.daily import DailyAnalyzer

        return DailyAnalyzer(self.cfg, self.db, None).push(day)


class _EitherEvent:
    """Read-only view over several threading.Event objects: set as soon as one of them is."""

    def __init__(self, *events: threading.Event):
        self._events = events

    def is_set(self) -> bool:
        return any(e.is_set() for e in self._events)

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        while not self.is_set():
            left = None if deadline is None else deadline - time.monotonic()
            if left is not None and left <= 0:
                return False
            self._events[0].wait(0.05 if left is None else min(0.05, left))
        return True


def _scoped(client: Any, purpose: str) -> Any:
    """client.scoped(purpose) when the client can tag its usage (MimoClient), the client itself otherwise."""
    fn = getattr(client, "scoped", None)
    if callable(fn):
        try:
            return fn(purpose)
        except Exception:  # noqa: BLE001
            return client
    return client


def run_scheduler(cfg: Config, db: DB, stop_event: threading.Event | None = None,
                  jobs: dict[str, Callable] | None = None) -> Scheduler:
    s = Scheduler(cfg, db, stop_event, jobs)
    s.run()
    return s


__all__: Iterable[str] = (
    "Scheduler", "run_scheduler", "parse_hhmm", "next_daily_due", "is_due", "daily_days_due",
    "requeue_transient_failures",
    "KV_PROCESS", "KV_CLEANUP", "KV_DAILY_DAY", "KV_DAILY_AT", "KV_DAILY_RECHECK",
    "KV_FAILED_REQUEUE_AT", "KV_FAILED_ATTEMPTS", "KV_PUSH_PENDING", "KV_DAILY_PARTIAL", "KV_BRIEF_DAY",
)
