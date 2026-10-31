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
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime, timedelta
from typing import Any, Callable, Iterable

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
                 jobs: dict[str, Callable] | None = None, tick_s: float = 5.0):
        self.cfg = cfg
        self.db = db
        self.stop_event = stop_event or threading.Event()
        self.tick_s = float(tick_s)
        jobs = jobs or {}
        self.jobs: dict[str, Callable] = {
            "process": jobs.get("process") or self._default_process,
            "cleanup": jobs.get("cleanup") or self._default_cleanup,
            "daily": jobs.get("daily") or self._default_daily,
            "push": jobs.get("push") or self._default_push,     # fn(day) -> {channel: error|None}
        }
        self.runs: dict[str, int] = {"process": 0, "cleanup": 0, "daily": 0}
        self.errors: dict[str, int] = {"process": 0, "cleanup": 0, "daily": 0}
        self._daily_retry_after: datetime | None = None
        self._processor = None
        self._vad = None
        self._client_warned = False

    # --- loop ---------------------------------------------------------------
    def run(self) -> None:
        log.info("scheduler started: process every %s min, cleanup every %s min, daily at %s",
                 self.cfg.pipeline.process_interval_min, self.cfg.storage.cleanup_interval_min, self.cfg.analysis.daily_time)
        while not self.stop_event.is_set():
            try:
                self.run_once()
            except Exception as e:  # noqa: BLE001  (never let the loop die)
                log.exception("scheduler tick failed: %s", e)
            self.stop_event.wait(self.tick_s)
        log.info("scheduler stopped")

    def run_once(self, now: datetime | None = None) -> list[str]:
        """Run every job that is due at `now`. Returns the names of the jobs that ran."""
        now = now or datetime.now()
        ran: list[str] = []
        if is_due(self.db.get_kv(KV_PROCESS), float(self.cfg.pipeline.process_interval_min), now):
            self._requeue_failed(now)
            self._run_job("process", now)
            ran.append("process")
        if self.stop_event.is_set():
            return ran
        if bool(self.cfg.storage.cleanup_enabled) and is_due(self.db.get_kv(KV_CLEANUP), float(self.cfg.storage.cleanup_interval_min), now):
            self._run_job("cleanup", now)
            ran.append("cleanup")
        if self.stop_event.is_set():
            return ran
        if self._daily_due(now):
            days = self._days_due(now)
            if days and "process" not in ran:
                # transcribe what was recorded in the last minutes before summarising the day
                self._run_job("process", now)
                ran.append("process")
            for day in days:
                if self.stop_event.is_set():
                    break
                if self._run_daily(day, now):
                    ran.append("daily")
        if not self.stop_event.is_set() and self._recheck_yesterday(now):
            ran.append("daily")
        if not self.stop_event.is_set() and self._retry_pushes(now):
            ran.append("push")
        return ran

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
        try:
            result = self.jobs[name]()
            self.runs[name] += 1
            log.info("job %s done in %.1fs%s", name, time.monotonic() - t0, f": {result}" if result is not None else "")
        except Exception as e:  # noqa: BLE001
            self.errors[name] += 1
            log.exception("job %s failed: %s", name, e)

    def _daily_due(self, now: datetime) -> bool:
        if self._daily_retry_after is not None and now < self._daily_retry_after:
            return False
        try:
            return bool(daily_days_due(now, str(self.cfg.analysis.daily_time), self.db.get_kv(KV_DAILY_DAY)))
        except ValueError as e:
            log.error("analysis.daily_time invalid: %s", e)
            self._daily_retry_after = now + timedelta(hours=1)
            return False

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
        try:
            res = self.jobs["daily"](day, push)
        except Exception as e:  # noqa: BLE001
            self.errors["daily"] += 1
            self._daily_retry_after = now + timedelta(minutes=DAILY_RETRY_MIN)
            log.exception("daily %s failed (retry in %.0f min): %s", day, DAILY_RETRY_MIN, e)
            return False
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
        from .llm.client import MimoClient, MissingAPIKey

        try:
            return MimoClient(self.cfg, rpm=int(self.cfg.mimo.rpm))
        except MissingAPIKey as e:
            if not self._client_warned:
                log.warning("%s -- processing/daily jobs are skipped until a key is configured", e)
                self._client_warned = True
            return None

    def _default_process(self):
        if self._processor is None:
            client = self._client()
            if client is None:
                return "skipped (no api key)"
            from .speech.pipeline import Processor

            self._processor = Processor(self.cfg, self.db, client, stop_event=self.stop_event)
        proc = self._processor
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

        analyzer = DailyAnalyzer(self.cfg, self.db, client)
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
        return result

    def _default_push(self, day: str) -> dict[str, str | None]:
        """Push an already generated digest (no LLM call needed when the row exists)."""
        from .analysis.daily import DailyAnalyzer

        return DailyAnalyzer(self.cfg, self.db, None).push(day)


def run_scheduler(cfg: Config, db: DB, stop_event: threading.Event | None = None,
                  jobs: dict[str, Callable] | None = None) -> Scheduler:
    s = Scheduler(cfg, db, stop_event, jobs)
    s.run()
    return s


__all__: Iterable[str] = (
    "Scheduler", "run_scheduler", "parse_hhmm", "next_daily_due", "is_due", "daily_days_due",
    "requeue_transient_failures",
    "KV_PROCESS", "KV_CLEANUP", "KV_DAILY_DAY", "KV_DAILY_AT", "KV_DAILY_RECHECK",
    "KV_FAILED_REQUEUE_AT", "KV_FAILED_ATTEMPTS", "KV_PUSH_PENDING", "KV_DAILY_PARTIAL",
)
