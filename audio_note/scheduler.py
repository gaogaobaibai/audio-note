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
from typing import Callable, Iterable

from .config import Config
from .db import DB

log = logging.getLogger(__name__)

KV_PROCESS = "last_process_at"
KV_CLEANUP = "last_cleanup_at"
KV_DAILY_DAY = "last_daily_day"
KV_DAILY_AT = "last_daily_at"
KV_DAILY_RECHECK = "last_daily_recheck_day"   # yesterday already re-checked for late content

DAILY_RETRY_MIN = 30.0   # after a failed daily run, wait this long before trying again


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


def daily_days_due(now: datetime, hhmm: str, last_daily_day: str | None) -> list[str]:
    """Which days need a daily report right now, oldest first.

    * yesterday: when the last report is older than yesterday (or there never was one) -> catch-up,
      regardless of the time of day (yesterday is complete).
    * today:     once `now` is past hh:mm and today's report has not been made yet.
    """
    today: date = now.date()
    yesterday = today - timedelta(days=1)
    last = (last_daily_day or "").strip()
    out: list[str] = []
    if not last or last < yesterday.isoformat():
        out.append(yesterday.isoformat())
    h, m = parse_hhmm(hhmm)
    if now >= now.replace(hour=h, minute=m, second=0, microsecond=0) and (not last or last < today.isoformat()):
        out.append(today.isoformat())
    return out


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
            days = daily_days_due(now, str(self.cfg.analysis.daily_time), self.db.get_kv(KV_DAILY_DAY))
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
        return ran

    def _recheck_yesterday(self, now: datetime) -> bool:
        """Once per day: yesterday's report was made at daily_time, but speech after that (and
        segments still pending at that moment) landed later. If the utterance count changed since the
        report was written (or there was no report because nothing was processed yet), regenerate and
        push it. Waits while yesterday still has unprocessed segments."""
        if self._daily_retry_after is not None and now < self._daily_retry_after:
            return False
        yesterday = (now.date() - timedelta(days=1)).isoformat()
        if (self.db.get_kv(KV_DAILY_RECHECK) or "") >= yesterday:
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
            self.jobs["daily"](day, push)
        except Exception as e:  # noqa: BLE001
            self.errors["daily"] += 1
            self._daily_retry_after = now + timedelta(minutes=DAILY_RETRY_MIN)
            log.exception("daily %s failed (retry in %.0f min): %s", day, DAILY_RETRY_MIN, e)
            return False
        self.runs["daily"] += 1
        self._mark_daily(day, now)
        log.info("daily %s done in %.1fs (push=%s)", day, time.monotonic() - t0, push)
        return True

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

    def _default_daily(self, day: str, push: bool):
        client = self._client()
        if client is None:
            raise RuntimeError("no MiMo API key")
        from .analysis.daily import DailyAnalyzer

        analyzer = DailyAnalyzer(self.cfg, self.db, client)
        result = analyzer.run(day)
        if push:
            try:
                analyzer.push(day)
            except Exception as e:  # noqa: BLE001  (a failed push must not re-run the LLM)
                log.exception("daily push %s failed: %s", day, e)
        return result


def run_scheduler(cfg: Config, db: DB, stop_event: threading.Event | None = None,
                  jobs: dict[str, Callable] | None = None) -> Scheduler:
    s = Scheduler(cfg, db, stop_event, jobs)
    s.run()
    return s


__all__: Iterable[str] = (
    "Scheduler", "run_scheduler", "parse_hhmm", "next_daily_due", "is_due", "daily_days_due",
    "KV_PROCESS", "KV_CLEANUP", "KV_DAILY_DAY", "KV_DAILY_AT", "KV_DAILY_RECHECK",
)
