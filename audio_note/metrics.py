"""Accounting of what the machine and the API were asked to do, aggregated per day (tables usage_local / usage_llm).

    with metrics.stage(db, "diarize", audio_s=len(audio) / 16000):
        turns = engine.diarize(audio)                  # wall seconds + process CPU seconds are added to today's row

    metrics.install(client, db)                        # every MimoClient request -> usage_llm (per purpose and model)
    metrics.summary(db, days=7)                        # {"local": [...], "llm": [...], "totals": {...}}

CPU time is the whole process's (time.process_time): that is the only way to include onnxruntime's native
threads; work running in other threads at the same moment is therefore counted too.
A failed write is logged and forgotten: accounting never gets in the way of the work it measures.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Iterator

log = logging.getLogger(__name__)

LOCAL_FIELDS = ("n", "audio_s", "wall_s", "cpu_s")
LLM_FIELDS = ("calls", "errors", "prompt_tokens", "completion_tokens", "audio_s", "wall_s")


class StageRecord:
    """What stage() measured; `audio_s` / `n` may be set inside the block when they are only known later."""

    __slots__ = ("name", "audio_s", "n", "wall_s", "cpu_s")

    def __init__(self, name: str, audio_s: float = 0.0, n: int = 1):
        self.name = name
        self.audio_s = float(audio_s or 0.0)
        self.n = int(n)
        self.wall_s = 0.0
        self.cpu_s = 0.0


@contextmanager
def stage(db: Any, name: str, audio_s: float = 0.0, n: int = 1) -> Iterator[StageRecord]:
    """Measure the block (wall clock + process CPU) and add it to usage_local[today, name].
    `db` may be None (measure only). The row is written whether the block succeeds or raises."""
    rec = StageRecord(name, audio_s, n)
    t0 = time.perf_counter()
    c0 = time.process_time()
    try:
        yield rec
    finally:
        rec.wall_s = max(0.0, time.perf_counter() - t0)
        rec.cpu_s = max(0.0, time.process_time() - c0)
        if db is not None:
            try:
                db.add_usage_local(name, n=rec.n, audio_s=rec.audio_s, wall_s=rec.wall_s, cpu_s=rec.cpu_s)
            except Exception as e:  # noqa: BLE001 - e.g. the database was closed by a profile switch
                log.debug("usage_local not recorded for %s: %s", name, e)


def add_local(db: Any, name: str, *, audio_s: float = 0.0, wall_s: float = 0.0, cpu_s: float = 0.0, n: int = 1) -> None:
    """Add an already measured stage (for code that times itself)."""
    try:
        db.add_usage_local(name, n=n, audio_s=audio_s, wall_s=wall_s, cpu_s=cpu_s)
    except Exception as e:  # noqa: BLE001
        log.debug("usage_local not recorded for %s: %s", name, e)


def add_llm(db: Any, rec: dict[str, Any]) -> None:
    """Add one MimoClient usage record ({purpose, model, ok, prompt_tokens, completion_tokens, audio_s, wall_s})
    to usage_llm[today, purpose, model]. A failed request counts as a call and as an error."""
    try:
        db.add_usage_llm(str(rec.get("purpose") or "other"), str(rec.get("model") or "?"), calls=1,
                         errors=0 if rec.get("ok", True) else 1,
                         prompt_tokens=int(rec.get("prompt_tokens") or 0), completion_tokens=int(rec.get("completion_tokens") or 0),
                         audio_s=float(rec.get("audio_s") or 0.0), wall_s=float(rec.get("wall_s") or 0.0))
    except Exception as e:  # noqa: BLE001
        log.debug("usage_llm not recorded: %s", e)


def install(client: Any, db: Any) -> None:
    """Send the client's usage records to `db` (sets client.on_usage). Safe on any object: a test double
    that cannot take the attribute is left alone."""
    if client is None or db is None:
        return
    try:
        client.on_usage = lambda rec, _db=db: add_llm(_db, rec)
    except Exception as e:  # noqa: BLE001
        log.debug("metrics not installed on %r: %s", type(client).__name__, e)


def _first_day(days: int) -> str:
    return (datetime.now().date() - timedelta(days=max(1, int(days)) - 1)).isoformat()


def summary(db: Any, days: int = 7) -> dict[str, Any]:
    """Usage of the last `days` days (today included):

        {"days": 7, "from": "2026-11-05",
         "local": [{"day", "stage", "n", "audio_s", "wall_s", "cpu_s"}, ...],              # day desc, stage
         "llm":   [{"day", "purpose", "model", "calls", "errors", "prompt_tokens", "completion_tokens", "audio_s", "wall_s"}, ...],
         "totals": {"local": {n, audio_s, wall_s, cpu_s}, "llm": {calls, errors, prompt_tokens, completion_tokens, audio_s, wall_s},
                    "by_stage": {stage: {...}}, "by_purpose": {purpose: {...}}}}
    """
    days = max(1, int(days or 7))
    start = _first_day(days)
    local = [dict(r) for r in db.query(
        "SELECT day, stage, n, audio_s, wall_s, cpu_s FROM usage_local WHERE day >= ? ORDER BY day DESC, stage", (start,))]
    llm = [dict(r) for r in db.query(
        "SELECT day, purpose, model, calls, errors, prompt_tokens, completion_tokens, audio_s, wall_s FROM usage_llm"
        " WHERE day >= ? ORDER BY day DESC, purpose, model", (start,))]

    def total(rows: list[dict[str, Any]], fields: tuple[str, ...], key: str | None = None) -> Any:
        if key is None:
            return {f: _round(sum(r[f] or 0 for r in rows)) for f in fields}
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            acc = out.setdefault(str(r[key]), {f: 0 for f in fields})
            for f in fields:
                acc[f] += r[f] or 0
        return {k: {f: _round(v) for f, v in acc.items()} for k, acc in sorted(out.items())}

    for rows in (local, llm):
        for r in rows:
            for k, v in r.items():
                if isinstance(v, float):
                    r[k] = _round(v)
    return {"days": days, "from": start, "local": local, "llm": llm,
            "totals": {"local": total(local, LOCAL_FIELDS), "llm": total(llm, LLM_FIELDS),
                       "by_stage": total(local, LOCAL_FIELDS, "stage"), "by_purpose": total(llm, LLM_FIELDS, "purpose")}}


def _round(v: Any) -> Any:
    return round(v, 3) if isinstance(v, float) else v


__all__ = ["stage", "add_local", "add_llm", "install", "summary", "StageRecord"]
