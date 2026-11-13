"""audio_note.metrics (stage timing, LLM usage, summary) and audio_note.governor (threads, priorities, battery).

The governor is only ever exercised against a fake kernel32: the tests never change the priority of the
pytest process or of a thread.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from audio_note import governor, metrics
from audio_note.config import DEFAULTS


# --- metrics.stage ---------------------------------------------------------------------------------------------

def _burn(seconds: float) -> None:
    """Keep this thread busy for `seconds` of wall time (the process CPU clock on Windows ticks every ~16 ms)."""
    end = time.perf_counter() + seconds
    x = 0
    while time.perf_counter() < end:
        x += 1


def test_stage_accumulates_wall_and_cpu_per_day(db):
    with metrics.stage(db, "diarize", audio_s=12.0) as rec:
        _burn(0.12)
    assert rec.name == "diarize" and rec.wall_s >= 0.12 and rec.cpu_s >= 0.015 and rec.audio_s == 12.0
    with metrics.stage(db, "diarize") as rec2:
        rec2.audio_s = 8.0                                                 # only known once the work is done
        rec2.n = 3
    with metrics.stage(db, "embed", audio_s=1.0, n=2):
        pass
    rows = {r["stage"]: r for r in db.query("SELECT * FROM usage_local")}
    d = rows["diarize"]
    assert d["day"] == datetime.now().strftime("%Y-%m-%d") and d["n"] == 4 and d["audio_s"] == 20.0
    assert d["wall_s"] == pytest.approx(rec.wall_s + rec2.wall_s) and d["cpu_s"] == pytest.approx(rec.cpu_s + rec2.cpu_s)
    assert rows["embed"]["n"] == 2 and rows["embed"]["audio_s"] == 1.0


def test_stage_records_a_failing_block_and_never_hides_its_error(db):
    with pytest.raises(ValueError, match="model exploded"):
        with metrics.stage(db, "identify", audio_s=2.0):
            raise ValueError("model exploded")
    assert db.one("SELECT n, audio_s FROM usage_local WHERE stage='identify'")["n"] == 1


def test_accounting_failures_are_swallowed(db):
    with metrics.stage(None, "gate", audio_s=1.0) as rec:                  # no database: measure only
        pass
    assert rec.wall_s >= 0
    db.close_all()                                                        # e.g. a profile switch closed it under a running stage
    with metrics.stage(db, "diarize", audio_s=1.0):
        pass
    metrics.add_local(db, "embed", cpu_s=1.0)
    metrics.add_llm(db, {"purpose": "asr", "model": "m", "ok": True})
    metrics.install(None, db)
    metrics.install(object(), db)                                         # an object that takes no attributes


# --- LLM usage ---------------------------------------------------------------------------------------------------

def test_add_llm_counts_calls_errors_tokens_and_audio(db):
    metrics.add_llm(db, {"purpose": "asr", "model": "mimo-v2.5-asr", "ok": True, "prompt_tokens": 44, "completion_tokens": 20,
                         "audio_s": 9.0, "wall_s": 1.25})
    metrics.add_llm(db, {"purpose": "asr", "model": "mimo-v2.5-asr", "ok": False, "wall_s": 30.0})
    metrics.add_llm(db, {"model": "mimo-v2.6-flash"})                      # no purpose -> "other", ok by default
    rows = {(r["purpose"], r["model"]): dict(r) for r in db.query("SELECT * FROM usage_llm")}
    asr = rows[("asr", "mimo-v2.5-asr")]
    assert (asr["calls"], asr["errors"], asr["prompt_tokens"], asr["completion_tokens"], asr["audio_s"], asr["wall_s"]) == (2, 1, 44, 20, 9.0, 31.25)
    assert rows[("other", "mimo-v2.6-flash")]["calls"] == 1 and rows[("other", "mimo-v2.6-flash")]["errors"] == 0


def test_install_routes_a_real_clients_usage_into_the_table(cfg_factory):
    from audio_note.db import DB
    from audio_note.llm.client import MimoClient

    cfg = cfg_factory({"mimo": {"api_key": "sk-fake-metrics"}})
    db = DB(cfg.db_path)
    client = MimoClient(cfg, rpm=100000)
    client.limiter.interval = 0.0
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=30)
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="好", tool_calls=None), finish_reason="stop")], usage=usage)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: reply)))
    metrics.install(client, db)
    client.scoped("refine").text("s", "u")
    client.scoped("refine").text("s", "u")
    client.text("s", "u")
    rows = {r["purpose"]: dict(r) for r in db.query("SELECT * FROM usage_llm")}
    assert rows["refine"]["calls"] == 2 and rows["refine"]["prompt_tokens"] == 200 and rows["refine"]["completion_tokens"] == 60
    assert rows["refine"]["model"] == "mimo-v2.6-flash" and rows["other"]["calls"] == 1


def test_summary_shape_window_and_totals(db):
    today = datetime.now().date()
    d = lambda k: (today - timedelta(days=k)).isoformat()  # noqa: E731
    db.add_usage_local("diarize", audio_s=100.0, wall_s=10.0, cpu_s=20.0, day=d(0))
    db.add_usage_local("embed", audio_s=100.0, wall_s=1.0, cpu_s=1.5, n=4, day=d(0))
    db.add_usage_local("diarize", audio_s=50.0, wall_s=5.0, cpu_s=9.0, day=d(6))
    db.add_usage_local("diarize", audio_s=1.0, wall_s=1.0, cpu_s=1.0, day=d(7))            # one day too old for days=7
    db.add_usage_llm("asr", "mimo-v2.5-asr", prompt_tokens=10, completion_tokens=5, audio_s=90.0, wall_s=7.0, day=d(0))
    db.add_usage_llm("refine", "mimo-v2.6-flash", errors=1, prompt_tokens=900, completion_tokens=300, wall_s=40.0, day=d(1))
    s = metrics.summary(db, days=7)
    assert set(s) == {"days", "from", "local", "llm", "totals"} and s["days"] == 7 and s["from"] == d(6)
    assert [(r["day"], r["stage"]) for r in s["local"]] == [(d(0), "diarize"), (d(0), "embed"), (d(6), "diarize")]
    assert set(s["local"][0]) == {"day", "stage", "n", "audio_s", "wall_s", "cpu_s"}
    assert set(s["llm"][0]) == {"day", "purpose", "model", "calls", "errors", "prompt_tokens", "completion_tokens", "audio_s", "wall_s"}
    t = s["totals"]
    assert t["local"] == {"n": 6, "audio_s": 250.0, "wall_s": 16.0, "cpu_s": 30.5}
    assert t["llm"] == {"calls": 2, "errors": 1, "prompt_tokens": 910, "completion_tokens": 305, "audio_s": 90.0, "wall_s": 47.0}
    assert t["by_stage"]["diarize"] == {"n": 2, "audio_s": 150.0, "wall_s": 15.0, "cpu_s": 29.0} and set(t["by_stage"]) == {"diarize", "embed"}
    assert t["by_purpose"]["refine"]["errors"] == 1 and set(t["by_purpose"]) == {"asr", "refine"}
    assert metrics.summary(db, days=1)["totals"]["local"]["n"] == 5 and len(metrics.summary(db, days=30)["local"]) == 4
    empty = metrics.summary(db.__class__(db.path.parent / "empty.sqlite3"))
    assert empty["local"] == [] and empty["llm"] == [] and empty["totals"]["llm"]["calls"] == 0 and empty["totals"]["by_stage"] == {}


# --- governor.threads ----------------------------------------------------------------------------------------------

def _cfg(**sections):
    return {k: dict(v) for k, v in sections.items()}


def test_threads_follow_mode_unless_set_explicitly():
    assert governor.threads(_cfg(performance={"mode": "eco", "threads": 0}, speakers={"num_threads": 0})) == 1
    assert governor.threads(_cfg(performance={"mode": "balanced", "threads": 0}, speakers={"num_threads": 0})) == 2
    assert governor.threads(_cfg(performance={"mode": "fast", "threads": 0}, speakers={"num_threads": 0})) == 4
    assert governor.threads(_cfg(performance={"mode": "eco", "threads": 3}, speakers={"num_threads": 0})) == 3
    assert governor.threads(_cfg(performance={"mode": "eco", "threads": 3}, speakers={"num_threads": 6})) == 6     # speakers wins
    assert governor.threads(_cfg(performance={"mode": "FAST "})) == 4 and governor.mode(_cfg(performance={"mode": "FAST "})) == "fast"
    # junk never breaks model loading
    assert governor.threads({}) == 2 and governor.threads(None) == 2 and governor.threads(_cfg(performance={"mode": "turbo"})) == 2
    assert governor.threads(_cfg(performance={"threads": "many"}, speakers={"num_threads": None})) == 2
    assert governor.threads(_cfg(performance={"threads": 10_000})) == governor.MAX_THREADS
    assert governor.threads(_cfg(speakers={"num_threads": -4})) == 2


def test_default_config_means_two_threads(cfg):
    assert DEFAULTS["speakers"]["num_threads"] == 0 and DEFAULTS["performance"] == {
        "mode": "balanced", "threads": 0, "background_priority": True, "defer_on_battery": False}
    assert governor.threads(cfg) == 2 and governor.defer_reason(cfg) is None


# --- priorities / power (fake kernel32 only) ------------------------------------------------------------------------

class FakeKernel:
    def __init__(self, ac: int = 1, ok: bool = True):
        self.calls: list[tuple] = []
        self.ac, self.ok = ac, ok

    def GetCurrentProcess(self):  # noqa: N802
        return -1

    def GetCurrentThread(self):  # noqa: N802
        return -2

    def SetPriorityClass(self, handle, cls):  # noqa: N802
        self.calls.append(("class", handle, cls))
        return 1 if self.ok else 0

    def SetThreadPriority(self, handle, level):  # noqa: N802
        self.calls.append(("thread", handle, level))
        return 1 if self.ok else 0

    def GetSystemPowerStatus(self, ref):  # noqa: N802
        ref._obj.ACLineStatus = self.ac
        return 1 if self.ok else 0


def test_everything_is_a_no_op_without_kernel32(monkeypatch):
    monkeypatch.setattr(governor, "_kernel32", lambda: None)
    on = _cfg(performance={"background_priority": True, "defer_on_battery": True})
    assert governor.apply_process_priority(on) is False
    assert governor.boost_current_thread() is None and governor.restore_current_thread() is None
    assert governor.on_battery() is None and governor.defer_reason(on) is None


def test_process_priority_only_when_asked(monkeypatch):
    k = FakeKernel()
    monkeypatch.setattr(governor, "_kernel32", lambda: k)
    assert governor.apply_process_priority(_cfg(performance={"background_priority": False})) is False
    assert governor.apply_process_priority({}) is False and k.calls == []
    assert governor.apply_process_priority(_cfg(performance={"background_priority": True})) is True
    assert k.calls == [("class", -1, governor.BELOW_NORMAL_PRIORITY_CLASS)]
    governor.boost_current_thread()
    governor.restore_current_thread()
    assert k.calls[1:] == [("thread", -2, governor.THREAD_PRIORITY_HIGHEST), ("thread", -2, governor.THREAD_PRIORITY_NORMAL)]
    refused = FakeKernel(ok=False)
    monkeypatch.setattr(governor, "_kernel32", lambda: refused)
    assert governor.apply_process_priority(_cfg(performance={"background_priority": True})) is False


def test_calls_that_raise_are_swallowed(monkeypatch):
    class Broken:
        def __getattr__(self, name):
            raise OSError("access denied")

    monkeypatch.setattr(governor, "_kernel32", lambda: Broken())
    cfg = _cfg(performance={"background_priority": True, "defer_on_battery": True})
    assert governor.apply_process_priority(cfg) is False and governor.on_battery() is None and governor.defer_reason(cfg) is None
    governor.boost_current_thread()
    governor.restore_current_thread()


@pytest.mark.parametrize("ac,expected", [(0, True), (1, False), (255, None)])
def test_on_battery_and_defer_reason(monkeypatch, ac, expected):
    monkeypatch.setattr(governor, "_kernel32", lambda: FakeKernel(ac=ac))
    assert governor.on_battery() is expected
    assert governor.defer_reason(_cfg(performance={"defer_on_battery": False})) is None            # never asked for
    reason = governor.defer_reason(_cfg(performance={"defer_on_battery": True}))
    assert reason == (governor.BATTERY_REASON if expected is True else None)
    if reason:
        assert "电池" in reason


def test_real_platform_queries_do_not_raise():
    """The real kernel32 (or its absence): read-only queries only, nothing is changed."""
    assert governor.on_battery() in (True, False, None)
    assert governor.defer_reason(_cfg(performance={"defer_on_battery": True})) in (None, governor.BATTERY_REASON)


def test_service_applies_the_priority_and_survives_a_failure(monkeypatch, cfg):
    from audio_note import service

    seen: list = []
    monkeypatch.setattr(governor, "apply_process_priority", lambda c: seen.append(c) or True)
    service._apply_priority(cfg)
    assert seen == [cfg]
    monkeypatch.setattr(governor, "apply_process_priority", lambda c: (_ for _ in ()).throw(OSError("denied")))
    service._apply_priority(cfg)                                           # never raises
