"""audio_note.chain: the post-processing chain (step order, missing modules, failure isolation, lazy client,
stop handling) and the job_* wrappers the runtime calls.

Steps are throw-away modules registered in sys.modules; chain.STEPS is pointed at them.
"""
from __future__ import annotations

import sys
import threading
import types
from types import SimpleNamespace

import pytest

from audio_note import chain


@pytest.fixture
def steps(monkeypatch):
    """Register fake step modules; returns (calls, add). add(name, fn=None) creates `v1chain_<name>:run`."""
    calls: list[tuple] = []

    def add(name: str, fn=None, attr: str = "run") -> str:
        mod = types.ModuleType(f"v1chain_{name}")

        def default(cfg, db, client, *, stop_event=None, events=None, limit=None):
            calls.append((name, client, limit, stop_event, events))
            return {"done": name}

        setattr(mod, attr, fn or default)
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
        return f"{mod.__name__}:{attr}"

    return calls, add


def test_default_steps_and_targets_follow_the_spec():
    assert chain.STEPS == (("sessions", "audio_note.memory.sessions:sync"),
                           ("rescore", "audio_note.speech.scoring:run_pending"),      # speakers v5: no LLM, numpy only
                           ("refine", "audio_note.speech.refine:run_pending"),
                           ("episodes", "audio_note.memory.episodes:run_pending"))
    assert chain.CONSOLIDATE == "audio_note.memory.consolidate:run" and chain.BRIEF == "audio_note.memory.brief:run"


def test_steps_run_in_order_with_the_fixed_signature(monkeypatch, steps):
    calls, add = steps
    monkeypatch.setattr(chain, "STEPS", (("sessions", add("a")), ("refine", add("b")), ("episodes", add("c"))))
    stop, bus = threading.Event(), object()
    made: list[int] = []
    client = SimpleNamespace(name="client")

    def factory():
        made.append(1)
        return client

    out = chain.run_chain("CFG", "DB", factory, stop_event=stop, events=bus, limit=3)
    assert out["ran"] == ["sessions", "refine", "episodes"] and out["skipped"] == [] and out["failed"] == {}
    assert out["results"] == {"sessions": {"done": "a"}, "refine": {"done": "b"}, "episodes": {"done": "c"}} and out["wall_s"] >= 0
    assert [c[0] for c in calls] == ["a", "b", "c"] and made == [1]            # one client for the whole chain
    assert all(c[1] is client and c[2] == 3 and c[3] is stop and c[4] is bus for c in calls)


def test_missing_modules_are_skipped_silently_and_need_no_client(monkeypatch, steps, caplog):
    calls, add = steps
    monkeypatch.setattr(chain, "STEPS", (("sessions", "audio_note.not_written_yet.sessions:sync"),
                                         ("refine", "v1chain_missing_pkg.refine:run_pending"),
                                         ("episodes", add("c", attr="other_name").replace("other_name", "run_pending"))))
    made: list[int] = []
    with caplog.at_level("WARNING"):
        out = chain.run_chain(None, None, lambda: made.append(1))
    assert out["ran"] == [] and out["skipped"] == ["sessions", "refine", "episodes"] and out["failed"] == {} and out["results"] == {}
    assert made == [] and caplog.records == []                                 # nothing ran: the client was never built
    assert chain.has_steps() is False and chain.available("v1chain_c:other_name") is True
    with pytest.raises(chain.StepUnavailable, match="不存在"):
        chain.resolve("audio_note.not_written_yet.sessions:sync")
    with pytest.raises(chain.StepUnavailable, match="没有 run_pending"):
        chain.resolve("v1chain_c:run_pending")


def test_the_real_chain_is_safe_to_run_whatever_is_installed(db, cfg):
    """Against the real STEPS: steps that are not installed are skipped, installed ones run without a client."""
    out = chain.run_chain(cfg, db, lambda: None)
    assert set(out["ran"]) | set(out["skipped"]) | set(out["failed"]) == {"sessions", "rescore", "refine", "episodes"}
    assert out["failed"] == {}


def test_a_failing_step_is_isolated(monkeypatch, steps, caplog):
    calls, add = steps

    def boom(cfg, db, client, *, stop_event=None, events=None, limit=None):
        raise RuntimeError("refine exploded")

    monkeypatch.setattr(chain, "STEPS", (("sessions", add("a")), ("refine", add("b", boom)), ("episodes", add("c"))))
    with caplog.at_level("ERROR"):
        out = chain.run_chain(None, None, None)
    assert out["ran"] == ["sessions", "episodes"] and out["failed"] == {"refine": "RuntimeError: refine exploded"}
    assert [c[0] for c in calls] == ["a", "c"] and any("refine" in r.getMessage() for r in caplog.records)
    assert calls[0][1] is None                                                 # no factory: steps get client=None


def test_a_module_that_exists_but_cannot_be_imported_is_reported_not_hidden(monkeypatch, steps, tmp_path):
    calls, add = steps
    pkg = tmp_path / "v1chain_broken.py"
    pkg.write_text("import module_that_does_not_exist_anywhere\n\ndef run(*a, **k):\n    return {}\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "v1chain_broken", raising=False)
    monkeypatch.setattr(chain, "STEPS", (("sessions", "v1chain_broken:run"), ("refine", add("b"))))
    out = chain.run_chain(None, None, None)
    assert out["ran"] == ["refine"] and "sessions" in out["failed"] and "module_that_does_not_exist_anywhere" in out["failed"]["sessions"]
    assert chain.available("v1chain_broken:run") is True                       # "installed" - running it shows the error


def test_only_and_stop_event(monkeypatch, steps):
    calls, add = steps
    stop = threading.Event()

    def stops(cfg, db, client, *, stop_event=None, events=None, limit=None):
        calls.append(("a",))
        stop_event.set()
        return {"ok": 1}

    monkeypatch.setattr(chain, "STEPS", (("sessions", add("a", stops)), ("refine", add("b")), ("episodes", add("c"))))
    out = chain.run_chain(None, None, None, stop_event=stop)
    assert out["ran"] == ["sessions"] and out["skipped"] == ["refine", "episodes"] and len(calls) == 1
    calls.clear()
    monkeypatch.setattr(chain, "STEPS", (("sessions", add("a")), ("refine", add("b")), ("episodes", add("c"))))
    out = chain.run_chain(None, None, None, only=("refine",))
    assert out["ran"] == ["refine"] and out["skipped"] == [] and [c[0] for c in calls] == ["b"]
    assert chain.run_chain(None, None, None, only=())["ran"] == []


def test_client_is_scoped_per_step_and_a_failing_factory_means_no_client(monkeypatch, steps):
    calls, add = steps
    monkeypatch.setattr(chain, "STEPS", (("sessions", add("a")), ("refine", add("b")), ("episodes", add("c"))))

    class Client:
        def __init__(self, purpose=None):
            self.purpose = purpose

        def scoped(self, purpose):
            return Client(purpose)

    chain.run_chain(None, None, lambda: Client())
    assert [c[1].purpose for c in calls] == ["sessions", "refine", "episode"]
    calls.clear()
    out = chain.run_chain(None, None, lambda: (_ for _ in ()).throw(RuntimeError("no api key")))
    assert out["ran"] == ["sessions", "refine", "episodes"] and all(c[1] is None for c in calls)


def test_non_dict_results_are_wrapped(monkeypatch, steps):
    calls, add = steps
    monkeypatch.setattr(chain, "STEPS", (("sessions", add("a", lambda *a, **k: 7)),))
    assert chain.run_chain(None, None, None)["results"] == {"sessions": {"result": 7}}
    assert chain.call_step(add("z", lambda *a, **k: None), None, None, None) == {"result": None}


# --- job wrappers ---------------------------------------------------------------------------------------------------

class FakeRuntime:
    def __init__(self, client="CLIENT", stop=None):
        self.cfg, self.db, self.events = "CFG", "DB", "BUS"
        self._client, self._stop = client, stop

    def client(self):
        if self._client is None:
            raise RuntimeError("没有配置 MiMo API key")
        return self._client

    def job_stop_event(self):
        return self._stop


def test_job_chain_passes_the_runtime_parts(monkeypatch, steps):
    calls, add = steps
    monkeypatch.setattr(chain, "STEPS", (("sessions", add("a")), ("refine", add("b")), ("episodes", add("c"))))
    stop = threading.Event()
    out = chain.job_chain(FakeRuntime(stop=stop), limit=5)
    assert out["ran"] == ["sessions", "refine", "episodes"]
    assert all(c[1] == "CLIENT" and c[2] == 5 and c[3] is stop and c[4] == "BUS" for c in calls)
    calls.clear()
    assert chain.job_chain(FakeRuntime(client=None), only=["sessions"])["ran"] == ["sessions"] and calls[0][1] is None


def test_job_refine_and_episodes_cut_sessions_first_and_need_their_module(monkeypatch, steps):
    calls, add = steps
    monkeypatch.setattr(chain, "STEPS", (("sessions", add("a")), ("refine", add("b")), ("episodes", "audio_note.not_written_yet.episodes:run")))
    assert chain.job_refine(FakeRuntime(), limit=2)["ran"] == ["sessions", "refine"]
    assert [c[0] for c in calls] == ["a", "b"] and calls[1][2] == 2
    with pytest.raises(chain.StepUnavailable, match="会话摘要功能在这个版本里还不可用"):
        chain.job_episodes(FakeRuntime())

    def boom(*a, **k):
        raise ValueError("bad session")

    monkeypatch.setattr(chain, "STEPS", (("sessions", add("a")), ("refine", add("b", boom)), ("episodes", add("c"))))
    with pytest.raises(RuntimeError, match="会话校对失败：ValueError: bad session"):
        chain.job_refine(FakeRuntime())


def test_job_consolidate_and_brief(monkeypatch, steps):
    calls, add = steps
    monkeypatch.setattr(chain, "CONSOLIDATE", add("cons"))
    monkeypatch.setattr(chain, "BRIEF", "audio_note.not_written_yet.brief:run")
    assert chain.job_consolidate(FakeRuntime(), limit=9) == {"done": "cons"}
    assert calls == [("cons", "CLIENT", 9, None, "BUS")]
    assert chain.job_consolidate(FakeRuntime(client=None)) == {"done": "cons"} and calls[-1][1] is None
    with pytest.raises(chain.StepUnavailable, match="晨报功能在这个版本里还不可用"):
        chain.job_brief(FakeRuntime())
