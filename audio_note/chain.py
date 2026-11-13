"""Post-processing chain: what happens to new sentences after they are stored, in the background.

    sessions.sync  ->  scoring.run_pending  ->  refine.run_pending  ->  episodes.run_pending     (STEPS, in this order)
    after the daily digest:  memory.consolidate.run        every morning:  memory.brief.run

Every step is a plain function with one fixed signature

    fn(cfg, db, client, *, stop_event=None, events=None, limit=None) -> dict

(`client` is None when no API key is configured: the step decides what it can still do). Steps are named by
"module:function" strings and imported when they run, so this module depends on none of them:

* a step whose module does not exist yet is skipped silently (the feature is simply not installed);
* a step that raises is logged and reported, the following steps still run;
* the MiMo client is only built when the first step that exists is about to run.

The scheduler runs the chain after every `process`, on a tick that finds finished-but-untreated sessions, and
right before the daily digest. Runtime jobs (`chain`, `refine`, `episodes`, `consolidate`, `brief`) call the
job_* wrappers at the bottom.
"""
from __future__ import annotations

import importlib
import logging
import time
from typing import Any, Callable, Iterable

log = logging.getLogger(__name__)

#: (name, "module:function") in running order
STEPS: tuple[tuple[str, str], ...] = (
    ("sessions", "audio_note.memory.sessions:sync"),
    ("rescore", "audio_note.speech.scoring:run_pending"),
    ("refine", "audio_note.speech.refine:run_pending"),
    ("episodes", "audio_note.memory.episodes:run_pending"),
)
#: same signature, run by the scheduler outside the chain
CONSOLIDATE = "audio_note.memory.consolidate:run"
BRIEF = "audio_note.memory.brief:run"
#: usage_llm purpose given to the client a step receives (a step may scope further)
PURPOSES = {"sessions": "sessions", "refine": "refine", "episodes": "episode", "consolidate": "consolidate", "brief": "brief"}
LABELS = {"sessions": "会话切分", "rescore": "重判说话人", "refine": "会话校对", "episodes": "会话摘要", "consolidate": "记忆整理",
          "brief": "晨报"}


class StepUnavailable(RuntimeError):
    """The module (or function) behind a step does not exist in this installation."""


def resolve(target: str) -> Callable[..., Any]:
    """Import "package.module:function". Raises StepUnavailable when that module (or one of its parent
    packages) or the function does not exist; an import error INSIDE an existing module propagates."""
    mod_name, _, fn_name = str(target).partition(":")
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        missing = e.name or ""
        if missing == mod_name or mod_name.startswith(missing + "."):
            raise StepUnavailable(f"模块 {mod_name} 不存在") from e
        raise
    fn = getattr(mod, fn_name, None) if fn_name else None
    if not callable(fn):
        raise StepUnavailable(f"模块 {mod_name} 里没有 {fn_name or '(未指定函数)'}")
    return fn


def available(target: str) -> bool:
    try:
        resolve(target)
        return True
    except StepUnavailable:
        return False
    except Exception:  # noqa: BLE001 - it exists but is broken: running it will report the error
        return True


def has_steps(names: Iterable[str] | None = None) -> bool:
    """True when at least one chain step (of `names`, default all) is installed."""
    wanted = set(names) if names is not None else None
    return any(available(t) for n, t in STEPS if wanted is None or n in wanted)


def _scoped(client: Any, name: str) -> Any:
    scoped = getattr(client, "scoped", None) if client is not None else None
    if callable(scoped):
        try:
            return scoped(PURPOSES.get(name, name))
        except Exception:  # noqa: BLE001
            return client
    return client


class _LazyClient:
    """Calls the factory at most once, the first time a step needs the client; a failure means None."""

    def __init__(self, factory: Callable[[], Any] | None):
        self._factory = factory
        self._done = False
        self._client: Any = None

    def get(self) -> Any:
        if not self._done:
            self._done = True
            if self._factory is not None:
                try:
                    self._client = self._factory()
                except Exception as e:  # noqa: BLE001 - no key / bad config: steps run without a client
                    log.debug("chain: no LLM client (%s)", e)
                    self._client = None
        return self._client


def call_step(target: str, cfg: Any, db: Any, client: Any, *, name: str | None = None, stop_event: Any = None,
              events: Any = None, limit: int | None = None) -> dict[str, Any]:
    """Run one step function by its target. Raises StepUnavailable when it is not installed and whatever
    the step raises otherwise. A non-dict result is wrapped as {"result": ...}."""
    fn = resolve(target)
    res = fn(cfg, db, _scoped(client, name) if name else client, stop_event=stop_event, events=events, limit=limit)
    return res if isinstance(res, dict) else {"result": res}


def run_chain(cfg: Any, db: Any, client_factory: Callable[[], Any] | None, *, stop_event: Any = None, events: Any = None,
              only: Iterable[str] | None = None, limit: int | None = None) -> dict[str, Any]:
    """Run the steps of STEPS in order (`only`: just these names).

    Returns {"ran": [names], "skipped": [names], "failed": {name: "Type: message"}, "results": {name: dict},
             "wall_s": seconds}.
    Missing steps land in "skipped" without a log line; a failing step lands in "failed" (logged with its
    traceback) and does not stop the others; once `stop_event` is set the remaining steps are skipped."""
    wanted = {str(n) for n in only} if only is not None else None
    out: dict[str, Any] = {"ran": [], "skipped": [], "failed": {}, "results": {}}
    lazy = _LazyClient(client_factory)
    t0 = time.perf_counter()
    for name, target in STEPS:
        if wanted is not None and name not in wanted:
            continue
        if stop_event is not None and stop_event.is_set():
            out["skipped"].append(name)
            continue
        try:
            fn = resolve(target)
        except StepUnavailable:
            out["skipped"].append(name)
            continue
        except Exception as e:  # noqa: BLE001 - the module exists but cannot be imported
            log.exception("chain step %s cannot be loaded: %s", name, e)
            out["failed"][name] = f"{type(e).__name__}: {e}"
            continue
        t1 = time.perf_counter()
        try:
            res = fn(cfg, db, _scoped(lazy.get(), name), stop_event=stop_event, events=events, limit=limit)
        except Exception as e:  # noqa: BLE001 - one step must not take the others down
            log.exception("chain step %s failed: %s", name, e)
            out["failed"][name] = f"{type(e).__name__}: {e}"
            continue
        out["ran"].append(name)
        out["results"][name] = res if isinstance(res, dict) else {"result": res}
        log.debug("chain step %s done in %.1fs: %s", name, time.perf_counter() - t1, out["results"][name])
    out["wall_s"] = round(time.perf_counter() - t0, 3)
    return out


# --- runtime jobs (targets of runtime.JOB_TARGETS: fn(rt, **kwargs) -> dict) ---------------------------------

def _rt_client_factory(rt: Any) -> Callable[[], Any]:
    def factory() -> Any:
        try:
            return rt.client()
        except Exception:  # noqa: BLE001 - JobError: no API key
            return None

    return factory


def _rt_stop(rt: Any) -> Any:
    fn = getattr(rt, "job_stop_event", None)
    return fn() if callable(fn) else None


def _require(name: str, target: str) -> None:
    try:
        resolve(target)
    except StepUnavailable as e:
        raise StepUnavailable(f"{LABELS.get(name, name)}功能在这个版本里还不可用（{e}）") from e


def job_chain(rt: Any, only: Iterable[str] | None = None, limit: int | None = None) -> dict[str, Any]:
    """整理新内容: the whole chain, now."""
    return run_chain(rt.cfg, rt.db, _rt_client_factory(rt), stop_event=_rt_stop(rt), events=getattr(rt, "events", None),
                     only=only, limit=limit)


def _job_steps(rt: Any, name: str, limit: int | None) -> dict[str, Any]:
    target = dict(STEPS)[name]
    _require(name, target)
    out = run_chain(rt.cfg, rt.db, _rt_client_factory(rt), stop_event=_rt_stop(rt), events=getattr(rt, "events", None),
                    only=("sessions", name), limit=limit)
    if name in out["failed"]:
        raise RuntimeError(f"{LABELS.get(name, name)}失败：{out['failed'][name]}")
    return out


def job_refine(rt: Any, limit: int | None = None) -> dict[str, Any]:
    """校对会话: cut sessions, then refine the finished ones."""
    return _job_steps(rt, "refine", limit)


def job_episodes(rt: Any, limit: int | None = None) -> dict[str, Any]:
    """会话摘要: cut sessions, then summarise the finished ones and extract memory items."""
    return _job_steps(rt, "episodes", limit)


def _job_single(rt: Any, name: str, target: str, limit: int | None) -> dict[str, Any]:
    _require(name, target)
    client = _rt_client_factory(rt)()
    return call_step(target, rt.cfg, rt.db, client, name=name, stop_event=_rt_stop(rt), events=getattr(rt, "events", None),
                     limit=limit)


def job_consolidate(rt: Any, limit: int | None = None) -> dict[str, Any]:
    """整理记忆: merge / supersede / close memory items (normally right after the daily digest)."""
    return _job_single(rt, "consolidate", CONSOLIDATE, limit)


def job_brief(rt: Any, limit: int | None = None) -> dict[str, Any]:
    """晨报: push the morning brief now."""
    return _job_single(rt, "brief", BRIEF, limit)


__all__ = ["STEPS", "CONSOLIDATE", "BRIEF", "StepUnavailable", "resolve", "available", "has_steps", "call_step", "run_chain",
           "job_chain", "job_refine", "job_episodes", "job_consolidate", "job_brief"]
