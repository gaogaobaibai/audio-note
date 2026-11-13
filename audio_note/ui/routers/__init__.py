"""The web API, one APIRouter per domain. `create_app` includes them in the order of ROUTER_MODULES.

    system       status, recording, service, jobs, logs
    profiles     profile list / create / switch / delete
    transcripts  days, transcript, segments, search, context
    audio        audio clips, import
    daily        daily digests, memory notes
    enroll       POST /api/speakers/{name}/samples, POST /api/speakers/label   (the two enrolment flows)
    speakers     list, patch, delete, calibrate, samples, negatives, review queue
    voices       unidentified voices: list / name / reject
    utterances   reject, confirm, preview-split, split
    chat         NDJSON memory chat
    settings     devices, config, test-api, push test, glossary
    events       SSE + metrics                    (package A1; optional: skipped while the file is absent)
    inbox / lexicon / sessions / memory           (packages B2 / B3; empty stubs until they land)

Rules for a router module:

* module-level `router = APIRouter(prefix="/api")`; handlers take `rt: RT` (ui/deps.py) - one Runtime per request;
* anything a test may monkeypatch (speech.voices, audio.capture, service.running_pid, the Runtime instance's
  methods, ...) is looked up at CALL time: `from ...x import name` inside the handler or `module.name(...)`,
  never hoisted to a module-level `from x import name`. That also keeps `import audio_note.ui.app` free of
  numpy / model imports;
* static paths before parameterised ones that could swallow them (`/speakers/label` vs `/speakers/{sid}`);
* errors: raise HTTPException(status, "中文说明") - see ui/errors.py for the shapes.

Order matters only where two routes of the same method could match one path; `enroll` sits before `speakers`
so its static POST paths are tried first. To add a router: create the module, append its name here.
"""
from __future__ import annotations

import importlib
import logging
from typing import Any

log = logging.getLogger(__name__)

#: router modules in inclusion order (append new ones at the end)
ROUTER_MODULES: list[str] = [
    "system",
    "profiles",
    "transcripts",
    "audio",
    "daily",
    "enroll",
    "speakers",
    "voices",
    "utterances",
    "chat",
    "settings",
    "events",
    "inbox",
    "lexicon",
    "sessions",
    "memory",
]
#: modules written by another package that may not exist yet: their absence is normal, not an error
OPTIONAL_MODULES: frozenset[str] = frozenset({"events"})


def load_routers(names: list[str] | None = None) -> list[tuple[str, Any]]:
    """Import every module of ROUTER_MODULES and return [(name, router)] in that order. A module that fails
    to import (or has no `router`) is logged and skipped: one broken feature must not take the whole UI down."""
    out: list[tuple[str, Any]] = []
    for name in (ROUTER_MODULES if names is None else names):
        qualified = f"{__name__}.{name}"
        try:
            mod = importlib.import_module(qualified)
        except ModuleNotFoundError as e:
            if e.name == qualified and name in OPTIONAL_MODULES:
                log.debug("router %s not present, skipped", name)
            else:
                log.exception("router %s skipped: %s", name, e)
            continue
        except Exception as e:  # noqa: BLE001 - e.g. a syntax error or a failing import inside the module
            log.exception("router %s skipped: %s: %s", name, type(e).__name__, e)
            continue
        router = getattr(mod, "router", None)
        if router is None or not hasattr(router, "routes"):
            log.error("router %s skipped: the module defines no `router`", name)
            continue
        out.append((name, router))
    return out
