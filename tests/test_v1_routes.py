"""Characterization of the HTTP route table (SPEC v1 section 8: existing paths must not change).

BASELINE is the route table of the pre-split monolith (commit a1f36ac): 54 API routes + GET / + the /static
mount. The split into ui/routers/* must keep every one of them, resolving to the same handler, whatever new
endpoints later packages add next to them.
"""
from __future__ import annotations

import logging
import re

import pytest
from starlette.routing import Match, Mount

from audio_note.runtime import Runtime
from audio_note.ui.app import create_app

#: (method, path template, handler name) in the registration order of the monolith
BASELINE: list[tuple[str, str, str]] = [
    ("GET", "/api/status", "status"),
    ("POST", "/api/recording/start", "recording_start"),
    ("POST", "/api/recording/stop", "recording_stop"),
    ("POST", "/api/service/start", "service_start"),
    ("POST", "/api/service/stop", "service_stop"),
    ("GET", "/api/profiles", "profiles_list"),
    ("POST", "/api/profiles", "profiles_create"),
    ("POST", "/api/profiles/{name}/switch", "profiles_switch"),
    ("DELETE", "/api/profiles/{name}", "profiles_delete"),
    ("GET", "/api/jobs", "jobs"),
    ("POST", "/api/jobs/{name}", "job_start"),
    ("GET", "/api/days", "days"),
    ("GET", "/api/transcript/{day}", "transcript"),
    ("GET", "/api/segments", "segments"),
    ("GET", "/api/search", "search"),
    ("GET", "/api/context/{uid}", "context"),
    ("GET", "/api/audio/utterance/{uid}", "audio_utterance"),
    ("GET", "/api/audio/segment/{seg_id}", "audio_segment"),
    ("GET", "/api/daily", "daily_list"),
    ("GET", "/api/daily/{day}", "daily_get"),
    ("GET", "/api/notes", "notes"),
    ("DELETE", "/api/notes/{note_id}", "note_delete"),
    ("GET", "/api/speakers", "speakers_list"),
    ("GET", "/api/voices", "voices_list"),
    ("POST", "/api/voices/name", "voices_name"),
    ("GET", "/api/speakers/{sid}/samples", "speaker_samples"),
    ("DELETE", "/api/speakers/{sid}/samples/outliers", "speaker_delete_outlier_samples"),
    ("GET", "/api/speakers/samples/{emb_id}/audio", "speaker_sample_audio"),
    ("DELETE", "/api/speakers/samples/{emb_id}", "speaker_sample_delete"),
    ("GET", "/api/speakers/{sid}/negatives", "speaker_negatives"),
    ("DELETE", "/api/speakers/negatives/{neg_id}", "speaker_negative_delete"),
    ("GET", "/api/speakers/negatives/{neg_id}/audio", "speaker_negative_audio"),
    ("GET", "/api/speakers/{sid}/review", "speaker_review"),
    ("POST", "/api/utterances/{uid}/reject", "utterance_reject"),
    ("POST", "/api/utterances/{uid}/confirm", "utterance_confirm"),
    ("POST", "/api/voices/reject", "voices_reject"),
    ("POST", "/api/speakers/{name}/samples", "speaker_add_sample"),
    ("POST", "/api/speakers/label", "speaker_label"),
    ("PATCH", "/api/speakers/{sid}", "speaker_update"),
    ("DELETE", "/api/speakers/{sid}", "speaker_delete"),
    ("POST", "/api/speakers/calibrate", "speakers_calibrate"),
    ("GET", "/api/utterances/{uid}/preview-split", "utterance_preview_split"),
    ("POST", "/api/utterances/{uid}/split", "utterance_split"),
    ("POST", "/api/chat", "chat"),
    ("GET", "/api/devices", "devices"),
    ("GET", "/api/devices/system", "system_devices"),
    ("GET", "/api/config", "config_get"),
    ("PUT", "/api/config", "config_put"),
    ("POST", "/api/config/test-api", "config_test_api"),
    ("POST", "/api/push/test", "push_test"),
    ("GET", "/api/glossary", "glossary_get"),
    ("PUT", "/api/glossary", "glossary_put"),
    ("GET", "/api/logs", "logs"),
    ("POST", "/api/import", "import_audio"),
    ("GET", "/", "index"),
]


def _isolated(tmp_path, **extra) -> dict:
    """Config overrides for a Runtime under test: no vector model, and the glossary inside the temp dir
    (its default location is <project root>/glossary.txt, which a test must never write)."""
    return {"memory": {"retrieval": "fts", "embedding_model": "hash"},
            "correction": {"glossary_file": str(tmp_path / "glossary.txt")}, **extra}


#: a concrete value for every path parameter of the baseline
_PARAM_VALUES = {"day": "2026-10-30", "name": "demo"}


@pytest.fixture
def app(cfg_factory, tmp_path):
    cfg = cfg_factory(_isolated(tmp_path))
    rt = Runtime(cfg)
    try:
        yield create_app(rt)
    finally:
        rt.shutdown()
        from audio_note.logging_setup import _remove_own_handlers

        _remove_own_handlers(logging.getLogger())


def _flat(app) -> list:
    """Every route of the app in matching order, included routers expanded. Older FastAPI copies the routes of
    an included router into app.routes; newer versions keep one lazy entry per router (iter_route_contexts
    expands those). Each item has .path / .name / .methods / .matches."""
    try:
        from fastapi.routing import iter_route_contexts
    except ImportError:
        return list(app.routes)
    return list(iter_route_contexts(app.routes))


def _is_mount(r) -> bool:
    return isinstance(getattr(r, "route", r), Mount)


def _table(app) -> list[tuple[str, str, str]]:
    """(method, path, handler name) of every plain route, in matching order (HEAD is implied by GET)."""
    rows = []
    for r in _flat(app):
        if _is_mount(r):
            continue
        for m in sorted(getattr(r, "methods", None) or []):
            if m != "HEAD":
                rows.append((m, r.path, r.name))
    return rows


def _concrete(template: str) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: _PARAM_VALUES.get(m.group(1), "7"), template)


def _resolve(app, method: str, path: str):
    """The route a request is dispatched to: the first full match in registration order (Starlette's rule; a
    router included as one entry answers "full" exactly when one of its routes does, so expanding is the same)."""
    scope = {"type": "http", "method": method, "path": path, "root_path": "", "query_string": b"", "headers": []}
    for r in _flat(app):
        match, _ = r.matches(scope)
        if match == Match.FULL:
            return r
    return None


def test_baseline_snapshot_is_complete():
    assert len(BASELINE) == 55 and len({(m, p) for m, p, _ in BASELINE}) == 55
    assert sum(1 for _, p, _ in BASELINE if p.startswith("/api/")) == 54


def test_every_baseline_route_is_still_registered(app):
    have = {(m, p) for m, p, _ in _table(app)}
    missing = [(m, p) for m, p, _ in BASELINE if (m, p) not in have]
    assert missing == []
    mounts = [r for r in _flat(app) if _is_mount(r)]
    assert [(m.path, m.name) for m in mounts] == [("/static", "static")]


def test_baseline_routes_keep_their_handler_names(app):
    names = {(m, p): n for m, p, n in _table(app)}
    assert {(m, p): names.get((m, p)) for m, p, _ in BASELINE} == {(m, p): n for m, p, n in BASELINE}


@pytest.mark.parametrize("method,template,name", BASELINE, ids=[f"{m} {p}" for m, p, _ in BASELINE])
def test_baseline_request_resolves_to_the_same_handler(app, method: str, template: str, name: str):
    """No route added before or after may shadow an existing one (static paths vs {param} paths)."""
    r = _resolve(app, method, _concrete(template))
    assert r is not None and (r.path, r.name) == (template, name)


def test_no_method_and_path_is_registered_twice(app):
    rows = [(m, p) for m, p, _ in _table(app)]
    dup = sorted({x for x in rows if rows.count(x) > 1})
    assert dup == []


def test_everything_that_is_not_baseline_lives_under_api(app):
    base = {(m, p) for m, p, _ in BASELINE}
    extra = [(m, p) for m, p, _ in _table(app) if (m, p) not in base]
    assert all(p.startswith("/api/") for _, p in extra), extra


# --- assembly: routers.ROUTER_MODULES drives create_app -------------------------------------------------------------

def test_router_modules_are_listed_in_a_fixed_order_with_the_stubs_of_later_packages():
    from audio_note.ui import routers

    names = routers.ROUTER_MODULES
    core = ["system", "profiles", "transcripts", "audio", "daily", "enroll", "speakers", "voices", "utterances", "chat", "settings"]
    assert names[:len(core)] == core                      # enroll before speakers: its static POST paths go first
    assert len(names) == len(set(names))
    for later in ("events", "inbox", "lexicon", "sessions", "memory"):
        assert later in names
    assert routers.OPTIONAL_MODULES == {"events"}


@pytest.mark.parametrize("name", ["inbox", "lexicon", "sessions", "memory"])
def test_feature_routers_of_later_packages_exist_and_use_the_api_prefix(name: str):
    import importlib

    from fastapi import APIRouter

    mod = importlib.import_module(f"audio_note.ui.routers.{name}")
    assert isinstance(mod.router, APIRouter) and mod.router.prefix == "/api"
    assert (mod.__doc__ or "").strip()
    assert all(r.path.startswith("/api/") for r in mod.router.routes if hasattr(r, "path"))


def test_every_router_module_of_this_package_defines_an_api_router():
    import importlib

    from fastapi import APIRouter

    from audio_note.ui import routers

    for name in routers.ROUTER_MODULES:
        if name in routers.OPTIONAL_MODULES:
            continue
        mod = importlib.import_module(f"audio_note.ui.routers.{name}")
        assert isinstance(mod.router, APIRouter) and mod.router.prefix == "/api", name


def test_load_routers_keeps_the_order_and_skips_what_does_not_import(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    import importlib
    import sys
    import types

    from audio_note.ui import routers

    def fake_import(qualified: str):
        if qualified.endswith(".explodes"):
            raise RuntimeError("boom at import")
        return importlib.import_module(qualified)

    monkeypatch.setattr(routers, "importlib", types.SimpleNamespace(import_module=fake_import))
    monkeypatch.setitem(sys.modules, "audio_note.ui.routers.no_router_attr", types.ModuleType("audio_note.ui.routers.no_router_attr"))
    monkeypatch.setitem(sys.modules, "audio_note.ui.routers.events", None)          # "the file is not there"
    with caplog.at_level(logging.DEBUG, logger="audio_note.ui.routers"):
        got = routers.load_routers(["daily", "explodes", "does_not_exist", "no_router_attr", "events", "system"])
    assert [n for n, _ in got] == ["daily", "system"]
    by_level = {r.levelname: [] for r in caplog.records}
    for r in caplog.records:
        by_level[r.levelname].append(r.getMessage())
    errors = "\n".join(by_level.get("ERROR", []))
    assert "explodes" in errors and "does_not_exist" in errors and "no_router_attr" in errors
    assert "events" not in errors                                                    # optional: its absence is normal
    assert any("events" in m for m in by_level.get("DEBUG", []))


def test_app_is_created_without_the_optional_events_router(cfg_factory, tmp_path, monkeypatch: pytest.MonkeyPatch):
    import sys

    monkeypatch.setitem(sys.modules, "audio_note.ui.routers.events", None)
    cfg = cfg_factory(_isolated(tmp_path))
    rt = Runtime(cfg)
    try:
        app = create_app(rt)
        have = {(m, p) for m, p, _ in _table(app)}
        assert all((m, p) in have for m, p, _ in BASELINE)
        assert ("GET", "/api/events") not in have
    finally:
        rt.shutdown()
        from audio_note.logging_setup import _remove_own_handlers

        _remove_own_handlers(logging.getLogger())


def test_a_broken_router_module_does_not_take_the_app_down(cfg_factory, tmp_path, monkeypatch: pytest.MonkeyPatch):
    """One feature that fails to import is skipped (and logged); everything else is served."""
    import importlib
    import types

    from fastapi.testclient import TestClient

    from audio_note.ui import routers

    def fake_import(qualified: str):
        if qualified.endswith(".memory"):
            raise ImportError("half-written module")
        return importlib.import_module(qualified)

    monkeypatch.setattr(routers, "importlib", types.SimpleNamespace(import_module=fake_import))
    cfg = cfg_factory(_isolated(tmp_path))
    rt = Runtime(cfg)
    try:
        with TestClient(create_app(rt), raise_server_exceptions=False) as c:
            assert c.get("/api/status").status_code == 200
            assert c.get("/api/days").json() == []
    finally:
        rt.shutdown()
        from audio_note.logging_setup import _remove_own_handlers

        _remove_own_handlers(logging.getLogger())


def test_create_app_is_a_small_composition_root():
    """The split stays split: app.py wires things together and holds no handlers of its own."""
    import inspect

    from audio_note.ui import app as app_mod

    src = inspect.getsource(app_mod)
    assert len(src.splitlines()) < 120
    assert "@app." not in src and "@router." not in src
    assert app_mod.create_app.__module__ == "audio_note.ui.app"
