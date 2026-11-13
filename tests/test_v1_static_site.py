"""ui/static_site.py: the page and its files (SPEC v1 section 9.2).

* /static answers `Cache-Control: no-cache` from a StaticFiles subclass (no BaseHTTPMiddleware around the app);
* `.js` / `.mjs` are always `text/javascript` (native ES modules; Python's mimetypes reads the Windows registry);
* the `__V__` stamp = package version + newest mtime over ALL files under static/, sub-folders included;
* GET / never answers 500 because a file is missing.

The tests build their own static directory: the real one belongs to the frontend package and changes under us.
"""
from __future__ import annotations

import logging
import mimetypes
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio_note.runtime import Runtime
from audio_note.ui import static_site
from audio_note.ui.app import create_app

INDEX = '<!doctype html><title>audio-note</title><link rel="stylesheet" href="/static/style.css?v=__V__">' \
        '<script type="module" src="/static/app.js?v=__V__"></script>'


def _isolated(tmp_path, **extra) -> dict:
    """Config overrides for a Runtime under test: no vector model, and the glossary inside the temp dir
    (its default location is <project root>/glossary.txt, which a test must never write)."""
    return {"memory": {"retrieval": "fts", "embedding_model": "hash"},
            "correction": {"glossary_file": str(tmp_path / "glossary.txt")}, **extra}


@pytest.fixture
def rt(cfg_factory, tmp_path) -> Runtime:
    r = Runtime(cfg_factory(_isolated(tmp_path)))
    yield r
    r.shutdown()
    from audio_note.logging_setup import _remove_own_handlers

    _remove_own_handlers(logging.getLogger())


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty static directory the app under test serves instead of the real one."""
    d = tmp_path / "static"
    d.mkdir()
    monkeypatch.setattr(static_site, "STATIC_DIR", d)
    return d


def _put(path: Path, text: str, mtime: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


# --- the stamp -----------------------------------------------------------------------------------------------------------

def test_newest_mtime_covers_every_file_in_every_sub_folder(site: Path):
    assert static_site.newest_mtime(site) == 0                                   # nothing there yet
    _put(site / "index.html", INDEX, 1_700_000_000)
    _put(site / "app.js", 'import "./js/main.js";', 1_700_000_100)
    assert static_site.newest_mtime(site) == 1_700_000_100
    _put(site / "js" / "pages" / "timeline.js", "export {};", 1_700_000_500)     # a module deep down, not named anywhere
    assert static_site.newest_mtime(site) == 1_700_000_500
    _put(site / "css" / "tokens.css", ":root{}", 1_700_000_300)                  # older than the newest: no change
    assert static_site.newest_mtime(site) == 1_700_000_500
    (site / "empty-folder").mkdir()
    assert static_site.newest_mtime(site) == 1_700_000_500
    assert static_site.newest_mtime(site / "does-not-exist") == 0
    assert static_site.newest_mtime() == static_site.newest_mtime(site)         # default: the module's STATIC_DIR, read at call time


def test_asset_stamp_is_version_plus_newest_mtime(site: Path, monkeypatch: pytest.MonkeyPatch):
    import audio_note

    _put(site / "js" / "main.js", "export {};", 1_700_000_777)
    monkeypatch.setattr(audio_note, "__version__", "9.8.7", raising=False)
    assert static_site.package_version() == "9.8.7" and static_site.asset_stamp(site) == "9.8.7-1700000777"
    # without the attribute: the installed distribution's version, never an exception
    monkeypatch.delattr(audio_note, "__version__", raising=False)
    v = static_site.package_version()
    assert isinstance(v, str) and v and static_site.asset_stamp(site) == f"{v}-1700000777"


# --- GET / -----------------------------------------------------------------------------------------------------------------

def test_index_is_stamped_and_does_not_need_app_js_or_style_css(rt: Runtime, site: Path):
    _put(site / "index.html", INDEX, 1_700_000_000)                              # the only file: no app.js, no style.css
    with TestClient(create_app(rt), raise_server_exceptions=False) as c:
        r = c.get("/")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html") and r.headers["cache-control"] == "no-cache"
        stamp = static_site.asset_stamp(site)
        assert stamp.endswith("-1700000000") and "__V__" not in r.text
        assert f"/static/app.js?v={stamp}" in r.text and f"/static/style.css?v={stamp}" in r.text
        assert c.get("/static/app.js").status_code == 404                        # its own request says so
        # any file changing anywhere under static/ changes the stamp of the next page load (no restart)
        _put(site / "js" / "ui" / "modal.mjs", "export {};", 1_700_009_999)
        assert f"?v={static_site.package_version()}-1700009999" in c.get("/").text


def test_missing_index_html_is_a_readable_404_not_a_500(rt: Runtime, site: Path):
    with TestClient(create_app(rt), raise_server_exceptions=False) as c:
        r = c.get("/")
        assert r.status_code == 404 and "index.html" in r.json()["error"]
        assert c.get("/api/status").status_code == 200


def test_without_a_static_directory_the_api_still_works(rt: Runtime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(static_site, "STATIC_DIR", tmp_path / "nowhere")
    app = create_app(rt)
    assert static_site.mount_static(app) is False
    with TestClient(app, raise_server_exceptions=False) as c:
        assert c.get("/").status_code == 404 and c.get("/static/app.js").status_code == 404
        assert c.get("/api/status").status_code == 200


# --- /static ---------------------------------------------------------------------------------------------------------------

def test_static_files_are_no_cache_with_pinned_types(rt: Runtime, site: Path, monkeypatch: pytest.MonkeyPatch):
    # a machine whose registry maps .js to something a browser will not run as a module
    import starlette.responses

    wrong = lambda url, strict=True: ("text/plain", None)  # noqa: E731
    monkeypatch.setattr(mimetypes, "guess_type", wrong)
    monkeypatch.setattr(starlette.responses, "guess_type", wrong)       # FileResponse holds its own reference
    _put(site / "index.html", INDEX, 1_700_000_000)
    _put(site / "app.js", 'import "./js/main.js";', 1_700_000_000)
    _put(site / "js" / "main.js", "export const x = 1;", 1_700_000_000)
    _put(site / "js" / "core" / "util.mjs", "export const y = 2;", 1_700_000_000)
    _put(site / "css" / "base.css", "body{}", 1_700_000_000)
    _put(site / "img" / "logo.svg", "<svg xmlns='http://www.w3.org/2000/svg'/>", 1_700_000_000)
    _put(site / "data.json", "{}", 1_700_000_000)
    _put(site / "notes.txt", "plain", 1_700_000_000)
    with TestClient(create_app(rt), raise_server_exceptions=False) as c:
        expect = {"/static/app.js": "text/javascript; charset=utf-8", "/static/js/main.js": "text/javascript; charset=utf-8",
                  "/static/js/core/util.mjs": "text/javascript; charset=utf-8", "/static/css/base.css": "text/css; charset=utf-8",
                  "/static/img/logo.svg": "image/svg+xml", "/static/data.json": "application/json"}
        for url, ctype in expect.items():
            r = c.get(url)
            assert r.status_code == 200 and r.headers["content-type"] == ctype, url
            assert r.headers["cache-control"] == "no-cache" and r.headers["etag"], url
        assert c.get("/static/notes.txt").headers["content-type"] == "text/plain; charset=utf-8"   # not pinned: as guessed
        raw = starlette.responses.FileResponse(site / "app.js")
        assert raw.media_type == "text/plain"                                                     # the guess really is wrong here
        assert c.get("/static/js/main.js").text == "export const x = 1;"
        # the stamped URL of index.html is the same file
        assert c.get("/static/app.js", params={"v": "1.0.0-1"}).text == 'import "./js/main.js";'
        # revalidation: 304, still no-cache
        etag = c.get("/static/js/main.js").headers["etag"]
        r = c.get("/static/js/main.js", headers={"if-none-match": etag})
        assert r.status_code == 304 and r.headers["cache-control"] == "no-cache" and r.content == b""
        # errors are JSON like the rest of the API, and not cacheable either
        r = c.get("/static/js/nope.js")
        assert r.status_code == 404 and set(r.json()) == {"error"} and r.headers["cache-control"] == "no-cache"
        r = c.post("/static/app.js")
        assert r.status_code == 405 and r.headers["allow"] == "GET, HEAD" and set(r.json()) == {"error"}
        assert c.get("/static/../static_site.py").status_code == 404                                # no way out of the directory
        assert c.get("/static/%2e%2e/app.py").status_code == 404
        assert c.head("/static/app.js").status_code == 200


def test_api_responses_are_not_touched_by_the_static_cache_policy(rt: Runtime, site: Path):
    _put(site / "index.html", INDEX, 1_700_000_000)
    with TestClient(create_app(rt), raise_server_exceptions=False) as c:
        assert "cache-control" not in c.get("/api/status").headers
        assert "cache-control" not in c.get("/api/nope").headers


def test_the_real_static_directory_serves_a_page(rt: Runtime):
    """Whatever the frontend package ships: the page loads and its entry script is JavaScript."""
    if not (static_site.STATIC_DIR / "index.html").exists():
        pytest.skip("no index.html in ui/static")
    with TestClient(create_app(rt), raise_server_exceptions=False) as c:
        r = c.get("/")
        assert r.status_code == 200 and "audio-note" in r.text and "__V__" not in r.text
        assert static_site.asset_stamp() in r.text
        if (static_site.STATIC_DIR / "app.js").exists():
            r = c.get("/static/app.js")
            assert r.status_code == 200 and r.headers["content-type"] == "text/javascript; charset=utf-8"


def test_every_script_and_stylesheet_of_the_real_static_directory_has_the_right_type(rt: Runtime):
    """Native ES modules: a browser refuses a module (and ignores a stylesheet) served with another content type."""
    root = static_site.STATIC_DIR
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".js", ".mjs", ".css")] if root.exists() else []
    if not files:
        pytest.skip("no scripts in ui/static")
    with TestClient(create_app(rt), raise_server_exceptions=False) as c:
        for p in files:
            r = c.get("/static/" + p.relative_to(root).as_posix())
            if r.status_code == 404 and not p.exists():
                continue                                                         # removed by the frontend package meanwhile
            want = "text/css; charset=utf-8" if p.suffix.lower() == ".css" else "text/javascript; charset=utf-8"
            assert r.status_code == 200 and r.headers["content-type"] == want and r.headers["cache-control"] == "no-cache", p
