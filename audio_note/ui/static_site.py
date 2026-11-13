"""The single-page app: `GET /` and the files under /static.

    mount_static(app)       # NoCacheStaticFiles at /static + the index route; a no-op when ui/static is absent

Cache policy (the page is small and local, an upgrade must never show a stale file):

* every static response carries `Cache-Control: no-cache`, so the browser revalidates on each load (ETag /
  Last-Modified -> 304). This matters for the native ES modules under static/js: only the entry points in
  index.html carry a version stamp, the modules they import are fetched by their plain URLs.
* `__V__` in index.html becomes `<package version>-<newest mtime of ANY file under static/>`.
* `.js` / `.mjs` are always served as `text/javascript` (Python's `mimetypes` reads the Windows registry, and a
  browser refuses to run a module that arrives with another type).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

#: content types that must not depend on the machine's registry / mime.types
MEDIA_TYPES = {
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".json": "application/json",
    ".map": "application/json",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
}


class NoCacheStaticFiles(StaticFiles):
    """StaticFiles whose responses (200, 304 and errors alike) say `Cache-Control: no-cache`, with pinned
    content types for the file kinds the page is made of. Replaces the BaseHTTPMiddleware that used to wrap
    every request of the app just to set this one header."""

    async def get_response(self, path: str, scope: Any) -> Response:
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            headers = dict(exc.headers or {})
            headers.setdefault("Cache-Control", "no-cache")
            raise StarletteHTTPException(exc.status_code, exc.detail, headers=headers) from None
        response.headers["Cache-Control"] = "no-cache"
        media_type = MEDIA_TYPES.get(os.path.splitext(path)[1].lower())
        if media_type and response.status_code != 304:
            response.headers["content-type"] = media_type
        return response


def package_version() -> str:
    """`audio_note.__version__`, else the installed distribution's version, else "0"."""
    try:
        import audio_note

        v = getattr(audio_note, "__version__", None)
        if v:
            return str(v)
    except Exception:  # noqa: BLE001
        pass
    try:
        from importlib.metadata import version

        return version("audio-note")
    except Exception:  # noqa: BLE001
        return "0"


def newest_mtime(static_dir: Path | str | None = None) -> int:
    """Newest modification time (whole seconds) over every file under the static directory, sub-folders
    included; 0 when there is none. Unreadable entries are skipped."""
    newest = 0.0
    stack = [str(static_dir if static_dir is not None else STATIC_DIR)]
    while stack:
        try:
            entries = list(os.scandir(stack.pop()))
        except OSError:
            continue
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path)
                elif e.is_file():
                    newest = max(newest, e.stat().st_mtime)
            except OSError:
                continue
    return int(newest)


def asset_stamp(static_dir: Path | str | None = None) -> str:
    """Cache-busting stamp for the asset URLs in index.html: changes with the package version and with any
    edit to any static file."""
    return f"{package_version()}-{newest_mtime(static_dir)}"


def render_index(static_dir: Path | str | None = None) -> Response:
    """index.html with `__V__` replaced. Missing style.css / app.js do not matter here (their own requests
    answer 404); only a missing index.html is an error, and then a readable 404, not a 500."""
    d = Path(static_dir) if static_dir is not None else STATIC_DIR
    try:
        html = (d / "index.html").read_text(encoding="utf-8")
    except OSError:
        raise HTTPException(404, "界面文件缺失（ui/static/index.html），请重新安装 audio-note")
    return Response(html.replace("__V__", asset_stamp(d)), media_type="text/html", headers={"Cache-Control": "no-cache"})


def mount_static(app: Any, static_dir: Path | str | None = None) -> bool:
    """Mount /static and register `GET /`. Returns False (and registers nothing) when the directory does
    not exist, e.g. an API-only install."""
    d = Path(static_dir) if static_dir is not None else STATIC_DIR
    if not d.exists():
        return False
    app.mount("/static", NoCacheStaticFiles(directory=str(d)), name="static")

    def index() -> Response:
        return render_index(d)

    app.add_api_route("/", index, methods=["GET"])
    return True
