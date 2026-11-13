"""HTTP API + static files for the local web UI: the composition root.

    app = create_app(runtime)          # runtime: audio_note.runtime.Runtime, or a RuntimeHolder

Everything is JSON under /api; the single-page app in ui/static is served at /. The API has no authentication:
ui.host must stay 127.0.0.1, and deps.LocalGuard refuses requests that are not addressed to this machine or that
another site's page tries to make through the browser.

Where things live (one file per concern, see each module's docstring):

    deps.py             the Runtime of a request (RT), argument parsers, the rewriting() gate, LocalGuard
    errors.py           every failure is {"error": "..."}: install_error_handlers
    serializers.py      utt_dict - the one place an utterance becomes JSON
    settings_schema.py  what the settings page may change and how values are validated
    static_site.py      GET / and /static (no-cache static files, asset stamp)
    routers/            the endpoints, one APIRouter per domain; routers.ROUTER_MODULES is the inclusion order

Endpoints (see the routers for parameters)
  status / service        GET /api/status   POST /api/recording/{start,stop}   POST /api/service/{start,stop}
  jobs                    GET /api/jobs     POST /api/jobs/{name}   (any job the runtime knows: runtime.JOBS)
  profiles                GET,POST /api/profiles   POST /api/profiles/{name}/switch   DELETE /api/profiles/{name}
  transcripts             GET /api/days  /api/transcript/{day}  /api/search  /api/context/{id}  /api/segments
  audio                   GET /api/audio/utterance/{id}  /api/audio/segment/{id}?from=&to=   POST /api/import
  daily digests / notes   GET /api/daily  /api/daily/{day}   GET,DELETE /api/notes
  speakers                GET /api/speakers  POST /api/speakers/{name}/samples?force=  POST /api/speakers/label
                          PATCH,DELETE /api/speakers/{id}  GET /api/speakers/{id}/samples  DELETE /api/speakers/samples/{id}
                          DELETE /api/speakers/{id}/samples/outliers  POST /api/speakers/calibrate
                          GET /api/speakers/{id}/negatives  GET,DELETE /api/speakers/negatives/{id}[/audio]
                          GET /api/speakers/{id}/review   POST /api/utterances/{id}/reject  POST /api/utterances/{id}/confirm
  unknown voices          GET /api/voices   POST /api/voices/name   POST /api/voices/reject   (name / reject a voice group)
  mixed utterances        POST /api/utterances/{id}/split {auto | at: [s, ...]}   GET /api/utterances/{id}/preview-split?auto=1
  memory chat             POST /api/chat   (NDJSON stream: tool / answer / error / ping events)
  settings                GET,PUT /api/config   GET,PUT /api/glossary   GET /api/devices[/system]   POST /api/config/test-api
                          POST /api/push/test   GET /api/logs
"""
from __future__ import annotations

import logging

from fastapi import FastAPI

from ..runtime import Runtime, RuntimeHolder
from .deps import install_local_guard
from .errors import install_error_handlers
from .routers import load_routers
from .static_site import mount_static

log = logging.getLogger(__name__)


def create_app(rt_or_holder: Runtime | RuntimeHolder) -> FastAPI:
    """Build the app around a Runtime (or the RuntimeHolder that lets the UI switch profiles).

    Nothing here binds to an event loop or needs a lifespan hook: UIServer runs uvicorn with lifespan off, and
    the tests drive one app from several TestClients (several loops) at once.
    """
    app = FastAPI(title="audio-note", docs_url=None, redoc_url=None, openapi_url=None)
    holder = rt_or_holder if isinstance(rt_or_holder, RuntimeHolder) else RuntimeHolder(rt_or_holder)
    app.state.holder = holder
    install_error_handlers(app)
    install_local_guard(app, holder)
    for _name, router in load_routers():
        app.include_router(router)
    mount_static(app)
    return app


# Names the single-file app.py used to define, kept importable from here.
from .serializers import utt_dict as _utt_dict  # noqa: E402,F401
from .settings_schema import EDITABLE, SECRET_KEYS, coerce_setting  # noqa: E402,F401
from .static_site import STATIC_DIR  # noqa: E402,F401

__all__ = ["create_app"]
