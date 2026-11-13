"""Error shapes of the web API: every failure is JSON `{"error": "<Chinese message>"}`.

    install_error_handlers(app)

| raised                                   | status | body                                                  |
|------------------------------------------|--------|-------------------------------------------------------|
| runtime.JobError / ReadOnly              | 409    | {"error": str(exc)}                                   |
| service.AlreadyRunning                   | 409    | {"error": str(exc)}                                   |
| HTTPException(status, "message")         | status | {"error": "message"}                                  |
| HTTPException(status, {...})             | status | the dict, "error" coerced to text (SampleMismatch)    |
| Starlette's own 404 / 405 (no such route)| same   | {"error": "..."}  (was {"detail": "Not Found"})       |
| RequestValidationError (bad path / query | 422    | {"error": "请求参数不正确：...", "detail": [{type, loc, |
|   / body type)                           |        |   msg}]}  (was {"detail": [...]})                     |
| anything else                            | 500    | {"error": "<TypeName>: <message>"} + a logged traceback|

Handler-level convention (in the routers): ValueError -> 400, LookupError / FileNotFoundError -> 404,
LLMError -> 502, FileExistsError / ProfileInUse -> 409.
"""
from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..runtime import JobError, ReadOnly
from ..service import AlreadyRunning

log = logging.getLogger(__name__)

#: messages for HTTP errors raised without one (Starlette's router / static files)
_STATUS_MESSAGES = {
    400: "请求不正确",
    401: "没有权限访问",
    403: "拒绝访问",
    404: "没有这个接口或页面",
    405: "这个接口不支持该请求方法",
}
_PARAM_PLACES = ("path", "query", "header", "cookie")
#: at most this many problems are spelled out in one 422 message
_MAX_PROBLEMS = 5


def err(status: int, msg: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=status, headers=headers or None)


def _where(loc: Any) -> str:
    """Readable name of the offending value: `uid`, `q`, `请求体`, `请求体.name`."""
    parts = [str(x) for x in (loc or ())]
    if parts and parts[0] in _PARAM_PLACES:
        parts = parts[1:]
    elif parts and parts[0] == "body":
        parts = ["请求体", *parts[1:]]
    return ".".join(parts) or "请求"


def _problem(e: dict[str, Any]) -> str:
    kind = str(e.get("type") or "")
    loc = e.get("loc") or ()
    where = _where(loc)
    if kind == "missing":
        return "缺少请求体" if tuple(loc) == ("body",) else f"缺少参数 {where}"
    if kind == "json_invalid":
        return "请求体不是合法的 JSON"
    if kind.startswith("int_"):
        return f"{where} 应为整数"
    if kind.startswith(("float_", "decimal_")):
        return f"{where} 应为数字"
    if kind.startswith("bool_"):
        return f"{where} 应为 true/false"
    if kind in ("dict_type", "model_attributes_type", "dataclass_type"):
        return f"{where} 应为 JSON 对象"
    if kind in ("list_type", "tuple_type", "set_type"):
        return f"{where} 应为列表"
    if kind.startswith("string_"):
        return f"{where} 应为文本"
    return f"{where} 不合法（{e.get('msg') or kind}）"


def validation_message(errors: list[dict[str, Any]]) -> str:
    """One Chinese sentence for a RequestValidationError (pydantic's own messages are English)."""
    problems = list(dict.fromkeys(_problem(e) for e in errors))
    more = f" 等 {len(problems)} 处" if len(problems) > _MAX_PROBLEMS else ""
    return "请求参数不正确：" + "；".join(problems[:_MAX_PROBLEMS]) + more


def install_error_handlers(app: FastAPI) -> None:
    async def _conflict(request: Request, exc: Exception) -> JSONResponse:
        return err(409, str(exc))

    for cls in (JobError, ReadOnly, AlreadyRunning):
        app.add_exception_handler(cls, _conflict)

    # registered for Starlette's class: fastapi.HTTPException derives from it, and the router / StaticFiles
    # raise the base class for "no such route" / "method not allowed" / "no such file"
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        headers = getattr(exc, "headers", None)
        if isinstance(exc.detail, dict):
            body = dict(exc.detail)
            body["error"] = str(body.get("error") or "")
            return JSONResponse(body, status_code=exc.status_code, headers=headers or None)
        detail = str(exc.detail) if exc.detail is not None else ""
        try:
            phrase = HTTPStatus(exc.status_code).phrase
        except ValueError:
            phrase = ""
        if not detail or detail == phrase:       # raised without a message of its own
            detail = _STATUS_MESSAGES.get(exc.status_code, detail or f"HTTP {exc.status_code}")
        return err(exc.status_code, detail, headers)

    app.add_exception_handler(StarletteHTTPException, _http_error)

    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        raw = list(exc.errors())
        # the offending input is not echoed back (it can be a whole upload); type / loc / msg are enough to debug
        detail = [{"type": str(e.get("type") or ""), "loc": [x if isinstance(x, int) else str(x) for x in e.get("loc") or ()],
                   "msg": str(e.get("msg") or "")} for e in raw]
        return JSONResponse({"error": validation_message(raw), "detail": detail}, status_code=422)

    app.add_exception_handler(RequestValidationError, _validation_error)

    async def _any_error(request: Request, exc: Exception) -> JSONResponse:
        log.exception("api %s %s failed: %s", request.method, request.url.path, exc)
        return err(500, f"{type(exc).__name__}: {exc}")

    app.add_exception_handler(Exception, _any_error)
