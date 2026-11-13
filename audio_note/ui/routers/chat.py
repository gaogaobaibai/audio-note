"""Memory chat.

    POST /api/chat {question, history?: [{role, content}], strategy?, retrieval?, expand?}

Answers with an NDJSON stream (`application/x-ndjson`, one JSON object per line):

    {"type": "start", "strategy", "retrieval", "model"}
    {"type": "tool", "name", "args", "result" (first 1500 chars), "lines"}      one per tool call
    {"type": "answer", "text", "stats": {..., "seconds"}}   |   {"type": "error", "message"}
    {"type": "ping"}                                         after 15 s of silence

Validation errors (400) and a missing API key (409, JobError from rt.chat) are plain JSON, sent before the stream
starts. The reader on the page ignores event types it does not know, so new ones can be added.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Any

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import StreamingResponse

from ..deps import RT, bool_arg

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

#: how many previous turns of the page's conversation are passed to the model
MAX_HISTORY = 20


@router.post("/chat")
def chat(rt: RT, body: dict[str, Any] = Body(...)) -> StreamingResponse:
    question = str(body.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "问题不能为空")
    history = [{"role": m["role"], "content": str(m.get("content") or "")}
               for m in (body.get("history") or []) if isinstance(m, dict) and m.get("role") in ("user", "assistant")][-MAX_HISTORY:]
    strategy = body.get("strategy") or None
    retrieval = body.get("retrieval") or None
    expand = body.get("expand")
    chat_obj = rt.chat(strategy, retrieval, None if expand is None else bool_arg(expand, "expand"))   # raises JobError without a key
    q: queue.Queue = queue.Queue()

    def on_tool(name: str, args: dict, result: str) -> None:
        q.put({"type": "tool", "name": name, "args": args, "result": (result or "")[:1500],
               "lines": (result or "").count("\n") + 1})

    chat_obj.on_tool = on_tool

    def work() -> None:
        t0 = time.monotonic()
        try:
            answer = chat_obj.ask(question, history)
            stats = dict(chat_obj.last_stats)
            stats["seconds"] = round(time.monotonic() - t0, 1)
            q.put({"type": "answer", "text": answer, "stats": stats})
        except Exception as e:  # noqa: BLE001
            log.exception("chat failed: %s", e)
            q.put({"type": "error", "message": str(e)})
        finally:
            q.put(None)

    threading.Thread(target=work, name="chat", daemon=True).start()

    def gen():
        yield json.dumps({"type": "start", "strategy": chat_obj.strategy, "retrieval": chat_obj.retriever.mode,
                          "model": chat_obj.model}, ensure_ascii=False) + "\n"
        while True:
            try:
                item = q.get(timeout=15.0)
            except queue.Empty:
                yield json.dumps({"type": "ping"}) + "\n"
                continue
            if item is None:
                break
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
