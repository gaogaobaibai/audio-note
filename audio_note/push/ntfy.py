"""ntfy channel: POST {server}/{topic} with the Markdown body and ``Title`` / ``Markdown`` headers.

HTTP headers must be ASCII, so a non-ASCII title is moved to the first line of the body and
``Title`` is set to ``audio_note``. ``msg.url`` becomes the ``Click`` action.
"""
from __future__ import annotations

import logging
from typing import Any

from .base import ChannelNotConfigured, Message, http_post, truncate_bytes

logger = logging.getLogger(__name__)

DEFAULT_SERVER = "https://ntfy.sh"
FALLBACK_TITLE = "audio_note"
BODY_LIMIT_BYTES = 4000  # ntfy default message limit is 4096 bytes


def build_url(server: str, topic: str) -> str:
    return f"{(server or DEFAULT_SERVER).rstrip('/')}/{topic.strip('/')}"


def build_payload(msg: Message) -> tuple[bytes, dict[str, str]]:
    """Return ``(body_bytes, headers)`` (pure)."""
    title = (msg.title or FALLBACK_TITLE).strip()
    body = msg.body_md or ""
    if not title.isascii():
        body = f"**{title}**\n\n{body}" if body else f"**{title}**"
        title = FALLBACK_TITLE
    headers = {"Title": title, "Markdown": "yes"}
    if msg.url and msg.url.isascii():
        headers["Click"] = msg.url
    elif msg.url:
        body = f"{body}\n\n{msg.url}"
    body = truncate_bytes(body, BODY_LIMIT_BYTES)
    return body.encode("utf-8"), headers


class NtfyChannel:
    name = "ntfy"

    def __init__(self, cfg: Any) -> None:
        sub = (cfg.get("push", {}) or {}).get("ntfy", {}) or {}
        self.topic: str = (sub.get("topic") or "").strip()
        self.server: str = (sub.get("server") or DEFAULT_SERVER).strip()
        if not self.topic:
            raise ChannelNotConfigured("push.ntfy.topic 为空")
        self.url = build_url(self.server, self.topic)

    def send(self, msg: Message) -> None:
        body, headers = build_payload(msg)
        resp = http_post(self.url, content=body, headers=headers)
        status = int(getattr(resp, "status_code", 0) or 0)
        if not 200 <= status < 300:
            raise RuntimeError(f"ntfy: HTTP {status} {(getattr(resp, 'text', '') or '')[:200]}")
