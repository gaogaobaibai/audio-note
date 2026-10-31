"""Server酱 Turbo channel: POST https://sctapi.ftqq.com/{sendkey}.send with title + desp (markdown)."""
from __future__ import annotations

import logging
from typing import Any

from .base import ChannelNotConfigured, Message, body_with_url, expect_json, http_post, truncate_bytes

logger = logging.getLogger(__name__)

API = "https://sctapi.ftqq.com/{sendkey}.send"
TITLE_MAX_CHARS = 32          # documented limit
DESP_LIMIT_BYTES = 32 * 1024  # documented limit


def build_url(sendkey: str) -> str:
    return API.format(sendkey=sendkey)


def build_payload(msg: Message) -> dict[str, Any]:
    """{"title", "desp"} payload (pure)."""
    title = (msg.title or "audio_note")[:TITLE_MAX_CHARS]
    return {"title": title, "desp": truncate_bytes(body_with_url(msg), DESP_LIMIT_BYTES)}


class ServerChanChannel:
    name = "serverchan"

    def __init__(self, cfg: Any) -> None:
        sub = (cfg.get("push", {}) or {}).get("serverchan", {}) or {}
        self.sendkey: str = (sub.get("sendkey") or "").strip()
        if not self.sendkey:
            raise ChannelNotConfigured("push.serverchan.sendkey 为空")
        self.url = build_url(self.sendkey)

    def send(self, msg: Message) -> None:
        resp = http_post(self.url, json=build_payload(msg), headers={"Content-Type": "application/json"})
        data = expect_json(resp, "serverchan")
        if data.get("code", 0) != 0:
            raise RuntimeError(f"serverchan: code={data.get('code')} message={data.get('message')}")
