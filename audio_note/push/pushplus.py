"""PushPlus channel: POST https://www.pushplus.plus/send with template=markdown; success code == 200."""
from __future__ import annotations

import logging
from typing import Any

from .base import ChannelNotConfigured, Message, body_with_url, expect_json, http_post, truncate_bytes

logger = logging.getLogger(__name__)

API = "https://www.pushplus.plus/send"
CONTENT_LIMIT_BYTES = 32 * 1024


def build_payload(msg: Message, token: str = "") -> dict[str, Any]:
    """{"token","title","content","template":"markdown"} payload (pure)."""
    return {
        "token": token,
        "title": msg.title or "audio_note",
        "content": truncate_bytes(body_with_url(msg), CONTENT_LIMIT_BYTES),
        "template": "markdown",
    }


class PushPlusChannel:
    name = "pushplus"

    def __init__(self, cfg: Any) -> None:
        sub = (cfg.get("push", {}) or {}).get("pushplus", {}) or {}
        self.token: str = (sub.get("token") or "").strip()
        if not self.token:
            raise ChannelNotConfigured("push.pushplus.token 为空")

    def send(self, msg: Message) -> None:
        resp = http_post(API, json=build_payload(msg, self.token), headers={"Content-Type": "application/json"})
        data = expect_json(resp, "pushplus")
        if data.get("code") != 200:
            raise RuntimeError(f"pushplus: code={data.get('code')} msg={data.get('msg')}")
