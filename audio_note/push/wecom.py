"""WeCom (企业微信) group bot webhook channel: markdown messages, <= 4096 bytes each.

Long bodies are split into several posts (line boundaries preferred); every post must
return ``errcode == 0``.
"""
from __future__ import annotations

import logging
from typing import Any

from .base import ChannelNotConfigured, Message, body_with_url, expect_json, http_post, split_text

logger = logging.getLogger(__name__)

LIMIT_BYTES = 4096
TAG_ROOM_BYTES = 12  # "\n(999/999)"


def build_payloads(msg: Message) -> list[dict[str, Any]]:
    """Markdown payloads for ``msg`` (pure): title as a heading, body split at 4096 bytes."""
    full = f"## {msg.title}\n{body_with_url(msg)}" if msg.title else body_with_url(msg)
    chunks = split_text(full, LIMIT_BYTES)
    if len(chunks) > 1:
        # leave room for a "\n(i/n)" tag on every part, then re-split so no text is lost
        chunks = split_text(full, LIMIT_BYTES - TAG_ROOM_BYTES)
        total = len(chunks)
        chunks = [f"{c.rstrip()}\n({i}/{total})" for i, c in enumerate(chunks, 1)]
    return [{"msgtype": "markdown", "markdown": {"content": c}} for c in chunks]


# alias so every channel exposes the same name
def build_payload(msg: Message) -> dict[str, Any]:
    return build_payloads(msg)[0]


class WecomChannel:
    name = "wecom"

    def __init__(self, cfg: Any) -> None:
        sub = (cfg.get("push", {}) or {}).get("wecom", {}) or {}
        self.webhook: str = (sub.get("webhook") or "").strip()
        if not self.webhook:
            raise ChannelNotConfigured("push.wecom.webhook 为空")

    def send(self, msg: Message) -> None:
        for payload in build_payloads(msg):
            resp = http_post(self.webhook, json=payload, headers={"Content-Type": "application/json"})
            data = expect_json(resp, "wecom")
            if data.get("errcode", 0) != 0:
                raise RuntimeError(f"wecom: errcode={data.get('errcode')} errmsg={data.get('errmsg')}")
