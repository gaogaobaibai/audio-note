"""Feishu (飞书) custom bot webhook channel.

Sends an interactive card (schema 2.0, markdown element). If the card is rejected by the
webhook, falls back to a plain ``text`` message. Optional signature: when
``cfg.push.feishu.secret`` is set, ``timestamp`` + ``sign`` are added where
``sign = base64(HMAC-SHA256(key=f"{ts}\\n{secret}", msg=b""))``.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import time
from typing import Any

from .base import ChannelNotConfigured, Message, body_with_url, expect_json, http_post, markdown_to_text, truncate_bytes

logger = logging.getLogger(__name__)

BODY_LIMIT_BYTES = 20 * 1024
TEXT_LIMIT_BYTES = 20 * 1024


def sign(timestamp: int | str, secret: str) -> str:
    """Feishu custom-bot signature: base64(HMAC-SHA256(key=ts + "\\n" + secret, msg=b""))."""
    key = f"{timestamp}\n{secret}".encode("utf-8")
    digest = hmac.new(key, msg=b"", digestmod=hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


def _with_sign(payload: dict[str, Any], secret: str | None, timestamp: int | None) -> dict[str, Any]:
    if secret:
        ts = int(timestamp if timestamp is not None else time.time())
        payload = {"timestamp": str(ts), "sign": sign(ts, secret), **payload}
    return payload


def build_payload(msg: Message, secret: str | None = None, timestamp: int | None = None) -> dict[str, Any]:
    """Interactive-card payload for ``msg`` (pure)."""
    body = truncate_bytes(body_with_url(msg), BODY_LIMIT_BYTES)
    card = {
        "schema": "2.0",
        "header": {"title": {"tag": "plain_text", "content": msg.title or "audio_note"}},
        "body": {"elements": [{"tag": "markdown", "content": body}]},
    }
    return _with_sign({"msg_type": "interactive", "card": card}, secret, timestamp)


def build_text_payload(msg: Message, secret: str | None = None, timestamp: int | None = None) -> dict[str, Any]:
    """Plain ``text`` payload used as a fallback when the card is rejected (pure)."""
    plain = markdown_to_text(body_with_url(msg))
    text = f"{msg.title}\n{plain}" if msg.title else plain
    text = truncate_bytes(text, TEXT_LIMIT_BYTES)
    return _with_sign({"msg_type": "text", "content": {"text": text}}, secret, timestamp)


class FeishuChannel:
    name = "feishu"

    def __init__(self, cfg: Any) -> None:
        sub = (cfg.get("push", {}) or {}).get("feishu", {}) or {}
        self.webhook: str = (sub.get("webhook") or "").strip()
        self.secret: str = (sub.get("secret") or "").strip()
        if not self.webhook:
            raise ChannelNotConfigured("push.feishu.webhook 为空")

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = http_post(self.webhook, json=payload, headers={"Content-Type": "application/json"})
        status = int(getattr(resp, "status_code", 0) or 0)
        if 400 <= status < 500:
            # the webhook answers some card validation errors with HTTP 400 + {"code": 9499/11246, ...};
            # surface that as a rejection so send() can fall back to plain text
            try:
                data = resp.json()
            except Exception:  # noqa: BLE001
                data = None
            if isinstance(data, dict) and data.get("code", data.get("StatusCode")) not in (None, 0):
                return data
        return expect_json(resp, "feishu")

    def send(self, msg: Message) -> None:
        data = self._post(build_payload(msg, self.secret or None))
        if data.get("code", data.get("StatusCode", 0)) == 0:
            return
        logger.warning("飞书卡片被拒绝 (code=%s msg=%s)，改用纯文本重发", data.get("code"), data.get("msg"))
        data = self._post(build_text_payload(msg, self.secret or None))
        if data.get("code", data.get("StatusCode", 0)) != 0:
            raise RuntimeError(f"feishu: code={data.get('code')} msg={data.get('msg')}")
