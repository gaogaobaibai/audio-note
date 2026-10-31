"""WxPusher channel (微信公众号推送).

Two modes: ``spt`` (极简推送, simple-push endpoint) or ``app_token`` + ``uids`` (standard endpoint).
``contentType = 3`` means Markdown. Success is ``code == 1000``.
"""
from __future__ import annotations

import logging
from typing import Any

from .base import ChannelNotConfigured, Message, body_with_url, expect_json, http_post, truncate_bytes

logger = logging.getLogger(__name__)

SIMPLE_API = "https://wxpusher.zjiecode.com/api/send/message/simple-push"
STANDARD_API = "https://wxpusher.zjiecode.com/api/send/message"
CONTENT_LIMIT_BYTES = 40 * 1024
SUMMARY_MAX_CHARS = 20  # documented limit for the summary shown in the WeChat list


def build_payload(
    msg: Message,
    *,
    spt: str = "",
    app_token: str = "",
    uids: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return ``(url, json)`` for ``msg`` (pure). ``spt`` takes precedence over app_token/uids."""
    content = truncate_bytes(body_with_url(msg), CONTENT_LIMIT_BYTES)
    summary = (msg.title or "audio_note")[:SUMMARY_MAX_CHARS]
    base: dict[str, Any] = {"content": content, "summary": summary, "contentType": 3}
    if msg.url:
        base["url"] = msg.url
    if spt:
        return SIMPLE_API, {"spt": spt, **base}
    return STANDARD_API, {"appToken": app_token, **base, "uids": list(uids or [])}


class WxPusherChannel:
    name = "wxpusher"

    def __init__(self, cfg: Any) -> None:
        sub = (cfg.get("push", {}) or {}).get("wxpusher", {}) or {}
        self.spt: str = (sub.get("spt") or "").strip()
        self.app_token: str = (sub.get("app_token") or "").strip()
        uids = sub.get("uids") or []
        self.uids: list[str] = [str(u) for u in ([uids] if isinstance(uids, str) else uids) if str(u).strip()]
        if not self.spt and not (self.app_token and self.uids):
            raise ChannelNotConfigured("push.wxpusher 需要 spt，或 app_token + uids")

    def send(self, msg: Message) -> None:
        url, payload = build_payload(msg, spt=self.spt, app_token=self.app_token, uids=self.uids)
        resp = http_post(url, json=payload, headers={"Content-Type": "application/json"})
        data = expect_json(resp, "wxpusher")
        if data.get("code") != 1000:
            raise RuntimeError(f"wxpusher: code={data.get('code')} msg={data.get('msg')}")
