"""Push notification core: Message, Channel protocol, Notifier and shared helpers.

Usage:
    from audio_note.push import Message, Notifier
    Notifier(cfg).send(Message("每日记录 2026-10-31", "# 概览\\n..."))  # -> {"feishu": None, "toast": "error..."}

Channels are built from ``cfg.push.channels`` (in order). Unknown names and channels whose
required config is empty are skipped with a warning. ``Notifier.send`` never raises: every
channel error is caught, logged and returned in the result dict.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

import httpx

logger = logging.getLogger(__name__)

TIMEOUT_S = 10.0

#: channel name -> "module:ClassName" (imported lazily to avoid circular imports)
CHANNEL_REGISTRY: dict[str, str] = {
    "feishu": "audio_note.push.feishu:FeishuChannel",
    "wecom": "audio_note.push.wecom:WecomChannel",
    "serverchan": "audio_note.push.serverchan:ServerChanChannel",
    "pushplus": "audio_note.push.pushplus:PushPlusChannel",
    "wxpusher": "audio_note.push.wxpusher:WxPusherChannel",
    "ntfy": "audio_note.push.ntfy:NtfyChannel",
    "toast": "audio_note.push.toast:ToastChannel",
}


@dataclass
class Message:
    """A notification: ``title`` plus Markdown ``body_md`` and an optional link."""

    title: str
    body_md: str
    url: str | None = None


@runtime_checkable
class Channel(Protocol):
    """A push channel. ``send`` raises (RuntimeError) on failure."""

    name: str

    def send(self, msg: Message) -> None: ...


class ChannelNotConfigured(ValueError):
    """Raised by a channel constructor when its required config is empty."""


# ----------------------------------------------------------------------------- helpers

def body_with_url(msg: Message) -> str:
    """Body Markdown with the optional link appended as a final line."""
    body = msg.body_md or ""
    if msg.url:
        body = f"{body.rstrip()}\n\n{msg.url}" if body.strip() else msg.url
    return body


def utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _split_long_line(line: str, limit_bytes: int) -> list[str]:
    """Split one string into pieces whose UTF-8 length is <= limit_bytes (never cuts a code point)."""
    out: list[str] = []
    buf: list[str] = []
    size = 0
    for ch in line:
        n = len(ch.encode("utf-8"))
        if size + n > limit_bytes and buf:
            out.append("".join(buf))
            buf, size = [], 0
        buf.append(ch)
        size += n
    if buf:
        out.append("".join(buf))
    return out


def split_text(text: str, limit_bytes: int) -> list[str]:
    """Split ``text`` into chunks of at most ``limit_bytes`` UTF-8 bytes, preferring line boundaries.

    Always returns at least one element (``[""]`` for empty input). Lines longer than the
    limit are cut at character boundaries so multi-byte characters are never broken.
    """
    if limit_bytes <= 0:
        raise ValueError("limit_bytes must be positive")
    if utf8_len(text) <= limit_bytes:
        return [text]
    chunks: list[str] = []
    cur = ""
    for line in text.splitlines(keepends=True):
        pieces = [line] if utf8_len(line) <= limit_bytes else _split_long_line(line, limit_bytes)
        for piece in pieces:
            if cur and utf8_len(cur) + utf8_len(piece) > limit_bytes:
                chunks.append(cur)
                cur = ""
            cur += piece
    if cur:
        chunks.append(cur)
    return chunks or [""]


def truncate_bytes(text: str, limit_bytes: int, marker: str = "\n…(已截断)") -> str:
    """Return ``text`` if it fits in ``limit_bytes`` UTF-8 bytes, else a cut prefix plus ``marker``."""
    if utf8_len(text) <= limit_bytes:
        return text
    room = max(limit_bytes - utf8_len(marker), 0)
    head = _split_long_line(text, room)[0] if room > 0 else ""
    return head + marker


_MD_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"```.*?```", re.S), ""),                    # fenced code blocks
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),          # images -> alt text
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),           # links -> text
    (re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*", re.M), ""),              # headings
    (re.compile(r"^[ \t]{0,3}>[ \t]?", re.M), ""),                   # blockquotes
    (re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+", re.M), ""),       # list markers
    (re.compile(r"^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$", re.M), ""),  # horizontal rules
    (re.compile(r"(\*\*|__)(.+?)\1", re.S), r"\2"),          # bold
    (re.compile(r"(\*|_)(?=\S)(.+?)(?<=\S)\1", re.S), r"\2"),  # italic
    (re.compile(r"~~(.+?)~~", re.S), r"\1"),                 # strikethrough
    (re.compile(r"`([^`]*)`"), r"\1"),                       # inline code
    (re.compile(r"<[^>\n]+>"), ""),                          # inline html tags
]


def markdown_to_text(md: str) -> str:
    """Best-effort conversion of Markdown to plain text (for channels without Markdown support)."""
    text = md or ""
    for pat, repl in _MD_PATTERNS:
        text = pat.sub(repl, text)
    lines = [ln.rstrip() for ln in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def http_post(
    url: str,
    *,
    json: Any = None,
    content: bytes | str | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = TIMEOUT_S,
) -> httpx.Response:
    """POST via ``httpx.post`` (looked up at call time so tests can monkeypatch it)."""
    return httpx.post(url, json=json, content=content, headers=headers, timeout=timeout)


def expect_json(resp: httpx.Response, where: str) -> dict[str, Any]:
    """Raise RuntimeError unless ``resp`` is a 2xx JSON object; return the parsed body."""
    status = int(getattr(resp, "status_code", 0) or 0)
    if not 200 <= status < 300:
        snippet = (getattr(resp, "text", "") or "")[:200]
        raise RuntimeError(f"{where}: HTTP {status} {snippet}")
    try:
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"{where}: 响应不是 JSON: {e}") from e
    if not isinstance(data, dict):
        raise RuntimeError(f"{where}: 响应格式异常: {str(data)[:200]}")
    return data


# ----------------------------------------------------------------------------- notifier

def _load_channel_class(spec: str) -> type:
    import importlib

    mod_name, cls_name = spec.split(":", 1)
    return getattr(importlib.import_module(mod_name), cls_name)


def build_channel(name: str, cfg: Any) -> Channel | None:
    """Instantiate one channel by name; returns None (after a warning) if unknown or unconfigured."""
    spec = CHANNEL_REGISTRY.get(name)
    if spec is None:
        logger.warning("未知推送渠道 %r，已跳过（可用: %s）", name, ", ".join(CHANNEL_REGISTRY))
        return None
    try:
        cls = _load_channel_class(spec)
        return cls(cfg)
    except ChannelNotConfigured as e:
        logger.warning("推送渠道 %s 未配置，已跳过: %s", name, e)
    except Exception as e:  # noqa: BLE001 - never let one broken channel block the rest
        logger.warning("推送渠道 %s 初始化失败，已跳过: %s", name, e)
    return None


def get_channels(cfg: Any) -> list[Channel]:
    """Build the configured channels from ``cfg.push.channels`` in order (duplicates removed)."""
    push_cfg = (cfg or {}).get("push", {}) if isinstance(cfg, dict) else {}
    names = push_cfg.get("channels") or []
    if isinstance(names, str):
        names = [names]
    channels: list[Channel] = []
    seen: set[str] = set()
    for raw in names:
        name = str(raw).strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        ch = build_channel(name, cfg)
        if ch is not None:
            channels.append(ch)
    if not channels:
        logger.warning("没有可用的推送渠道（push.channels=%r）", names)
    return channels


class Notifier:
    """Fan a Message out to every configured channel. Never raises."""

    def __init__(self, cfg: Any, channels: list[Channel] | None = None) -> None:
        self.cfg = cfg
        self.channels: list[Channel] = channels if channels is not None else get_channels(cfg)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.channels]

    def send(self, msg: Message) -> dict[str, str | None]:
        """Send to all channels; returns {channel_name: None (ok) | error string}."""
        results: dict[str, str | None] = {}
        if not self.channels:
            logger.warning("推送未发送：没有配置渠道（标题: %s）", msg.title)
            return results
        for ch in self.channels:
            try:
                ch.send(msg)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}" if not isinstance(e, RuntimeError) else str(e)
                logger.error("推送失败 [%s] %s: %s", ch.name, msg.title, err)
                results[ch.name] = err
            else:
                logger.info("推送成功 [%s] %s", ch.name, msg.title)
                results[ch.name] = None
        return results

    def send_text(self, title: str, body_md: str, url: str | None = None) -> dict[str, str | None]:
        return self.send(Message(title=title, body_md=body_md, url=url))


SendFn = Callable[[Message], None]
