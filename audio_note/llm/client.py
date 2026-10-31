"""MiMo chat client (OpenAI-compatible). One place for retries, thinking toggle, JSON mode, token accounting.

Verified against https://mimo.mi.com/docs (2026-10):
  base_url  https://api.xiaomimimo.com/v1   auth: Bearer (openai SDK) or `api-key` header
  models    mimo-v2.6-flash (cheap, default) / mimo-v2.6-pro ; thinking ON by default ->
            pass extra_body={"thinking": {"type": "disabled"}} for deterministic cheap calls
  json mode response_format={"type": "json_object"}
  errors    429 rate limit (backoff), 402 balance, 421 content filter, 503 overloaded
"""
from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from typing import Any, Callable

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from ..config import Config, is_placeholder_key

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class MissingAPIKey(LLMError):
    pass


class ContentFiltered(LLMError):
    """HTTP 421: the input was rejected by the content filter. Deterministic, never retried;
    the pipeline skips the affected ASR chunk instead of failing the whole segment."""


RETRYABLE_STATUS = {408, 409, 429}           # plus every 5xx
RATE_LIMIT_BASE_DELAY_S = 10.0               # 429: 10 / 20 / 40 / 60 s so three retries span a per-minute window
RATE_LIMIT_MAX_DELAY_S = 60.0
BASE_DELAY_S = 2.0                           # 5xx / network: 2 / 4 / 8 s
MAX_DELAY_S = 60.0


def _retry_after_seconds(err: Exception) -> float | None:
    """Seconds requested by a Retry-After header (integer seconds or HTTP-date), or None."""
    resp = getattr(err, "response", None)
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    try:
        raw = headers.get("retry-after")
    except Exception:  # noqa: BLE001
        return None
    if not raw:
        return None
    raw = str(raw).strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(raw)
        return max(0.0, dt.timestamp() - time.time())
    except Exception:  # noqa: BLE001
        return None


def _is_retryable_status(code: int) -> bool:
    return code in RETRYABLE_STATUS or code >= 500


class RateLimiter:
    """Simple token-bucket on requests/minute shared by all threads of a process."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / max(1, rpm)
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
                now = time.monotonic()
            self._next = now + self.interval


class MimoClient:
    def __init__(self, cfg: Config, rpm: int = 90):
        m = cfg.mimo
        self.cfg = cfg
        if not m.api_key or is_placeholder_key(m.api_key):
            raise MissingAPIKey("mimo.api_key is empty: fill config.yaml or set MIMO_API_KEY")
        self.client = OpenAI(api_key=m.api_key, base_url=m.base_url, timeout=float(m.timeout_s), max_retries=0)
        self.chat_model = m.chat_model
        self.max_retries = int(m.max_retries)
        self.temperature = float(m.temperature)
        self.limiter = RateLimiter(rpm)
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "audio_seconds": 0.0}
        self._ulock = threading.Lock()

    # --- low level ---------------------------------------------------------
    def _with_retry(self, fn: Callable[[], Any], what: str) -> Any:
        """Retry policy: 429 -> 10/20/40/60 s backoff (Retry-After honoured when larger);
        408/409/5xx/network -> 2/4/8 s; every other 4xx (400/413/415/422, 401/402/403/404) is
        deterministic and raised at once; 421 raises ContentFiltered."""
        delay = BASE_DELAY_S
        rl_delay = RATE_LIMIT_BASE_DELAY_S
        for attempt in range(self.max_retries + 1):
            rate_limited = False
            try:
                self.limiter.wait()
                return fn()
            except RateLimitError as e:
                err: Exception = e
                rate_limited = True
            except APIStatusError as e:
                code = int(getattr(e, "status_code", 0) or 0)
                if code == 421:
                    raise ContentFiltered(f"{what}: content filtered (421)") from e
                if not _is_retryable_status(code):
                    raise LLMError(f"{what}: HTTP {code} {getattr(e, 'message', e)}") from e
                err = e
                rate_limited = code == 429
            except (APIConnectionError, APITimeoutError) as e:
                err = e
            if attempt >= self.max_retries:
                raise LLMError(f"{what}: giving up after {attempt + 1} attempts: {err}") from err
            if rate_limited:
                sleep = rl_delay * (1 + random.random() * 0.5)
                rl_delay = min(rl_delay * 2, RATE_LIMIT_MAX_DELAY_S)
            else:
                sleep = delay * (1 + random.random() * 0.5)
                delay = min(delay * 2, MAX_DELAY_S)
            hinted = _retry_after_seconds(err)
            if hinted is not None and hinted > sleep:
                sleep = min(hinted + 0.5, 300.0)
            log.warning("%s failed (%s); retry in %.1fs", what, err, sleep)
            time.sleep(sleep)
        raise LLMError(what)  # unreachable

    def _account(self, resp: Any) -> None:
        u = getattr(resp, "usage", None)
        with self._ulock:
            self.usage["calls"] += 1
            if u:
                self.usage["prompt_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
                self.usage["completion_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)
                secs = getattr(u, "seconds", None)
                if secs:
                    self.usage["audio_seconds"] += float(secs)

    # --- chat ---------------------------------------------------------------
    def chat(self, messages: list[dict[str, Any]], *, model: str | None = None, temperature: float | None = None,
             max_tokens: int = 4096, json_mode: bool = False, thinking: bool = False,
             tools: list[dict] | None = None, tool_choice: Any = None) -> Any:
        """Returns the raw ChatCompletion. Use .choices[0].message."""
        # `max_tokens` is the field every OpenAI-compatible server (MiMo included) honours;
        # `max_completion_tokens` is OpenAI-only and would be silently ignored or rejected elsewhere.
        kwargs: dict[str, Any] = dict(
            model=model or self.chat_model, messages=messages, max_tokens=max_tokens,
            extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        )
        if not thinking:
            kwargs["temperature"] = self.temperature if temperature is None else temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        resp = self._with_retry(lambda: self.client.chat.completions.create(**kwargs), "chat")
        self._account(resp)
        return resp

    def text(self, system: str, user: str, **kw: Any) -> str:
        resp = self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
        return (resp.choices[0].message.content or "").strip()

    def json(self, system: str, user: str, **kw: Any) -> Any:
        """JSON-mode call; tolerant parse (strips code fences, finds first {...} or [...])."""
        txt = self.text(system, user, json_mode=True, **kw)
        return parse_json_loose(txt)

    # --- audio (ASR) ---------------------------------------------------------
    def transcribe_bytes(self, wav_or_mp3: bytes, mime: str = "audio/wav", language: str = "auto",
                         model: str | None = None) -> tuple[str, float]:
        """mimo-v2.5-asr via chat.completions. Returns (text, billed_seconds).
        wav/mp3 only; base64 must be <= 10 MB."""
        import base64

        b64 = base64.b64encode(wav_or_mp3).decode()
        if len(b64) > 10 * 1024 * 1024:
            raise LLMError(f"audio too large for ASR: {len(b64)} base64 bytes > 10 MB")
        kwargs: dict[str, Any] = dict(
            model=model or self.cfg.mimo.asr_model,
            messages=[{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": f"data:{mime};base64,{b64}"}}]}],
            extra_body={"asr_options": {"language": language}},
        )
        resp = self._with_retry(lambda: self.client.chat.completions.create(**kwargs), "asr")
        self._account(resp)
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        if choice.finish_reason == "length":
            log.warning("ASR output truncated (finish_reason=length); shorten chunks")
        secs = float(getattr(resp.usage, "seconds", 0) or 0) if getattr(resp, "usage", None) else 0.0
        return text, secs


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def parse_json_loose(txt: str) -> Any:
    s = _FENCE.sub("", txt.strip())
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    for open_, close in (("{", "}"), ("[", "]")):
        i, j = s.find(open_), s.rfind(close)
        if i != -1 and j > i:
            try:
                return json.loads(s[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"model did not return JSON: {txt[:200]!r}")
