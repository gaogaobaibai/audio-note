"""MiMo chat client (OpenAI-compatible). One place for retries, thinking toggle, JSON mode / JSON schema,
audio (ASR + "listening" chat models), streaming, rate limiting and usage accounting.

Verified against https://mimo.mi.com/docs (2026-10) and by probing the live API (2026-11, docs/SPEC_v1.md section 1):
  base_url  https://api.xiaomimimo.com/v1   auth: Bearer (openai SDK) or `api-key` header
  models    mimo-v2.6-flash (cheap, default) / mimo-v2.6-pro ; thinking ON by default ->
            pass extra_body={"thinking": {"type": "disabled"}} for deterministic cheap calls
  json      response_format={"type": "json_object"} (key names may drift) or
            {"type": "json_schema", "json_schema": {"name", "schema", "strict": true}} (enforced by the server)
  asr       mimo-v2.5-asr takes ONE user message holding an input_audio part. A system message, or a plain-text
            user message before the audio, works as a soft hot-word / context prompt; a text part inside the audio
            message is rejected with HTTP 400, and asr_options.hotwords / prompt are silently ignored.
            The model may answer HTTP 200 with finish_reason=content_filter and an English refusal: that is
            raised as ContentFiltered, never returned as a transcript.
  listen    mimo-v2.6-flash / pro hear audio: one user message with input_audio parts followed by a text part
  errors    429 rate limit (backoff), 402 balance, 421 content filter, 503 overloaded

Usage: every call ends in one record {purpose, model, ok, prompt_tokens, completion_tokens, audio_s, wall_s}
handed to `client.on_usage` (metrics.install wires it to the usage_llm table). `client.scoped("refine")`
returns a view of the same client (same SDK object, same rate limiter, same counters) that tags its calls
with that purpose. The rate limiter is shared by every client of the process that talks to the same
(base_url, api_key).
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import random
import re
import threading
import time
from typing import Any, Callable, Iterator

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from ..config import Config, is_placeholder_key

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    #: HTTP status of the failed request when the server answered with one (None: network / parsing / other)
    status_code: int | None = None


class MissingAPIKey(LLMError):
    pass


class ContentFiltered(LLMError):
    """The input or the output was rejected by the content filter (HTTP 421, or HTTP 200 with
    finish_reason=content_filter). Deterministic, never retried; the pipeline skips the affected ASR
    chunk instead of failing the whole segment."""


RETRYABLE_STATUS = {408, 409, 429}           # plus every 5xx
RATE_LIMIT_BASE_DELAY_S = 10.0               # 429: 10 / 20 / 40 / 60 s so three retries span a per-minute window
RATE_LIMIT_MAX_DELAY_S = 60.0
BASE_DELAY_S = 2.0                           # 5xx / network: 2 / 4 / 8 s
MAX_DELAY_S = 60.0
MAX_AUDIO_B64 = 10 * 1024 * 1024             # the API refuses more than 10 MB of base64 audio per request
SCHEMA_REJECT_STATUS = {400, 422}            # a server that does not take response_format=json_schema answers with these
DEFAULT_PURPOSE = "other"


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


_LIMITERS: dict[tuple[str, str], RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def shared_limiter(base_url: str, api_key: str, rpm: int) -> RateLimiter:
    """The process-wide rate limiter of one account: every MimoClient built for the same (base_url, api_key)
    gets the same object, so the runtime's client, the scheduler's and a CLI helper's stay under ONE
    requests-per-minute cap together. The newest `rpm` wins (a config reload changes the cap in place).
    The key itself is not kept: the registry is indexed by its hash."""
    key = (str(base_url or "").rstrip("/"), hashlib.sha256(str(api_key or "").encode("utf-8")).hexdigest())
    with _LIMITERS_LOCK:
        lim = _LIMITERS.get(key)
        if lim is None:
            lim = _LIMITERS[key] = RateLimiter(rpm)
        else:
            lim.interval = 60.0 / max(1, int(rpm))
        return lim


def _audio_mime(data: bytes) -> str:
    """wav unless the bytes clearly are an mp3 (the two formats the API accepts)."""
    head = bytes(data[:4])
    if head[:3] == b"ID3" or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return "audio/mpeg"
    return "audio/wav"


def _audio_part(data: bytes, mime: str | None = None) -> tuple[dict[str, Any], int]:
    b64 = base64.b64encode(data).decode()
    return {"type": "input_audio", "input_audio": {"data": f"data:{mime or _audio_mime(data)};base64,{b64}"}}, len(b64)


class MimoClient:
    def __init__(self, cfg: Config, rpm: int = 90):
        m = cfg.mimo
        self._root = self            # a scoped() view points at the client it was made from
        self.purpose: str | None = None
        self.cfg = cfg
        if not m.api_key or is_placeholder_key(m.api_key):
            raise MissingAPIKey("mimo.api_key is empty: fill config.yaml or set MIMO_API_KEY")
        self.client = OpenAI(api_key=m.api_key, base_url=m.base_url, timeout=float(m.timeout_s), max_retries=0)
        self.chat_model = m.chat_model
        self.max_retries = int(m.max_retries)
        self.temperature = float(m.temperature)
        self.limiter = shared_limiter(m.base_url, m.api_key, rpm)
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "audio_seconds": 0.0}
        self._ulock = threading.Lock()
        #: called after every request with {purpose, model, ok, prompt_tokens, completion_tokens, audio_s, wall_s}
        self.on_usage: Callable[[dict], None] | None = None
        self._schema_rejected: set[tuple[str, str]] = set()     # (model, schema name) the server refused
        self._stream_usage_ok = True                             # stream_options.include_usage accepted so far

    # --- purpose-tagged views ------------------------------------------------------
    def scoped(self, purpose: str) -> "MimoClient":
        """A view of this client whose calls are accounted under `purpose` (asr / refine / daily / chat ...).
        It shares everything with the client it came from (SDK object, rate limiter, usage counters,
        on_usage): reading any other attribute goes to that client, so later changes are seen by both."""
        view = object.__new__(type(self))
        object.__setattr__(view, "_root", self.__dict__.get("_root") or self)
        object.__setattr__(view, "purpose", str(purpose) if purpose else None)
        return view

    def __getattr__(self, name: str) -> Any:
        # only reached when normal lookup fails: a scoped view resolves everything but `purpose` on its root
        root = self.__dict__.get("_root")
        if root is None or root is self or name.startswith("__"):
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(root, name)

    def __setattr__(self, name: str, value: Any) -> None:
        root = self.__dict__.get("_root")
        if root is not None and root is not self and name != "purpose":
            setattr(root, name, value)
        else:
            object.__setattr__(self, name, value)

    def _purpose(self, purpose: str | None, default: str = DEFAULT_PURPOSE) -> str:
        return str(purpose or self.__dict__.get("purpose") or default)

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
                    ex: LLMError = ContentFiltered(f"{what}: content filtered (421)")
                    ex.status_code = code
                    raise ex from e
                if not _is_retryable_status(code):
                    ex = LLMError(f"{what}: HTTP {code} {getattr(e, 'message', e)}")
                    ex.status_code = code
                    raise ex from e
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

    def _account(self, resp: Any, purpose: str | None = None, model: str | None = None, wall_s: float = 0.0) -> None:
        """Count one successful request (process counters + on_usage record)."""
        u = getattr(resp, "usage", None)
        pt = ct = 0
        secs = 0.0
        if u:
            pt = int(getattr(u, "prompt_tokens", 0) or 0)
            ct = int(getattr(u, "completion_tokens", 0) or 0)
            secs = float(getattr(u, "seconds", 0) or 0)
        with self._ulock:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += pt
            self.usage["completion_tokens"] += ct
            if secs:
                self.usage["audio_seconds"] += secs
        self._report(purpose, model, True, pt, ct, secs, wall_s)

    def _report(self, purpose: str | None, model: str | None, ok: bool, prompt_tokens: int = 0, completion_tokens: int = 0,
                audio_s: float = 0.0, wall_s: float = 0.0) -> None:
        cb = self.on_usage
        if cb is None:
            return
        try:
            cb({"purpose": self._purpose(purpose), "model": str(model or self.chat_model), "ok": bool(ok),
                "prompt_tokens": int(prompt_tokens), "completion_tokens": int(completion_tokens),
                "audio_s": float(audio_s), "wall_s": float(wall_s)})
        except Exception as e:  # noqa: BLE001 - accounting never breaks a request
            log.debug("on_usage callback failed: %s", e)

    def _request(self, kwargs: dict[str, Any], what: str, purpose: str | None) -> Any:
        """One chat.completions call with retries; a failure is reported to on_usage (ok=False) and re-raised."""
        t0 = time.perf_counter()
        try:
            resp = self._with_retry(lambda: self.client.chat.completions.create(**kwargs), what)
        except LLMError:
            self._report(purpose, kwargs.get("model"), False, wall_s=time.perf_counter() - t0)
            raise
        self._account(resp, purpose, kwargs.get("model"), time.perf_counter() - t0)
        return resp

    # --- chat ---------------------------------------------------------------
    def _chat_kwargs(self, messages: list[dict[str, Any]], model: str | None, temperature: float | None, max_tokens: int,
                     thinking: bool) -> dict[str, Any]:
        # `max_tokens` is the field every OpenAI-compatible server (MiMo included) honours;
        # `max_completion_tokens` is OpenAI-only and would be silently ignored or rejected elsewhere.
        kwargs: dict[str, Any] = dict(
            model=model or self.chat_model, messages=messages, max_tokens=max_tokens,
            extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        )
        if not thinking:
            kwargs["temperature"] = self.temperature if temperature is None else temperature
        return kwargs

    def chat(self, messages: list[dict[str, Any]], *, model: str | None = None, temperature: float | None = None,
             max_tokens: int = 4096, json_mode: bool = False, json_schema: dict[str, Any] | None = None,
             thinking: bool = False, tools: list[dict] | None = None, tool_choice: Any = None,
             purpose: str | None = None) -> Any:
        """Returns the raw ChatCompletion. Use .choices[0].message.

        json_schema = {"name": str, "schema": dict}: the server enforces the schema (strict). A server that
        rejects the schema with HTTP 400 / 422 gets the same request once more in plain JSON mode, and that
        (model, schema name) pair goes straight to JSON mode from then on."""
        kwargs = self._chat_kwargs(messages, model, temperature, max_tokens, thinking)
        schema_key: tuple[str, str] | None = None
        if json_schema is not None:
            name = str(json_schema.get("name") or "result")
            schema_key = (str(kwargs["model"]), name)
            if schema_key in self._schema_rejected:
                kwargs["response_format"] = {"type": "json_object"}
                schema_key = None
            else:
                kwargs["response_format"] = {"type": "json_schema",
                                             "json_schema": {"name": name, "schema": json_schema.get("schema") or {}, "strict": True}}
        elif json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        try:
            return self._request(kwargs, "chat", purpose)
        except LLMError as e:
            if schema_key is None or isinstance(e, ContentFiltered) or e.status_code not in SCHEMA_REJECT_STATUS:
                raise
            log.warning("model %s rejected json_schema %r (%s); falling back to json_object", schema_key[0], schema_key[1], e)
            self._schema_rejected.add(schema_key)
            kwargs["response_format"] = {"type": "json_object"}
            return self._request(kwargs, "chat", purpose)

    def text(self, system: str, user: str, **kw: Any) -> str:
        resp = self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
        return (resp.choices[0].message.content or "").strip()

    def json(self, system: str, user: str, *, schema: dict[str, Any] | None = None, schema_name: str = "result",
             **kw: Any) -> Any:
        """JSON call; tolerant parse (strips code fences, finds first {...} or [...]).
        With `schema` (a JSON Schema dict) the server enforces the shape (response_format=json_schema, with the
        fallback described in chat()); without it plain JSON mode is used."""
        if schema is not None:
            txt = self.text(system, user, json_schema={"name": schema_name, "schema": schema}, **kw)
        else:
            txt = self.text(system, user, json_mode=True, **kw)
        return parse_json_loose(txt)

    def chat_stream(self, messages: list[dict[str, Any]], *, model: str | None = None, temperature: float | None = None,
                    max_tokens: int = 4096, thinking: bool = False, purpose: str | None = None) -> Iterator[str]:
        """Yield the answer's text pieces as they arrive (no tools, no JSON mode). The retry policy covers
        opening the stream; an error in the middle of it raises LLMError. Usage is accounted when the
        stream ends (also when the consumer stops early)."""
        kwargs = self._chat_kwargs(messages, model, temperature, max_tokens, thinking)
        kwargs["stream"] = True
        purpose = self._purpose(purpose, "chat")
        t0 = time.perf_counter()

        def open_stream() -> Any:
            if self._stream_usage_ok:
                try:
                    return self._with_retry(lambda: self.client.chat.completions.create(
                        **kwargs, stream_options={"include_usage": True}), "chat_stream")
                except LLMError as e:
                    if isinstance(e, ContentFiltered) or e.status_code not in SCHEMA_REJECT_STATUS:
                        raise
                    log.info("server rejected stream_options.include_usage (%s); streaming without usage", e)
                    self._stream_usage_ok = False
            return self._with_retry(lambda: self.client.chat.completions.create(**kwargs), "chat_stream")

        try:
            stream = open_stream()
        except LLMError:
            self._report(purpose, kwargs["model"], False, wall_s=time.perf_counter() - t0)
            raise
        return self._iter_stream(stream, purpose, kwargs["model"], t0)

    def _iter_stream(self, stream: Any, purpose: str, model: str, t0: float) -> Iterator[str]:
        usage: Any = None
        ok = False
        try:
            for chunk in stream:
                u = getattr(chunk, "usage", None)
                if u:
                    usage = u
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                piece = getattr(delta, "content", None) if delta is not None else None
                if piece:
                    yield piece
            ok = True
        except (APIConnectionError, APITimeoutError, APIStatusError) as e:
            raise LLMError(f"chat_stream: interrupted: {e}") from e
        except GeneratorExit:
            ok = True              # the consumer stopped reading: not a failure
            raise
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass
            wall = time.perf_counter() - t0
            if ok:
                self._account(type("StreamEnd", (), {"usage": usage})(), purpose, model, wall)
            else:
                self._report(purpose, model, False, wall_s=wall)

    # --- audio (ASR) ---------------------------------------------------------
    def transcribe_bytes(self, wav_or_mp3: bytes, mime: str = "audio/wav", language: str = "auto",
                         model: str | None = None, *, prompt: str | None = None, context: str | None = None) -> tuple[str, float]:
        """mimo-v2.5-asr via chat.completions. Returns (text, billed_seconds).
        wav/mp3 only; base64 must be <= 10 MB.

        prompt  -> a system message (soft hot words, e.g. "可能出现的专有名词：甲、乙")
        context -> a plain-text user message placed before the audio
        With both None the request is exactly the one this method always sent. Beware: on audio without
        speech the model may echo the prompt back as the "transcript" (the caller's guard handles that).
        A reply with finish_reason=content_filter raises ContentFiltered."""
        part, n_b64 = _audio_part(wav_or_mp3, mime)
        if n_b64 > MAX_AUDIO_B64:
            raise LLMError(f"audio too large for ASR: {n_b64} base64 bytes > 10 MB")
        messages: list[dict[str, Any]] = []
        if prompt:
            messages.append({"role": "system", "content": str(prompt)})
        if context:
            messages.append({"role": "user", "content": str(context)})
        messages.append({"role": "user", "content": [part]})
        kwargs: dict[str, Any] = dict(
            model=model or self.cfg.mimo.asr_model,
            messages=messages,
            extra_body={"asr_options": {"language": language}},
        )
        resp = self._request(kwargs, "asr", self._purpose(None, "asr"))
        choice = resp.choices[0]
        if choice.finish_reason == "content_filter":
            raise ContentFiltered("asr: content filtered (finish_reason=content_filter)")
        text = (choice.message.content or "").strip()
        if choice.finish_reason == "length":
            log.warning("ASR output truncated (finish_reason=length); shorten chunks")
        secs = float(getattr(resp.usage, "seconds", 0) or 0) if getattr(resp, "usage", None) else 0.0
        return text, secs

    def listen(self, clips: list[bytes], text: str, *, model: str | None = None, json_schema: dict[str, Any] | None = None,
               thinking: bool = False, max_tokens: int = 600, purpose: str = "listen") -> str:
        """Let a chat model that hears audio (mimo.listen_model) answer `text` about the clips: ONE user
        message made of the clips (wav / mp3 bytes, in order) followed by the text. Returns the reply text.
        Never show it an existing transcript or candidates (it anchors on them): ask it to transcribe blind."""
        content: list[dict[str, Any]] = []
        total = 0
        for clip in clips:
            part, n_b64 = _audio_part(clip)
            total += n_b64
            content.append(part)
        if total > MAX_AUDIO_B64:
            raise LLMError(f"audio too large to listen to: {total} base64 bytes > 10 MB")
        content.append({"type": "text", "text": str(text)})
        listen_model = model or self.cfg.mimo.get("listen_model") or self.chat_model
        resp = self.chat([{"role": "user", "content": content}], model=listen_model, max_tokens=max_tokens,
                         json_schema=json_schema, thinking=thinking, purpose=purpose)
        choice = resp.choices[0]
        if choice.finish_reason == "content_filter":
            raise ContentFiltered("listen: content filtered (finish_reason=content_filter)")
        return (choice.message.content or "").strip()


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
