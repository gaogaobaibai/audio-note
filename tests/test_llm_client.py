"""Tests for audio_note.llm.client: parse_json_loose, RateLimiter, MissingAPIKey, retry logic, request shaping.

The OpenAI SDK object inside MimoClient is replaced by a fake; nothing goes on the wire.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import httpx
import openai
import pytest

from audio_note.llm import client as client_mod
from audio_note.llm.client import LLMError, MimoClient, MissingAPIKey, RateLimiter, parse_json_loose


# ---------------------------------------------------------------------------
# parse_json_loose
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("txt,expected", [
    ('{"a": 1}', {"a": 1}),
    ('  \n{"a": [1, 2]}\n', {"a": [1, 2]}),
    ('```json\n{"edits": []}\n```', {"edits": []}),
    ('```\n{"edits": []}\n```', {"edits": []}),
    ('好的，结果如下：{"a": "中文"} 以上。', {"a": "中文"}),
    ('前言 [1, 2, 3] 后记', [1, 2, 3]),
    ('[{"i": 0}]', [{"i": 0}]),
    ('噪音 {"outer": {"inner": [1]}} 噪音', {"outer": {"inner": [1]}}),
    ('{"a": 1}\n\n补充说明：无。', {"a": 1}),
    ('"just a string"', "just a string"),
    ("42", 42),
])
def test_parse_json_loose_variants(txt, expected):
    assert parse_json_loose(txt) == expected


@pytest.mark.parametrize("txt", ["", "no json here", "{broken", "```json\n{\n```", "[1, 2"])
def test_parse_json_loose_raises(txt):
    with pytest.raises(LLMError):
        parse_json_loose(txt)


def test_parse_json_loose_stray_trailing_brace_falls_back_to_list():
    # Documented quirk: the first-'{' .. last-'}' heuristic fails on a stray trailing brace and
    # the list fallback then returns the inner list instead of raising.
    assert parse_json_loose('x {"outer": {"inner": [1]}} y }') == [1]


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------

def test_rate_limiter_spacing():
    rl = RateLimiter(rpm=600)  # 0.1 s interval
    assert rl.interval == pytest.approx(0.1)
    t0 = time.monotonic()
    rl.wait()
    first = time.monotonic() - t0
    assert first < 0.05  # first call is immediate
    rl.wait()
    rl.wait()
    total = time.monotonic() - t0
    # two sleeps of ~0.1 s; Windows' monotonic clock has ~16 ms granularity, so allow slack
    assert 0.15 <= total < 0.6


def test_rate_limiter_no_wait_after_idle():
    rl = RateLimiter(rpm=6000)  # 10 ms
    rl.wait()
    time.sleep(0.03)
    t0 = time.monotonic()
    rl.wait()
    assert time.monotonic() - t0 < 0.01


def test_rate_limiter_rpm_floor():
    assert RateLimiter(0).interval == 60.0
    assert RateLimiter(-5).interval == 60.0


# ---------------------------------------------------------------------------
# MimoClient construction
# ---------------------------------------------------------------------------

def test_missing_api_key(cfg):
    assert cfg.mimo.api_key == ""
    with pytest.raises(MissingAPIKey) as ei:
        MimoClient(cfg)
    assert isinstance(ei.value, LLMError)
    assert "MIMO_API_KEY" in str(ei.value)


def test_env_api_key_overrides(cfg_factory, monkeypatch):
    monkeypatch.setenv("MIMO_API_KEY", "sk-test-env")
    cfg = cfg_factory()
    assert cfg.mimo.api_key == "sk-test-env"
    c = MimoClient(cfg, rpm=6000)
    assert c.chat_model == "mimo-v2.6-flash"
    assert c.max_retries == 3
    assert c.usage["calls"] == 0


# ---------------------------------------------------------------------------
# helpers to build a client with a fake OpenAI SDK underneath
# ---------------------------------------------------------------------------

def rate_limit_error(msg: str = "slow down") -> openai.RateLimitError:
    return openai.RateLimitError(msg, response=httpx.Response(429, request=httpx.Request("POST", "http://x")), body=None)


def status_error(code: int, msg: str = "err") -> openai.APIStatusError:
    return openai.APIStatusError(msg, response=httpx.Response(code, request=httpx.Request("POST", "http://x")), body=None)


def completion(content: str = "ok", prompt=10, completion_=5, seconds=None, finish_reason="stop"):
    usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion_)
    if seconds is not None:
        usage.seconds = seconds
    msg = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)], usage=usage)


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)  # each item: a response or an exception to raise
        self.kwargs: list[dict] = []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        r = self.responses.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


@pytest.fixture
def client(cfg_factory, monkeypatch):
    """MimoClient with an API key, no retry sleeps, no rate limiting, fake SDK."""
    cfg = cfg_factory({"mimo": {"api_key": "sk-fake", "max_retries": 2, "temperature": 0.3, "asr_language": "zh"}})
    c = MimoClient(cfg, rpm=100000)
    c.limiter.interval = 0.0
    sleeps: list[float] = []
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: sleeps.append(s))
    c._test_sleeps = sleeps  # type: ignore[attr-defined]

    def install(*responses):
        fc = FakeCompletions(responses)
        c.client = SimpleNamespace(chat=SimpleNamespace(completions=fc))
        return fc

    c.install = install  # type: ignore[attr-defined]
    return c


# ---------------------------------------------------------------------------
# _with_retry
# ---------------------------------------------------------------------------

def test_with_retry_recovers_from_rate_limit(client):
    fc = client.install(rate_limit_error(), rate_limit_error(), completion("第三次成功"))
    resp = client.chat([{"role": "user", "content": "hi"}])
    assert resp.choices[0].message.content == "第三次成功"
    assert len(fc.kwargs) == 3
    assert len(client._test_sleeps) == 2
    # 429 uses the long ladder so three retries span a per-minute quota window: 10*(1..1.5), then 20*(1..1.5)
    assert 10.0 <= client._test_sleeps[0] <= 15.0
    assert 20.0 <= client._test_sleeps[1] <= 30.0
    assert client.usage["calls"] == 1 and client.usage["prompt_tokens"] == 10 and client.usage["completion_tokens"] == 5


def test_with_retry_honours_retry_after_header(client):
    resp = httpx.Response(429, request=httpx.Request("POST", "http://x"), headers={"Retry-After": "45"})
    err = openai.RateLimitError("slow", response=resp, body=None)
    client.install(err, completion("ok"))
    assert client.text("s", "u") == "ok"
    assert client._test_sleeps[0] >= 45.0


@pytest.mark.parametrize("code", [400, 413, 415, 422])
def test_with_retry_deterministic_4xx_not_retried(client, code):
    fc = client.install(status_error(code, "bad request"), completion("never"))
    with pytest.raises(LLMError) as ei:
        client.text("s", "u")
    assert f"HTTP {code}" in str(ei.value)
    assert len(fc.kwargs) == 1 and client._test_sleeps == []


def test_placeholder_api_key_counts_as_missing(cfg_factory):
    cfg = cfg_factory({"mimo": {"api_key": "sk-填这里"}})
    assert cfg.mimo.api_key == ""          # normalised by load_config
    cfg["mimo"]["api_key"] = "<your-key>"  # and the client guards on its own
    with pytest.raises(MissingAPIKey):
        MimoClient(cfg)


def test_with_retry_gives_up_after_max_retries(client):
    fc = client.install(rate_limit_error(), rate_limit_error(), rate_limit_error(), completion("never"))
    with pytest.raises(LLMError) as ei:
        client.chat([{"role": "user", "content": "hi"}])
    assert "giving up after 3 attempts" in str(ei.value)
    assert isinstance(ei.value.__cause__, openai.RateLimitError)
    assert len(fc.kwargs) == 3  # max_retries=2 -> 3 attempts
    assert client.usage["calls"] == 0


@pytest.mark.parametrize("code", [401, 402, 403, 404])
def test_with_retry_non_retryable_status(client, code):
    fc = client.install(status_error(code), completion("never"))
    with pytest.raises(LLMError) as ei:
        client.text("s", "u")
    assert f"HTTP {code}" in str(ei.value)
    assert len(fc.kwargs) == 1 and client._test_sleeps == []


def test_with_retry_content_filter_421(client):
    from audio_note.llm.client import ContentFiltered

    fc = client.install(status_error(421), completion("never"))
    with pytest.raises(ContentFiltered, match="content filtered") as ei:
        client.text("s", "u")
    assert isinstance(ei.value, LLMError)
    assert len(fc.kwargs) == 1 and client._test_sleeps == []


def test_with_retry_retries_503_and_connection_errors(client):
    client.install(
        status_error(503, "overloaded"),
        openai.APIConnectionError(request=httpx.Request("POST", "http://x")),
        completion("ok"),
    )
    assert client.text("s", "u") == "ok"
    assert len(client._test_sleeps) == 2


def test_with_retry_does_not_catch_unrelated_exceptions(client):
    client.install(ValueError("bug"))
    with pytest.raises(ValueError):
        client.text("s", "u")


# ---------------------------------------------------------------------------
# chat / text / json request shaping
# ---------------------------------------------------------------------------

def test_chat_request_kwargs_defaults(client):
    fc = client.install(completion("x"))
    client.chat([{"role": "user", "content": "hi"}])
    kw = fc.kwargs[0]
    assert kw["model"] == "mimo-v2.6-flash"
    assert kw["extra_body"] == {"thinking": {"type": "disabled"}}
    assert kw["temperature"] == pytest.approx(0.3)
    assert kw["max_tokens"] == 4096 and "max_completion_tokens" not in kw
    assert "response_format" not in kw and "tools" not in kw


def test_chat_request_kwargs_options(client):
    fc = client.install(completion("x"))
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    client.chat([{"role": "user", "content": "hi"}], model="mimo-v2.6-pro", temperature=0.9, max_tokens=99,
                json_mode=True, thinking=True, tools=tools, tool_choice="auto")
    kw = fc.kwargs[0]
    assert kw["model"] == "mimo-v2.6-pro"
    assert kw["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "temperature" not in kw  # thinking mode drops temperature
    assert kw["max_tokens"] == 99
    assert kw["response_format"] == {"type": "json_object"}
    assert kw["tools"] is tools and kw["tool_choice"] == "auto"


def test_text_and_json(client):
    client.install(completion("  回答  "))
    assert client.text("系统", "用户") == "回答"
    fc = client.install(completion('```json\n{"edits": [{"i": 0}]}\n```'))
    assert client.json("系统", "用户") == {"edits": [{"i": 0}]}
    assert fc.kwargs[0]["response_format"] == {"type": "json_object"}
    assert fc.kwargs[0]["messages"] == [{"role": "system", "content": "系统"}, {"role": "user", "content": "用户"}]
    client.install(completion("not json"))
    with pytest.raises(LLMError):
        client.json("s", "u")
    client.install(completion(None))
    assert client.text("s", "u") == ""


# ---------------------------------------------------------------------------
# transcribe_bytes
# ---------------------------------------------------------------------------

def test_transcribe_bytes_request_and_usage(client):
    fc = client.install(completion("你好，世界。", seconds=12.5))
    text, secs = client.transcribe_bytes(b"RIFFfakewav", "audio/wav", "zh")
    assert text == "你好，世界。" and secs == 12.5
    kw = fc.kwargs[0]
    assert kw["model"] == "mimo-v2.5-asr"
    assert kw["extra_body"] == {"asr_options": {"language": "zh"}}
    content = kw["messages"][0]["content"][0]
    assert content["type"] == "input_audio"
    assert content["input_audio"]["data"].startswith("data:audio/wav;base64,")
    import base64
    assert base64.b64decode(content["input_audio"]["data"].split(",", 1)[1]) == b"RIFFfakewav"
    assert client.usage["audio_seconds"] == 12.5 and client.usage["calls"] == 1
    # explicit model override + no usage.seconds -> 0
    fc = client.install(completion("x"))
    text, secs = client.transcribe_bytes(b"abc", model="other-asr")
    assert secs == 0.0 and fc.kwargs[0]["model"] == "other-asr"


def test_transcribe_bytes_rejects_oversize(client):
    fc = client.install(completion("never"))
    with pytest.raises(LLMError, match="too large"):
        client.transcribe_bytes(b"\x00" * (8 * 1024 * 1024))  # 8 MB raw -> ~10.7 MB base64
    assert fc.kwargs == []


def test_transcribe_bytes_retries_rate_limit(client):
    client.install(rate_limit_error(), completion("ok", seconds=1.0))
    assert client.transcribe_bytes(b"abc") == ("ok", 1.0)
