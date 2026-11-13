"""audio_note.llm.client 1.0: json_schema with fallback, streaming, ASR prompt / context, content_filter on a
200 reply, listen(), scoped() views, on_usage records and the process-wide rate limiter.

The OpenAI SDK object inside MimoClient is replaced by a fake; nothing goes on the wire.
"""
from __future__ import annotations

import base64
from types import SimpleNamespace

import httpx
import openai
import pytest

from audio_note.llm import client as client_mod
from audio_note.llm.client import ContentFiltered, LLMError, MimoClient, shared_limiter

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["edits"],
          "properties": {"edits": {"type": "array", "items": {"type": "object"}}}}


def status_error(code: int, msg: str = "err") -> openai.APIStatusError:
    return openai.APIStatusError(msg, response=httpx.Response(code, request=httpx.Request("POST", "http://x")), body=None)


def completion(content: str | None = "ok", prompt=10, completion_=5, seconds=None, finish_reason="stop"):
    usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion_)
    if seconds is not None:
        usage.seconds = seconds
    msg = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)], usage=usage)


def chunk(text: str | None = None, usage=None):
    choices = [SimpleNamespace(delta=SimpleNamespace(content=text))] if text is not None else []
    return SimpleNamespace(choices=choices, usage=usage)


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.kwargs: list[dict] = []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        r = self.responses.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


@pytest.fixture
def client(cfg_factory, monkeypatch):
    cfg = cfg_factory({"mimo": {"api_key": "sk-fake-v1", "max_retries": 1, "temperature": 0.3}})
    c = MimoClient(cfg, rpm=100000)
    c.limiter.interval = 0.0
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)
    records: list[dict] = []
    c.on_usage = records.append
    c.records = records  # type: ignore[attr-defined]

    def install(*responses):
        fc = FakeCompletions(responses)
        c.client = SimpleNamespace(chat=SimpleNamespace(completions=fc))
        return fc

    c.install = install  # type: ignore[attr-defined]
    return c


# --- json_schema ---------------------------------------------------------------------------------------------

def test_json_schema_request_shape(client):
    fc = client.install(completion('{"edits": []}'))
    out = client.json("系统", "用户", schema=SCHEMA, schema_name="correction", max_tokens=1500)
    assert out == {"edits": []}
    kw = fc.kwargs[0]
    assert kw["response_format"] == {"type": "json_schema", "json_schema": {"name": "correction", "schema": SCHEMA, "strict": True}}
    assert kw["max_tokens"] == 1500 and kw["extra_body"] == {"thinking": {"type": "disabled"}} and kw["temperature"] == pytest.approx(0.3)
    assert kw["messages"] == [{"role": "system", "content": "系统"}, {"role": "user", "content": "用户"}]
    # chat() takes the same thing directly; json_schema wins over json_mode
    fc = client.install(completion("{}"))
    client.chat([{"role": "user", "content": "hi"}], json_schema={"name": "r", "schema": SCHEMA}, json_mode=True, thinking=True)
    assert fc.kwargs[0]["response_format"]["type"] == "json_schema" and "temperature" not in fc.kwargs[0]


def test_json_without_schema_is_unchanged(client):
    fc = client.install(completion('{"a": 1}'))
    assert client.json("s", "u") == {"a": 1}
    assert fc.kwargs[0]["response_format"] == {"type": "json_object"}
    assert set(fc.kwargs[0]) == {"model", "messages", "max_tokens", "extra_body", "temperature", "response_format"}


@pytest.mark.parametrize("code", [400, 422])
def test_rejected_schema_falls_back_to_json_object_once_and_is_remembered(client, code):
    fc = client.install(status_error(code, "response_format.json_schema is not supported"), completion('{"edits": [1]}'),
                        completion('{"edits": [2]}'))
    assert client.json("s", "u", schema=SCHEMA, schema_name="fix") == {"edits": [1]}
    assert [k["response_format"]["type"] for k in fc.kwargs] == ["json_schema", "json_object"]
    assert fc.kwargs[0]["messages"] == fc.kwargs[1]["messages"]
    # the same (model, schema name) goes straight to JSON mode afterwards; another schema name is tried again
    assert client.json("s", "u", schema=SCHEMA, schema_name="fix") == {"edits": [2]}
    assert [k["response_format"]["type"] for k in fc.kwargs] == ["json_schema", "json_object", "json_object"]
    fc = client.install(completion("{}"))
    client.json("s", "u", schema=SCHEMA, schema_name="other")
    assert fc.kwargs[0]["response_format"]["type"] == "json_schema"
    # usage: the rejected attempt is an error, the retries are calls
    assert [r["ok"] for r in client.records] == [False, True, True, True] and client.usage["calls"] == 3


def test_schema_fallback_does_not_hide_other_errors(client):
    fc = client.install(status_error(401, "bad key"), completion("never"))
    with pytest.raises(LLMError, match="HTTP 401") as ei:
        client.json("s", "u", schema=SCHEMA)
    assert ei.value.status_code == 401 and len(fc.kwargs) == 1
    fc = client.install(status_error(421), completion("never"))
    with pytest.raises(ContentFiltered):
        client.json("s", "u", schema=SCHEMA)
    assert len(fc.kwargs) == 1
    # a plain json_mode call that gets a 400 is not retried either
    fc = client.install(status_error(400), completion("never"))
    with pytest.raises(LLMError, match="HTTP 400"):
        client.json("s", "u")
    assert len(fc.kwargs) == 1


# --- ASR ---------------------------------------------------------------------------------------------------------

def test_transcribe_default_request_is_exactly_the_old_one(client):
    fc = client.install(completion("你好。", seconds=3.0))
    assert client.transcribe_bytes(b"RIFFfakewav", "audio/wav", "zh") == ("你好。", 3.0)
    b64 = base64.b64encode(b"RIFFfakewav").decode()
    assert fc.kwargs[0] == {
        "model": "mimo-v2.5-asr",
        "messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": f"data:audio/wav;base64,{b64}"}}]}],
        "extra_body": {"asr_options": {"language": "zh"}},
    }
    # prompt=None / context=None (and empty strings) change nothing
    fc = client.install(completion("x"), completion("y"))
    client.transcribe_bytes(b"RIFFfakewav", "audio/wav", "zh", prompt=None, context=None)
    client.transcribe_bytes(b"RIFFfakewav", "audio/wav", "zh", prompt="", context="")
    assert fc.kwargs[0] == fc.kwargs[1] and len(fc.kwargs[0]["messages"]) == 1 and set(fc.kwargs[0]) == {"model", "messages", "extra_body"}


def test_transcribe_prompt_is_a_system_message_and_context_a_text_message_before_the_audio(client):
    fc = client.install(completion("今天先把灰度发布过一遍。", seconds=4.0))
    text, secs = client.transcribe_bytes(b"RIFFx", prompt="可能出现的专有名词：灰度发布、王敏", context="背景：项目晨会。")
    assert (text, secs) == ("今天先把灰度发布过一遍。", 4.0)
    msgs = fc.kwargs[0]["messages"]
    assert msgs[0] == {"role": "system", "content": "可能出现的专有名词：灰度发布、王敏"}
    assert msgs[1] == {"role": "user", "content": "背景：项目晨会。"}
    assert msgs[2]["role"] == "user" and [p["type"] for p in msgs[2]["content"]] == ["input_audio"]   # never a text part next to the audio
    assert fc.kwargs[0]["extra_body"] == {"asr_options": {"language": "auto"}} and len(msgs) == 3
    fc = client.install(completion("x"))
    client.transcribe_bytes(b"RIFFx", prompt="热词")
    assert [m["role"] for m in fc.kwargs[0]["messages"]] == ["system", "user"]


def test_content_filter_on_a_200_reply_is_raised_not_returned(client):
    client.install(completion("I'm sorry, but I can't assist with that.", seconds=2.0, finish_reason="content_filter"))
    with pytest.raises(ContentFiltered, match="content_filter"):
        client.transcribe_bytes(b"RIFFx")
    assert client.usage["calls"] == 1 and client.usage["audio_seconds"] == 2.0     # the request happened and was billed
    assert client.records[-1]["purpose"] == "asr" and client.records[-1]["audio_s"] == 2.0
    client.install(completion("refusal", finish_reason="content_filter"))
    with pytest.raises(ContentFiltered):
        client.listen([b"RIFFx"], "请逐字转写这段录音，只输出转写文本。")


# --- listen ------------------------------------------------------------------------------------------------------

def test_listen_sends_clips_then_text_in_one_user_message(client):
    fc = client.install(completion(" 早上好，今天先过灰度发布。 "))
    out = client.listen([b"RIFFone", b"ID3mp3bytes"], "请逐字转写这段录音，只输出转写文本。")
    assert out == "早上好，今天先过灰度发布。"
    kw = fc.kwargs[0]
    assert kw["model"] == "mimo-v2.6-flash" and kw["max_tokens"] == 600
    assert kw["extra_body"] == {"thinking": {"type": "disabled"}} and "response_format" not in kw
    assert len(kw["messages"]) == 1 and kw["messages"][0]["role"] == "user"
    parts = kw["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["input_audio", "input_audio", "text"]
    assert parts[0]["input_audio"]["data"] == "data:audio/wav;base64," + base64.b64encode(b"RIFFone").decode()
    assert parts[1]["input_audio"]["data"].startswith("data:audio/mpeg;base64,")
    assert parts[2] == {"type": "text", "text": "请逐字转写这段录音，只输出转写文本。"}
    assert client.records[-1]["purpose"] == "listen" and client.records[-1]["model"] == "mimo-v2.6-flash"


def test_listen_options_and_size_limit(client, cfg_factory):
    client.cfg["mimo"]["listen_model"] = "mimo-v2.6-pro"
    fc = client.install(completion('{"text": "a"}'))
    client.listen([b"RIFFx"], "t", json_schema={"name": "heard", "schema": SCHEMA}, thinking=True, max_tokens=50, purpose="hearing")
    kw = fc.kwargs[0]
    assert kw["model"] == "mimo-v2.6-pro" and kw["max_tokens"] == 50 and kw["extra_body"] == {"thinking": {"type": "enabled"}}
    assert kw["response_format"]["json_schema"]["name"] == "heard" and client.records[-1]["purpose"] == "hearing"
    fc = client.install(completion("x"))
    client.listen([b"RIFFx"], "t", model="other-omni")
    assert fc.kwargs[0]["model"] == "other-omni"
    fc = client.install(completion("never"))
    with pytest.raises(LLMError, match="too large"):
        client.listen([b"\x00" * (4 * 1024 * 1024), b"\x00" * (4 * 1024 * 1024)], "t")
    assert fc.kwargs == []


# --- streaming -----------------------------------------------------------------------------------------------------

def test_chat_stream_yields_text_and_accounts_usage(client):
    usage = SimpleNamespace(prompt_tokens=30, completion_tokens=12)
    fc = client.install([chunk("你"), chunk("好"), chunk(""), chunk(None), chunk("。"), chunk(None, usage=usage)])
    assert list(client.chat_stream([{"role": "user", "content": "hi"}], max_tokens=99)) == ["你", "好", "。"]
    kw = fc.kwargs[0]
    assert kw["stream"] is True and kw["stream_options"] == {"include_usage": True} and kw["max_tokens"] == 99
    assert kw["extra_body"] == {"thinking": {"type": "disabled"}} and "response_format" not in kw and "tools" not in kw
    assert client.usage["calls"] == 1 and client.usage["prompt_tokens"] == 30 and client.usage["completion_tokens"] == 12
    assert client.records == [{"purpose": "chat", "model": "mimo-v2.6-flash", "ok": True, "prompt_tokens": 30,
                               "completion_tokens": 12, "audio_s": 0.0, "wall_s": pytest.approx(0.0, abs=5.0)}]


def test_chat_stream_without_stream_options_support(client):
    fc = client.install(status_error(400, "unknown field stream_options"), [chunk("a"), chunk("b")], [chunk("c")])
    assert "".join(client.chat_stream([{"role": "user", "content": "x"}])) == "ab"
    assert "stream_options" in fc.kwargs[0] and "stream_options" not in fc.kwargs[1]
    assert "".join(client.chat_stream([{"role": "user", "content": "x"}])) == "c"       # remembered: not offered again
    assert "stream_options" not in fc.kwargs[2] and client.usage["calls"] == 2


def test_chat_stream_errors_and_early_stop(client):
    client.install(status_error(401, "bad key"))
    with pytest.raises(LLMError, match="HTTP 401"):
        client.chat_stream([{"role": "user", "content": "x"}])            # opening the stream fails at call time
    assert client.records[-1]["ok"] is False

    def broken():
        yield chunk("部分")
        raise openai.APIConnectionError(request=httpx.Request("POST", "http://x"))

    client.install(broken())
    got: list[str] = []
    with pytest.raises(LLMError, match="interrupted"):
        for piece in client.chat_stream([{"role": "user", "content": "x"}]):
            got.append(piece)
    assert got == ["部分"] and client.records[-1]["ok"] is False
    calls = client.usage["calls"]
    client.install([chunk("一"), chunk("二"), chunk("三")])
    gen = client.chat_stream([{"role": "user", "content": "x"}])
    assert next(gen) == "一"
    gen.close()                                                           # the consumer stops reading: still one accounted call
    assert client.usage["calls"] == calls + 1 and client.records[-1]["ok"] is True


# --- scoped views / usage ------------------------------------------------------------------------------------------

def test_scoped_view_shares_everything_but_the_purpose(client):
    view = client.scoped("refine")
    assert isinstance(view, MimoClient) and view is not client and view.purpose == "refine" and client.purpose is None
    assert view.limiter is client.limiter and view.usage is client.usage and view.client is client.client and view.cfg is client.cfg
    fc = client.install(completion("a"), completion("b"), completion("c", seconds=1.5), completion("d"))   # swapped AFTER the view was made
    assert view.text("s", "u") == "a" and client.text("s", "u") == "b"
    view.transcribe_bytes(b"RIFFx")
    view.text("s", "u", purpose="episode")                                # an explicit purpose wins
    assert len(fc.kwargs) == 4 and client.usage["calls"] == 4
    assert [r["purpose"] for r in client.records] == ["refine", "other", "refine", "episode"]
    assert [r["model"] for r in client.records] == ["mimo-v2.6-flash", "mimo-v2.6-flash", "mimo-v2.5-asr", "mimo-v2.6-flash"]
    nested = view.scoped("daily")
    assert nested.purpose == "daily" and nested.client is client.client
    # attribute writes on a view land on the client it came from
    seen: list[dict] = []
    view.on_usage = seen.append
    assert client.on_usage == seen.append
    client.install(completion("e"))
    client.text("s", "u")
    assert len(seen) == 1 and seen[0]["purpose"] == "other"
    with pytest.raises(AttributeError):
        view.no_such_attribute  # noqa: B018
    with pytest.raises(AttributeError):
        client.no_such_attribute  # noqa: B018


def test_on_usage_records_failures_and_never_breaks_a_call(client):
    client.install(status_error(400, "bad"))
    with pytest.raises(LLMError):
        client.text("s", "u", purpose="daily")
    rec = client.records[-1]
    assert rec["ok"] is False and rec["purpose"] == "daily" and rec["prompt_tokens"] == 0 and rec["wall_s"] >= 0
    assert set(rec) == {"purpose", "model", "ok", "prompt_tokens", "completion_tokens", "audio_s", "wall_s"}
    client.on_usage = lambda rec: (_ for _ in ()).throw(RuntimeError("sink broke"))
    client.install(completion("fine"))
    assert client.text("s", "u") == "fine"
    client.on_usage = None
    client.install(completion("fine"))
    assert client.text("s", "u") == "fine"


def test_rate_limiter_is_shared_per_account(cfg_factory):
    cfg = cfg_factory({"mimo": {"api_key": "sk-shared-limiter"}})
    a, b = MimoClient(cfg, rpm=90), MimoClient(cfg, rpm=90)
    assert a.limiter is b.limiter and a.usage is not b.usage
    assert a.limiter is shared_limiter(cfg.mimo.base_url, "sk-shared-limiter", 90)
    c = MimoClient(cfg, rpm=600)                                           # the newest rpm wins, for everyone
    assert c.limiter is a.limiter and a.limiter.interval == pytest.approx(0.1)
    cfg["mimo"]["api_key"] = "sk-another-account"
    assert MimoClient(cfg, rpm=90).limiter is not a.limiter
    cfg["mimo"]["base_url"] = "https://example.invalid/v1"
    cfg["mimo"]["api_key"] = "sk-shared-limiter"
    assert MimoClient(cfg, rpm=90).limiter is not a.limiter
    assert not any("sk-shared-limiter" in str(k) for k in client_mod._LIMITERS)      # the registry never holds a key
