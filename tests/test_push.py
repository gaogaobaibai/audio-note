"""Tests for audio_note.push: payload formats, signing, splitting, Notifier aggregation. No network."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import subprocess
from typing import Any

import httpx
import pytest
import yaml

from audio_note.config import load_config
from audio_note.push import Message, Notifier, get_channels, split_text
from audio_note.push import base, feishu, ntfy, pushplus, serverchan, toast, wecom, wxpusher


# ----------------------------------------------------------------------------- fixtures

class FakeResponse:
    def __init__(self, data: Any = None, status_code: int = 200, text: str = ""):
        self._data = data
        self.status_code = status_code
        self.text = text or (json.dumps(data) if data is not None else "")

    def json(self) -> Any:
        if self._data is None:
            raise ValueError("no json")
        return self._data


class FakePost:
    """Records every httpx.post call; returns queued responses (last one repeats)."""

    def __init__(self, *responses: FakeResponse):
        self.responses = list(responses) or [FakeResponse({"code": 0})]
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        resp = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return resp


@pytest.fixture
def fake_post(monkeypatch):
    def factory(*responses: FakeResponse) -> FakePost:
        fp = FakePost(*responses)
        monkeypatch.setattr(httpx, "post", fp)
        return fp

    return factory


@pytest.fixture
def make_cfg(tmp_path, monkeypatch):
    def factory(push: dict[str, Any]):
        path = tmp_path / "config.yaml"
        path.write_text(
            yaml.safe_dump({"paths": {"data_dir": str(tmp_path / "data")}, "push": push}, allow_unicode=True),
            encoding="utf-8",
        )
        monkeypatch.setenv("AUDIO_NOTE_CONFIG", str(path))
        return load_config(path)

    return factory


@pytest.fixture(autouse=True)
def no_real_subprocess(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("subprocess.run must be mocked in push tests")

    monkeypatch.setattr(subprocess, "run", boom)


MSG = Message(title="每日记录 2026-10-31", body_md="# 概览\n- **张三** 说了点什么\n- 待办：买牛奶", url="https://example.com/x")


# ----------------------------------------------------------------------------- helpers

def test_split_text_prefers_lines_and_respects_bytes():
    text = "\n".join(f"第{i}行 中文内容" for i in range(50))
    chunks = split_text(text, 100)
    assert len(chunks) > 1
    assert "".join(chunks) == text
    assert all(len(c.encode("utf-8")) <= 100 for c in chunks)
    assert all(c.endswith("\n") for c in chunks[:-1])  # split on line boundaries


def test_split_text_long_line_never_breaks_codepoint():
    text = "汉" * 100  # 300 bytes, single line
    chunks = split_text(text, 64)
    assert "".join(chunks) == text
    assert all(len(c.encode("utf-8")) <= 64 for c in chunks)
    assert all(len(c) == 21 for c in chunks[:-1])
    assert split_text("", 10) == [""]
    assert split_text("short", 10) == ["short"]


def test_markdown_to_text():
    md = "# 标题\n\n- **粗体** 和 *斜体* 以及 `代码`\n> 引用 [链接](https://x.y)\n\n```\ncode block\n```\n---\n"
    text = base.markdown_to_text(md)
    assert text == "标题\n\n粗体 和 斜体 以及 代码\n引用 链接"


def test_truncate_bytes():
    out = base.truncate_bytes("汉" * 100, 50)
    assert len(out.encode("utf-8")) <= 50
    assert out.endswith("…(已截断)")
    assert base.truncate_bytes("ok", 50) == "ok"


# ----------------------------------------------------------------------------- feishu

def test_feishu_signature_matches_hmac_vector():
    ts, secret = 1730000000, "s3cret"
    expected = base64.b64encode(
        hmac.new(f"{ts}\n{secret}".encode("utf-8"), b"", hashlib.sha256).digest()
    ).decode()
    assert feishu.sign(ts, secret) == expected
    payload = feishu.build_payload(MSG, secret=secret, timestamp=ts)
    assert payload["timestamp"] == str(ts)
    assert payload["sign"] == expected
    assert "sign" not in feishu.build_payload(MSG)


def test_feishu_card_payload_format():
    p = feishu.build_payload(MSG)
    assert p["msg_type"] == "interactive"
    card = p["card"]
    assert card["schema"] == "2.0"
    assert card["header"]["title"] == {"tag": "plain_text", "content": MSG.title}
    el = card["body"]["elements"][0]
    assert el["tag"] == "markdown"
    assert el["content"].startswith(MSG.body_md)
    assert MSG.url in el["content"]


def test_feishu_body_limited_to_20kb():
    big = Message("t", "好" * 20000)
    p = feishu.build_payload(big)
    assert len(p["card"]["body"]["elements"][0]["content"].encode("utf-8")) <= 20 * 1024


def test_feishu_send_ok_and_fallback_to_text(fake_post, make_cfg):
    cfg = make_cfg({"channels": ["feishu"], "feishu": {"webhook": "https://open.feishu.cn/hook/abc", "secret": "k"}})
    fp = fake_post(FakeResponse({"code": 0, "msg": "success"}))
    ch = feishu.FeishuChannel(cfg)
    ch.send(MSG)
    assert fp.calls[0]["url"] == "https://open.feishu.cn/hook/abc"
    assert fp.calls[0]["json"]["msg_type"] == "interactive"
    assert fp.calls[0]["json"]["sign"] == feishu.sign(fp.calls[0]["json"]["timestamp"], "k")
    assert fp.calls[0]["timeout"] == 10.0

    # card rejected -> text fallback accepted
    fp = fake_post(FakeResponse({"code": 19001, "msg": "param invalid"}), FakeResponse({"code": 0}))
    ch.send(MSG)
    assert [c["json"]["msg_type"] for c in fp.calls] == ["interactive", "text"]
    txt = fp.calls[1]["json"]["content"]["text"]
    assert txt.startswith(MSG.title) and "**" not in txt

    # both rejected -> RuntimeError
    fp = fake_post(FakeResponse({"code": 19001, "msg": "bad"}))
    with pytest.raises(RuntimeError):
        ch.send(MSG)
    # card schema errors come back as HTTP 400 + JSON code: still a rejection -> text fallback
    fp = fake_post(FakeResponse({"code": 9499, "msg": "Bad Request"}, status_code=400), FakeResponse({"code": 0}))
    ch.send(MSG)
    assert [c["json"]["msg_type"] for c in fp.calls] == ["interactive", "text"]
    # http error without a JSON rejection body -> RuntimeError
    fake_post(FakeResponse(None, status_code=500, text="boom"))
    with pytest.raises(RuntimeError, match="HTTP 500"):
        ch.send(MSG)
    fake_post(FakeResponse(None, status_code=404, text="not found"))
    with pytest.raises(RuntimeError, match="HTTP 404"):
        ch.send(MSG)


# ----------------------------------------------------------------------------- wecom

def test_wecom_splits_at_4096_bytes(fake_post, make_cfg):
    cfg = make_cfg({"channels": ["wecom"], "wecom": {"webhook": "https://qyapi.weixin.qq.com/hook"}})
    body = "\n".join(f"第{i}行：" + "内容" * 20 for i in range(200))  # ~ 26 KB
    payloads = wecom.build_payloads(Message("标题", body))
    assert len(payloads) > 1
    for i, p in enumerate(payloads, 1):
        assert p["msgtype"] == "markdown"
        assert len(p["markdown"]["content"].encode("utf-8")) <= 4096
        assert p["markdown"]["content"].endswith(f"({i}/{len(payloads)})")
    joined = "".join(p["markdown"]["content"] for p in payloads)
    assert "第199行" in joined and "## 标题" in joined

    fp = fake_post(FakeResponse({"errcode": 0, "errmsg": "ok"}))
    wecom.WecomChannel(cfg).send(Message("标题", body))
    assert len(fp.calls) == len(payloads)

    single = wecom.build_payloads(MSG)
    assert len(single) == 1 and "(1/1)" not in single[0]["markdown"]["content"]

    fake_post(FakeResponse({"errcode": 93000, "errmsg": "invalid webhook"}))
    with pytest.raises(RuntimeError, match="93000"):
        wecom.WecomChannel(cfg).send(MSG)


# ----------------------------------------------------------------------------- serverchan / pushplus / wxpusher

def test_serverchan(fake_post, make_cfg):
    cfg = make_cfg({"channels": ["serverchan"], "serverchan": {"sendkey": "SCT123"}})
    fp = fake_post(FakeResponse({"code": 0}))
    serverchan.ServerChanChannel(cfg).send(MSG)
    assert fp.calls[0]["url"] == "https://sctapi.ftqq.com/SCT123.send"
    assert fp.calls[0]["json"]["title"] == MSG.title
    assert fp.calls[0]["json"]["desp"].startswith(MSG.body_md)
    assert len(serverchan.build_payload(Message("x" * 100, "b"))["title"]) == 32
    fake_post(FakeResponse({"code": 40001, "message": "bad key"}))
    with pytest.raises(RuntimeError):
        serverchan.ServerChanChannel(cfg).send(MSG)


def test_pushplus(fake_post, make_cfg):
    cfg = make_cfg({"channels": ["pushplus"], "pushplus": {"token": "tok"}})
    fp = fake_post(FakeResponse({"code": 200, "msg": "请求成功"}))
    pushplus.PushPlusChannel(cfg).send(MSG)
    assert fp.calls[0]["url"] == "https://www.pushplus.plus/send"
    j = fp.calls[0]["json"]
    assert j["token"] == "tok" and j["title"] == MSG.title and j["template"] == "markdown"
    assert j["content"].startswith(MSG.body_md)
    fake_post(FakeResponse({"code": 903, "msg": "无效的用户token"}))
    with pytest.raises(RuntimeError, match="903"):
        pushplus.PushPlusChannel(cfg).send(MSG)


def test_wxpusher_spt_and_apptoken(fake_post, make_cfg):
    url, j = wxpusher.build_payload(MSG, spt="SPT_x")
    assert url == "https://wxpusher.zjiecode.com/api/send/message/simple-push"
    assert j["spt"] == "SPT_x" and j["contentType"] == 3 and "appToken" not in j
    assert j["summary"] == MSG.title[:20] and j["content"].startswith(MSG.body_md)

    url, j = wxpusher.build_payload(MSG, app_token="AT_x", uids=["UID_1", "UID_2"])
    assert url == "https://wxpusher.zjiecode.com/api/send/message"
    assert j["appToken"] == "AT_x" and j["uids"] == ["UID_1", "UID_2"] and "spt" not in j

    cfg = make_cfg({"channels": ["wxpusher"], "wxpusher": {"app_token": "AT_x", "uids": ["UID_1"]}})
    fp = fake_post(FakeResponse({"code": 1000, "msg": "处理成功"}))
    wxpusher.WxPusherChannel(cfg).send(MSG)
    assert fp.calls[0]["url"].endswith("/api/send/message")
    fake_post(FakeResponse({"code": 1001, "msg": "appToken 错误"}))
    with pytest.raises(RuntimeError, match="1001"):
        wxpusher.WxPusherChannel(cfg).send(MSG)

    with pytest.raises(base.ChannelNotConfigured):
        wxpusher.WxPusherChannel(make_cfg({"wxpusher": {"app_token": "AT_x", "uids": []}}))


# ----------------------------------------------------------------------------- ntfy

def test_ntfy_headers_ascii_only(fake_post, make_cfg):
    body, headers = ntfy.build_payload(MSG)
    assert headers["Title"] == "audio_note"  # Chinese title moved into body
    assert headers["Markdown"] == "yes" and headers["Click"] == MSG.url
    assert body.decode("utf-8").startswith(f"**{MSG.title}**\n\n{MSG.body_md}")
    assert all(v.isascii() for v in headers.values())

    body, headers = ntfy.build_payload(Message("Daily digest", "hi"))
    assert headers["Title"] == "Daily digest" and body == b"hi" and "Click" not in headers

    cfg = make_cfg({"channels": ["ntfy"], "ntfy": {"topic": "my-topic", "server": "https://ntfy.example/"}})
    fp = fake_post(FakeResponse({"id": "x"}))
    ntfy.NtfyChannel(cfg).send(MSG)
    assert fp.calls[0]["url"] == "https://ntfy.example/my-topic"
    assert isinstance(fp.calls[0]["content"], bytes) and fp.calls[0]["json"] is None
    fake_post(FakeResponse(None, status_code=403, text="forbidden"))
    with pytest.raises(RuntimeError, match="403"):
        ntfy.NtfyChannel(cfg).send(MSG)


# ----------------------------------------------------------------------------- toast

def test_toast_builds_safe_powershell(monkeypatch, make_cfg):
    runs: list[dict[str, Any]] = []

    def fake_run(cmd, **kw):
        runs.append({"cmd": cmd, **kw})
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(toast, "is_windows", lambda: True)

    evil = Message("标题 <b> & 'quote'", "**粗体**\n- 条目 \"x\"; Remove-Item C:\\ ; $(calc)\n" + "长" * 500)
    p = toast.build_payload(evil)
    assert len(p["body"]) <= 200 and "**" not in p["body"]
    assert "&lt;b&gt; &amp; 'quote'" in p["xml"] and "&quot;x&quot;" in p["xml"]
    assert "Remove-Item" not in p["script"]  # text travels base64, never as code
    # toasts are shown under PowerShell's registered AUMID: an unregistered id ('audio_note') is silently dropped
    assert "CreateToastNotifier('{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe')" in p["script"]
    assert "ToastText02" in p["xml"]
    assert toast.build_payload(Message("", "x"))["title"] == "audio-note"

    toast.ToastChannel(make_cfg({"channels": ["toast"]})).send(evil)
    cmd = runs[0]["cmd"]
    assert cmd[0].lower().startswith("powershell") and "-NoProfile" in cmd and "-NonInteractive" in cmd
    enc = cmd[cmd.index("-EncodedCommand") + 1]
    script = base64.b64decode(enc).decode("utf-16-le")
    assert script == p["script"]
    xml_b64 = script.split("FromBase64String('")[1].split("')")[0]
    assert base64.b64decode(xml_b64).decode("utf-8") == p["xml"]

    def fail_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, b"", "出错了".encode("utf-8"))

    monkeypatch.setattr(subprocess, "run", fail_run)
    with pytest.raises(RuntimeError, match="出错了"):
        toast.ToastChannel(make_cfg({"channels": ["toast"]})).send(MSG)


def test_toast_skipped_off_windows(monkeypatch, make_cfg):
    monkeypatch.setattr(toast, "is_windows", lambda: False)
    assert get_channels(make_cfg({"channels": ["toast"]})) == []


# ----------------------------------------------------------------------------- notifier

def test_notifier_skips_unknown_and_unconfigured(make_cfg, caplog):
    cfg = make_cfg({
        "channels": ["feishu", "bogus", "serverchan", "ntfy", "serverchan"],
        "feishu": {"webhook": ""},               # unconfigured -> skipped
        "serverchan": {"sendkey": "SCT1"},
        "ntfy": {"topic": "t"},
    })
    with caplog.at_level("WARNING", logger="audio_note.push"):
        n = Notifier(cfg)
    assert n.names == ["serverchan", "ntfy"]
    assert any("bogus" in r.message for r in caplog.records)
    assert any("feishu" in r.message for r in caplog.records)


def test_notifier_aggregates_errors_and_never_raises(make_cfg, fake_post, caplog):
    cfg = make_cfg({
        "channels": ["serverchan", "pushplus", "ntfy"],
        "serverchan": {"sendkey": "SCT1"},
        "pushplus": {"token": "tok"},
        "ntfy": {"topic": "t"},
    })
    n = Notifier(cfg)

    class Exploding:
        name = "boom"

        def send(self, msg):
            raise ValueError("kaboom")

    n.channels.append(Exploding())
    # serverchan ok (code 0), pushplus fails (needs 200), ntfy ok (2xx)
    fake_post(FakeResponse({"code": 0}))
    with caplog.at_level("INFO", logger="audio_note.push"):
        res = n.send(MSG)
    assert list(res) == ["serverchan", "pushplus", "ntfy", "boom"]
    assert res["serverchan"] is None and res["ntfy"] is None
    assert "code=0" in res["pushplus"]
    assert "kaboom" in res["boom"]
    assert any("推送失败 [pushplus]" in r.message for r in caplog.records)
    assert any("推送成功 [serverchan]" in r.message for r in caplog.records)


def test_notifier_without_channels_returns_empty(make_cfg):
    cfg = make_cfg({"channels": [], "feishu": {"webhook": ""}})
    assert Notifier(cfg).send(MSG) == {}


def test_channel_protocol_and_registry():
    assert set(base.CHANNEL_REGISTRY) == {"feishu", "wecom", "serverchan", "pushplus", "wxpusher", "ntfy", "toast"}
    for spec in base.CHANNEL_REGISTRY.values():
        cls = base._load_channel_class(spec)
        assert isinstance(cls.name, str) and callable(cls.send)
