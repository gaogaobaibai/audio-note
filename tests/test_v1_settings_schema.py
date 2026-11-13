"""ui/settings_schema.py: the settings whitelist follows config.DEFAULTS at run time, and GET /api/config
ships the validation rules as data ("schema") so the page stops duplicating them (SPEC v1 sections 4.1 and 8)."""
from __future__ import annotations

import copy
import logging

import pytest
import yaml
from fastapi import HTTPException
from fastapi.testclient import TestClient

from audio_note import config
from audio_note.runtime import Runtime
from audio_note.ui import settings_schema as ss
from audio_note.ui.app import create_app

#: the SPEC 4.1 keys, with a default each (what package A1 adds to config.DEFAULTS)
V1_KEYS = {
    "performance": {"mode": "balanced", "threads": 0, "background_priority": True, "defer_on_battery": False},
    "pipeline": {"realtime": True},
    "mimo": {"listen_model": "mimo-v2.6-flash", "asr_prompt": True, "asr_prompt_max_terms": 50},
    "ui": {"events": True},
}


def _defaults(with_v1: bool) -> dict:
    """A copy of config.DEFAULTS that certainly has / certainly lacks the SPEC 4.1 keys."""
    d = copy.deepcopy(config.DEFAULTS)
    for section, keys in V1_KEYS.items():
        if with_v1:
            sec = d.setdefault(section, {})
            for k, v in keys.items():
                sec.setdefault(k, v)
        elif section == "performance":
            d.pop(section, None)
        else:
            for k in keys:
                d.get(section, {}).pop(k, None)
    return d


def _isolated(tmp_path, **extra) -> dict:
    """Config overrides for a Runtime under test: no vector model, and the glossary inside the temp dir
    (its default location is <project root>/glossary.txt, which a test must never write)."""
    return {"memory": {"retrieval": "fts", "embedding_model": "hash"},
            "correction": {"glossary_file": str(tmp_path / "glossary.txt")}, **extra}


@pytest.fixture
def rt(cfg_factory, tmp_path) -> Runtime:
    r = Runtime(cfg_factory(_isolated(tmp_path)))
    yield r
    r.shutdown()
    from audio_note.logging_setup import _remove_own_handlers

    _remove_own_handlers(logging.getLogger())


@pytest.fixture
def client(rt: Runtime) -> TestClient:
    with TestClient(create_app(rt), raise_server_exceptions=False) as c:
        yield c


# --- the whitelist follows config.DEFAULTS ---------------------------------------------------------------------------

def test_whitelist_names_the_v1_keys():
    for section, keys in V1_KEYS.items():
        assert set(keys) <= ss.EDITABLE[section], section


def test_keys_missing_from_defaults_are_neither_offered_nor_accepted(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "DEFAULTS", _defaults(with_v1=False))
    known = ss.editable()
    assert "performance" not in known and "events" not in known["ui"] and "realtime" not in known["pipeline"]
    assert not {"listen_model", "asr_prompt", "asr_prompt_max_terms"} & known["mimo"]
    assert not ss.is_editable("performance", "mode") and ss.is_editable("audio", "gain")
    sc = ss.schema()
    assert "performance" not in sc["sections"] and not any(k.startswith("performance.") for k in sc["keys"])
    assert "ui.events" not in sc["keys"] and "mimo.listen_model" not in sc["defaults"]
    with pytest.raises(HTTPException) as ex:
        ss.coerce_setting(("performance", "mode"), "eco")
    assert ex.value.status_code == 400 and ex.value.detail == "不允许修改 performance.mode"
    # speakers.num_threads = 0 means "follow performance.*": meaningless (and refused) without that section
    assert ss.range_for(("speakers", "num_threads")) == (1, 64)
    with pytest.raises(HTTPException):
        ss.coerce_setting(("speakers", "num_threads"), 0)


def test_keys_present_in_defaults_are_offered_and_validated(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "DEFAULTS", _defaults(with_v1=True))
    known = ss.editable()
    for section, keys in V1_KEYS.items():
        assert set(keys) <= known[section], section
    c = ss.coerce_setting
    assert c(("performance", "mode"), " eco ") == "eco" and c(("performance", "threads"), "2") == 2
    assert c(("performance", "background_priority"), "false") is False and c(("performance", "defer_on_battery"), True) is True
    assert c(("pipeline", "realtime"), "off") is False and c(("ui", "events"), "1") is True
    assert c(("mimo", "listen_model"), " mimo-v2.6-pro ") == "mimo-v2.6-pro"
    assert c(("mimo", "asr_prompt"), "no") is False and c(("mimo", "asr_prompt_max_terms"), 80.0) == 80
    assert c(("speakers", "num_threads"), 0) == 0 and ss.range_for(("speakers", "num_threads")) == (0, 64)
    for path, bad, msg in ((("performance", "mode"), "turbo", "performance.mode 只能是 eco / balanced / fast"),
                           (("performance", "threads"), 65, "performance.threads 应在 0–64 之间"),
                           (("performance", "threads"), -1, "performance.threads 应在 0–64 之间"),
                           (("performance", "threads"), "many", "performance.threads 应为整数"),
                           (("performance", "background_priority"), "maybe", "performance.background_priority 应为 true/false"),
                           (("mimo", "asr_prompt_max_terms"), 501, "mimo.asr_prompt_max_terms 应在 0–500 之间"),
                           (("ui", "events"), [], "ui.events 应为 true/false"),
                           (("performance",), {"mode": "eco"}, "performance 是一组设置，不能整个替换")):
        with pytest.raises(HTTPException) as ex:
            c(path, bad)
        assert ex.value.status_code == 400 and ex.value.detail == msg, path


def test_editable_reads_defaults_at_call_time(monkeypatch: pytest.MonkeyPatch):
    """A key another package adds to config.DEFAULTS after import is picked up without touching this module."""
    d = _defaults(with_v1=False)
    monkeypatch.setattr(config, "DEFAULTS", d)
    assert "performance" not in ss.editable()
    d["performance"] = {"mode": "balanced"}
    assert ss.editable()["performance"] == {"mode"} and ss.schema()["keys"]["performance.mode"]["choices"] == ["eco", "balanced", "fast"]


# --- schema as data ---------------------------------------------------------------------------------------------------

def test_schema_describes_every_editable_leaf():
    sc = ss.schema()
    assert set(sc) == {"sections", "keys", "choices", "ranges", "nullable", "secret", "defaults"}
    known = ss.editable()
    assert sc["sections"] == [s for s in ss.EDITABLE if s in known]
    # every editable key is there (groups such as push.feishu expanded to their leaves)
    for section, keys in known.items():
        for key in keys:
            d = config.DEFAULTS[section][key]
            if isinstance(d, dict):
                assert all(f"{section}.{key}.{k2}" in sc["keys"] for k2 in d), (section, key)
            else:
                assert f"{section}.{key}" in sc["keys"], (section, key)
    for dotted, e in sc["keys"].items():
        assert e["section"] == dotted.split(".")[0] and f"{e['section']}.{e['key']}" == dotted
        assert e["type"] in ("bool", "int", "float", "str", "list", "device")
        assert set(e) <= {"section", "key", "type", "default", "min", "max", "choices", "nullable", "secret", "format"}
    assert set(sc["defaults"]) == set(sc["keys"])


def test_schema_details():
    sc = ss.schema()
    k = sc["keys"]
    assert k["audio.gain"] == {"section": "audio", "key": "gain", "type": "float", "default": 1.0, "min": 0.05, "max": 50.0}
    assert k["audio.codec"]["choices"] == ["opus", "flac", "wav"] and sc["choices"]["audio.codec"] == ["opus", "flac", "wav"]
    assert k["audio.device"] == {"section": "audio", "key": "device", "type": "device", "default": None, "nullable": True}
    assert k["audio.min_speech_dbfs"]["nullable"] is True and k["audio.min_speech_dbfs"]["type"] == "float"
    assert {"audio.device", "audio.min_speech_dbfs", "audio.system_device"} <= set(sc["nullable"]) and sc["nullable"] == sorted(sc["nullable"])
    assert k["storage.cleanup_enabled"]["type"] == "bool" and k["ui.port"] == {"section": "ui", "key": "port", "type": "int",
                                                                              "default": 8765, "min": 1024, "max": 65535}
    assert sc["ranges"]["ui.port"] == [1024, 65535] and sc["ranges"]["audio.vad_threshold"] == [0.0, 1.0]
    assert k["push.channels"]["type"] == "list" and k["push.channels"]["choices"] == list(ss._PUSH_CHANNELS)
    assert k["push.ntfy.server"]["format"] == "url" and k["mimo.base_url"]["format"] == "url"
    assert k["analysis.daily_time"]["format"] == "HH:MM"
    assert k["push.wxpusher.uids"] == {"section": "push", "key": "wxpusher.uids", "type": "list", "default": []}
    # secrets: flagged, and their defaults are blank whatever config.DEFAULTS says
    assert sc["secret"] == sorted(".".join(p) for p in ss.SECRET_KEYS)
    assert all(k[s]["secret"] is True and k[s]["default"] == "" and sc["defaults"][s] == "" for s in sc["secret"])


def test_schema_agrees_with_coerce_setting():
    """The data the page gets and the validation the server runs cannot drift apart: same tables."""
    sc = ss.schema()
    for dotted, e in sc["keys"].items():
        path = tuple(dotted.split("."))
        if e.get("secret"):
            continue
        assert ss.coerce_setting(path, e["default"]) == e["default"], dotted        # every default is a valid value
        if "min" in e:
            assert ss.coerce_setting(path, e["min"]) == e["min"] and ss.coerce_setting(path, e["max"]) == e["max"]
            for outside in (e["min"] - 1, e["max"] + 1):
                with pytest.raises(HTTPException):
                    ss.coerce_setting(path, outside)
        if "choices" in e and e["type"] == "str":
            assert all(ss.coerce_setting(path, c) == c for c in e["choices"])
            with pytest.raises(HTTPException):
                ss.coerce_setting(path, "definitely-not-a-choice")
        if e.get("nullable"):
            assert ss.coerce_setting(path, "") is None and ss.coerce_setting(path, None) is None
        if e["type"] == "bool":
            assert ss.coerce_setting(path, "false") is False and ss.coerce_setting(path, "true") is True


def test_original_tables_are_unchanged():
    """The pre-split rules moved here verbatim (spot checks of every kind of rule; later packages only add keys)."""
    assert "enabled" in ss.EDITABLE["correction"] and "level" in ss.EDITABLE["log"] and "rpm" not in ss.EDITABLE["mimo"]
    assert {("mimo", "api_key"), ("push", "feishu", "webhook"), ("push", "feishu", "secret"), ("push", "wecom", "webhook"),
            ("push", "serverchan", "sendkey"), ("push", "pushplus", "token"), ("push", "wxpusher", "spt"),
            ("push", "wxpusher", "app_token")} <= ss.SECRET_KEYS
    assert ss._CHOICES[("memory", "retrieval")] == ("fts", "vector", "hybrid")
    assert ss._RANGES[("speakers", "min_label_sample_s")] == (1.0, 60.0) and ss._RANGES[("ui", "port")] == (1024, 65535)
    assert ("audio", "min_speech_dbfs") in ss._NULLABLE
    c = ss.coerce_setting
    assert c(("push", "channels"), "toast， ntfy,toast") == ["toast", "ntfy"]
    assert c(("audio", "device"), " 3 ") == 3 and c(("audio", "device"), "USB 麦克风") == "USB 麦克风" and c(("audio", "device"), "") is None
    assert c(("storage", "keep_audio_days"), "15") == 15 and c(("audio", "vad_threshold"), "0.6") == 0.6
    assert c(("audio", "min_speech_dbfs"), " ") is None and c(("mimo", "asr_language"), " zh ") == "zh"
    for path, bad, msg in ((("mimo", "rpm_nope"), 1, "不允许修改 mimo.rpm_nope"),
                           (("push", "channels"), "feishu,nope", "未知推送渠道: nope（可选 toast, feishu, wecom, serverchan, pushplus, wxpusher, ntfy）"),
                           (("audio", "device"), True, "audio.device 应为设备序号或名称"),
                           (("correction", "enabled"), "maybe", "correction.enabled 应为 true/false"),
                           (("ui", "port"), 80, "ui.port 应在 1024–65535 之间"),
                           (("storage", "keep_audio_days"), 1.5, "storage.keep_audio_days 应为整数"),
                           (("audio", "vad_threshold"), "abc", "audio.vad_threshold 应为数字"),
                           (("mimo", "chat_model"), {"x": 1}, "mimo.chat_model 应为文本"),
                           (("mimo", "asr_language"), "fr", "mimo.asr_language 只能是 auto / zh / en"),
                           (("mimo", "base_url"), "ftp://x", "mimo.base_url 应以 http:// 或 https:// 开头")):
        with pytest.raises(HTTPException) as ex:
            c(path, bad)
        assert ex.value.status_code == 400 and ex.value.detail == msg, path


# --- through the API -----------------------------------------------------------------------------------------------------

def test_config_get_returns_the_schema_and_never_a_secret(client: TestClient, rt: Runtime):
    rt.cfg["mimo"]["api_key"] = "sk-THIS-MUST-NOT-LEAK-0123456789"
    rt.cfg["push"]["feishu"]["secret"] = "feishu-THIS-MUST-NOT-LEAK"
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert body["schema"] == ss.schema()
    assert set(body["values"]) == set(body["defaults"]) == set(body["schema"]["sections"])
    assert "MUST-NOT-LEAK" not in r.text
    assert body["secrets"]["mimo.api_key"] == {"set": True, "hint": "sk-…6789"} and body["values"]["mimo"]["api_key"] == ""
    # what the old page relied on is all still there
    for key in ("path", "exists", "values", "secrets", "profile", "profile_label", "base_config_path", "env_api_key", "defaults",
                "glossary_path", "data_dir", "models_dir"):
        assert key in body


def test_config_put_accepts_v1_keys_only_once_defaults_have_them(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "DEFAULTS", _defaults(with_v1=False))
    g = client.get("/api/config").json()
    assert "performance" not in g["values"] and "performance" not in g["defaults"] and "events" not in g["values"]["ui"]
    for body, path in (({"performance": {"mode": "eco"}}, "performance.mode"), ({"ui": {"events": False}}, "ui.events"),
                       ({"pipeline": {"realtime": False}}, "pipeline.realtime"), ({"mimo": {"asr_prompt": False}}, "mimo.asr_prompt")):
        r = client.put("/api/config", json=body)
        assert r.status_code == 400 and r.json() == {"error": f"不允许修改 {path}"}, body
    assert not rt.config_path.exists() or "performance" not in rt.config_path.read_text(encoding="utf-8")
    monkeypatch.setattr(config, "DEFAULTS", _defaults(with_v1=True))
    g = client.get("/api/config").json()
    assert g["defaults"]["performance"]["mode"] == "balanced" and "events" in g["values"]["ui"]
    r = client.put("/api/config", json={"performance": {"mode": "eco", "threads": "2"}, "ui": {"events": "false"},
                                        "mimo": {"asr_prompt_max_terms": 80}})
    assert r.status_code == 200, r.text
    assert r.json()["saved"] == ["mimo.asr_prompt_max_terms", "performance.mode", "performance.threads", "ui.events"]
    written = yaml.safe_load(rt.config_path.read_text(encoding="utf-8"))
    assert written["performance"] == {"mode": "eco", "threads": 2} and written["ui"]["events"] is False
    assert written["mimo"]["asr_prompt_max_terms"] == 80
    r = client.put("/api/config", json={"performance": {"mode": "turbo"}})
    assert r.status_code == 400 and r.json() == {"error": "performance.mode 只能是 eco / balanced / fast"}


def test_saving_everything_the_page_shows_changes_nothing(client: TestClient, rt: Runtime):
    """GET -> PUT round trip with the live DEFAULTS: every value the server hands out passes its own validation."""
    g = client.get("/api/config").json()
    r = client.put("/api/config", json={sec: dict(vals) for sec, vals in g["values"].items()})
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "changed": [], "notes": ["没有需要保存的修改"], "saved": []}
