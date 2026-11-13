"""Settings page: devices, config.yaml, connectivity tests, the glossary file.

    GET  /api/devices            GET /api/devices/system
    GET  /api/config             PUT /api/config {section: {key: value | {subkey: value}}}
    POST /api/config/test-api    POST /api/push/test {text?}
    GET  /api/glossary           PUT /api/glossary {text, reindex?}

What may be changed and how each value is validated lives in ui/settings_schema.py; GET /api/config also
returns those rules as data ("schema") so the page does not keep its own copy.
"""
from __future__ import annotations

import copy
import logging
import os
import time
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from ... import config as config_mod
from ...config import is_placeholder_key
from ...runtime import JobError, ReadOnly, profile_label
from .. import settings_schema, yaml_edit
from ..deps import RT, bool_arg
from ..serializers import hint

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------------------------- devices
@router.get("/devices")
def devices() -> list[dict[str, Any]]:
    try:
        from ...audio.capture import list_input_devices

        return [{"index": d.index, "name": d.name, "hostapi": d.hostapi, "channels": d.channels, "default_sr": d.default_sr}
                for d in list_input_devices()]
    except Exception as e:  # noqa: BLE001
        log.warning("list devices failed: %s", e)
        return []


@router.get("/devices/system")
def system_devices() -> dict[str, Any]:
    try:
        from ...audio.loopback import available, list_loopback_devices

        ok = available()
        devs = list_loopback_devices() if ok else []
        return {"available": ok, "devices": [{"index": d.index, "name": d.name, "channels": d.channels,
                                               "default_sr": d.default_sr, "is_default": d.is_default} for d in devs]}
    except Exception as e:  # noqa: BLE001
        log.warning("list system devices failed: %s", e)
        return {"available": False, "devices": [], "error": str(e)}


# ---------------------------------------------------------------------------------------------- config
@router.get("/config")
def config_get(rt: RT) -> dict[str, Any]:
    cfg = rt.cfg
    defaults = config_mod.DEFAULTS
    editable = settings_schema.editable()
    values: dict[str, Any] = {}
    for section, keys in editable.items():
        sec = cfg.get(section) or {}
        # deep copy: the nested push.* dicts are the live config's own objects and get their secrets blanked below
        values[section] = copy.deepcopy({k: sec.get(k, defaults.get(section, {}).get(k)) for k in sorted(keys)})
    values.setdefault("mimo", {})["api_key"] = ""
    secrets: dict[str, Any] = {}
    for path in settings_schema.SECRET_KEYS:
        cur: Any = cfg
        for k in path:
            cur = (cur or {}).get(k) if isinstance(cur, dict) else None
        secrets[".".join(path)] = {"set": bool(cur), "hint": hint(cur) if cur else ""}
        # never ship the secret itself to the page
        node = values
        for k in path[:-1]:
            node = node.setdefault(k, {}) if isinstance(node.get(k), dict) or k not in node else node[k]
        if isinstance(node, dict):
            node[path[-1]] = ""
    return {"path": str(rt.config_path), "exists": rt.config_path.exists(), "values": values, "secrets": secrets,
            "profile": rt.profile, "profile_label": profile_label(rt.profile), "base_config_path": cfg.get("_base_config_path"),
            "env_api_key": bool(os.environ.get("MIMO_API_KEY")),
            "defaults": {s: {k: defaults[s].get(k) for k in keys} for s, keys in editable.items()},
            "glossary_path": str(cfg.glossary_path), "data_dir": str(cfg.data_dir), "models_dir": str(cfg.models_dir),
            "schema": settings_schema.schema()}


@router.put("/config")
def config_put(rt: RT, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    updates: dict[tuple[str, ...], Any] = {}
    if not isinstance(body, dict) or any(not isinstance(v, dict) for v in body.values()):
        raise HTTPException(400, "请求体应为 {小节: {键: 值}}")
    editable = settings_schema.editable()
    leaves: list[tuple[tuple[str, ...], Any]] = []
    for path, value in yaml_edit.flatten(body).items():
        if len(path) < 2:
            raise HTTPException(400, f"不允许修改 {'.'.join(path)}")
        section, key = path[0], path[1]
        if section not in editable or key not in editable[section]:
            raise HTTPException(400, f"不允许修改 {'.'.join(path)}")
        if isinstance(value, dict):
            if not isinstance(settings_schema._default_for(path), dict):
                raise HTTPException(400, f"{'.'.join(path)} 不是一组设置")
            for k2, v2 in value.items():
                leaves.append((path + (str(k2),), v2))
        else:
            leaves.append((path, value))
    for path, value in leaves:
        if path in settings_schema.SECRET_KEYS:
            if value is None or str(value).strip() == "":
                continue   # empty secret field = keep what is there
            if path == ("mimo", "api_key") and is_placeholder_key(value):
                raise HTTPException(400, "API key 看起来是占位符")
            if isinstance(value, (dict, list, bool)):
                raise HTTPException(400, f"{'.'.join(path)} 应为文本")
            updates[path] = str(value).strip()
            continue
        value = settings_schema.coerce_setting(path, value)
        if path == ("analysis", "daily_time"):
            from ...scheduler import parse_hhmm

            try:
                parse_hhmm(str(value))
            except ValueError:
                raise HTTPException(400, "日报时间格式应为 HH:MM")
        # unchanged values are not written: on a profile that would pin inherited settings into
        # profiles/<name>/config.yaml and silently cut them off from the shared root config.yaml
        if settings_schema._effective(rt.cfg, path) == value:
            continue
        updates[path] = value
    if not updates:
        return {"ok": True, "changed": [], "notes": ["没有需要保存的修改"], "saved": []}
    p = rt.config_path
    text = p.read_text(encoding="utf-8") if p.exists() else "# audio_note 配置（由网页界面生成）\n"
    res = yaml_edit.set_values(text, updates)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(res.text, encoding="utf-8")
    os.replace(tmp, p)
    info = rt.reload_config()
    notes = list(info["notes"])
    if res.comments_lost:
        notes.append("config.yaml 里的注释无法保留，文件已按结构重写（值都在）")
    return {"ok": True, "changed": info["changed"], "restarted_recorder": info["restarted_recorder"], "notes": notes,
            "saved": sorted(".".join(k) for k in updates)}


@router.post("/config/test-api")
def config_test_api(rt: RT) -> dict[str, Any]:
    client = rt.client()
    t0 = time.monotonic()
    resp = client.chat([{"role": "user", "content": "回复“ok”两个字母即可。"}], max_tokens=8)
    text = (resp.choices[0].message.content or "").strip()
    return {"ok": True, "model": client.chat_model, "reply": text[:50], "seconds": round(time.monotonic() - t0, 2)}


@router.post("/push/test")
def push_test(rt: RT, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    from ...push.base import Message, Notifier

    text = str(body.get("text") or "这是一条来自 audio-note 的测试消息。")
    n = Notifier(rt.cfg)
    if not n.channels:
        raise HTTPException(400, "没有配置可用的推送渠道（检查 push.channels 和对应的密钥）")
    results = n.send(Message("audio-note 测试", text))
    return {"ok": any(v is None for v in results.values()), "channels": results}


# ---------------------------------------------------------------------------------------------- glossary
# Compatibility with the pre-1.0 "glossary" text box: the words live in the lexicon now (speech.lexicon), and
# these two endpoints are its plain-text export / import. Speaker names are not part of the text.
@router.get("/glossary")
def glossary_get(rt: RT) -> dict[str, Any]:
    """The lexicon as glossary text: glossary.txt as the user wrote it (comments kept) without entries that
    were switched off, plus the entries learned elsewhere (answers, hand edits, the lexicon page)."""
    p = rt.cfg.glossary_path
    try:
        from ...speech import lexicon

        lexicon.sync_sources(rt.db, rt.cfg)
        text = lexicon.glossary_text(rt.db, rt.cfg)
    except Exception as e:  # noqa: BLE001 - the text box must keep working whatever the lexicon does
        log.warning("glossary export fell back to the file: %s: %s", type(e).__name__, e)
        text = p.read_text(encoding="utf-8") if p.exists() else ""
    return {"path": str(p), "text": text}


@router.put("/glossary")
def glossary_put(rt: RT, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Save the text box: the file is written as typed, and its terms become THE vocabulary - new ones are
    added to the lexicon, entries (other than speaker names) that are no longer listed are switched off."""
    from ...speech import glossary_file

    p = rt.cfg.glossary_path
    text = str(body.get("text") or "")
    reindex = bool_arg(body.get("reindex"), "reindex")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")
    terms = glossary_file.parse_glossary(text)
    try:
        from ...speech import lexicon

        lexicon.import_terms(rt.db, terms, source="glossary", replace=True, events=rt.events)
        lexicon.sync_sources(rt.db, rt.cfg)          # remembers the file as imported (and keeps speakers in step)
    except Exception as e:  # noqa: BLE001 - the file is saved; the next sync imports it
        log.warning("glossary saved but not imported into the lexicon: %s: %s", type(e).__name__, e)
    extra: dict[str, Any] = {}
    if reindex:
        try:
            st = rt.submit("reindex", vectors=False)
            extra["job"] = st.as_dict()
        except (JobError, ReadOnly) as e:
            extra["job_error"] = str(e)
    return {"ok": True, "terms": len(terms), **extra}
