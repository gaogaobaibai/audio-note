"""What the settings page may change, and how each value is validated (GET / PUT /api/config).

    EDITABLE        section -> keys on the whitelist (anything else is rejected)
    SECRET_KEYS     paths never returned to the page; an empty value on PUT keeps what is there
    _CHOICES / _RANGES / _NULLABLE   per-path constraints
    editable()      EDITABLE restricted to the keys config.DEFAULTS knows RIGHT NOW
    coerce_setting(path, value)      validate / coerce one leaf against the type of its DEFAULTS entry
    schema()        the same rules as data, so the page does not duplicate them

Adding a setting (any package): add the key to config.DEFAULTS + config.example.yaml, then to EDITABLE here
(plus _RANGES / _CHOICES when it has limits; SECRET_KEYS when it is a secret). A key listed here that
config.DEFAULTS does not (yet) contain is simply not offered and not accepted, so this file may be ahead of
config.py - the type of the default drives the validation, there is nothing to validate against without one.
"""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from .. import config as _config

#: config sections / keys the settings page may change (anything else is rejected)
EDITABLE: dict[str, set[str]] = {
    "mimo": {"api_key", "base_url", "chat_model", "analysis_model", "asr_model", "asr_language", "asr_concurrency",
             "listen_model", "asr_prompt", "asr_prompt_max_terms"},
    "audio": {"device", "vad_threshold", "min_silence_s", "min_speech_s", "max_segment_s", "gain", "codec",
              "system_audio", "system_device", "system_gain", "min_speech_dbfs"},
    "storage": {"cleanup_enabled", "keep_audio_days", "max_disk_gb"},
    "speakers": {"identify_threshold", "cluster_threshold", "unknown_label", "num_threads",
                 "voice_group_threshold", "infer_relations", "relation_min_utterances", "relation_min_confidence",
                 "sample_consistency_min", "turn_check_min_s", "turn_reject_margin", "min_label_sample_s",
                 "negative_margin", "system_turn_share",
                 "scoring", "sure_norm", "guess_norm", "adapt", "smooth"},
    "correction": {"enabled", "mode"},
    "inbox": {"daily_max"},
    "pipeline": {"process_interval_min", "min_segment_age_s", "realtime"},
    "performance": {"mode", "threads", "background_priority", "defer_on_battery"},
    "memory": {"strategy", "retrieval", "query_expansion", "prefetch_k", "chat_model", "max_tool_rounds", "embedding_model",
               "session_gap_min", "hybrid_keywords"},
    "analysis": {"daily_time", "push_daily", "session_gap_min", "catchup_days"},
    "push": {"channels", "feishu", "wecom", "serverchan", "pushplus", "wxpusher", "ntfy"},
    "ui": {"port", "open_browser", "events"},
    "log": {"level"},
}
SECRET_KEYS = {("mimo", "api_key"), ("push", "feishu", "webhook"), ("push", "feishu", "secret"), ("push", "wecom", "webhook"),
               ("push", "serverchan", "sendkey"), ("push", "pushplus", "token"), ("push", "wxpusher", "spt"),
               ("push", "wxpusher", "app_token")}


_CHOICES: dict[tuple[str, ...], tuple[str, ...]] = {
    ("mimo", "asr_language"): ("auto", "zh", "en"),
    ("audio", "codec"): ("opus", "flac", "wav"),
    ("memory", "strategy"): ("agentic", "prefetch", "rag"),
    ("memory", "retrieval"): ("fts", "vector", "hybrid"),
    ("memory", "hybrid_keywords"): ("content", "strict", "loose", "off"),
    ("log", "level"): ("DEBUG", "INFO", "WARNING", "ERROR"),
    ("performance", "mode"): ("eco", "balanced", "fast"),
    ("correction", "mode"): ("session", "segment", "off"),
    ("speakers", "scoring"): ("v5", "legacy"),
}
_RANGES: dict[tuple[str, ...], tuple[float, float]] = {
    ("audio", "vad_threshold"): (0.0, 1.0), ("audio", "min_silence_s"): (0.2, 60.0), ("audio", "min_speech_s"): (0.0, 10.0),
    ("audio", "max_segment_s"): (10.0, 3600.0), ("audio", "gain"): (0.05, 50.0), ("audio", "system_gain"): (0.05, 50.0),
    ("audio", "min_speech_dbfs"): (-90.0, 0.0),
    ("storage", "keep_audio_days"): (0, 100000), ("storage", "max_disk_gb"): (0, 100000),
    # num_threads: 0 = follow performance.mode / performance.threads (SPEC v1 section 4.1)
    ("speakers", "identify_threshold"): (0.0, 1.0), ("speakers", "cluster_threshold"): (0.0, 1.0), ("speakers", "num_threads"): (0, 64),
    ("speakers", "voice_group_threshold"): (0.0, 1.0), ("speakers", "relation_min_utterances"): (1, 1000),
    ("speakers", "relation_min_confidence"): (0.0, 1.0), ("speakers", "sample_consistency_min"): (0.0, 1.0),
    ("speakers", "turn_check_min_s"): (0.0, 60.0), ("speakers", "turn_reject_margin"): (0.0, 1.0),
    ("speakers", "min_label_sample_s"): (1.0, 60.0), ("speakers", "negative_margin"): (-1.0, 1.0),
    ("speakers", "system_turn_share"): (0.0, 1.0),
    ("speakers", "sure_norm"): (0.05, 0.8), ("speakers", "guess_norm"): (0.0, 0.8),
    ("pipeline", "process_interval_min"): (1, 1440), ("pipeline", "min_segment_age_s"): (0, 3600),
    ("mimo", "asr_concurrency"): (1, 32), ("memory", "prefetch_k"): (1, 100), ("memory", "max_tool_rounds"): (0, 50),
    ("memory", "session_gap_min"): (1, 240),
    ("analysis", "session_gap_min"): (1, 1440), ("analysis", "catchup_days"): (1, 365), ("ui", "port"): (1024, 65535),
    ("performance", "threads"): (0, 64), ("mimo", "asr_prompt_max_terms"): (0, 500),
    ("inbox", "daily_max"): (0, 200),
}
_PUSH_CHANNELS = ("toast", "feishu", "wecom", "serverchan", "pushplus", "wxpusher", "ntfy")
#: numeric settings that may also be null (= feature off); an empty form field means null
_NULLABLE = {("audio", "min_speech_dbfs")}
#: device selectors: null (system default), a device index, or a name substring
_DEVICE_KEYS = (("audio", "device"), ("audio", "system_device"))
#: text settings that must be an http(s) URL when not empty
_URL_KEYS = (("mimo", "base_url"), ("push", "ntfy", "server"))


def _defaults() -> dict[str, Any]:
    """config.DEFAULTS as it is right now (read through the module: another package may extend it)."""
    return _config.DEFAULTS


def _default_for(path: tuple[str, ...]) -> Any:
    node: Any = _defaults()
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return _NO_DEFAULT
        node = node[k]
    return node


_NO_DEFAULT = object()


def editable() -> dict[str, set[str]]:
    """EDITABLE restricted to what config.DEFAULTS knows at this moment: {section: {key, ...}} without empty
    sections. A whitelisted key whose default has not landed in config.py is neither offered nor accepted."""
    defaults = _defaults()
    out: dict[str, set[str]] = {}
    for section, keys in EDITABLE.items():
        sec = defaults.get(section)
        if not isinstance(sec, dict):
            continue
        known = {k for k in keys if k in sec}
        if known:
            out[section] = known
    return out


def is_editable(section: str, key: str) -> bool:
    return key in editable().get(section, ())


def range_for(path: tuple[str, ...]) -> tuple[float, float] | None:
    """(lo, hi) of a numeric setting, or None. speakers.num_threads may be 0 (= follow performance.*) only
    once config.DEFAULTS has the performance section that gives 0 a meaning."""
    r = _RANGES.get(path)
    if r is not None and path == ("speakers", "num_threads") and not isinstance(_defaults().get("performance"), dict):
        return (max(1, r[0]), r[1])
    return r


def coerce_setting(path: tuple[str, ...], value: Any) -> Any:
    """Validate / coerce one config leaf against the type of its DEFAULTS entry (form values arrive as
    strings or JSON scalars). Raises HTTPException(400) with a readable message."""
    dotted = ".".join(path)
    default = _default_for(path)
    if default is _NO_DEFAULT:
        raise HTTPException(400, f"不允许修改 {dotted}")
    if isinstance(default, dict):
        raise HTTPException(400, f"{dotted} 是一组设置，不能整个替换")
    if path == ("push", "channels"):
        if isinstance(value, str):
            value = [v.strip() for v in value.replace("，", ",").split(",") if v.strip()]
        if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
            raise HTTPException(400, "push.channels 必须是渠道名列表")
        bad = [v for v in value if v not in _PUSH_CHANNELS]
        if bad:
            raise HTTPException(400, f"未知推送渠道: {', '.join(bad)}（可选 {', '.join(_PUSH_CHANNELS)}）")
        return list(dict.fromkeys(value))
    if isinstance(default, list):
        if isinstance(value, str):
            value = [v.strip() for v in value.replace("，", ",").split(",") if v.strip()]
        if not isinstance(value, list):
            raise HTTPException(400, f"{dotted} 必须是列表")
        return [str(v) for v in value]
    if path in _NULLABLE and (value is None or (isinstance(value, str) and not value.strip())):
        return None
    if path in _DEVICE_KEYS:
        if value is None or value == "":
            return None
        if isinstance(value, bool) or isinstance(value, (dict, list)):
            raise HTTPException(400, f"{dotted} 应为设备序号或名称")
        if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
            return int(value)
        return str(value)
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false", "1", "0", "yes", "no", "on", "off"):
            return value.strip().lower() in ("true", "1", "yes", "on")
        raise HTTPException(400, f"{dotted} 应为 true/false")
    if isinstance(default, int) and not isinstance(default, bool):
        if isinstance(value, bool) or isinstance(value, (dict, list)):
            raise HTTPException(400, f"{dotted} 应为整数")
        try:
            iv = int(float(value)) if not isinstance(value, int) else value
        except (TypeError, ValueError):
            raise HTTPException(400, f"{dotted} 应为整数")
        if isinstance(value, float) and abs(value - iv) > 1e-9:
            raise HTTPException(400, f"{dotted} 应为整数")
        value = iv
    elif isinstance(default, float):
        if isinstance(value, bool) or isinstance(value, (dict, list)):
            raise HTTPException(400, f"{dotted} 应为数字")
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise HTTPException(400, f"{dotted} 应为数字")
    elif isinstance(default, str) or default is None:
        if value is None and default is None:
            return None
        if isinstance(value, (dict, list, bool)):
            raise HTTPException(400, f"{dotted} 应为文本")
        value = str(value).strip() if not isinstance(value, str) else value.strip()
        if path in _CHOICES and value not in _CHOICES[path]:
            raise HTTPException(400, f"{dotted} 只能是 {' / '.join(_CHOICES[path])}")
        if path in _URL_KEYS and value and not value.startswith(("http://", "https://")):
            raise HTTPException(400, f"{dotted} 应以 http:// 或 https:// 开头")
        return value
    rng = range_for(path)
    if rng is not None:
        lo, hi = rng
        if not (lo <= value <= hi):
            raise HTTPException(400, f"{dotted} 应在 {lo:g}–{hi:g} 之间")
    return value


def _effective(cfg: Any, path: tuple[str, ...]) -> Any:
    """Current value of a path in the live config (_NO_DEFAULT when absent)."""
    node: Any = cfg
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return _NO_DEFAULT
        node = node[k]
    return node


# ---------------------------------------------------------------------------------------------- schema as data
def _type_name(path: tuple[str, ...], default: Any) -> str:
    if path in _DEVICE_KEYS:
        return "device"
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, list):
        return "list"
    return "str"


def _leaves() -> list[tuple[tuple[str, ...], Any]]:
    """Every editable leaf path with its default (the push.<channel> groups are expanded one level)."""
    defaults = _defaults()
    out: list[tuple[tuple[str, ...], Any]] = []
    for section, keys in editable().items():
        for key in sorted(keys):
            d = defaults[section][key]
            if isinstance(d, dict):
                out.extend(((section, key, str(k2)), v2) for k2, v2 in d.items() if not isinstance(v2, dict))
            else:
                out.append(((section, key), d))
    return out


def schema() -> dict[str, Any]:
    """The validation rules of coerce_setting as data (GET /api/config -> "schema"), keyed by dotted path:

        sections   editable section names, in display order
        keys       {"audio.gain": {"section", "key", "type", "default", "min"?, "max"?, "choices"?, "nullable"?,
                                   "secret"?, "format"?}, ...}     type: bool | int | float | str | list | device
        choices    {"audio.codec": ["opus", "flac", "wav"], ...}
        ranges     {"audio.gain": [0.05, 50.0], ...}
        nullable   ["audio.device", ...]          an empty field means null
        secret     ["mimo.api_key", ...]          never returned; an empty field keeps the stored value
        defaults   {"audio.gain": 1.0, ...}       (secrets: always "")
    """
    keys: dict[str, dict[str, Any]] = {}
    for path, default in _leaves():
        dotted = ".".join(path)
        secret = path in SECRET_KEYS
        e: dict[str, Any] = {"section": path[0], "key": ".".join(path[1:]), "type": _type_name(path, default),
                             "default": "" if secret else default}
        if path == ("push", "channels"):
            e["choices"] = list(_PUSH_CHANNELS)
        elif path in _CHOICES:
            e["choices"] = list(_CHOICES[path])
        rng = range_for(path)
        if rng is not None:
            e["min"], e["max"] = rng
        if path in _NULLABLE or path in _DEVICE_KEYS:
            e["nullable"] = True
        if secret:
            e["secret"] = True
        if path in _URL_KEYS:
            e["format"] = "url"
        elif path == ("analysis", "daily_time"):
            e["format"] = "HH:MM"
        keys[dotted] = e
    known = editable()
    return {
        "sections": [s for s in EDITABLE if s in known],
        "keys": keys,
        "choices": {k: e["choices"] for k, e in keys.items() if "choices" in e},
        "ranges": {k: [e["min"], e["max"]] for k, e in keys.items() if "min" in e},
        "nullable": sorted(k for k, e in keys.items() if e.get("nullable")),
        "secret": sorted(k for k, e in keys.items() if e.get("secret")),
        "defaults": {k: e["default"] for k, e in keys.items()},
    }
