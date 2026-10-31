"""Tests for audio_note.config: example file matches DEFAULTS, _merge semantics, env/explicit path overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from audio_note import config as cfgmod
from audio_note.config import DEFAULTS, Config, _merge, load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = PROJECT_ROOT / "config.example.yaml"


def _assert_keys_subset(sub: dict, ref: dict, path: str = "") -> None:
    for k, v in sub.items():
        here = f"{path}.{k}" if path else k
        assert k in ref, f"unknown key {here} (not in DEFAULTS)"
        if isinstance(v, dict):
            assert isinstance(ref[k], dict), f"{here} is a mapping in the example but not in DEFAULTS"
            _assert_keys_subset(v, ref[k], here)


def _flatten(d: dict, path: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        here = f"{path}.{k}" if path else k
        if isinstance(v, dict):
            out.update(_flatten(v, here))
        else:
            out[here] = v
    return out


def test_example_loads_and_keys_are_subset_of_defaults() -> None:
    with open(EXAMPLE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict) and data
    _assert_keys_subset(data, DEFAULTS)
    # every top-level section of DEFAULTS is documented in the example
    assert set(data) == set(DEFAULTS)


def test_example_values_match_defaults_except_api_key() -> None:
    with open(EXAMPLE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    flat_ex = _flatten(data)
    flat_def = _flatten(DEFAULTS)
    # the shipped example must not contain a non-empty placeholder: every "missing key" guard
    # (`init` hint, `status`, MimoClient) keys off an empty string
    assert flat_ex["mimo.api_key"] == ""
    assert set(flat_ex) == set(flat_def), "example must list every leaf key of DEFAULTS"
    for key, val in flat_ex.items():
        if key == "mimo.api_key":
            continue
        assert val == flat_def[key], f"{key}: example={val!r} default={flat_def[key]!r}"
        assert type(val) is type(flat_def[key]), f"{key}: type {type(val)} != {type(flat_def[key])}"


def test_merge_is_recursive_and_non_destructive() -> None:
    base = {"a": {"x": 1, "y": {"z": 2}}, "b": [1, 2], "c": "keep"}
    snapshot = copy.deepcopy(base)
    out = _merge(base, {"a": {"y": {"z": 3, "w": 4}}, "b": [9], "d": None})
    assert out == {"a": {"x": 1, "y": {"z": 3, "w": 4}}, "b": [9], "c": "keep", "d": None}
    assert base == snapshot  # input untouched
    out["a"]["y"]["z"] = 100
    assert base["a"]["y"]["z"] == 2  # deep copy, no aliasing
    # dict in override replaces a non-dict in base, and vice versa
    assert _merge({"a": 1}, {"a": {"b": 2}}) == {"a": {"b": 2}}
    assert _merge({"a": {"b": 2}}, {"a": 1}) == {"a": 1}
    # None / empty override returns a copy of base
    assert _merge(base, None) == base  # type: ignore[arg-type]
    assert _merge(base, {}) == base


def test_load_config_uses_env_path_and_merges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    p = tmp_path / "cfg.yaml"
    p.write_text(
        f"paths:\n  data_dir: {data_dir.as_posix()}\nmimo:\n  api_key: from-file\naudio:\n  vad_threshold: 0.7\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AUDIO_NOTE_CONFIG", str(p))
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg["_config_path"] == str(p)
    assert cfg.mimo.api_key == "from-file"
    assert cfg.audio.vad_threshold == 0.7
    assert cfg.audio.sample_rate == DEFAULTS["audio"]["sample_rate"]  # untouched default
    assert cfg.data_dir == data_dir
    assert cfg.db_path == data_dir / "audio_note.sqlite3"
    assert cfg.transcripts_dir == data_dir / "transcripts"
    assert cfg.models_dir == cfgmod.PROJECT_ROOT / "models"
    assert cfg.model_path("vad_model") == cfgmod.PROJECT_ROOT / "models" / "silero_vad.onnx"


def test_env_mimo_api_key_overrides_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(f"paths:\n  data_dir: {(tmp_path / 'd').as_posix()}\nmimo:\n  api_key: from-file\n", encoding="utf-8")
    monkeypatch.setenv("AUDIO_NOTE_CONFIG", str(p))
    monkeypatch.setenv("MIMO_API_KEY", "from-env")
    cfg = load_config()
    assert cfg.mimo.api_key == "from-env"
    # empty env var does not override
    monkeypatch.setenv("MIMO_API_KEY", "")
    assert load_config().mimo.api_key == "from-file"


def test_missing_file_gives_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "nope.yaml"
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    cfg = load_config(missing)
    assert cfg["_config_path"] == str(missing)
    for section, body in DEFAULTS.items():
        assert dict(cfg[section]) == body
    assert cfg.mimo.api_key == ""


def test_explicit_path_beats_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("log:\n  level: DEBUG\n", encoding="utf-8")
    b.write_text("log:\n  level: ERROR\n", encoding="utf-8")
    monkeypatch.setenv("AUDIO_NOTE_CONFIG", str(a))
    assert load_config().log.level == "DEBUG"
    assert load_config(b).log.level == "ERROR"


def test_section_attribute_access() -> None:
    cfg = Config(copy.deepcopy(DEFAULTS))
    assert cfg.push.feishu.webhook == ""
    assert cfg["push"]["channels"] == ["toast"]
    with pytest.raises(AttributeError):
        _ = cfg.push.no_such_key
    # top-level attribute writes land in the dict
    cfg.extra = 1
    assert cfg["extra"] == 1
    # nested sections are wrapped on each access (a shallow copy), so nested attribute
    # writes do NOT propagate; mutate via item access instead
    cfg.log.level = "WARNING"
    assert cfg.log.level == "INFO"
    cfg["log"]["level"] = "WARNING"
    assert cfg.log.level == "WARNING"
