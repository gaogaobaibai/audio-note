"""Configuration loading: defaults merged with config.yaml.

Usage:
    from audio_note.config import load_config
    cfg = load_config()            # finds config.yaml next to the project root (or $AUDIO_NOTE_CONFIG)
    cfg.mimo.api_key, cfg.audio.sample_rate, cfg["audio"]["sample_rate"]
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, Any] = {
    "paths": {
        "data_dir": "data",            # relative to project root unless absolute
        "models_dir": "models",
    },
    "mimo": {
        "api_key": "",
        "base_url": "https://api.xiaomimimo.com/v1",
        "chat_model": "mimo-v2.6-flash",     # cheap default; mimo-v2.6-pro for higher quality
        "analysis_model": "",                 # empty = chat_model; e.g. mimo-v2.6-pro for the daily digest
        "asr_model": "mimo-v2.5-asr",
        "asr_language": "auto",        # auto | zh | en
        "asr_max_chunk_s": 60,         # ASR quality degrades past ~2 min per request; split at quiet points
        "rpm": 90,                     # client-side request rate cap (API limit is 100 RPM)
        "timeout_s": 120,
        "max_retries": 3,
        "temperature": 0.2,
    },
    "audio": {
        "device": None,                # None = system default input; or device index; or name substring
        "sample_rate": 16000,
        "channels": 1,
        "block_ms": 32,                # 512 samples @16k = silero window
        "vad_threshold": 0.5,
        "min_silence_s": 2.0,          # end a segment after this much silence (trailing silence is trimmed)
        "min_speech_s": 0.3,
        "max_segment_s": 300,          # hard cut so single files stay ASR-friendly
        "pre_roll_s": 0.5,             # audio kept before speech onset
        "post_roll_s": 0.5,
        "codec": "opus",               # opus | flac | wav
        "opus_bitrate": 24000,
        "gain": 1.0,
    },
    "storage": {
        "cleanup_enabled": True,
        "cleanup_interval_min": 30,
        "min_speech_ratio": 0.15,      # delete segment files whose recheck finds less speech than this
        "min_speech_total_s": 1.0,
        "keep_audio_days": 30,         # 0 = keep forever; transcripts are never deleted
        "max_disk_gb": 20,
    },
    "speakers": {
        "embedding_model": "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
        "segmentation_model": "sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
        "vad_model": "silero_vad.onnx",
        "num_threads": 4,
        "identify_threshold": 0.45,    # cosine similarity to call a cluster a known speaker
        "cluster_threshold": 0.5,      # sherpa-onnx fast clustering threshold
        "min_turn_s": 0.3,
        "unknown_label": "未知",
        "auto_enroll_from_diarization": False,
    },
    "correction": {
        "enabled": True,
        "glossary_file": "glossary.txt",   # one term per line, injected into the prompt
        "max_chars_per_call": 3000,
    },
    "pipeline": {
        "process_interval_min": 10,    # how often to pick up new segments for ASR
        "min_segment_age_s": 20,       # don't touch files still being written
        "transcripts_dir": "transcripts",
    },
    "memory": {
        "top_k": 20,
        "max_context_chars": 16000,
        "max_tool_rounds": 8,          # agentic retrieval loop budget per question
        "chat_model": "",              # empty = mimo.chat_model
    },
    "analysis": {
        "daily_time": "22:30",
        "push_daily": True,
        "lookback_days": 7,
    },
    "push": {
        "channels": ["toast"],         # any of: toast, feishu, wecom, serverchan, pushplus, wxpusher, ntfy
        "feishu": {"webhook": "", "secret": ""},
        "wecom": {"webhook": ""},
        "serverchan": {"sendkey": ""},
        "pushplus": {"token": ""},
        "wxpusher": {"spt": "", "app_token": "", "uids": []},
        "ntfy": {"topic": "", "server": "https://ntfy.sh"},
    },
    "log": {"level": "INFO", "file": "audio_note.log"},
}


class Section(dict):
    """dict with attribute access, recursively."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Section(v) if isinstance(v, dict) and not isinstance(v, Section) else v

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def find_config_path(explicit: str | os.PathLike | None = None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get("AUDIO_NOTE_CONFIG")
    if env:
        return Path(env)
    return PROJECT_ROOT / "config.yaml"


class Config(Section):
    @property
    def data_dir(self) -> Path:
        return self._abs(self["paths"]["data_dir"])

    @property
    def models_dir(self) -> Path:
        return self._abs(self["paths"]["models_dir"])

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / self["pipeline"]["transcripts_dir"]

    @property
    def daily_dir(self) -> Path:
        return self.data_dir / "daily"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "audio_note.sqlite3"

    @property
    def speakers_dir(self) -> Path:
        return self.data_dir / "speakers"

    @property
    def glossary_path(self) -> Path:
        return self._abs(self["correction"]["glossary_file"])

    def model_path(self, key: str) -> Path:
        p = Path(self["speakers"][key])
        return p if p.is_absolute() else self.models_dir / p

    def _abs(self, p: str | os.PathLike) -> Path:
        p = Path(p)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.audio_dir, self.transcripts_dir, self.speakers_dir, self.daily_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_config(path: str | os.PathLike | None = None) -> Config:
    p = find_config_path(path)
    user: dict = {}
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    merged = _merge(DEFAULTS, user)
    # env override for the secret so it never has to live in the file
    env_key = os.environ.get("MIMO_API_KEY")
    if env_key:
        merged["mimo"]["api_key"] = env_key
    cfg = Config(merged)
    cfg["_config_path"] = str(p)
    return cfg
