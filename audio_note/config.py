"""Configuration loading: defaults merged with config.yaml.

Usage:
    from audio_note.config import load_config
    cfg = load_config()            # finds config.yaml next to the project root (or $AUDIO_NOTE_CONFIG)
    cfg.mimo.api_key, cfg.audio.sample_rate, cfg["audio"]["sample_rate"]
"""
from __future__ import annotations

import copy
import os
import re
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
        "asr_concurrency": 4,          # parallel ASR requests per segment (latency ~0.8x realtime each; API cap 100 RPM)
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
        "system_audio": False,         # also capture what the speakers / headphones play (meetings, calls): WASAPI loopback
        "system_device": None,         # None = default output device; or an output device index / name substring
        "system_gain": 1.0,            # level of the system audio in the mix
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
        "voice_group_threshold": 0.55,  # cosine threshold for grouping unknown voices across recordings (higher = finer groups)
        "infer_relations": True,        # after the daily digest, ask the LLM how each speaker relates to the user
        "relation_min_utterances": 15,  # a speaker needs at least this many utterances before relation inference
        "relation_min_confidence": 0.6, # inferred relations below this confidence are not written
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
        "retrieval": "hybrid",         # fts | vector | hybrid   (vector/hybrid need the local embedding model)
        "embedding_model": "BAAI/bge-small-zh-v1.5",   # fastembed model id (downloaded into models_dir/embeddings)
        "embedding_threads": 2,
        "query_expansion": False,      # ask the LLM for synonyms before keyword search (1 extra call per query)
        "strategy": "prefetch",        # agentic | prefetch | rag   (see docs/DESIGN.md §7)
        "prefetch_k": 12,              # hits injected before the first model turn (prefetch / rag)
        "rrf_k": 60,                   # reciprocal-rank-fusion constant for hybrid retrieval
        "fts_weight": 1.0,             # weight of the keyword list in hybrid fusion (vector list = 1.0)
        "hybrid_fts_strict": True,     # hybrid: keyword list = exact AND matches only (the loose OR list adds noise)
    },
    "analysis": {
        "daily_time": "22:30",
        "push_daily": True,
        "session_gap_min": 20,         # a gap longer than this splits the day into separate sessions
        "max_block_chars": 6000,       # max characters per map-step LLM call
        "catchup_days": 7,             # after downtime, how many missed days still get a digest
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
    "ui": {
        "enabled": True,               # serve the web UI from `audio-note run` (always on for `audio-note ui`)
        "host": "127.0.0.1",           # local only; never expose the UI (no auth) on a public interface
        "port": 8765,
        "open_browser": False,         # `audio-note run`: open the UI in the browser on start
    },
    "log": {"level": "INFO", "file": "audio_note.log"},
}


#: values that mean "not filled in yet" (the example / init templates and common habits)
API_KEY_PLACEHOLDERS = frozenset({
    "sk-填这里", "填这里", "sk-xxx", "sk-xxxx", "sk-...", "xxx", "your-key", "your_api_key", "your-api-key",
    "<your-key>", "<api-key>", "todo", "changeme", "none", "null",
})


def is_placeholder_key(key: Any) -> bool:
    """True for an empty value or a known placeholder, so every 'missing key' guard treats it as unset."""
    if key is None:
        return True
    s = str(key).strip()
    if not s:
        return True
    if s.lower() in API_KEY_PLACEHOLDERS or s in API_KEY_PLACEHOLDERS:
        return True
    return ("填" in s) or (s.startswith("<") and s.endswith(">"))


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


# --- profiles -------------------------------------------------------------------------------------
# A profile ("档案") is one person's / one purpose's data: profiles/<name>/config.yaml whose values are
# layered over the root config.yaml (shared settings such as the API key live there once) and whose
# data_dir defaults to profiles/<name>/data. The root config.yaml + data/ pair is the profile named
# DEFAULT_PROFILE. profiles/current remembers the last profile chosen (CLI `profile use`, UI switcher).
# Resolution order for `load_config()` with no explicit path:
#   AUDIO_NOTE_CONFIG (explicit file, no layering) > AUDIO_NOTE_PROFILE > profiles/current > default.

PROFILES_DIR = PROJECT_ROOT / "profiles"
CURRENT_PROFILE_FILE = PROFILES_DIR / "current"
DEFAULT_PROFILE = "default"
PROFILE_NAME_RE = re.compile(r"^[\w\-\u4e00-\u9fff]{1,32}$")
RESERVED_PROFILE_NAMES = frozenset({DEFAULT_PROFILE, "current", "custom", "data", "."})
#: Windows device names: a directory called NUL / COM1 cannot be created or deleted normally
_WINDOWS_DEVICE_NAMES = frozenset({"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))})


def valid_profile_name(name: str) -> bool:
    return (bool(name) and bool(PROFILE_NAME_RE.match(name)) and name not in RESERVED_PROFILE_NAMES
            and not name.startswith(".") and name.lower() not in _WINDOWS_DEVICE_NAMES)


def canonical_profile_name(name: str) -> str | None:
    """The existing profile whose name matches `name` ignoring case (NTFS is case-insensitive, so
    'Demo' and 'demo' would otherwise be two names for one directory); None when no profile matches."""
    name = (name or "").strip()
    if not name:
        return None
    if name == DEFAULT_PROFILE:
        return DEFAULT_PROFILE
    for existing in list_profiles():
        if existing.lower() == name.lower():
            return existing
    return None


def profile_dir(name: str) -> Path:
    return PROFILES_DIR / name


def profile_config_path(name: str) -> Path:
    """Config file of a profile; the default profile is the root config.yaml."""
    if name == DEFAULT_PROFILE:
        return PROJECT_ROOT / "config.yaml"
    return profile_dir(name) / "config.yaml"


def profile_data_dir(name: str) -> str:
    """Relative data_dir written into a new profile's config (relative to the project root)."""
    return f"profiles/{name}/data"


def list_profiles() -> list[str]:
    """Profile names: the default one first, then every profiles/<name>/config.yaml (sorted)."""
    names = [DEFAULT_PROFILE]
    if PROFILES_DIR.exists():
        for d in sorted(PROFILES_DIR.iterdir()):
            if d.is_dir() and (d / "config.yaml").exists() and valid_profile_name(d.name):
                names.append(d.name)
    return names


def profile_exists(name: str) -> bool:
    """True when a profile of exactly this name exists (case-sensitive; see canonical_profile_name)."""
    return name == DEFAULT_PROFILE or (valid_profile_name(name) and profile_config_path(name).exists()
                                       and name in list_profiles())


def profile_glossary_file(name: str) -> str:
    return f"profiles/{name}/glossary.txt"


def current_profile_name() -> str:
    """The remembered profile (AUDIO_NOTE_PROFILE, else profiles/current), falling back to the default."""
    env = os.environ.get("AUDIO_NOTE_PROFILE")
    if env:
        return env.strip()
    try:
        name = CURRENT_PROFILE_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return DEFAULT_PROFILE
    return canonical_profile_name(name) or DEFAULT_PROFILE


def set_current_profile(name: str) -> None:
    name = canonical_profile_name(name) or name
    if not profile_exists(name):
        raise ValueError(f"档案不存在: {name}")
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    CURRENT_PROFILE_FILE.write_text(name + "\n", encoding="utf-8")


#: keys never copied into a new profile by create_profile(copy_from=...): secrets stay in the root file
#: (one place to rotate them), paths / ui / log are per installation
_NOT_COPIED_SECTIONS = ("paths", "ui", "log")
_NOT_COPIED_KEYS = (("mimo", "api_key"), ("push", "feishu", "webhook"), ("push", "feishu", "secret"), ("push", "wecom", "webhook"),
                    ("push", "serverchan", "sendkey"), ("push", "pushplus", "token"), ("push", "wxpusher", "spt"),
                    ("push", "wxpusher", "app_token"), ("correction", "glossary_file"))


def create_profile(name: str, copy_from: str | os.PathLike | None = None) -> Path:
    """Create profiles/<name>/config.yaml (+ data/ and an empty glossary.txt of its own). Only
    paths.data_dir and correction.glossary_file are written: everything else is inherited from the root
    config.yaml. With `copy_from` (another profile's config file) its non-secret settings are copied as
    a starting point (secrets, paths, ui, log are left out so they keep coming from the root file) and
    its glossary file is copied too."""
    name = (name or "").strip()
    if not valid_profile_name(name):
        raise ValueError("档案名只能用中英文、数字、下划线、短横线（1–32 字），且不能叫 default/current 或 Windows 保留名")
    if canonical_profile_name(name):
        raise FileExistsError(f"档案已存在: {canonical_profile_name(name)}")
    d = profile_dir(name)
    cfg_path = d / "config.yaml"
    if cfg_path.exists():
        raise FileExistsError(f"档案已存在: {name}")
    try:
        d.mkdir(parents=True, exist_ok=True)
        (d / "data").mkdir(exist_ok=True)
    except OSError as e:
        raise ValueError(f"无法创建档案目录 {d}: {e}") from e
    own: dict = {"paths": {"data_dir": profile_data_dir(name)}, "correction": {"glossary_file": profile_glossary_file(name)}}
    glossary_src: Path | None = None
    if copy_from and Path(copy_from).exists():
        src = _read_yaml(Path(copy_from))
        for sec in _NOT_COPIED_SECTIONS:
            src.pop(sec, None)
        for path in _NOT_COPIED_KEYS:
            node: Any = src
            for k in path[:-1]:
                node = node.get(k) if isinstance(node, dict) else None
            if isinstance(node, dict):
                node.pop(path[-1], None)
        try:
            src_cfg = load_config(copy_from)
            if src_cfg.glossary_path.exists():
                glossary_src = src_cfg.glossary_path
        except Exception:  # noqa: BLE001
            glossary_src = None
        merged = _merge(src, own)
        text = (f"# audio_note 档案「{name}」的配置（从 {Path(copy_from).as_posix()} 复制，不含密钥 / 路径）：\n"
                f"# 没写的键继承项目根目录的 config.yaml。\n" + yaml.safe_dump(merged, allow_unicode=True, sort_keys=False))
    else:
        text = (f"# audio_note 档案「{name}」的配置：没写的键继承项目根目录的 config.yaml（API key 等共享设置放那里）。\n"
                f"paths:\n  data_dir: {profile_data_dir(name)}\n"
                f"correction:\n  glossary_file: {profile_glossary_file(name)}   # 这个档案自己的词表\n")
    cfg_path.write_text(text, encoding="utf-8", newline="\n")
    gl = d / "glossary.txt"
    if glossary_src is not None and glossary_src.exists():
        gl.write_text(glossary_src.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    elif not gl.exists():
        gl.write_text("# 人名 / 产品名 / 术语，每行一个；转写纠错和检索时使用\n", encoding="utf-8", newline="\n")
    return cfg_path


class ProfileInUse(OSError):
    """Some process still has files of the profile open (database, audio)."""


def delete_profile(name: str) -> Path:
    """Remove profiles/<name> entirely (audio, database, config). The default profile cannot be deleted.
    The directory is renamed first: on Windows the rename fails atomically while any file inside is open
    (another `audio-note ui`, a chat REPL), so a profile is either removed whole or left untouched."""
    import shutil
    import time

    name = canonical_profile_name(name) or name
    if name == DEFAULT_PROFILE or not valid_profile_name(name):
        raise ValueError("默认档案不能删除")
    d = profile_dir(name)
    if not (d / "config.yaml").exists():
        raise FileNotFoundError(f"档案不存在: {name}")
    trash = PROFILES_DIR / f".deleting-{name}-{int(time.time())}"
    try:
        os.rename(d, trash)
    except OSError as e:
        raise ProfileInUse(f"档案 {name} 的文件正被别的程序使用（另一个 audio-note 窗口或命令行？），先关闭它再删除: {e}") from e
    if current_profile_name() == name:
        try:
            CURRENT_PROFILE_FILE.unlink()
        except OSError:
            pass
    try:
        shutil.rmtree(trash)
    except OSError as e:
        raise ProfileInUse(f"档案 {name} 已移到 {trash}，但部分文件无法删除: {e}") from e
    return d


def resolve_config(explicit: str | os.PathLike | None = None, profile: str | None = None) -> tuple[str | None, Path, Path | None]:
    """(profile name or None for a custom file, config path, base config path or None).

    explicit path / AUDIO_NOTE_CONFIG -> that single file, no profile layering.
    profile (or AUDIO_NOTE_PROFILE / profiles/current) -> profiles/<name>/config.yaml over the root config.yaml."""
    if explicit:
        return None, Path(explicit), None
    env = os.environ.get("AUDIO_NOTE_CONFIG")
    if env and not profile:
        return None, Path(env), None
    name = (profile or "").strip() or current_profile_name()
    name = canonical_profile_name(name) or name
    if name != DEFAULT_PROFILE and not profile_exists(name):
        raise FileNotFoundError(f"档案不存在: {name}（audio-note profile list 查看，profile create {name} 新建）")
    if name == DEFAULT_PROFILE:
        return DEFAULT_PROFILE, PROJECT_ROOT / "config.yaml", None
    return name, profile_config_path(name), PROJECT_ROOT / "config.yaml"


def find_config_path(explicit: str | os.PathLike | None = None) -> Path:
    return resolve_config(explicit)[1]


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
    def embeddings_dir(self) -> Path:
        return self.models_dir / "embeddings"

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


def _read_yaml(p: Path) -> dict:
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def load_config(path: str | os.PathLike | None = None, profile: str | None = None,
                base: str | os.PathLike | None = None) -> Config:
    """Merged config. `path` = explicit file (no profile layering unless `base` is given);
    `profile` = profile name; neither = AUDIO_NOTE_CONFIG / AUDIO_NOTE_PROFILE / profiles/current / root."""
    if path is not None:
        name, p, base_p = None, Path(path), (Path(base) if base else None)
    else:
        name, p, base_p = resolve_config(None, profile)
    user = _read_yaml(p)
    merged = _merge(DEFAULTS, _merge(_read_yaml(base_p), user) if base_p else user)
    # env override for the secret so it never has to live in the file
    env_key = os.environ.get("MIMO_API_KEY")
    if env_key:
        merged["mimo"]["api_key"] = env_key
    # a placeholder copied from config.example.yaml must behave exactly like an empty key
    if is_placeholder_key(merged["mimo"].get("api_key")):
        merged["mimo"]["api_key"] = ""
    cfg = Config(merged)
    cfg["_config_path"] = str(p)
    cfg["_base_config_path"] = str(base_p) if base_p else None
    cfg["_profile"] = name
    return cfg
