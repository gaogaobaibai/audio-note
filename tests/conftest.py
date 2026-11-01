"""Shared pytest fixtures: temp config (via AUDIO_NOTE_CONFIG), temp DB, fake VAD, session-scoped models.

Nothing here touches the network; MimoClient is never constructed with a real key.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # tests never download models, whatever a code path under test asks for
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
# Real speech recording for the tests that run the CPU models: 16 kHz mono wav, silent for the first 29 s, speech after.
# Not in the repository (tests/data/ is ignored); those tests skip without it. Override with AUDIO_NOTE_TEST_WAV.
SAMPLE_WAV = Path(os.environ.get("AUDIO_NOTE_TEST_WAV") or PROJECT_ROOT / "tests" / "data" / "sample.wav")


#: Tests that need the files from scripts/download_models.ps1 without using a model fixture: they go through the
#: real "models present?" checks of the API / CLI / recorder. Skipped on a checkout that has no models yet.
_NEEDS_MODELS = {
    "test_api_label_and_voices_name_report_mismatch_and_accept_force",
    "test_api_split_needs_api_key_then_splits",
    "test_api_split_runs_under_the_scheduler_lock_preview_does_not",
    "test_delete_speaker_releases_hand_labelled_rows_and_relation_cache",
    "test_inactive_stream_raises_immediately",
    "test_job_already_running_and_read_only_are_409",
    "test_partial_write_failure_keeps_recording",
    "test_reanchors_after_gap_in_delivered_samples",
    "test_recording_start_and_stop",
    "test_speaker_add_sample_rejections",
    "test_speaker_label_error_cases",
    "test_speakers_voices_assign_after_a_scan_asks_for_a_fresh_number",
    "test_speakers_voices_assign_too_short_is_cli_error",
    "test_speakers_voices_engine_failure_is_a_cli_error",
    "test_speakers_voices_scan_list_and_assign",
    "test_speakers_voices_shows_keys_and_assign_accepts_index_or_key",
    "test_stall_raises_so_the_service_reopens_the_device",
    "test_status_shape_on_empty_db",
    "test_voices_name_and_label_hold_the_busy_counter",
    "test_voices_name_assigns_group_and_starts_relabel",
    "test_voices_name_errors",
    "test_voices_name_mismatch_is_400_with_flag_and_force_retries",
    "test_voices_name_relabel_is_skipped_when_a_job_blocks_it",
}


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    from audio_note.config import DEFAULTS

    sp = DEFAULTS["speakers"]
    if all((MODELS_DIR / sp[k]).exists() for k in ("vad_model", "embedding_model", "segmentation_model")):
        return
    skip = pytest.mark.skip(reason="models missing: run scripts/download_models.ps1")
    for item in items:
        if item.name in _NEEDS_MODELS:
            item.add_marker(skip)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@pytest.fixture
def cfg_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., Any]:
    """Build a Config whose data_dir is a fresh temp dir. Pass nested overrides as a dict.

    The yaml is written to tmp_path and exported through AUDIO_NOTE_CONFIG so that
    `load_config()` with no arguments picks it up, exactly like the CLI would.
    """
    from audio_note.config import load_config

    monkeypatch.delenv("MIMO_API_KEY", raising=False)

    def make(overrides: dict | None = None, name: str = "config.yaml") -> Any:
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        user = {
            "paths": {"data_dir": str(data_dir), "models_dir": str(MODELS_DIR)},
            "mimo": {"api_key": ""},
        }
        user = _deep_merge(user, overrides or {})
        p = tmp_path / name
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(user, f, allow_unicode=True)
        monkeypatch.setenv("AUDIO_NOTE_CONFIG", str(p))
        cfg = load_config()
        assert cfg.data_dir == data_dir
        cfg.ensure_dirs()
        return cfg

    return make


@pytest.fixture
def cfg(cfg_factory: Callable[..., Any]) -> Any:
    return cfg_factory()


@pytest.fixture
def db(cfg: Any) -> Any:
    from audio_note.db import DB

    return DB(cfg.db_path)


class FakeVAD:
    """Stand-in for SileroVAD: returns scripted probabilities per window (cycled / default)."""

    def __init__(self, probs: list[float] | None = None, default: float = 0.0):
        self.script = list(probs or [])
        self.default = default
        self.calls = 0
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1
        self.calls = 0

    def prob(self, window: np.ndarray) -> float:
        i = self.calls
        self.calls += 1
        if i < len(self.script):
            return float(self.script[i])
        return float(self.default)


@pytest.fixture
def fake_vad_factory() -> Callable[..., FakeVAD]:
    return FakeVAD


@pytest.fixture(scope="session")
def silero() -> Any:
    """Real Silero VAD, loaded once per session (read-only model file)."""
    from audio_note.audio.vad import SileroVAD

    p = MODELS_DIR / "silero_vad.onnx"
    if not p.exists():
        pytest.skip(f"model missing: {p}")
    return SileroVAD(p, threads=1)


@pytest.fixture(scope="session")
def speaker_engine(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """SpeakerEngine with the CAM++ extractor loaded once; the diarizer stays lazy.
    Its data_dir (where enrollment sample copies land) is a session temp dir, never the repo."""
    from audio_note.config import Config, DEFAULTS
    from audio_note.speech.speakers import SpeakerEngine

    import copy

    emb = MODELS_DIR / DEFAULTS["speakers"]["embedding_model"]
    if not emb.exists():
        pytest.skip(f"model missing: {emb}")
    c = copy.deepcopy(DEFAULTS)
    c["paths"]["models_dir"] = str(MODELS_DIR)
    c["paths"]["data_dir"] = str(tmp_path_factory.mktemp("speaker_engine_data"))
    c["speakers"]["num_threads"] = 2
    return SpeakerEngine(Config(c))


def sine(seconds: float, freq: float = 440.0, sr: int = 16000, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def noise(seconds: float, sr: int = 16000, amp: float = 0.1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(seconds * sr))).astype(np.float32)


def silence(seconds: float, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def chirp(seconds: float, sr: int = 16000, amp: float = 0.3) -> np.ndarray:
    """Linear sweep 100 Hz -> 1 kHz; its CAM++ embedding is nearly orthogonal to white noise."""
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * (100 + 900 * t / max(seconds, 1e-6)) * t)).astype(np.float32)
