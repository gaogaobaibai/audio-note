"""Speakers v3b §B (docs/SPEC_speakers_v3b.md): the loudness gate for far-away / unclear voices.

audio.loudness.speech_dbfs, the pipeline dropping faint turns (scripted engine + real numpy audio), loudness_db /
faint_s bookkeeping, relabel backfill, voices excluding faint sentences, the `loudness` CLI / UI job, the transcript
API field and the settings validation. No network (ASR is a fake), no microphone, no heavy models.
"""
from __future__ import annotations

import io
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import yaml
from fastapi.testclient import TestClient
from rich.console import Console

from audio_note import cli
from audio_note.audio.loudness import (DEFAULT_MIN_SPEECH_DBFS, SILENCE_DBFS, backfill_loudness, is_faint, loudness_stats,
                                       min_speech_dbfs, speech_dbfs)
from audio_note.audio.storage import segment_path
from audio_note.audio.vad import SAMPLE_RATE as SR
from audio_note.config import Config, load_config
from audio_note.db import DB
from audio_note.runtime import JOB_LABELS, JOBS, Runtime
from audio_note.speech import voices
from audio_note.speech.asr import ASRResult
from audio_note.speech.pipeline import Processor, drop_faint_turns
from audio_note.speech.speakers import Turn
from audio_note.ui.app import create_app

from .conftest import silence, sine

DAY = "2026-10-30"
LOUD_AMP = 0.3                      # sine amplitude -> RMS 0.212 -> -13.5 dBFS (near-field speech)
FAINT_AMP = 0.003                   # -> -53.5 dBFS (somebody two metres away)


def _db_of_sine(amp: float) -> float:
    return 20 * math.log10(amp / math.sqrt(2))


# --- fixtures / helpers ---------------------------------------------------------------------------------

@pytest.fixture
def pcfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """Pipeline config (same shape as tests/test_pipeline.py); the glossary lives in tmp so nothing touches the repo."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "paths": {"data_dir": str(tmp_path / "data")},
        "pipeline": {"min_segment_age_s": 0},
        "correction": {"enabled": False, "glossary_file": str(tmp_path / "glossary.txt")},
    }), encoding="utf-8")
    monkeypatch.setenv("AUDIO_NOTE_CONFIG", str(cfg_path))
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    c = load_config()
    c.ensure_dirs()
    return c


@pytest.fixture
def pdb(pcfg: Config) -> DB:
    return DB(pcfg.db_path)


def _seg(cfg: Config, db: DB, when: datetime, audio: np.ndarray, status: str = "new") -> tuple[int, Path]:
    p = segment_path(cfg, when, ".wav")
    sf.write(p, audio, SR, subtype="PCM_16")
    sid = db.add_segment(p, when, len(audio) / SR, speech_s=len(audio) / SR)
    if status != "new":
        db.set_segment_status(sid, status)
    return sid, p


def _utt(db: DB, seg: int, hhmm: str, text: str, start_s: float, end_s: float, who: str | None = None,
         speaker_id: int | None = None, cluster: int = 0, loudness_db: float | None = None, source: str | None = None,
         day: str = DAY) -> int:
    return db.add_utterances([{
        "segment_id": seg, "start_s": start_s, "end_s": end_s, "abs_start": f"{day}T{hhmm}:00", "day": day,
        "cluster_label": cluster, "speaker_id": speaker_id, "speaker_name": who, "speaker_score": 0.6 if speaker_id else None,
        "text_raw": text, "text": text, "text_seg": text, "loudness_db": loudness_db, "speaker_source": source,
    }])[0]


class ScriptedEngine:
    """Diarization returns the scripted turns; identification labels every turn by cluster (0 -> speaker `a`)."""

    def __init__(self, turns: list[Turn], a_id: int | None = None):
        self.turns = turns
        self.a_id = a_id
        self.identified: list[list[Turn]] = []

    def diarize(self, audio):
        return [Turn(t.start, t.end, t.cluster) for t in self.turns]

    def identify_turns(self, audio, turns, db, segment_id=None):
        self.identified.append([Turn(t.start, t.end, t.cluster) for t in turns])
        for t in turns:
            if t.cluster == 0 and self.a_id is not None:
                t.speaker_id, t.speaker_name, t.score = self.a_id, "甲", 0.8
            else:
                t.speaker_id, t.speaker_name, t.score = None, None, 0.1
        return turns

    identify_utterances = identify_turns


class FakeASR:
    def __init__(self, text: str = "你好"):
        self.text = text
        self.calls: list[float] = []

    def transcribe(self, audio):
        self.calls.append(round(len(audio) / SR, 2))
        return ASRResult(f"{self.text}{len(self.calls)}", round(len(audio) / SR, 1), 1)


def _proc(cfg: Config, db: DB, engine: Any, asr: FakeASR | None = None) -> Processor:
    p = Processor(cfg, db, client=object(), engine=engine, asr=asr or FakeASR(), corrector=None)
    p._corrector_tried = True
    return p


def _two_voices() -> np.ndarray:
    """0-3 s a loud voice, 3-6 s a faint one (same waveform, 100x quieter)."""
    return np.concatenate([sine(3.0, amp=LOUD_AMP), sine(3.0, amp=FAINT_AMP)])


# =========================================================================================================
# 1. speech_dbfs / config helpers
# =========================================================================================================

def test_speech_dbfs_silence_empty_and_sine_levels():
    assert speech_dbfs(np.zeros(0, dtype=np.float32)) == SILENCE_DBFS == -120.0
    assert speech_dbfs(silence(1.0)) == SILENCE_DBFS
    assert speech_dbfs(np.zeros(100, dtype=np.float32)) == SILENCE_DBFS            # shorter than one frame
    # a sine of amplitude A has RMS A/sqrt(2): -20 dBFS at A = 0.1*sqrt(2)
    assert speech_dbfs(sine(1.0, amp=0.1 * math.sqrt(2))) == pytest.approx(-20.0, abs=0.05)
    assert speech_dbfs(sine(2.0, amp=LOUD_AMP)) == pytest.approx(_db_of_sine(LOUD_AMP), abs=0.05)
    assert speech_dbfs(sine(2.0, amp=FAINT_AMP)) == pytest.approx(_db_of_sine(FAINT_AMP), abs=0.1)
    # float64 / 2-D input and a clip shorter than a frame still give a level
    assert speech_dbfs(sine(0.03, amp=0.1 * math.sqrt(2)).astype(np.float64)) == pytest.approx(-20.0, abs=0.3)
    assert speech_dbfs(np.full(5, 0.5)) == pytest.approx(20 * math.log10(0.5), abs=1e-6)


def test_speech_dbfs_uses_the_loudest_half_of_the_frames():
    """Half speech, half pause: the level is that of the speech, not dragged down by the pause."""
    loud = sine(1.0, amp=0.1 * math.sqrt(2))               # -20 dBFS
    mixed = np.concatenate([loud, silence(1.0)])
    assert speech_dbfs(mixed) == pytest.approx(-20.0, abs=0.05)
    # three quarters pause: the loudest half still contains silence, so the level drops by ~3 dB
    mostly_quiet = np.concatenate([loud[: SR // 2], silence(1.5)])
    assert -23.5 < speech_dbfs(mostly_quiet) < -22.5
    # with top_frac=1.0 the pause counts fully
    assert speech_dbfs(mixed, top_frac=1.0) == pytest.approx(-20.0 - 10 * math.log10(2), abs=0.05)
    # frames of a clipped region win over a quiet one regardless of order
    assert speech_dbfs(np.concatenate([silence(1.0), loud])) == pytest.approx(-20.0, abs=0.05)


def test_min_speech_dbfs_default_null_and_bad_values():
    assert DEFAULT_MIN_SPEECH_DBFS == -40.0
    cfg = Config({"audio": {"min_speech_dbfs": -40.0}})
    assert min_speech_dbfs(cfg) == -40.0
    assert min_speech_dbfs(Config({"audio": {"min_speech_dbfs": None}})) is None
    assert min_speech_dbfs(Config({"audio": {"min_speech_dbfs": ""}})) is None
    assert min_speech_dbfs(Config({"audio": {"min_speech_dbfs": "-35"}})) == -35.0
    assert min_speech_dbfs(Config({"audio": {"min_speech_dbfs": "abc"}})) is None
    assert min_speech_dbfs(Config({"audio": {}})) == DEFAULT_MIN_SPEECH_DBFS
    assert min_speech_dbfs(Config({})) == DEFAULT_MIN_SPEECH_DBFS
    assert is_faint(-50.0, -40.0) and not is_faint(-30.0, -40.0) and not is_faint(-40.0, -40.0)
    assert not is_faint(None, -40.0) and not is_faint(-50.0, None)


def test_config_default_and_example_carry_the_gate():
    from audio_note.config import DEFAULTS, PROJECT_ROOT

    assert DEFAULTS["audio"]["min_speech_dbfs"] == -40.0
    with open(PROJECT_ROOT / "config.example.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["audio"]["min_speech_dbfs"] == -40.0


def test_drop_faint_turns_pure_function():
    audio = _two_voices()
    turns = [Turn(0.1, 2.9, 0), Turn(3.1, 5.9, 1), Turn(5.95, 6.0, 1)]
    kept, dropped, faint_s = drop_faint_turns(audio, turns, -40.0)
    assert [t.cluster for t in kept] == [0] and dropped == 2 and faint_s == pytest.approx(2.85, abs=1e-6)
    assert drop_faint_turns(audio, turns, None) == (turns, 0, 0.0)
    assert drop_faint_turns(audio, [], -40.0) == ([], 0, 0.0)
    # a gate below everything keeps everything; one above everything drops everything
    assert len(drop_faint_turns(audio, turns, -90.0)[0]) == 3
    assert drop_faint_turns(audio, turns, -5.0)[1] == 3
    # a turn past the end of the audio is empty -> silent -> faint
    assert drop_faint_turns(audio, [Turn(7.0, 8.0, 0)], -40.0)[1] == 1


# =========================================================================================================
# 2. pipeline
# =========================================================================================================

def test_process_segment_drops_the_faint_turn_and_stores_levels(pcfg: Config, pdb: DB, caplog):
    seg, path = _seg(pcfg, pdb, datetime(2026, 10, 30, 9, 0), _two_voices())
    a = pdb.upsert_speaker("甲")
    engine = ScriptedEngine([Turn(0.1, 2.9, 0), Turn(3.1, 5.9, 1)], a)
    asr = FakeASR()
    with caplog.at_level(logging.INFO, logger="audio_note.speech.pipeline"):
        n = _proc(pcfg, pdb, engine, asr).process_segment(seg)
    assert n == 1 and len(asr.calls) == 1
    assert [[t.cluster for t in ts] for ts in engine.identified] == [[0]]            # the faint turn never reached identification
    assert any(f"segment #{seg}: dropped 1 faint turns (2.8 s, < -40 dBFS)" in r.getMessage() for r in caplog.records)
    s = pdb.get_segment(seg)
    assert s["status"] == "done" and s["faint_s"] == pytest.approx(2.8, abs=1e-6) and path.exists()
    rows = pdb.query("SELECT * FROM utterances WHERE segment_id=?", (seg,))
    assert len(rows) == 1 and rows[0]["speaker_id"] == a and rows[0]["cluster_label"] == 0
    assert rows[0]["loudness_db"] == pytest.approx(_db_of_sine(LOUD_AMP), abs=0.2)


def test_process_segment_all_faint_is_no_speech(pcfg: Config, pdb: DB):
    seg, path = _seg(pcfg, pdb, datetime(2026, 10, 30, 9, 0), sine(4.0, amp=FAINT_AMP))
    engine = ScriptedEngine([Turn(0.0, 2.0, 0), Turn(2.0, 4.0, 1)])
    asr = FakeASR()
    assert _proc(pcfg, pdb, engine, asr).process_segment(seg) == 0
    s = pdb.get_segment(seg)
    assert s["status"] == "no_speech" and s["faint_s"] == pytest.approx(4.0) and not path.exists()
    assert asr.calls == [] and engine.identified == []
    assert pdb.query("SELECT id FROM utterances WHERE segment_id=?", (seg,)) == []


def test_process_segment_gate_off_keeps_the_faint_turn(pcfg: Config, pdb: DB):
    pcfg["audio"]["min_speech_dbfs"] = None
    seg, _ = _seg(pcfg, pdb, datetime(2026, 10, 30, 9, 0), _two_voices())
    engine = ScriptedEngine([Turn(0.1, 2.9, 0), Turn(3.1, 5.9, 1)])
    assert _proc(pcfg, pdb, engine).process_segment(seg) == 2
    rows = pdb.query("SELECT loudness_db FROM utterances WHERE segment_id=? ORDER BY start_s", (seg,))
    assert rows[0]["loudness_db"] == pytest.approx(_db_of_sine(LOUD_AMP), abs=0.2)
    assert rows[1]["loudness_db"] == pytest.approx(_db_of_sine(FAINT_AMP), abs=0.3)
    assert pdb.get_segment(seg)["faint_s"] == 0.0
    # a stricter gate (in the file, reloaded) drops the louder one too
    pcfg["audio"]["min_speech_dbfs"] = -10.0
    seg2, _ = _seg(pcfg, pdb, datetime(2026, 10, 30, 10, 0), _two_voices())
    assert _proc(pcfg, pdb, ScriptedEngine([Turn(0.1, 2.9, 0), Turn(3.1, 5.9, 1)])).process_segment(seg2) == 0
    assert pdb.get_segment(seg2)["status"] == "no_speech"


def test_relabel_backfills_levels_and_skips_faint_sentences(pcfg: Config, pdb: DB):
    """Rows written before the gate existed have no level: relabel computes it, identifies only the loud ones
    and clears the score of the faint ones (speaker left as it is)."""
    seg, _ = _seg(pcfg, pdb, datetime(2026, 10, 30, 9, 0), _two_voices(), status="done")
    a = pdb.upsert_speaker("甲")
    pdb.add_speaker_embedding(a, np.ones(8, dtype=np.float32), None, 3.0)
    loud = _utt(pdb, seg, "09:00", "近的", 0.1, 2.9, "未知", None, cluster=0)
    faint = _utt(pdb, seg, "09:00", "远的", 3.1, 5.9, "乙", None, cluster=1)
    pdb.execute("UPDATE utterances SET speaker_score=0.42 WHERE id=?", (faint,))
    engine = ScriptedEngine([], a)
    proc = Processor(pcfg, pdb, engine=engine)
    assert proc.relabel_segment(seg) == 1
    assert [[(t.start, t.cluster) for t in ts] for ts in engine.identified] == [[(0.1, 0)]]
    rows = {r["id"]: r for r in pdb.query("SELECT * FROM utterances WHERE segment_id=?", (seg,))}
    assert rows[loud]["speaker_id"] == a and rows[loud]["speaker_name"] == "甲" and rows[loud]["speaker_score"] == pytest.approx(0.8)
    assert rows[loud]["loudness_db"] == pytest.approx(_db_of_sine(LOUD_AMP), abs=0.2)
    assert rows[faint]["loudness_db"] == pytest.approx(_db_of_sine(FAINT_AMP), abs=0.3)
    assert (rows[faint]["speaker_id"], rows[faint]["speaker_name"], rows[faint]["speaker_score"]) == (None, "乙", None)
    # second pass: levels are stored, nothing changes
    assert proc.relabel_segment(seg) == 0
    # with the gate off the faint one is identified as well (cluster 1 -> nobody: "乙" becomes 未知, score 0.1)
    pcfg["audio"]["min_speech_dbfs"] = None
    assert proc.relabel_segment(seg) == 1
    row = pdb.one("SELECT speaker_name, speaker_score FROM utterances WHERE id=?", (faint,))
    assert row["speaker_name"] == "未知" and row["speaker_score"] == pytest.approx(0.1)
    # a stored level is trusted over the audio (not recomputed)
    pdb.execute("UPDATE utterances SET loudness_db=-99 WHERE id=?", (loud,))
    pcfg["audio"]["min_speech_dbfs"] = -40.0
    proc.relabel_segment(seg)
    assert pdb.one("SELECT loudness_db, speaker_score FROM utterances WHERE id=?", (loud,))["speaker_score"] is None


def test_split_pieces_carry_their_own_level(pcfg: Config, pdb: DB):
    seg, _ = _seg(pcfg, pdb, datetime(2026, 10, 30, 9, 0), _two_voices(), status="done")
    mixed = _utt(pdb, seg, "09:00", "混", 0.0, 6.0, "未知", None, cluster=0)
    pcfg["audio"]["min_speech_dbfs"] = None
    res = _proc(pcfg, pdb, ScriptedEngine([])).split_utterance(mixed, at=[3.0])
    rows = pdb.query("SELECT loudness_db FROM utterances WHERE segment_id=? ORDER BY start_s", (seg,))
    assert len(res["pieces"]) == 2 and len(rows) == 2
    assert rows[0]["loudness_db"] == pytest.approx(_db_of_sine(LOUD_AMP), abs=0.3)
    assert rows[1]["loudness_db"] == pytest.approx(_db_of_sine(FAINT_AMP), abs=0.5)


# =========================================================================================================
# 3. backfill / stats / voices
# =========================================================================================================

def test_backfill_loudness_and_stats(pcfg: Config, pdb: DB):
    seg, path = _seg(pcfg, pdb, datetime(2026, 10, 30, 9, 0), _two_voices(), status="done")
    gone, gone_path = _seg(pcfg, pdb, datetime(2026, 10, 30, 10, 0), sine(2.0), status="done")
    gone_path.unlink()
    pending, _ = _seg(pcfg, pdb, datetime(2026, 10, 30, 11, 0), sine(2.0), status="new")
    a = pdb.upsert_speaker("甲")
    u1 = _utt(pdb, seg, "09:00", "近", 0.0, 3.0, "甲", a)
    u2 = _utt(pdb, seg, "09:00", "远", 3.0, 6.0, "未知", None, cluster=1)
    u3 = _utt(pdb, seg, "09:00", "有", 0.0, 1.0, "甲", a, loudness_db=-25.0)
    u4 = _utt(pdb, gone, "10:00", "没音频", 0.0, 1.0, "甲", a)
    u5 = _utt(pdb, pending, "11:00", "还没处理", 0.0, 1.0, "甲", a)
    assert backfill_loudness(pdb) == 2
    rows = {r["id"]: r["loudness_db"] for r in pdb.query("SELECT id, loudness_db FROM utterances")}
    assert rows[u1] == pytest.approx(_db_of_sine(LOUD_AMP), abs=0.2) and rows[u2] == pytest.approx(_db_of_sine(FAINT_AMP), abs=0.3)
    assert rows[u3] == -25.0 and rows[u4] is None and rows[u5] is None
    assert backfill_loudness(pdb) == 0
    st = loudness_stats(pdb, -40.0)
    assert st["total"] == 5 and st["missing"] == 2 and st["below"] == 1 and st["threshold"] == -40.0
    by = {s["name"]: s for s in st["speakers"]}
    assert by["甲"]["n"] == 2 and by["甲"]["below"] == 0 and by["甲"]["median"] == pytest.approx((-25.0 + _db_of_sine(LOUD_AMP)) / 2, abs=0.2)
    assert by["未知"]["n"] == 1 and by["未知"]["below"] == 1 and by["未知"]["p10"] == by["未知"]["p90"]
    assert st["speakers"][0]["name"] == "甲"                                          # most sentences first
    assert loudness_stats(pdb, None)["below"] == 0
    assert loudness_stats(pdb, -40.0, day="2026-10-31")["total"] == 0
    # day / segment filters of the backfill
    _utt(pdb, seg, "09:01", "再来", 0.0, 1.0, "甲", a)
    assert backfill_loudness(pdb, day="2026-10-31") == 0 and backfill_loudness(pdb, segment_id=seg) == 1


def test_voices_grouping_and_naming_ignore_faint_sentences(cfg, db):
    seg, _ = _seg(cfg, db, datetime(2026, 10, 30, 9, 0), _two_voices(), status="done")
    emb = np.random.default_rng(1).standard_normal(192).astype(np.float32)
    db.upsert_cluster_embedding(seg, 0, emb, 3.0)
    db.upsert_cluster_embedding(seg, 1, -emb, 3.0)
    loud = _utt(db, seg, "09:00", "近的", 0.0, 3.0, "未知", None, cluster=0, loudness_db=-20.0)
    faint = _utt(db, seg, "09:00", "远的", 3.0, 6.0, "未知", None, cluster=1, loudness_db=-55.0)
    unknown_level = _utt(db, seg, "09:00", "没算过", 0.0, 1.0, "未知", None, cluster=0, loudness_db=None)
    out = voices.group_unknown_voices(db, cfg)
    assert out["unknown_utterances"] == 2                                            # the faint one is not counted
    assert [g["members"] for g in out["groups"]] == [[[seg, 0]]]                     # cluster 1 has only faint sentences
    assert {s["utterance_id"] for s in out["groups"][0]["samples"]} == {loud, unknown_level}
    assert voices.unknown_utterance_count(db) == 3 and voices.unknown_utterance_count(db, -40.0) == 2
    # gate off: everything is back
    cfg["audio"]["min_speech_dbfs"] = None
    out = voices.group_unknown_voices(db, cfg)
    assert out["unknown_utterances"] == 3 and len(out["groups"]) == 2
    cfg["audio"]["min_speech_dbfs"] = -40.0

    class Engine:
        clips: list[float] = []

        def enroll_from_audio(self, db_, name, audio, source=None, copy_as_source=False, **kw):
            Engine.clips.append(round(len(audio) / SR, 1))
            assert kw.get("origin") == "voice"
            sid = db_.upsert_speaker(name)
            db_.add_speaker_embedding(sid, emb, source, len(audio) / SR, origin=kw.get("origin"))
            return sid

    # naming cluster 1 (whose only sentence is faint): no sample can be cut and the stored cluster embedding is a
    # far-away voice too, so a brand-new name is refused (and leaves no empty speaker row)
    with pytest.raises(ValueError, match="太轻"):
        voices.name_voice(db, cfg, Engine(), [[seg, 1]], "乙")
    assert Engine.clips == [] and db.one("SELECT id FROM speakers WHERE name='乙'") is None
    # naming cluster 0: the sample is cut from the loud sentences only (3 s + 1 s), faint ones never enter samples
    res = voices.name_voice(db, cfg, Engine(), [[seg, 0], [seg, 1]], "甲")
    assert Engine.clips == [4.0] and res["samples_added"] == 1 and res["utterances_assigned"] == 3
    assert db.one("SELECT origin FROM speaker_embeddings")["origin"] == "voice"
    assert db.one("SELECT speaker_name FROM utterances WHERE id=?", (faint,))["speaker_name"] == "甲"


# =========================================================================================================
# 4. runtime job, API, settings
# =========================================================================================================

@pytest.fixture
def ui_cfg(cfg_factory, tmp_path: Path):
    cfg = cfg_factory({"memory": {"retrieval": "fts", "embedding_model": "hash"},
                       "correction": {"enabled": False, "glossary_file": str(tmp_path / "glossary.txt")}})
    yield cfg
    from audio_note.logging_setup import _remove_own_handlers

    _remove_own_handlers(logging.getLogger())


@pytest.fixture
def rt(ui_cfg) -> Runtime:
    r = Runtime(ui_cfg)
    yield r
    r.shutdown()


@pytest.fixture
def client(rt: Runtime) -> TestClient:
    with TestClient(create_app(rt), raise_server_exceptions=False) as c:
        yield c


def test_loudness_job_computes_missing_levels_and_counts(rt: Runtime):
    assert "loudness" in JOBS and JOB_LABELS["loudness"] == "统计音量分布" and "loudness" in rt.jobs
    seg, _ = _seg(rt.cfg, rt.db, datetime(2026, 10, 30, 9, 0), _two_voices(), status="done")
    a = rt.db.upsert_speaker("甲")
    _utt(rt.db, seg, "09:00", "近", 0.0, 3.0, "甲", a)
    _utt(rt.db, seg, "09:00", "远", 3.0, 6.0, "未知", None, cluster=1)
    res = rt.run_job("loudness")
    assert res["computed"] == 2 and res["below"] == 1 and res["threshold"] == -40.0 and res["total"] == 2 and res["missing"] == 0
    assert {s["name"] for s in res["speakers"]} == {"甲", "未知"}
    assert rt.db.one("SELECT COUNT(*) AS n FROM utterances")["n"] == 2                 # statistics only: nothing deleted
    assert rt.run_job("loudness")["computed"] == 0
    rt.cfg["audio"]["min_speech_dbfs"] = None
    res = rt.run_job("loudness")
    assert res["below"] == 0 and res["threshold"] is None
    st = rt.submit("loudness")
    assert rt.wait_jobs(5) and st.result["computed"] == 0 and st.runs == 1


def test_api_jobs_loudness_and_transcript_levels(client: TestClient, rt: Runtime):
    seg, _ = _seg(rt.cfg, rt.db, datetime(2026, 10, 30, 9, 0), _two_voices(), status="done")
    a = rt.db.upsert_speaker("甲")
    u1 = _utt(rt.db, seg, "09:00", "近", 0.0, 3.0, "甲", a, loudness_db=-26.04)
    u2 = _utt(rt.db, seg, "09:00", "远", 3.0, 6.0, "未知", None, cluster=1)
    r = client.post("/api/jobs/loudness")
    assert r.status_code == 200 and r.json()["job"]["name"] == "loudness"
    assert rt.wait_jobs(5)
    job = client.get("/api/jobs").json()["loudness"]
    assert job["label"] == "统计音量分布" and job["error"] is None
    assert job["result"]["computed"] == 1 and job["result"]["below"] == 1 and job["result"]["threshold"] == -40.0
    tr = client.get(f"/api/transcript/{DAY}").json()
    assert tr["min_speech_dbfs"] == -40.0
    utts = {u["id"]: u for s in tr["sessions"] for u in s["utterances"]}
    assert utts[u1]["loudness_db"] == -26.0
    assert utts[u2]["loudness_db"] == pytest.approx(_db_of_sine(FAINT_AMP), abs=0.3)
    assert client.get(f"/api/context/{u1}").json()["utterances"][0]["loudness_db"] == -26.0
    # /api/speakers leaves the faint sentence out of the unidentified count
    assert client.get("/api/speakers").json()["unknown_utterances"] == 0
    rt.cfg["audio"]["min_speech_dbfs"] = None
    assert client.get("/api/speakers").json()["unknown_utterances"] == 1
    assert client.get(f"/api/transcript/{DAY}").json()["min_speech_dbfs"] is None


def test_api_config_min_speech_dbfs_validation_and_null(client: TestClient, rt: Runtime):
    got = client.get("/api/config").json()
    assert got["values"]["audio"]["min_speech_dbfs"] == -40.0 and got["defaults"]["audio"]["min_speech_dbfs"] == -40.0
    r = client.put("/api/config", json={"audio": {"min_speech_dbfs": -200}})
    assert r.status_code == 400 and "-90" in r.json()["error"]
    assert client.put("/api/config", json={"audio": {"min_speech_dbfs": 5}}).status_code == 400
    assert client.put("/api/config", json={"audio": {"min_speech_dbfs": "loud"}}).status_code == 400
    assert client.put("/api/config", json={"audio": {"min_speech_dbfs": True}}).status_code == 400
    r = client.put("/api/config", json={"audio": {"min_speech_dbfs": "-35.5"}})
    assert r.status_code == 200, r.text
    assert rt.cfg.audio.min_speech_dbfs == -35.5 and "min_speech_dbfs: -35.5" in rt.config_path.read_text(encoding="utf-8")
    # an empty field means null = gate off
    r = client.put("/api/config", json={"audio": {"min_speech_dbfs": ""}})
    assert r.status_code == 200 and r.json()["saved"] == ["audio.min_speech_dbfs"]
    assert rt.cfg.audio.min_speech_dbfs is None
    assert yaml.safe_load(rt.config_path.read_text(encoding="utf-8"))["audio"]["min_speech_dbfs"] is None
    assert client.get("/api/config").json()["values"]["audio"]["min_speech_dbfs"] is None
    r = client.put("/api/config", json={"audio": {"min_speech_dbfs": None}})
    assert r.status_code == 200 and r.json()["saved"] == []                           # unchanged: nothing written
    r = client.put("/api/config", json={"audio": {"min_speech_dbfs": -40}})
    assert r.status_code == 200 and rt.cfg.audio.min_speech_dbfs == -40.0


# =========================================================================================================
# 5. CLI
# =========================================================================================================

@pytest.fixture
def out(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    buf = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=buf, width=400, force_terminal=False, color_system=None,
                                                 highlight=False, soft_wrap=True))
    return buf


@pytest.fixture
def cli_cfg(cfg_factory):
    cfg = cfg_factory({"correction": {"enabled": False}})
    yield cfg
    from audio_note.logging_setup import _remove_own_handlers

    _remove_own_handlers(logging.getLogger())


def test_cli_loudness_stats_and_apply(cli_cfg, out: io.StringIO):
    db = DB(cli_cfg.db_path)
    seg, _ = _seg(cli_cfg, db, datetime(2026, 10, 30, 9, 0), _two_voices(), status="done")
    a = db.upsert_speaker("甲")
    near = _utt(db, seg, "09:00", "近的", 0.0, 3.0, "甲", a)
    far = _utt(db, seg, "09:00", "远的", 3.0, 6.0, "未知", None, cluster=1)
    kept = _utt(db, seg, "09:00", "手标的远句", 3.0, 6.0, "乙", None, cluster=1, source="user")
    other_day = _utt(db, seg, "09:00", "别天的远句", 3.0, 6.0, "未知", None, cluster=1, day="2026-10-31")
    db.add_vectors("hash", [(far, np.ones(4, dtype=np.float32))])
    assert cli.main(["loudness"]) == 0
    text = out.getvalue()
    assert "补算了 4 句的音量" in text and "门槛 -40" in text and "3 句低于 -40 dBFS" in text
    assert "--apply --yes" in text and "甲" in text and "未知" in text
    assert db.one("SELECT loudness_db FROM utterances WHERE id=?", (near,))["loudness_db"] == pytest.approx(_db_of_sine(LOUD_AMP), abs=0.2)
    # --apply without --yes: nothing deleted, exit 1
    out.truncate(0), out.seek(0)
    assert cli.main(["loudness", "--apply"]) == 1
    assert "将删除 2 句" in out.getvalue() and "--yes" in out.getvalue()
    assert db.one("SELECT COUNT(*) AS n FROM utterances")["n"] == 4
    # --day limits both the statistics and the deletion
    out.truncate(0), out.seek(0)
    assert cli.main(["loudness", "--day", "2026-10-31", "--apply", "--yes"]) == 0
    assert "已删除 1 句" in out.getvalue()
    assert db.one("SELECT id FROM utterances WHERE id=?", (other_day,)) is None
    out.truncate(0), out.seek(0)
    assert cli.main(["loudness", "--apply", "-y"]) == 0
    text = out.getvalue()
    assert "将删除 1 句" in text and "已删除 1 句，重写了 1 天的记录文件" in text
    ids = {r["id"] for r in db.query("SELECT id FROM utterances")}
    assert ids == {near, kept}                                                        # the hand-labelled faint one stays
    assert db.one("SELECT COUNT(*) AS n FROM utterance_vectors")["n"] == 0            # cascaded
    assert db.one("SELECT COUNT(*) AS n FROM utterance_tokens")["n"] == 2
    md = (cli_cfg.transcripts_dir / f"{DAY}.md").read_text(encoding="utf-8")
    assert "远的" not in md and "近的" in md and "手标的远句" in md
    out.truncate(0), out.seek(0)
    assert cli.main(["loudness", "--apply", "--yes"]) == 0
    assert "没有可删除的句子" in out.getvalue()
    assert cli.main(["loudness", "--day", "20261030"]) == 2                           # bad day = CLIError


def test_cli_loudness_gate_off_and_empty(cli_cfg, out: io.StringIO, monkeypatch: pytest.MonkeyPatch):
    assert cli.main(["loudness"]) == 0
    assert "（没有记录）" in out.getvalue()
    db = DB(cli_cfg.db_path)
    seg, _ = _seg(cli_cfg, db, datetime(2026, 10, 30, 9, 0), sine(2.0, amp=FAINT_AMP), status="done")
    _utt(db, seg, "09:00", "远的", 0.0, 2.0, "未知", None)
    cli_cfg["audio"]["min_speech_dbfs"] = None
    monkeypatch.setattr(cli, "load_config", lambda *a, **k: cli_cfg)
    out.truncate(0), out.seek(0)
    assert cli.main(["loudness", "--apply", "--yes"]) == 0
    assert "门槛已关闭" in out.getvalue() and "不会过滤" in out.getvalue()
    assert db.one("SELECT COUNT(*) AS n FROM utterances")["n"] == 1
