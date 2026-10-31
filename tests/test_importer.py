"""Tests for audio_note.audio.importer: start-time guessing, decoding, import_file on synthetic + real sample."""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_note.audio.importer import _name_has_time, decode_any, guess_start_time, import_file
from audio_note.audio.storage import read_segment
from audio_note.audio.vad import SAMPLE_RATE, segment_audio
from audio_note.audio.recorder import gate_config_from

from .conftest import SAMPLE_WAV, FakeVAD, noise, silence, sine


@pytest.mark.parametrize("name,expected", [
    ("20261031_091500.wav", datetime(2026, 10, 31, 9, 15, 0)),
    ("20261031-091500.m4a", datetime(2026, 10, 31, 9, 15, 0)),
    ("20261031091500.mp3", datetime(2026, 10, 31, 9, 15, 0)),
    ("2026-10-31 09-15-00.wav", datetime(2026, 10, 31, 9, 15, 0)),
    ("2026-10-31_09-15-00.wav", datetime(2026, 10, 31, 9, 15, 0)),
    ("rec_2026-10-31T09_15_00_meeting.flac", datetime(2026, 10, 31, 9, 15, 0)),
    ("Recording 2026_10_31 09.wav".replace(" 09", "_091500"), datetime(2026, 10, 31, 9, 15, 0)),
    ("voice-2026-10-31-09-15-00.ogg", datetime(2026, 10, 31, 9, 15, 0)),
])
def test_guess_start_time_patterns(tmp_path, name, expected):
    p = tmp_path / name
    p.write_bytes(b"")
    assert guess_start_time(p) == expected
    assert _name_has_time(p)


@pytest.mark.parametrize("name", ["meeting.wav", "voice 001.m4a", "20261031.wav", "123456789012.wav"])
def test_guess_start_time_falls_back_to_mtime(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(b"")
    ts = time.mktime(datetime(2026, 10, 30, 18, 30, 0).timetuple())
    os.utime(p, (ts, ts))
    assert guess_start_time(p) == datetime(2026, 10, 30, 18, 30, 0)
    assert not _name_has_time(p)


def test_guess_start_time_invalid_date_falls_back(tmp_path):
    p = tmp_path / "20261331_091500.wav"  # month 13
    p.write_bytes(b"")
    ts = time.mktime(datetime(2026, 10, 30, 18, 30, 0).timetuple())
    os.utime(p, (ts, ts))
    assert guess_start_time(p) == datetime(2026, 10, 30, 18, 30, 0)
    assert not _name_has_time(p)   # the mtime path must then also subtract the duration


@pytest.mark.parametrize("name", ["20261031_0915.wav", "2026-10-31 09-15.m4a", "rec_20261031-0915_x.mp3"])
def test_name_time_without_seconds(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(b"")
    assert guess_start_time(p) == datetime(2026, 10, 31, 9, 15, 0)
    assert _name_has_time(p)


def test_import_file_name_without_seconds_and_invalid_date(cfg, db, tmp_path):
    # HHMM in the name: taken as the START, duration not subtracted
    p = tmp_path / "20261031_0915.wav"
    sf.write(p, noise(4.0, amp=0.05), SAMPLE_RATE)
    ids = import_file(cfg, db, p, vad=FakeVAD(default=1.0))
    started = datetime.fromisoformat(db.get_segment(ids[0])["started_at"])
    assert abs((started - datetime(2026, 10, 31, 9, 15, 0)).total_seconds()) < 0.01
    # invalid date in the name: treated as no name time -> mtime (end) minus duration
    q = tmp_path / "20261331_091500.wav"
    sf.write(q, noise(4.0, amp=0.05), SAMPLE_RATE)
    end = datetime(2026, 10, 30, 18, 30, 0)
    ts = time.mktime(end.timetuple())
    os.utime(q, (ts, ts))
    ids = import_file(cfg, db, q, vad=FakeVAD(default=1.0))
    started = datetime.fromisoformat(db.get_segment(ids[0])["started_at"])
    assert abs((started - (end - timedelta(seconds=4))).total_seconds()) < 0.01


def test_decode_any_wav_and_resample(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(p, sine(0.5, sr=44100), 44100)
    a = decode_any(p)
    assert a.dtype == np.float32 and a.ndim == 1
    assert abs(len(a) - 0.5 * SAMPLE_RATE) <= 1


def test_decode_any_unreadable_raises(tmp_path):
    p = tmp_path / "junk.xyz"
    p.write_bytes(b"\x00\x01\x02 definitely not audio")
    with pytest.raises(Exception):
        decode_any(p)


def test_import_file_silence_yields_nothing(cfg, db, tmp_path):
    p = tmp_path / "quiet_20261031_090000.wav"
    sf.write(p, silence(3.0), SAMPLE_RATE)
    assert import_file(cfg, db, p, vad=FakeVAD(default=0.0)) == []
    assert db.query("SELECT * FROM segments") == []
    assert list(cfg.audio_dir.rglob("*.opus")) == []


def test_import_file_scripted_vad_sets_times_and_rows(cfg, db, tmp_path):
    # 12 s: speech 2..5 s and 8..10 s (window indices), silence elsewhere.
    # The gap (3 s) exceeds the default min_silence_s of 2.0 s so two segments result.
    assert float(cfg.audio.min_silence_s) == 2.0
    n_win = 12 * SAMPLE_RATE // 512
    probs = [0.0] * n_win
    for i in range(n_win):
        t = i * 512 / SAMPLE_RATE
        if 2.0 <= t < 5.0 or 8.0 <= t < 10.0:
            probs[i] = 0.95
    p = tmp_path / "talk.wav"
    sf.write(p, noise(12.0, amp=0.05), SAMPLE_RATE)
    start = datetime(2026, 10, 31, 9, 0, 0)
    ids = import_file(cfg, db, p, start_time=start, vad=FakeVAD(probs, default=0.0))
    assert len(ids) == 2
    rows = [db.get_segment(i) for i in ids]
    for r in rows:
        assert r["source"] == "import" and r["origin"] == str(p) and r["status"] == "new"
        assert r["speech_s"] > 0 and Path(r["path"]).exists() and Path(r["path"]).suffix == ".opus"
        assert cfg.audio_dir in Path(r["path"]).parents
    t0 = datetime.fromisoformat(rows[0]["started_at"])
    t1 = datetime.fromisoformat(rows[1]["started_at"])
    # The pre-roll ring (0.5 s) also holds the min_speech_s onset windows (0.3 s), so the audio
    # kept before speech onset is pre_roll_s - min_speech_s ~ 0.2 s (not the full 0.5 s).
    lead = float(cfg.audio.pre_roll_s) - float(cfg.audio.min_speech_s)
    assert abs((t0 - start).total_seconds() - (2.0 - lead)) < 0.1
    assert abs((t1 - start).total_seconds() - (8.0 - lead)) < 0.1
    assert rows[0]["speech_s"] == pytest.approx(3.0, abs=0.1)
    assert rows[1]["speech_s"] == pytest.approx(2.0, abs=0.1)


def test_import_file_without_name_time_uses_mtime_minus_duration(cfg, db, tmp_path):
    p = tmp_path / "memo.wav"
    sf.write(p, noise(4.0, amp=0.05), SAMPLE_RATE)
    end = datetime(2026, 10, 30, 18, 30, 0)
    ts = time.mktime(end.timetuple())
    os.utime(p, (ts, ts))
    ids = import_file(cfg, db, p, vad=FakeVAD(default=1.0))
    assert len(ids) == 1
    started = datetime.fromisoformat(db.get_segment(ids[0])["started_at"])
    # whole file is speech -> segment starts at sample 0 -> start = mtime - 4 s
    assert abs((started - (end - timedelta(seconds=4))).total_seconds()) < 0.01


@pytest.mark.skipif(not SAMPLE_WAV.exists(), reason="real sample recording not available")
def test_import_real_sample(cfg_factory, db, silero):
    # With the default 2.0 s hangover this 272 s sample yields 9 segments; 1.0 s gives ~29.
    cfg = cfg_factory({"audio": {"min_silence_s": 1.0}})
    audio = read_segment(SAMPLE_WAV)
    dur = len(audio) / SAMPLE_RATE
    assert dur > 60
    segs = segment_audio(audio, silero, gate_config_from(cfg))
    assert len(segs) >= 10
    total_speech = sum(s.speech_s for s in segs)
    assert 0.3 * dur < total_speech < dur  # a talk: mostly speech, but not everything
    assert all(s.speech_s > 0 for s in segs)
    assert all(s.duration_s <= float(cfg.audio.max_segment_s) + 0.1 for s in segs)
    # segments are ordered and non-overlapping beyond the pre-roll
    starts = [s.start_sample for s in segs]
    assert starts == sorted(starts)

    ids = import_file(cfg, db, SAMPLE_WAV, start_time=datetime(2026, 10, 31, 9, 0, 0), vad=silero)
    assert len(ids) >= 10
    rows = db.query("SELECT * FROM segments ORDER BY started_at")
    assert len(rows) == len(ids)
    assert all(r["speech_s"] > 0 and r["source"] == "import" for r in rows)
    files = list(cfg.audio_dir.rglob("*.opus"))
    assert len(files) == len(ids)
    stored = sum(f.stat().st_size for f in files)
    assert stored < SAMPLE_WAV.stat().st_size / 5  # opus + speech-only gating saves a lot
