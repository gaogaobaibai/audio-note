"""Tests for audio_note.audio.storage: compression mapping, file roundtrip, finalize, partial recovery, cleanup."""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_note.audio import storage
from audio_note.audio.storage import (
    PartialWriter, cleanup, dir_size_bytes, finalize_segment, opus_compression_level, read_segment,
    recover_partials, segment_path, write_segment_file,
)
from audio_note.audio.vad import SAMPLE_RATE

from .conftest import FakeVAD, noise, silence, sine


def test_opus_compression_level_mapping():
    assert opus_compression_level(24000) == pytest.approx((257 - 24) / 250)
    assert 0.9 < opus_compression_level(24000) < 0.95
    assert opus_compression_level(132000) == pytest.approx(0.5)
    assert opus_compression_level(0) == 0.99          # clamped high
    assert opus_compression_level(300000) == 0.0      # clamped low
    assert opus_compression_level(64000) > opus_compression_level(96000)  # lower bitrate -> higher level


def test_segment_path_layout(cfg):
    t = datetime(2026, 10, 31, 8, 19, 10, 7000)
    p = segment_path(cfg, t, ".opus")
    assert p == cfg.audio_dir / "2026-10-31" / "081910_007.opus"
    assert p.parent.is_dir()
    assert cfg.data_dir in p.parents  # never outside the temp data dir


@pytest.mark.parametrize("codec,ext", [("opus", ".opus"), ("flac", ".flac"), ("wav", ".wav")])
def test_write_read_roundtrip(cfg_factory, codec, ext):
    cfg = cfg_factory({"audio": {"codec": codec}})
    t = datetime(2026, 10, 31, 8, 19, 10, 500000)
    audio = sine(1.0, 440.0, amp=0.3)
    p = write_segment_file(cfg, audio, t)
    assert p.suffix == ext and p.exists() and p.stat().st_size > 0
    back = read_segment(p)
    assert back.dtype == np.float32 and back.ndim == 1
    # opus may add a few ms of codec delay; lengths must agree within 60 ms
    assert abs(len(back) - len(audio)) <= int(0.06 * SAMPLE_RATE)
    n = min(len(back), len(audio))
    if codec == "opus":
        # correlate allowing for a small shift
        best = max(float(np.corrcoef(audio[: n - 400], back[s: s + n - 400])[0, 1]) for s in range(0, 400, 8))
        assert best > 0.8
    else:
        np.testing.assert_allclose(back[:n], audio[:n], atol=2e-4)


def test_opus_is_much_smaller_than_wav(cfg_factory):
    cfg_o = cfg_factory({"audio": {"codec": "opus"}}, name="o.yaml")
    audio = noise(4.0, amp=0.05)
    p_opus = write_segment_file(cfg_o, audio, datetime(2026, 1, 1, 0, 0, 0))
    cfg_w = cfg_factory({"audio": {"codec": "wav"}}, name="w.yaml")
    p_wav = write_segment_file(cfg_w, audio, datetime(2026, 1, 1, 0, 0, 1))
    assert p_opus.stat().st_size < p_wav.stat().st_size / 4


def test_read_segment_resamples_and_downmixes(tmp_path):
    p = tmp_path / "stereo48k.wav"
    sr = 48000
    t = np.arange(sr) / sr
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t)], axis=1).astype(np.float32) * 0.5
    sf.write(p, stereo, sr)
    a = read_segment(p)
    assert a.ndim == 1 and len(a) == SAMPLE_RATE
    assert np.abs(a).max() == pytest.approx(0.5, abs=0.02)


def test_partial_writer(cfg):
    t = datetime(2026, 10, 31, 10, 0, 0, 42000)
    w = PartialWriter(cfg, t)
    assert w.path.name == "100000_042_partial.wav"
    w.write(np.zeros(512, dtype=np.float32))
    w.write(np.ones(512, dtype=np.float32) * 0.1)
    assert w.frames == 1024
    w.close()
    a, sr = sf.read(w.path, dtype="float32")
    assert sr == SAMPLE_RATE and len(a) == 1024
    w2 = PartialWriter(cfg, t + timedelta(seconds=1))
    w2.write(np.zeros(100, dtype=np.float32))
    w2.abandon()
    assert not w2.path.exists()
    w2.abandon()  # second abandon is harmless


def test_finalize_segment_drops_when_speech_below_min(cfg, db):
    t = datetime(2026, 10, 31, 10, 0, 0)
    partial = segment_path(cfg, t, "_partial.wav")
    sf.write(partial, silence(1.0), SAMPLE_RATE)
    assert float(cfg.storage.min_speech_total_s) == 1.0
    sid = finalize_segment(cfg, db, silence(3.0), t, speech_s=0.9, partial=partial)
    assert sid is None
    assert not partial.exists()
    assert db.query("SELECT * FROM segments") == []
    assert list(cfg.audio_dir.glob("*/*.opus")) == []


def test_finalize_segment_writes_file_and_row(cfg, db):
    t = datetime(2026, 10, 31, 10, 0, 0, 250000)
    partial = segment_path(cfg, t, "_partial.wav")
    sf.write(partial, silence(1.0), SAMPLE_RATE)
    audio = noise(2.0, amp=0.05)
    sid = finalize_segment(cfg, db, audio, t, speech_s=1.5, source="import", origin="orig.mp3", partial=partial)
    assert isinstance(sid, int)
    assert not partial.exists()
    row = db.get_segment(sid)
    p = Path(row["path"])
    assert p == cfg.audio_dir / "2026-10-31" / "100000_250.opus" and p.exists()
    assert row["duration_s"] == pytest.approx(2.0)
    assert row["speech_s"] == 1.5 and row["source"] == "import" and row["origin"] == "orig.mp3"
    assert row["status"] == "new" and row["started_at"] == "2026-10-31T10:00:00.250"
    # exactly on the threshold is kept
    sid2 = finalize_segment(cfg, db, audio, t + timedelta(seconds=5), speech_s=1.0)
    assert sid2 is not None


def _old(p: Path, seconds: float = 60.0) -> None:
    ts = time.time() - seconds
    os.utime(p, (ts, ts))


def test_recover_partials_finalizes_with_proper_name(cfg, db):
    day = cfg.audio_dir / "2026-10-30"
    day.mkdir()
    p = day / "120000_123_partial.wav"
    sf.write(p, noise(2.0, amp=0.05), SAMPLE_RATE)
    _old(p)
    n = recover_partials(cfg, db, FakeVAD(default=1.0))
    assert n == 1
    assert not p.exists()
    rows = db.query("SELECT * FROM segments")
    assert len(rows) == 1
    r = rows[0]
    assert r["started_at"] == "2026-10-30T12:00:00.123"
    assert Path(r["path"]) == day / "120000_123.opus" and Path(r["path"]).exists()
    assert r["source"] == "mic" and r["speech_s"] == pytest.approx(2.0, abs=0.05)
    # second run finds nothing
    assert recover_partials(cfg, db, FakeVAD(default=1.0)) == 0


def test_recover_partials_skips_fresh_deletes_empty_and_silent(cfg, db):
    day = cfg.audio_dir / "2026-10-30"
    day.mkdir()
    fresh = day / "120000_000_partial.wav"
    sf.write(fresh, noise(1.0), SAMPLE_RATE)  # mtime = now -> skipped
    empty = day / "120100_000_partial.wav"
    sf.write(empty, np.zeros(0, dtype=np.float32), SAMPLE_RATE)
    _old(empty)
    silent = day / "120200_000_partial.wav"
    sf.write(silent, silence(3.0), SAMPLE_RATE)
    _old(silent)
    n = recover_partials(cfg, db, FakeVAD(default=0.0))
    assert n == 0
    assert fresh.exists()
    assert not empty.exists()
    assert not silent.exists()  # finalize dropped it -> partial removed
    assert db.query("SELECT * FROM segments") == []


def test_recover_partials_survives_bad_file(cfg, db, caplog):
    day = cfg.audio_dir / "2026-10-30"
    day.mkdir()
    bad = day / "garbage_partial.wav"
    bad.write_bytes(b"not a wav")
    _old(bad)
    assert recover_partials(cfg, db, FakeVAD(default=1.0)) == 0
    assert bad.exists()  # left for the human to inspect


def _seg_with_file(cfg, db, started: datetime, status: str, speech_s=None, seconds: float = 1.0, codec_ext=".wav"):
    p = segment_path(cfg, started, codec_ext)
    sf.write(p, noise(seconds, amp=0.05), SAMPLE_RATE)
    sid = db.add_segment(p, started, seconds, speech_s=speech_s)
    if status != "new":
        db.set_segment_status(sid, status)
    return sid, p


def test_cleanup_recheck_marks_no_speech(cfg_factory, db):
    cfg = cfg_factory({"storage": {"keep_audio_days": 0, "max_disk_gb": 0}})
    now = datetime.now()
    s_quiet, p_quiet = _seg_with_file(cfg, db, now - timedelta(minutes=10), "new", speech_s=None, seconds=4.0)
    s_known, p_known = _seg_with_file(cfg, db, now - timedelta(minutes=9), "new", speech_s=2.0)
    s_done, p_done = _seg_with_file(cfg, db, now - timedelta(minutes=8), "done", speech_s=None)
    stats = cleanup(cfg, db, FakeVAD(default=0.0))
    assert stats["rechecked"] == 2 and stats["no_speech"] == 1
    assert stats["expired"] == 0 and stats["disk_cap"] == 0
    assert db.get_segment(s_quiet)["status"] == "no_speech" and not p_quiet.exists()
    assert db.get_segment(s_known)["status"] == "new" and p_known.exists()  # speech_s already set -> untouched
    d = db.get_segment(s_done)
    assert d["status"] == "done" and p_done.exists() and d["speech_s"] == 0.0  # done rows get measured, never deleted here


def test_cleanup_recheck_keeps_speech_and_ratio(cfg_factory, db):
    cfg = cfg_factory({"storage": {"keep_audio_days": 0, "max_disk_gb": 0, "min_speech_ratio": 0.15}})
    now = datetime.now()
    # all windows speech -> ratio 1.0, speech_s ~ 4 s -> kept
    s_ok, p_ok = _seg_with_file(cfg, db, now - timedelta(minutes=10), "new", seconds=4.0)
    stats = cleanup(cfg, db, FakeVAD(default=1.0))
    assert stats["no_speech"] == 0
    r = db.get_segment(s_ok)
    assert r["status"] == "new" and p_ok.exists() and r["speech_s"] == pytest.approx(4.0, abs=0.05)
    # enough absolute speech but tiny ratio -> dropped (10 s file, only 1.0 s speech ~ 0.1 ratio)
    s_lo, p_lo = _seg_with_file(cfg, db, now - timedelta(minutes=5), "new", seconds=10.0)
    probs = [1.0] * 32  # 32 windows ≈ 1.02 s >= min_speech_total_s, ratio ≈ 0.1 < 0.15
    stats = cleanup(cfg, db, FakeVAD(probs, default=0.0))
    assert stats["no_speech"] == 1
    assert db.get_segment(s_lo)["status"] == "no_speech" and not p_lo.exists()


def test_cleanup_unreadable_marks_failed(cfg_factory, db):
    cfg = cfg_factory({"storage": {"keep_audio_days": 0, "max_disk_gb": 0}})
    p = cfg.audio_dir / "2026-10-30" / "bad.wav"
    p.parent.mkdir()
    p.write_bytes(b"xx")
    sid = db.add_segment(p, datetime(2026, 10, 30, 1, 0, 0), 1.0)
    cleanup(cfg, db, FakeVAD())
    r = db.get_segment(sid)
    assert r["status"] == "failed" and "unreadable" in r["error"]


def test_cleanup_expiry_only_done_and_only_old(cfg_factory, db):
    cfg = cfg_factory({"storage": {"keep_audio_days": 30, "max_disk_gb": 0}})
    now = datetime.now()
    old_done, p1 = _seg_with_file(cfg, db, now - timedelta(days=40), "done", speech_s=1.0)
    old_new, p2 = _seg_with_file(cfg, db, now - timedelta(days=41), "new", speech_s=1.0)  # different day folder
    new_done, p3 = _seg_with_file(cfg, db, now - timedelta(days=5), "done", speech_s=1.0)
    stats = cleanup(cfg, db, vad=None)
    assert stats == {"rechecked": 0, "no_speech": 0, "expired": 1, "disk_cap": 0}
    assert db.get_segment(old_done)["status"] == "deleted" and not p1.exists()
    assert db.get_segment(old_new)["status"] == "new" and p2.exists()
    assert db.get_segment(new_done)["status"] == "done" and p3.exists()
    # the day folder of the deleted file is now empty -> removed
    assert not p1.parent.exists()
    assert p2.parent.exists()
    # already-deleted rows are not re-processed
    assert cleanup(cfg, db, vad=None)["expired"] == 0


def test_cleanup_keep_forever_when_zero(cfg_factory, db):
    cfg = cfg_factory({"storage": {"keep_audio_days": 0, "max_disk_gb": 0}})
    sid, p = _seg_with_file(cfg, db, datetime.now() - timedelta(days=400), "done", speech_s=1.0)
    assert cleanup(cfg, db)["expired"] == 0
    assert p.exists() and db.get_segment(sid)["status"] == "done"


def test_cleanup_disk_cap_deletes_oldest_done_first(cfg_factory, db):
    # tiny cap: 1 byte above 0 -> any audio exceeds it
    cfg = cfg_factory({"storage": {"keep_audio_days": 0, "max_disk_gb": 1e-9}})
    now = datetime.now()
    a, pa = _seg_with_file(cfg, db, now - timedelta(hours=3), "done", speech_s=1.0, seconds=2.0)
    b, pb = _seg_with_file(cfg, db, now - timedelta(hours=2), "done", speech_s=1.0, seconds=2.0)
    c, pc = _seg_with_file(cfg, db, now - timedelta(hours=1), "new", speech_s=1.0, seconds=2.0)
    assert dir_size_bytes(cfg.audio_dir) > 1
    stats = cleanup(cfg, db)
    assert stats["disk_cap"] == 2
    assert db.get_segment(a)["status"] == "deleted" and not pa.exists()
    assert db.get_segment(b)["status"] == "deleted" and not pb.exists()
    assert db.get_segment(c)["status"] == "new" and pc.exists()  # untranscribed audio is never purged


def test_finalize_same_timestamp_never_overwrites(cfg, db):
    """Two segments with the same started_at (two imports with the same --start) get distinct files/rows."""
    t = datetime(2026, 10, 31, 9, 0, 0)
    a = sine(2.0, 440.0, amp=0.2)
    b = sine(2.0, 880.0, amp=0.3)
    s1 = finalize_segment(cfg, db, a, t, speech_s=1.5, source="import", origin="a.wav")
    s2 = finalize_segment(cfg, db, b, t, speech_s=1.5, source="import", origin="b.wav")
    assert s1 is not None and s2 is not None and s1 != s2
    p1, p2 = Path(db.get_segment(s1)["path"]), Path(db.get_segment(s2)["path"])
    assert p1 != p2 and p1.exists() and p2.exists()
    assert p1.name == "090000_000.opus" and p2.name == "090000_000_1.opus"
    assert np.abs(read_segment(p1)).max() < np.abs(read_segment(p2)).max()   # a's audio untouched
    # a row whose file was expired still blocks its path: the next one goes to _2
    p2.unlink()
    s3 = finalize_segment(cfg, db, b, t, speech_s=1.5)
    assert Path(db.get_segment(s3)["path"]).name == "090000_000_2.opus"


def test_finalize_drops_long_segment_with_tiny_speech_ratio(cfg, db):
    """A gate kept open by hysteresis-band noise: 20 s with 1.5 s of speech (ratio 0.075 < 0.15)."""
    assert float(cfg.storage.min_speech_ratio) == 0.15
    t = datetime(2026, 10, 31, 11, 0, 0)
    partial = segment_path(cfg, t, "_partial.wav")
    sf.write(partial, silence(1.0), SAMPLE_RATE)
    assert finalize_segment(cfg, db, noise(20.0, amp=0.05), t, speech_s=1.5, partial=partial) is None
    assert not partial.exists() and db.query("SELECT * FROM segments") == []
    # the ratio rule does not touch short segments (a brief reply with pre/post-roll is legitimately low)
    assert finalize_segment(cfg, db, noise(5.0, amp=0.05), t, speech_s=1.0) is not None
    # and a long segment with real speech is kept
    assert finalize_segment(cfg, db, noise(20.0, amp=0.05), t + timedelta(minutes=1), speech_s=6.0) is not None


def test_cleanup_rechecks_new_rows_with_low_stored_ratio(cfg_factory, db):
    cfg = cfg_factory({"storage": {"keep_audio_days": 0, "max_disk_gb": 0}})
    now = datetime.now()
    # stored ratio 1.2/12 = 0.1 < 0.15 (a row written before the finalize-time rule existed)
    s_lo, p_lo = _seg_with_file(cfg, db, now - timedelta(minutes=10), "new", speech_s=1.2, seconds=12.0)
    # re-measured with a VAD that finds speech everywhere -> kept, speech_s updated
    stats = cleanup(cfg, db, FakeVAD(default=1.0))
    assert stats["rechecked"] == 1 and stats["no_speech"] == 0
    assert db.get_segment(s_lo)["speech_s"] == pytest.approx(12.0, abs=0.05) and p_lo.exists()
    # same row, VAD now confirms the low ratio -> deleted
    db.execute("UPDATE segments SET speech_s=1.2 WHERE id=?", (s_lo,))
    stats = cleanup(cfg, db, FakeVAD([1.0] * 38, default=0.0))
    assert stats["no_speech"] == 1 and not p_lo.exists()


def test_cleanup_expires_old_failed_segments_too(cfg_factory, db):
    cfg = cfg_factory({"storage": {"keep_audio_days": 30, "max_disk_gb": 0}})
    now = datetime.now()
    old_failed, p = _seg_with_file(cfg, db, now - timedelta(days=45), "failed", speech_s=2.0)
    fresh_failed, p2 = _seg_with_file(cfg, db, now - timedelta(days=2), "failed", speech_s=2.0)
    assert cleanup(cfg, db)["expired"] == 1
    assert db.get_segment(old_failed)["status"] == "deleted" and not p.exists()
    assert db.get_segment(fresh_failed)["status"] == "failed" and p2.exists()


def test_cleanup_removes_empty_day_dirs(cfg_factory, db):
    cfg = cfg_factory({"storage": {"keep_audio_days": 0, "max_disk_gb": 0}})
    empty = cfg.audio_dir / "2026-01-01"
    empty.mkdir()
    full = cfg.audio_dir / "2026-01-02"
    full.mkdir()
    (full / "x.opus").write_bytes(b"0")
    cleanup(cfg, db)
    assert not empty.exists() and full.exists()
    assert dir_size_bytes(cfg.audio_dir) == 1
