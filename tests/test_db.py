"""Tests for audio_note.db: schema, segments, speakers/embeddings, utterances + FTS triggers, kv."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from audio_note.db import DB

EXPECTED_TABLES = {
    "segments", "transcripts", "speakers", "speaker_embeddings", "utterances", "utterance_tokens",
    "utterances_fts", "daily_summaries", "memory_notes", "kv",
}


def test_schema_created_and_idempotent(cfg):
    db = DB(cfg.db_path)
    names = {r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    assert EXPECTED_TABLES <= names
    triggers = {r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type='trigger'")}
    assert {"utt_tok_ai", "utt_tok_ad", "utt_tok_au"} <= triggers
    assert db.one("PRAGMA journal_mode")[0] == "wal"
    assert db.one("PRAGMA foreign_keys")[0] == 1
    # opening again on the same file must not fail (CREATE IF NOT EXISTS everywhere)
    DB(cfg.db_path)
    assert cfg.db_path.exists()


def test_add_segment_and_status(db):
    t = datetime(2026, 10, 31, 9, 15, 0, 123000)
    sid = db.add_segment("x/091500_123.opus", t, 12.5, source="mic", speech_s=3.2)
    row = db.get_segment(sid)
    assert row["path"] == "x/091500_123.opus"
    assert row["started_at"] == "2026-10-31T09:15:00.123"
    assert row["duration_s"] == 12.5 and row["sample_rate"] == 16000
    assert row["status"] == "new" and row["speech_s"] == 3.2 and row["error"] is None
    db.set_segment_status(sid, "failed", "boom")
    row = db.get_segment(sid)
    assert row["status"] == "failed" and row["error"] == "boom"
    assert db.get_segment(999) is None
    # path is unique
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        db.add_segment("x/091500_123.opus", t, 1.0)


def test_pending_segments_age_filter(db):
    t = datetime(2026, 10, 31, 9, 0, 0)
    a = db.add_segment("a.opus", t, 1.0)
    b = db.add_segment("b.opus", datetime(2026, 10, 31, 8, 0, 0), 1.0)
    done = db.add_segment("c.opus", t, 1.0)
    db.set_segment_status(done, "done")
    # fresh rows are visible with no age requirement, hidden behind a 1h age
    ids = [r["id"] for r in db.pending_segments(0)]
    assert ids == [b, a]  # ordered by started_at
    assert db.pending_segments(3600) == []
    # back-date one row's created_at -> it becomes pending
    db.execute("UPDATE segments SET created_at=datetime('now','localtime','-2 hours') WHERE id=?", (a,))
    assert [r["id"] for r in db.pending_segments(3600)] == [a]
    assert len(db.pending_segments(0, limit=1)) == 1


def test_speakers_and_embeddings_roundtrip(db):
    sid = db.upsert_speaker("张三", "同事")
    assert db.upsert_speaker("张三") == sid
    assert db.one("SELECT notes FROM speakers WHERE id=?", (sid,))["notes"] == "同事"
    db.upsert_speaker("张三", "更新")
    assert db.one("SELECT notes FROM speakers WHERE id=?", (sid,))["notes"] == "更新"
    other = db.upsert_speaker("李四")
    assert other != sid

    e1 = np.random.default_rng(1).standard_normal(192).astype(np.float32)
    e2 = np.arange(192, dtype=np.float64)  # float64 input is cast to float32
    db.add_speaker_embedding(sid, e1, "a.wav", 3.5)
    db.add_speaker_embedding(other, e2, None, None)
    rows = db.speaker_embeddings()
    assert len(rows) == 2
    by_id = {r[0]: r for r in rows}
    assert by_id[sid][1] == "张三"
    np.testing.assert_array_equal(by_id[sid][2], e1)
    assert by_id[sid][2].dtype == np.float32
    np.testing.assert_array_equal(by_id[other][2], e2.astype(np.float32))
    assert db.one("SELECT dim FROM speaker_embeddings WHERE speaker_id=?", (sid,))["dim"] == 192
    # cascade: deleting a speaker drops its embeddings
    db.execute("DELETE FROM speakers WHERE id=?", (other,))
    assert len(db.speaker_embeddings()) == 1


def _utt(seg: int, i: int, text: str, text_seg: str | None, **kw):
    r = {"segment_id": seg, "start_s": float(i), "end_s": float(i + 1), "abs_start": f"2026-10-31T09:00:{i:02d}",
         "day": "2026-10-31", "text_raw": text, "text": text}
    if text_seg is not None:
        r["text_seg"] = text_seg
    r.update(kw)
    return r


def test_add_utterances_and_fts_triggers(db):
    seg = db.add_segment("s.opus", datetime(2026, 10, 31, 9, 0, 0), 60.0)
    spk = db.upsert_speaker("张三")
    ids = db.add_utterances([
        _utt(seg, 0, "今天我们讨论一下预算", "今天 我们 讨论 一下 预算", speaker_id=spk, speaker_name="张三", speaker_score=0.8, cluster_label=0),
        _utt(seg, 1, "好的", "好的"),
        _utt(seg, 2, "no tokens", None),
    ])
    assert len(ids) == 3
    rows = db.utterances_for_day("2026-10-31")
    assert [r["id"] for r in rows] == ids
    assert rows[0]["speaker_name"] == "张三" and rows[0]["speaker_id"] == spk and rows[0]["cluster_label"] == 0
    assert db.utterances_for_day("2026-11-01") == []
    assert db.one("SELECT COUNT(*) AS n FROM utterance_tokens")["n"] == 2

    hits = db.query("SELECT rowid FROM utterances_fts WHERE utterances_fts MATCH ?", ("预算",))
    assert [h["rowid"] for h in hits] == [ids[0]]
    hits = db.query("SELECT rowid FROM utterances_fts WHERE utterances_fts MATCH ? ORDER BY bm25(utterances_fts)", ("讨论 AND 预算",))
    assert [h["rowid"] for h in hits] == [ids[0]]
    assert db.query("SELECT rowid FROM utterances_fts WHERE utterances_fts MATCH ?", ("不存在",)) == []

    # update trigger keeps the index in sync
    db.execute("UPDATE utterance_tokens SET text_seg=? WHERE utterance_id=?", ("明天 开会", ids[0]))
    assert db.query("SELECT rowid FROM utterances_fts WHERE utterances_fts MATCH ?", ("预算",)) == []
    assert [h["rowid"] for h in db.query("SELECT rowid FROM utterances_fts WHERE utterances_fts MATCH ?", ("开会",))] == [ids[0]]

    # deleting utterances cascades to tokens and (via trigger) out of the FTS index
    db.delete_utterances_for_segment(seg)
    assert db.utterances_for_day("2026-10-31") == []
    assert db.one("SELECT COUNT(*) AS n FROM utterance_tokens")["n"] == 0
    assert db.query("SELECT rowid FROM utterances_fts WHERE utterances_fts MATCH ?", ("开会",)) == []
    assert db.query("SELECT rowid FROM utterances_fts WHERE utterances_fts MATCH ?", ("好的",)) == []


def test_add_utterances_is_transactional(db):
    seg = db.add_segment("s.opus", datetime(2026, 10, 31, 9, 0, 0), 60.0)
    bad = _utt(seg, 1, "x", "x")
    del bad["text"]  # KeyError inside the transaction -> rollback
    with pytest.raises(KeyError):
        db.add_utterances([_utt(seg, 0, "ok", "ok"), bad])
    assert db.utterances_for_day("2026-10-31") == []
    assert db.one("SELECT COUNT(*) AS n FROM utterance_tokens")["n"] == 0
    # the connection is usable afterwards
    assert db.add_utterances([_utt(seg, 0, "ok", "ok")])


def test_speaker_delete_sets_utterance_speaker_null(db):
    seg = db.add_segment("s.opus", datetime(2026, 10, 31, 9, 0, 0), 60.0)
    spk = db.upsert_speaker("王五")
    uid = db.add_utterances([_utt(seg, 0, "hi", "hi", speaker_id=spk, speaker_name="王五")])[0]
    db.execute("DELETE FROM speakers WHERE id=?", (spk,))
    row = db.one("SELECT speaker_id, speaker_name FROM utterances WHERE id=?", (uid,))
    assert row["speaker_id"] is None and row["speaker_name"] == "王五"


def test_kv_roundtrip(db):
    assert db.get_kv("missing") is None
    assert db.get_kv("missing", "d") == "d"
    db.set_kv("last_daily_day", "2026-10-30")
    db.set_kv("obj", {"a": [1, 2], "中": "文"})
    assert db.get_kv("last_daily_day") == "2026-10-30"
    assert db.get_kv("obj") == {"a": [1, 2], "中": "文"}
    db.set_kv("last_daily_day", "2026-10-31")
    assert db.get_kv("last_daily_day") == "2026-10-31"


def test_transaction_helper_rolls_back(db):
    with pytest.raises(RuntimeError):
        with db.transaction() as c:
            c.execute("INSERT INTO kv(key,value) VALUES ('k','1')")
            raise RuntimeError("abort")
    assert db.get_kv("k") is None
    with db.transaction() as c:
        c.execute("INSERT INTO kv(key,value) VALUES ('k','2')")
    assert db.get_kv("k") == 2


def test_per_thread_connections(db):
    import threading

    conns = {}

    def worker():
        conns["t"] = db.conn()
        db.set_kv("from_thread", 1)

    th = threading.Thread(target=worker)
    th.start()
    th.join()
    assert conns["t"] is not db.conn()
    assert db.get_kv("from_thread") == 1
