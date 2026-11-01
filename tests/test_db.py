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


# --- speakers v2: migration, self speaker, relations, people_table, cluster_embeddings -------------------

def _old_schema() -> str:
    """The pre-v2 SCHEMA: no cluster_embeddings table (the speakers / utterances CREATE TABLEs never had the new columns)."""
    from audio_note.db import SCHEMA

    head, _sep, _tail = SCHEMA.partition("CREATE TABLE IF NOT EXISTS cluster_embeddings")
    assert "cluster_embeddings" not in head
    return head


def test_old_database_is_upgraded_in_place_and_idempotently(cfg):
    import sqlite3

    raw = sqlite3.connect(cfg.db_path)
    raw.executescript(_old_schema())
    raw.execute("INSERT INTO speakers(name) VALUES ('老用户')")
    raw.commit()
    raw.close()
    assert {r[1] for r in sqlite3.connect(cfg.db_path).execute("PRAGMA table_info(speakers)")} == {"id", "name", "notes", "created_at"}

    db = DB(cfg.db_path)
    spk_cols = {r["name"] for r in db.query("PRAGMA table_info(speakers)")}
    assert {"is_self", "relation", "relation_source", "relation_note", "relation_updated_at"} <= spk_cols
    assert "speaker_source" in {r["name"] for r in db.query("PRAGMA table_info(utterances)")}
    assert "cluster_embeddings" in {r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type='table'")}
    row = db.one("SELECT * FROM speakers WHERE name='老用户'")
    assert row["is_self"] == 0 and row["relation"] is None and row["relation_source"] is None
    # opening again (and calling _migrate directly) adds nothing more
    assert DB._migrate(db.conn()) == []
    DB(cfg.db_path)
    assert len([r for r in db.query("PRAGMA table_info(speakers)") if r["name"] == "is_self"]) == 1
    # a fresh database gets the same columns through the same path
    other = DB(cfg.db_path.parent / "fresh.sqlite3")
    assert {r["name"] for r in other.query("PRAGMA table_info(speakers)")} == spk_cols
    assert other.one("SELECT is_self FROM speakers") is None


def test_set_self_speaker_is_unique_and_clearable(db):
    a, b = db.upsert_speaker("甲"), db.upsert_speaker("乙")
    assert db.self_speaker() is None
    db.set_self_speaker(a)
    assert db.self_speaker()["id"] == a and db.self_speaker()["name"] == "甲"
    db.set_self_speaker(b)
    assert db.self_speaker()["id"] == b
    assert [r["is_self"] for r in db.query("SELECT is_self FROM speakers ORDER BY id")] == [0, 1]
    assert db.one("SELECT COUNT(*) AS n FROM speakers WHERE is_self=1")["n"] == 1
    db.set_self_speaker(None)
    assert db.self_speaker() is None and db.one("SELECT COUNT(*) AS n FROM speakers WHERE is_self=1")["n"] == 0
    db.set_self_speaker(999)                         # unknown id: nothing flagged, no error
    assert db.self_speaker() is None


def test_set_relation_writes_and_clears(db):
    sid = db.upsert_speaker("王敏")
    db.set_relation(sid, "同事", "user", note="一起做项目")
    row = db.one("SELECT * FROM speakers WHERE id=?", (sid,))
    assert row["relation"] == "同事" and row["relation_source"] == "user" and row["relation_note"] == "一起做项目"
    assert row["relation_updated_at"] and datetime.fromisoformat(row["relation_updated_at"])
    # an inferred ('auto') relation never replaces the user's own label: the write reports False
    assert db.set_relation(sid, "朋友", "auto", note="称呼老王") is False
    row = db.one("SELECT * FROM speakers WHERE id=?", (sid,))
    assert (row["relation"], row["relation_source"], row["relation_note"]) == ("同事", "user", "一起做项目")
    assert db.set_relation(sid, None, None) is True
    assert db.set_relation(sid, "朋友", "auto", note="称呼老王") is True
    row = db.one("SELECT * FROM speakers WHERE id=?", (sid,))
    assert (row["relation"], row["relation_source"], row["relation_note"]) == ("朋友", "auto", "称呼老王")
    assert db.set_relation(sid, "同事", "user") is True                     # the user's label replaces an inferred one
    assert db.one("SELECT relation_source FROM speakers WHERE id=?", (sid,))["relation_source"] == "user"
    for empty in (None, "", "  "):
        db.set_relation(sid, "领导", "user", note="x")
        db.set_relation(sid, empty, "user", note="still given")
        row = db.one("SELECT * FROM speakers WHERE id=?", (sid,))
        assert row["relation"] is None and row["relation_source"] is None and row["relation_note"] is None
        assert row["relation_updated_at"] is None
    db.set_relation(sid, "  导师 ", None)
    assert db.one("SELECT relation, relation_source FROM speakers WHERE id=?", (sid,))["relation"] == "导师"


def test_people_table_order_and_fields(db):
    seg = db.add_segment("s.opus", datetime(2026, 10, 31, 9, 0, 0), 60.0)
    me, wang, li, xiao = (db.upsert_speaker(n) for n in ("张伟", "王敏", "李强", "小雨"))
    rows = []
    for i, (sid, name) in enumerate([(wang, "王敏")] * 3 + [(me, "张伟")] * 1 + [(li, "李强")] * 2):
        rows.append(_utt(seg, i, "t", None, speaker_id=sid, speaker_name=name))
    db.add_utterances(rows)
    db.set_self_speaker(me)
    db.set_relation(wang, "同事", "user")
    db.set_relation(li, "朋友", "auto", note="叫他老李")
    people = db.people_table()
    assert [p["name"] for p in people] == ["张伟", "王敏", "李强", "小雨"]        # self first, then utterances desc
    assert people[0] == {"id": me, "name": "张伟", "is_self": True, "relation": None, "relation_source": None,
                         "relation_note": None, "utterances": 1}
    assert people[1] == {"id": wang, "name": "王敏", "is_self": False, "relation": "同事", "relation_source": "user",
                         "relation_note": None, "utterances": 3}
    assert people[2]["relation"] == "朋友" and people[2]["relation_source"] == "auto" and people[2]["relation_note"] == "叫他老李"
    assert people[3] == {"id": xiao, "name": "小雨", "is_self": False, "relation": None, "relation_source": None,
                         "relation_note": None, "utterances": 0}
    assert all(isinstance(p["is_self"], bool) for p in people)
    assert DB(db.path).people_table() == people
    db.execute("DELETE FROM speakers")
    assert db.people_table() == []


def test_cluster_embeddings_roundtrip_upsert_and_cascade(db):
    a = db.add_segment("a.opus", datetime(2026, 10, 31, 9, 0, 0), 60.0)
    b = db.add_segment("b.opus", datetime(2026, 10, 31, 10, 0, 0), 60.0)
    assert db.cluster_embeddings() == [] and db.cluster_embeddings([a]) == [] and db.cluster_embeddings([]) == []
    e0 = np.random.default_rng(0).standard_normal(192).astype(np.float32)
    e1 = np.arange(192, dtype=np.float64)                  # cast to float32
    db.upsert_cluster_embedding(a, 0, e0, 3.5)
    db.upsert_cluster_embedding(a, 1, e1, 1.25)
    db.upsert_cluster_embedding(b, 0, e0, 7.0)
    rows = db.cluster_embeddings()
    assert [(r[0], r[1], r[3]) for r in rows] == [(a, 0, 3.5), (a, 1, 1.25), (b, 0, 7.0)]
    np.testing.assert_array_equal(rows[0][2], e0)
    np.testing.assert_array_equal(rows[1][2], e1.astype(np.float32))
    assert rows[1][2].dtype == np.float32 and db.one("SELECT dim FROM cluster_embeddings WHERE segment_id=? AND cluster_label=1", (a,))["dim"] == 192
    assert [(r[0], r[1]) for r in db.cluster_embeddings([b])] == [(b, 0)]
    assert [(r[0], r[1]) for r in db.cluster_embeddings([b, a])] == [(a, 0), (a, 1), (b, 0)]
    # upsert replaces the embedding and speech_s of an existing (segment, cluster)
    db.upsert_cluster_embedding(a, 0, e1, 9.0)
    rows = db.cluster_embeddings([a])
    assert len(rows) == 2 and rows[0][3] == 9.0
    np.testing.assert_array_equal(rows[0][2], e1.astype(np.float32))
    assert db.one("SELECT COUNT(*) AS n FROM cluster_embeddings")["n"] == 3
    # explicit delete per segment, and ON DELETE CASCADE from segments
    assert db.delete_cluster_embeddings(a) == 2 and db.delete_cluster_embeddings(a) == 0
    assert [(r[0], r[1]) for r in db.cluster_embeddings()] == [(b, 0)]
    db.execute("DELETE FROM segments WHERE id=?", (b,))
    assert db.cluster_embeddings() == []


def test_add_utterances_accepts_speaker_source(db):
    seg = db.add_segment("s.opus", datetime(2026, 10, 31, 9, 0, 0), 60.0)
    ids = db.add_utterances([_utt(seg, 0, "a", None, speaker_source="user"), _utt(seg, 1, "b", None)])
    assert [r["speaker_source"] for r in db.query("SELECT speaker_source FROM utterances ORDER BY id")] == ["user", None]
    assert len(ids) == 2


# --- review fixes: concurrent upgrade, speaker delete, scan sentinel -----------------------------------------

def test_migrate_tolerates_a_concurrent_upgrade(cfg):
    """Finding 6: two processes upgrading the same old database at once - the loser's ALTER TABLE fails with
    'duplicate column name', which is ignored; any other OperationalError still propagates."""
    import sqlite3

    from audio_note.db import MIGRATION_COLUMNS

    raw = sqlite3.connect(cfg.db_path)
    raw.executescript(_old_schema())
    raw.close()
    db = DB(cfg.db_path)                                            # the winner upgraded the file
    assert "is_self" in {r["name"] for r in db.query("PRAGMA table_info(speakers)")}

    class StalePragma:
        """The loser: its PRAGMA table_info ran before the winner's ALTER, so every column looks missing."""

        def __init__(self, conn):
            self.conn, self.alters = conn, []

        def execute(self, sql, *a):
            if sql.startswith("PRAGMA"):
                return type("Cur", (), {"fetchall": lambda s: []})()
            self.alters.append(sql)
            return self.conn.execute(sql, *a)

    loser = StalePragma(db.conn())
    assert DB._migrate(loser) == [] and len(loser.alters) == len(MIGRATION_COLUMNS)
    assert len([r for r in db.query("PRAGMA table_info(speakers)") if r["name"] == "is_self"]) == 1

    class Locked(StalePragma):
        def execute(self, sql, *a):
            if sql.startswith("ALTER"):
                raise sqlite3.OperationalError("database is locked")
            return super().execute(sql, *a)

    with pytest.raises(sqlite3.OperationalError, match="locked"):
        DB._migrate(Locked(db.conn()))


def test_delete_speaker_releases_rows_and_relation_cache(db):
    """Finding 1: the rows go back to unidentified without a 'user' mark; the kv cache of the id is dropped."""
    seg = db.add_segment("s.opus", datetime(2026, 10, 31, 9, 0, 0), 60.0)
    a = db.upsert_speaker("甲")
    b = db.upsert_speaker("乙")
    db.add_utterances([_utt(seg, 0, "x", None, speaker_id=a, speaker_name="甲", speaker_score=0.9, speaker_source="user"),
                       _utt(seg, 1, "y", None, speaker_id=a, speaker_name="甲"),
                       _utt(seg, 2, "z", None, speaker_id=b, speaker_name="乙", speaker_source="user")])
    db.set_kv(f"relation_inferred_n:{a}", 9)
    db.set_kv(f"relation_inferred_n:{b}", 9)
    assert db.delete_speaker(a, "未知") == 2
    rows = db.query("SELECT speaker_id, speaker_name, speaker_score, speaker_source FROM utterances ORDER BY id")
    assert [tuple(r) for r in rows[:2]] == [(None, "未知", None, None)] * 2
    assert (rows[2]["speaker_id"], rows[2]["speaker_source"]) == (b, "user")
    assert db.get_kv(f"relation_inferred_n:{a}") is None and db.get_kv(f"relation_inferred_n:{b}") == 9
    assert db.one("SELECT COUNT(*) AS n FROM speakers")["n"] == 1 and db.delete_speaker(999, "未知") == 0


def test_cluster_embeddings_sentinel_is_hidden_and_cleared(db):
    """Finding 5: the 'scanned, nothing to embed' sentinel row is invisible to cluster_embeddings() and
    removed together with the real rows."""
    a = db.add_segment("a.opus", datetime(2026, 10, 31, 9, 0, 0), 60.0)
    db.mark_cluster_embeddings_empty(a)
    db.mark_cluster_embeddings_empty(a)                                   # idempotent
    assert db.cluster_embeddings() == [] and db.cluster_embeddings([a]) == []
    row = db.one("SELECT cluster_label, dim, speech_s, embedding FROM cluster_embeddings WHERE segment_id=?", (a,))
    assert (row["cluster_label"], row["dim"], row["speech_s"], bytes(row["embedding"])) == (-1, 0, 0.0, b"")
    db.upsert_cluster_embedding(a, 0, np.ones(4, dtype=np.float32), 1.0)
    assert [r[1] for r in db.cluster_embeddings([a])] == [0]
    assert db.delete_cluster_embeddings(a) == 2 and db.cluster_embeddings() == []
