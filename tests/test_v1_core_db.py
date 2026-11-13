"""audio_note.db 1.0: versioned migrations (PRAGMA user_version), the pre-upgrade backup, the refusal of newer
files, the v2 data steps, re-entrant transactions and the small generic helpers (usage, job history, maintenance).

Old databases are built from the pre-1.0 part of SCHEMA with plain sqlite3, exactly like tests/test_db.py does.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from audio_note import db as db_mod
from audio_note.db import (BACKUP_KEEP, DB, MIGRATION_COLUMNS, MIGRATIONS, SCHEMA, SCHEMA_VERSION, STOPPED_ERROR, MigrationError,
                           SchemaTooNew, memory_fingerprint, now_iso)

NEW_TABLES = {"lexicon", "lexicon_aliases", "utterance_revisions", "inbox_items", "utterance_voice", "speaker_centroids", "voices",
              "sessions", "memory_items", "memory_item_tokens", "memory_items_fts", "memory_vectors", "chat_conversations",
              "chat_messages", "usage_local", "usage_llm", "job_runs"}
NEW_COLUMNS = {
    "utterances": {"session_id", "text_source", "edited_at", "flags", "speaker_guess_id", "speaker_guess_score", "speaker_meta"},
    "speaker_embeddings": {"model"}, "speaker_negatives": {"model"},
    "cluster_embeddings": {"model", "voice_id", "loudness_db", "system_share"},
    "segments": {"realtime"},
}


def _old_schema() -> str:
    """The 0.3.x tables: everything before the 1.0 block of SCHEMA (no migrated columns, user_version 0)."""
    head, sep, _tail = SCHEMA.partition("-- ===== 1.0 (schema version 2)")
    assert sep, "the 1.0 block marker moved"
    return head


def _make_old_db(path: Path, segments: int = 2, notes: list[tuple] | None = None) -> None:
    """A 0.3.x database with `segments` recordings (one of them stamped in SQLite's space format) and notes."""
    raw = sqlite3.connect(path)
    raw.executescript(_old_schema())
    for i in range(segments):
        raw.execute("INSERT INTO segments(path, started_at, duration_s, status, updated_at) VALUES (?,?,?,?,?)",
                    (f"a{i}.opus", f"2026-11-0{i + 1}T09:00:00.000", 10.0, "done",
                     "2026-11-01 09:00:05" if i == 0 else "2026-11-02T09:00:05"))
    for row in notes or []:
        raw.execute("INSERT INTO memory_notes(kind, title, content, day_from, day_to, source_utterance_ids) VALUES (?,?,?,?,?,?)", row)
    raw.commit()
    assert raw.execute("PRAGMA user_version").fetchone()[0] == 0
    raw.close()


def _names(db: DB, kind: str = "table") -> set[str]:
    return {r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type=?", (kind,))}


# --- schema ---------------------------------------------------------------------------------------------------

def test_fresh_database_has_every_1_0_table_column_and_index(cfg):
    db = DB(cfg.db_path)
    assert db.schema_version() == SCHEMA_VERSION == 2
    assert db.migrated == (2,) and db.last_backup is None                 # nothing to back up in an empty file
    assert NEW_TABLES <= _names(db) | _names(db, "view")
    for table, cols in NEW_COLUMNS.items():
        assert cols <= {r["name"] for r in db.query(f"PRAGMA table_info({table})")}, table
    idx = _names(db, "index")
    assert {"idx_utt_session", "idx_cluster_voice", "idx_lex_alias", "idx_rev_utt", "idx_inbox_open_key", "idx_inbox_status",
            "idx_sessions_day", "idx_sessions_started", "idx_items_kind", "idx_items_subject", "idx_items_fp"} <= idx
    assert {"item_tok_ai", "item_tok_ad", "item_tok_au"} <= _names(db, "trigger")
    assert not (cfg.data_dir / "backups").exists()
    # a second open is a no-op
    again = DB(cfg.db_path)
    assert again.migrated == () and again.last_backup is None and again.schema_version() == 2


def test_migrated_columns_are_not_referenced_by_schema_objects():
    """SCHEMA runs before the columns exist on an old file: an index on one of them must live in a data step."""
    for table, col, _decl in MIGRATION_COLUMNS:
        for stmt in SCHEMA.split(";"):
            if "CREATE INDEX" in stmt.upper() or "CREATE UNIQUE INDEX" in stmt.upper():
                assert not (f"ON {table}(" in stmt and col in stmt.split("(", 1)[1]), (table, col, stmt.strip())
    assert [m[0] for m in MIGRATIONS] == sorted(m[0] for m in MIGRATIONS) and MIGRATIONS[-1][0] == SCHEMA_VERSION


def test_memory_items_fts_follows_its_token_table(db):
    iid = int(db.execute("INSERT INTO memory_items(kind, title) VALUES ('todo', '买牛奶')").lastrowid)
    db.execute("INSERT INTO memory_item_tokens(item_id, text_seg) VALUES (?, ?)", (iid, "买 牛奶"))
    hit = lambda q: [r[0] for r in db.query("SELECT rowid FROM memory_items_fts WHERE memory_items_fts MATCH ?", (q,))]  # noqa: E731
    assert hit("牛奶") == [iid]
    db.execute("UPDATE memory_item_tokens SET text_seg='买 豆浆' WHERE item_id=?", (iid,))
    assert hit("牛奶") == [] and hit("豆浆") == [iid]
    db.execute("DELETE FROM memory_items WHERE id=?", (iid,))                 # cascades to the tokens, the trigger cleans the index
    assert hit("豆浆") == []


def test_inbox_open_dedupe_key_is_unique_only_while_open(db):
    db.execute("INSERT INTO inbox_items(kind, dedupe_key, title) VALUES ('term', 'hui du', '是「灰度」吗')")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO inbox_items(kind, dedupe_key, title) VALUES ('term', 'hui du', '又问一次')")
    db.execute("UPDATE inbox_items SET status='answered'")
    db.execute("INSERT INTO inbox_items(kind, dedupe_key, title) VALUES ('term', 'hui du', '以后再出现可以再问')")
    db.execute("INSERT INTO inbox_items(kind, title) VALUES ('voice', '没有 key 的不受限制')")
    db.execute("INSERT INTO inbox_items(kind, title) VALUES ('voice', '没有 key 的不受限制')")
    assert db.one("SELECT COUNT(*) AS n FROM inbox_items")["n"] == 4


# --- upgrade of an old database ----------------------------------------------------------------------------------

def test_old_database_is_backed_up_then_upgraded(cfg):
    notes = [("todo", "买 牛奶！", "下班顺路", "2026-11-01", "2026-11-02", "[3, 5]"),
             ("person", "王敏", "同事", "2026-11-01", None, None),
             ("decision", "周五发版", "", None, "2026-11-02", "not json"),
             ("todo", "交 报告", "x", "2026-11-02", "2026-11-02", '{"a": 1}')]
    _make_old_db(cfg.db_path, segments=2, notes=notes)
    size_before = cfg.db_path.stat().st_size

    db = DB(cfg.db_path)
    assert db.schema_version() == 2 and db.migrated == (2,)
    # 1. the copy: old layout, old version, same rows, made before anything changed
    backups = sorted((cfg.data_dir / "backups").glob("*.sqlite3"))
    assert [db.last_backup] == backups and backups[0].name.startswith("audio_note-v0-") and backups[0].stat().st_size >= size_before / 2
    stamp = backups[0].stem.split("-v0-")[1]
    assert datetime.strptime(stamp, "%Y%m%d-%H%M%S")
    old = sqlite3.connect(backups[0])
    assert old.execute("PRAGMA user_version").fetchone()[0] == 0
    assert old.execute("SELECT COUNT(*) FROM segments").fetchone()[0] == 2
    assert old.execute("SELECT COUNT(*) FROM memory_notes").fetchone()[0] == 4
    assert "memory_items" not in {r[0] for r in old.execute("SELECT name FROM sqlite_master")}
    assert "realtime" not in {r[1] for r in old.execute("PRAGMA table_info(segments)")}
    assert old.execute("SELECT updated_at FROM segments ORDER BY id").fetchone()[0] == "2026-11-01 09:00:05"
    old.close()
    assert not list((cfg.data_dir / "backups").glob("*.partial"))
    # 2. timestamps: one format
    assert [r["updated_at"] for r in db.query("SELECT updated_at FROM segments ORDER BY id")] == ["2026-11-01T09:00:05", "2026-11-02T09:00:05"]
    # 3. notes -> items (ids kept, the old table stays)
    items = db.query("SELECT * FROM memory_items ORDER BY id")
    assert [(r["id"], r["kind"], r["title"], r["source"], r["status"]) for r in items] == [
        (1, "todo", "买 牛奶！", "legacy", "active"), (2, "person", "王敏", "legacy", "active"),
        (3, "decision", "周五发版", "legacy", "active"), (4, "todo", "交 报告", "legacy", "active")]
    assert [r["evidence"] for r in items] == ["[3, 5]", None, None, None]
    assert [(r["day"], r["valid_from"]) for r in items] == [("2026-11-02", "2026-11-01"), ("2026-11-01", "2026-11-01"),
                                                           ("2026-11-02", None), ("2026-11-02", "2026-11-02")]
    assert [r["fingerprint"] for r in items] == [memory_fingerprint(r["kind"], r["title"]) for r in items]
    assert items[0]["fingerprint"] == memory_fingerprint("todo", "买牛奶") and items[0]["content"] == "下班顺路"
    assert db.one("SELECT COUNT(*) AS n FROM memory_notes")["n"] == 4
    # 4. the rest of the upgrade
    assert NEW_TABLES <= _names(db) and {"idx_utt_session", "idx_cluster_voice"} <= _names(db, "index")
    assert db.one("PRAGMA integrity_check")[0] == "ok"
    # opening again changes nothing and makes no second copy
    db2 = DB(cfg.db_path)
    assert db2.migrated == () and db2.last_backup is None
    assert db2.one("SELECT COUNT(*) AS n FROM memory_items")["n"] == 4
    assert len(list((cfg.data_dir / "backups").glob("*.sqlite3"))) == 1


def test_copy_notes_to_items_is_idempotent_and_skips_what_is_already_there(db):
    """The note copy can be run again later (notes written by a pre-1.0 code path after the upgrade)."""
    from audio_note.db import copy_notes_to_items

    from audio_note.db import memory_fingerprint as fp

    db.execute("INSERT INTO memory_items(kind, title, source, fingerprint) VALUES ('todo', '用户自己加的', 'user', ?)", (fp("todo", "用户自己加的"),))
    db.execute("INSERT INTO memory_items(kind, title, status, fingerprint) VALUES ('todo', '不要了', 'rejected', ?)", (fp("todo", "不要了"),))
    db.execute("INSERT INTO memory_notes(kind, title, content) VALUES ('todo', '用户 自己加的！', '同一件事，写法不同')")
    db.execute("INSERT INTO memory_notes(kind, title, content) VALUES ('todo', '不要了', '墓碑不会被复活')")
    db.execute("INSERT INTO memory_notes(kind, title, content, day_from) VALUES ('person', '李四', '朋友', '2026-11-03')")
    with db.transaction() as c:
        assert copy_notes_to_items(c) == 1
    with db.transaction() as c:
        assert copy_notes_to_items(c) == 0
    rows = [(r["kind"], r["title"], r["source"], r["status"], r["day"]) for r in db.query("SELECT * FROM memory_items ORDER BY id")]
    assert rows == [("todo", "用户自己加的", "user", "active", None), ("todo", "不要了", "auto", "rejected", None),
                    ("person", "李四", "legacy", "active", "2026-11-03")]
    assert db.one("SELECT COUNT(*) AS n FROM memory_notes")["n"] == 3


def test_only_the_newest_backups_are_kept(cfg):
    bdir = cfg.data_dir / "backups"
    bdir.mkdir()
    for i in range(5):                                                     # leftovers of earlier upgrades, oldest first
        f = bdir / f"audio_note-v0-2026010{i + 1}-000000.sqlite3"
        f.write_bytes(b"old")
        t = time.time() - 86400 * (10 - i)
        import os
        os.utime(f, (t, t))
    (bdir / "unrelated.txt").write_text("keep me", encoding="utf-8")
    _make_old_db(cfg.db_path)
    db = DB(cfg.db_path)
    left = sorted(p.name for p in bdir.glob("*.sqlite3"))
    assert len(left) == BACKUP_KEEP == 3 and db.last_backup.name in left
    assert left[:2] == ["audio_note-v0-20260104-000000.sqlite3", "audio_note-v0-20260105-000000.sqlite3"]
    assert (bdir / "unrelated.txt").exists()


def test_a_failed_backup_stops_the_upgrade_and_changes_nothing(cfg):
    _make_old_db(cfg.db_path, notes=[("todo", "甲", "x", None, None, None)])
    (cfg.data_dir / "backups").write_text("a file where the directory should be", encoding="utf-8")
    with pytest.raises(MigrationError, match="备份失败"):
        DB(cfg.db_path)
    raw = sqlite3.connect(cfg.db_path)                                     # the handle was released, the file is as it was
    assert raw.execute("PRAGMA user_version").fetchone()[0] == 0
    assert "memory_items" not in {r[0] for r in raw.execute("SELECT name FROM sqlite_master")}
    assert "realtime" not in {r[1] for r in raw.execute("PRAGMA table_info(segments)")}
    raw.close()
    (cfg.data_dir / "backups").unlink()
    assert DB(cfg.db_path).schema_version() == 2                           # once the obstacle is gone it upgrades


def test_a_failing_data_step_rolls_back_and_keeps_the_old_version(cfg, monkeypatch):
    _make_old_db(cfg.db_path, notes=[("todo", "甲", "x", None, None, None)])

    def boom(c):
        c.execute("INSERT INTO memory_items(kind, title) VALUES ('todo', 'half done')")
        raise RuntimeError("step broke")

    monkeypatch.setattr(db_mod, "MIGRATIONS", [(2, "broken", boom)])
    with pytest.raises(MigrationError, match="v2 失败.*step broke"):
        DB(cfg.db_path)
    raw = sqlite3.connect(cfg.db_path)
    assert raw.execute("PRAGMA user_version").fetchone()[0] == 0
    assert raw.execute("SELECT COUNT(*) FROM memory_items").fetchone()[0] == 0      # rolled back with the step
    raw.close()
    monkeypatch.undo()
    db = DB(cfg.db_path)
    assert db.schema_version() == 2 and db.one("SELECT COUNT(*) AS n FROM memory_items")["n"] == 1


def test_a_database_from_a_newer_version_is_refused_untouched(cfg):
    raw = sqlite3.connect(cfg.db_path)
    raw.execute("CREATE TABLE segments(id INTEGER PRIMARY KEY, future_column TEXT)")
    raw.execute(f"PRAGMA user_version={SCHEMA_VERSION + 1}")
    raw.commit()
    raw.close()
    before = cfg.db_path.read_bytes()
    with pytest.raises(SchemaTooNew, match="更新的版本"):
        DB(cfg.db_path)
    assert cfg.db_path.read_bytes() == before                              # not even the journal mode was changed
    assert not Path(str(cfg.db_path) + "-wal").exists()
    cfg.db_path.unlink()                                                   # nothing holds the file open (Windows)


def test_two_threads_upgrading_at_once_run_the_step_once(cfg):
    _make_old_db(cfg.db_path, notes=[(("todo"), f"事项{i}", "x", None, None, None) for i in range(30)])
    out: list = []
    barrier = threading.Barrier(4)

    def open_it():
        barrier.wait(5)
        try:
            d = DB(cfg.db_path)
            out.append((d.schema_version(), d.migrated))
        except Exception as e:  # noqa: BLE001
            out.append(e)

    ts = [threading.Thread(target=open_it) for _ in range(4)]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    assert all(isinstance(o, tuple) and o[0] == 2 for o in out), out
    assert sorted(o[1] for o in out).count((2,)) == 1                      # exactly one of them ran the data step
    db = DB(cfg.db_path)
    assert db.one("SELECT COUNT(*) AS n FROM memory_items")["n"] == 30
    assert len(list((cfg.data_dir / "backups").glob("*.sqlite3"))) <= BACKUP_KEEP


def test_memory_fingerprint_ignores_spacing_punctuation_width_and_case():
    fp = memory_fingerprint("todo", "买牛奶")
    assert len(fp) == 16 and int(fp, 16) >= 0
    assert fp == memory_fingerprint("todo", " 买 牛奶！") == memory_fingerprint("TODO ", "买，牛奶。")
    assert memory_fingerprint("todo", "Fix API") == memory_fingerprint("todo", "fix-api") == memory_fingerprint("todo", "ＦＩＸ　ＡＰＩ")
    assert fp != memory_fingerprint("decision", "买牛奶") and fp != memory_fingerprint("todo", "买豆浆")
    assert memory_fingerprint("todo", "") == memory_fingerprint("todo", None)  # type: ignore[arg-type]


# --- the stale-processing bug ----------------------------------------------------------------------------------------

def test_reset_stale_processing_works_on_rows_stamped_by_now_iso(db):
    """Regression: set_segment_status writes 'YYYY-MM-DDTHH:MM:SS', the old query compared it with SQLite's
    'YYYY-MM-DD HH:MM:SS' as strings, so a segment stuck in 'processing' was never reclaimed on the same day."""
    t = datetime.now().replace(microsecond=0)
    stale = db.add_segment("stale.opus", t, 5.0)
    fresh = db.add_segment("fresh.opus", t, 5.0)
    legacy = db.add_segment("legacy.opus", t, 5.0)
    for sid in (stale, fresh, legacy):
        db.set_segment_status(sid, "processing")
    assert "T" in db.get_segment(fresh)["updated_at"] and db.get_segment(fresh)["updated_at"] <= now_iso()
    db.execute("UPDATE segments SET updated_at=? WHERE id=?", ((datetime.now() - timedelta(hours=2)).isoformat(timespec="seconds"), stale))
    db.execute("UPDATE segments SET updated_at=datetime('now','localtime','-2 hours') WHERE id=?", (legacy,))   # space format
    assert db.reset_stale_processing(600) == 2
    assert [db.get_segment(s)["status"] for s in (stale, fresh, legacy)] == ["new", "processing", "new"]
    assert db.reset_stale_processing(600) == 0
    assert db.reset_stale_processing(0) == 1                               # the fresh one, once no age is required


def test_new_segments_are_stamped_in_the_iso_format(db):
    sid = db.add_segment("x.opus", datetime(2026, 11, 11, 9, 0), 3.0)
    assert datetime.fromisoformat(db.get_segment(sid)["updated_at"]) and "T" in db.get_segment(sid)["updated_at"]


def test_unfinished_segments_counts_recent_failures_in_either_format(db):
    day = "2026-11-10"
    recent = db.add_segment("r.opus", datetime(2026, 11, 10, 9, 0), 3.0)
    old = db.add_segment("o.opus", datetime(2026, 11, 10, 10, 0), 3.0)
    legacy = db.add_segment("l.opus", datetime(2026, 11, 10, 11, 0), 3.0)
    done = db.add_segment("d.opus", datetime(2026, 11, 10, 12, 0), 3.0)
    db.set_segment_status(done, "done")
    for sid in (recent, old, legacy):
        db.set_segment_status(sid, "failed", "boom")                       # stamped now, 'T' format
    assert db.unfinished_segments_for_day(day) == 3
    db.execute("UPDATE segments SET updated_at=? WHERE id=?", ((datetime.now() - timedelta(days=3)).isoformat(timespec="seconds"), old))
    db.execute("UPDATE segments SET updated_at=datetime('now','localtime','-3 days') WHERE id=?", (legacy,))
    assert db.unfinished_segments_for_day(day) == 1
    assert db.unfinished_segments_for_day(day, failed_within_s=86400 * 7) == 3
    assert db.unfinished_segments_for_day("2026-11-09") == 0


def test_requeue_stopped_only_touches_segments_interrupted_by_a_stop(db):
    t = datetime(2026, 11, 11, 9, 0)
    stopped, failed, fine = (db.add_segment(f"{n}.opus", t, 3.0) for n in ("s", "f", "ok"))
    db.set_segment_status(stopped, "failed", STOPPED_ERROR)
    db.set_segment_status(failed, "failed", "LLMError: asr: HTTP 400 bad audio")
    db.set_segment_status(fine, "done")
    assert STOPPED_ERROR == "RuntimeError: stopped" and db.requeue_stopped() == 1
    assert [(db.get_segment(s)["status"], db.get_segment(s)["error"]) for s in (stopped, failed, fine)] == [
        ("new", None), ("failed", "LLMError: asr: HTTP 400 bad audio"), ("done", None)]
    assert db.requeue_stopped() == 0


# --- transactions -----------------------------------------------------------------------------------------------------

def test_transactions_nest_with_savepoints(db):
    seg = db.add_segment("s.opus", datetime(2026, 11, 11, 9, 0), 10.0)
    row = {"segment_id": seg, "start_s": 0.0, "end_s": 1.0, "abs_start": "2026-11-11T09:00:00", "day": "2026-11-11",
           "text_raw": "一", "text": "一"}
    with db.transaction() as c:
        assert c.in_transaction
        ids = db.add_utterances([row])                                     # opens its own _Tx: used to raise "cannot start a transaction"
        c.execute("INSERT INTO kv(key, value) VALUES ('outer', '1')")
        with pytest.raises(ValueError):
            with db.transaction() as inner:
                inner.execute("INSERT INTO kv(key, value) VALUES ('inner', '1')")
                raise ValueError("only the inner block is undone")
        assert c.in_transaction
    assert not db.conn().in_transaction
    assert db.get_kv("outer") == 1 and db.get_kv("inner") is None and len(ids) == 1
    assert db.one("SELECT COUNT(*) AS n FROM utterances")["n"] == 1
    # an outer failure takes the inner (released) work with it
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.add_utterances([row, row])
            raise RuntimeError("abort all")
    assert db.one("SELECT COUNT(*) AS n FROM utterances")["n"] == 1 and not db.conn().in_transaction


def test_transaction_takes_the_write_lock_up_front(cfg):
    """BEGIN IMMEDIATE: a second writer waits at BEGIN (and times out there) instead of failing in the middle."""
    a = DB(cfg.db_path)
    other = sqlite3.connect(cfg.db_path, timeout=0.2, isolation_level=None)
    try:
        with a.transaction() as c:
            c.execute("SELECT 1").fetchone()                                # no write yet: a DEFERRED transaction would not hold the lock
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                other.execute("BEGIN IMMEDIATE")
        other.execute("BEGIN IMMEDIATE")
        other.execute("ROLLBACK")
    finally:
        other.close()


# --- helpers ----------------------------------------------------------------------------------------------------------

def test_add_utterances_takes_the_new_optional_keys(db):
    seg = db.add_segment("s.opus", datetime(2026, 11, 11, 9, 0), 10.0)
    base = {"segment_id": seg, "start_s": 0.0, "end_s": 1.0, "abs_start": "2026-11-11T09:00:00", "day": "2026-11-11",
            "text_raw": "原文", "text": "改后"}
    sid = db.upsert_speaker("甲")
    a, b, c = db.add_utterances([
        dict(base),
        dict(base, session_id=7, text_source="lexicon", flags={"guard": "loop"}, speaker_guess_id=sid, speaker_guess_score=0.41,
             speaker_meta={"method": "centroid:far", "raw": 0.52, "norm": 2.1}),
        dict(base, flags='{"media": true}', speaker_meta=None),
    ])
    rows = {r["id"]: r for r in db.query("SELECT * FROM utterances")}
    assert [rows[a][k] for k in ("session_id", "text_source", "flags", "speaker_guess_id", "speaker_guess_score", "speaker_meta")] == [None] * 6
    assert rows[b]["session_id"] == 7 and rows[b]["text_source"] == "lexicon" and json.loads(rows[b]["flags"]) == {"guard": "loop"}
    assert rows[b]["speaker_guess_id"] == sid and rows[b]["speaker_guess_score"] == 0.41
    assert json.loads(rows[b]["speaker_meta"]) == {"method": "centroid:far", "raw": 0.52, "norm": 2.1}
    assert rows[c]["flags"] == '{"media": true}' and rows[c]["speaker_meta"] is None


def test_usage_rows_accumulate_per_day(db):
    db.add_usage_local("diarize", audio_s=10.0, wall_s=1.5, cpu_s=2.0)
    db.add_usage_local("diarize", n=2, audio_s=5.0, wall_s=0.5, cpu_s=1.0)
    db.add_usage_local("embed", day="2026-11-01", cpu_s=0.25)
    today = datetime.now().strftime("%Y-%m-%d")
    assert [tuple(r) for r in db.query("SELECT day, stage, n, audio_s, wall_s, cpu_s FROM usage_local ORDER BY day, stage")] == [
        ("2026-11-01", "embed", 1, 0.0, 0.0, 0.25), (today, "diarize", 3, 15.0, 2.0, 3.0)]
    db.add_usage_llm("asr", "mimo-v2.5-asr", prompt_tokens=40, completion_tokens=20, audio_s=9.5, wall_s=1.0)
    db.add_usage_llm("asr", "mimo-v2.5-asr", errors=1, wall_s=2.0)
    db.add_usage_llm("refine", "mimo-v2.6-flash", prompt_tokens=900, completion_tokens=300)
    rows = {(r["purpose"], r["model"]): r for r in db.query("SELECT * FROM usage_llm")}
    asr = rows[("asr", "mimo-v2.5-asr")]
    assert (asr["day"], asr["calls"], asr["errors"], asr["prompt_tokens"], asr["completion_tokens"], asr["audio_s"], asr["wall_s"]) == (
        today, 2, 1, 40, 20, 9.5, 3.0)
    assert rows[("refine", "mimo-v2.6-flash")]["calls"] == 1


def test_job_run_helpers_and_maintenance(db):
    rid = db.start_job_run("reindex", "ui")
    orphan = db.start_job_run("daily", "scheduler")
    assert [r["status"] for r in db.query("SELECT status FROM job_runs ORDER BY id")] == ["running", "running"]
    db.finish_job_run(rid, "done", {"keywords": 3, "路径": "x"})
    row = db.one("SELECT * FROM job_runs WHERE id=?", (rid,))
    assert row["status"] == "done" and json.loads(row["result"]) == {"keywords": 3, "路径": "x"} and row["finished_at"] and row["error"] is None
    assert db.close_orphan_job_runs() == 1
    row = db.one("SELECT * FROM job_runs WHERE id=?", (orphan,))
    assert (row["status"], row["error"]) == ("failed", "interrupted") and row["finished_at"]
    old = db.add_job_run("process", "scheduler", "failed", "2020-01-01T00:00:00", error="boom")
    kept = db.add_job_run("process", "scheduler", "done", now_iso(), {"processed": 2})
    out = db.maintenance()
    assert out["job_runs_pruned"] == 1 and out["vacuum"] is False and out["wal_checkpoint"] in ("ok", "busy") and out["db_bytes"] > 0
    ids = {r["id"] for r in db.query("SELECT id FROM job_runs")}
    assert old not in ids and {rid, orphan, kept} <= ids
    assert db.maintenance(vacuum=True)["vacuum"] is True and db.one("PRAGMA integrity_check")[0] == "ok"
