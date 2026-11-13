"""memory.forget: forgetting a stretch of recordings across every layer (SPEC v1 section 7.3).

The central test plants canary sentences in every place text can live - sentences, tokens / FTS, vectors, voice
vectors, revisions, the segment transcript, cluster embeddings, sessions, memory items, the digest, both Markdown
files, the inbox, a stored chat, a pre-upgrade backup, an audio file - forgets them, and then proves that neither
any table nor any file under the data directory still contains a trace.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from audio_note.db import DB
from audio_note.memory import forget as forget_mod
from audio_note.memory import items, sessions, vectors
from audio_note.memory.embed import HashEmbedder
from audio_note.memory.forget import forget
from audio_note.memory.index import tokenize
from audio_note.speech.markdown import write_day_markdown

DAY, OTHER = "2026-11-03", "2026-11-04"
#: fragments that must be gone afterwards (latin + CJK, so both the raw text and the jieba tokens are covered)
CANARIES = ("QZX749", "蓝鲸暗号", "KLM382", "银杏口令", "VBN615")
KEEP = "保留句子ZYX001"


@pytest.fixture
def fcfg(cfg_factory):
    return cfg_factory({"memory": {"embedding_model": "hash", "retrieval": "hybrid"}})


def _seg(db: DB, cfg, name: str, start: datetime, dur: float = 30.0, content: bytes = b"") -> int:
    path = cfg.audio_dir / name
    path.write_bytes(content or (b"OggS" + name.encode() * 20))
    sid = db.add_segment(path, start, dur)
    db.set_segment_status(sid, "done")
    return sid


def _utts(db: DB, seg: int, day: str, rows: list[tuple[str, float, str, int]]) -> list[int]:
    """rows: (HH:MM:SS, start_s, text, cluster)"""
    return db.add_utterances([{
        "segment_id": seg, "start_s": s, "end_s": s + 2.0, "abs_start": f"{day}T{t}.000", "day": day, "cluster_label": cl,
        "speaker_name": "甲" if cl == 0 else "乙", "text_raw": text, "text": text, "text_seg": tokenize(text)}
        for t, s, text, cl in rows])


def _transcript(db: DB, seg: int, ids: list[int], with_ids: bool) -> None:
    rows = db.query(f"SELECT id, start_s, end_s, text FROM utterances WHERE id IN ({','.join('?' * len(ids))}) ORDER BY start_s", ids)
    its = [{"i": i, "start_s": r["start_s"], "end_s": r["end_s"], "text_raw": r["text"], "text": r["text"],
            **({"utterance_id": int(r["id"])} if with_ids else {})} for i, r in enumerate(rows)]
    db.execute("INSERT INTO transcripts(segment_id, asr_model, raw_json, text_raw, text_corrected, correction_json) VALUES (?,?,?,?,?,?)",
               (seg, "fake", json.dumps(its, ensure_ascii=False), "\n".join(r["text"] for r in rows),
                "\n".join(r["text"] for r in rows),
                json.dumps([{"i": i, "from": r["text"][:3], "to": r["text"][:3], "reason": "x"} for i, r in enumerate(rows)],
                           ensure_ascii=False)))


@pytest.fixture
def world(fcfg):
    """A small database with every layer filled. Returns a dict of the ids the tests refer to."""
    cfg = fcfg
    db = DB(cfg.db_path)
    a = _seg(db, cfg, "a.opus", datetime(2026, 11, 3, 10, 0, 0))
    b = _seg(db, cfg, "b.opus", datetime(2026, 11, 3, 10, 1, 0))
    c = _seg(db, cfg, "c.opus", datetime(2026, 11, 3, 15, 0, 0))
    d = _seg(db, cfg, "d.opus", datetime(2026, 11, 4, 9, 0, 0))
    ua = _utts(db, a, DAY, [("10:00:02", 2.0, "第一句提到QZX749的安排", 0), ("10:00:06", 6.0, "蓝鲸暗号是今晚八点", 1)])
    ub = _utts(db, b, DAY, [("10:01:02", 2.0, f"{KEEP}甲说的话", 0), ("10:01:06", 6.0, "中间这句有KLM382和银杏口令", 1),
                            ("10:01:10", 10.0, f"{KEEP}最后一句", 0)])
    uc = _utts(db, c, DAY, [("15:00:02", 2.0, f"下午的{KEEP}闲聊", 0), ("15:00:06", 6.0, f"下午第二句{KEEP}", 0)])
    ud = _utts(db, d, OTHER, [("09:00:02", 2.0, f"第二天的{KEEP}", 0)])
    _transcript(db, a, ua, with_ids=False)
    _transcript(db, b, ub, with_ids=False)          # matched by bounds
    _transcript(db, c, uc, with_ids=True)
    _transcript(db, d, ud, with_ids=True)
    for seg, labels in ((a, (0, 1)), (b, (0, 1)), (c, (0,)), (d, (0,))):
        for label in labels:
            db.upsert_cluster_embedding(seg, label, np.ones(8, dtype=np.float32), 4.0)
    emb = HashEmbedder(64)
    emb.views = ("plain", "window")
    vectors.index_missing(db, emb)
    for uid in ua + ub + uc + ud:
        db.execute("INSERT INTO utterance_voice(utterance_id, model, dim, embedding, speech_s) VALUES (?,?,?,?,?)",
                   (uid, "fake", 8, np.ones(8, dtype=np.float32).tobytes(), 2.0))
    # an earlier wording of a forgotten sentence lives in the revision history
    db.execute("INSERT INTO utterance_revisions(utterance_id, old_text, new_text, source, detail) VALUES (?,?,?,?,?)",
               (ua[0], "第一句提到VBN615的安排", "第一句提到QZX749的安排", "user",
                json.dumps({"edits": [{"from": "VBN615", "to": "QZX749"}]})))
    sessions.sync(cfg, db, now=datetime(2026, 11, 5))
    sess = {r["id"]: r["session_id"] for r in db.query("SELECT id, session_id FROM utterances")}
    db.execute("UPDATE sessions SET title=?, summary=? WHERE id=?", ("讨论QZX749", "蓝鲸暗号定在今晚八点", sess[ua[0]]))
    it_gone, _ = items.upsert(db, cfg, {"kind": "fact", "title": "蓝鲸暗号是今晚八点", "content": "QZX749 相关", "evidence": ua,
                                        "day": DAY, "subject": "QZX749"}, embedder=emb)
    it_trim, _ = items.upsert(db, cfg, {"kind": "fact", "title": f"{KEEP}的事实", "evidence": [ua[1], uc[0]], "day": DAY}, embedder=emb)
    it_digest, _ = items.upsert(db, cfg, {"kind": "todo", "title": "确认KLM382的口令", "day": DAY}, source="digest")
    it_user, _ = items.upsert(db, cfg, {"kind": "note", "title": f"{KEEP}用户自己写的", "day": DAY}, source="user")
    it_other, _ = items.upsert(db, cfg, {"kind": "fact", "title": f"{KEEP}第二天的事", "evidence": ud, "day": OTHER})
    db.execute("INSERT INTO daily_summaries(day, summary_md, summary_json, utterance_count) VALUES (?,?,?,?)",
               (DAY, "# 日报\n提到了QZX749和蓝鲸暗号", json.dumps({"overview": "银杏口令 KLM382"}, ensure_ascii=False), 7))
    (cfg.daily_dir / f"{DAY}.md").write_text("# 日报\n提到了QZX749和蓝鲸暗号\n", encoding="utf-8")
    db.execute("INSERT INTO daily_summaries(day, summary_md, utterance_count) VALUES (?,?,?)", (OTHER, f"# {KEEP}", 1))
    write_day_markdown(db, cfg, DAY)
    write_day_markdown(db, cfg, OTHER)
    db.execute("INSERT INTO inbox_items(kind, title, payload, utterance_id) VALUES (?,?,?,?)",
               ("term", "QZX749 是什么？", json.dumps({"span": "QZX749", "sentence": "第一句提到QZX749的安排"}, ensure_ascii=False), ua[0]))
    db.execute("INSERT INTO inbox_items(kind, title, payload, utterance_id) VALUES (?,?,?,?)",
               ("term", f"{KEEP} 怎么写？", json.dumps({"span": KEEP, "utterance_id": uc[0], "examples": [
                   {"utterance_id": ub[1], "sentence": "中间这句有KLM382和银杏口令"},
                   {"utterance_id": uc[0], "sentence": f"下午的{KEEP}闲聊"}]}, ensure_ascii=False), uc[0]))
    conv = db.execute("INSERT INTO chat_conversations(title) VALUES ('问答')").lastrowid
    db.execute("INSERT INTO chat_messages(conversation_id, role, content, meta) VALUES (?,?,?,?)",
               (conv, "assistant", "暗号是蓝鲸暗号", json.dumps({"citations": [ua[1]]})))
    db.execute("INSERT INTO chat_messages(conversation_id, role, content, meta) VALUES (?,?,?,?)",
               (conv, "assistant", f"{KEEP}的回答", json.dumps({"citations": [{"id": ud[0]}]})))
    # a pre-upgrade copy of the database that still holds the sentences
    bdir = cfg.data_dir / "backups"
    bdir.mkdir()
    (bdir / "audio_note-v0-20261104-000000.sqlite3").write_bytes("旧库里的QZX749和蓝鲸暗号".encode("utf-8"))
    return {"cfg": cfg, "db": db, "segs": (a, b, c, d), "ua": ua, "ub": ub, "uc": uc, "ud": ud, "emb": emb,
            "items": {"gone": it_gone, "trim": it_trim, "digest": it_digest, "user": it_user, "other": it_other}}


def _table_leaks(db: DB, needles=CANARIES) -> list[str]:
    """Every (table.column) of the database whose text still contains a needle."""
    leaks = []
    tables = [r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")]
    for t in tables:
        try:
            cols = [r["name"] for r in db.query(f"PRAGMA table_xinfo('{t}')")]
        except sqlite3.Error:
            continue
        for col in cols:
            for n in needles:
                try:
                    hit = db.one(f"SELECT 1 FROM \"{t}\" WHERE CAST(\"{col}\" AS TEXT) LIKE ? LIMIT 1", (f"%{n}%",))
                except sqlite3.Error:
                    continue
                if hit is not None:
                    leaks.append(f"{t}.{col}:{n}")
    return leaks


def _file_leaks(root: Path, needles=CANARIES) -> list[str]:
    """Every file under `root` whose bytes still contain a needle (UTF-8 or UTF-16, as SQLite may store text)."""
    leaks = []
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        data = f.read_bytes()
        for n in needles:
            if n.encode("utf-8") in data or n.encode("utf-16-le") in data or n.lower().encode("utf-8") in data:
                leaks.append(f"{f.relative_to(root)}:{n}")
    return leaks


def _snapshot(db: DB, root: Path) -> tuple:
    counts = tuple((t, db.one(f"SELECT COUNT(*) AS n FROM {t}")["n"]) for t in (
        "utterances", "utterance_tokens", "utterance_vectors", "memory_vectors", "utterance_voice", "utterance_revisions",
        "transcripts", "cluster_embeddings", "sessions", "memory_items", "memory_item_tokens", "daily_summaries",
        "inbox_items", "chat_messages", "segments"))
    files = tuple(sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and not p.name.endswith(("-wal", "-shm"))))
    return counts, files


# ---------------------------------------------------------------------------------------------- the proof
def test_nothing_of_the_forgotten_text_survives_in_any_table_or_file(world):
    cfg, db = world["cfg"], world["db"]
    ua, ub, uc = world["ua"], world["ub"], world["uc"]
    a, b, c, d = world["segs"]
    assert _table_leaks(db) and _file_leaks(cfg.data_dir)              # the canaries really are everywhere first

    target = [*ua, ub[1]]                                               # all of segment a, the middle sentence of b
    before = _snapshot(db, cfg.data_dir)
    dry = forget(db, cfg, utterance_ids=target)
    assert dry["dry_run"] is True and _snapshot(db, cfg.data_dir) == before          # a dry run changes nothing

    rep = forget(db, cfg, utterance_ids=target, dry_run=False)
    counted = ("utterances", "chars", "days", "segments", "audio_files", "audio_bytes", "audio_kept", "transcripts",
               "cluster_embeddings", "vectors", "voice_vectors", "revisions", "sessions", "items", "item_ids", "digests",
               "markdown", "inbox", "chat_messages", "backups")
    assert {k: dry[k] for k in counted} == {k: rep[k] for k in counted}               # ... and predicts the real run
    assert rep["errors"] == [] and rep["dry_run"] is False

    # --- the proof: no table, no file
    assert _table_leaks(db) == []
    db.close_all()                                                       # everything flushed; the files are at rest
    assert _file_leaks(cfg.data_dir) == []
    db = DB(cfg.db_path)

    # --- what the report says happened
    assert rep["utterances"] == 3 and rep["days"] == [DAY]
    assert rep["segments"] == {"whole": 1, "partial": 1, "skipped": 0}
    assert rep["audio_files"] == 1 and rep["audio_kept"] == [b] and rep["audio_bytes"] > 0
    assert rep["transcripts"] == 2 and rep["cluster_embeddings"] == 3          # a: 2 clusters; b: the cluster only the forgotten sentence had
    assert rep["vectors"] == 6 and rep["voice_vectors"] == 3 and rep["revisions"] == 1
    assert rep["items"] == {"tombstoned": 1, "trimmed": 1, "removed": 1, "unverified": 1}
    assert rep["digests"] == [DAY] and rep["markdown"] == [DAY]
    assert rep["inbox"] == 2 and rep["chat_messages"] == 1 and rep["backups"] == ["audio_note-v0-20261104-000000.sqlite3"]
    assert rep["physical"]["vacuum"] is True and rep["physical"]["wal_checkpoint"] == "ok"
    assert any("录音文件保留" in n for n in rep["notes"])

    # --- and what is left is intact
    assert not (cfg.audio_dir / "a.opus").exists() and (cfg.audio_dir / "b.opus").exists()
    assert db.one("SELECT status FROM segments WHERE id=?", (a,))["status"] == "deleted"
    assert [r["id"] for r in db.query("SELECT id FROM utterances ORDER BY id")] == [ub[0], ub[2], *uc, *world["ud"]]
    tr = db.one("SELECT raw_json, text_raw, correction_json FROM transcripts WHERE segment_id=?", (b,))
    kept = json.loads(tr["raw_json"])
    assert [it["i"] for it in kept] == [0, 1] and all(KEEP in it["text"] for it in kept)
    assert tr["text_raw"].count("\n") == 1 and [e["i"] for e in json.loads(tr["correction_json"])] == [0, 1]
    assert db.one("SELECT COUNT(*) AS n FROM transcripts WHERE segment_id=?", (a,))["n"] == 0
    assert db.query("SELECT cluster_label FROM cluster_embeddings WHERE segment_id=?", (b,))[0]["cluster_label"] == 0
    its = world["items"]
    gone = db.one("SELECT * FROM memory_items WHERE id=?", (its["gone"],))
    assert gone["status"] == "rejected" and gone["title"] == "" and gone["content"] == "" and gone["subject"] is None
    assert gone["fingerprint"] == items.fingerprint("fact", "蓝鲸暗号是今晚八点")                 # the tombstone still guards the title
    assert items.upsert(db, cfg, {"kind": "fact", "title": "蓝鲸暗号是今晚八点"}) == (its["gone"], "skipped")
    assert items.get(db, its["trim"])["evidence"] == [uc[0]] and items.get(db, its["trim"])["status"] == "active"
    assert items.get(db, its["digest"]) is None                                           # rebuilt by the next digest
    assert items.get(db, its["user"])["status"] == "active" and items.get(db, its["other"])["status"] == "active"
    assert db.one("SELECT 1 FROM daily_summaries WHERE day=?", (DAY,)) is None and not (cfg.daily_dir / f"{DAY}.md").exists()
    assert db.one("SELECT 1 FROM daily_summaries WHERE day=?", (OTHER,)) is not None
    assert db.get_kv(forget_mod.KV_REBUILD_DAYS) == [DAY]
    md = (cfg.transcripts_dir / f"{DAY}.md").read_text(encoding="utf-8")
    assert KEEP in md and "a.opus" not in md
    inbox = db.query("SELECT payload FROM inbox_items")
    assert len(inbox) == 1 and [e["utterance_id"] for e in json.loads(inbox[0]["payload"])["examples"]] == [uc[0]]
    assert [m["content"] for m in db.query("SELECT content FROM chat_messages")] == [f"{KEEP}的回答"]
    # sessions: the morning session lost three of five sentences, kept its row with fresh counts and no summary
    s = db.one("SELECT * FROM sessions WHERE id=(SELECT session_id FROM utterances WHERE id=?)", (ub[0],))
    assert s["utterance_count"] == 2 and s["title"] is None and s["summary"] is None and s["digested_hash"] is None
    assert (s["raw_hash"], s["text_hash"]) == db.session_hashes(int(s["id"]))
    # retrieval: neither keywords nor vectors find the forgotten sentences
    from audio_note.memory.retrieval import Retriever

    r = Retriever(cfg, db, embedder=world["emb"])
    for q in ("QZX749", "蓝鲸暗号", "KLM382 银杏口令"):
        assert all(h.id not in target for h in r.search(q, limit=20).hits)
    assert items.search(db, cfg, "蓝鲸暗号", status="all", embedder=world["emb"]) == []
    db.close_all()


def test_forget_a_whole_day_removes_audio_sessions_and_markdown(world):
    cfg, db = world["cfg"], world["db"]
    a, b, c, d = world["segs"]
    silent = _seg(db, cfg, "silent.opus", datetime(2026, 11, 3, 23, 0, 0))
    db.set_segment_status(silent, "no_speech")
    rep = forget(db, cfg, day=DAY, dry_run=False)
    assert rep["utterances"] == 7 and rep["segments"]["whole"] == 4 and rep["segments"]["partial"] == 0
    assert rep["audio_files"] == 4 and rep["audio_kept"] == []                    # the silent recording of that day goes too
    assert sorted(p.name for p in cfg.audio_dir.iterdir()) == ["d.opus"]
    assert rep["sessions"]["removed"] == 2 and db.one("SELECT COUNT(*) AS n FROM sessions")["n"] == 1
    assert not (cfg.transcripts_dir / f"{DAY}.md").exists() and (cfg.transcripts_dir / f"{OTHER}.md").exists()
    assert db.one("SELECT COUNT(*) AS n FROM utterances WHERE day=?", (DAY,))["n"] == 0
    assert db.one("SELECT COUNT(*) AS n FROM utterances WHERE day=?", (OTHER,))["n"] == 1
    assert {r["status"] for r in db.query("SELECT status FROM segments WHERE id IN (?,?,?,?)", (a, b, c, silent))} == {"deleted"}
    assert db.one("SELECT status FROM segments WHERE id=?", (d,))["status"] == "done"
    assert _table_leaks(db) == []
    # idempotent: nothing is left to forget
    again = forget(db, cfg, day=DAY, dry_run=False)
    assert again["utterances"] == 0 and again["segments"]["whole"] == 0 and again["errors"] == []
    db.close_all()
    assert _file_leaks(cfg.data_dir) == []


def test_time_range_and_session_scopes(world):
    cfg, db = world["cfg"], world["db"]
    ua, ub, uc = world["ua"], world["ub"], world["uc"]
    # HH:MM needs the day; the range is [from, to) on the start of a sentence
    rep = forget(db, cfg, day=DAY, time_from="10:00", time_to="10:01:05")
    assert rep["utterances"] == 3 and rep["scope"]["time_from"] == f"{DAY}T10:00:00" and rep["segments"]["whole"] == 1
    rep = forget(db, cfg, time_from=f"{DAY}T15:00", time_to=f"{DAY} 16:00")
    assert rep["utterances"] == 2 and rep["segments"] == {"whole": 1, "partial": 0, "skipped": 0}
    assert forget(db, cfg, day=DAY, time_from="14:00", time_to="24:00")["utterances"] == 2
    # a session
    sid = db.one("SELECT session_id FROM utterances WHERE id=?", (uc[0],))["session_id"]
    rep = forget(db, cfg, session_id=sid, dry_run=False)
    assert rep["utterances"] == 2 and rep["sessions"] == {"removed": 1, "updated": 0} and sessions.get(db, sid) is None
    assert items.get(db, world["items"]["trim"])["evidence"] == [ua[1]]
    # the union of several selectors
    rep = forget(db, cfg, utterance_ids=[ua[0]], day=OTHER)
    assert rep["utterances"] == 2 and rep["days"] == [DAY, OTHER]
    assert forget(db, cfg, utterance_ids=[999999])["utterances"] == 0


def test_a_hole_in_a_session_makes_two_sessions(fcfg):
    db = DB(fcfg.db_path)
    seg = _seg(db, fcfg, "long.opus", datetime(2026, 11, 3, 9, 0, 0), dur=1200.0)
    rows = [(f"09:{m:02d}:00", m * 60.0, f"第{m}分钟说的话", 0) for m in range(0, 16, 2)]      # 09:00 .. 09:14, every 2 min
    ids = _utts(db, seg, DAY, rows)
    sessions.sync(fcfg, db, now=datetime(2026, 11, 5))
    assert db.one("SELECT COUNT(*) AS n FROM sessions")["n"] == 1
    rep = forget(db, fcfg, utterance_ids=ids[2:6], dry_run=False)                   # 09:04 .. 09:10 -> an 8 minute hole
    assert rep["sessions"] == {"removed": 0, "updated": 1}
    got = db.query("SELECT utterance_count, status FROM sessions ORDER BY started_at")
    assert [(s["utterance_count"], s["status"]) for s in got] == [(2, "closed"), (2, "closed")]
    assert db.one("SELECT COUNT(*) AS n FROM utterances WHERE session_id IS NULL")["n"] == 0


def test_a_segment_being_transcribed_is_skipped(world):
    cfg, db = world["cfg"], world["db"]
    a = world["segs"][0]
    db.set_segment_status(a, "processing")
    rep = forget(db, cfg, utterance_ids=world["ua"], dry_run=False)
    assert rep["utterances"] == 0 and rep["segments"]["skipped"] == 1 and "正在转写" in rep["errors"][0]
    assert db.one("SELECT COUNT(*) AS n FROM utterances WHERE segment_id=?", (a,))["n"] == 2
    assert (cfg.audio_dir / "a.opus").exists()


def test_voice_samples_cut_from_forgotten_audio_are_reported_not_deleted(world):
    cfg, db = world["cfg"], world["db"]
    sp = db.upsert_speaker("甲")
    on = db.add_speaker_embedding(sp, np.ones(8, dtype=np.float32), str(cfg.audio_dir / "b.opus") + "#5.5-8.5", 3.0)
    off = db.add_speaker_embedding(sp, np.ones(8, dtype=np.float32), str(cfg.audio_dir / "b.opus") + "#10.0-12.0", 2.0)
    rep = forget(db, cfg, utterance_ids=[world["ub"][1]], dry_run=False)
    assert rep["voice_samples_kept"] == [on] and any("声音样本" in n for n in rep["notes"])
    assert db.one("SELECT COUNT(*) AS n FROM speaker_embeddings WHERE id IN (?,?)", (on, off))["n"] == 2


def test_options_and_argument_errors(world):
    cfg, db = world["cfg"], world["db"]
    with pytest.raises(ValueError, match="请指定要忘记的范围"):
        forget(db, cfg)
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        forget(db, cfg, day="11月3号")
    with pytest.raises(ValueError, match="只有时间没有日期"):
        forget(db, cfg, time_from="10:00")
    with pytest.raises(ValueError, match="不是有效的时间"):
        forget(db, cfg, day=DAY, time_from="上午")
    with pytest.raises(ValueError, match="晚于"):
        forget(db, cfg, day=DAY, time_from="11:00", time_to="10:00")
    # keeping the backups and skipping the vacuum are possible, and reported
    rep = forget(db, cfg, utterance_ids=world["ua"], dry_run=False, purge_backups=False, vacuum=False)
    assert rep["backups"] == [] and rep["physical"]["vacuum"] is False
    assert (cfg.data_dir / "backups" / "audio_note-v0-20261104-000000.sqlite3").exists()
    # a backup taken BEFORE the sentences existed does not contain them and stays
    old = cfg.data_dir / "backups" / "audio_note-v0-20200101-000000.sqlite3"
    old.write_bytes(b"old")
    os.utime(old, (946684800, 946684800))
    rep = forget(db, cfg, utterance_ids=[world["ub"][1]], dry_run=False)
    assert rep["backups"] == ["audio_note-v0-20261104-000000.sqlite3"] and old.exists()


def test_forget_on_an_empty_database(fcfg):
    db = DB(fcfg.db_path)
    rep = forget(db, fcfg, day=DAY, dry_run=False)
    assert rep["utterances"] == 0 and rep["errors"] == [] and rep["notes"] == [] and rep["physical"]["wal_checkpoint"] == "ok"
