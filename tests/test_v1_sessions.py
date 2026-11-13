"""memory.sessions: cutting sentences into sessions (SPEC v1 section 7.1) and the HTTP / chain surface of it."""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta

import pytest

from audio_note import chain
from audio_note.db import DB
from audio_note.events import EventBus
from audio_note.memory import sessions

DAY = "2026-11-03"
NOW = datetime(2026, 11, 3, 23, 0, 0)


@pytest.fixture
def seg(db):
    return db.add_segment("x.opus", datetime(2026, 11, 3, 9, 0, 0), 3600.0)


def add(db: DB, seg: int, when: str, text: str = "一句话", dur: float = 2.0, speaker: str | None = None,
        speaker_id: int | None = None) -> int:
    """One sentence at 'HH:MM[:SS]' of DAY (or a full ISO time)."""
    ts = when if "T" in when else f"{DAY}T{when if when.count(':') == 2 else when + ':00'}"
    return db.add_utterances([{"segment_id": seg, "start_s": 0.0, "end_s": dur, "abs_start": ts, "day": ts[:10],
                               "speaker_name": speaker, "speaker_id": speaker_id, "text_raw": text, "text": text}])[0]


def session_of(db: DB, uid: int) -> int | None:
    return db.one("SELECT session_id FROM utterances WHERE id=?", (uid,))["session_id"]


def rows(db: DB) -> list[dict]:
    return [dict(r) for r in db.query("SELECT * FROM sessions ORDER BY started_at, id")]


# ---------------------------------------------------------------------------------------------- cutting
def test_gap_cuts_sessions_and_fills_every_column(cfg, db, seg):
    a = [add(db, seg, "09:00:00", "早上好"), add(db, seg, "09:02:00", "开始吧", dur=4.0), add(db, seg, "09:06:30", "好的")]
    b = [add(db, seg, "09:12:00", "另一段"), add(db, seg, "09:12:10", "继续说")]            # 5.5 min after 09:06:32
    out = sessions.sync(cfg, db, now=NOW)
    assert (out["assigned"], out["created"], out["closed"], out["open"], out["changed"]) == (5, 2, 2, 0, True)
    s1, s2 = rows(db)
    assert {session_of(db, u) for u in a} == {s1["id"]} and {session_of(db, u) for u in b} == {s2["id"]}
    assert s1["day"] == DAY and s1["started_at"] == f"{DAY}T09:00:00.000" and s1["ended_at"] == f"{DAY}T09:06:32.000"
    assert (s1["utterance_count"], s1["speech_s"], s1["chars"], s1["status"]) == (3, 8.0, 3 + 3 + 2, "closed")
    assert (s1["raw_hash"], s1["text_hash"]) == db.session_hashes(s1["id"]) and s1["raw_hash"] and s1["text_hash"]
    assert s1["refined_hash"] is None and s1["digested_hash"] is None          # never written here: "hash differs" = work to do


def test_a_gap_of_exactly_the_limit_stays_in_the_session(cfg, db, seg):
    add(db, seg, "09:00:00", dur=0.0)
    add(db, seg, "09:05:00")            # exactly 5 minutes after the end of the first
    add(db, seg, "09:10:03")            # 5 min 1 s after the end of the second (09:05:02)
    sessions.sync(cfg, db, now=NOW)
    assert [s["utterance_count"] for s in rows(db)] == [2, 1]


def test_gap_comes_from_the_config(cfg_factory):
    cfg = cfg_factory({"memory": {"session_gap_min": 1}})
    db = DB(cfg.db_path)
    seg = db.add_segment("x.opus", datetime(2026, 11, 3, 9, 0, 0), 600.0)
    for t in ("09:00:00", "09:00:30", "09:02:00"):
        add(db, seg, t)
    sessions.sync(cfg, db, now=NOW)
    assert [s["utterance_count"] for s in rows(db)] == [2, 1]
    assert sessions.gap_minutes(cfg) == 1.0
    cfg["memory"]["session_gap_min"] = "nonsense"
    assert sessions.gap_minutes(cfg) == sessions.DEFAULT_GAP_MIN
    cfg["memory"]["session_gap_min"] = 0
    assert sessions.gap_minutes(cfg) == sessions.DEFAULT_GAP_MIN


def test_sync_is_idempotent_and_cheap_when_nothing_changed(cfg, db, seg):
    for m in range(0, 30, 2):
        add(db, seg, f"09:{m:02d}:00")
    first = sessions.sync(cfg, db, now=NOW)
    before = rows(db)
    again = sessions.sync(cfg, db, now=NOW)
    assert first["changed"] is True and again["changed"] is False
    assert {k: again[k] for k in ("assigned", "created", "updated", "merged", "removed", "closed", "reopened")} == \
        {"assigned": 0, "created": 0, "updated": 0, "merged": 0, "removed": 0, "closed": 0, "reopened": 0}
    assert rows(db) == before                                                   # not even updated_at moved
    assert again["wall_ms"] < 50


def test_open_until_the_gap_has_passed_then_closed(cfg, db, seg):
    add(db, seg, "09:00:00")
    out = sessions.sync(cfg, db, now=datetime(2026, 11, 3, 9, 3, 0))
    assert (out["created"], out["closed"], out["open"]) == (1, 0, 1) and rows(db)[0]["status"] == "open"
    assert sessions.sync(cfg, db, now=datetime(2026, 11, 3, 9, 4, 59))["changed"] is False
    out = sessions.sync(cfg, db, now=datetime(2026, 11, 3, 9, 5, 3))
    assert out["closed"] == 1 and out["open"] == 0 and rows(db)[0]["status"] == "closed"


def test_new_sentences_extend_an_open_session_and_reopen_a_closed_one(cfg, db, seg):
    add(db, seg, "09:00:00")
    sessions.sync(cfg, db, now=datetime(2026, 11, 3, 9, 1, 0))
    sid = rows(db)[0]["id"]
    h0 = rows(db)[0]["raw_hash"]
    # still open: the next sentence joins it
    u2 = add(db, seg, "09:03:00")
    out = sessions.sync(cfg, db, now=datetime(2026, 11, 3, 9, 3, 30))
    assert (out["assigned"], out["created"], out["updated"], out["reopened"]) == (1, 0, 1, 0)
    assert session_of(db, u2) == sid and rows(db)[0]["raw_hash"] != h0 and rows(db)[0]["utterance_count"] == 2
    # closed, refined and digested ...
    sessions.sync(cfg, db, now=datetime(2026, 11, 3, 9, 30, 0))
    s = rows(db)[0]
    assert s["status"] == "closed"
    db.execute("UPDATE sessions SET refined_hash=raw_hash, digested_hash=text_hash, title='晨会', summary='说了两句' WHERE id=?", (sid,))
    # ... then a late sentence arrives (a slow segment): open again, and the hashes say "do the work again"
    u3 = add(db, seg, "09:06:00")
    out = sessions.sync(cfg, db, now=datetime(2026, 11, 3, 9, 8, 0))
    s = rows(db)[0]
    assert out["reopened"] == 1 and s["status"] == "open" and session_of(db, u3) == sid and s["utterance_count"] == 3
    assert s["refined_hash"] != s["raw_hash"] and s["digested_hash"] != s["text_hash"]
    assert s["title"] == "晨会"                                                 # what the episode step wrote stays until it runs again
    # an old recording imported later lands in its (closed) session and leaves it closed, in one call
    u4 = add(db, seg, "09:07:00")
    sessions.sync(cfg, db, now=datetime(2026, 11, 3, 9, 30, 0))
    out = sessions.sync(cfg, db, now=NOW)
    assert rows(db)[0]["status"] == "closed" and session_of(db, u4) == sid and out["reopened"] == 0


def test_a_sentence_that_bridges_two_sessions_merges_them(cfg, db, seg):
    a = add(db, seg, "09:00:00")
    b = add(db, seg, "09:08:00")
    sessions.sync(cfg, db, now=NOW)
    first, second = (s["id"] for s in rows(db))
    db.execute("UPDATE memory_items SET session_id=NULL")
    item = db.execute("INSERT INTO memory_items(kind, title, session_id) VALUES ('fact', '第二段里的事', ?)", (second,)).lastrowid
    db.execute("INSERT INTO inbox_items(kind, title, session_id) VALUES ('term', '问题', ?)", (second,))
    db.execute("INSERT INTO memory_vectors(kind, ref_id, view, model, dim, vec) VALUES ('session', ?, '', 'm', 1, X'00')", (second,))
    mid = add(db, seg, "09:04:00")                                  # 4 min after a, 4 min before b
    out = sessions.sync(cfg, db, now=NOW)
    assert (out["merged"], out["created"], out["assigned"]) == (1, 0, 1)
    got = rows(db)
    assert len(got) == 1 and got[0]["id"] == first and got[0]["utterance_count"] == 3       # the earlier session survives
    assert {session_of(db, u) for u in (a, b, mid)} == {first}
    assert got[0]["started_at"].endswith("09:00:00.000") and got[0]["ended_at"].endswith("09:08:02.000")
    assert db.one("SELECT session_id FROM memory_items WHERE id=?", (item,))["session_id"] == first
    assert db.one("SELECT session_id FROM inbox_items")["session_id"] == first
    assert db.one("SELECT COUNT(*) AS n FROM memory_vectors WHERE kind='session'")["n"] == 0


def test_sessions_cross_midnight_and_belong_to_the_day_they_started(cfg, db, seg):
    ids = [add(db, seg, "2026-11-03T23:57:00"), add(db, seg, "2026-11-03T23:59:30"), add(db, seg, "2026-11-04T00:02:00"),
           add(db, seg, "2026-11-04T00:04:00")]
    add(db, seg, "2026-11-04T08:00:00")
    sessions.sync(cfg, db, now=datetime(2026, 11, 5))
    night, morning = rows(db)
    assert night["day"] == "2026-11-03" and night["utterance_count"] == 4 and {session_of(db, u) for u in ids} == {night["id"]}
    assert [s["id"] for s in sessions.for_day(db, "2026-11-03")] == [night["id"]]
    both = sessions.for_day(db, "2026-11-04")                         # the night session has sentences on the 4th too
    assert [s["id"] for s in both] == [night["id"], morning["id"]] and both[0]["crosses_midnight"] is True
    assert both[1]["crosses_midnight"] is False and sessions.for_day(db, "2026-11-05") == []


def test_out_of_order_arrival_gives_the_same_sessions(cfg, db, seg):
    times = ["09:00:00", "09:01:00", "09:20:00", "09:21:00", "09:22:00", "10:00:00"]
    for t in reversed(times):                                         # newest first, one sync per sentence
        add(db, seg, t)
        sessions.sync(cfg, db, now=NOW)
    assert [s["utterance_count"] for s in rows(db)] == [2, 3, 1]
    assert db.one("SELECT COUNT(*) AS n FROM utterances WHERE session_id IS NULL")["n"] == 0


def test_deletions_made_elsewhere_are_repaired(cfg, db, seg):
    a = [add(db, seg, f"09:{m:02d}:00") for m in (0, 2, 4, 6, 8, 10, 12, 14)]
    b = [add(db, seg, "10:00:00"), add(db, seg, "10:01:00")]
    sessions.sync(cfg, db, now=NOW)
    sa, sb = (s["id"] for s in rows(db))
    # a session loses everything (the segment was processed again): the row goes
    db.execute(f"DELETE FROM utterances WHERE id IN ({b[0]},{b[1]})")
    out = sessions.sync(cfg, db, now=NOW)
    assert out["removed"] == 1 and [s["id"] for s in rows(db)] == [sa]
    # a session loses its middle: counts, hashes, and a cut at the hole (09:02 .. 09:12 is longer than the gap)
    db.execute(f"DELETE FROM utterances WHERE id IN ({a[2]},{a[3]},{a[4]},{a[5]})")
    out = sessions.sync(cfg, db, now=NOW)
    got = rows(db)
    assert out["updated"] == 1 and out["created"] == 1 and [s["utterance_count"] for s in got] == [2, 2]
    assert all((s["raw_hash"], s["text_hash"]) == db.session_hashes(s["id"]) for s in got)
    # sentences pointing at a session that no longer exists are picked up again
    db.execute("DELETE FROM sessions WHERE id=?", (got[1]["id"],))
    out = sessions.sync(cfg, db, now=NOW)
    assert out["assigned"] == 2 and len(rows(db)) == 2 and sessions.sync(cfg, db, now=NOW)["changed"] is False


def test_refresh_follows_a_text_edit(cfg, db, seg):
    u = add(db, seg, "09:00:00", "原来的话")
    sessions.sync(cfg, db, now=NOW)
    sid = session_of(db, u)
    before = rows(db)[0]
    db.execute("UPDATE utterances SET text='改过的话更长一点' WHERE id=?", (u,))
    assert sessions.refresh(db, sid) is True
    after = rows(db)[0]
    assert after["raw_hash"] == before["raw_hash"] and after["text_hash"] != before["text_hash"] and after["chars"] == 8
    db.execute("DELETE FROM utterances WHERE id=?", (u,))
    assert sessions.refresh(db, sid) is False and rows(db) == []


# ---------------------------------------------------------------------------------------------- scale, stop, events
def test_two_thousand_sentences_are_cut_in_well_under_a_second(cfg, db, seg):
    base = datetime(2026, 11, 3, 8, 0, 0)
    batch = []
    t = base
    for i in range(2000):
        t += timedelta(seconds=3 if i % 40 else 400)                # a 400 s pause every 40 sentences
        batch.append({"segment_id": seg, "start_s": 0.0, "end_s": 2.0, "abs_start": t.isoformat(timespec="milliseconds"),
                      "day": t.strftime("%Y-%m-%d"), "text_raw": f"第{i}句", "text": f"第{i}句"})
    db.add_utterances(batch)
    t0 = time.perf_counter()
    out = sessions.sync(cfg, db, now=datetime(2026, 11, 10))
    wall = time.perf_counter() - t0
    assert out["assigned"] == 2000 and out["created"] == 50 and out["closed"] == 50
    assert wall < 1.0, f"sync of 2000 sentences took {wall:.2f}s"
    t0 = time.perf_counter()
    assert sessions.sync(cfg, db, now=datetime(2026, 11, 10))["changed"] is False
    assert time.perf_counter() - t0 < 0.05


def test_batches_and_stop_event(cfg, db, seg, monkeypatch):
    monkeypatch.setattr(sessions, "BATCH", 3)
    for m in range(8):
        add(db, seg, f"09:{m:02d}:00")
    stop = threading.Event()
    stop.set()
    out = sessions.sync(cfg, db, stop_event=stop, now=NOW)            # a stop before the first batch: nothing assigned, no error
    assert out["assigned"] == 0 and db.one("SELECT COUNT(*) AS n FROM utterances WHERE session_id IS NULL")["n"] == 8
    out = sessions.sync(cfg, db, now=NOW)                             # several batches, still one session
    assert out["assigned"] == 8 and [s["utterance_count"] for s in rows(db)] == [8]


def test_limit_is_ignored_and_client_is_optional(cfg, db, seg):
    for m in range(0, 40, 10):
        add(db, seg, f"09:{m:02d}:00")
    out = sessions.sync(cfg, db, None, limit=1, now=NOW)
    assert out["assigned"] == 4 and out["created"] == 4


def test_events_name_what_happened(cfg, db, seg):
    bus = EventBus()
    add(db, seg, "09:00:00")
    sessions.sync(cfg, db, events=bus, now=datetime(2026, 11, 3, 9, 1, 0))
    add(db, seg, "09:02:00")
    sessions.sync(cfg, db, events=bus, now=datetime(2026, 11, 3, 9, 3, 0))
    sessions.sync(cfg, db, events=bus, now=NOW)
    sessions.sync(cfg, db, events=bus, now=NOW)                       # nothing changed: no event
    evs = [(e["type"], e["change"], e["day"]) for e in bus.recent()]
    assert evs == [("session", "created", DAY), ("session", "updated", DAY), ("session", "closed", DAY)]
    assert bus.recent()[0]["id"] == rows(db)[0]["id"]


def test_the_chain_runs_it_first_and_the_scheduler_sees_no_leftovers(cfg, db, seg):
    assert chain.STEPS[0] == ("sessions", "audio_note.memory.sessions:sync")
    add(db, seg, "09:00:00")
    out = chain.run_chain(cfg, db, lambda: None, only=("sessions",))
    assert out["ran"] == ["sessions"] and out["failed"] == {} and out["results"]["sessions"]["assigned"] == 1
    assert chain.run_chain(cfg, DB(cfg.db_path), None, only=("sessions",))["results"]["sessions"]["changed"] is False


# ---------------------------------------------------------------------------------------------- reading
def test_for_day_get_and_lines(cfg, db, seg):
    me = db.upsert_speaker("甲")
    u1 = add(db, seg, "09:00:00", "大家  好\n开始", speaker="甲", speaker_id=me)
    u2 = add(db, seg, "09:00:10", "好的", speaker=None)
    u3 = add(db, seg, "09:00:20", "那我先说", speaker="甲", speaker_id=me)
    add(db, seg, "11:00:00", "另一场")
    sessions.sync(cfg, db, now=NOW)
    day = sessions.for_day(db, DAY, unknown="未知")
    assert [s["utterance_count"] for s in day] == [3, 1]
    s = day[0]
    assert (s["start"], s["end"], s["status"], s["duration_s"], s["refined"], s["digested"]) == ("09:00", "09:00", "closed", 22.0, False, False)
    assert s["topics"] == [] and s["participants"] == [] and s["title"] is None
    assert s["speakers"] == [{"id": me, "name": "甲", "n": 2, "speech_s": 4.0}, {"id": None, "name": "未知", "n": 1, "speech_s": 2.0}]
    db.execute("UPDATE sessions SET topics='[\"发布\"]', participants='[\"甲\"]', refined_hash=raw_hash, meta='{\"k\": 1}' WHERE id=?", (s["id"],))
    one = sessions.get(db, s["id"], with_utterances=True)
    assert one["topics"] == ["发布"] and one["participants"] == ["甲"] and one["refined"] is True and one["meta"] == {"k": 1}
    assert [u["id"] for u in one["utterances"]] == [u1, u2, u3] and "utterances" not in sessions.get(db, s["id"])
    assert sessions.get(db, 99999) is None
    assert sessions.lines(db, s["id"]) == [
        {"i": 0, "id": u1, "time": "09:00", "speaker": "甲", "text": "大家 好 开始"},
        {"i": 1, "id": u2, "time": "09:00", "speaker": "未知", "text": "好的"},
        {"i": 2, "id": u3, "time": "09:00", "speaker": "甲", "text": "那我先说"}]
    assert sessions.lines(db, 99999) == []
    assert sessions.ids_for(db, [u1, u3, 424242]) == [s["id"]]
