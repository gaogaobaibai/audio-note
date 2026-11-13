"""0.9.0 release decisions: what this half-way build of the 1.0 upgrade turns on by default, and what it keeps
out of sight until the second half lands."""
from __future__ import annotations

import re
from pathlib import Path

from audio_note import metrics
from audio_note.cli import helpers
from audio_note.db import DB
from audio_note.speech import correct

STATIC = Path(__file__).resolve().parents[1] / "audio_note" / "ui" / "static" / "js"


def test_per_segment_llm_correction_stays_the_default_until_session_refine_ships(cfg_factory):
    # session mode relies on speech.refine, which 0.9 does not ship: as the default it would silently drop the
    # LLM correction the owner had before the upgrade
    cfg = cfg_factory()
    assert cfg.correction.mode == "segment" and correct.correction_mode(cfg) == correct.DEFAULT_MODE == "segment"


def test_cli_clients_record_their_usage(cfg_factory):
    cfg = cfg_factory()
    db = DB(cfg.db_path)

    class Client:
        on_usage = None

    c = helpers._meter(Client(), db)
    assert callable(c.on_usage)
    c.on_usage({"purpose": "correct", "model": "m", "ok": True, "prompt_tokens": 10, "completion_tokens": 3})
    c.on_usage({"purpose": "correct", "model": "m", "ok": False})
    row = db.one("SELECT calls, errors, prompt_tokens, completion_tokens FROM usage_llm WHERE purpose='correct'")
    assert tuple(row) == (2, 1, 10, 3)
    assert metrics.summary(db, 1)["totals"]["llm"]["calls"] == 2
    assert helpers._meter(object(), db) is not None          # a double that cannot take the attribute is left alone


def test_unfinished_pages_are_out_of_the_nav():
    main = (STATIC / "main.js").read_text(encoding="utf-8")
    for page in ("inbox", "memory"):
        line = next(l for l in main.splitlines() if l.startswith(f'definePage("{page}"'))
        assert "nav: false" in line, page
    shell = (STATIC / "ui" / "shell.js").read_text(encoding="utf-8")
    # the inbox comes back into the nav (and the bell shows) only while it really holds a question
    assert 'p.navWhen === "inbox" && inboxOpen() > 0' in shell and "bell.hidden = !n;" in shell
    for page in ("inbox", "memory"):
        text = (STATIC / "pages" / f"{page}.js").read_text(encoding="utf-8")
        assert "下一版" in text and not re.search("升级后这里", text), page


def test_cli_prints_citation_markers_literally(capsys):
    # rich would read "[#12]" as a colour tag and drop it; answers must keep their citations
    helpers._print_result("灰度定在周五（[#12]、[#13]）")
    helpers._print_result({"text": "见 [#7]"})
    helpers._print_result({"答案": "[#9] 和 [bold]"})
    helpers._print_result([{"speaker": "[#1]", "text": "[#2] 好", "start": 1.0}])
    out = capsys.readouterr().out
    for marker in ("[#12]", "[#13]", "[#7]", "[#9]", "[bold]", "[#1]", "[#2]"):
        assert marker in out, marker


def test_memory_item_line_says_the_title_once():
    from audio_note.cli import memory_tree

    line = memory_tree._item_line({"id": 2, "day": "2026-11-02", "kind": "todo", "title": "完成压测",
                                   "content": "完成压测（截止：下午）", "owner": "某人"})
    assert line.count("完成压测") == 1 and "截止：下午" in line
    other = memory_tree._item_line({"id": 3, "day": "", "kind": "decision", "title": "灰度", "content": "定在周五 [#4]"})
    assert "灰度：定在周五 \[#4]" in other


# --------------------------------------------------------------------------- forget: safety fixes found before 0.9
def _forget_world(cfg_factory, tmp_path):
    """A profile whose database also holds a segment recorded by ANOTHER profile (a copied / restored database
    keeps absolute paths): forgetting that day must not delete the other profile's file."""
    from audio_note.db import DB

    cfg = cfg_factory()
    db = DB(cfg.db_path)
    other = tmp_path / "other_profile" / "audio" / "2026-11-04"
    other.mkdir(parents=True)
    foreign = other / "100000_000.opus"
    foreign.write_bytes(b"x" * 10)
    own_dir = cfg.audio_dir / "2026-11-04"
    own_dir.mkdir(parents=True, exist_ok=True)
    own = own_dir / "110000_000.opus"
    own.write_bytes(b"y" * 10)
    ids = []
    for i, (path, hhmm) in enumerate(((foreign, "10:00:00"), (own, "11:00:00"))):
        sid = db.execute("INSERT INTO segments(path, started_at, duration_s, status, created_at) VALUES (?,?,?,?,?)",
                         (str(path), f"2026-11-04T{hhmm}", 5.0, "done", "2026-11-04 12:00:00")).lastrowid
        uid = db.execute("INSERT INTO utterances(segment_id, day, abs_start, start_s, end_s, text, text_raw, created_at)"
                         " VALUES (?,?,?,?,?,?,?,?)", (sid, "2026-11-04", f"2026-11-04T{hhmm}", 0.0, 4.0, f"句子{i}", f"句子{i}",
                                                       "2026-11-04 12:00:00")).lastrowid
        ids.append(uid)
    return cfg, db, foreign, own, ids


def test_forget_never_deletes_audio_outside_the_profile(cfg_factory, tmp_path):
    from audio_note.memory import forget as fg

    cfg, db, foreign, own, _ = _forget_world(cfg_factory, tmp_path)
    dry = fg.forget(db, cfg, day="2026-11-04")
    assert dry["audio_files"] == 1 and len(dry["audio_outside"]) == 1 and any("不在这个档案" in n for n in dry["notes"])
    rep = fg.forget(db, cfg, day="2026-11-04", dry_run=False, purge_backups=False)
    assert foreign.exists(), "a file of another profile was deleted"
    assert not own.exists() and rep["audio_files"] == 1


def test_forgetting_part_of_a_day_keeps_its_evidence_less_items(cfg_factory, tmp_path):
    from audio_note.memory import forget as fg

    cfg, db, _, _, ids = _forget_world(cfg_factory, tmp_path)
    db.execute("INSERT INTO memory_items(kind, title, content, day, source, status, fingerprint, created_at, updated_at)"
               " VALUES ('todo', '交周报', '交周报', '2026-11-04', 'digest', 'active', 'fp-todo', '2026-11-04', '2026-11-04')")
    part = fg.forget(db, cfg, utterance_ids=[ids[1]], dry_run=False, purge_backups=False)
    assert part["items"]["removed"] == 0 and part["items"]["unverified"] == 1
    assert db.one("SELECT status FROM memory_items WHERE fingerprint='fp-todo'")["status"] == "active"
    whole = fg.forget(db, cfg, day="2026-11-04", dry_run=True)
    assert whole["items"]["removed"] == 1                    # a whole day still takes them along (the digest rebuilds)


def test_forget_tombstones_do_not_block_the_name_forever(cfg_factory, tmp_path):
    import json

    from audio_note.memory import forget as fg

    cfg, db, _, _, ids = _forget_world(cfg_factory, tmp_path)
    db.execute("INSERT INTO memory_items(kind, title, content, day, source, status, fingerprint, evidence, created_at, updated_at)"
               " VALUES ('person', '某同事', '一起开会', '2026-11-04', 'digest', 'active', 'fp-person', ?, '2026-11-04', '2026-11-04')",
               (json.dumps([ids[1]]),))
    rep = fg.forget(db, cfg, utterance_ids=[ids[1]], dry_run=False, purge_backups=False)
    assert rep["items"]["tombstoned"] == 1
    row = db.one("SELECT status, title, fingerprint FROM memory_items WHERE kind='person'")
    assert row["status"] == "rejected" and row["title"] == "" and row["fingerprint"] is None


def test_scheduler_rewrites_digests_removed_by_forget(cfg_factory, tmp_path):
    from datetime import datetime

    from audio_note.memory.forget import KV_REBUILD_DAYS
    from audio_note.scheduler import Scheduler

    cfg, db, _, _, _ = _forget_world(cfg_factory, tmp_path)
    calls = []
    db.set_kv(KV_REBUILD_DAYS, ["2026-11-03", "2026-11-04"])           # 11-03 has nothing left
    sch = Scheduler(cfg, db, jobs={"daily": lambda day, push: calls.append((day, push)) or {"day": day}})
    now = datetime(2026, 11, 13, 9, 0)
    assert sch._rebuild_forgotten(now) is False and db.get_kv(KV_REBUILD_DAYS) == ["2026-11-04"]
    assert sch._rebuild_forgotten(now) is True and calls == [("2026-11-04", False)]   # never pushed again
    assert db.get_kv(KV_REBUILD_DAYS) == []
