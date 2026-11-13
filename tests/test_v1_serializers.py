"""ui/serializers.py: utt_dict is the one place an utterance row becomes JSON (SPEC v1 section 8).

Existing keys never change; the schema-v2 columns add `session_id`, `text_source`, `flags`, `speaker_guess`
when the row has them; `marks` comes through a hook that package B2 fills (speech.revisions.marks).
"""
from __future__ import annotations

import logging
import sqlite3
import sys
import types
from datetime import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

import audio_note.speech as speech_pkg
from audio_note.runtime import Runtime
from audio_note.ui import serializers as sz
from audio_note.ui.app import create_app

DAY = "2026-10-30"
LEGACY_KEYS = {"id", "segment_id", "day", "abs_start", "time", "start_s", "end_s", "speaker", "speaker_id", "score", "text",
               "cluster", "loudness_db", "system_share"}


def _row(**over: Any) -> dict[str, Any]:
    """An utterances row as a dict with only the pre-v1 columns."""
    r = {"id": 5, "segment_id": 2, "day": DAY, "abs_start": f"{DAY}T09:00:07", "start_s": 1.0, "end_s": 3.5,
         "speaker_name": "张三", "speaker_id": 1, "speaker_score": 0.81234, "text": "今天讨论了预算问题",
         "text_raw": "今天讨论了预算问题", "cluster_label": 0}
    r.update(over)
    return r


class _FakeDB:
    def __init__(self, speakers: dict[int, str]):
        self.speakers = speakers
        self.queries: list[str] = []

    def query(self, sql: str, params: Any = ()) -> list[dict[str, Any]]:
        self.queries.append(sql)
        return [{"id": i, "name": n} for i, n in self.speakers.items()]


@pytest.fixture
def no_revisions(monkeypatch: pytest.MonkeyPatch):
    """speech.revisions absent (the state before package B2 lands), whatever the tree contains."""
    monkeypatch.setitem(sys.modules, "audio_note.speech.revisions", None)
    monkeypatch.delattr(speech_pkg, "revisions", raising=False)


def _fake_revisions(monkeypatch: pytest.MonkeyPatch, marks_fn: Any) -> None:
    mod = types.ModuleType("audio_note.speech.revisions")
    if marks_fn is not None:
        mod.marks = marks_fn                                         # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "audio_note.speech.revisions", mod)
    monkeypatch.setattr(speech_pkg, "revisions", mod, raising=False)


def _isolated(tmp_path, **extra) -> dict:
    """Config overrides for a Runtime under test: no vector model, and the glossary inside the temp dir
    (its default location is <project root>/glossary.txt, which a test must never write)."""
    return {"memory": {"retrieval": "fts", "embedding_model": "hash"},
            "correction": {"glossary_file": str(tmp_path / "glossary.txt")}, **extra}


# --- the legacy shape ------------------------------------------------------------------------------------------------

def test_row_without_the_new_columns_gives_exactly_the_old_shape():
    d = sz.utt_dict(_row(), "未知")
    assert d == {"id": 5, "segment_id": 2, "day": DAY, "abs_start": f"{DAY}T09:00:07", "time": "09:00:07", "start_s": 1.0,
                 "end_s": 3.5, "speaker": "张三", "speaker_id": 1, "score": 0.812, "text": "今天讨论了预算问题", "cluster": 0,
                 "loudness_db": None, "system_share": None}
    d = sz.utt_dict(_row(speaker_name=None, speaker_id=None, speaker_score=None, text_raw="今天讨论了预算", loudness_db=-31.26,
                         system_share=0.12345), "未知", True)
    assert set(d) == LEGACY_KEYS | {"text_raw", "audio"}
    assert d["speaker"] == "未知" and d["score"] is None and d["text_raw"] == "今天讨论了预算" and d["audio"] is True
    assert d["loudness_db"] == -31.3 and d["system_share"] == 0.123
    assert set(sz.utt_dict(_row(text=None, text_raw=None), "未知", False)) == LEGACY_KEYS | {"audio"}


def test_sqlite_rows_from_an_old_table_or_a_narrow_select_work():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute("CREATE TABLE utterances (id INTEGER PRIMARY KEY, segment_id INT, day TEXT, abs_start TEXT, start_s REAL, end_s REAL,"
                " speaker_name TEXT, speaker_id INT, speaker_score REAL, text TEXT, text_raw TEXT, cluster_label INT)")
    con.execute("INSERT INTO utterances VALUES (1, 2, ?, ?, 0.0, 1.0, NULL, NULL, NULL, '午饭吃什么', '午饭吃什么', 3)", (DAY, f"{DAY}T12:00:00"))
    row = con.execute("SELECT * FROM utterances").fetchone()
    d = sz.utt_dict(row, "未知")
    assert set(d) == LEGACY_KEYS and d["cluster"] == 3 and d["loudness_db"] is None
    assert sz.col(row, "flags", "absent") == "absent" and sz.col(row, "text") == "午饭吃什么"
    con.close()


# --- schema v2 columns -------------------------------------------------------------------------------------------------

def test_new_columns_are_emitted_when_the_row_has_them():
    r = _row(session_id=12, text_source="lexicon", flags='{"guard": "loop", "media": true}', speaker_guess_id=None,
             speaker_guess_score=None)
    d = sz.utt_dict(r, "未知")
    assert set(d) == LEGACY_KEYS | {"session_id", "text_source", "flags"}
    assert d["session_id"] == 12 and d["text_source"] == "lexicon" and d["flags"] == {"guard": "loop", "media": True}
    # NULL columns: the keys stay (stable shape for the page), with null / {}
    d = sz.utt_dict(_row(session_id=None, text_source=None, flags=None), "未知")
    assert d["session_id"] is None and d["text_source"] is None and d["flags"] == {}
    assert sz.utt_dict(_row(text_source=""), "未知")["text_source"] is None


@pytest.mark.parametrize("raw,expect", [(None, {}), ("", {}), ("{broken", {}), ("[1, 2]", {}), ('"text"', {}), ("{}", {}),
                                        ('{"media": true}', {"media": True}), ({"guard": "echo"}, {"guard": "echo"}), (7, {})])
def test_flags_are_parsed_and_never_break_the_row(raw: Any, expect: dict):
    assert sz.parse_flags(raw) == expect
    assert sz.utt_dict(_row(flags=raw), "未知")["flags"] == expect


def test_speaker_guess_only_for_a_sentence_nobody_is_assigned_to():
    names = {1: "张三", 4: "王敏"}
    unknown = _row(speaker_name=None, speaker_id=None, speaker_score=None, speaker_guess_id=4, speaker_guess_score=0.41267)
    d = sz.utt_dict(unknown, "未知", speaker_names=names)
    assert d["speaker_guess"] == {"id": 4, "name": "王敏", "score": 0.413} and d["speaker"] == "未知" and d["speaker_id"] is None
    assert sz.utt_dict(unknown, "未知")["speaker_guess"] == {"id": 4, "name": None, "score": 0.413}       # no name table given
    no_score = dict(unknown, speaker_guess_score=None)
    assert sz.utt_dict(no_score, "未知", speaker_names=names)["speaker_guess"] == {"id": 4, "name": "王敏", "score": None}
    gone = dict(unknown, speaker_guess_id=99)                                                             # speaker deleted since
    assert sz.utt_dict(gone, "未知", speaker_names=names)["speaker_guess"] == {"id": 99, "name": None, "score": 0.413}
    assert "speaker_guess" not in sz.utt_dict(_row(speaker_guess_id=None, speaker_guess_score=None), "未知", speaker_names=names)
    # a stale guess next to a decided speaker is not shown
    assert "speaker_guess" not in sz.utt_dict(_row(speaker_guess_id=4, speaker_guess_score=0.4), "未知", speaker_names=names)


# --- the marks hook (package B2) ----------------------------------------------------------------------------------------

def test_marks_are_only_emitted_when_given():
    assert "marks" not in sz.utt_dict(_row(), "未知")
    assert sz.utt_dict(_row(), "未知", marks=[])["marks"] == []
    m = [{"start": 2, "end": 4, "kind": "fixed", "from": "鱼酸"}]
    assert sz.utt_dict(_row(), "未知", marks=m)["marks"] == m


def test_marks_hook_is_silent_until_the_revisions_module_exists(no_revisions):
    assert sz.marks_for(_FakeDB({}), [_row()]) is None
    names, marks = sz.utterance_extras(_FakeDB({}), [_row()])
    assert names == {} and marks is None
    out = sz.utt_dicts(_FakeDB({}), [_row(), _row(id=6)], "未知")
    assert [set(d) for d in out] == [LEGACY_KEYS, LEGACY_KEYS]


def test_marks_hook_delegates_to_speech_revisions(monkeypatch: pytest.MonkeyPatch):
    seen: list[tuple] = []

    def marks(db, rows):
        seen.append((db, [r["id"] for r in rows]))
        return {5: [{"start": 0, "end": 2, "kind": "uncertain", "inbox_id": 3}]}

    _fake_revisions(monkeypatch, marks)
    db = _FakeDB({})
    out = sz.utt_dicts(db, iter([_row(), _row(id=6)]), "未知", audio_ok=lambda seg_id: seg_id == 2)
    assert seen == [(db, [5, 6])]                                                # one batched call
    assert out[0]["marks"] == [{"start": 0, "end": 2, "kind": "uncertain", "inbox_id": 3}] and out[1]["marks"] == []
    assert out[0]["audio"] is True and db.queries == []                           # no guess -> the speakers table is not read


@pytest.mark.parametrize("marks_fn", [None, lambda db, rows: (_ for _ in ()).throw(RuntimeError("half-migrated table")),
                                      lambda db, rows: ["not", "a", "dict"]])
def test_a_missing_or_failing_marks_function_never_breaks_reading(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
                                                                marks_fn: Any):
    _fake_revisions(monkeypatch, marks_fn)
    with caplog.at_level(logging.WARNING, logger="audio_note.ui.serializers"):
        assert sz.marks_for(_FakeDB({}), [_row()]) is None
    assert "marks" not in sz.utt_dicts(_FakeDB({}), [_row()], "未知")[0]


def test_utterance_extras_reads_the_speakers_table_only_when_a_row_has_a_guess(no_revisions):
    db = _FakeDB({1: "张三", 4: "王敏"})
    assert sz.utterance_extras(db, [_row(), _row(id=6, speaker_guess_id=None)]) == ({}, None) and db.queries == []
    guessed = _row(id=7, speaker_name=None, speaker_id=None, speaker_score=None, speaker_guess_id=4, speaker_guess_score=0.5)
    names, _ = sz.utterance_extras(db, [_row(), guessed])
    assert names == {1: "张三", 4: "王敏"} and len(db.queries) == 1
    out = sz.utt_dicts(db, [_row(), guessed], "未知")
    assert "speaker_guess" not in out[0] and out[1]["speaker_guess"] == {"id": 4, "name": "王敏", "score": 0.5}


def test_hint_masks_secrets():
    assert sz.hint(None) == "" and sz.hint("") == ""
    assert sz.hint("short") == "…rt" and sz.hint("sk-0123456789abcdef") == "sk-…cdef"


# --- through the API -------------------------------------------------------------------------------------------------------

@pytest.fixture
def rt(cfg_factory, tmp_path) -> Runtime:
    r = Runtime(cfg_factory(_isolated(tmp_path)))
    yield r
    r.shutdown()
    from audio_note.logging_setup import _remove_own_handlers

    _remove_own_handlers(logging.getLogger())


def _seed(rt: Runtime) -> list[int]:
    db = rt.db
    seg = db.add_segment(rt.cfg.audio_dir / "x.wav", datetime(2026, 10, 30, 9, 0), 10.0, speech_s=10.0)
    db.set_segment_status(seg, "done")
    sid = db.upsert_speaker("张三")
    return db.add_utterances([
        {"segment_id": seg, "start_s": 0.0, "end_s": 1.0, "abs_start": f"{DAY}T09:00:00", "day": DAY, "cluster_label": 0,
         "speaker_id": sid, "speaker_name": "张三", "speaker_score": 0.8, "text_raw": "今天讨论了预算问题", "text": "今天讨论了预算问题"},
        {"segment_id": seg, "start_s": 1.0, "end_s": 2.0, "abs_start": f"{DAY}T09:00:01", "day": DAY, "cluster_label": 1,
         "speaker_id": None, "speaker_name": None, "speaker_score": None, "text_raw": "午饭吃什么", "text": "午饭吃什么"}])


def _columns(rt: Runtime) -> set[str]:
    return {r["name"] for r in rt.db.query("PRAGMA table_info(utterances)")}


def test_transcript_and_context_share_one_shape_and_carry_the_marks(rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    a, b = _seed(rt)
    _fake_revisions(monkeypatch, lambda db, rows: {a: [{"start": 5, "end": 7, "kind": "fixed", "from": "鱼酸"}]})
    with TestClient(create_app(rt), raise_server_exceptions=False) as c:
        t = c.get(f"/api/transcript/{DAY}").json()
        utts = t["sessions"][0]["utterances"]
        ctx = c.get(f"/api/context/{a}", params={"before": 0, "after": 1}).json()["utterances"]
    assert [u["id"] for u in utts] == [a, b] == [u["id"] for u in ctx]
    assert utts[0]["marks"] == [{"start": 5, "end": 7, "kind": "fixed", "from": "鱼酸"}] and utts[1]["marks"] == []
    for u, v in zip(utts, ctx):
        assert LEGACY_KEYS <= set(u) and u["audio"] is False
        assert {k: x for k, x in u.items() if k != "audio"} == v            # context = the same dict minus "audio"
    assert utts[1]["speaker"] == "未知" and utts[0]["score"] == 0.8


def test_transcript_emits_the_schema_v2_fields_when_the_database_has_the_columns(rt: Runtime, no_revisions):
    a, b = _seed(rt)
    cols = _columns(rt)
    needed = {"session_id", "text_source", "flags", "speaker_guess_id", "speaker_guess_score"}
    if not needed <= cols:
        with TestClient(create_app(rt), raise_server_exceptions=False) as c:
            utts = c.get(f"/api/transcript/{DAY}").json()["sessions"][0]["utterances"]
        assert all(not (needed & set(u)) and "speaker_guess" not in u for u in utts)     # old database: old shape
        pytest.skip("this database has no schema-v2 utterance columns yet")
    sid = rt.db.one("SELECT id FROM speakers WHERE name='张三'")["id"]
    rt.db.execute("UPDATE utterances SET session_id=3, text_source='refine', flags=? WHERE id=?", ('{"media": true}', a))
    rt.db.execute("UPDATE utterances SET speaker_guess_id=?, speaker_guess_score=0.4267 WHERE id=?", (sid, b))
    with TestClient(create_app(rt), raise_server_exceptions=False) as c:
        u1, u2 = c.get(f"/api/transcript/{DAY}").json()["sessions"][0]["utterances"]
    assert (u1["session_id"], u1["text_source"], u1["flags"]) == (3, "refine", {"media": True}) and "speaker_guess" not in u1
    assert (u2["session_id"], u2["text_source"], u2["flags"]) == (None, None, {})
    assert u2["speaker_guess"] == {"id": sid, "name": "张三", "score": 0.427} and u2["speaker"] == "未知" and "marks" not in u2
