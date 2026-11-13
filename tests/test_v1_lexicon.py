"""audio_note.speech.lexicon: the one vocabulary (SPEC v1 section 6.1). No network, no models.

All names and sentences are invented: 星河 is a project, 兴和 / 新河 what a recogniser might write instead.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from audio_note.db import DB
from audio_note.speech import correct, glossary_file, lexicon, pinyin, revisions
from audio_note.speech.lexicon import Lexicon

DAY = "2026-11-02"
needs_pypinyin = pytest.mark.skipif(not pinyin.available(), reason="pypinyin is not installed")


def seed(db: DB, cfg: Any, texts: list[str], day: str = DAY) -> list[int]:
    """One done segment (no audio file) with one sentence per text."""
    seg = db.add_segment(cfg.audio_dir / f"{day}-{len(texts)}.wav", datetime.fromisoformat(f"{day}T09:00:00"), 60.0, speech_s=30.0)
    db.set_segment_status(seg, "done")
    return db.add_utterances([{
        "segment_id": seg, "start_s": float(i), "end_s": float(i + 1), "abs_start": f"{day}T09:{i:02d}:00", "day": day,
        "cluster_label": 0, "speaker_id": None, "speaker_name": None, "speaker_score": None, "text_raw": t, "text": t,
        "text_seg": t} for i, t in enumerate(texts)])


def texts_of(db: DB) -> list[str]:
    return [r["text"] for r in db.query("SELECT text FROM utterances ORDER BY id")]


# ---------------------------------------------------------------------------------------------- entries and aliases
def test_upsert_term_creates_once_and_updates_without_losing_what_is_there(db):
    a = lexicon.upsert_term(db, "  星河  ", kind="project", note="正在做的项目")
    assert lexicon.upsert_term(db, "星河") == a                                   # same entry, defaults change nothing
    t = lexicon.get_term(db, a)
    assert (t["term"], t["kind"], t["note"], t["source"], t["status"], t["hotword"]) == ("星河", "project", "正在做的项目", "user", "active", True)
    if pinyin.available():
        assert t["pinyin"] == "xing he"
    b = lexicon.upsert_term(db, "OKR")
    assert lexicon.upsert_term(db, "okr", kind="term") == b                        # Latin spellings: one entry in any case
    lexicon.upsert_term(db, "星河", status="candidate")                           # never downgrades an active entry
    assert lexicon.get_term(db, a)["status"] == "active"
    c = lexicon.upsert_term(db, "新词", source="auto", status="candidate")
    lexicon.upsert_term(db, "新词", source="answer", kind="product")              # confirmed: active, source and kind follow
    t = lexicon.get_term(db, c)
    assert (t["status"], t["source"], t["kind"]) == ("active", "answer", "product")
    for bad in ("", "   ", "，。", "字" * 41):
        with pytest.raises(ValueError):
            lexicon.upsert_term(db, bad)
    with pytest.raises(ValueError):
        lexicon.upsert_term(db, "词", kind="animal")


def test_alias_scope_is_decided_from_what_would_be_safe(db, cfg):
    star = lexicon.upsert_term(db, "星河", kind="project")
    other = lexicon.upsert_term(db, "新和")
    rows = {a["alias"]: a for a in []}

    def scope(alias: str, tid: int = star, **kw: Any) -> str:
        aid = lexicon.add_alias(db, tid, alias, cfg=cfg, **kw)
        rows[alias] = lexicon.get_alias(db, aid)
        return rows[alias]["scope"]

    assert scope("兴和") == "auto"                                # not a dictionary word to speak of
    assert scope("河") == "hint"                                  # one character
    assert scope("新河") == "hint"                                # an everyday dictionary word (frequency above the limit)
    assert scope("新和") == "hint"                                # the spelling of another entry
    assert lexicon.auto_scope(db, star, "河")[1] == "single" and lexicon.auto_scope(db, star, "新和")[1] == "term"
    assert lexicon.auto_scope(db, star, "新河", cfg)[1] == "common"
    # the same mis-hearing for two entries: nobody can tell which one is meant, both become hints
    assert scope("星和") == "auto"
    assert scope("星和", other) == "hint"
    assert lexicon.get_alias(db, rows["星和"]["id"])["scope"] == "hint"
    # an explicit scope is taken as given, and adding the pair again updates instead of duplicating
    assert scope("新河", scope="auto") == "auto"
    assert len([a for a in lexicon.get_term(db, star)["aliases"] if a["alias"] == "新河"]) == 1
    # an "aka" is another name: never rewritten, whatever was asked
    assert scope("银河计划", kind="aka", scope="auto") == "hint" and rows["银河计划"]["kind"] == "aka"
    with pytest.raises(ValueError):
        lexicon.add_alias(db, star, "星河")                       # itself
    with pytest.raises(LookupError):
        lexicon.add_alias(db, 999, "某某")
    with pytest.raises(ValueError):
        lexicon.add_alias(db, star, "某某", scope="always")


def test_alias_frequency_limit_comes_from_the_config(db, cfg_factory):
    cfg = cfg_factory({"correction": {"alias_auto_max_freq": 1000000}})
    tid = lexicon.upsert_term(db, "星河")
    assert lexicon.word_frequency("新河") > lexicon.DEFAULT_ALIAS_AUTO_MAX_FREQ and lexicon.word_frequency("兴和") <= 50
    assert lexicon.auto_scope(db, tid, "新河")[0] == "hint" and lexicon.auto_scope(db, tid, "新河", cfg)[0] == "auto"
    assert lexicon.word_frequency("这不是一个词语") == 0


# ---------------------------------------------------------------------------------------------- the snapshot
def test_load_is_cached_per_database_and_every_write_invalidates(db, cfg, tmp_path):
    a = Lexicon.load(db)
    assert Lexicon.load(db) is a and a.terms == []
    tid = lexicon.upsert_term(db, "星河")
    b = Lexicon.load(db)
    assert b is not a and [t.term for t in b.active] == ["星河"] and Lexicon.load(db) is b
    aid = lexicon.add_alias(db, tid, "兴和")
    c = Lexicon.load(db)
    assert c is not b and c.alias_by_id[aid][1].alias == "兴和"
    lexicon.touch(db, lexicon_ids=[tid, tid], alias_ids=[aid])
    d = Lexicon.load(db)
    assert d.by_id[tid].hits == 2 and d.alias_by_id[aid][1].hits == 1 and d.by_id[tid].last_hit_at
    # another connection / process writing is seen too: the version lives in the database
    other = DB(cfg.db_path)
    lexicon.update_term(other, tid, note="改过")
    assert Lexicon.load(db).by_id[tid].note == "改过"
    other.close_all()
    # two database files never share a snapshot
    db2 = DB(tmp_path / "other.sqlite3")
    assert Lexicon.load(db2).terms == [] and Lexicon.load(db).by_term["星河"].id == tid
    db2.close_all()


def test_find_in_prefers_the_longest_spelling_and_respects_latin_word_edges(db):
    star = lexicon.upsert_term(db, "星河")
    app = lexicon.upsert_term(db, "星河App")
    sdk = lexicon.upsert_term(db, "SDK")
    lexicon.add_alias(db, star, "兴和")
    lexicon.add_alias(db, sdk, "sdk包", scope="auto")
    lex = Lexicon.load(db)
    hits = lex.find_in("兴和和星河app都接了SDK，MSDKX不算，sdk包也是")
    assert [(h.text, h.term.id, h.alias.alias if h.alias else None) for h in hits] == [
        ("兴和", star, "兴和"), ("星河app", app, None), ("SDK", sdk, None), ("sdk包", sdk, "sdk包")]
    assert [(h.start, h.end) for h in hits][:2] == [(0, 2), (3, 8)]
    assert lex.find_in("") == [] and Lexicon().find_in("星河") == []
    assert lex.resolve("兴和").id == star and lex.resolve("SDK").id == sdk and lex.resolve("没有") is None
    assert lex.is_term("星河") and not lex.is_term("兴和")


def test_apply_rewrites_only_safe_aliases_and_reports_positions(db, cfg):
    star = lexicon.upsert_term(db, "星河", kind="project")
    road = lexicon.upsert_term(db, "兴和路", kind="place")
    auto = lexicon.add_alias(db, star, "兴和", cfg=cfg)
    hint = lexicon.add_alias(db, star, "新河", cfg=cfg)                     # common word -> hint
    aka = lexicon.add_alias(db, star, "银河计划", kind="aka")
    lex = Lexicon.load(db)
    text = "兴和的排期和兴和路无关，新河也不是，银河计划是别称，兴和要上线"
    new, edits = lex.apply(text)
    assert new == "星河的排期和兴和路无关，新河也不是，银河计划是别称，星河要上线"
    assert [(e["from"], e["to"], e["alias_id"], e["lexicon_id"], e["reason"]) for e in edits] == [
        ("兴和", "星河", auto, star, "词库"), ("兴和", "星河", auto, star, "词库")]
    for e in edits:
        assert text[e["pos"]:e["pos"] + 2] == "兴和" and new[e["start"]:e["end"]] == "星河"
    assert lex.apply("没有要改的") == ("没有要改的", [])
    # explicit request for one alias: the hint is applied too, nothing else
    forced, e2 = lex.apply(text, only_alias_ids=[hint], force=True)
    assert forced.count("星河") == 1 and "新河" not in forced and forced.count("兴和") == 3 and len(e2) == 1
    assert lex.apply(text, only_alias_ids=[aka], force=True)[1] == []       # an aka is never rewritten
    assert lex.apply(text, skip_alias_ids=[auto]) == (text, [])
    assert lex.apply(text, only_lexicon_ids=[road]) == (text, [])
    # a disabled entry's aliases stop applying
    lexicon.update_term(db, star, status="disabled")
    assert Lexicon.load(db).apply(text) == (text, [])


def test_apply_with_longer_and_shorter_replacements_keeps_offsets_right(db):
    t1 = lexicon.upsert_term(db, "灰度发布")
    t2 = lexicon.upsert_term(db, "岚")
    lexicon.add_alias(db, t1, "灰度", scope="auto")
    lexicon.add_alias(db, t2, "小兰姐", scope="auto")
    new, edits = Lexicon.load(db).apply("先做灰度，再问小兰姐，最后灰度")
    assert new == "先做灰度发布，再问岚，最后灰度发布"
    assert [new[e["start"]:e["end"]] for e in edits] == ["灰度发布", "岚", "灰度发布"]
    assert Lexicon.load(db).apply("已经是灰度发布了")[1] == []                 # inside the longer canonical spelling: untouched


@needs_pypinyin
def test_near_finds_sound_alikes_that_are_not_spelled_out(db):
    star = lexicon.upsert_term(db, "星河", kind="project")
    lexicon.upsert_term(db, "陈默", kind="person")
    lexicon.upsert_term(db, "南湖路", kind="place")
    lexicon.upsert_term(db, "CDN")
    lexicon.upsert_term(db, "默")                                               # single syllable: never a candidate
    lexicon.add_alias(db, star, "兴和")
    lex = Lexicon.load(db)
    text = "幸和的排期，沉默说南山路的cbn要续费，星河和兴和都在"
    spans = lex.near_spans(text)
    got = {(h.term.term, h.text) for h in spans}
    assert got == {("星河", "幸和"), ("陈默", "沉默"), ("南湖路", "南山路"), ("CDN", "cbn")}
    assert all(text[h.start:h.end] == h.text for h in spans)
    assert [h.similarity for h in spans] == sorted((h.similarity for h in spans), reverse=True)
    assert [t.term for t in lex.near(text)][:2] == ["星河", "陈默"] and len(lex.near(text, limit=1)) == 1
    # the strict threshold drops the "one syllable of three" and "one letter of three" candidates
    assert {h.term.term for h in lex.near_spans(text, min_sim=pinyin.DEFAULT_MIN_SIM)} == {"星河", "陈默"}
    assert lex.near_spans("今天天气不错") == [] and lex.near("") == []


def test_hotwords_order_and_filters(db):
    sid = db.upsert_speaker("王敏")
    lexicon.sync_sources(db, type("C", (), {"glossary_path": None})())
    g = lexicon.import_terms(db, ["埋点"], source="glossary")
    assert g["added"] == 1
    ans = lexicon.upsert_term(db, "星河", source="answer")
    usr = lexicon.upsert_term(db, "压测", source="user")
    lexicon.upsert_term(db, "默")                                               # one character: no hot word
    off = lexicon.upsert_term(db, "不要", hotword=False)
    cand = lexicon.upsert_term(db, "也许", source="auto", status="candidate")
    lexicon.touch(db, lexicon_ids=[usr, usr])
    lex = Lexicon.load(db)
    assert lex.hotwords(50) == ["王敏", "压测", "星河", "埋点"]                 # speaker, confirmed (most hit first), glossary
    assert lex.hotwords(2) == ["王敏", "压测"] and lex.hotwords(0) == []
    assert lex.hotwords(50, context="刚才在说埋点的事")[0] == "埋点"           # what the context is about comes first
    assert off and cand and lex.by_term["王敏"].speaker_id == sid


def test_prompt_rows_expand_and_jieba_words(db):
    star = lexicon.upsert_term(db, "星河", kind="project", note="正在做的项目")
    lexicon.add_alias(db, star, "兴和")
    lexicon.add_alias(db, star, "银河计划", kind="aka")
    lexicon.upsert_term(db, "王敏", kind="person")
    lex = Lexicon.load(db)
    assert Lexicon.prompt_rows(lex.active) == "星河｜正在做的项目｜兴和\n王敏｜人名｜"
    assert lex.expand("银河计划什么时候上线") == ["星河", "兴和"]
    assert lex.expand("星河呢") == ["兴和", "银河计划"] and lex.expand("别的") == []
    assert lexicon.jieba_words(db) == ["星河", "王敏", "银河计划"]


def test_list_terms_filters_and_counts(db):
    a = lexicon.upsert_term(db, "星河", kind="project", note="项目")
    lexicon.add_alias(db, a, "兴和")
    lexicon.upsert_term(db, "王敏", kind="person")
    off = lexicon.upsert_term(db, "旧词")
    lexicon.update_term(db, off, status="disabled")
    names = lambda **kw: [t["term"] for t in lexicon.list_terms(db, **kw)]      # noqa: E731
    assert sorted(names()) == ["星河", "王敏"] and sorted(names(status="all")) == ["旧词", "星河", "王敏"]
    assert names(status="disabled") == ["旧词"] and names(kind="person") == ["王敏"]
    assert names(q="兴和") == ["星河"] and names(q="项目") == ["星河"] and names(q="没有") == []
    if pinyin.available():
        assert names(q="xing") == ["星河"]
    assert lexicon.counts(db) == {"active": 2, "candidate": 0, "disabled": 1, "aliases": 1, "by_kind": {"project": 1, "person": 1}}


def test_update_and_delete(db, cfg):
    a = lexicon.upsert_term(db, "星河")
    b = lexicon.upsert_term(db, "压测")
    aid = lexicon.add_alias(db, a, "兴和")
    t = lexicon.update_term(db, a, term="星河计划", kind="project", note="  说明  ", hotword=False, pinyin_key="Xing  He ji hua")
    assert (t["term"], t["kind"], t["note"], t["hotword"], t["pinyin"]) == ("星河计划", "project", "说明", False, "xing he ji hua")
    assert [x["alias"] for x in t["aliases"]] == ["兴和"]                       # aliases survive a rename
    with pytest.raises(ValueError, match="已经有"):
        lexicon.update_term(db, a, term="压测")
    with pytest.raises(LookupError):
        lexicon.update_term(db, 999, note="x")
    with pytest.raises(ValueError):
        lexicon.update_term(db, b, status="gone")
    assert lexicon.delete_alias(db, aid) is True and lexicon.delete_alias(db, aid) is False
    assert lexicon.delete_term(db, cfg, b) == {"deleted": True, "disabled": False} and lexicon.get_term(db, b) is None
    with pytest.raises(LookupError):
        lexicon.delete_term(db, cfg, b)


# ---------------------------------------------------------------------------------------------- glossary.txt / speakers
def test_sync_sources_imports_glossary_and_speakers_idempotently(db, cfg):
    cfg.glossary_path.write_text("# 词表\n星河 # 项目\n压测\n星河\n", encoding="utf-8")
    sid = db.upsert_speaker("王敏")
    me = db.upsert_speaker("我")
    assert lexicon.sync_sources(db, cfg) == {"glossary_added": 2, "speakers_added": 2, "speakers_renamed": 0}
    assert lexicon.sync_sources(db, cfg) == {"glossary_added": 0, "speakers_added": 0, "speakers_renamed": 0}
    by = {t["term"]: t for t in lexicon.list_terms(db)}
    assert set(by) == {"星河", "压测", "王敏", "我"}
    assert (by["星河"]["source"], by["星河"]["kind"]) == ("glossary", "term")
    assert (by["王敏"]["source"], by["王敏"]["kind"], by["王敏"]["speaker_id"], by["王敏"]["hotword"]) == ("speaker", "person", sid, True)
    assert by["我"]["hotword"] is False and by["我"]["speaker_id"] == me        # a one-character name is no hot word
    # a renamed speaker renames its entry; a new line in the file is picked up; the rest is untouched
    db.execute("UPDATE speakers SET name='王敏敏' WHERE id=?", (sid,))
    cfg.glossary_path.write_text("星河\n压测\n埋点\n", encoding="utf-8")
    assert lexicon.sync_sources(db, cfg) == {"glossary_added": 1, "speakers_added": 0, "speakers_renamed": 1}
    assert lexicon.get_term(db, by["王敏"]["id"])["term"] == "王敏敏"
    # a glossary term that is also a speaker's name: one entry, linked to the speaker
    s2 = db.upsert_speaker("埋点")
    lexicon.sync_sources(db, cfg)
    t = [t for t in lexicon.list_terms(db) if t["term"] == "埋点"]
    assert len(t) == 1 and t[0]["speaker_id"] == s2 and t[0]["kind"] == "person" and t[0]["source"] == "glossary"


def test_sync_respects_tombstones_and_delete_makes_them_where_needed(db, cfg):
    cfg.glossary_path.write_text("星河\n压测\n", encoding="utf-8")
    sid = db.upsert_speaker("王敏")
    lexicon.sync_sources(db, cfg)
    by = {t["term"]: t["id"] for t in lexicon.list_terms(db)}
    # an entry the glossary file / the speakers table would bring back is switched off, not removed
    assert lexicon.delete_term(db, cfg, by["星河"]) == {"deleted": False, "disabled": True}
    assert lexicon.delete_term(db, cfg, by["王敏"]) == {"deleted": False, "disabled": True}
    cfg.glossary_path.write_text("星河\n压测\n埋点\n", encoding="utf-8")       # the file changes: re-imported
    assert lexicon.sync_sources(db, cfg)["glossary_added"] == 1
    states = {t["term"]: t["status"] for t in lexicon.list_terms(db, status="all")}
    assert states == {"星河": "disabled", "压测": "active", "王敏": "disabled", "埋点": "active"}
    # the explicit path brings one back
    lexicon.upsert_term(db, "星河")
    assert lexicon.get_term(db, by["星河"])["status"] == "active"
    # an unreadable file is skipped (and retried next time), speakers still sync
    cfg.glossary_path.write_bytes(bytes([0xFF, 0xFE, 0x00, 0x41]))
    db.upsert_speaker("李四")
    assert lexicon.sync_sources(db, cfg) == {"glossary_added": 0, "speakers_added": 1, "speakers_renamed": 0}
    assert sid


def test_sync_throttle_and_never_raises(db, cfg, monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(lexicon, "sync_sources", lambda d, c: calls.append(1) or (_ for _ in ()).throw(RuntimeError("boom")))
    lexicon._last_sync.clear()
    lexicon.sync_sources_throttled(db, cfg)
    lexicon.sync_sources_throttled(db, cfg)
    assert calls == [1]                                                          # second call inside the window: skipped
    lexicon.sync_sources_throttled(db, cfg, every_s=0)
    assert calls == [1, 1]


def test_import_replace_export_and_glossary_text_round_trip(db, cfg):
    db.upsert_speaker("王敏")
    cfg.glossary_path.write_text("# 我的词表\n星河 # 项目\n\n压测\n", encoding="utf-8")
    lexicon.sync_sources(db, cfg)
    learned = lexicon.upsert_term(db, "埋点", source="answer")
    assert lexicon.export_terms(db) == ["星河", "压测", "埋点"] and lexicon.export_glossary(db) == "星河\n压测\n埋点\n"
    assert lexicon.export_terms(db, include_speakers=True) == ["星河", "压测", "王敏", "埋点"]
    text = lexicon.glossary_text(db, cfg)
    assert text == "# 我的词表\n星河 # 项目\n\n压测\n埋点\n"                    # the file as written + what was learned
    # saving the text back changes nothing
    assert lexicon.import_terms(db, glossary_file.parse_glossary(text), replace=True) == {"added": 0, "enabled": 0, "disabled": 0, "total": 3}
    # dropping a line switches that entry off; speakers are never touched; a new line is added
    res = lexicon.import_terms(db, ["星河", "埋点", "灰度"], replace=True)
    assert res == {"added": 1, "enabled": 0, "disabled": 1, "total": 3}
    states = {t["term"]: t["status"] for t in lexicon.list_terms(db, status="all")}
    assert states == {"星河": "active", "压测": "disabled", "王敏": "active", "埋点": "active", "灰度": "active"}
    # the export no longer shows the switched-off line of the file
    assert lexicon.glossary_text(db, cfg) == "# 我的词表\n星河 # 项目\n\n埋点\n灰度\n"
    # without replace an import only adds / re-enables
    assert lexicon.import_terms(db, ["压测", "", "，"]) == {"added": 0, "enabled": 1, "disabled": 0, "total": 1}
    assert learned and Lexicon.load(db).by_term["压测"].status == "active"
    assert lexicon.export_glossary(DB(cfg.data_dir / "empty.sqlite3")) == ""


# ---------------------------------------------------------------------------------------------- history
def test_apply_alias_to_history_dry_run_limit_and_revisions(db, cfg):
    ids = seed(db, cfg, ["兴和的排期定了", "这句没有", "兴和和兴和路都提到了", "又说兴和"])
    star = lexicon.upsert_term(db, "星河")
    lexicon.upsert_term(db, "兴和路")
    aid = lexicon.add_alias(db, star, "兴和")
    dry = lexicon.apply_alias_to_history(db, cfg, aid, dry_run=True)
    assert (dry["total"], dry["applied"], dry["dry_run"], dry["alias"], dry["term"]) == (3, 0, True, "兴和", "星河")
    assert [m["utterance_id"] for m in dry["matches"]] == [ids[3], ids[2], ids[0]]            # newest first
    assert dry["matches"][1] == {"utterance_id": ids[2], "day": DAY, "time": "09:02:00", "before": "兴和和兴和路都提到了",
                                 "after": "星河和兴和路都提到了"}
    assert texts_of(db)[0] == "兴和的排期定了" and lexicon.count_alias_matches(db, aid) == 3
    assert lexicon.count_alias_matches(db, aid, exclude_utterance_id=ids[0]) == 2
    one = lexicon.apply_alias_to_history(db, cfg, aid, limit=1)
    assert (one["applied"], one["changed"], one["total"]) == (1, [ids[3]], 3)
    rest = lexicon.apply_alias_to_history(db, cfg, aid, source="answer", detail={"inbox_id": 7})
    assert rest["applied"] == 2 and texts_of(db) == ["星河的排期定了", "这句没有", "星河和兴和路都提到了", "又说星河"]
    h = revisions.history(db, ids[0])
    assert len(h) == 1 and h[0]["source"] == "answer" and h[0]["detail"]["inbox_id"] == 7
    assert h[0]["detail"]["edits"] == [{"from": "兴和", "to": "星河", "reason": "词库", "alias_id": aid, "lexicon_id": star,
                                        "start": 0, "end": 2}]
    assert db.one("SELECT text_source FROM utterances WHERE id=?", (ids[0],))["text_source"] == "answer"
    assert lexicon.apply_alias_to_history(db, cfg, aid)["total"] == 0                          # idempotent
    t = lexicon.get_term(db, star)
    assert t["hits"] == 3 and t["aliases"][0]["hits"] == 3
    assert (cfg.transcripts_dir / f"{DAY}.md").read_text(encoding="utf-8").count("星河") == 3   # one rewrite at the end
    with pytest.raises(LookupError):
        lexicon.apply_alias_to_history(db, cfg, 999)
    aka = lexicon.add_alias(db, star, "银河计划", kind="aka")
    with pytest.raises(ValueError):
        lexicon.apply_alias_to_history(db, cfg, aka)


def test_history_passes_restrict_by_day_and_ids_and_match_latin_in_any_case(db, cfg):
    a = seed(db, cfg, ["cdn要续费", "CDN的报价"], day="2026-11-02")
    b = seed(db, cfg, ["Cdn明年再说", "100%_cdn"], day="2026-11-03")
    tid = lexicon.upsert_term(db, "内容分发")
    aid = lexicon.add_alias(db, tid, "cdn", scope="auto")
    assert lexicon.apply_alias_to_history(db, cfg, aid, dry_run=True)["total"] == 4
    assert lexicon.apply_alias_to_history(db, cfg, aid, dry_run=True, day="2026-11-03")["total"] == 2
    assert lexicon.apply_alias_to_history(db, cfg, aid, dry_run=True, utterance_ids=[a[0], b[0]])["match_ids"] == [b[0], a[0]]
    assert lexicon.apply_alias_to_history(db, cfg, aid, dry_run=True, utterance_ids=[])["total"] == 0


def test_recorrect_runs_the_first_pass_over_history_and_skips_undone_rewrites(db, cfg):
    ids = seed(db, cfg, ["兴和的排期", "新河边上", "兴和又延期", "没事"])
    seed(db, cfg, ["兴和第二天"], day="2026-11-03")
    star = lexicon.upsert_term(db, "星河")
    auto = lexicon.add_alias(db, star, "兴和", cfg=cfg)
    lexicon.add_alias(db, star, "新河", cfg=cfg)                                 # hint: never applied by a history pass
    assert lexicon.recorrect(db, cfg, dry_run=True) == {"scanned": 5, "changed": 3, "edits": 3, "dry_run": True}
    assert lexicon.recorrect(db, cfg, day=DAY) == {"scanned": 4, "changed": 2, "edits": 2, "dry_run": False}
    assert texts_of(db) == ["星河的排期", "新河边上", "星河又延期", "没事", "兴和第二天"]
    assert revisions.history(db, ids[0])[0]["source"] == "lexicon"
    # the user takes one back: later passes leave that sentence alone
    revisions.undo(db, cfg, ids[0])
    assert revisions.undone_aliases(db, ids) == {ids[0]: {auto}}
    assert lexicon.recorrect(db, cfg)["changed"] == 1 and texts_of(db)[0] == "兴和的排期" and texts_of(db)[4] == "星河第二天"
    assert lexicon.apply_alias_to_history(db, cfg, auto, dry_run=True)["total"] == 0
    assert lexicon.recorrect(db, cfg, lexicon_id=999)["changed"] == 0
    assert lexicon.recorrect(DB(cfg.data_dir / "empty.sqlite3"), cfg) == {"scanned": 0, "changed": 0, "edits": 0, "dry_run": False}


def test_job_recorrect_is_the_runtime_job_target(db, cfg):
    from audio_note import runtime

    assert runtime.JOB_TARGETS["recorrect"] == "audio_note.speech.lexicon:job_recorrect" and "recorrect" in runtime.JOBS
    seed(db, cfg, ["兴和的排期"])
    cfg.glossary_path.write_text("压测\n", encoding="utf-8")
    star = lexicon.upsert_term(db, "星河")
    lexicon.add_alias(db, star, "兴和")
    rt = type("RT", (), {"db": db, "cfg": cfg, "events": None})()
    assert lexicon.job_recorrect(rt, dry_run=True)["changed"] == 1 and texts_of(db) == ["兴和的排期"]
    assert lexicon.job_recorrect(rt) == {"scanned": 1, "changed": 1, "edits": 1, "dry_run": False}
    assert "压测" in Lexicon.load(db).by_term                                    # the job syncs the sources first


# ---------------------------------------------------------------------------------------------- the old loaders
def test_the_old_glossary_loader_is_a_wrapper_of_the_shared_parser(tmp_path):
    p = tmp_path / "g.txt"
    p.write_text("# 表\n灰度发布 # 上线方式\nC#\n灰度发布\n", encoding="utf-8-sig")
    assert correct.load_glossary(p) == glossary_file.read_glossary(p) == ["灰度发布", "C#"]
    assert correct.load_glossary(None) == [] and correct.load_glossary(tmp_path / "missing.txt") == []


def test_events_are_published_on_writes(db, cfg):
    from audio_note.events import EventBus

    bus = EventBus()
    sub = bus.subscribe(since=0)
    tid = lexicon.upsert_term(db, "星河", events=bus)
    aid = lexicon.add_alias(db, tid, "兴和", events=bus)
    lexicon.update_term(db, tid, note="x", events=bus)
    lexicon.delete_alias(db, aid, events=bus)
    lexicon.delete_term(db, cfg, tid, events=bus)
    lexicon.import_terms(db, ["压测"], events=bus)
    seen = []
    while (ev := sub.get(timeout=0.01)) is not None:
        seen.append((ev["type"], ev["change"]))
    assert seen == [("lexicon", "term"), ("lexicon", "alias"), ("lexicon", "term"), ("lexicon", "alias"), ("lexicon", "term"),
                    ("lexicon", "import")]


def test_a_new_entry_becomes_one_search_token_in_the_sentences_that_contain_it(db, cfg):
    from audio_note.memory import index

    a, b = seed(db, cfg, ["岚禾的排期定了", "别的话"])
    with db.transaction() as con:                                               # as the index stood before the word was known
        con.execute("UPDATE utterance_tokens SET text_seg=? WHERE utterance_id=?", (index.tokenize("岚禾的排期定了"), a))
    fts = lambda w: [int(r[0]) for r in db.query("SELECT rowid FROM utterances_fts WHERE utterances_fts MATCH ?", (f'"{w}"',))]  # noqa: E731
    before = db.one("SELECT text_seg FROM utterance_tokens WHERE utterance_id=?", (a,))["text_seg"]
    lexicon.upsert_term(db, "岚禾", kind="project")
    after = db.one("SELECT text_seg FROM utterance_tokens WHERE utterance_id=?", (a,))["text_seg"]
    assert "岚禾" in after.split() and fts("岚禾") == [a]
    assert before != after or "岚禾" in before.split()
    assert lexicon.reindex_spelling(db, "没有出现过的词") == 0


def test_a_database_recreated_under_the_same_path_never_reuses_a_snapshot(cfg, tmp_path):
    path = tmp_path / "again.sqlite3"
    one = DB(path)
    tid = lexicon.upsert_term(one, "星河")
    lexicon.add_alias(one, tid, "兴和")
    assert [t.term for t in Lexicon.load(one).active] == ["星河"]
    one.close_all()
    for ext in ("", "-wal", "-shm"):
        p = tmp_path / f"again.sqlite3{ext}"
        if p.exists():
            p.unlink()
    two = DB(path)                                                              # same path, another database
    assert Lexicon.load(two).terms == []
    t2 = lexicon.upsert_term(two, "压测")
    lexicon.add_alias(two, t2, "牙测")                                           # the same number of writes as before
    assert [t.term for t in Lexicon.load(two).active] == ["压测"]
    two.close_all()


def test_rows_written_with_plain_sql_are_noticed(db):
    assert Lexicon.load(db).terms == []
    tid = db.execute("INSERT INTO lexicon(term, kind) VALUES ('周岚', 'person')").lastrowid      # another module, not this API
    assert [t.term for t in Lexicon.load(db).active] == ["周岚"]
    db.execute("INSERT INTO lexicon_aliases(lexicon_id, alias, kind) VALUES (?, '岚姐', 'aka')", (tid,))
    assert Lexicon.load(db).resolve("岚姐").term == "周岚"
    db.execute("DELETE FROM lexicon_aliases")
    assert Lexicon.load(db).resolve("岚姐") is None


def test_word_frequency_ignores_what_was_registered_with_the_tokenizer(db):
    import jieba

    assert lexicon.word_frequency("新和") == 0
    jieba.add_word("新和")                                                       # what registering a spelling does to the live table
    assert (jieba.dt.FREQ.get("新和") or 0) > lexicon.DEFAULT_ALIAS_AUTO_MAX_FREQ
    assert lexicon.word_frequency("新和") == 0
    tid = lexicon.upsert_term(db, "星河")
    assert lexicon.auto_scope(db, tid, "新和") == ("auto", "ok")
    assert lexicon.word_frequency("沉默") > 2000 and lexicon.word_frequency("") == 0


def test_an_alias_that_would_change_a_number_is_never_automatic(db, cfg):
    t30 = lexicon.upsert_term(db, "朗月T30", kind="product")
    assert lexicon.auto_scope(db, t30, "朗月T三十", cfg) == ("hint", "number")
    assert lexicon.auto_scope(db, t30, "朗悦T30", cfg) == ("auto", "ok")           # the digits stay as they are
    price = lexicon.upsert_term(db, "三千二")
    aid = lexicon.add_alias(db, price, "三千五", cfg=cfg)
    assert lexicon.get_alias(db, aid)["scope"] == "hint" and Lexicon.load(db).apply("目标三千五")[1] == []
