"""HTTP and CLI surface of the text features (SPEC v1 section 8, rows marked B2): editing a sentence, its
revisions, the inbox, the lexicon, the glossary compatibility endpoints, marks in the transcript.

No network, no microphone, no models. Names and sentences are invented.
"""
from __future__ import annotations

import io
import json
import logging
import sys
import types
from datetime import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from rich.console import Console

from audio_note import cli, inbox
from audio_note.runtime import Runtime
from audio_note.speech import lexicon, pinyin, revisions
from audio_note.ui.app import create_app

DAY = "2026-11-02"
needs_pypinyin = pytest.mark.skipif(not pinyin.available(), reason="pypinyin is not installed")


@pytest.fixture
def rt(cfg_factory) -> Runtime:
    r = Runtime(cfg_factory({"memory": {"retrieval": "fts", "embedding_model": "hash"}}))
    yield r
    r.shutdown()
    from audio_note.logging_setup import _remove_own_handlers

    _remove_own_handlers(logging.getLogger())


@pytest.fixture
def client(rt: Runtime) -> TestClient:
    with TestClient(create_app(rt), raise_server_exceptions=False) as c:
        yield c


def seed(rt: Any, texts: list[str], day: str = DAY, raw: list[str] | None = None) -> list[int]:
    db, cfg = rt.db, rt.cfg
    seg = db.add_segment(cfg.audio_dir / f"{day}-{len(texts)}-{texts[0][:2]}.wav", datetime.fromisoformat(f"{day}T09:00:00"), 60.0,
                         speech_s=30.0)
    db.set_segment_status(seg, "done")
    return db.add_utterances([{
        "segment_id": seg, "start_s": float(i), "end_s": float(i + 1), "abs_start": f"{day}T09:{i:02d}:00", "day": day,
        "cluster_label": 0, "speaker_id": None, "speaker_name": None, "speaker_score": None,
        "text_raw": (raw[i] if raw else t), "text": t, "text_seg": " ".join(t)} for i, t in enumerate(texts)])


def texts_of(rt: Any) -> list[str]:
    return [r["text"] for r in rt.db.query("SELECT text FROM utterances ORDER BY id")]


def events_of(sub: Any) -> list[str]:
    out = []
    while (ev := sub.get(timeout=0.01)) is not None:
        out.append(ev["type"])
    return out


def read_only(rt: Runtime, monkeypatch: pytest.MonkeyPatch, pid: int = 424242) -> None:
    """Make this process "read-only": the service lock belongs to another (fake) live process."""
    from audio_note import service

    service.lock_path(rt.cfg).write_text(f"{pid}\n2026-10-31T09:00:00\n1.0\n", encoding="utf-8")
    monkeypatch.setattr(service, "_same_process", lambda p, ctime: p == pid)


# ---------------------------------------------------------------------------------------------- routes
def test_the_b2_rows_of_the_api_table_are_served(client: TestClient):
    try:
        from fastapi.routing import iter_route_contexts

        routes = [c.route for c in iter_route_contexts(client.app.routes)]
    except ImportError:
        routes = list(client.app.routes)
    have = {(m, getattr(r, "path", "")) for r in routes for m in (getattr(r, "methods", None) or ())}
    for method, path in [
        ("PATCH", "/api/utterances/{uid}"), ("GET", "/api/utterances/{uid}/revisions"), ("POST", "/api/utterances/{uid}/undo"),
        ("GET", "/api/inbox"), ("POST", "/api/inbox/{item_id}/answer"), ("POST", "/api/inbox/{item_id}/dismiss"),
        ("POST", "/api/inbox/{item_id}/snooze"), ("POST", "/api/inbox/{item_id}/reopen"),
        ("GET", "/api/lexicon"), ("POST", "/api/lexicon"), ("PATCH", "/api/lexicon/{term_id}"), ("DELETE", "/api/lexicon/{term_id}"),
        ("POST", "/api/lexicon/{term_id}/aliases"), ("DELETE", "/api/lexicon/aliases/{alias_id}"),
        ("POST", "/api/lexicon/aliases/{alias_id}/apply"), ("GET", "/api/glossary"), ("PUT", "/api/glossary"),
        ("GET", "/api/transcript/{day}"),
    ]:
        assert (method, path) in have, (method, path)


# ---------------------------------------------------------------------------------------------- editing a sentence
def test_patch_utterance_text_revisions_and_undo(client: TestClient, rt: Runtime):
    a, b = seed(rt, ["今天讨论了预算问题", "明天去出差"])
    sub = rt.events.subscribe()
    r = client.patch(f"/api/utterances/{a}", json={"text": " 今天讨论了排期问题 "})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["ok"] is True and out["changed"] is True and out["revision_id"] and out["learned"] is None
    u = out["utterance"]
    assert (u["id"], u["text"], u["text_raw"], u["text_source"], u["audio"]) == (a, "今天讨论了排期问题", "今天讨论了预算问题", "user", False)
    assert u["marks"] == [{"start": 5, "end": 7, "kind": "fixed", "from": "预算", "source": "user"}]
    assert "utterance" in events_of(sub)
    assert "今天讨论了排期问题" in (rt.cfg.transcripts_dir / f"{DAY}.md").read_text(encoding="utf-8")
    # the same text again: nothing happens
    again = client.patch(f"/api/utterances/{a}", json={"text": "今天讨论了排期问题"}).json()
    assert again["changed"] is False and again["revision_id"] is None
    # history
    h = client.get(f"/api/utterances/{a}/revisions").json()
    assert (h["id"], h["text"], h["text_raw"], h["text_source"]) == (a, "今天讨论了排期问题", "今天讨论了预算问题", "user")
    assert [(x["source"], x["old_text"], x["new_text"]) for x in h["revisions"]] == [("user", "今天讨论了预算问题", "今天讨论了排期问题")]
    assert h["revisions"][0]["detail"] == {"edits": [{"from": "预算", "to": "排期", "start": 5, "end": 7}]}
    assert client.get(f"/api/utterances/{b}/revisions").json()["revisions"] == []
    # undo
    r = client.post(f"/api/utterances/{a}/undo", json={})
    assert r.status_code == 200 and r.json()["undone"] == out["revision_id"]
    assert r.json()["utterance"]["text"] == "今天讨论了预算问题" and r.json()["utterance"]["marks"] == []
    assert r.json()["utterance"]["text_source"] is None
    r = client.post(f"/api/utterances/{a}/undo")
    assert r.status_code == 400 and r.json() == {"error": "这句话没有可以撤销的修改"}
    # errors
    assert client.patch(f"/api/utterances/{a}", json={"text": 5}).json() == {"error": "text 应为文字"}
    assert client.patch(f"/api/utterances/{a}", json={"text": "   "}).json() == {"error": "文字不能为空"}
    assert client.patch(f"/api/utterances/{a}", json={"text": "x", "remember": "maybe"}).status_code == 400
    assert client.patch("/api/utterances/999", json={"text": "x"}).status_code == 404
    assert client.get("/api/utterances/999/revisions").status_code == 404
    assert client.post("/api/utterances/999/undo").status_code == 404
    assert client.post(f"/api/utterances/{a}/undo", json={"revision_id": "x"}).status_code == 400
    assert client.post(f"/api/utterances/{a}/undo", json={"revision_id": 999}).status_code == 404
    assert texts_of(rt) == ["今天讨论了预算问题", "明天去出差"]


@needs_pypinyin
def test_a_hand_edit_teaches_the_lexicon_and_offers_the_other_places(client: TestClient, rt: Runtime):
    ids = seed(rt, ["明天和兰姐开会", "兰姐说可以", "无关的话"])
    out = client.patch(f"/api/utterances/{ids[0]}", json={"text": "明天和岚姐开会"}).json()
    learned = out["learned"]
    assert (learned["term"], learned["alias"], learned["scope"], learned["other_matches"], learned["created_term"]) == (
        "岚姐", "兰姐", "auto", 1, True)
    assert out["utterance"]["marks"] == [{"start": 3, "end": 4, "kind": "fixed", "from": "兰", "source": "user"}]
    # the waiting question, and the direct way: apply the alias to the history
    box = client.get("/api/inbox").json()
    assert box["counts"]["open"] == 1 and box["items"][0]["id"] == learned["inbox_id"] and box["items"][0]["kind"] == "rule"
    assert box["items"][0]["utterance"]["id"] == ids[0]
    dry = client.post(f"/api/lexicon/aliases/{learned['alias_id']}/apply", json={"dry_run": True}).json()
    assert (dry["total"], dry["applied"], dry["dry_run"]) == (1, 0, True) and dry["matches"][0]["after"] == "岚姐说可以"
    assert "match_ids" not in dry and texts_of(rt)[1] == "兰姐说可以"
    done = client.post(f"/api/lexicon/aliases/{learned['alias_id']}/apply", json={}).json()
    assert done["applied"] == 1 and done["changed"] == [ids[1]] and texts_of(rt)[1] == "岚姐说可以"
    assert client.get("/api/inbox").json()["counts"]["open"] == 0                # the question became moot
    # "remember" makes a rule out of an edit the server would not have learned from
    (c,) = seed(rt, ["我们明天在见"], day="2026-11-03")
    assert client.patch(f"/api/utterances/{c}", json={"text": "我们明天再见"}).json()["learned"] is None
    client.post(f"/api/utterances/{c}/undo")
    got = client.patch(f"/api/utterances/{c}", json={"text": "我们明天再见", "remember": True}).json()["learned"]
    assert (got["term"], got["alias"]) == ("再见", "在见")


def test_typing_an_automatic_fix_away_counts_as_undoing_it(client: TestClient, rt: Runtime):
    (a,) = seed(rt, ["兴和的排期"])
    tid = lexicon.upsert_term(rt.db, "星河")
    aid = lexicon.add_alias(rt.db, tid, "兴和")
    assert lexicon.recorrect(rt.db, rt.cfg)["changed"] == 1
    out = client.patch(f"/api/utterances/{a}", json={"text": "兴和的排期"}).json()
    assert out["changed"] is True and out["learned"] is None and out["utterance"]["text"] == "兴和的排期"
    assert [h["source"] for h in revisions.history(rt.db, a)] == ["undo", "lexicon"]
    assert revisions.undone_aliases(rt.db, [a]) == {a: {aid}}
    assert lexicon.recorrect(rt.db, rt.cfg)["changed"] == 0                      # the rule leaves this sentence alone now


def test_text_changes_are_refused_in_a_read_only_process(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    (a,) = seed(rt, ["兴和的排期"])
    q = inbox.add(rt.db, "term", title="？", payload={"span": "兴和", "utterance_id": a, "candidates": ["星河"]}, cfg=rt.cfg)
    tid = lexicon.upsert_term(rt.db, "星河")
    aid = lexicon.add_alias(rt.db, tid, "兴和")
    read_only(rt, monkeypatch)
    for method, url, body in [("PATCH", f"/api/utterances/{a}", {"text": "改"}), ("POST", f"/api/utterances/{a}/undo", {}),
                              ("POST", f"/api/inbox/{q}/answer", {"choice": 0}), ("POST", f"/api/inbox/{q}/reopen", {}),
                              ("POST", f"/api/lexicon/aliases/{aid}/apply", {})]:
        r = client.request(method, url, json=body)
        assert r.status_code == 409 and "error" in r.json(), (method, url)
    assert texts_of(rt) == ["兴和的排期"] and inbox.get(rt.db, q)["status"] == "open"
    # reading and a dry run still work
    assert client.get("/api/inbox").json()["counts"]["open"] == 1
    assert client.post(f"/api/lexicon/aliases/{aid}/apply", json={"dry_run": True}).json()["total"] == 1


# ---------------------------------------------------------------------------------------------- inbox
@needs_pypinyin
def test_inbox_endpoints(client: TestClient, rt: Runtime):
    ids = seed(rt, ["今天过一下兴和的排期", "兴和要上线了", "牙测报告", "三千五的目标"])
    q = inbox.add(rt.db, "term", title="「兴和」是哪个词？", cfg=rt.cfg, priority=3, payload={
        "span": "兴和", "sentence": "今天过一下兴和的排期", "utterance_id": ids[0], "start": 5, "end": 7, "candidates": ["星河"],
        "term_kind": "name"})
    q2 = inbox.add(rt.db, "term", title="「牙测」是哪个词？", cfg=rt.cfg, payload={
        "span": "牙测", "utterance_id": ids[2], "candidates": ["压测"], "term_kind": "term"})
    q3 = inbox.add(rt.db, "term", title="「三千五」对吗？", cfg=rt.cfg, payload={
        "span": "三千五", "utterance_id": ids[3], "candidates": ["三千二"], "term_kind": "number"})
    box = client.get("/api/inbox").json()
    assert box["counts"] == {"open": 3, "by_kind": {"term": 3}, "queued": 0, "snoozed": 0}
    assert [it["id"] for it in box["items"]][0] == q and box["items"][0]["payload"]["candidates"] == ["星河"]
    first = box["items"][0]["utterance"]
    assert first["id"] == ids[0] and first["marks"] == [{"start": 5, "end": 7, "kind": "uncertain", "inbox_id": q, "candidates": ["星河"]}]
    assert client.get("/api/inbox", params={"kind": "voice"}).json()["items"] == []
    assert len(client.get("/api/inbox", params={"limit": 1, "offset": 1}).json()["items"]) == 1
    assert client.get("/api/inbox", params={"status": "nope"}).status_code == 400
    assert rt.status()["inbox_open"] == 3

    sub = rt.events.subscribe()
    r = client.post(f"/api/inbox/{q}/answer", json={"choice": 0})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["ok"] is True and out["counts"]["open"] == 2 and out["item"]["status"] == "answered"
    assert (out["result"]["term"], out["result"]["alias"], out["result"]["scope"]) == ("星河", "兴和", "auto")
    assert sorted(out["result"]["changed"]) == ids[:2] and texts_of(rt)[:2] == ["今天过一下星河的排期", "星河要上线了"]
    assert out["item"]["utterance"]["marks"] == [{"start": 5, "end": 7, "kind": "fixed", "from": "兴和", "source": "answer"}]
    seen = events_of(sub)
    assert {"inbox", "utterance", "lexicon"} <= set(seen)
    # the answer can be taken back as a whole
    back = client.post(f"/api/inbox/{q}/reopen").json()
    assert back["reverted"] == 2 and back["item"]["status"] == "open" and back["counts"]["open"] == 3
    assert texts_of(rt)[:2] == ["今天过一下兴和的排期", "兴和要上线了"] and client.get("/api/lexicon").json()["terms"] == []
    # typed answer, only this sentence
    out = client.post(f"/api/inbox/{q}/answer", json={"text": "星河", "apply_history": False}).json()
    assert out["result"]["changed"] == [ids[0]] and texts_of(rt)[1] == "兴和要上线了"
    # "it is right" / dismiss / snooze
    out = client.post(f"/api/inbox/{q2}/answer", json={"as_correct": True}).json()
    assert out["result"]["as_correct"] is True and out["item"]["payload"]["never"] is True
    out = client.post(f"/api/inbox/{q3}/snooze", json={"days": 3}).json()
    assert out["item"]["status"] == "snoozed" and out["counts"] == {"open": 0, "by_kind": {}, "queued": 0, "snoozed": 1}
    out = client.post(f"/api/inbox/{q3}/dismiss", json={"never": True}).json()
    assert out["item"]["status"] == "dismissed" and out["item"]["payload"]["never"] is True and out["counts"]["snoozed"] == 0
    assert [it["id"] for it in client.get("/api/inbox", params={"status": "all"}).json()["items"]] and rt.status()["inbox_open"] == 0
    # errors
    assert client.post(f"/api/inbox/{q3}/dismiss").json() == {"error": "这个问题已经处理过了"}
    assert client.post("/api/inbox/999/answer", json={"choice": 0}).status_code == 404
    assert client.post("/api/inbox/999/dismiss").status_code == 404 and client.post("/api/inbox/999/snooze").status_code == 404
    assert client.post("/api/inbox/999/reopen").status_code == 404
    for body in ({"choice": [1]}, {"text": 5}, {"as_correct": "perhaps"}, {"apply_history": "x"}):
        assert client.post(f"/api/inbox/{q}/answer", json=body).status_code == 400
    v = inbox.add(rt.db, "voice", title="是谁？", payload={})
    r = client.post(f"/api/inbox/{v}/answer", json={"text": "王敏"})
    if "voice" not in inbox._HANDLERS:                                           # the speaker package registers it when present
        assert r.status_code == 400 and r.json() == {"error": "这类问题现在还不能在这里回答"}
    assert client.post(f"/api/inbox/{v}/snooze", json={"days": 0}).status_code == 400


# ---------------------------------------------------------------------------------------------- lexicon
def test_lexicon_crud(client: TestClient, rt: Runtime):
    rt.cfg.glossary_path.write_text("灰度发布\n", encoding="utf-8")
    sid = rt.db.upsert_speaker("王敏")
    first = client.get("/api/lexicon").json()                                    # glossary + speakers are there on the first visit
    assert sorted(t["term"] for t in first["terms"]) == ["灰度发布", "王敏"] and first["counts"]["active"] == 2
    sub = rt.events.subscribe()
    r = client.post("/api/lexicon", json={"term": "星河", "kind": "project", "note": "正在做的项目",
                                         "aliases": ["兴和", {"alias": "银河计划", "kind": "aka"}, {"alias": "新河", "scope": "auto"}]})
    assert r.status_code == 200, r.text
    t = r.json()["term"]
    assert r.json()["created"] is True and (t["term"], t["kind"], t["note"], t["source"], t["hotword"]) == ("星河", "project", "正在做的项目", "user", True)
    assert [(a["alias"], a["kind"], a["scope"]) for a in t["aliases"]] == [("兴和", "mishear", "auto"), ("银河计划", "aka", "hint"),
                                                                           ("新河", "mishear", "auto")]
    assert client.post("/api/lexicon", json={"term": "星河"}).json()["created"] is False
    assert "lexicon" in events_of(sub)
    tid = t["id"]
    # list filters
    assert [x["term"] for x in client.get("/api/lexicon", params={"q": "兴和"}).json()["terms"]] == ["星河"]
    assert [x["term"] for x in client.get("/api/lexicon", params={"kind": "person"}).json()["terms"]] == ["王敏"]
    assert client.get("/api/lexicon", params={"kind": "animal"}).status_code == 400
    assert client.get("/api/lexicon", params={"status": "gone"}).status_code == 400
    # patch
    t = client.patch(f"/api/lexicon/{tid}", json={"note": "改过的说明", "hotword": False, "kind": "product"}).json()["term"]
    assert (t["note"], t["hotword"], t["kind"]) == ("改过的说明", False, "product")
    assert client.patch(f"/api/lexicon/{tid}", json={"term": "灰度发布"}).json() == {"error": "已经有这个词条了"}
    assert client.patch(f"/api/lexicon/{tid}", json={"status": "gone"}).status_code == 400
    assert client.patch("/api/lexicon/999", json={"note": "x"}).status_code == 404
    person = [x for x in first["terms"] if x["term"] == "王敏"][0]
    assert client.patch(f"/api/lexicon/{person['id']}", json={"term": "王敏敏"}).status_code == 400    # rename speakers where they live
    # aliases
    seed(rt, ["星和的排期", "又是星和"])
    r = client.post(f"/api/lexicon/{tid}/aliases", json={"alias": "星和"}).json()
    assert (r["alias"]["alias"], r["alias"]["scope"], r["alias"]["term"], r["matches"]) == ("星和", "auto", "星河", 2)
    assert len(r["term"]["aliases"]) == 4
    assert client.post(f"/api/lexicon/{tid}/aliases", json={"alias": "星河"}).json() == {"error": "别名不能和词条本身一样"}
    assert client.post(f"/api/lexicon/{tid}/aliases", json={"alias": " "}).status_code == 400
    assert client.post(f"/api/lexicon/{tid}/aliases", json={"alias": "某某", "scope": "always"}).status_code == 400
    assert client.post("/api/lexicon/999/aliases", json={"alias": "某某"}).status_code == 404
    assert client.post(f"/api/lexicon/aliases/{r['alias']['id']}/apply", json={"limit": 1}).json()["applied"] == 1
    assert client.post("/api/lexicon/aliases/999/apply").status_code == 404
    assert client.delete(f"/api/lexicon/aliases/{r['alias']['id']}").json() == {"ok": True}
    assert client.delete(f"/api/lexicon/aliases/{r['alias']['id']}").status_code == 404
    # delete: a real removal, and a tombstone for what would come straight back
    assert client.delete(f"/api/lexicon/{tid}").json() == {"ok": True, "deleted": True, "disabled": False}
    assert client.delete(f"/api/lexicon/{person['id']}").json() == {"ok": True, "deleted": False, "disabled": True}
    assert client.delete("/api/lexicon/999").status_code == 404
    assert [x["term"] for x in client.get("/api/lexicon").json()["terms"]] == ["灰度发布"]
    assert [x["term"] for x in client.get("/api/lexicon", params={"status": "disabled"}).json()["terms"]] == ["王敏"]
    assert client.patch(f"/api/lexicon/{person['id']}", json={"status": "active"}).json()["term"]["status"] == "active"
    for body in ({"term": ""}, {"term": 5}, {"term": "词", "aliases": "x"}, {"term": "词", "kind": "animal"},
                 {"term": "词甲", "aliases": [5]}, {"term": "词乙", "hotword": "maybe"}):
        assert client.post("/api/lexicon", json=body).status_code == 400, body
    assert sid


def test_glossary_endpoints_are_the_lexicon_as_text(client: TestClient, rt: Runtime):
    p = rt.cfg.glossary_path
    rt.db.upsert_speaker("王敏")
    assert client.get("/api/glossary").json() == {"path": str(p), "text": ""}    # speaker names are not glossary text
    assert client.put("/api/glossary", json={"text": "# 术语\n星河 # 项目\n\n压测  \n"}).json() == {"ok": True, "terms": 2}
    assert p.read_text(encoding="utf-8") == "# 术语\n星河 # 项目\n\n压测  \n"
    assert sorted(t["term"] for t in client.get("/api/lexicon").json()["terms"]) == ["压测", "星河", "王敏"]
    # what was learned elsewhere shows up in the export, after the user's own lines
    lexicon.upsert_term(rt.db, "岚姐", source="answer")
    text = client.get("/api/glossary").json()["text"]
    assert text == "# 术语\n星河 # 项目\n\n压测  \n岚姐\n"
    # saving it back changes nothing; dropping a line switches that entry off (not the speaker)
    assert client.put("/api/glossary", json={"text": text}).json() == {"ok": True, "terms": 3}
    assert client.put("/api/glossary", json={"text": "# 术语\n星河 # 项目\n岚姐\n埋点\n"}).json()["terms"] == 3
    states = {t["term"]: t["status"] for t in client.get("/api/lexicon", params={"status": "all"}).json()["terms"]}
    assert states == {"星河": "active", "压测": "disabled", "王敏": "active", "岚姐": "active", "埋点": "active"}
    assert client.get("/api/glossary").json()["text"] == "# 术语\n星河 # 项目\n岚姐\n埋点\n"
    # an entry switched off on the lexicon page disappears from the text although the file still has the line
    star = [t for t in client.get("/api/lexicon").json()["terms"] if t["term"] == "星河"][0]
    assert client.delete(f"/api/lexicon/{star['id']}").json() == {"ok": True, "deleted": False, "disabled": True}
    assert client.get("/api/glossary").json()["text"] == "# 术语\n岚姐\n埋点\n"


# ---------------------------------------------------------------------------------------------- transcript
def test_transcript_carries_marks_and_session_meta(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    a, b = seed(rt, ["星河的排期", "兴和要上线"], raw=["兴和的排期", "兴和要上线"])
    q = inbox.add(rt.db, "term", title="「兴和」是哪个词？", cfg=rt.cfg, payload={
        "span": "兴和", "utterance_id": b, "candidates": ["星河"], "term_kind": "name"})
    out = client.get(f"/api/transcript/{DAY}").json()
    utts = [u for s in out["sessions"] for u in s["utterances"]]                 # the existing shape is unchanged
    assert [u["id"] for u in utts] == [a, b] and out["sessions"][0]["index"] == 1
    assert utts[0]["marks"] == [{"start": 0, "end": 2, "kind": "fixed", "from": "兴和", "source": "llm"}]
    assert utts[1]["marks"] == [{"start": 0, "end": 2, "kind": "uncertain", "inbox_id": q, "candidates": ["星河"]}]
    assert isinstance(out["session_meta"], list)
    ctx = client.get(f"/api/context/{b}").json()["utterances"]
    assert [u["marks"] for u in ctx if u["id"] == b] == [utts[1]["marks"]]
    # with the memory layer's sessions: handed through as they are
    fake = types.ModuleType("audio_note.memory.sessions")
    fake.for_day = lambda db, day: [{"id": 3, "day": day, "title": "晨会", "kind": "meeting", "summary": "排期"}]
    monkeypatch.setitem(sys.modules, "audio_note.memory.sessions", fake)
    assert client.get(f"/api/transcript/{DAY}").json()["session_meta"] == [
        {"id": 3, "day": DAY, "title": "晨会", "kind": "meeting", "summary": "排期"}]
    # a module without the function, a failing one, an absent one: an empty list, never an error
    fake.for_day = lambda db, day: (_ for _ in ()).throw(RuntimeError("boom"))
    assert client.get(f"/api/transcript/{DAY}").json()["session_meta"] == []
    del fake.for_day
    assert client.get(f"/api/transcript/{DAY}").json()["session_meta"] == []
    monkeypatch.setitem(sys.modules, "audio_note.memory.sessions", None)         # import raises ImportError
    r = client.get(f"/api/transcript/{DAY}")
    assert r.status_code == 200 and r.json()["session_meta"] == []


# ---------------------------------------------------------------------------------------------- CLI
@pytest.fixture
def out(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    buf = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=buf, width=400, force_terminal=False, color_system=None, highlight=False,
                                                 soft_wrap=True))
    yield buf
    from audio_note.logging_setup import _remove_own_handlers

    _remove_own_handlers(logging.getLogger())


def run_cli(out: io.StringIO, *argv: str) -> tuple[int, str]:
    out.seek(0)
    out.truncate()
    code = cli.main(list(argv))
    return code, out.getvalue()


@needs_pypinyin
def test_cli_lexicon_and_inbox_commands(cfg_factory, out: io.StringIO, tmp_path):
    cfg = cfg_factory({"memory": {"retrieval": "fts", "embedding_model": "hash"}})
    from audio_note.db import DB

    db = DB(cfg.db_path)
    rt = types.SimpleNamespace(db=db, cfg=cfg)
    ids = seed(rt, ["今天过一下兴和的排期", "兴和要上线了", "牙测报告", "[音乐] 兰姐来了"])
    assert {"lexicon", "inbox"} <= set(cli.COMMAND_MODULES)

    code, text = run_cli(out, "lexicon", "list")
    assert code == 0 and "词库是空的" in text
    code, text = run_cli(out, "lexicon", "add", "星河", "--kind", "project", "--note", "正在做的项目", "--alias", "兴和")
    assert code == 0 and "已记下「星河」" in text and "听成「兴和」时：自动替换" in text
    code, text = run_cli(out, "lexicon", "alias", "星河", "银河计划", "--aka")
    assert code == 0 and "另一个叫法" in text
    code, text = run_cli(out, "lexicon", "alias", "星河", "新河")
    assert code == 0 and "不会自动替换" in text
    code, text = run_cli(out, "lexicon", "alias", "没有这个词", "某某")
    assert code == 2 and "词库里没有" in text
    code, text = run_cli(out, "lexicon", "list", "--json")
    doc = json.loads(text)
    assert code == 0 and [t["term"] for t in doc["terms"]] == ["星河"] and doc["counts"]["aliases"] == 3
    assert [(a["alias"], a["kind"], a["scope"]) for a in doc["terms"][0]["aliases"]] == [
        ("兴和", "mishear", "auto"), ("银河计划", "aka", "hint"), ("新河", "mishear", "hint")]
    code, text = run_cli(out, "lexicon", "list", "-q", "兴和")
    assert code == 0 and "星河" in text and "?新河" in text and "项目" in text

    # apply: a preview first, then for real (everything through revisions)
    code, text = run_cli(out, "lexicon", "apply")
    assert code == 0 and "其中 2 句（2 处）会被修改" in text and texts_of(rt)[0] == "今天过一下兴和的排期"
    code, text = run_cli(out, "lexicon", "apply", "--day", DAY, "--yes")
    assert code == 0 and "改了 2 句" in text and texts_of(rt)[:2] == ["今天过一下星河的排期", "星河要上线了"]
    assert revisions.history(db, ids[0])[0]["source"] == "lexicon"
    code, text = run_cli(out, "lexicon", "apply", "--term", "星河", "--alias", "兴和")
    assert code == 0 and "没有需要修改的句子" in text
    code, text = run_cli(out, "lexicon", "apply", "--alias", "兴和")
    assert code == 2 and "--term" in text
    code, text = run_cli(out, "lexicon", "apply", "--term", "星河", "--alias", "不存在")
    assert code == 2 and "没有别名" in text

    # import / export
    src = tmp_path / "words.txt"
    src.write_text("# 词表\n压测\n灰度发布 # 上线方式\n", encoding="utf-8")
    code, text = run_cli(out, "lexicon", "import", str(src))
    assert code == 0 and "读到 2 个词：新增 2" in text
    code, text = run_cli(out, "lexicon", "import", str(tmp_path / "missing.txt"))
    assert code == 2
    code, text = run_cli(out, "lexicon", "export")
    assert code == 0 and text == "星河\n压测\n灰度发布\n"
    dst = tmp_path / "out.txt"
    code, text = run_cli(out, "lexicon", "export", "--out", str(dst))
    assert code == 0 and dst.read_text(encoding="utf-8") == "星河\n压测\n灰度发布\n"
    code, text = run_cli(out, "lexicon", "import", str(src), "--replace")
    assert code == 0 and "停用了不在文件里的 1 个" in text

    # inbox
    code, text = run_cli(out, "inbox", "list")
    assert code == 0 and "没有要确认的事" in text
    q = inbox.add(db, "term", title="「牙测」是哪个词？", cfg=cfg, payload={
        "span": "牙测", "sentence": "牙测报告", "utterance_id": ids[2], "candidates": ["压测", "押测"], "term_kind": "term"})
    q2 = inbox.add(db, "term", title="「兰姐」是谁？", cfg=cfg, payload={
        "span": "兰姐", "sentence": "[音乐] 兰姐来了", "utterance_id": ids[3], "candidates": ["岚姐"], "term_kind": "name"})
    code, text = run_cli(out, "inbox", "list")
    assert code == 0 and "「牙测」是哪个词？" in text and "[0] 压测" in text and "[音乐] 兰姐来了" in text     # brackets are not markup
    code, text = run_cli(out, "inbox", "list", "--json")
    doc = json.loads(text)
    assert code == 0 and doc["counts"]["open"] == 2 and {it["id"] for it in doc["items"]} == {q, q2}
    code, text = run_cli(out, "inbox", "answer", str(q), "--choice", "0", "--correct")
    assert code == 2 and "只能用一个" in text
    code, text = run_cli(out, "inbox", "answer", str(q), "--choice", "0")
    assert code == 0 and "「牙测」→「压测」" in text and "改了 1 句" in text and texts_of(rt)[2] == "压测报告"
    code, text = run_cli(out, "inbox", "answer", str(q), "--choice", "0")
    assert code == 2 and "已经处理过" in text
    code, text = run_cli(out, "inbox", "dismiss", str(q2), "--never")
    assert code == 0 and "以后不再问" in text and inbox.get(db, q2)["payload"]["never"] is True
    code, text = run_cli(out, "inbox", "dismiss", "999")
    assert code == 2
    code, text = run_cli(out, "inbox", "list", "--status", "all", "--kind", "term")
    assert code == 0 and "已回答" in text and "已忽略" in text
    db.close_all()


def test_recorrect_job_runs_through_the_jobs_endpoint(client: TestClient, rt: Runtime):
    ids = seed(rt, ["兴和的排期", "无关"])
    tid = lexicon.upsert_term(rt.db, "星河")
    lexicon.add_alias(rt.db, tid, "兴和")
    r = client.post("/api/jobs/recorrect", json={"dry_run": "true", "lexicon_id": str(tid)})
    assert r.status_code == 200 and rt.wait_jobs(10)
    assert rt.jobs["recorrect"].result == {"scanned": 2, "changed": 1, "edits": 1, "dry_run": True} and texts_of(rt)[0] == "兴和的排期"
    r = client.post("/api/jobs/recorrect", json={"day": DAY})
    assert r.status_code == 200 and r.json()["job"]["label"] == "按词库重校历史" and rt.wait_jobs(10)
    assert rt.jobs["recorrect"].error is None and rt.jobs["recorrect"].result["changed"] == 1
    assert texts_of(rt) == ["星河的排期", "无关"] and revisions.history(rt.db, ids[0])[0]["source"] == "lexicon"
