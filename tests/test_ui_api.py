"""audio_note.ui.app (FastAPI JSON API) through fastapi.testclient over a real Runtime, plus UIServer.

No network: mimo.api_key is "" (chat / test-api answer 409), the memory chat is a fake object, push
goes through a stub Notifier. No microphone: the recorder loop is a fake thread body. Audio-related
endpoints use real wav files written into the temp data dir; enrollment / labelling / import run the
real CPU models (CAM++ embedding, Silero VAD) on the sample recording from conftest.SAMPLE_WAV.
"""
from __future__ import annotations

import io
import json
import logging
import os
import socket
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
import soundfile as sf
import yaml
from fastapi.testclient import TestClient

from audio_note import runtime as rt_mod
from audio_note.audio.storage import segment_path
from audio_note.audio.vad import SAMPLE_RATE as SR
from audio_note.db import DB
from audio_note.runtime import JOBS, Runtime
from audio_note.ui.app import create_app
from audio_note.ui.server import UIServer, port_free

from .conftest import SAMPLE_WAV

DAY = "2026-10-30"
needs_sample = pytest.mark.skipif(not SAMPLE_WAV.exists(), reason="real sample recording not available")


# --- fixtures / helpers ----------------------------------------------------------------------------

@pytest.fixture
def ui_cfg(cfg_factory, tmp_path: Path):
    cfg = cfg_factory({"memory": {"retrieval": "fts", "embedding_model": "hash"},
                       "correction": {"glossary_file": str(tmp_path / "glossary.txt")}})
    yield cfg
    from audio_note.logging_setup import _remove_own_handlers

    _remove_own_handlers(logging.getLogger())


@pytest.fixture
def rt(ui_cfg) -> Runtime:
    r = Runtime(ui_cfg)
    yield r
    r.shutdown()


@pytest.fixture
def client(rt: Runtime) -> TestClient:
    with TestClient(create_app(rt), raise_server_exceptions=False) as c:
        yield c


def _wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _wav_len(data: bytes) -> int:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    assert sr == SR
    return len(audio)


def _sample_slice(a: float, b: float) -> np.ndarray:
    audio, sr = sf.read(str(SAMPLE_WAV), dtype="float32")
    assert sr == SR
    return audio[int(a * SR):int(b * SR)]


def _tone(seconds: float, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _seg(cfg, db: DB, when: datetime, audio: np.ndarray | None = None, status: str = "done", with_file: bool = True) -> tuple[int, Path]:
    audio = _tone(5.0) if audio is None else audio
    p = segment_path(cfg, when, ".wav")
    if with_file:
        sf.write(p, audio, SR, subtype="PCM_16")
    sid = db.add_segment(p, when, len(audio) / SR, speech_s=len(audio) / SR)
    if status != "new":
        db.set_segment_status(sid, status)
    return sid, p


def _utt(db: DB, seg: int, day: str, hhmm: str, text: str, who: str | None = None, speaker_id: int | None = None,
         start_s: float = 0.0, end_s: float = 1.0, score: float | None = None) -> int:
    return db.add_utterances([{
        "segment_id": seg, "start_s": start_s, "end_s": end_s, "abs_start": f"{day}T{hhmm}:00", "day": day,
        "cluster_label": 0, "speaker_id": speaker_id, "speaker_name": who, "speaker_score": score,
        "text_raw": text, "text": text,
    }])[0]


def _seed_day(cfg, db: DB) -> dict[str, Any]:
    """One done segment on DAY with three utterances: 张三 x2 at 09:00/09:01, an unknown one at 09:40."""
    seg, p = _seg(cfg, db, datetime(2026, 10, 30, 9, 0))
    sid = db.upsert_speaker("张三", "同事")
    ids = [
        _utt(db, seg, DAY, "09:00", "今天讨论了预算问题", "张三", sid, 0.0, 1.0, 0.8),
        _utt(db, seg, DAY, "09:01", "明天去北京出差", "张三", sid, 1.0, 2.0, 0.7),
        _utt(db, seg, DAY, "09:40", "午饭吃什么", None, None, 2.0, 3.0),
    ]
    return {"seg": seg, "path": p, "speaker": sid, "ids": ids}


# --- status / errors / static -----------------------------------------------------------------------

def test_status_endpoint(client: TestClient):
    r = client.get("/api/status")
    assert r.status_code == 200
    st = r.json()
    assert st["pid"] == os.getpid() and st["service"]["running"] is False and st["api_key"] is False
    assert set(st["jobs"]) == set(JOBS) and st["pending"] == 0


def test_unexpected_error_is_json_500(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    def boom():
        raise RuntimeError("kaboom")

    monkeypatch.setattr(rt, "status", boom)
    r = client.get("/api/status")
    assert r.status_code == 500 and r.json() == {"error": "RuntimeError: kaboom"}


def test_static_index_and_assets(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html") and "audio-note" in r.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/nope.js").status_code == 404


# --- days / transcript / segments -------------------------------------------------------------------

def test_days_lists_utterance_days_pending_segments_and_daily(client: TestClient, rt: Runtime):
    _seed_day(rt.cfg, rt.db)
    _seg(rt.cfg, rt.db, datetime(2026, 10, 29, 8, 0), status="new", with_file=False)
    _seg(rt.cfg, rt.db, datetime(2026, 10, 29, 8, 5), status="failed", with_file=False)
    _seg(rt.cfg, rt.db, datetime(2026, 10, 28, 8, 5), status="deleted", with_file=False)
    rt.db.execute("INSERT INTO daily_summaries(day, summary_md, utterance_count, created_at) VALUES (?,?,?,?)", (DAY, "# x", 3, "2026-10-30 22:30:00"))
    days = client.get("/api/days").json()
    assert [d["day"] for d in days] == ["2026-10-30", "2026-10-29"]            # deleted-only day is not listed
    d0 = days[0]
    assert d0["utterances"] == 3 and d0["first"] == "09:00" and d0["last"] == "09:40" and d0["speech_min"] == 0.1
    assert d0["segments"] == 1 and d0["pending"] == 0 and d0["failed"] == 0
    assert d0["daily"] == {"created_at": "2026-10-30 22:30:00", "pushed_at": None, "utterance_count": 3}
    assert d0["speakers"] == [{"name": "张三", "n": 2}, {"name": "未知", "n": 1}]
    d1 = days[1]
    assert d1["utterances"] == 0 and d1["segments"] == 2 and d1["pending"] == 1 and d1["failed"] == 1 and d1["daily"] is None


def test_transcript_groups_sessions_and_filters_by_speaker(client: TestClient, rt: Runtime):
    seed = _seed_day(rt.cfg, rt.db)
    gone, _ = _seg(rt.cfg, rt.db, datetime(2026, 10, 30, 9, 41), with_file=False)   # audio expired
    _utt(rt.db, gone, DAY, "09:41", "没有音频的句子", "张三", seed["speaker"])
    r = client.get(f"/api/transcript/{DAY}")
    assert r.status_code == 200
    t = r.json()
    assert t["day"] == DAY and t["unknown_label"] == "未知" and t["markdown_path"] is None
    assert [s["index"] for s in t["sessions"]] == [1, 2]                           # 09:01 -> 09:40 is > session_gap_min
    s1, s2 = t["sessions"]
    assert (s1["start"], s1["end"]) == ("09:00", "09:01") and len(s1["utterances"]) == 2
    u = s1["utterances"][0]
    assert u["speaker"] == "张三" and u["score"] == 0.8 and u["time"] == "09:00:00" and u["audio"] is True
    assert u["text"] == "今天讨论了预算问题" and "text_raw" not in u and u["segment_id"] == seed["seg"]
    assert [x["speaker"] for x in s2["utterances"]] == ["未知", "张三"]
    assert [x["audio"] for x in s2["utterances"]] == [True, False]
    assert t["speakers"] == {"张三": 3, "未知": 1}
    assert [s["id"] for s in t["segments"]] == [seed["seg"], gone] and t["segments"][0]["status"] == "done"
    # speaker filter drops sessions without that speaker's lines
    f = client.get(f"/api/transcript/{DAY}", params={"speaker": "未知"}).json()
    assert [s["index"] for s in f["sessions"]] == [2] and len(f["sessions"][0]["utterances"]) == 1
    assert client.get(f"/api/transcript/{DAY}", params={"speaker": "李四"}).json()["sessions"] == []
    # a day without content still answers
    e = client.get("/api/transcript/2026-01-01").json()
    assert e["sessions"] == [] and e["segments"] == [] and e["speakers"] == {}


def test_transcript_bad_date_is_400(client: TestClient):
    r = client.get("/api/transcript/2026-1-1")
    assert r.status_code == 400 and "YYYY-MM-DD" in r.json()["error"]
    assert client.get("/api/daily/20261030").status_code == 400
    assert client.get("/api/search", params={"q": "x", "day_from": "bad"}).status_code == 400


def test_segments_list_filters(client: TestClient, rt: Runtime):
    seed = _seed_day(rt.cfg, rt.db)
    new, _ = _seg(rt.cfg, rt.db, datetime(2026, 10, 29, 8, 0), status="new", with_file=False)
    _seg(rt.cfg, rt.db, datetime(2026, 10, 28, 8, 0), status="deleted", with_file=False)
    rows = client.get("/api/segments").json()
    assert [r["id"] for r in rows] == [seed["seg"], new] and "path" not in rows[0]
    assert rows[0]["audio"] is True and rows[0]["name"] == seed["path"].name and rows[1]["audio"] is False
    assert [r["id"] for r in client.get("/api/segments", params={"status": "new"}).json()] == [new]
    assert [r["id"] for r in client.get("/api/segments", params={"day": DAY}).json()] == [seed["seg"]]
    assert client.get("/api/segments", params={"day": "x"}).status_code == 400
    assert len(client.get("/api/segments", params={"limit": 1}).json()) == 1


# --- search / context ---------------------------------------------------------------------------------

def test_search_fts_mode(client: TestClient, rt: Runtime):
    from audio_note.memory import index as mem_index

    _seed_day(rt.cfg, rt.db)
    mem_index.reindex(rt.db)
    r = client.get("/api/search", params={"q": "预算"})
    assert r.status_code == 200
    res = r.json()
    assert res["how"] == "全文检索" and res["expansions"] == []
    assert len(res["hits"]) == 1
    h = res["hits"][0]
    assert h["text"] == "今天讨论了预算问题" and h["speaker"] == "张三" and h["day"] == DAY and h["time"] == "09:00"
    assert h["source"] == "fts" and isinstance(h["score"], float) and h["segment_id"] >= 1
    assert client.get("/api/search", params={"q": "  "}).json() == {"hits": [], "how": ""}
    assert client.get("/api/search", params={"q": "预算", "speaker": "李四"}).json()["hits"] == []
    assert client.get("/api/search", params={"q": "预算", "speaker": "张三", "day_from": DAY, "day_to": DAY, "mode": "fts"}).json()["hits"][0]["id"] == h["id"]
    assert client.get("/api/search", params={"q": "北京", "day_to": "2026-10-29"}).json()["hits"] == []
    assert set(rt._retrievers) == {("fts", False)}


def test_context_window_and_404(client: TestClient, rt: Runtime):
    seed = _seed_day(rt.cfg, rt.db)
    a, b, c = seed["ids"]
    r = client.get(f"/api/context/{b}", params={"before": 1, "after": 1})
    assert r.status_code == 200
    assert r.json()["id"] == b and [u["id"] for u in r.json()["utterances"]] == [a, b, c]
    assert [u["id"] for u in client.get(f"/api/context/{a}", params={"before": 0, "after": 0}).json()["utterances"]] == [a]
    r = client.get("/api/context/999")
    assert r.status_code == 404 and r.json() == {"error": "没有这条记录"}


# --- audio ----------------------------------------------------------------------------------------------

def test_audio_utterance_returns_padded_wav_and_404_when_file_gone(client: TestClient, rt: Runtime):
    seed = _seed_day(rt.cfg, rt.db)
    uid = seed["ids"][1]                                   # 1.0s-2.0s of a 5 s file
    r = client.get(f"/api/audio/utterance/{uid}", params={"pad": 0.5})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert r.headers["cache-control"] == "no-store"       # v3b: utterance ids get reused by splits, never cache the clip
    assert _wav_len(r.content) == int(2.0 * SR)           # 0.5 .. 2.5
    assert _wav_len(client.get(f"/api/audio/utterance/{uid}", params={"pad": 0}).content) == SR
    assert _wav_len(client.get(f"/api/audio/utterance/{uid}", params={"pad": 99}).content) == 5 * SR     # pad capped, clipped to file
    assert client.get("/api/audio/utterance/999").status_code == 404
    seed["path"].unlink()
    r = client.get(f"/api/audio/utterance/{uid}")
    assert r.status_code == 404 and "已删除" in r.json()["error"]


def test_audio_segment_range(client: TestClient, rt: Runtime):
    seg, _ = _seg(rt.cfg, rt.db, datetime(2026, 10, 30, 9, 0), audio=_tone(4.0))
    r = client.get(f"/api/audio/segment/{seg}", params={"from": 1, "to": 2.5})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and _wav_len(r.content) == int(1.5 * SR)
    assert _wav_len(client.get(f"/api/audio/segment/{seg}").content) == 4 * SR                      # whole file
    assert _wav_len(client.get(f"/api/audio/segment/{seg}", params={"from": 3, "to": 1}).content) == SR   # to<=from: until the end
    assert client.get("/api/audio/segment/999").status_code == 404


# --- daily / notes --------------------------------------------------------------------------------------

def test_daily_list_and_get_with_stale_flag(client: TestClient, rt: Runtime):
    _seed_day(rt.cfg, rt.db)                                # 3 utterances on DAY
    js = {"overview": "总" * 200, "partial": True, "blocks": [{"i": 1}], "todos": []}
    rt.db.execute("INSERT INTO daily_summaries(day, summary_md, summary_json, model, utterance_count, created_at, pushed_at) VALUES (?,?,?,?,?,?,?)",
                  (DAY, "# 日报", json.dumps(js, ensure_ascii=False), "mimo-v2.6-pro", 2, "2026-10-30 22:30:00", "2026-10-30 22:31:00"))
    rt.db.execute("INSERT INTO daily_summaries(day, summary_md, summary_json, utterance_count) VALUES (?,?,?,?)",
                  ("2026-10-29", "# 旧", "not json", 0))
    lst = client.get("/api/daily").json()
    assert [d["day"] for d in lst] == [DAY, "2026-10-29"]
    d0 = lst[0]
    assert d0["stale"] is True and d0["now_count"] == 3 and d0["utterance_count"] == 2 and d0["partial"] is True
    assert d0["overview"] == "总" * 160 and d0["model"] == "mimo-v2.6-pro" and d0["pushed_at"] == "2026-10-30 22:31:00"
    assert lst[1]["stale"] is False and lst[1]["partial"] is False and lst[1]["overview"] == ""
    g = client.get(f"/api/daily/{DAY}").json()
    assert g["exists"] is True and g["markdown"] == "# 日报" and g["stale"] is True and g["now_count"] == 3
    assert "blocks" not in g["json"] and g["json"]["partial"] is True and g["json"]["todos"] == []
    assert client.get("/api/daily/2026-10-29").json()["json"] == {}
    assert client.get("/api/daily/2026-10-01").json() == {"day": "2026-10-01", "exists": False, "now_count": 0}


def test_notes_list_filter_and_delete(client: TestClient, rt: Runtime):
    for kind, title, content, upd in (("todo", "买菜", "明天买菜", "2026-10-30 10:00:00"),
                                      ("person", "张三", "同事，负责预算", "2026-10-30 11:00:00"),
                                      ("topic", "预算", "本月预算讨论", "2026-10-30 11:00:00")):
        rt.db.execute("INSERT INTO memory_notes(kind, title, content, day_from, day_to, updated_at) VALUES (?,?,?,?,?,?)",
                      (kind, title, content, DAY, DAY, upd))
    rows = client.get("/api/notes").json()
    assert [r["title"] for r in rows] == ["预算", "张三", "买菜"]                 # updated_at desc, id desc
    assert set(rows[0]) == {"id", "kind", "title", "content", "day_from", "day_to", "updated_at"}
    assert [r["title"] for r in client.get("/api/notes", params={"kind": "todo"}).json()] == ["买菜"]
    assert [r["title"] for r in client.get("/api/notes", params={"q": "预算"}).json()] == ["预算", "张三"]
    assert [r["title"] for r in client.get("/api/notes", params={"q": "预算", "kind": "person"}).json()] == ["张三"]
    assert len(client.get("/api/notes", params={"limit": 1}).json()) == 1
    nid = rows[2]["id"]
    assert client.delete(f"/api/notes/{nid}").json() == {"ok": True, "deleted": 1}
    assert client.delete(f"/api/notes/{nid}").json() == {"ok": True, "deleted": 0}
    assert len(client.get("/api/notes").json()) == 2


# --- speakers ---------------------------------------------------------------------------------------------

def _speaker_with_samples(rt: Runtime, name: str, n: int = 1, wav: Path | None = None) -> int:
    sid = rt.db.upsert_speaker(name)
    rng = np.random.default_rng(hash(name) % 1000)
    base = rng.normal(size=16).astype(np.float32)
    for i in range(n):
        rt.db.add_speaker_embedding(sid, base + 0.05 * rng.normal(size=16).astype(np.float32), str(wav) if wav else None, 5.0 + i)
    return sid


def test_speakers_list(client: TestClient, rt: Runtime):
    seed = _seed_day(rt.cfg, rt.db)
    rt.db.add_speaker_embedding(seed["speaker"], np.ones(8, dtype=np.float32), None, 7.5)
    other = _speaker_with_samples(rt, "李四", 2)
    r = client.get("/api/speakers").json()
    assert r["unknown_utterances"] == 1 and r["identify_threshold"] == 0.45 and r["unknown_label"] == "未知"
    by_name = {s["name"]: s for s in r["speakers"]}
    assert by_name["张三"]["samples"] == 1 and by_name["张三"]["duration_s"] == 7.5 and by_name["张三"]["utterances"] == 2
    assert by_name["张三"]["notes"] == "同事" and by_name["张三"]["id"] == seed["speaker"]
    assert by_name["李四"] == {**by_name["李四"], "id": other, "samples": 2, "duration_s": 11.0, "utterances": 0, "notes": ""}


def test_speaker_rename_updates_utterances_and_rejects_duplicates(client: TestClient, rt: Runtime):
    seed = _seed_day(rt.cfg, rt.db)
    sid = seed["speaker"]
    _speaker_with_samples(rt, "王五")
    r = client.patch(f"/api/speakers/{sid}", json={"name": "李雷", "notes": "邻居"})
    assert r.status_code == 200 and r.json() == {"ok": True, "renamed_days": 1}
    assert rt.db.one("SELECT name, notes FROM speakers WHERE id=?", (sid,))[:] == ("李雷", "邻居")
    names = [u["speaker_name"] for u in rt.db.query("SELECT speaker_name FROM utterances ORDER BY id")]
    assert names == ["李雷", "李雷", None]
    assert (rt.cfg.transcripts_dir / f"{DAY}.md").exists()                      # day markdown rewritten
    assert "李雷" in (rt.cfg.transcripts_dir / f"{DAY}.md").read_text(encoding="utf-8")
    # same name again / notes only: no rename
    assert client.patch(f"/api/speakers/{sid}", json={"name": "李雷"}).json() == {"ok": True, "renamed_days": 0}
    assert client.patch(f"/api/speakers/{sid}", json={"notes": ""}).json()["renamed_days"] == 0
    assert rt.db.one("SELECT notes FROM speakers WHERE id=?", (sid,))["notes"] is None
    r = client.patch(f"/api/speakers/{sid}", json={"name": "王五"})
    assert r.status_code == 400 and "已经有叫 王五" in r.json()["error"]
    assert rt.db.one("SELECT name FROM speakers WHERE id=?", (sid,))["name"] == "李雷"
    assert client.patch("/api/speakers/999", json={"name": "x"}).status_code == 404


def test_speaker_delete_unlabels_utterances(client: TestClient, rt: Runtime):
    seed = _seed_day(rt.cfg, rt.db)
    sid = seed["speaker"]
    rt.db.add_speaker_embedding(sid, np.ones(8, dtype=np.float32), None, 5.0)
    r = client.delete(f"/api/speakers/{sid}")
    assert r.status_code == 200 and r.json() == {"ok": True, "unlabeled_days": 1}
    rows = rt.db.query("SELECT speaker_id, speaker_name, speaker_score FROM utterances ORDER BY id")
    assert [tuple(x) for x in rows] == [(None, "未知", None), (None, "未知", None), (None, None, None)]
    assert rt.db.one("SELECT COUNT(*) AS n FROM speakers")["n"] == 0
    assert rt.db.one("SELECT COUNT(*) AS n FROM speaker_embeddings")["n"] == 0
    assert client.get("/api/speakers").json()["unknown_utterances"] == 3
    assert client.delete(f"/api/speakers/{sid}").status_code == 404


def test_speaker_samples_list_audio_and_delete(client: TestClient, rt: Runtime, tmp_path: Path):
    wav = tmp_path / "样本.wav"
    sf.write(str(wav), _tone(3.0), SR, subtype="PCM_16")
    sid = rt.db.upsert_speaker("甲")
    e1 = rt.db.add_speaker_embedding(sid, np.ones(8, dtype=np.float32), str(wav), 3.0)
    e2 = rt.db.add_speaker_embedding(sid, np.ones(8, dtype=np.float32), f"{wav}#1.0-2.0", 1.0)
    e3 = rt.db.add_speaker_embedding(sid, np.ones(8, dtype=np.float32), str(tmp_path / "gone.wav"), 1.0)
    e4 = rt.db.add_speaker_embedding(sid, np.ones(8, dtype=np.float32), None, 1.0)
    rows = client.get(f"/api/speakers/{sid}/samples").json()
    assert [r["id"] for r in rows] == [e1, e2, e3, e4]
    assert [r["file"] for r in rows] == ["样本.wav", "样本.wav", "gone.wav", ""]
    assert [r["audio"] for r in rows] == [True, True, False, False]
    assert rows[0]["duration_s"] == 3.0 and rows[0]["source_path"] == str(wav)
    assert client.get("/api/speakers/999/samples").json() == []
    assert _wav_len(client.get(f"/api/speakers/samples/{e1}/audio").content) == 3 * SR
    assert _wav_len(client.get(f"/api/speakers/samples/{e2}/audio").content) == SR         # sliced by the #a-b suffix
    assert client.get(f"/api/speakers/samples/{e3}/audio").status_code == 404
    assert client.get(f"/api/speakers/samples/{e4}/audio").status_code == 404
    assert client.get("/api/speakers/samples/999/audio").status_code == 404
    assert client.delete(f"/api/speakers/samples/{e2}").json() == {"ok": True, "deleted": 1}
    assert client.delete(f"/api/speakers/samples/{e2}").json() == {"ok": True, "deleted": 0}
    assert [r["id"] for r in client.get(f"/api/speakers/{sid}/samples").json()] == [e1, e3, e4]


@needs_sample
def test_speaker_add_sample_enrolls_real_audio(client: TestClient, rt: Runtime):
    r = client.post("/api/speakers/雷军/samples", params={"filename": "lei.wav", "notes": "测试"}, content=SAMPLE_WAV.read_bytes())
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["ok"] is True and out["name"] == "雷军" and out["samples"] == 1 and out["speech_s"] >= 10.0
    row = rt.db.one("SELECT id, notes FROM speakers WHERE name='雷军'")
    assert row["id"] == out["id"] and row["notes"] == "测试"
    emb = rt.db.query("SELECT source_path, duration_s, dim FROM speaker_embeddings WHERE speaker_id=?", (out["id"],))
    assert len(emb) == 1 and emb[0]["dim"] > 0 and Path(emb[0]["source_path"]).exists()
    assert rt.cfg.speakers_dir in Path(emb[0]["source_path"]).parents              # speech copy kept under data/speakers
    assert list((rt.cfg.data_dir / "uploads").glob("enroll_*")) == []           # upload temp file removed
    # a second sample for the same name adds to the same speaker
    r2 = client.post("/api/speakers/雷军/samples", params={"filename": "lei2.wav"}, content=_wav_bytes(_sample_slice(30, 60)))
    assert r2.status_code == 200 and r2.json()["id"] == out["id"] and r2.json()["samples"] == 2


def test_speaker_add_sample_rejections(client: TestClient, rt: Runtime):
    r = client.post("/api/speakers/甲/samples", params={"filename": "a.wav"}, content=b"RIFF" * 10)
    assert r.status_code == 400 and "没有收到音频数据" in r.json()["error"]
    r = client.post("/api/speakers/甲/samples", params={"filename": "notes.txt"}, content=_wav_bytes(_tone(2.0)))
    assert r.status_code == 400 and "不支持的音频格式 .txt" in r.json()["error"]
    r = client.post("/api/speakers/%20%20/samples", params={"filename": "a.wav"}, content=_wav_bytes(_tone(2.0)))
    assert r.status_code == 400 and "姓名不能为空" in r.json()["error"]
    # too little speech (0.5 s of silence): the speaker row created for it is removed again
    r = client.post("/api/speakers/乙/samples", params={"filename": "s.wav"}, content=_wav_bytes(np.zeros(SR // 2, dtype=np.float32)))
    assert r.status_code == 400 and "语音太少" in r.json()["error"]
    assert rt.db.one("SELECT COUNT(*) AS n FROM speakers")["n"] == 0
    assert list((rt.cfg.data_dir / "uploads").glob("*")) == []


@needs_sample
def test_speaker_label_by_utterance_enrolls_and_relabels(client: TestClient, rt: Runtime):
    seg, p = _seg(rt.cfg, rt.db, datetime(2026, 10, 30, 9, 0), audio=_sample_slice(30, 42))
    uid = _utt(rt.db, seg, DAY, "09:00", "大家好", None, None, 0.0, 8.0)
    r = client.post("/api/speakers/label", json={"name": " 雷军 ", "utterance_id": uid})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["ok"] is True and out["name"] == "雷军" and out["samples"] == 1 and out["segment_id"] == seg
    # the labelled utterance is assigned by hand (speaker_source='user'), so relabel has nothing left to change
    assert "relabel_error" not in out and out["relabeled_changed"] == 0 and out["assigned"] == 1
    emb = rt.db.query("SELECT source_path, duration_s FROM speaker_embeddings")
    assert len(emb) == 1 and emb[0]["source_path"] == f"{p}#0.0-8.0" and 1.0 <= emb[0]["duration_s"] <= 8.0
    u = rt.db.one("SELECT speaker_id, speaker_name, speaker_score, speaker_source FROM utterances WHERE id=?", (uid,))
    assert u["speaker_name"] == "雷军" and u["speaker_id"] == out["id"] and u["speaker_score"] is None and u["speaker_source"] == "user"
    assert (rt.cfg.transcripts_dir / f"{DAY}.md").exists()


def test_speaker_label_error_cases(client: TestClient, rt: Runtime):
    seg, p = _seg(rt.cfg, rt.db, datetime(2026, 10, 30, 9, 0), audio=_tone(5.0))
    uid = _utt(rt.db, seg, DAY, "09:00", "x", None, None, 0.0, 0.3)
    gone, _ = _seg(rt.cfg, rt.db, datetime(2026, 10, 30, 10, 0), with_file=False)
    gone_uid = _utt(rt.db, gone, DAY, "10:00", "y", None, None, 0.0, 2.0)
    post = lambda body: client.post("/api/speakers/label", json=body)  # noqa: E731
    assert post({"name": "", "utterance_id": uid}).status_code == 400
    assert post({"name": "甲", "utterance_id": 999}).status_code == 404
    r = post({"name": "甲"})
    assert r.status_code == 400 and "需要 utterance_id" in r.json()["error"]
    r = post({"name": "甲", "segment_id": seg, "from": 2.0, "to": 1.0})
    assert r.status_code == 400 and "结束时间必须大于开始时间" in r.json()["error"]
    r = post({"name": "甲", "segment_id": 999, "from": 0, "to": 1})
    assert r.status_code == 404 and "已不存在" in r.json()["error"]
    assert post({"name": "甲", "utterance_id": gone_uid}).status_code == 404
    r = post({"name": "甲", "utterance_id": uid})                               # 0.3 s: too little speech for a sample,
    assert r.status_code == 200 and r.json()["samples"] == 0 and "语音" in r.json()["sample_note"]   # but the sentence is assigned
    assert r.json()["assigned"] == 1 and rt.db.one("SELECT speaker_name, speaker_source FROM utterances WHERE id=?", (uid,))["speaker_name"] == "甲"
    assert rt.db.one("SELECT COUNT(*) AS n FROM speaker_embeddings")["n"] == 0


def test_speakers_calibrate(client: TestClient, rt: Runtime):
    r = client.post("/api/speakers/calibrate")
    assert r.status_code == 400 and "至少需要 2 个注册样本" in r.json()["error"]
    _speaker_with_samples(rt, "甲", 1)
    assert client.post("/api/speakers/calibrate").status_code == 400
    _speaker_with_samples(rt, "乙", 2)
    r = client.post("/api/speakers/calibrate")
    assert r.status_code == 200
    out = r.json()
    assert out["n_embeddings"] == 3 and out["n_speakers"] == 2 and out["current"] == 0.45
    assert out["same"]["n"] == 1 and out["cross"]["n"] == 2 and 0.3 <= out["suggested"] <= 0.7


# --- chat ----------------------------------------------------------------------------------------------------

class FakeChat:
    def __init__(self):
        self.strategy, self.model = "prefetch", "mimo-v2.6-flash"
        self.retriever = SimpleNamespace(mode="fts")
        self.last_stats = {"strategy": "prefetch", "llm_calls": 1, "tool_calls": 1}
        self.on_tool = None
        self.calls: list = []
        self.fail: Exception | None = None

    def ask(self, question, history=None):
        self.calls.append((question, history))
        if self.fail:
            raise self.fail
        self.on_tool("search_transcripts", {"query": question}, "第一行\n第二行")
        return "答案：" + question


def _ndjson(client: TestClient, body: dict) -> list[dict]:
    with client.stream("POST", "/api/chat", json=body) as r:
        assert r.status_code == 200, r.read()
        assert r.headers["content-type"].startswith("application/x-ndjson")
        return [json.loads(ln) for ln in r.iter_lines() if ln.strip()]


def test_chat_streams_start_tool_and_answer_events(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    fake = FakeChat()
    seen: list = []

    def chat(strategy=None, retrieval=None, expand=None):
        seen.append((strategy, retrieval, expand))
        return fake

    monkeypatch.setattr(rt, "chat", chat)
    history = [{"role": "user", "content": "之前"}, {"role": "system", "content": "x"}, {"role": "assistant", "content": None}]
    events = _ndjson(client, {"question": " 昨天说了什么 ", "history": history, "strategy": "rag", "retrieval": "fts", "expand": 0})
    assert [e["type"] for e in events] == ["start", "tool", "answer"]
    assert events[0] == {"type": "start", "strategy": "prefetch", "retrieval": "fts", "model": "mimo-v2.6-flash"}
    assert events[1] == {"type": "tool", "name": "search_transcripts", "args": {"query": "昨天说了什么"}, "result": "第一行\n第二行", "lines": 2}
    assert events[2]["text"] == "答案：昨天说了什么"
    stats = events[2]["stats"]
    assert stats["llm_calls"] == 1 and isinstance(stats["seconds"], float)
    assert seen == [("rag", "fts", False)]
    assert fake.calls == [("昨天说了什么", [{"role": "user", "content": "之前"}, {"role": "assistant", "content": ""}])]
    # defaults: no strategy / retrieval / expand given
    _ndjson(client, {"question": "q"})
    assert seen[-1] == (None, None, None)


def test_chat_error_event_and_validation(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    fake = FakeChat()
    fake.fail = RuntimeError("模型挂了")
    monkeypatch.setattr(rt, "chat", lambda *a, **k: fake)
    events = _ndjson(client, {"question": "q"})
    assert [e["type"] for e in events] == ["start", "error"] and events[1]["message"] == "模型挂了"
    r = client.post("/api/chat", json={"question": "   "})
    assert r.status_code == 400 and r.json() == {"error": "问题不能为空"}
    assert client.post("/api/chat", json={}).status_code == 400


def test_chat_without_api_key_is_409(client: TestClient):
    r = client.post("/api/chat", json={"question": "你好"})
    assert r.status_code == 409 and "MiMo API key" in r.json()["error"]
    r = client.post("/api/config/test-api")
    assert r.status_code == 409 and "API key" in r.json()["error"]


def test_config_test_api_with_fake_client(client: TestClient, rt: Runtime):
    calls: list = []

    def chat(messages, max_tokens=None):
        calls.append((messages, max_tokens))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="  ok \n"))])

    rt._client = SimpleNamespace(chat=chat, chat_model="mimo-v2.6-flash")
    r = client.post("/api/config/test-api")
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is True and out["model"] == "mimo-v2.6-flash" and out["reply"] == "ok" and out["seconds"] >= 0
    assert calls[0][1] == 8 and calls[0][0][0]["role"] == "user"


# --- config / glossary ------------------------------------------------------------------------------------------

def _write_config(rt: Runtime, extra: str = "") -> None:
    cfg = rt.cfg
    rt.config_path.write_text(
        "# keep this comment\n"
        "paths:\n"
        f"  data_dir: '{cfg.data_dir}'\n"
        f"  models_dir: '{cfg.models_dir}'\n"
        "mimo:\n"
        "  api_key: \"\"   # secret\n"
        "  chat_model: mimo-v2.6-flash   # model\n"
        "memory:\n"
        "  retrieval: fts\n"
        "  embedding_model: hash\n"
        "correction:\n"
        f"  glossary_file: '{cfg['correction']['glossary_file']}'\n"
        + extra, encoding="utf-8")


def test_config_get_masks_secrets(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    rt.cfg["mimo"]["api_key"] = "sk-abcdefghijklmnop"
    rt.cfg["push"]["feishu"]["webhook"] = "https://x"
    r = client.get("/api/config")
    assert r.status_code == 200
    assert "abcdefghijklmnop" not in r.text and "https://x" not in r.text
    out = r.json()
    assert out["path"] == str(rt.config_path) and out["exists"] is True and out["env_api_key"] is False
    assert out["values"]["mimo"]["api_key"] == "" and out["values"]["mimo"]["chat_model"] == "mimo-v2.6-flash"
    assert "rpm" not in out["values"]["mimo"] and "paths" not in out["values"]                 # non-editable keys are not shown
    assert out["values"]["push"]["feishu"] == {"webhook": "", "secret": ""}
    assert out["values"]["push"]["channels"] == ["toast"] and out["values"]["analysis"]["daily_time"] == "22:30"
    assert out["secrets"]["mimo.api_key"] == {"set": True, "hint": "sk-…mnop"}
    assert out["secrets"]["push.feishu.webhook"] == {"set": True, "hint": "…/x"}
    assert out["secrets"]["push.wecom.webhook"] == {"set": False, "hint": ""}
    assert out["defaults"]["ui"]["port"] == 8765 and out["defaults"]["mimo"]["api_key"] == ""
    assert out["glossary_path"] == str(rt.cfg.glossary_path) and out["data_dir"] == str(rt.cfg.data_dir)
    # masking must not touch the live config: a second GET sees the same secrets
    again = client.get("/api/config").json()
    assert again["secrets"]["push.feishu.webhook"]["set"] is True and rt.cfg.push.feishu.webhook == "https://x"
    assert rt.cfg.mimo.api_key == "sk-abcdefghijklmnop"
    monkeypatch.setenv("MIMO_API_KEY", "sk-from-env")
    assert client.get("/api/config").json()["env_api_key"] is True


def test_config_put_writes_yaml_keeping_comments_and_reloads(client: TestClient, rt: Runtime):
    _write_config(rt)
    body = {"mimo": {"chat_model": "mimo-v2.6-pro"}, "analysis": {"daily_time": "23:00"},
            "push": {"channels": ["toast", "ntfy"], "ntfy": {"topic": "notes"}}}
    r = client.put("/api/config", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["ok"] is True and out["restarted_recorder"] is False
    assert out["saved"] == ["analysis.daily_time", "mimo.chat_model", "push.channels", "push.ntfy.topic"]
    assert set(out["changed"]) == {"analysis", "mimo", "push"}
    assert not any("注释" in n for n in out["notes"])
    text = rt.config_path.read_text(encoding="utf-8")
    assert "# keep this comment" in text and "# model" in text and "# secret" in text
    data = yaml.safe_load(text)
    assert data["mimo"]["chat_model"] == "mimo-v2.6-pro" and data["mimo"]["api_key"] == ""
    assert data["analysis"] == {"daily_time": "23:00"} and data["push"] == {"channels": ["toast", "ntfy"], "ntfy": {"topic": "notes"}}
    assert data["paths"]["data_dir"] == str(rt.cfg.data_dir)
    assert rt.cfg.mimo.chat_model == "mimo-v2.6-pro" and rt.cfg.analysis.daily_time == "23:00"
    assert rt.status()["daily_time"] == "23:00" and rt.sched.cfg is rt.cfg
    assert not rt.config_path.with_suffix(".yaml.tmp").exists()
    # ui / log changes only produce a restart note
    out = client.put("/api/config", json={"ui": {"port": 9000}, "log": {"level": "DEBUG"}}).json()
    assert out["changed"] == ["log", "ui"] and any("重启" in n for n in out["notes"])


def test_config_put_validation(client: TestClient, rt: Runtime):
    _write_config(rt)
    before = rt.config_path.read_text(encoding="utf-8")
    r = client.put("/api/config", json={"mimo": {"rpm": 5}})
    assert r.status_code == 400 and r.json() == {"error": "不允许修改 mimo.rpm"}
    r = client.put("/api/config", json={"paths": {"data_dir": "C:/elsewhere"}})
    assert r.status_code == 400 and "paths.data_dir" in r.json()["error"]
    r = client.put("/api/config", json={"mimo": {"api_key": "sk-xxx"}})
    assert r.status_code == 400 and "占位符" in r.json()["error"]
    r = client.put("/api/config", json={"analysis": {"daily_time": "25:00"}})
    assert r.status_code == 400 and r.json() == {"error": "日报时间格式应为 HH:MM"}
    assert client.put("/api/config", json={"analysis": {"daily_time": "2230"}}).status_code == 400
    assert rt.config_path.read_text(encoding="utf-8") == before


def test_config_put_empty_secret_keeps_old_value(client: TestClient, rt: Runtime):
    _write_config(rt, "push:\n  feishu: {webhook: 'https://old.example/hook', secret: ''}   # inline\n")
    rt.config_path.write_text(rt.config_path.read_text(encoding="utf-8").replace('api_key: ""', 'api_key: "sk-old-secret-value-1234"'), encoding="utf-8")
    rt.reload_config()
    assert rt.cfg.mimo.api_key == "sk-old-secret-value-1234"
    r = client.put("/api/config", json={"mimo": {"api_key": ""}, "push": {"feishu": {"webhook": "", "secret": None}}})
    assert r.status_code == 200 and r.json() == {"ok": True, "changed": [], "notes": ["没有需要保存的修改"], "saved": []}
    data = yaml.safe_load(rt.config_path.read_text(encoding="utf-8"))
    assert data["mimo"]["api_key"] == "sk-old-secret-value-1234" and data["push"]["feishu"]["webhook"] == "https://old.example/hook"
    r = client.put("/api/config", json={"push": {"feishu": {"webhook": "", "secret": "new-secret"}}, "mimo": {"api_key": "   "}})
    assert r.status_code == 200 and r.json()["saved"] == ["push.feishu.secret"]
    text = rt.config_path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert data["push"]["feishu"] == {"webhook": "https://old.example/hook", "secret": "new-secret"} and "# inline" in text
    assert data["mimo"]["api_key"] == "sk-old-secret-value-1234" and rt.cfg.push.feishu.secret == "new-secret"
    r = client.put("/api/config", json={"mimo": {"api_key": "sk-brand-new-key-value"}})
    assert r.status_code == 200 and r.json()["saved"] == ["mimo.api_key"]
    assert rt.cfg.mimo.api_key == "sk-brand-new-key-value" and "sk-old-secret" not in rt.config_path.read_text(encoding="utf-8")


def test_glossary_get_put_and_reindex_job(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    p = rt.cfg.glossary_path
    assert not p.exists()
    assert client.get("/api/glossary").json() == {"path": str(p), "text": ""}
    r = client.put("/api/glossary", json={"text": "# 术语\n小米\n\nMiMo  \n"})
    assert r.status_code == 200 and r.json() == {"ok": True, "terms": 2}
    assert p.read_bytes() == "# 术语\n小米\n\nMiMo  \n".encode("utf-8")           # written verbatim, LF endings
    assert client.get("/api/glossary").json()["text"] == "# 术语\n小米\n\nMiMo  \n"
    seen: list = []
    monkeypatch.setattr(rt, "_job_reindex", lambda vectors=True: seen.append(vectors) or {"keywords": 0})
    r = client.put("/api/glossary", json={"text": "小米", "reindex": True}).json()
    assert r["terms"] == 1 and r["job"]["name"] == "reindex" and r["job"]["args"] == {"vectors": False}
    assert rt.wait_jobs(5) and seen == [False]
    assert client.put("/api/glossary", json={}).json() == {"ok": True, "terms": 0} and p.read_text(encoding="utf-8") == ""


# --- logs / devices / push ------------------------------------------------------------------------------------------

def test_logs_endpoint(client: TestClient, rt: Runtime):
    p = rt.cfg.data_dir / "audio_note.log"
    assert client.get("/api/logs").json() == {"path": str(p), "lines": [], "size": 0}
    p.write_text("".join(f"line {i} 中文\n" for i in range(300)), encoding="utf-8")
    out = client.get("/api/logs").json()
    assert len(out["lines"]) == 200 and out["lines"][-1] == "line 299 中文" and out["lines"][0] == "line 100 中文"
    assert out["size"] == p.stat().st_size
    assert len(client.get("/api/logs", params={"lines": 10}).json()["lines"]) == 10
    assert len(client.get("/api/logs", params={"lines": 1}).json()["lines"]) == 10          # floor
    assert len(client.get("/api/logs", params={"lines": 9999}).json()["lines"]) == 300
    cli = client.get("/api/logs", params={"which": "cli"}).json()
    assert cli["path"] == str(rt.cfg.data_dir / "audio_note.cli.log") and cli["lines"] == []


def test_devices_endpoint_returns_list(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    import audio_note.audio.capture as capture

    assert isinstance(client.get("/api/devices").json(), list)
    dev = SimpleNamespace(index=3, name="Fake Mic", hostapi="WASAPI", channels=2, default_sr=48000.0)
    monkeypatch.setattr(capture, "list_input_devices", lambda: [dev])
    assert client.get("/api/devices").json() == [{"index": 3, "name": "Fake Mic", "hostapi": "WASAPI", "channels": 2, "default_sr": 48000.0}]

    def broken():
        raise RuntimeError("PortAudio not initialized")

    monkeypatch.setattr(capture, "list_input_devices", broken)
    assert client.get("/api/devices").json() == []


def test_push_test_endpoint(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    import sys
    import types

    rt.cfg["push"]["channels"] = []
    r = client.post("/api/push/test", json={"text": "hi"})
    assert r.status_code == 400 and "推送渠道" in r.json()["error"]
    sent: list = []
    results = {"toast": None, "feishu": "HTTP 500"}

    class Message:
        def __init__(self, title, body_md, url=None):
            self.title, self.body_md = title, body_md

    class Notifier:
        def __init__(self, cfg):
            self.channels = ["toast", "feishu"]

        def send(self, msg):
            sent.append(msg)
            return dict(results)

    mod = types.ModuleType("audio_note.push.base")
    mod.Message, mod.Notifier = Message, Notifier
    monkeypatch.setitem(sys.modules, "audio_note.push.base", mod)
    r = client.post("/api/push/test", json={"text": "你好"})
    assert r.status_code == 200 and r.json() == {"ok": True, "channels": results}
    assert sent[0].title == "audio-note 测试" and sent[0].body_md == "你好"
    results["toast"] = "no toast"
    assert client.post("/api/push/test").json()["ok"] is False
    assert sent[1].body_md.startswith("这是一条来自 audio-note 的测试消息")


# --- import ---------------------------------------------------------------------------------------------------------

@needs_sample
def test_import_wav_creates_segments_with_real_vad(client: TestClient, rt: Runtime):
    data = _wav_bytes(_sample_slice(30, 60))
    r = client.post("/api/import", params={"filename": "../会议 录音.wav", "start": "2026-10-30 09:00"}, content=data)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["ok"] is True and out["file"] == "会议_录音.wav" and out["segments"] >= 1
    assert (rt.cfg.data_dir / "imports" / "会议_录音.wav").read_bytes() == data
    rows = rt.db.query("SELECT started_at, source, origin, status, speech_s FROM segments ORDER BY started_at")
    assert len(rows) == out["segments"] and all(r["source"] == "import" and r["status"] == "new" for r in rows)
    assert rows[0]["started_at"].startswith("2026-10-30T09:00") and rows[0]["speech_s"] > 1.0
    assert rows[0]["origin"].endswith("会议_录音.wav")
    assert client.get("/api/status").json()["pending"] == out["segments"]
    # same name again: the upload is kept under a numbered name
    r2 = client.post("/api/import", params={"filename": "会议 录音.wav", "start": "2026-10-30 10:00"}, content=data)
    assert r2.status_code == 200 and r2.json()["file"] == "会议_录音_1.wav"


def test_import_rejections(client: TestClient, rt: Runtime):
    r = client.post("/api/import", content=b"RIFF")
    assert r.status_code == 400 and "没有收到音频数据" in r.json()["error"]
    r = client.post("/api/import", params={"filename": "a.wav", "start": "昨天"}, content=_wav_bytes(_tone(2.0)))
    assert r.status_code == 400 and "开始时间格式" in r.json()["error"]
    assert rt.db.query("SELECT id FROM segments") == []


# --- jobs ------------------------------------------------------------------------------------------------------------

def test_jobs_endpoints(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    r = client.post("/api/jobs/frobnicate")
    assert r.status_code == 404 and r.json() == {"error": "未知任务 frobnicate"}
    assert set(client.get("/api/jobs").json()) == set(JOBS)
    seen: dict = {}
    monkeypatch.setattr(rt, "_job_cleanup", lambda: {"expired": 2})
    monkeypatch.setattr(rt, "_job_process", lambda retry_failed=False, limit=None: seen.update(retry_failed=retry_failed, limit=limit) or {"done": 1})
    r = client.post("/api/jobs/cleanup")
    assert r.status_code == 200
    job = r.json()["job"]
    assert r.json()["ok"] is True and job["name"] == "cleanup" and job["label"] == "清理音频" and job["started_at"]
    assert rt.wait_jobs(5)
    assert client.get("/api/jobs").json()["cleanup"]["result"] == {"expired": 2}
    r = client.post("/api/jobs/process", json={"retry_failed": True, "limit": 3})
    assert r.status_code == 200 and r.json()["job"]["args"] == {"retry_failed": True, "limit": 3}
    assert rt.wait_jobs(5) and seen == {"retry_failed": True, "limit": 3}
    assert client.get("/api/jobs").json()["process"]["result"] == {"done": 1}
    # argument validation happens before anything is queued
    assert client.post("/api/jobs/daily", json={"day": "bad"}).status_code == 400
    assert client.post("/api/jobs/push", json={}).status_code == 400
    assert client.post("/api/jobs/relabel", json={"day": "2026-13-01x"}).status_code == 400
    assert not any(j["running"] for j in client.get("/api/jobs").json().values())


def test_job_already_running_and_read_only_are_409(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    import threading

    from audio_note import service

    release = threading.Event()
    monkeypatch.setattr(rt, "_job_cleanup", lambda: release.wait(10) or {})
    assert client.post("/api/jobs/cleanup").status_code == 200
    r = client.post("/api/jobs/cleanup")
    assert r.status_code == 409 and "已在运行中" in r.json()["error"]
    release.set()
    assert rt.wait_jobs(5)
    # another process owns the service: jobs and recording are refused, browsing still works
    service.lock_path(rt.cfg).write_text("424242\n2026-10-31T09:00:00\n1.0\n", encoding="utf-8")
    monkeypatch.setattr(service, "_same_process", lambda pid, ctime: pid == 424242)
    r = client.post("/api/jobs/cleanup")
    assert r.status_code == 409 and "424242" in r.json()["error"]
    r = client.post("/api/service/start", json={"record": False})
    assert r.status_code == 409 and "已在运行" in r.json()["error"]
    assert client.post("/api/recording/start").status_code == 409
    st = client.get("/api/status").json()["service"]
    assert st["read_only"] is True and st["other_pid"] == 424242
    assert client.get("/api/days").status_code == 200


# --- recording / service ------------------------------------------------------------------------------------------------

def _fake_recorder_loop(cfg, db, stop_event, ref):
    ref["recorder"] = SimpleNamespace(is_speaking=False, segments_written=1, speech_total_s=2.0, started_at=datetime.now())
    stop_event.wait(30)


def test_recording_start_and_stop(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rt_mod, "_recorder_loop", _fake_recorder_loop)
    r = client.post("/api/recording/start")
    assert r.status_code == 200 and r.json() == {"ok": True, "recording": True, "last_error": None}
    st = client.get("/api/status").json()["service"]
    assert st["running"] is True and st["scheduler"] is True and st["recording"] is True
    assert st["segments_written"] == 1 and st["speech_total_s"] == 2.0 and st["recorder_started_at"] and st["is_speaking"] is False
    assert client.post("/api/recording/start").json()["recording"] is True             # idempotent
    r = client.post("/api/recording/stop")
    assert r.status_code == 200 and r.json() == {"ok": True, "recording": False}
    st = client.get("/api/status").json()["service"]
    assert st["recording"] is False and st["running"] is True                            # service (lock + scheduler) stays
    assert client.post("/api/recording/stop").json()["recording"] is False


def test_recording_start_without_models_is_400(cfg_factory, tmp_path: Path):
    cfg = cfg_factory({"paths": {"models_dir": str(tmp_path / "no_models")}, "memory": {"retrieval": "fts", "embedding_model": "hash"}})
    r = Runtime(cfg)
    try:
        with TestClient(create_app(r), raise_server_exceptions=False) as c:
            assert set(c.get("/api/status").json()["models_missing"]) == {"vad_model", "embedding_model", "segmentation_model"}
            resp = c.post("/api/recording/start")
            assert resp.status_code == 400 and "模型文件缺失" in resp.json()["error"]
            assert not r.service_running() and not r.recording()
    finally:
        r.shutdown()
        from audio_note.logging_setup import _remove_own_handlers

        _remove_own_handlers(logging.getLogger())


def test_service_start_and_stop(client: TestClient, rt: Runtime):
    from audio_note import service

    r = client.post("/api/service/start", json={"record": False})
    assert r.status_code == 200
    svc = r.json()["service"]
    assert r.json()["ok"] is True and svc["running"] is True and svc["scheduler"] is True and svc["recording"] is False
    assert service.running_pid(rt.cfg) == os.getpid()
    r = client.post("/api/service/stop")
    assert r.status_code == 200 and r.json()["service"]["running"] is False and r.json()["service"]["scheduler"] is False
    assert service.running_pid(rt.cfg) is None


# --- UIServer ------------------------------------------------------------------------------------------------------------

def test_ui_server_serves_the_app_on_a_free_port(rt: Runtime):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert port_free("127.0.0.1", port)
    srv = UIServer(create_app(rt), "127.0.0.1", port)
    assert srv.url == f"http://127.0.0.1:{port}"
    url = srv.start()
    try:
        assert url == srv.url and srv._thread is not None and srv._thread.is_alive()
        assert not port_free("127.0.0.1", port)
        r = httpx.get(url + "/api/status", timeout=10)
        assert r.status_code == 200 and r.json()["pid"] == os.getpid()
        assert "audio-note" in httpx.get(url + "/", timeout=10).text
        with pytest.raises(RuntimeError, match="已被占用"):
            UIServer(create_app(rt), "127.0.0.1", port).start()
    finally:
        srv.stop()
    assert srv._thread is None
    assert UIServer(None, "0.0.0.0", 1).url == "http://localhost:1"


def test_port_free_is_false_for_a_bound_socket():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        assert port_free("127.0.0.1", port) is False
    assert port_free("127.0.0.1", port) is True
