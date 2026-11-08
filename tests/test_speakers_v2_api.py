"""Speakers v2 through the JSON API (spec §5): self / relation fields on /api/speakers, PATCH is_self / relation,
GET /api/voices + POST /api/voices/name, the whole_cluster / is_self options of POST /api/speakers/label, the
voices / relations jobs and the new /api/status and /api/config keys.

No network: the relations job is driven by a stubbed infer_relations; no microphone. One test runs the real
CAM++ engine on the sample recording (skipped when it is missing); the others use a fake engine that writes
real speaker_embeddings rows, so the DB side of the flow is exercised end-to-end.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import yaml
from fastapi.testclient import TestClient

from audio_note.analysis.relations import RELATION_CHOICES
from audio_note.audio.storage import segment_path
from audio_note.audio.vad import SAMPLE_RATE as SR
from audio_note.db import DB
from audio_note.runtime import Runtime
from audio_note.ui.app import create_app

from .conftest import SAMPLE_WAV

DAY = "2026-10-30"
DAY2 = "2026-10-31"
needs_sample = pytest.mark.skipif(not SAMPLE_WAV.exists(), reason="real sample recording not available")


# --- fixtures / helpers (same shape as tests/test_ui_api.py) ---------------------------------------------

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


def _tone(seconds: float, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _sample_slice(a: float, b: float) -> np.ndarray:
    audio, sr = sf.read(str(SAMPLE_WAV), dtype="float32")
    assert sr == SR
    return audio[int(a * SR):int(b * SR)]


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
         start_s: float = 0.0, end_s: float = 1.0, score: float | None = None, cluster: int | None = 0) -> int:
    return db.add_utterances([{
        "segment_id": seg, "start_s": start_s, "end_s": end_s, "abs_start": f"{day}T{hhmm}:00", "day": day,
        "cluster_label": cluster, "speaker_id": speaker_id, "speaker_name": who, "speaker_score": score,
        "text_raw": text, "text": text,
    }])[0]


def _speaker_with_samples(rt: Runtime, name: str, n: int = 1, wav: Path | None = None) -> int:
    sid = rt.db.upsert_speaker(name)
    rng = np.random.default_rng(hash(name) % 1000)
    base = rng.normal(size=16).astype(np.float32)
    for i in range(n):
        rt.db.add_speaker_embedding(sid, base + 0.05 * rng.normal(size=16).astype(np.float32), str(wav) if wav else None, 5.0 + i)
    return sid


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _voice(seed: int, dim: int = 192) -> np.ndarray:
    return _unit(np.random.default_rng(seed).standard_normal(dim))


def _near(base: np.ndarray, seed: int, noise: float = 0.03) -> np.ndarray:
    return _unit(base + noise * np.random.default_rng(seed).standard_normal(base.shape[0]))


class FakeEngine:
    """SpeakerEngine stand-in for the API: writes real speaker_embeddings rows, identifies nothing."""

    def __init__(self, too_short: bool = False):
        self.too_short = too_short
        self.calls: list[tuple] = []

    def enroll_from_audio(self, db, name, audio, source=None, copy_as_source=False, **kw):
        self.calls.append(("enroll_from_audio", name, source, round(len(audio) / SR, 1)))
        if self.too_short:
            raise ValueError("选中的范围只有 0.3s 语音（至少需要 1s），未注册")
        sid = db.upsert_speaker(name)
        db.add_speaker_embedding(sid, _voice(hash(name) % 97), source, len(audio) / SR)
        return sid

    def enroll_from_segment_turns(self, db, name, seg_path, turns, **kw):
        self.calls.append(("label", name, Path(seg_path), turns))
        if self.too_short:
            raise ValueError("选中的范围只有 0.3s 语音（至少需要 1s），未注册")
        sid = db.upsert_speaker(name)
        db.add_speaker_embedding(sid, _voice(hash(name) % 97), f"{seg_path}#{turns[0][0]:.1f}-{turns[-1][1]:.1f}", 2.0)
        return sid

    def cluster_embedding(self, audio, turns):
        return _voice(1)

    def identify_turns(self, audio, turns, db, segment_id=None):
        self.calls.append(("identify", segment_id, len(turns)))
        return turns                                          # nobody recognised


@pytest.fixture
def fake_engine(rt: Runtime, monkeypatch: pytest.MonkeyPatch) -> FakeEngine:
    eng = FakeEngine()
    monkeypatch.setattr(rt, "engine", lambda: eng)
    return eng


def _two_unknown_groups(rt: Runtime) -> dict[str, Any]:
    """Two recordings on two days with synthetic cluster embeddings: voice A in both (should merge), voice B once."""
    db, cfg = rt.db, rt.cfg
    a, b = _voice(11), _voice(22)
    seg1, p1 = _seg(cfg, db, datetime(2026, 10, 30, 9, 0), audio=_tone(10.0))
    seg2, p2 = _seg(cfg, db, datetime(2026, 10, 31, 10, 0), audio=_tone(10.0))
    u = {
        "a1": _utt(db, seg1, DAY, "09:00", "早上好", None, None, 0.0, 3.0, cluster=0),
        "a2": _utt(db, seg1, DAY, "09:00", "今天开会", None, None, 3.0, 5.0, cluster=0),
        "b1": _utt(db, seg1, DAY, "09:01", "好的", None, None, 5.0, 6.0, cluster=1),
        "a3": _utt(db, seg2, DAY2, "10:00", "明天再说", None, None, 0.0, 4.0, cluster=0),
    }
    db.upsert_cluster_embedding(seg1, 0, a, 5.0)
    db.upsert_cluster_embedding(seg1, 1, b, 1.0)
    db.upsert_cluster_embedding(seg2, 0, _near(a, 3), 4.0)
    return {"seg1": seg1, "seg2": seg2, "p1": p1, "p2": p2, "utts": u}


# --- GET /api/speakers, PATCH is_self / relation ----------------------------------------------------------

def test_speakers_list_has_self_and_relation_fields(client: TestClient, rt: Runtime):
    me = _speaker_with_samples(rt, "张伟")
    other = _speaker_with_samples(rt, "王敏", 2)
    r = client.get("/api/speakers").json()
    assert r["self_id"] is None and r["relation_choices"] == list(RELATION_CHOICES) and "同事" in r["relation_choices"]
    by_name = {s["name"]: s for s in r["speakers"]}
    for s in by_name.values():
        assert s["is_self"] is False and s["relation"] is None and s["relation_source"] is None and s["relation_note"] is None
    rt.db.set_self_speaker(me)
    rt.db.set_relation(other, "同事", "auto", "叫他王总")
    r = client.get("/api/speakers").json()
    assert r["self_id"] == me
    by_name = {s["name"]: s for s in r["speakers"]}
    assert by_name["张伟"]["is_self"] is True and by_name["张伟"]["samples"] == 1
    assert by_name["王敏"] == {**by_name["王敏"], "is_self": False, "relation": "同事", "relation_source": "auto",
                               "relation_note": "叫他王总", "samples": 2}


def test_patch_is_self_is_unique_and_clearable(client: TestClient, rt: Runtime):
    a = _speaker_with_samples(rt, "甲")
    b = _speaker_with_samples(rt, "乙")
    r = client.patch(f"/api/speakers/{a}", json={"is_self": True})
    assert r.status_code == 200 and r.json() == {"ok": True, "renamed_days": 0}
    assert rt.db.self_speaker()["id"] == a and client.get("/api/speakers").json()["self_id"] == a
    # the second "self" replaces the first: never two rows with is_self=1
    assert client.patch(f"/api/speakers/{b}", json={"is_self": True}).status_code == 200
    assert [r["id"] for r in rt.db.query("SELECT id FROM speakers WHERE is_self=1")] == [b]
    assert client.get("/api/status").json()["self_name"] == "乙"
    # false on somebody who is not self changes nothing; false on self clears it
    assert client.patch(f"/api/speakers/{a}", json={"is_self": False}).status_code == 200
    assert rt.db.self_speaker()["id"] == b
    assert client.patch(f"/api/speakers/{b}", json={"is_self": False}).status_code == 200
    assert rt.db.self_speaker() is None and client.get("/api/speakers").json()["self_id"] is None
    assert client.patch("/api/speakers/999", json={"is_self": True}).status_code == 404


def test_patch_relation_and_note(client: TestClient, rt: Runtime):
    sid = _speaker_with_samples(rt, "李强")
    rt.db.set_relation(sid, "同学", "auto", "推断：一起上课")
    r = client.patch(f"/api/speakers/{sid}", json={"relation": " 朋友 ", "relation_note": "大学室友"})
    assert r.status_code == 200
    row = rt.db.one("SELECT relation, relation_source, relation_note, relation_updated_at FROM speakers WHERE id=?", (sid,))
    assert tuple(row)[:3] == ("朋友", "user", "大学室友") and row["relation_updated_at"]
    # a relation without a note replaces the inferred reason; the note alone only touches the note
    assert client.patch(f"/api/speakers/{sid}", json={"relation": "好朋友"}).status_code == 200
    assert tuple(rt.db.one("SELECT relation, relation_source, relation_note FROM speakers WHERE id=?", (sid,))) == ("好朋友", "user", None)
    assert client.patch(f"/api/speakers/{sid}", json={"relation_note": "周末常约"}).status_code == 200
    assert tuple(rt.db.one("SELECT relation, relation_source, relation_note FROM speakers WHERE id=?", (sid,))) == ("好朋友", "user", "周末常约")
    # a custom relation (not in the choices) is allowed; "" clears all three columns
    assert client.patch(f"/api/speakers/{sid}", json={"relation": "表哥"}).status_code == 200
    assert rt.db.one("SELECT relation FROM speakers WHERE id=?", (sid,))["relation"] == "表哥"
    assert client.patch(f"/api/speakers/{sid}", json={"relation": ""}).status_code == 200
    row = rt.db.one("SELECT relation, relation_source, relation_note, relation_updated_at FROM speakers WHERE id=?", (sid,))
    assert tuple(row) == (None, None, None, None)
    # name / notes handling is unchanged when combined with the new fields
    r = client.patch(f"/api/speakers/{sid}", json={"name": "李强强", "relation": "同事", "notes": "备注"})
    assert r.status_code == 200 and r.json() == {"ok": True, "renamed_days": 0}
    assert tuple(rt.db.one("SELECT name, notes, relation FROM speakers WHERE id=?", (sid,))) == ("李强强", "备注", "同事")


# --- GET /api/voices --------------------------------------------------------------------------------------

def test_voices_empty_db(client: TestClient):
    r = client.get("/api/voices")
    assert r.status_code == 200
    assert r.json() == {"groups": [], "missing_segments": 0, "threshold": 0.55, "unknown_utterances": 0, "scanning": False}


def test_voices_groups_synthetic_embeddings_and_reports_scanning(client: TestClient, rt: Runtime):
    d = _two_unknown_groups(rt)
    _seg(rt.cfg, rt.db, datetime(2026, 10, 29, 8, 0), audio=_tone(3.0))          # done, nothing unidentified: not "missing"
    gone, _ = _seg(rt.cfg, rt.db, datetime(2026, 10, 29, 9, 0), audio=_tone(3.0))
    _utt(rt.db, gone, "2026-10-29", "09:00", "没扫描过", None, None, 0.0, 2.0)      # unidentified, no cluster embedding yet
    out = client.get("/api/voices").json()
    assert out["missing_segments"] == 1 and out["unknown_utterances"] == 5 and out["scanning"] is False
    assert len(out["groups"]) == 2
    big, small = out["groups"]
    assert big["members"] == [[d["seg1"], 0], [d["seg2"], 0]] and big["utterances"] == 3 and big["speech_s"] == 9.0
    assert big["days"] == [DAY, DAY2] and big["n_days"] == 2 and big["segments"] == 2 and big["closest"] is None
    assert [s["utterance_id"] for s in big["samples"]] == [d["utts"]["a3"], d["utts"]["a1"], d["utts"]["a2"]]   # longest first
    assert big["samples"][0] == {"utterance_id": d["utts"]["a3"], "day": DAY2, "time": "10:00", "text": "明天再说", "seconds": 4.0, "audio": True}
    assert small["members"] == [[d["seg1"], 1]] and small["utterances"] == 1 and small["key"] != big["key"]
    # the "scanning" flag mirrors the voices job
    rt.jobs["voices"].running = True
    try:
        assert client.get("/api/voices").json()["scanning"] is True
    finally:
        rt.jobs["voices"].running = False
    # an enrolled speaker close to a group shows up as "closest"
    sid = rt.db.upsert_speaker("张三")
    rt.db.add_speaker_embedding(sid, _near(_voice(22), 9), None, 4.0)
    small = client.get("/api/voices").json()["groups"][1]
    assert small["closest"]["id"] == sid and small["closest"]["name"] == "张三" and small["closest"]["score"] > 0.8


# --- POST /api/voices/name --------------------------------------------------------------------------------

def test_voices_name_assigns_group_and_starts_relabel(client: TestClient, rt: Runtime, fake_engine: FakeEngine, monkeypatch: pytest.MonkeyPatch):
    d = _two_unknown_groups(rt)
    relabel_calls: list[dict] = []
    monkeypatch.setattr(rt, "_job_relabel", lambda day=None, segment_id=None: relabel_calls.append({"day": day, "segment_id": segment_id}) or {"segments": 0})
    big = client.get("/api/voices").json()["groups"][0]
    r = client.post("/api/voices/name", json={"members": big["members"], "name": " 张伟 ", "is_self": True, "relation": "同事"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["ok"] is True and out["name"] == "张伟" and out["utterances_assigned"] == 3 and out["samples_added"] == 2
    assert out["segments"] == [d["seg1"], d["seg2"]] and out["days"] == [DAY, DAY2] and out["relabel_started"] is True
    assert "reindexed" in out
    sid = out["id"]
    # both members gave a sample (longest first), the source carries the segment path and the clip range
    assert [c[0] for c in fake_engine.calls] == ["enroll_from_audio", "enroll_from_audio"]
    assert fake_engine.calls[0][2] == f"{d['p1']}#0.0-5.0" and fake_engine.calls[1][2] == f"{d['p2']}#0.0-4.0"
    rows = {r["id"]: r for r in rt.db.query("SELECT id, speaker_id, speaker_name, speaker_score, speaker_source FROM utterances")}
    for k in ("a1", "a2", "a3"):
        u = rows[d["utts"][k]]
        assert (u["speaker_id"], u["speaker_name"], u["speaker_score"], u["speaker_source"]) == (sid, "张伟", None, "user")
    assert rows[d["utts"]["b1"]]["speaker_id"] is None and rows[d["utts"]["b1"]]["speaker_source"] is None
    me = rt.db.self_speaker()
    assert me["id"] == sid and me["relation"] == "同事" and me["relation_source"] == "user"
    assert (rt.cfg.transcripts_dir / f"{DAY}.md").exists() and "张伟" in (rt.cfg.transcripts_dir / f"{DAY2}.md").read_text(encoding="utf-8")
    assert rt.wait_jobs(5) and relabel_calls == [{"day": None, "segment_id": None}]       # the other recordings get re-identified
    st = client.get("/api/status").json()
    assert st["self_name"] == "张伟" and st["unknown_utterances"] == 1
    # the last group: nothing unidentified is left afterwards, so no relabel is started; auto_relabel=false never starts one
    small = client.get("/api/voices").json()["groups"][0]
    assert small["members"] == [[d["seg1"], 1]]
    r = client.post("/api/voices/name", json={"members": small["members"], "name": "王敏", "auto_relabel": False})
    assert r.status_code == 200 and r.json()["relabel_started"] is False and r.json()["utterances_assigned"] == 1
    assert client.get("/api/voices").json()["groups"] == [] and client.get("/api/speakers").json()["unknown_utterances"] == 0
    assert rt.wait_jobs(5) and len(relabel_calls) == 1
    assert rt.db.self_speaker()["id"] == sid                                                # naming somebody else keeps "self"


def test_voices_name_relabel_is_skipped_when_a_job_blocks_it(client: TestClient, rt: Runtime, fake_engine: FakeEngine, monkeypatch: pytest.MonkeyPatch):
    import threading

    d = _two_unknown_groups(rt)
    release = threading.Event()
    monkeypatch.setattr(rt, "_job_relabel", lambda day=None, segment_id=None: release.wait(10) or {})
    rt.submit("relabel")                                       # already running -> JobError on the next submit, ignored
    try:
        r = client.post("/api/voices/name", json={"members": [[d["seg1"], 1]], "name": "王敏"})
        assert r.status_code == 200 and r.json()["relabel_started"] is False and r.json()["utterances_assigned"] == 1
    finally:
        release.set()
    assert rt.wait_jobs(5)


def test_voices_name_errors(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    d = _two_unknown_groups(rt)
    post = lambda body: client.post("/api/voices/name", json=body)  # noqa: E731
    r = post({"members": [[d["seg1"], 0]], "name": "  "})
    assert r.status_code == 400 and "姓名" in r.json()["error"]
    r = post({"members": [], "name": "甲"})
    assert r.status_code == 400 and "members" in r.json()["error"]
    assert post({"name": "甲"}).status_code == 400
    assert post({"members": "1,0", "name": "甲"}).status_code == 400
    r = post({"members": [["x", 0]], "name": "甲"})
    assert r.status_code == 400 and "members" in r.json()["error"]
    assert post({"members": [[1]], "name": "甲"}).status_code == 400
    # too little speech in a brand-new speaker's group: 400 and no empty speaker row is left behind
    monkeypatch.setattr(rt, "engine", lambda: FakeEngine(too_short=True))
    r = post({"members": [[d["seg1"], 1]], "name": "新人"})
    assert r.status_code == 400 and "语音太短" in r.json()["error"]
    assert rt.db.one("SELECT COUNT(*) AS n FROM speakers")["n"] == 0
    assert rt.db.one("SELECT speaker_id FROM utterances WHERE id=?", (d["utts"]["b1"],))["speaker_id"] is None
    assert rt.wait_jobs(5) and not any(j["running"] for j in client.get("/api/jobs").json().values())


@needs_sample
def test_voices_scan_and_name_with_real_engine(client: TestClient, rt: Runtime):
    seg, p = _seg(rt.cfg, rt.db, datetime(2026, 10, 30, 9, 0), audio=_sample_slice(29, 37))
    u1 = _utt(rt.db, seg, DAY, "09:00", "大家好我是雷军", None, None, 0.0, 4.0, cluster=0)
    u2 = _utt(rt.db, seg, DAY, "09:00", "今天发布新产品", None, None, 4.0, 8.0, cluster=0)
    before = client.get("/api/voices").json()
    assert before["groups"] == [] and before["missing_segments"] == 1 and before["unknown_utterances"] == 2
    r = client.post("/api/jobs/voices")
    assert r.status_code == 200 and r.json()["job"]["name"] == "voices" and r.json()["job"]["args"] == {}
    assert rt.wait_jobs(60)
    job = client.get("/api/jobs").json()["voices"]
    assert job["error"] is None and job["result"] == {"segments": 1, "clusters": 1, "skipped": 0}
    out = client.get("/api/voices").json()
    assert out["missing_segments"] == 0 and out["scanning"] is False and len(out["groups"]) == 1
    g = out["groups"][0]
    assert g["members"] == [[seg, 0]] and g["utterances"] == 2 and {s["utterance_id"] for s in g["samples"]} == {u1, u2}
    assert all(s["audio"] for s in g["samples"])
    r = client.post("/api/voices/name", json={"members": g["members"], "name": "雷军", "is_self": True, "relation": "导师"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["samples_added"] == 1 and out["utterances_assigned"] == 2 and out["relabel_started"] is False
    emb = rt.db.query("SELECT speaker_id, source_path, duration_s, dim FROM speaker_embeddings")
    assert len(emb) == 1 and emb[0]["speaker_id"] == out["id"] and emb[0]["dim"] == 192
    assert Path(emb[0]["source_path"]).exists() and rt.cfg.speakers_dir in Path(emb[0]["source_path"]).parents   # speech-only copy
    assert 1.0 <= emb[0]["duration_s"] <= 8.0
    utts = rt.db.query("SELECT speaker_name, speaker_source FROM utterances ORDER BY id")
    assert [tuple(u) for u in utts] == [("雷军", "user"), ("雷军", "user")]
    st = client.get("/api/status").json()
    assert st["self_name"] == "雷军" and st["unknown_utterances"] == 0
    sp = client.get("/api/speakers").json()
    assert sp["self_id"] == out["id"] and sp["speakers"][0]["relation"] == "导师" and sp["speakers"][0]["relation_source"] == "user"
    assert client.get("/api/voices").json()["groups"] == []


# --- POST /api/speakers/label: whole_cluster / is_self / relation -----------------------------------------

def test_label_whole_cluster_assigns_same_cluster_and_sets_self(client: TestClient, rt: Runtime, fake_engine: FakeEngine):
    seg, p = _seg(rt.cfg, rt.db, datetime(2026, 10, 30, 9, 0), audio=_tone(10.0))
    u1 = _utt(rt.db, seg, DAY, "09:00", "第一句", None, None, 0.0, 2.0, cluster=0)
    u2 = _utt(rt.db, seg, DAY, "09:00", "第二句", None, None, 2.0, 4.0, cluster=0)
    u3 = _utt(rt.db, seg, DAY, "09:00", "别人说的", None, None, 4.0, 5.0, cluster=1)
    other_seg, _ = _seg(rt.cfg, rt.db, datetime(2026, 10, 30, 10, 0), audio=_tone(5.0))
    u4 = _utt(rt.db, other_seg, DAY, "10:00", "另一段录音", None, None, 0.0, 1.0, cluster=0)      # same label, other segment
    r = client.post("/api/speakers/label", json={"name": "张伟", "utterance_id": u1, "is_self": True, "relation": "同事"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["assigned"] == 2 and out["samples"] == 1 and out["segment_id"] == seg and out["relabeled_changed"] == 0
    assert "relabel_error" not in out
    sid = out["id"]
    rows = {r["id"]: r for r in rt.db.query("SELECT id, speaker_id, speaker_name, speaker_score, speaker_source FROM utterances")}
    for uid in (u1, u2):
        assert (rows[uid]["speaker_id"], rows[uid]["speaker_name"], rows[uid]["speaker_score"], rows[uid]["speaker_source"]) == (sid, "张伟", None, "user")
    assert rows[u3]["speaker_id"] is None and rows[u3]["speaker_source"] is None
    assert rows[u4]["speaker_id"] is None
    assert fake_engine.calls[0] == ("label", "张伟", p, [(0.0, 2.0)])
    assert ("identify", seg, 3) in fake_engine.calls                                 # relabel ran on the segment, skipping the user rows
    me = rt.db.self_speaker()
    assert me["id"] == sid and me["relation"] == "同事" and me["relation_source"] == "user"
    assert "张伟" in (rt.cfg.transcripts_dir / f"{DAY}.md").read_text(encoding="utf-8")
    assert client.get("/api/status").json()["unknown_utterances"] == 2
    # whole_cluster=false labels just the one utterance; is_self / relation absent keep the existing flags
    r = client.post("/api/speakers/label", json={"name": "王敏", "utterance_id": u3, "whole_cluster": False})
    assert r.status_code == 200 and r.json()["assigned"] == 1
    assert rt.db.one("SELECT speaker_name, speaker_source FROM utterances WHERE id=?", (u3,))[:] == ("王敏", "user")
    assert rt.db.self_speaker()["id"] == sid and rt.db.one("SELECT relation FROM speakers WHERE name='王敏'")["relation"] is None
    # the segment + from/to form labels the utterances the range covers (with their cluster mates)
    u5 = _utt(rt.db, other_seg, DAY, "10:00", "再一句", None, None, 1.0, 2.0, cluster=0)
    u6 = _utt(rt.db, other_seg, DAY, "10:00", "第三人", None, None, 2.0, 3.0, cluster=2)
    r = client.post("/api/speakers/label", json={"name": "张伟", "segment_id": other_seg, "from": 0.0, "to": 1.5})
    assert r.status_code == 200 and r.json()["assigned"] == 2 and r.json()["id"] == sid
    assert [tuple(x) for x in rt.db.query("SELECT id, speaker_id FROM utterances WHERE segment_id=? ORDER BY id", (other_seg,))] == \
        [(u4, sid), (u5, sid), (u6, None)]


def test_label_too_short_leaves_nothing_assigned(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rt, "engine", lambda: FakeEngine(too_short=True))
    seg, _ = _seg(rt.cfg, rt.db, datetime(2026, 10, 30, 9, 0), audio=_tone(5.0))
    uid = _utt(rt.db, seg, DAY, "09:00", "嗯", None, None, 0.0, 0.3)
    r = client.post("/api/speakers/label", json={"name": "甲", "utterance_id": uid, "is_self": True})
    assert r.status_code == 400 and "语音" in r.json()["error"]
    assert rt.db.one("SELECT speaker_id, speaker_source FROM utterances WHERE id=?", (uid,))[:] == (None, None)
    assert rt.db.self_speaker() is None


# --- jobs: voices / relations ------------------------------------------------------------------------------

def test_jobs_voices_and_relations_submission(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    from audio_note.analysis import relations

    monkeypatch.setattr(rt, "_job_voices", lambda: {"segments": 2, "clusters": 3, "skipped": 0})
    r = client.post("/api/jobs/voices", json={"anything": 1})
    assert r.status_code == 200 and r.json()["job"]["name"] == "voices" and r.json()["job"]["label"] == "扫描未识别的声音"
    assert r.json()["job"]["args"] == {}
    assert rt.wait_jobs(5) and client.get("/api/jobs").json()["voices"]["result"] == {"segments": 2, "clusters": 3, "skipped": 0}

    # relations: the LLM is stubbed (no key is configured, so the runtime client must be replaced too)
    seen: list[dict] = []

    def fake_infer(cfg, db, client_, speaker_ids=None, force=False):
        seen.append({"speaker_ids": speaker_ids, "force": force})
        return [{"id": 1, "name": "王敏", "relation": "同事", "confidence": 0.8, "reason": "叫他王总", "applied": True, "skipped": None},
                {"id": 2, "name": "李强", "relation": None, "confidence": None, "reason": None, "applied": False, "skipped": "语句太少"}]

    monkeypatch.setattr(relations, "infer_relations", fake_infer)
    monkeypatch.setattr(rt, "client", lambda: object())
    r = client.post("/api/jobs/relations", json={})
    assert r.status_code == 200 and r.json()["job"]["label"] == "推断关系" and r.json()["job"]["args"] == {"force": False}
    assert rt.wait_jobs(5)
    res = client.get("/api/jobs").json()["relations"]["result"]
    assert res["speakers"] == 2 and res["applied"] == 1 and res["results"][0]["name"] == "王敏"
    r = client.post("/api/jobs/relations", json={"speaker_id": "7", "force": True})
    assert r.status_code == 200 and r.json()["job"]["args"] == {"speaker_id": 7, "force": True}
    assert rt.wait_jobs(5) and seen == [{"speaker_ids": None, "force": False}, {"speaker_ids": [7], "force": True}]
    # argument validation happens before anything is queued
    for bad in ("abc", 0, -1, True, [1]):
        r = client.post("/api/jobs/relations", json={"speaker_id": bad})
        assert r.status_code == 400 and "speaker_id" in r.json()["error"], bad
    assert len(seen) == 2 and not any(j["running"] for j in client.get("/api/jobs").json().values())


def test_relations_job_without_api_key_fails_in_the_job(client: TestClient, rt: Runtime):
    assert client.post("/api/jobs/relations").status_code == 200
    assert rt.wait_jobs(5)
    job = client.get("/api/jobs").json()["relations"]
    assert job["running"] is False and job["error"] and "API key" in job["error"]


# --- status / config ---------------------------------------------------------------------------------------

def test_status_has_self_name_and_unknown_count(client: TestClient, rt: Runtime):
    st = client.get("/api/status").json()
    assert st["self_name"] is None and st["unknown_utterances"] == 0
    seg, _ = _seg(rt.cfg, rt.db, datetime(2026, 10, 30, 9, 0))
    sid = _speaker_with_samples(rt, "张伟")
    _utt(rt.db, seg, DAY, "09:00", "我说的", "张伟", sid)
    _utt(rt.db, seg, DAY, "09:01", "谁说的", None, None, 1.0, 2.0)
    _utt(rt.db, seg, DAY, "09:02", "不知道", None, None, 2.0, 3.0)
    rt.db.set_self_speaker(sid)
    st = client.get("/api/status").json()
    assert st["self_name"] == "张伟" and st["unknown_utterances"] == 2


def test_config_exposes_and_validates_speakers_v2_keys(client: TestClient, rt: Runtime):
    rt.config_path.write_text(yaml.safe_dump({"paths": {"data_dir": str(rt.cfg.data_dir), "models_dir": str(rt.cfg.models_dir)},
                                              "mimo": {"api_key": ""}}, allow_unicode=True), encoding="utf-8")
    cfg = client.get("/api/config").json()
    spk = cfg["values"]["speakers"]
    assert spk["voice_group_threshold"] == 0.55 and spk["infer_relations"] is True
    assert spk["relation_min_utterances"] == 15 and spk["relation_min_confidence"] == 0.6
    assert cfg["defaults"]["speakers"]["voice_group_threshold"] == 0.55
    for body, needle in ((({"speakers": {"voice_group_threshold": 1.5}}), "0–1"),
                         (({"speakers": {"relation_min_utterances": 0}}), "1–1000"),
                         (({"speakers": {"relation_min_confidence": -0.1}}), "0–1"),
                         (({"speakers": {"infer_relations": "maybe"}}), "true/false")):
        r = client.put("/api/config", json=body)
        assert r.status_code == 400 and needle in r.json()["error"], (body, r.text)
    r = client.put("/api/config", json={"speakers": {"voice_group_threshold": "0.6", "infer_relations": False,
                                                     "relation_min_utterances": 20, "relation_min_confidence": 0.7}})
    assert r.status_code == 200, r.text
    assert r.json()["saved"] == ["speakers.infer_relations", "speakers.relation_min_confidence",
                                 "speakers.relation_min_utterances", "speakers.voice_group_threshold"]
    data = yaml.safe_load(rt.config_path.read_text(encoding="utf-8"))["speakers"]
    assert data == {"voice_group_threshold": 0.6, "infer_relations": False, "relation_min_utterances": 20, "relation_min_confidence": 0.7}
    assert rt.cfg.speakers.voice_group_threshold == 0.6 and client.get("/api/voices").json()["threshold"] == 0.6


# --- review fixes --------------------------------------------------------------------------------------------

def test_delete_speaker_releases_hand_labelled_rows_and_relation_cache(client: TestClient, rt: Runtime, fake_engine: FakeEngine):
    """Finding 1: after DELETE the rows lose their 'user' mark (the FK only nulls speaker_id), so grouping and
    relabel pick them up again, and the relation inference cache of the id is gone."""
    d = _two_unknown_groups(rt)
    big = client.get("/api/voices").json()["groups"][0]
    sid = client.post("/api/voices/name", json={"members": big["members"], "name": "张伟", "auto_relabel": False}).json()["id"]
    rt.db.set_kv(f"relation_inferred_n:{sid}", 3)
    r = client.delete(f"/api/speakers/{sid}")
    assert r.status_code == 200 and r.json()["unlabeled_days"] == 2
    rows = rt.db.query("SELECT speaker_id, speaker_name, speaker_score, speaker_source FROM utterances WHERE id IN (?,?,?)",
                       (d["utts"]["a1"], d["utts"]["a2"], d["utts"]["a3"]))
    assert len(rows) == 3 and all(tuple(u) == (None, "未知", None, None) for u in rows)
    assert rt.db.get_kv(f"relation_inferred_n:{sid}") is None
    assert len(client.get("/api/voices").json()["groups"]) == 2                 # released rows are grouped again
    other = _speaker_with_samples(rt, "王敏")
    from audio_note.speech.pipeline import Processor

    class Recognises(FakeEngine):
        def identify_turns(self, audio, turns, db, segment_id=None):
            for t in turns:
                t.speaker_id, t.speaker_name, t.score = other, "王敏", 0.8
            return turns

    assert Processor(rt.cfg, rt.db, engine=Recognises()).relabel_segment(d["seg1"]) == 3
    assert rt.db.one("SELECT speaker_name FROM utterances WHERE id=?", (d["utts"]["a1"],))["speaker_name"] == "王敏"
    assert rt.wait_jobs(5)


def test_label_whole_cluster_keeps_rows_hand_labelled_as_somebody_else(client: TestClient, rt: Runtime, fake_engine: FakeEngine):
    """Finding 3: the cluster expansion of /api/speakers/label skips rows the user labelled to another
    speaker (reported in skipped_user_labels); the explicitly chosen utterance is always relabelled."""
    seg, _ = _seg(rt.cfg, rt.db, datetime(2026, 10, 30, 9, 0), audio=_tone(10.0))
    u1 = _utt(rt.db, seg, DAY, "09:00", "第一句", None, None, 0.0, 2.0, cluster=0)
    u2 = _utt(rt.db, seg, DAY, "09:00", "第二句", None, None, 2.0, 4.0, cluster=0)
    u3 = _utt(rt.db, seg, DAY, "09:00", "张三亲口确认的", None, None, 4.0, 5.0, cluster=0)
    r = client.post("/api/speakers/label", json={"name": "张三", "utterance_id": u3, "whole_cluster": False})
    assert r.status_code == 200 and r.json()["assigned"] == 1 and r.json()["skipped_user_labels"] == 0
    zhang = r.json()["id"]
    r = client.post("/api/speakers/label", json={"name": "王敏", "utterance_id": u1})
    assert r.status_code == 200, r.text
    assert r.json()["assigned"] == 2 and r.json()["skipped_user_labels"] == 1
    rows = {x["id"]: x for x in rt.db.query("SELECT id, speaker_id, speaker_name, speaker_source FROM utterances")}
    assert rows[u1]["speaker_name"] == "王敏" and rows[u2]["speaker_name"] == "王敏" and rows[u2]["speaker_source"] == "user"
    assert (rows[u3]["speaker_id"], rows[u3]["speaker_source"]) == (zhang, "user")
    r = client.post("/api/speakers/label", json={"name": "王敏", "utterance_id": u3, "whole_cluster": False})
    assert r.status_code == 200 and r.json()["assigned"] == 1 and r.json()["skipped_user_labels"] == 0
    assert rt.db.one("SELECT speaker_name FROM utterances WHERE id=?", (u3,))["speaker_name"] == "王敏"
    assert rt.wait_jobs(5)


def test_patch_relation_clear_forgets_the_inference_cache(client: TestClient, rt: Runtime):
    """Finding 7b: clearing a relation drops kv relation_inferred_n:<sid>, so a manual 自动推断关系 asks again."""
    sid = _speaker_with_samples(rt, "王敏")
    rt.db.set_relation(sid, "同事", "auto", note="推断")
    rt.db.set_kv(f"relation_inferred_n:{sid}", 20)
    assert client.patch(f"/api/speakers/{sid}", json={"relation": "朋友"}).status_code == 200
    assert rt.db.get_kv(f"relation_inferred_n:{sid}") == 20                    # a label keeps the cache
    assert client.patch(f"/api/speakers/{sid}", json={"relation": ""}).status_code == 200
    assert rt.db.get_kv(f"relation_inferred_n:{sid}") is None
    assert tuple(rt.db.one("SELECT relation, relation_source FROM speakers WHERE id=?", (sid,))) == (None, None)


def test_voices_name_and_label_hold_the_busy_counter(client: TestClient, rt: Runtime, fake_engine: FakeEngine, monkeypatch: pytest.MonkeyPatch):
    """Finding 10: the heavy request handlers run inside Runtime.busy(), which a profile switch respects;
    once a switch claimed the runtime they are refused with 409 instead of running on a closing database."""
    from audio_note.speech import voices

    d = _two_unknown_groups(rt)
    seen: list[int] = []
    real = voices.name_voice
    monkeypatch.setattr(voices, "name_voice", lambda *a, **k: (seen.append(rt.request_busy()), real(*a, **k))[1])
    assert rt.request_busy() == 0
    r = client.post("/api/voices/name", json={"members": [[d["seg1"], 1]], "name": "王敏", "auto_relabel": False})
    assert r.status_code == 200, r.text
    assert seen == [1] and rt.request_busy() == 0
    orig = fake_engine.enroll_from_segment_turns
    monkeypatch.setattr(fake_engine, "enroll_from_segment_turns",
                        lambda db, name, seg_path, turns, **kw: (seen.append(rt.request_busy()), orig(db, name, seg_path, turns, **kw))[1])
    r = client.post("/api/speakers/label", json={"name": "张伟", "utterance_id": d["utts"]["a1"]})
    assert r.status_code == 200, r.text
    assert seen == [1, 1] and rt.request_busy() == 0
    rt.begin_switch()
    try:
        r = client.post("/api/voices/name", json={"members": [[d["seg2"], 0]], "name": "张伟"})
        assert r.status_code == 409 and "切换档案" in r.json()["error"]
        assert rt.db.one("SELECT speaker_id FROM utterances WHERE id=?", (d["utts"]["a3"],))["speaker_id"] is None
    finally:
        rt.end_switch()
    assert rt.wait_jobs(5)


def test_voices_name_and_job_report_missing_models_in_chinese(client: TestClient, rt: Runtime):
    """Finding 11: without the CAM++ model file the name request is a 400 and the voices job a JobError,
    both with the Chinese message, instead of an English 500 / RuntimeError."""
    d = _two_unknown_groups(rt)
    rt.cfg["speakers"]["embedding_model"] = "missing-model.onnx"
    assert rt._engine is None
    r = client.post("/api/voices/name", json={"members": [[d["seg1"], 1]], "name": "王敏"})
    assert r.status_code == 400 and "模型文件缺失" in r.json()["error"] and "download_models" in r.json()["error"]
    assert rt.db.one("SELECT COUNT(*) AS n FROM speakers")["n"] == 0
    assert rt.db.one("SELECT speaker_id FROM utterances WHERE id=?", (d["utts"]["b1"],))["speaker_id"] is None
    assert client.post("/api/jobs/voices").status_code == 200 and rt.wait_jobs(5)
    job = client.get("/api/jobs").json()["voices"]
    assert job["running"] is False and job["error"] and "模型文件缺失" in job["error"] and "Error" not in job["error"]


# --- v3: force / mismatch on the naming endpoints (more in tests/test_mixed_speakers.py) -----------------------------

def test_voices_name_mismatch_is_400_with_flag_and_force_retries(client: TestClient, rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    from audio_note.speech.speakers import SampleMismatch

    class Engine(FakeEngine):
        def enroll_from_audio(self, db, name, audio, source=None, copy_as_source=False, force=False, **kw):
            self.calls.append(("enroll_from_audio", name, force))
            if not force:
                raise SampleMismatch(name, 0.31, 0.5)
            return super().enroll_from_audio(db, name, audio, source, copy_as_source)

    eng = Engine()
    monkeypatch.setattr(rt, "engine", lambda: eng)
    d = _two_unknown_groups(rt)
    _speaker_with_samples(rt, "张三")
    r = client.post("/api/voices/name", json={"members": [[d["seg1"], 0]], "name": "张三", "auto_relabel": False})
    assert r.status_code == 400
    assert r.json()["mismatch"] is True and r.json()["score"] == 0.31 and r.json()["threshold"] == 0.5 and "张三" in r.json()["error"]
    assert rt.db.one("SELECT speaker_id FROM utterances WHERE id=?", (d["utts"]["a1"],))["speaker_id"] is None
    r = client.post("/api/voices/name", json={"members": [[d["seg1"], 0]], "name": "张三", "auto_relabel": False, "force": True})
    assert r.status_code == 200, r.text
    assert r.json()["samples_added"] == 1 and r.json()["utterances_assigned"] == 2
    assert ("enroll_from_audio", "张三", True) in eng.calls and ("enroll_from_audio", "张三", False) in eng.calls
    sp = {s["name"]: s for s in client.get("/api/speakers").json()["speakers"]}
    assert "outlier_samples" in sp["张三"]
    assert rt.wait_jobs(5)


def test_speakers_list_and_samples_expose_v3b_fields(client: TestClient, rt: Runtime):
    sid = _speaker_with_samples(rt, "张三", n=2)
    rows = client.get(f"/api/speakers/{sid}/samples").json()
    assert [r["origin"] for r in rows] == [None, None] and all(r["ambiguous"] is False for r in rows)
    sp = {s["name"]: s for s in client.get("/api/speakers").json()["speakers"]}
    assert sp["张三"]["outlier_samples"] == 0 and sp["张三"]["ambiguous_samples"] == 0
    rt.db.add_speaker_embedding(sid, _voice(5), "rec.wav", 2.0, origin="mic")
    assert client.get(f"/api/speakers/{sid}/samples").json()[-1]["origin"] == "mic"



def test_speakers_list_has_v4_counts_and_settings_are_editable(client: TestClient, rt: Runtime):
    sid = rt.db.upsert_speaker("甲")
    rt.db.add_speaker_embedding(sid, _voice(1), None, 3.0)
    sp = {s["id"]: s for s in client.get("/api/speakers").json()["speakers"]}[sid]
    assert (sp["negatives"], sp["rejections"], sp["risky_samples"]) == (0, 0, 0)
    rt.db.add_speaker_negative(sid, _voice(2), None, 1.0)
    assert {s["id"]: s for s in client.get("/api/speakers").json()["speakers"]}[sid]["negatives"] == 1
    cfg = client.get("/api/config").json()
    assert "negative_margin" in cfg["defaults"]["speakers"] and "system_turn_share" in cfg["defaults"]["speakers"]
    r = client.put("/api/config", json={"speakers": {"system_turn_share": 1.5}})
    assert r.status_code == 400


def test_label_whole_cluster_skips_vetoed_rows_and_an_explicit_pick_lifts_its_veto(client: TestClient, rt: Runtime,
                                                                                   fake_engine: FakeEngine):
    """v4 review fix: the cluster expansion of /api/speakers/label never reassigns a sentence the user vetoed for
    that speaker (「不是 TA」, counted in skipped_user_labels); the sentence picked explicitly is assigned and its
    veto for the speaker is lifted."""
    seg, _ = _seg(rt.cfg, rt.db, datetime(2026, 10, 30, 9, 0), audio=_tone(10.0))
    wang = rt.db.upsert_speaker("王敏")
    u1 = _utt(rt.db, seg, DAY, "09:00", "第一句", None, None, 0.0, 2.0, cluster=0)
    u2 = _utt(rt.db, seg, DAY, "09:00", "第二句", None, None, 2.0, 4.0, cluster=0)
    u3 = _utt(rt.db, seg, DAY, "09:00", "不是王敏", None, None, 4.0, 5.0, cluster=0)
    rt.db.reject_utterances(wang, [u1, u3], "未知")
    r = client.post("/api/speakers/label", json={"name": "王敏", "utterance_id": u1})
    assert r.status_code == 200, r.text
    assert r.json()["id"] == wang and r.json()["assigned"] == 2 and r.json()["skipped_user_labels"] == 1
    rows = {x["id"]: x for x in rt.db.query("SELECT id, speaker_id, speaker_source FROM utterances")}
    assert rows[u1]["speaker_id"] == wang and rows[u2]["speaker_id"] == wang and rows[u3]["speaker_id"] is None
    assert rt.db.rejected_speakers([u1, u2, u3]) == {u3: {wang}}
    r = client.post("/api/speakers/label", json={"name": "王敏", "utterance_id": u3, "whole_cluster": False})
    assert r.status_code == 200 and r.json()["assigned"] == 1
    assert rt.db.one("SELECT speaker_id FROM utterances WHERE id=?", (u3,))["speaker_id"] == wang
    assert rt.db.rejected_speakers([u3]) == {}
    assert rt.wait_jobs(5)
