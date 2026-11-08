"""Tests for audio_note.speech.pipeline (no network, no heavy models: engine/ASR/corrector are fakes)."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from audio_note.config import load_config
from audio_note.db import DB
from audio_note.llm.client import LLMError
from audio_note.speech.asr import ASRResult
from audio_note.speech.pipeline import Chunk, Processor, build_chunks, tokenize, transcribe_file, write_day_markdown
from audio_note.speech.speakers import Turn

SR = 16000


# --------------------------------------------------------------------------- fixtures
@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "data"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "paths": {"data_dir": str(data_dir)},
        "pipeline": {"min_segment_age_s": 0},
        "correction": {"glossary_file": str(tmp_path / "glossary.txt")},
    }), encoding="utf-8")
    monkeypatch.setenv("AUDIO_NOTE_CONFIG", str(cfg_path))
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    c = load_config()
    c.ensure_dirs()
    return c


@pytest.fixture
def db(cfg):
    return DB(cfg.db_path)


def make_segment(cfg, db, started: datetime, seconds: float = 3.0, seed: int = 0) -> int:
    rng = np.random.default_rng(seed)
    audio = (rng.standard_normal(int(seconds * SR)) * 0.05).astype(np.float32)
    d = cfg.audio_dir / started.strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    path = d / (started.strftime("%H%M%S_000") + ".wav")
    sf.write(path, audio, SR, format="WAV", subtype="PCM_16")
    return db.add_segment(path, started, seconds, SR, speech_s=seconds)


class FakeEngine:
    def __init__(self, turns: list[Turn] | None = None):
        self.turns = turns
        self.diarize_calls = 0
        self.identify_calls = 0

    def diarize(self, audio):
        self.diarize_calls += 1
        if self.turns is None:
            return []
        return [Turn(t.start, t.end, t.cluster) for t in self.turns]

    def identify_turns(self, audio, turns, db):
        self.identify_calls += 1
        for t, src in zip(turns, self.turns):
            t.speaker_id, t.speaker_name, t.score = src.speaker_id, src.speaker_name, src.score
        return turns


class FakeASR:
    def __init__(self, text: str = "你好世界"):
        self.text = text
        self.calls: list[int] = []

    def transcribe(self, audio):
        self.calls.append(len(audio))
        return ASRResult(self.text, 3.0, 1)


class EchoCorrector:
    def __init__(self):
        self.calls: list[tuple[list, list]] = []

    def correct(self, items, glossary):
        self.calls.append((items, list(glossary)))
        return [{"i": it["i"], "text": it["text"], "edits": []} for it in items]


class FailingCorrector:
    def correct(self, items, glossary):
        raise LLMError("boom")


class EditingCorrector:
    def correct(self, items, glossary):
        return [{"i": it["i"], "text": it["text"].replace("世界", "世间"), "edits": [{"from": "世界", "to": "世间", "reason": "test"}]}
                for it in items]


# --------------------------------------------------------------------------- build_chunks
def test_build_chunks_merges_same_cluster_within_gap():
    turns = [Turn(0.0, 2.0, 0), Turn(2.5, 4.0, 0), Turn(4.2, 6.0, 0)]
    chunks = build_chunks(turns, 10.0, 60.0, gap_s=1.0, pad_s=0.15)
    assert len(chunks) == 1
    c = chunks[0]
    assert (c.start_s, c.end_s, c.cluster) == (0.0, 6.0, 0)
    assert c.audio_start_s == 0.0            # clamped at file start
    assert c.audio_end_s == pytest.approx(6.15)


def test_build_chunks_does_not_merge_across_gap_or_cluster():
    turns = [Turn(0.0, 2.0, 0), Turn(3.5, 5.0, 0), Turn(5.2, 7.0, 1), Turn(7.1, 9.0, 0)]
    chunks = build_chunks(turns, 20.0, 60.0, gap_s=1.0)
    assert [(c.start_s, c.end_s, c.cluster) for c in chunks] == [(0.0, 2.0, 0), (3.5, 5.0, 0), (5.2, 7.0, 1), (7.1, 9.0, 0)]


def test_build_chunks_sorts_turns_and_carries_speaker_info():
    turns = [Turn(5.0, 7.0, 1, speaker_id=2, speaker_name="李四", score=0.7),
             Turn(0.0, 3.0, 0, speaker_id=1, speaker_name="张三", score=0.9)]
    chunks = build_chunks(turns, 10.0, 60.0)
    assert [c.speaker_name for c in chunks] == ["张三", "李四"]
    assert chunks[0].speaker_id == 1 and chunks[0].score == 0.9
    assert chunks[1].cluster == 1


def test_build_chunks_splits_at_max_chunk_on_turn_boundaries():
    turns = [Turn(i * 10.0, i * 10.0 + 9.5, 0) for i in range(7)]  # 7 turns, 0.5s gaps, 69.5s total
    chunks = build_chunks(turns, 100.0, 30.0, gap_s=1.0)
    assert all(c.duration_s <= 30.0 for c in chunks)
    assert [(c.start_s, c.end_s) for c in chunks] == [(0.0, 29.5), (30.0, 59.5), (60.0, 69.5)]


def test_build_chunks_splits_single_long_turn_evenly():
    chunks = build_chunks([Turn(0.0, 150.0, 0)], 150.0, 60.0)
    assert len(chunks) == 3
    assert all(c.duration_s == pytest.approx(50.0) for c in chunks)
    assert chunks[0].start_s == 0.0 and chunks[-1].end_s == pytest.approx(150.0)
    # padding stays inside the file
    assert chunks[-1].audio_end_s == 150.0
    assert chunks[1].audio_start_s == pytest.approx(50.0 - 0.15)


def test_build_chunks_drops_short_and_pads_and_clamps():
    turns = [Turn(0.05, 0.4, 0), Turn(2.0, 3.0, 1), Turn(9.95, 10.0, 2)]
    chunks = build_chunks(turns, 10.0, 60.0, pad_s=0.15, min_chunk_s=0.6)
    assert [c.cluster for c in chunks] == [1]
    c = chunks[0]
    assert c.audio_start_s == pytest.approx(1.85)
    assert c.audio_end_s == pytest.approx(3.15)
    audio = np.zeros(10 * SR, dtype=np.float32)
    assert len(c.slice(audio)) == int(round(1.3 * SR))


def test_build_chunks_clamps_turn_end_to_audio_length_and_empty():
    assert build_chunks([], 10.0, 60.0) == []
    chunks = build_chunks([Turn(8.0, 12.0, 0)], 10.0, 60.0)
    assert len(chunks) == 1 and chunks[0].end_s == 10.0 and chunks[0].audio_end_s == 10.0


# --------------------------------------------------------------------------- tokenize
def test_tokenize_falls_back_to_jieba():
    out = tokenize("今天天气很好")
    assert " " in out and "".join(out.split()) == "今天天气很好"


# --------------------------------------------------------------------------- process_segment
def test_process_segment_end_to_end(cfg, db):
    started = datetime(2026, 10, 31, 8, 19, 10)
    seg_id = make_segment(cfg, db, started, 3.0)
    (cfg.glossary_path).write_text("世界\n# comment\n", encoding="utf-8")
    db.upsert_speaker("张三")

    engine = FakeEngine([Turn(0.1, 1.4, 0, speaker_id=1, speaker_name="张三", score=0.8), Turn(1.6, 2.9, 1)])
    asr = FakeASR("你好世界")
    corr = EchoCorrector()
    p = Processor(cfg, db, client=object(), engine=engine, asr=asr, corrector=corr)

    n = p.process_segment(seg_id)
    assert n == 2
    assert engine.diarize_calls == 1 and engine.identify_calls == 1
    assert len(asr.calls) == 2
    # padded slices: 0.1-1.4 -> 0.0-1.55 ; 1.6-2.9 -> 1.45-3.0
    assert asr.calls[0] == int(round(1.55 * SR))
    assert asr.calls[1] == int(round(1.55 * SR))

    seg = db.get_segment(seg_id)
    assert seg["status"] == "done" and seg["error"] is None
    assert Path(seg["path"]).exists()

    utts = db.utterances_for_day("2026-10-31")
    assert len(utts) == 2
    u0, u1 = utts
    assert u0["speaker_name"] == "张三" and u0["speaker_id"] == 1 and u0["speaker_score"] == pytest.approx(0.8)
    assert u1["speaker_name"] == cfg.speakers.unknown_label and u1["speaker_id"] is None
    assert u0["abs_start"].startswith("2026-10-31T08:19:10.100")
    assert u1["abs_start"].startswith("2026-10-31T08:19:11.600")
    assert u0["text_raw"] == "你好世界" and u0["text"] == "你好世界"
    assert u0["cluster_label"] == 0 and u1["cluster_label"] == 1
    # FTS tokens were written and are searchable
    toks = db.query("SELECT text_seg FROM utterance_tokens ORDER BY utterance_id")
    assert len(toks) == 2 and "".join(toks[0]["text_seg"].split()) == "你好世界"
    assert db.query("SELECT rowid FROM utterances_fts WHERE utterances_fts MATCH '你好'")

    tr = db.one("SELECT * FROM transcripts WHERE segment_id=?", (seg_id,))
    assert tr is not None
    assert tr["asr_model"] == cfg.mimo.asr_model
    assert tr["text_raw"] == "你好世界\n你好世界"
    assert tr["text_corrected"] == "你好世界\n你好世界"
    raw = json.loads(tr["raw_json"])
    assert [r["speaker_name"] for r in raw] == ["张三", None]
    assert raw[0]["start_s"] == 0.1 and raw[1]["end_s"] == 2.9

    # corrector got the whole segment's items plus glossary with speaker names
    assert len(corr.calls) == 1
    items, glossary = corr.calls[0]
    assert items == [{"i": 0, "speaker": "张三", "text": "你好世界"}, {"i": 1, "speaker": "未知", "text": "你好世界"}]
    assert glossary == ["世界", "张三"]

    md = cfg.transcripts_dir / "2026-10-31.md"
    text = md.read_text(encoding="utf-8")
    assert text == (
        "# 2026-10-31 记录\n\n"
        f"## 08:19:10–08:19:13 (#{seg_id} 081910_000.wav)\n"
        "- **08:19:10 张三**：你好世界\n"
        "- **08:19:11 未知**：你好世界\n"
    )
    assert "\r" not in md.read_bytes().decode("utf-8")


def test_process_segment_applies_correction_and_records_edits(cfg, db):
    seg_id = make_segment(cfg, db, datetime(2026, 10, 31, 9, 0, 0), 3.0)
    p = Processor(cfg, db, client=object(), engine=FakeEngine([Turn(0.0, 3.0, 0)]), asr=FakeASR("你好世界"),
                  corrector=EditingCorrector())
    assert p.process_segment(seg_id) == 1
    u = db.utterances_for_day("2026-10-31")[0]
    assert u["text_raw"] == "你好世界" and u["text"] == "你好世间"
    tr = db.one("SELECT * FROM transcripts WHERE segment_id=?", (seg_id,))
    assert tr["text_corrected"] == "你好世间"
    edits = json.loads(tr["correction_json"])
    assert edits == [{"i": 0, "from": "世界", "to": "世间", "reason": "test"}]
    assert "你好世间" in (cfg.transcripts_dir / "2026-10-31.md").read_text(encoding="utf-8")


def test_process_segment_correction_failure_keeps_raw(cfg, db):
    seg_id = make_segment(cfg, db, datetime(2026, 10, 31, 9, 0, 0), 3.0)
    p = Processor(cfg, db, client=object(), engine=FakeEngine([Turn(0.0, 3.0, 0)]), asr=FakeASR("你好世界"),
                  corrector=FailingCorrector())
    assert p.process_segment(seg_id) == 1
    assert db.get_segment(seg_id)["status"] == "done"
    u = db.utterances_for_day("2026-10-31")[0]
    assert u["text"] == "你好世界"
    assert db.one("SELECT correction_json FROM transcripts WHERE segment_id=?", (seg_id,))["correction_json"] is None


def test_process_segment_correction_disabled_skips_corrector(cfg, db):
    cfg["correction"]["enabled"] = False
    seg_id = make_segment(cfg, db, datetime(2026, 10, 31, 9, 0, 0), 3.0)
    p = Processor(cfg, db, client=object(), engine=FakeEngine([Turn(0.0, 3.0, 0)]), asr=FakeASR("你好世界"))
    assert p.process_segment(seg_id) == 1
    assert p.corrector is None


def test_process_segment_no_turns_marks_no_speech_and_deletes_file(cfg, db):
    seg_id = make_segment(cfg, db, datetime(2026, 10, 31, 9, 0, 0), 3.0)
    path = Path(db.get_segment(seg_id)["path"])
    asr = FakeASR()
    p = Processor(cfg, db, client=object(), engine=FakeEngine(None), asr=asr, corrector=EchoCorrector())
    assert p.process_segment(seg_id) == 0
    assert db.get_segment(seg_id)["status"] == "no_speech"
    assert not path.exists()
    assert asr.calls == []
    assert db.utterances_for_day("2026-10-31") == []


def test_process_segment_empty_asr_text_writes_no_utterances(cfg, db):
    seg_id = make_segment(cfg, db, datetime(2026, 10, 31, 9, 0, 0), 3.0)
    p = Processor(cfg, db, client=object(), engine=FakeEngine([Turn(0.0, 3.0, 0)]), asr=FakeASR(""), corrector=EchoCorrector())
    assert p.process_segment(seg_id) == 0
    assert db.get_segment(seg_id)["status"] == "done"
    assert db.utterances_for_day("2026-10-31") == []


def test_process_segment_failure_marks_failed_and_retry_is_idempotent(cfg, db):
    seg_id = make_segment(cfg, db, datetime(2026, 10, 31, 9, 0, 0), 3.0)

    class BrokenASR:
        def transcribe(self, audio):
            raise RuntimeError("asr down")

    p = Processor(cfg, db, client=object(), engine=FakeEngine([Turn(0.0, 3.0, 0)]), asr=BrokenASR(), corrector=EchoCorrector())
    with pytest.raises(RuntimeError):
        p.process_segment(seg_id)
    seg = db.get_segment(seg_id)
    assert seg["status"] == "failed" and "asr down" in seg["error"]

    # retry with a working ASR: exactly one transcript / utterance, no duplicates
    p2 = Processor(cfg, db, client=object(), engine=FakeEngine([Turn(0.0, 3.0, 0)]), asr=FakeASR("你好世界"), corrector=EchoCorrector())
    stats = p2.process_pending(retry_failed=True)
    assert stats["processed"] == 1 and stats["done"] == 1 and stats["utterances"] == 1
    assert db.get_segment(seg_id)["status"] == "done"
    assert len(db.utterances_for_day("2026-10-31")) == 1
    assert len(db.query("SELECT id FROM transcripts WHERE segment_id=?", (seg_id,))) == 1


def test_process_segment_missing_file_fails(cfg, db):
    seg_id = make_segment(cfg, db, datetime(2026, 10, 31, 9, 0, 0), 3.0)
    Path(db.get_segment(seg_id)["path"]).unlink()
    p = Processor(cfg, db, client=object(), engine=FakeEngine([Turn(0.0, 3.0, 0)]), asr=FakeASR())
    with pytest.raises(FileNotFoundError):
        p.process_segment(seg_id)
    assert db.get_segment(seg_id)["status"] == "failed"


# --------------------------------------------------------------------------- process_pending
def test_process_pending_stats_limit_and_min_age(cfg, db):
    ids = [make_segment(cfg, db, datetime(2026, 10, 31, 9, i, 0), 3.0, seed=i) for i in range(3)]
    done_before = make_segment(cfg, db, datetime(2026, 10, 30, 9, 0, 0), 3.0, seed=9)
    db.set_segment_status(done_before, "done")

    engine = FakeEngine([Turn(0.0, 3.0, 0)])
    p = Processor(cfg, db, client=object(), engine=engine, asr=FakeASR("你好世界"), corrector=EchoCorrector())
    stats = p.process_pending(limit=2)
    assert stats == {"processed": 2, "done": 2, "no_speech": 0, "failed": 0, "utterances": 2}
    assert [db.get_segment(i)["status"] for i in ids] == ["done", "done", "new"]

    stats = p.process_pending()
    assert stats["processed"] == 1 and stats["done"] == 1
    assert db.get_segment(ids[2])["status"] == "done"
    assert p.process_pending()["processed"] == 0

    # segments younger than min_segment_age_s are left alone
    cfg["pipeline"]["min_segment_age_s"] = 3600
    young = make_segment(cfg, db, datetime(2026, 10, 31, 10, 0, 0), 3.0, seed=5)
    assert p.process_pending()["processed"] == 0
    assert db.get_segment(young)["status"] == "new"


def test_process_pending_counts_failures_and_no_speech(cfg, db):
    a = make_segment(cfg, db, datetime(2026, 10, 31, 9, 0, 0), 3.0, seed=1)
    b = make_segment(cfg, db, datetime(2026, 10, 31, 9, 1, 0), 3.0, seed=2)
    Path(db.get_segment(b)["path"]).unlink()
    p = Processor(cfg, db, client=object(), engine=FakeEngine(None), asr=FakeASR())
    stats = p.process_pending()
    assert stats["processed"] == 2 and stats["no_speech"] == 1 and stats["failed"] == 1 and stats["done"] == 0
    assert db.get_segment(a)["status"] == "no_speech" and db.get_segment(b)["status"] == "failed"


def test_process_pending_reclaims_stale_processing_rows(cfg, db):
    """A segment left in 'processing' by a killed run is picked up again; a fresh one is left alone."""
    stale = make_segment(cfg, db, datetime(2026, 10, 31, 9, 0, 0), 3.0, seed=1)
    fresh = make_segment(cfg, db, datetime(2026, 10, 31, 9, 1, 0), 3.0, seed=2)
    db.set_segment_status(stale, "processing", "half done")
    db.execute("UPDATE segments SET updated_at=datetime('now','localtime','-1 hour') WHERE id=?", (stale,))
    db.set_segment_status(fresh, "processing")
    p = Processor(cfg, db, client=object(), engine=FakeEngine([Turn(0.0, 3.0, 0)]), asr=FakeASR("你好世界"), corrector=EchoCorrector())
    stats = p.process_pending()
    assert stats["processed"] == 1 and stats["done"] == 1
    assert db.get_segment(stale)["status"] == "done" and db.get_segment(stale)["error"] is None
    assert db.get_segment(fresh)["status"] == "processing"


def test_process_pending_stops_between_segments_on_stop_event(cfg, db):
    import threading

    ids = [make_segment(cfg, db, datetime(2026, 10, 31, 9, i, 0), 3.0, seed=i) for i in range(3)]
    stop = threading.Event()

    class StoppingASR(FakeASR):
        def transcribe(self, audio):
            stop.set()   # a shutdown request arrives while the first segment is being transcribed
            return super().transcribe(audio)

    p = Processor(cfg, db, client=object(), engine=FakeEngine([Turn(0.0, 3.0, 0)]), asr=StoppingASR("你好"),
                  corrector=EchoCorrector(), stop_event=stop)
    stats = p.process_pending()
    assert stats["processed"] == 1 and stats["done"] == 1
    assert [db.get_segment(i)["status"] for i in ids] == ["done", "new", "new"]   # nothing left half-done


def test_content_filtered_chunk_is_skipped_not_fatal(cfg, db):
    from audio_note.llm.client import ContentFiltered

    seg_id = make_segment(cfg, db, datetime(2026, 10, 31, 9, 0, 0), 6.0)

    class FilteringASR(FakeASR):
        def transcribe(self, audio):
            self.calls.append(len(audio))
            if len(self.calls) == 2:
                raise ContentFiltered("asr: content filtered (421)")
            return ASRResult("你好世界", 3.0, 1)

    asr = FilteringASR()
    turns = [Turn(0.0, 1.5, 0), Turn(2.0, 3.5, 1), Turn(4.0, 5.5, 0)]
    p = Processor(cfg, db, client=object(), engine=FakeEngine(turns), asr=asr, corrector=EchoCorrector())
    assert p.process_segment(seg_id) == 2
    assert db.get_segment(seg_id)["status"] == "done"
    assert len(asr.calls) == 3 and p.skipped_chunks == 1
    utts = db.utterances_for_day("2026-10-31")
    assert [u["cluster_label"] for u in utts] == [0, 0]   # the filtered middle chunk is simply absent
    # a transient give-up (LLMError) still fails the segment so it is retried later
    class DownASR(FakeASR):
        def transcribe(self, audio):
            raise LLMError("asr: giving up after 4 attempts")

    seg2 = make_segment(cfg, db, datetime(2026, 10, 31, 10, 0, 0), 3.0, seed=3)
    p2 = Processor(cfg, db, client=object(), engine=FakeEngine([Turn(0.0, 3.0, 0)]), asr=DownASR(), corrector=EchoCorrector())
    with pytest.raises(LLMError):
        p2.process_segment(seg2)
    assert db.get_segment(seg2)["status"] == "failed"


def test_relabel_segment_updates_speakers_without_asr(cfg, db):
    seg_id = make_segment(cfg, db, datetime(2026, 10, 31, 8, 19, 10), 3.0)
    turns = [Turn(0.1, 1.4, 0), Turn(1.6, 2.9, 1)]
    asr = FakeASR("你好世界")
    p = Processor(cfg, db, client=object(), engine=FakeEngine(turns), asr=asr, corrector=EchoCorrector())
    assert p.process_segment(seg_id) == 2
    assert [u["speaker_name"] for u in db.utterances_for_day("2026-10-31")] == ["未知", "未知"]

    # the user enrolled 张三 later: a new engine now recognises cluster 0
    db.upsert_speaker("张三")
    relabelled = FakeEngine([Turn(0.1, 1.4, 0, speaker_id=1, speaker_name="张三", score=0.9), Turn(1.6, 2.9, 1)])
    p2 = Processor(cfg, db, engine=relabelled)   # no client, no ASR needed
    assert p2.relabel_segment(seg_id) == 1
    assert relabelled.diarize_calls == 0 and relabelled.identify_calls == 1 and len(asr.calls) == 2
    utts = db.utterances_for_day("2026-10-31")
    assert [(u["speaker_name"], u["speaker_id"]) for u in utts] == [("张三", 1), ("未知", None)]
    assert utts[0]["speaker_score"] == pytest.approx(0.9)
    assert "张三" in (cfg.transcripts_dir / "2026-10-31.md").read_text(encoding="utf-8")
    assert p2.relabel(day="2026-10-31") == {"segments": 1, "changed": 0, "skipped": 0}
    # a segment whose audio expired is skipped, not fatal
    Path(db.get_segment(seg_id)["path"]).unlink()
    assert p2.relabel()["skipped"] == 1


# --------------------------------------------------------------------------- lazy components
def test_lazy_client_without_key_raises_clear_error(cfg, db):
    p = Processor(cfg, db)
    with pytest.raises(LLMError, match="API key"):
        _ = p.client
    with pytest.raises(LLMError, match="API key"):
        _ = p.asr


def test_corrector_missing_module_is_skipped(cfg, db, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.endswith("speech.correct") or name == "correct":
            raise ImportError("not yet written")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    p = Processor(cfg, db, client=object())
    assert p.corrector is None


# --------------------------------------------------------------------------- write_day_markdown
def test_write_day_markdown_groups_by_segment_and_marks_imports(cfg, db):
    s1 = make_segment(cfg, db, datetime(2026, 10, 31, 8, 19, 5), 3.0, seed=1)
    s2 = db.add_segment(cfg.audio_dir / "x.wav", datetime(2026, 10, 31, 10, 0, 0), 125.0, SR, source="import",
                        origin="C:/recordings/meeting.m4a")
    db.add_utterances([
        {"segment_id": s1, "start_s": 0.5, "end_s": 2.0, "abs_start": "2026-10-31T08:19:05.500", "day": "2026-10-31",
         "cluster_label": 0, "speaker_id": None, "speaker_name": "张三", "speaker_score": 0.9, "text_raw": "a", "text": "今天我们先过一下"},
        {"segment_id": s2, "start_s": 0.0, "end_s": 1.0, "abs_start": "2026-10-31T10:00:00.000", "day": "2026-10-31",
         "cluster_label": 0, "speaker_id": None, "speaker_name": None, "speaker_score": None, "text_raw": "b", "text": "好的。\n第二行"},
        {"segment_id": s2, "start_s": 70.0, "end_s": 71.0, "abs_start": "2026-10-31T10:01:10.000", "day": "2026-10-31",
         "cluster_label": 1, "speaker_id": None, "speaker_name": "李四", "speaker_score": None, "text_raw": "c", "text": "没问题"},
    ])
    path = write_day_markdown(db, cfg, "2026-10-31")
    assert path == cfg.transcripts_dir / "2026-10-31.md"
    # the heading carries the segment id + file name so `speakers label` can be run from the transcript
    assert path.read_text(encoding="utf-8") == (
        "# 2026-10-31 记录\n\n"
        f"## 08:19:05–08:19:08 (#{s1} 081905_000.wav)\n"
        "- **08:19:05 张三**：今天我们先过一下\n\n"
        f"## 10:00:00–10:02:05 (#{s2} x.wav)（导入：meeting.m4a）\n"
        "- **10:00:00 未知**：好的。 第二行\n"
        "- **10:01:10 李四**：没问题\n"
    )
    # empty day still produces a valid file
    empty = write_day_markdown(db, cfg, "2026-11-01")
    assert empty.read_text(encoding="utf-8") == "# 2026-11-01 记录\n"


# --------------------------------------------------------------------------- transcribe_file
def test_transcribe_file_no_db_writes(cfg, db, tmp_path):
    wav = tmp_path / "clip.wav"
    sf.write(wav, (np.random.default_rng(3).standard_normal(3 * SR) * 0.05).astype(np.float32), SR)
    db.upsert_speaker("张三")
    engine = FakeEngine([Turn(0.0, 1.4, 0, speaker_id=1, speaker_name="张三", score=0.8), Turn(1.6, 3.0, 1)])
    out = transcribe_file(cfg, object(), wav, db=db, engine=engine, asr=FakeASR("你好世界"), corrector=EditingCorrector())
    assert [(o["speaker"], o["text_raw"], o["text"]) for o in out] == [("张三", "你好世界", "你好世间"), ("未知", "你好世界", "你好世间")]
    assert out[0]["start_s"] == 0.0 and out[1]["end_s"] == 3.0
    assert db.query("SELECT id FROM utterances") == [] and db.query("SELECT id FROM transcripts") == []
    assert db.query("SELECT id FROM segments") == []


# --------------------------------------------------------------------------- v3: split_utterance (more in tests/test_mixed_speakers.py)
def test_split_utterance_replaces_the_row_with_retranscribed_pieces(cfg, db):
    seg_id = make_segment(cfg, db, datetime(2026, 10, 31, 8, 19, 10), 6.0)
    turns = [Turn(0.0, 6.0, 0)]
    p = Processor(cfg, db, client=object(), engine=FakeEngine(turns), asr=FakeASR("混在一起"), corrector=EchoCorrector())
    assert p.process_segment(seg_id) == 1
    mixed = db.utterances_for_day("2026-10-31")[0]["id"]

    class SplitEngine(FakeEngine):
        def diarize(self, audio):
            return [Turn(0.0, 2.5, 0), Turn(2.5, 6.0, 1)]

        def identify_turns(self, audio, turns, db_, segment_id=None):
            for t in turns:
                t.speaker_id, t.speaker_name, t.score = (1, "张三", 0.9) if t.cluster == 0 else (None, None, 0.2)
            return turns

    db.upsert_speaker("张三")
    asr = FakeASR("拆开")
    corr = EchoCorrector()
    p2 = Processor(cfg, db, client=object(), engine=SplitEngine(turns), asr=asr, corrector=corr)
    res = p2.split_utterance(mixed, auto=True)
    assert res["removed"] == mixed and res["billed_s"] == 6.0 and len(asr.calls) == 2
    assert [(x["start_s"], x["end_s"], x["speaker"], x["speaker_id"], x["score"], x["text"]) for x in res["pieces"]] == \
        [(0.0, 2.5, "张三", 1, 0.9, "拆开"), (2.5, 6.0, "未知", None, 0.2, "拆开")]
    assert len(corr.calls) == 1 and [it["speaker"] for it in corr.calls[0][0]] == ["张三", "未知"]
    utts = db.utterances_for_day("2026-10-31")
    assert [(u["id"], u["cluster_label"], u["speaker_source"]) for u in utts] == [(res["pieces"][0]["id"], 0, None), (res["pieces"][1]["id"], 1, None)]
    assert utts[1]["abs_start"] == "2026-10-31T08:19:12.500"
    assert db.one("SELECT COUNT(*) AS n FROM utterances WHERE text='混在一起'")["n"] == 0
    assert db.one("SELECT COUNT(*) AS n FROM utterance_tokens")["n"] == 2
    md = (cfg.transcripts_dir / "2026-10-31.md").read_text(encoding="utf-8")
    assert "08:19:12 未知**：拆开" in md and "混在一起" not in md


# --------------------------------------------------------------------------- v3b: faint (far-away) turns (more in tests/test_loudness.py)
def test_process_segment_drops_faint_turn_and_stores_loudness(cfg, db):
    """The scripted engine hands back two turns; the second lies on 100x quieter audio and is dropped before
    identification (audio.min_speech_dbfs -40): its seconds land in segments.faint_s, the kept utterance stores its level."""
    from tests.conftest import sine

    started = datetime(2026, 10, 31, 9, 0, 0)
    d = cfg.audio_dir / "2026-10-31"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "090000_000.wav"
    sf.write(path, np.concatenate([sine(3.0, amp=0.3), sine(3.0, amp=0.003)]), SR, format="WAV", subtype="PCM_16")
    seg_id = db.add_segment(path, started, 6.0, SR, speech_s=6.0)
    engine = FakeEngine([Turn(0.1, 2.9, 0), Turn(3.1, 5.9, 1)])
    asr = FakeASR("你好世界")
    p = Processor(cfg, db, client=object(), engine=engine, asr=asr, corrector=EchoCorrector())
    assert p.process_segment(seg_id) == 1 and len(asr.calls) == 1
    seg = db.get_segment(seg_id)
    assert seg["status"] == "done" and seg["faint_s"] == pytest.approx(2.8, abs=1e-6)
    utts = db.utterances_for_day("2026-10-31")
    assert len(utts) == 1 and utts[0]["cluster_label"] == 0 and -14.5 < utts[0]["loudness_db"] < -12.5
    # gate off: both turns survive, nothing is counted as faint
    cfg["audio"]["min_speech_dbfs"] = None
    assert p.process_segment(seg_id) == 2 and db.get_segment(seg_id)["faint_s"] == 0.0
    assert [round(u["loudness_db"]) for u in db.utterances_for_day("2026-10-31")] == [-13, -53]



def test_process_segment_mic_only_has_no_system_share(cfg, db):
    seg_id = make_segment(cfg, db, datetime(2026, 10, 31, 10, 0, 0), 3.0)
    p = Processor(cfg, db, client=object(), engine=FakeEngine([Turn(0.0, 3.0, 0)]), asr=FakeASR("你好世界"), corrector=EchoCorrector())
    assert p.process_segment(seg_id) == 1
    assert db.query("SELECT system_share FROM utterances WHERE segment_id=?", (seg_id,))[0]["system_share"] is None


def test_process_segment_stores_system_share_from_channel_rms(cfg, db):
    from audio_note.speech.pipeline import system_share, channel_pairs

    seg_id = make_segment(cfg, db, datetime(2026, 10, 31, 10, 5, 0), 3.0)
    ch = np.asarray([0.1, 0.0] * 3 + [0.0, 0.1] * 3, dtype=np.float16).tobytes()       # 0-1.5 s mic, 1.5-3 s system
    db.execute("UPDATE segments SET channel_rms=?, channel_block_s=0.5 WHERE id=?", (ch, seg_id))
    p = Processor(cfg, db, client=object(), engine=FakeEngine([Turn(0.0, 1.4, 0), Turn(1.6, 3.0, 1)]), asr=FakeASR("你好"),
                  corrector=EchoCorrector())
    assert p.process_segment(seg_id) == 2
    shares = [r["system_share"] for r in db.query("SELECT system_share FROM utterances WHERE segment_id=? ORDER BY start_s", (seg_id,))]
    assert shares[0] == pytest.approx(0.0) and shares[1] == pytest.approx(1.0)
    assert system_share(channel_pairs(ch), 0.5, 1.0, 2.0) == pytest.approx(0.5)
