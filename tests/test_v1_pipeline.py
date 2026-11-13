"""Regression tests for the split of audio_note.speech.pipeline (1.0, SPEC_v1 section 2 / package A4).

No network, no models: the engine, ASR and corrector are fakes, as in tests/test_pipeline.py.

1. the facade: every old name is still there, and patching a name on `pipeline` reaches the code that moved out
2. stage seams: _voice / _transcribe / _correct, storage in one transaction, after-store hooks, stage timing
3. (a) a stop request in the middle of a segment leaves it retryable (Stopped -> 'new', not 'failed')
4. (b) a split keeps the pieces' correction edits and the transcripts row follows the rows that exist
5. (c) one glossary parser
6. (d) embedding survives a transient failure
"""
from __future__ import annotations

import inspect
import json
import re
import sys
import threading
import types
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

import audio_note
from audio_note.db import DB
from audio_note.llm.client import LLMError
from audio_note.speech import glossary_file, pipeline
from audio_note.speech.asr import ASRResult
from audio_note.speech.pipeline import Chunk, Processor, Stopped, StoreContext
from audio_note.speech.speakers import Turn

SR = 16000
SPEECH_DIR = Path(pipeline.__file__).resolve().parent
SPLIT_MODULES = ("chunking", "channels", "markdown", "glossary_file", "stage_voice", "stage_text", "processor", "relabel", "split")
EDIT = {"from": "世界", "to": "世间", "reason": "test"}


# --------------------------------------------------------------------------- fixtures / doubles
@pytest.fixture
def pcfg(cfg_factory, tmp_path: Path):
    """Pipeline config: no waiting for young segments, glossary in tmp, keyword retrieval (no embedding model)."""
    return cfg_factory({
        "pipeline": {"min_segment_age_s": 0},
        "correction": {"glossary_file": str(tmp_path / "glossary.txt")},
        "memory": {"retrieval": "fts"},
        "mimo": {"asr_concurrency": 1},
    })


@pytest.fixture
def pdb(pcfg) -> DB:
    return DB(pcfg.db_path)


def make_segment(cfg, db: DB, started: datetime, seconds: float = 3.0, seed: int = 0) -> int:
    rng = np.random.default_rng(seed)
    audio = (rng.standard_normal(int(seconds * SR)) * 0.05).astype(np.float32)
    d = cfg.audio_dir / started.strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    path = d / (started.strftime("%H%M%S_000") + ".wav")
    sf.write(path, audio, SR, format="WAV", subtype="PCM_16")
    return db.add_segment(path, started, seconds, SR, speech_s=seconds)


class Engine:
    """Scripted diarization; identification copies the scripted speaker of each turn (by position)."""

    def __init__(self, turns: list[Turn] | None = None):
        self.turns = turns or []
        self.diarize_calls = 0

    def diarize(self, audio):
        self.diarize_calls += 1
        return [Turn(t.start, t.end, t.cluster) for t in self.turns]

    def identify_turns(self, audio, turns, db, segment_id=None):
        for t, src in zip(turns, self.turns):
            t.speaker_id, t.speaker_name, t.score = src.speaker_id, src.speaker_name, src.score
        return turns


class ASR:
    """Returns the scripted texts in call order (the last one repeats); `on_call(k)` runs inside call number k."""

    def __init__(self, *texts: str, on_call: Any = None):
        self.texts = list(texts) or ["你好世界"]
        self.calls = 0
        self.on_call = on_call
        self._lock = threading.Lock()

    def transcribe(self, audio):
        with self._lock:
            self.calls += 1
            k = self.calls
        if self.on_call is not None:
            self.on_call(k)
        return ASRResult(self.texts[min(k, len(self.texts)) - 1], round(len(audio) / SR, 2), 1)


class Replace:
    """Corrector double: replaces 世界 with 世间 wherever it occurs and reports the edit."""

    def __init__(self):
        self.calls: list[tuple[list[dict], list[str]]] = []

    def correct(self, items, glossary):
        self.calls.append(([dict(it) for it in items], list(glossary)))
        return [{"i": it["i"], "text": it["text"].replace("世界", "世间"), "edits": [dict(EDIT)] if "世界" in it["text"] else []}
                for it in items]


def proc(cfg, db: DB, engine: Any, asr: Any, corrector: Any = None, cls: type = Processor, **kw: Any) -> Processor:
    p = cls(cfg, db, client=object(), engine=engine, asr=asr, corrector=corrector, **kw)
    p._corrector_tried = True          # corrector=None means "no correction", not "build the real one"
    return p


def utterances(db: DB, seg_id: int) -> list[Any]:
    return db.query("SELECT * FROM utterances WHERE segment_id=? ORDER BY start_s, id", (seg_id,))


def transcript(db: DB, seg_id: int) -> Any:
    rows = db.query("SELECT * FROM transcripts WHERE segment_id=?", (seg_id,))
    assert len(rows) == 1
    return rows[0]


# =========================================================================================================
# 1. the facade
# =========================================================================================================
OLD_NAMES = [
    # public
    "Chunk", "Processor", "build_chunks", "drop_faint_turns", "write_day_markdown", "segment_heading", "transcribe_file",
    "tokenize", "load_glossary", "system_share", "channel_pairs", "system_turn_share", "transcribe_chunks",
    # constants
    "STALE_PROCESSING_S", "SPLIT_MIN_PIECE_S", "SPLIT_EDGE_S", "SPLIT_MIN_TURN_S", "SPLIT_PAD_S", "DEFAULT_SYSTEM_TURN_SHARE",
    # private helpers
    "_segment_channels", "_clip", "_parse_iso", "_row_get", "_accepts_segment_id", "_index_correction", "_asr_concurrency",
    # what the old module imported (reachable as attributes, e.g. for monkeypatching)
    "read_segment", "speech_dbfs", "min_speech_dbfs", "is_faint", "SAMPLE_RATE", "Turn", "LLMError", "ContentFiltered",
    "Config", "DB", "log", "np", "json", "math", "inspect", "logging", "threading", "Path", "datetime", "timedelta",
    "dataclass", "Any", "Sequence",
]
OLD_ALL = ["Chunk", "Processor", "build_chunks", "drop_faint_turns", "write_day_markdown", "segment_heading", "transcribe_file",
           "tokenize", "load_glossary", "system_share", "channel_pairs", "system_turn_share",
           "SPLIT_MIN_PIECE_S", "SPLIT_EDGE_S", "SPLIT_MIN_TURN_S"]


def test_facade_keeps_every_name_of_the_old_module():
    missing = [n for n in OLD_NAMES if not hasattr(pipeline, n)]
    assert missing == []
    assert set(OLD_ALL) <= set(pipeline.__all__) and all(hasattr(pipeline, n) for n in pipeline.__all__)
    assert (pipeline.STALE_PROCESSING_S, pipeline.SPLIT_MIN_PIECE_S, pipeline.SPLIT_EDGE_S, pipeline.SPLIT_MIN_TURN_S,
            pipeline.SPLIT_PAD_S, pipeline.DEFAULT_SYSTEM_TURN_SHARE) == (600, 0.6, 0.3, 1.0, 0.15, 0.6)
    assert pipeline.log.name == "audio_note.speech.pipeline"


def test_processor_keeps_its_constructor_and_methods():
    from audio_note.speech.stage_text import TextStage
    from audio_note.speech.stage_voice import VoiceStage

    assert Processor.__bases__ == (VoiceStage, TextStage)
    params = list(inspect.signature(Processor.__init__).parameters.values())
    assert [p.name for p in params] == ["self", "cfg", "db", "client", "engine", "asr", "corrector", "stop_event"]
    assert all(p.default is None for p in params[3:])
    for name in ("process_pending", "process_segment", "relabel_segment", "relabel", "preview_split", "split_utterance",
                 "_voice", "_transcribe", "_correct", "_rows", "_store", "_after_store", "_voice_after_store",
                 "_text_after_store", "_voice_row", "_text_row", "_identify", "_match_clip", "_system_exclude", "_self_id", "_mark_system_turns",
                 "_enforce_system_turns", "_embed_new", "_register_terms", "_plan_split", "_pieces_auto", "_pieces_manual",
                 "_merge_short_pieces", "_utterance_for_split"):
        assert callable(getattr(Processor, name)), name
    for name in ("client", "engine", "asr", "corrector"):
        assert isinstance(inspect.getattr_static(Processor, name), property), name
    assert list(inspect.signature(Processor._voice).parameters) == ["self", "seg", "audio"]
    assert list(inspect.signature(Processor._transcribe).parameters) == ["self", "seg", "audio", "chunks"]
    assert list(inspect.signature(Processor._correct).parameters) == ["self", "spoken"]
    assert list(inspect.signature(Processor._after_store).parameters) == ["self", "seg_id", "utterance_ids", "days"]


def test_every_facade_lookup_in_the_split_modules_resolves():
    """The split modules reach the old globals as `F.name`: each such name must exist on the facade."""
    used: dict[str, set[str]] = {}
    for mod in SPLIT_MODULES:
        src = (SPEECH_DIR / f"{mod}.py").read_text(encoding="utf-8")
        for name in re.findall(r"(?<![A-Za-z0-9_.])F[.]([A-Za-z_][A-Za-z0-9_]*)", src):
            used.setdefault(name, set()).add(mod)
    assert len(used) > 20
    assert {n: sorted(m) for n, m in used.items() if not hasattr(pipeline, n)} == {}


def test_split_modules_log_under_the_facade_name_and_do_not_import_it_eagerly():
    import importlib

    for mod in SPLIT_MODULES:
        m = importlib.import_module(f"audio_note.speech.{mod}")
        if hasattr(m, "log"):
            assert m.log.name == "audio_note.speech.pipeline", mod
        src = (SPEECH_DIR / f"{mod}.py").read_text(encoding="utf-8")
        # only chunking.F may import the facade, and only inside a function (the facade imports all of them)
        eager = [ln for ln in src.splitlines() if re.match(r"(from [.] import pipeline|from [.]pipeline import|import audio_note[.]speech[.]pipeline)", ln)]
        assert eager == [], mod


def test_patching_the_facade_reaches_process_relabel_and_split(pcfg, pdb, monkeypatch):
    seg = make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, 0, 0), 6.0)
    written: list[str] = []
    read: list[str] = []
    real_read = pipeline.read_segment
    monkeypatch.setattr(pipeline, "write_day_markdown", lambda db, cfg, day: written.append(day))
    monkeypatch.setattr(pipeline, "read_segment", lambda path: read.append(Path(path).name) or real_read(path))
    monkeypatch.setattr(pipeline, "tokenize", lambda text: "TOK")
    monkeypatch.setattr(pipeline, "speech_dbfs", lambda clip: -12.5)
    monkeypatch.setattr(pipeline, "load_glossary", lambda cfg, db=None: ["星河App"])
    corr = Replace()
    p = proc(pcfg, pdb, Engine([Turn(0.0, 6.0, 0, speaker_id=None, speaker_name=None, score=0.3)]), ASR("你好世界"), corr)
    assert p.process_segment(seg) == 1
    (u,) = utterances(pdb, seg)
    assert written == ["2026-11-02"] and read == ["090000_000.wav"]
    assert not (pcfg.transcripts_dir / "2026-11-02.md").exists()            # the real writer never ran
    assert u["loudness_db"] == -12.5 and corr.calls[0][1] == ["星河App"]
    assert [r["text_seg"] for r in pdb.query("SELECT text_seg FROM utterance_tokens")] == ["TOK"]

    # relabel: reads the audio and rewrites the day through the same patched names
    pdb.upsert_speaker("张三")
    p2 = proc(pcfg, pdb, Engine([Turn(0.0, 6.0, 0, speaker_id=1, speaker_name="张三", score=0.9)]), ASR())
    assert p2.relabel_segment(seg) == 1
    assert written == ["2026-11-02"] * 2 and len(read) == 2

    # split: constants, storage helpers and the writer are looked up on the facade too
    monkeypatch.setattr(pipeline, "SPLIT_EDGE_S", 2.0)
    with pytest.raises(ValueError, match="必须在 2.0s 和 4.0s 之间"):
        p2.preview_split(u["id"], at=[1.5], auto=False)
    monkeypatch.setattr(pipeline, "SPLIT_EDGE_S", 0.3)
    res = p2.split_utterance(u["id"], at=[3.0])
    assert len(res["pieces"]) == 2 and written == ["2026-11-02"] * 3 and len(read) == 4       # preview + split read it again
    assert [r["text_seg"] for r in pdb.query("SELECT text_seg FROM utterance_tokens")] == ["TOK", "TOK"]


def test_patching_facade_functions_and_constants_reaches_the_processor(pcfg, pdb, monkeypatch):
    seg = make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, 0, 0), 6.0)
    seen: dict[str, Any] = {}

    def fake_build(turns, audio_len_s, max_chunk_s, **kw):
        seen["build"] = (len(turns), round(audio_len_s, 1), max_chunk_s)
        c = Chunk(1.0, 2.0, 9)
        c.audio_start_s, c.audio_end_s = 1.0, 2.0
        return [c]

    def fake_transcribe(asr, audio, chunks, concurrency=4, stop_event=None):
        seen["transcribe"] = (len(chunks), concurrency)
        return [(chunks[0], "打了补丁")], 1.0, 2

    monkeypatch.setattr(pipeline, "build_chunks", fake_build)
    monkeypatch.setattr(pipeline, "transcribe_chunks", fake_transcribe)
    monkeypatch.setattr(pipeline, "min_speech_dbfs", lambda cfg: seen.setdefault("gate", None))
    monkeypatch.setattr(pipeline, "STALE_PROCESSING_S", 77)
    monkeypatch.setattr(pdb, "reset_stale_processing", lambda age: seen.setdefault("stale", age) and 0)
    p = proc(pcfg, pdb, Engine([Turn(0.0, 3.0, 0), Turn(3.0, 6.0, 1)]), ASR())
    assert p.process_pending() == {"processed": 1, "done": 1, "no_speech": 0, "failed": 0, "utterances": 1}
    assert seen == {"stale": 77, "gate": None, "build": (2, 6.0, float(pcfg.mimo.asr_max_chunk_s)), "transcribe": (1, 1)}
    (u,) = utterances(pdb, seg)
    assert (u["text"], u["cluster_label"], u["start_s"]) == ("打了补丁", 9, 1.0) and p.skipped_chunks == 2


def test_all_three_correction_call_sites_use_the_one_helper(pcfg, pdb, tmp_path, monkeypatch):
    calls: list[int] = []
    real = pipeline.correct_spoken

    def spy(corrector, spoken, cfg, db=None):
        calls.append(len(spoken))
        return real(corrector, spoken, cfg, db)

    monkeypatch.setattr(pipeline, "correct_spoken", spy)
    seg = make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, 0, 0), 6.0)
    eng = Engine([Turn(0.0, 6.0, 0)])
    p = proc(pcfg, pdb, eng, ASR("你好世界"), Replace())
    assert p.process_segment(seg) == 1                                                  # 1. process
    uid = utterances(pdb, seg)[0]["id"]
    assert [x["text"] for x in p.split_utterance(uid, at=[3.0])["pieces"]] == ["你好世间", "你好世间"]   # 2. split
    wav = tmp_path / "clip.wav"
    sf.write(wav, (np.random.default_rng(3).standard_normal(3 * SR) * 0.05).astype(np.float32), SR)
    out = pipeline.transcribe_file(pcfg, object(), wav, db=pdb, engine=Engine([Turn(0.0, 3.0, 0)]), asr=ASR("你好世界"),
                                   corrector=Replace())                                # 3. one-off file
    assert [o["text"] for o in out] == ["你好世间"] and calls == [1, 2, 1]


def test_correct_spoken_never_raises_and_accepts_the_result_shapes(pcfg):
    spoken = [(Chunk(0.0, 1.0, 0, speaker_name="张三"), "你好世界"), (Chunk(1.0, 2.0, 1), "世界很大")]
    raw = ["你好世界", "世界很大"]
    assert pipeline.correct_spoken(None, spoken, pcfg) == (raw, None)
    assert pipeline.correct_spoken(Replace(), [], pcfg) == ([], None)

    class Boom:
        def __init__(self, exc):
            self.exc = exc

        def correct(self, items, glossary):
            raise self.exc

    assert pipeline.correct_spoken(Boom(LLMError("down")), spoken, pcfg) == (raw, None)
    assert pipeline.correct_spoken(Boom(KeyError("x")), spoken, pcfg) == (raw, None)

    class Shaped:
        def __init__(self, result):
            self.result, self.items = result, None

        def correct(self, items, glossary):
            self.items = items
            return self.result

    c = Shaped({"items": [{"i": 1, "text": " 世间很大 ", "edits": [dict(EDIT), "junk"]}]})
    texts, edits = pipeline.correct_spoken(c, spoken, pcfg)
    assert texts == ["你好世界", "世间很大"] and json.loads(edits) == [{"i": 1, **EDIT}]
    assert c.items == [{"i": 0, "speaker": "张三", "text": "你好世界"}, {"i": 1, "speaker": pcfg.speakers.unknown_label, "text": "世界很大"}]
    assert pipeline.correct_spoken(Shaped({"0": {"text": "改了"}}), spoken, pcfg) == (["改了", "世界很大"], None)
    assert pipeline.correct_spoken(Shaped("garbage"), spoken, pcfg) == (raw, None)
    assert pipeline.correct_spoken(Shaped([{"i": "x", "text": "坏的"}]), spoken, pcfg) == (raw, None)      # unusable -> raw


# =========================================================================================================
# 2. stage seams, storage, hooks, timing
# =========================================================================================================
def test_the_stage_methods_are_the_seams_of_process_segment(pcfg, pdb):
    """_process only talks to the stages through _voice / _transcribe / _correct: replacing them replaces the stage."""
    seg = make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, 0, 0), 3.0)
    order: list[str] = []

    class P(Processor):
        def _voice(self, seg, audio):
            order.append("voice")
            assert len(audio) == 3 * SR
            return [Turn(0.5, 2.5, 7, speaker_id=None, speaker_name=None, score=0.4)]

        def _transcribe(self, seg, audio, chunks):
            order.append("transcribe")
            return [(c, "原文") for c in chunks], 2.0, 0

        def _correct(self, spoken):
            order.append("correct")
            return ["改过的"], None

    p = P(pcfg, pdb, client=object(), engine=object(), asr=object())       # neither engine nor ASR is touched
    assert p.process_segment(seg) == 1 and order == ["voice", "transcribe", "correct"]
    (u,) = utterances(pdb, seg)
    assert (u["text_raw"], u["text"], u["cluster_label"], u["start_s"], u["end_s"]) == ("原文", "改过的", 7, 0.5, 2.5)
    assert pdb.get_segment(seg)["status"] == "done"

    class Silent(P):
        def _voice(self, seg, audio):
            return []

    seg2 = make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, 5, 0), 3.0, seed=1)
    path = Path(pdb.get_segment(seg2)["path"])
    assert Silent(pcfg, pdb, client=object(), engine=object(), asr=object()).process_segment(seg2) == 0
    assert pdb.get_segment(seg2)["status"] == "no_speech" and not path.exists()


def test_store_writes_transcript_and_utterances_in_one_transaction(pcfg, pdb, monkeypatch):
    seg = make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, 0, 0), 6.0)
    real = pipeline.insert_utterance
    n = []

    def second_row_fails(con, row, columns=None, above_id=None):
        n.append(1)
        if len(n) == 2:
            raise RuntimeError("disk full")
        return real(con, row, columns, above_id)

    monkeypatch.setattr(pipeline, "insert_utterance", second_row_fails)
    p = proc(pcfg, pdb, Engine([Turn(0.0, 3.0, 0), Turn(3.0, 6.0, 1)]), ASR("你好世界"), Replace())
    with pytest.raises(RuntimeError, match="disk full"):
        p.process_segment(seg)
    s = pdb.get_segment(seg)
    assert s["status"] == "failed" and "disk full" in s["error"]
    for table in ("transcripts", "utterances", "utterance_tokens"):
        assert pdb.one(f"SELECT COUNT(*) AS n FROM {table}")["n"] == 0, table        # nothing half-written
    monkeypatch.setattr(pipeline, "insert_utterance", real)
    assert p.process_pending(retry_failed=True)["utterances"] == 2
    assert pdb.one("SELECT COUNT(*) AS n FROM transcripts")["n"] == 1 and len(utterances(pdb, seg)) == 2


def test_insert_utterance_writes_known_columns_ignores_the_rest_and_can_force_a_high_id(pcfg, pdb):
    seg = make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, 0, 0), 3.0)
    row = {"segment_id": seg, "start_s": 0.0, "end_s": 1.0, "abs_start": "2026-11-02T09:00:00.000", "day": "2026-11-02",
           "text_raw": "原", "text": "文", "text_seg": "文", "speaker_source": "user", "flags": {"guard": "loop"},
           "id": 999, "no_such_column": 1}
    with pdb.transaction() as con:
        columns = pipeline.utterance_columns(con)
        a = pipeline.insert_utterance(con, row)
        b = pipeline.insert_utterance(con, {**row, "text_seg": ""}, columns, above_id=40)
        c = pipeline.insert_utterance(con, row, columns, above_id=3)
    assert "id" not in columns and {"segment_id", "text", "speaker_source", "loudness_db", "system_share"} <= set(columns)
    assert (b, c) == (41, 42) and a not in (999, b, c)
    got = pdb.one("SELECT * FROM utterances WHERE id=?", (a,))
    assert (got["speaker_source"], got["text"], got["cluster_label"], got["loudness_db"]) == ("user", "文", None, None)
    if "flags" in columns:                                      # a 1.0 column: picked up from the row dict as JSON
        assert json.loads(got["flags"]) == {"guard": "loop"}
    assert [r["utterance_id"] for r in pdb.query("SELECT utterance_id FROM utterance_tokens ORDER BY utterance_id")] == [a, c]


def test_after_store_hooks_run_in_order_with_the_context_and_never_fail_the_segment(pcfg, pdb, monkeypatch):
    seg = make_segment(pcfg, pdb, datetime(2026, 11, 2, 23, 59, 58), 6.0)
    seen: list[tuple] = []

    class P(Processor):
        def _voice_after_store(self, seg_id, utterance_ids, days):
            committed = [r["id"] for r in self.db.query("SELECT id FROM utterances WHERE segment_id=? ORDER BY id", (seg_id,))]
            seen.append(("voice", seg_id, list(utterance_ids), list(days), self.store_ctx, committed,
                         self.db.get_segment(seg_id)["status"]))

        def _text_after_store(self, seg_id, utterance_ids, days):
            seen.append(("text", seg_id, list(utterance_ids), list(days)))
            raise RuntimeError("hook exploded")

    monkeypatch.setattr(pipeline, "AFTER_STORE_HOOKS",
                        [lambda p_, seg_id, ids, days: seen.append(("extra", p_, seg_id, list(ids), list(days)))])
    p = proc(pcfg, pdb, Engine([Turn(0.0, 2.5, 0), Turn(3.0, 5.5, 1)]), ASR("你好世界", "天气不错"), Replace(), cls=P)
    assert p.process_segment(seg) == 2
    ids = [u["id"] for u in utterances(pdb, seg)]
    days = ["2026-11-02", "2026-11-03"]                         # the second sentence starts after midnight
    assert [s[0] for s in seen] == ["voice", "text", "extra"]
    assert seen[0][1:4] == (seg, ids, days) and seen[1][1:] == (seg, ids, days) and seen[2][1:] == (p, seg, ids, days)
    ctx = seen[0][4]
    assert isinstance(ctx, StoreContext) and ctx.origin == "process" and ctx.replaced_utterance_id is None
    assert ctx.utterance_ids == ids and ctx.days == days and int(ctx.seg["id"]) == seg and len(ctx.audio) == 6 * SR
    assert [(c.cluster, raw) for c, raw in ctx.spoken] == [(0, "你好世界"), (1, "天气不错")]
    assert [(r["text_raw"], r["text"], r["day"]) for r in ctx.rows] == [("你好世界", "你好世间", days[0]), ("天气不错", "天气不错", days[1])]
    assert ctx.edits == [{"i": 0, **EDIT}]
    assert seen[0][5] == ids and seen[0][6] == "processing"     # rows are committed, the segment is not closed yet
    # the failing hook changed nothing: the segment is done, the context is released
    assert pdb.get_segment(seg)["status"] == "done" and p.store_ctx is None
    assert all((pcfg.transcripts_dir / f"{d}.md").exists() for d in days)

    # nothing stored (ASR heard nothing) -> no hook call
    seen.clear()
    seg2 = make_segment(pcfg, pdb, datetime(2026, 11, 4, 9, 0, 0), 3.0, seed=2)
    assert proc(pcfg, pdb, Engine([Turn(0.0, 3.0, 0)]), ASR(""), cls=P).process_segment(seg2) == 0
    assert seen == [] and pdb.get_segment(seg2)["status"] == "done"


def test_register_after_store_adds_a_hook_once(pcfg, pdb):
    calls: list[tuple] = []

    def hook(processor, seg_id, utterance_ids, days):
        calls.append((seg_id, list(utterance_ids), list(days)))

    before = list(pipeline.AFTER_STORE_HOOKS)
    try:
        assert pipeline.register_after_store(hook) is hook and pipeline.register_after_store(hook) is hook
        assert pipeline.AFTER_STORE_HOOKS == before + [hook]
        seg = make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, 0, 0), 3.0)
        assert proc(pcfg, pdb, Engine([Turn(0.0, 3.0, 0)]), ASR()).process_segment(seg) == 1
        assert calls == [(seg, [utterances(pdb, seg)[0]["id"]], ["2026-11-02"])]
    finally:
        pipeline.AFTER_STORE_HOOKS[:] = before


def test_row_hooks_add_columns_and_scratch_travels_from_the_stages_to_the_hooks(pcfg, pdb):
    seg = make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, 0, 0), 6.0)
    seen: dict[str, Any] = {"rows": []}

    class P(Processor):
        def _voice(self, seg, audio):
            turns = super()._voice(seg, audio)
            self.scratch["clusters"] = sorted({t.cluster for t in turns})      # left for the hooks of this segment
            return turns

        def _voice_row(self, row, chunk):
            row["speaker_source"] = "user" if chunk.cluster == 0 else None      # any real column is stored with the row
            row["not_a_column"] = 1                                             # anything else is ignored

        def _text_row(self, row, chunk, raw):
            seen["rows"].append((chunk.cluster, raw, row["text"]))
            if chunk.cluster == 1:
                raise RuntimeError("row hook exploded")                         # costs the extra columns, not the sentence

        def _voice_after_store(self, seg_id, utterance_ids, days):
            seen["scratch"] = dict(self.scratch)
            seen["extra_is_scratch"] = self.store_ctx.extra is self.scratch

    eng = Engine([Turn(0.0, 2.5, 0), Turn(3.0, 5.5, 1)])
    p = proc(pcfg, pdb, eng, ASR("你好世界", "天气不错"), Replace(), cls=P)
    assert p.process_segment(seg) == 2
    us = utterances(pdb, seg)
    assert [(u["speaker_source"], u["text"]) for u in us] == [("user", "你好世间"), (None, "天气不错")]
    assert seen["rows"] == [(0, "你好世界", "你好世间"), (1, "天气不错", "天气不错")]
    # (the voice stage keeps its own entry there since speakers v5: the turns as the engine decided them)
    assert seen["scratch"]["clusters"] == [0, 1] and set(seen["scratch"]) <= {"clusters", "voice"}
    assert seen["extra_is_scratch"] is True
    assert p.scratch == {} and p.store_ctx is None                              # nothing of the segment is kept alive

    # the pieces of a split go through the same row hooks (piece 0 keeps cluster 0, piece 1 gets a new one)
    seen["rows"].clear()
    p2 = proc(pcfg, pdb, eng, ASR("你好", "世界"), Replace(), cls=P)
    p2.scratch["stale"] = True                                                  # a split starts with an empty scratch
    res = p2.split_utterance(us[0]["id"], at=[1.25])
    a, b = [x["id"] for x in res["pieces"]]
    rows = {u["id"]: u for u in utterances(pdb, seg)}
    assert (rows[a]["speaker_source"], rows[b]["speaker_source"]) == ("user", None)
    assert seen["rows"] == [(0, "你好", "你好"), (2, "世界", "世间")] and seen["scratch"] == {}

    # a segment that fails half-way does not leave its scratch behind either
    class Failing(P):
        def _transcribe(self, seg, audio, chunks):
            assert self.scratch["clusters"] == [0, 1]
            raise LLMError("asr: giving up after 4 attempts")

    seg2 = make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, 5, 0), 6.0, seed=1)
    p3 = proc(pcfg, pdb, eng, ASR(), cls=Failing)
    with pytest.raises(LLMError):
        p3.process_segment(seg2)
    assert p3.scratch == {} and p3.store_ctx is None and pdb.get_segment(seg2)["status"] == "failed"


def _fake_metrics(monkeypatch, fail: str | None = None) -> list[tuple[str, float]]:
    """Install a stand-in audio_note.metrics whose stage() records (name, audio_s); `fail`: 'enter' | 'exit' | None."""
    calls: list[tuple[str, float]] = []
    mod = types.ModuleType("audio_note.metrics")

    @contextmanager
    def stage(db, name, audio_s=0.0, n=1):
        if fail == "enter":
            raise RuntimeError("metrics down")
        calls.append((name, round(float(audio_s), 2)))
        yield
        if fail == "exit":
            raise RuntimeError("metrics down")

    mod.stage = stage
    monkeypatch.setitem(sys.modules, "audio_note.metrics", mod)
    monkeypatch.setattr(audio_note, "metrics", mod, raising=False)
    return calls


VOICE_STAGE_NAMES = {"gate", "diarize", "embed", "identify"}      # SPEC_v1 5.2: what the voice stage may account for


def test_stages_are_timed_through_metrics_stage(pcfg, pdb, monkeypatch):
    calls = _fake_metrics(monkeypatch)
    seg = make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, 0, 0), 6.0)
    p = proc(pcfg, pdb, Engine([Turn(0.0, 2.5, 0), Turn(3.0, 5.5, 1)]), ASR("你好世界"), Replace())
    assert p.process_segment(seg) == 2
    names = [name for name, _ in calls]
    # the voice stage accounts for its own steps (its internals may change), then ASR, correction, storage
    assert names[-3:] == ["asr", "correct", "store"]
    assert names[:-3] and set(names[:-3]) <= VOICE_STAGE_NAMES
    by = dict(calls)
    assert all(0.0 < audio_s <= 6.0 for name, audio_s in calls if name in VOICE_STAGE_NAMES)
    assert by["asr"] == pytest.approx(2.65 + 2.8)               # the padded audio that was sent: 0-2.65 s and 2.85-5.65 s
    assert by["correct"] == by["store"] == 0.0

    # no turns: only the voice stage ran
    calls.clear()
    seg2 = make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, 5, 0), 3.0, seed=1)
    assert proc(pcfg, pdb, Engine([]), ASR()).process_segment(seg2) == 0
    assert calls and {name for name, _ in calls} <= VOICE_STAGE_NAMES


@pytest.mark.parametrize("fail", ["enter", "exit", "missing"])
def test_broken_or_missing_metrics_never_gets_in_the_way(pcfg, pdb, monkeypatch, fail):
    if fail == "missing":
        monkeypatch.setitem(sys.modules, "audio_note.metrics", None)       # import audio_note.metrics -> ImportError
        monkeypatch.delattr(audio_note, "metrics", raising=False)
    else:
        _fake_metrics(monkeypatch, fail)
    seg = make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, 0, 0), 3.0)
    assert proc(pcfg, pdb, Engine([Turn(0.0, 3.0, 0)]), ASR("你好世界"), Replace()).process_segment(seg) == 1
    assert pdb.get_segment(seg)["status"] == "done" and utterances(pdb, seg)[0]["text"] == "你好世间"
    # the stage's own exception comes out unchanged
    with pytest.raises(KeyError, match="mine"):
        with pipeline.stage_timer(pdb, "asr", 1.0):
            raise KeyError("mine")


def test_stage_timer_reports_a_failing_stage_to_metrics_and_reraises(monkeypatch):
    seen: list[tuple[str, str | None]] = []
    mod = types.ModuleType("audio_note.metrics")

    @contextmanager
    def stage(db, name, audio_s=0.0, n=1):
        try:
            yield
        except Exception as e:
            seen.append((name, type(e).__name__))
            raise
        else:
            seen.append((name, None))

    mod.stage = stage
    monkeypatch.setitem(sys.modules, "audio_note.metrics", mod)
    monkeypatch.setattr(audio_note, "metrics", mod, raising=False)
    with pipeline.stage_timer(None, "store"):
        pass
    with pytest.raises(Stopped):
        with pipeline.stage_timer(None, "asr"):
            raise Stopped("stopped")
    assert seen == [("store", None), ("asr", "Stopped")]


# =========================================================================================================
# 3. (a) a stop request in the middle of a segment
# =========================================================================================================
THREE_TURNS = [Turn(0.0, 2.5, 0), Turn(3.0, 5.5, 1), Turn(6.0, 8.5, 0)]


def test_stopped_is_a_runtime_error_with_the_old_message():
    e = Stopped("stopped")
    assert isinstance(e, RuntimeError) and str(e) == "stopped" and pipeline.Stopped is Stopped
    stop = threading.Event()
    stop.set()
    asr = ASR()
    chunks = pipeline.build_chunks(THREE_TURNS, 9.0, 60.0)
    for workers in (1, 4):
        with pytest.raises(Stopped, match="^stopped$"):
            pipeline.transcribe_chunks(asr, np.zeros(9 * SR, dtype=np.float32), chunks, workers, stop_event=stop)
    assert asr.calls == 0


def test_stop_while_chunks_are_being_transcribed_leaves_the_segment_retryable(pcfg, pdb):
    seg = make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, 0, 0), 9.0)
    stop = threading.Event()
    asr = ASR("你好世界", on_call=lambda k: stop.set())      # the stop request arrives during the first chunk
    p = proc(pcfg, pdb, Engine(THREE_TURNS), asr, Replace(), stop_event=stop)
    with pytest.raises(Stopped) as ex:
        p.process_segment(seg)
    assert isinstance(ex.value, RuntimeError) and str(ex.value) == "stopped"
    s = pdb.get_segment(seg)
    assert s["status"] == "new" and s["error"] is None and Path(s["path"]).exists()
    assert asr.calls == 1                                           # the other two chunks were never sent
    assert utterances(pdb, seg) == [] and pdb.query("SELECT id FROM transcripts") == []

    # the service comes back: the same segment is processed from the start, nothing is left 'failed'
    stop.clear()
    asr2 = ASR("你好世界")
    stats = proc(pcfg, pdb, Engine(THREE_TURNS), asr2, Replace(), stop_event=stop).process_pending()
    assert stats == {"processed": 1, "done": 1, "no_speech": 0, "failed": 0, "utterances": 3}
    assert asr2.calls == 3 and pdb.get_segment(seg)["status"] == "done" and len(utterances(pdb, seg)) == 3
    assert pdb.one("SELECT COUNT(*) AS n FROM transcripts")["n"] == 1


def test_stop_with_parallel_asr_requests_is_not_a_failure_either(pcfg, pdb):
    pcfg["mimo"]["asr_concurrency"] = 2
    seg = make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, 0, 0), 12.0)
    stop = threading.Event()
    asr = ASR("你好", on_call=lambda k: stop.set())
    turns = [Turn(0.0, 2.5, 0), Turn(3.0, 5.5, 1), Turn(6.0, 8.5, 0), Turn(9.0, 11.5, 1)]
    with pytest.raises(Stopped):
        proc(pcfg, pdb, Engine(turns), asr, stop_event=stop).process_segment(seg)
    assert 1 <= asr.calls <= 2 and pdb.get_segment(seg)["status"] == "new" and utterances(pdb, seg) == []


def test_process_pending_ends_the_run_and_counts_nothing_for_the_interrupted_segment(pcfg, pdb, caplog):
    ids = [make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, i, 0), 6.0, seed=i) for i in range(3)]
    stop = threading.Event()
    asr = ASR("你好", on_call=lambda k: k == 3 and stop.set())      # first chunk of the SECOND segment
    p = proc(pcfg, pdb, Engine([Turn(0.0, 2.5, 0), Turn(3.0, 5.5, 1)]), asr, stop_event=stop)
    with caplog.at_level("INFO", logger="audio_note.speech.pipeline"):
        stats = p.process_pending()
    assert stats == {"processed": 1, "done": 1, "no_speech": 0, "failed": 0, "utterances": 2}
    assert [pdb.get_segment(i)["status"] for i in ids] == ["done", "new", "new"] and asr.calls == 3
    assert all(pdb.get_segment(i)["error"] is None for i in ids)
    assert f"stop requested during segment #{ids[1]}, 2 segment(s) left" in caplog.text
    assert not [r for r in caplog.records if r.levelname in ("ERROR", "CRITICAL")]


def test_a_stage_may_raise_stopped_itself_but_other_errors_still_fail_the_segment(pcfg, pdb):
    class Interrupted(Engine):
        def diarize(self, audio):
            raise Stopped("stopped")

    seg = make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, 0, 0), 3.0)
    with pytest.raises(Stopped):
        proc(pcfg, pdb, Interrupted(), ASR()).process_segment(seg)
    assert pdb.get_segment(seg)["status"] == "new"

    class Broken(Engine):
        def diarize(self, audio):
            raise RuntimeError("stopped")              # not the dedicated class: an ordinary failure

    with pytest.raises(RuntimeError):
        proc(pcfg, pdb, Broken(), ASR()).process_segment(seg)
    s = pdb.get_segment(seg)
    assert s["status"] == "failed" and s["error"] == "RuntimeError: stopped"


# =========================================================================================================
# 4. (b) split: the pieces' edits and the transcripts row
# =========================================================================================================
def _processed_two_sentences(pcfg, pdb) -> tuple[int, int, int, Engine]:
    seg = make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, 0, 0), 6.0)
    eng = Engine([Turn(0.0, 3.0, 0), Turn(3.0, 6.0, 1)])
    assert proc(pcfg, pdb, eng, ASR("你好世界", "大家好世界"), Replace()).process_segment(seg) == 2
    first, second = [u["id"] for u in utterances(pdb, seg)]
    return seg, first, second, eng


def _items(tr: Any) -> list[tuple]:
    return [(it["i"], it["utterance_id"], it["start_s"], it["end_s"], it["text_raw"], it["text"]) for it in json.loads(tr["raw_json"])]


def test_split_keeps_the_pieces_edits_and_the_transcript_follows_the_rows(pcfg, pdb):
    seg, first, second, eng = _processed_two_sentences(pcfg, pdb)
    tr = transcript(pdb, seg)
    assert json.loads(tr["correction_json"]) == [{"i": 0, **EDIT}, {"i": 1, **EDIT}]
    assert all("utterance_id" not in it for it in json.loads(tr["raw_json"]))       # process_segment's own format is unchanged

    seen: list[tuple] = []

    class P(Processor):
        def _text_after_store(self, seg_id, utterance_ids, days):
            seen.append((seg_id, list(utterance_ids), list(days), self.store_ctx))

    res = proc(pcfg, pdb, eng, ASR("世界很大", "天气不错"), Replace(), cls=P).split_utterance(first, at=[1.5])
    a, b = [x["id"] for x in res["pieces"]]
    assert [x["text"] for x in res["pieces"]] == ["世间很大", "天气不错"]

    tr = transcript(pdb, seg)
    assert _items(tr) == [(0, a, 0.0, 1.5, "世界很大", "世间很大"),
                          (1, b, 1.5, 3.0, "天气不错", "天气不错"),
                          (2, second, 3.0, 6.0, "大家好世界", "大家好世间")]
    assert json.loads(tr["correction_json"]) == [
        {"i": 0, **EDIT},       # the edit the corrector made on the first piece is kept
        {"i": 2, **EDIT},       # the untouched sentence's edit follows it to its new index
    ]
    assert tr["text_raw"] == "世界很大\n天气不错\n大家好世界" and tr["text_corrected"] == "世间很大\n天气不错\n大家好世间"
    # the transcript describes exactly the rows that exist: nothing is left of the removed one
    rows = utterances(pdb, seg)
    assert [(it[1], it[4], it[5]) for it in _items(tr)] == [(r["id"], r["text_raw"], r["text"]) for r in rows]
    assert first not in [it[1] for it in _items(tr)] and "你好世界" not in tr["raw_json"] + tr["text_raw"]
    # the after-store hooks get the pieces, with their edits
    ((sid, ids, days, ctx),) = seen
    assert (sid, ids, days) == (seg, [a, b], ["2026-11-02"])
    assert ctx.origin == "split" and ctx.replaced_utterance_id == first and ctx.utterance_ids == [a, b]
    assert ctx.edits == [{"i": 0, **EDIT}] and [raw for _, raw in ctx.spoken] == ["世界很大", "天气不错"]

    # splitting a piece again: the items written by the first split are found by their utterance id
    res2 = proc(pcfg, pdb, eng, ASR("天气", "不错世界"), Replace()).split_utterance(b, at=[0.75])
    c, d = [x["id"] for x in res2["pieces"]]
    tr = transcript(pdb, seg)
    assert _items(tr) == [(0, a, 0.0, 1.5, "世界很大", "世间很大"), (1, c, 1.5, 2.25, "天气", "天气"),
                          (2, d, 2.25, 3.0, "不错世界", "不错世间"), (3, second, 3.0, 6.0, "大家好世界", "大家好世间")]
    assert json.loads(tr["correction_json"]) == [{"i": 0, **EDIT}, {"i": 2, **EDIT}, {"i": 3, **EDIT}]
    assert [r["id"] for r in utterances(pdb, seg)] == [a, c, d, second]


def test_split_without_edits_clears_stale_ones_and_rebuilds_an_unreadable_transcript(pcfg, pdb):
    seg, first, second, eng = _processed_two_sentences(pcfg, pdb)
    # an old / damaged transcript: items unreadable, an edit pointing nowhere
    pdb.execute("UPDATE transcripts SET raw_json='not json', correction_json=? WHERE segment_id=?",
                (json.dumps([{"i": 5, "from": "甲", "to": "乙"}]), seg))
    res = proc(pcfg, pdb, eng, ASR("今天", "明天")).split_utterance(first, at=[1.5])      # no corrector: no edits
    a, b = [x["id"] for x in res["pieces"]]
    tr = transcript(pdb, seg)
    assert _items(tr) == [(0, a, 0.0, 1.5, "今天", "今天"), (1, b, 1.5, 3.0, "明天", "明天"),
                          (2, second, 3.0, 6.0, "大家好世界", "大家好世间")]       # described from the row itself
    assert json.loads(tr["raw_json"])[2]["speaker_name"] == pcfg.speakers.unknown_label
    assert tr["correction_json"] is None

    # no transcripts row at all (very old data): one is written for the rows that exist
    pdb.execute("DELETE FROM transcripts WHERE segment_id=?", (seg,))
    res2 = proc(pcfg, pdb, eng, ASR("大家好", "世界"), Replace()).split_utterance(second, at=[1.5])
    c, d = [x["id"] for x in res2["pieces"]]
    tr = transcript(pdb, seg)
    assert [it[:2] for it in _items(tr)] == [(0, a), (1, b), (2, c), (3, d)]
    assert json.loads(tr["correction_json"]) == [{"i": 3, **EDIT}] and tr["asr_model"] == pcfg.mimo.asr_model


def test_split_is_all_or_nothing_including_the_transcript(pcfg, pdb, monkeypatch):
    seg, first, second, eng = _processed_two_sentences(pcfg, pdb)
    before = (dict(transcript(pdb, seg)), [dict(u) for u in utterances(pdb, seg)])

    def boom(*a, **k):
        raise RuntimeError("transcript refresh failed")

    monkeypatch.setattr(pipeline, "refresh_transcript", boom)
    with pytest.raises(RuntimeError, match="refresh failed"):
        proc(pcfg, pdb, eng, ASR("今天", "明天"), Replace()).split_utterance(first, at=[1.5])
    assert (dict(transcript(pdb, seg)), [dict(u) for u in utterances(pdb, seg)]) == before


# =========================================================================================================
# 5. (c) one glossary parser
# =========================================================================================================
GLOSSARY_TEXT = "# 术语表\n\n灰度发布 # 上线方式\n  星河App  \nC#\nF# 语言\n灰度发布\n王敏\t# 同事\n#只有注释\n   # 缩进的注释\n"
GLOSSARY_TERMS = ["灰度发布", "星河App", "C#", "F# 语言", "王敏"]


def test_glossary_file_rules(tmp_path):
    assert glossary_file.parse_glossary(GLOSSARY_TEXT) == GLOSSARY_TERMS
    assert glossary_file.parse_glossary(GLOSSARY_TEXT.replace("\n", "\r\n")) == GLOSSARY_TERMS
    assert glossary_file.parse_glossary("") == [] and glossary_file.parse_glossary(None) == []
    assert [glossary_file.parse_line(s) for s in ("", "  ", "# x", " 词 ", "词 #", "词#1")] == [None, None, None, "词", "词", "词#1"]
    p = tmp_path / "glossary.txt"
    p.write_text(GLOSSARY_TEXT, encoding="utf-8-sig")              # Notepad's BOM is not part of the first term
    assert glossary_file.read_glossary(p) == GLOSSARY_TERMS and glossary_file.read_glossary(str(p)) == GLOSSARY_TERMS
    assert glossary_file.read_glossary(None) == [] and glossary_file.read_glossary("") == []
    assert glossary_file.read_glossary(tmp_path / "missing.txt") == [] and glossary_file.read_glossary(tmp_path) == []
    (tmp_path / "bad.txt").write_bytes(bytes([0xFF, 0xFE, 0x00, 0x41]))
    with pytest.raises(UnicodeDecodeError):
        glossary_file.read_glossary(tmp_path / "bad.txt")


def test_pipeline_glossary_uses_the_shared_parser_and_adds_speakers(pcfg, pdb, caplog):
    pcfg.glossary_path.write_text(GLOSSARY_TEXT, encoding="utf-8")
    assert pipeline.load_glossary(pcfg) == GLOSSARY_TERMS                          # inline comments no longer leak
    pdb.upsert_speaker("王敏")
    pdb.upsert_speaker("李四")
    assert pipeline.load_glossary(pcfg, pdb) == GLOSSARY_TERMS + ["李四"]            # no duplicate for 王敏
    # what the corrector is given
    seg = make_segment(pcfg, pdb, datetime(2026, 11, 2, 9, 0, 0), 3.0)
    corr = Replace()
    assert proc(pcfg, pdb, Engine([Turn(0.0, 3.0, 0)]), ASR(), corr).process_segment(seg) == 1
    assert corr.calls[0][1] == GLOSSARY_TERMS + ["李四"]
    # an unreadable file is a warning, not an error: the speakers are still there
    pcfg.glossary_path.write_bytes(bytes([0xFF, 0xFE, 0x00, 0x41]))
    with caplog.at_level("WARNING", logger="audio_note.speech.pipeline"):
        assert pipeline.load_glossary(pcfg, pdb) == ["王敏", "李四"]
    assert "cannot read glossary" in caplog.text


@pytest.mark.parametrize("module", ["audio_note.speech.correct", "audio_note.memory.index"])
def test_the_other_two_glossary_loaders_agree_with_the_shared_parser(tmp_path, module):
    import importlib

    p = tmp_path / "glossary.txt"
    p.write_text(GLOSSARY_TEXT, encoding="utf-8")
    assert importlib.import_module(module).load_glossary(p) == glossary_file.read_glossary(p)


# =========================================================================================================
# 6. (d) embeddings for retrieval
# =========================================================================================================
@pytest.fixture
def ecfg(cfg_factory, tmp_path: Path):
    return cfg_factory({
        "pipeline": {"min_segment_age_s": 0},
        "correction": {"glossary_file": str(tmp_path / "glossary.txt")},
        "memory": {"retrieval": "hybrid", "embedding_model": "hash"},
        "mimo": {"asr_concurrency": 1},
    })


def _flaky_embedder(monkeypatch, failures: int, load_error: Exception | None = None) -> dict[str, int]:
    """memory.embed.get_embedder -> a HashEmbedder whose first `failures` embedding passes raise."""
    import audio_note.memory.embed as emod

    state = {"built": 0, "loads": 0, "passes": 0, "left": failures}

    class Flaky(emod.HashEmbedder):
        def load(self):
            state["loads"] += 1
            if load_error is not None:
                raise load_error

        def embed_passages(self, texts):
            state["passes"] += 1
            if state["left"] > 0:
                state["left"] -= 1
                raise RuntimeError("onnxruntime hiccup")
            return super().embed_passages(texts)

    def build(cfg):
        state["built"] += 1
        return Flaky()

    monkeypatch.setattr(emod, "get_embedder", build)
    return state


def test_embedding_retries_with_the_next_segment_after_a_transient_failure(ecfg, monkeypatch, caplog):
    from audio_note.memory import vectors

    db = DB(ecfg.db_path)
    state = _flaky_embedder(monkeypatch, failures=1)
    p = proc(ecfg, db, Engine([Turn(0.0, 3.0, 0)]), ASR("你好世界"))
    seg1 = make_segment(ecfg, db, datetime(2026, 11, 2, 9, 0, 0), 3.0)
    with caplog.at_level("WARNING", logger="audio_note.speech.pipeline"):
        assert p.process_segment(seg1) == 1
    assert db.get_segment(seg1)["status"] == "done" and db.vector_count() == 0
    assert "the next segment retries" in caplog.text and p._embedder is not False and p._embedder is not None

    seg2 = make_segment(ecfg, db, datetime(2026, 11, 2, 9, 5, 0), 3.0, seed=1)
    assert p.process_segment(seg2) == 1
    assert db.vector_count("hash") == 2 and vectors.missing_count(db, "hash") == 0      # the first sentence was picked up too
    assert state["built"] == 1 and state["loads"] == 1 and state["passes"] == 2 and p._embed_failures == 0


def test_embedding_gives_up_when_the_model_cannot_be_loaded(ecfg, monkeypatch, caplog):
    db = DB(ecfg.db_path)
    state = _flaky_embedder(monkeypatch, failures=0, load_error=FileNotFoundError("model files missing"))
    p = proc(ecfg, db, Engine([Turn(0.0, 3.0, 0)]), ASR("你好世界"))
    with caplog.at_level("WARNING", logger="audio_note.speech.pipeline"):
        for i in range(3):
            seg = make_segment(ecfg, db, datetime(2026, 11, 2, 9, i, 0), 3.0, seed=i)
            assert p.process_segment(seg) == 1
    assert p._embedder is False and state == {"built": 1, "loads": 1, "passes": 0, "left": 0}     # asked once, not per segment
    assert caplog.text.count("embedding skipped") == 1 and "model files missing" in caplog.text
    assert db.vector_count() == 0


def test_embedding_stops_after_repeated_failures_in_a_row(ecfg, monkeypatch, caplog):
    db = DB(ecfg.db_path)
    monkeypatch.setattr(pipeline, "EMBED_MAX_FAILURES", 2)
    state = _flaky_embedder(monkeypatch, failures=99)
    p = proc(ecfg, db, Engine([Turn(0.0, 3.0, 0)]), ASR("你好世界"))
    with caplog.at_level("WARNING", logger="audio_note.speech.pipeline"):
        for i in range(4):
            seg = make_segment(ecfg, db, datetime(2026, 11, 2, 9, i, 0), 3.0, seed=i)
            assert p.process_segment(seg) == 1
    assert p._embedder is False and state["passes"] == 2 and state["built"] == 1
    assert caplog.text.count("the next segment retries") == 1 and caplog.text.count("embedding skipped") == 1
