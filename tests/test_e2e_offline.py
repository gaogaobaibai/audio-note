"""Offline end-to-end test: real VAD + real speaker models, fake MiMo.

import_file (real Silero VAD) -> Processor.process_pending (real SpeakerEngine, fake ASR / corrector)
-> transcripts Markdown -> DailyAnalyzer.run (fake map/reduce) -> memory.index.search -> MemoryChat.ask
(fake client issuing one tool call). Skipped when the sample recording is not available.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from .conftest import MODELS_DIR, SAMPLE_WAV

pytestmark = pytest.mark.skipif(not SAMPLE_WAV.exists(), reason="real sample recording not available")

ASR_TEXT = "这是第{n}段测试文本，雷军在发布会上讲话。"
MAP_RESULT = {
    "title": "发布会讲话", "summary": "雷军在发布会上讲话，介绍了测试文本。",
    "topics": ["发布会"], "decisions": ["继续测试"], "todos": ["整理发布会记录"],
    "people": ["雷军"], "mood": "平静", "open_questions": ["下一场发布会时间"],
}
REDUCE_RESULT = {
    "overview": "今天主要是雷军的发布会讲话。", "topics": ["发布会", "测试"],
    "decisions": ["继续测试"], "todos": [{"text": "整理发布会记录", "owner": "我"}],
    "people": [{"name": "雷军", "context": "在发布会上讲话"}], "mood": "平静",
    "open_loops": ["下一场发布会时间"], "timeline": [{"time": "09:00", "event": "发布会开始"}],
}
FINAL_ANSWER = "根据记录，10-30 09:00 雷军在发布会上讲话。"


def _message(content: str | None = None, tool_calls: list[Any] | None = None) -> Any:
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=None)


class FakeMimo:
    """Same method names as llm.client.MimoClient; nothing leaves the process."""

    def __init__(self) -> None:
        self.asr_calls = 0
        self.json_calls: list[tuple[str, str]] = []
        self.chat_calls: list[list[dict]] = []
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "audio_seconds": 0.0}

    # ASR
    def transcribe_bytes(self, wav: bytes, mime: str = "audio/wav", language: str = "auto",
                         model: str | None = None) -> tuple[str, float]:
        assert wav[:4] == b"RIFF" and mime == "audio/wav"
        self.asr_calls += 1
        self.usage["audio_seconds"] += 3.0
        return ASR_TEXT.format(n=self.asr_calls), 3.0

    # correction (edits) + daily analysis (map / reduce), routed on the prompt
    def json(self, system: str, user: str, **kw: Any) -> Any:
        self.json_calls.append((system, user))
        if "open_questions" in system:
            return dict(MAP_RESULT)
        if "open_loops" in system:
            return dict(REDUCE_RESULT)
        return {"edits": []}

    def text(self, system: str, user: str, **kw: Any) -> str:
        return json.dumps(self.json(system, user, **kw), ensure_ascii=False)

    # memory chat: one search_transcripts tool call, then the answer
    def chat(self, messages: list[dict], **kw: Any) -> Any:
        self.chat_calls.append([dict(m) for m in messages])
        if kw.get("tools") and len(self.chat_calls) == 1:
            call = SimpleNamespace(id="call_1", type="function",
                                   function=SimpleNamespace(name="search_transcripts",
                                                            arguments=json.dumps({"query": "雷军"}, ensure_ascii=False)))
            return _message(None, [call])
        return _message(FINAL_ANSWER, None)


@pytest.fixture
def e2e_cfg(cfg_factory, tmp_path: Path):
    glossary = tmp_path / "glossary.txt"
    glossary.write_text("# 测试词表\n雷军\n小米汽车\n", encoding="utf-8")
    return cfg_factory({
        "audio": {"min_silence_s": 1.0},            # 272 s sample -> ~29 segments (2.0 s gives only 9)
        "pipeline": {"min_segment_age_s": 0},       # freshly imported rows are picked up at once
        "speakers": {"num_threads": 2},
        # mode segment: this test follows the per-segment LLM corrector (since 1.0 the default first pass is the
        # lexicon's deterministic one, correction.mode = session, which makes no LLM call)
        "correction": {"glossary_file": str(glossary), "mode": "segment"},
    })


def test_offline_end_to_end(e2e_cfg, db, silero):
    cfg = e2e_cfg
    for key in ("vad_model", "embedding_model", "segmentation_model"):
        if not cfg.model_path(key).exists():
            pytest.skip(f"model missing: {cfg.model_path(key)}")
    assert cfg.models_dir == MODELS_DIR

    from audio_note.audio.importer import import_file
    from audio_note.analysis.daily import DailyAnalyzer
    from audio_note.memory import index as mem_index
    from audio_note.memory.chat import MemoryChat
    from audio_note.speech.pipeline import Processor
    from audio_note.speech.speakers import SpeakerEngine

    # 1. import with the real VAD
    start = datetime(2026, 10, 30, 9, 0, 0)
    seg_ids = import_file(cfg, db, SAMPLE_WAV, start_time=start, vad=silero)
    assert len(seg_ids) >= 10
    assert all(Path(r["path"]).exists() for r in db.query("SELECT path FROM segments"))
    assert all(str(cfg.data_dir) in r["path"] for r in db.query("SELECT path FROM segments"))

    # 2. process six segments: real diarization + identification, fake ASR / correction
    client = FakeMimo()
    proc = Processor(cfg, db, client=client, engine=SpeakerEngine(cfg))
    stats = proc.process_pending(limit=6)
    assert stats["processed"] == 6 and stats["failed"] == 0
    assert stats["done"] + stats["no_speech"] == 6
    assert stats["done"] >= 1 and stats["utterances"] >= 1
    assert client.asr_calls >= 1
    assert db.one("SELECT COUNT(*) AS n FROM segments WHERE status='new'")["n"] == len(seg_ids) - 6
    utts = db.query("SELECT * FROM utterances ORDER BY abs_start, start_s")
    assert len(utts) == stats["utterances"]
    assert all(u["day"] == "2026-10-30" for u in utts)
    assert all(u["speaker_name"] == cfg.speakers.unknown_label for u in utts)   # nobody enrolled
    assert all("雷军" in u["text"] and u["text"] == u["text_raw"] for u in utts)  # edits=[] keeps raw
    assert db.one("SELECT COUNT(*) AS n FROM utterance_tokens")["n"] == len(utts)
    assert db.one("SELECT COUNT(*) AS n FROM transcripts")["n"] == stats["done"]
    tr = db.one("SELECT * FROM transcripts ORDER BY id LIMIT 1")
    assert tr["asr_model"] == cfg.mimo.asr_model and json.loads(tr["raw_json"])
    # the corrector saw the glossary term and the unknown speaker label
    correction_prompts = [s + u for s, u in client.json_calls if "open_" not in s]
    assert correction_prompts and all("雷军" in p for p in correction_prompts)

    md_path = cfg.transcripts_dir / "2026-10-30.md"
    md = md_path.read_text(encoding="utf-8")
    assert md.startswith("# 2026-10-30 记录\n")
    assert f"（导入：{SAMPLE_WAV.name}）" in md
    assert re.search(r"^- \*\*09:\d\d:\d\d 未知\*\*：这是第\d+段测试文本", md, re.M)
    assert "\r" not in md

    # 3. daily digest with canned map / reduce
    analyzer = DailyAnalyzer(cfg, db, client)
    daily_md, daily_js = analyzer.run("2026-10-30")
    assert daily_md.startswith("# 2026-10-30 每日记录")
    for section in ("## 概览", "## 主要话题", "## 决定", "## 待办", "## 提到的人", "## 情绪", "## 未闭环", "## 时间线"):
        assert section in daily_md
    assert "雷军" in daily_md and "整理发布会记录" in daily_md
    assert daily_js["utterance_count"] == len(utts) and daily_js["block_count"] >= 1
    assert (cfg.daily_dir / "2026-10-30.md").read_text(encoding="utf-8") == daily_md
    row = db.one("SELECT * FROM daily_summaries WHERE day='2026-10-30'")
    assert row and row["utterance_count"] == len(utts) and row["pushed_at"] is None
    assert db.one("SELECT COUNT(*) AS n FROM memory_items WHERE kind='person' AND title='雷军'")["n"] == 1

    # 4. full-text search
    hits = mem_index.search(db, "雷军")
    assert hits and all("雷军" in h.text for h in hits)
    assert mem_index.search(db, "雷军", speaker="张三") == []
    assert mem_index.search(db, "发布会 讲话", day_from="2026-10-30", day_to="2026-10-30")

    # 5. memory chat: one tool round, then the answer
    chat_client = FakeMimo()
    chat = MemoryChat(cfg, db, chat_client)
    answer = chat.ask("雷军说了什么？", history=[])
    assert answer == FINAL_ANSWER
    assert len(chat_client.chat_calls) == 2
    second = chat_client.chat_calls[1]
    assert second[0]["role"] == "system" and "2026" in second[0]["content"]
    tool_msgs = [m for m in second if m["role"] == "tool"]
    assert len(tool_msgs) == 1 and tool_msgs[0]["tool_call_id"] == "call_1"
    assert "雷军" in tool_msgs[0]["content"] and "找到" in tool_msgs[0]["content"]
    assert any(m["role"] == "assistant" and m.get("tool_calls") for m in second)
