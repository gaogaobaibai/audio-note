"""Processing pipeline: segment file -> diarize -> identify -> ASR chunks -> correction -> DB -> daily Markdown.

See docs/DESIGN.md section 5. The Processor is meant to run in the scheduler thread while the
Recorder thread keeps writing new segments; the DB hands out one connection per thread.

This module is the FACADE of the pipeline: the code lives in the modules below, and every name the old
single-file pipeline.py had is still importable from here (`from audio_note.speech.pipeline import Processor`).

    chunking.py       Chunk, build_chunks, _clip; Stopped; stage_timer; F (late-bound view of this module)
    channels.py       channel_pairs, system_share, system_turn_share, _segment_channels
    markdown.py       write_day_markdown, write_days_markdown, segment_heading
    glossary_file.py  parse_glossary, read_glossary: the one glossary.txt parser
    stage_voice.py    VoiceStage (Processor mixin): _voice(seg, audio) -> turns; drop_faint_turns
    stage_text.py     TextStage (Processor mixin): _transcribe / _correct; transcribe_chunks, load_glossary,
                      correct_spoken
    processor.py      Processor(VoiceStage, TextStage): process_pending / process_segment, storage, after-store
                      hooks; tokenize; transcribe_file
    relabel.py        relabel_segment / relabel (Processor methods delegate)
    split.py          preview_split / split_utterance (Processor methods delegate); refresh_transcript

Monkeypatching: the modules above look up everything that used to be a global of this file THROUGH this module at
call time (chunking.F), so `monkeypatch.setattr(pipeline, "read_segment", fake)` reaches every caller, as it did
when all the code was here. Patch names on this module, not on the module that defines them.

Public pieces
-------------
Chunk                 one speaker-homogeneous stretch of audio sent to ASR as a unit
build_chunks(...)     pure function: diarization turns -> ASR chunks (merge / split / pad / drop)
Processor             process_pending() / process_segment(seg_id) / relabel_segment(seg_id) / split_utterance(utt_id)
write_day_markdown()  regenerate transcripts/YYYY-MM-DD.md from the DB
transcribe_file()     one-off diarize + identify + ASR of any audio file (no DB writes)
system_share()        share of a time range's energy that came from system audio (segments.channel_rms)
Stopped               raised when a stop request cuts a segment short (the segment goes back to 'new')
register_after_store  add a hook that runs after the utterances of a segment were committed (see processor.py)

Speakers v4: a turn whose system-audio share (segments.channel_rms, see system_share) reaches
speakers.system_turn_share is never identified as the self speaker (the other side of a meeting / call), and a
sentence is never re-identified as a speaker the user vetoed for it (utterance_rejections).
"""
from __future__ import annotations

# The plain imports of the old module stay available as attributes (pipeline.np, pipeline.json, ...).
import inspect  # noqa: F401
import json  # noqa: F401
import logging
import math  # noqa: F401
import threading  # noqa: F401
from dataclasses import dataclass  # noqa: F401
from datetime import datetime, timedelta  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any, Sequence  # noqa: F401

import numpy as np  # noqa: F401

from ..audio.loudness import is_faint, min_speech_dbfs, speech_dbfs  # noqa: F401
from ..audio.storage import read_segment  # noqa: F401
from ..audio.vad import SAMPLE_RATE  # noqa: F401
from ..config import Config  # noqa: F401
from ..db import DB  # noqa: F401
from ..llm.client import ContentFiltered, LLMError  # noqa: F401
from .channels import (  # noqa: F401
    DEFAULT_SYSTEM_TURN_SHARE,
    _segment_channels,
    channel_pairs,
    system_share,
    system_turn_share,
)
from .chunking import LOGGER_NAME, Chunk, Stopped, _clip, build_chunks, stage_timer  # noqa: F401
from .glossary_file import parse_glossary, read_glossary  # noqa: F401
from .markdown import _parse_iso, segment_heading, write_day_markdown, write_days_markdown  # noqa: F401
from .processor import (  # noqa: F401
    AFTER_STORE_HOOKS,
    EMBED_MAX_FAILURES,
    EMBED_PASS_LIMIT,
    STALE_PROCESSING_S,
    Processor,
    StoreContext,
    edits_list,
    insert_utterance,
    register_after_store,
    tokenize,
    transcribe_file,
    transcript_item,
    utterance_columns,
)
from .relabel import _row_get  # noqa: F401
from .speakers import Turn  # noqa: F401
from .split import (  # noqa: F401
    SPLIT_EDGE_S,
    SPLIT_MIN_PIECE_S,
    SPLIT_MIN_TURN_S,
    SPLIT_PAD_S,
    refresh_transcript,
)
from .stage_text import (  # noqa: F401
    TextStage,
    _asr_concurrency,
    _index_correction,
    correct_spoken,
    load_glossary,
    transcribe_chunks,
)
from .stage_voice import VoiceStage, _accepts_segment_id, drop_faint_turns  # noqa: F401

log = logging.getLogger(LOGGER_NAME)    # "audio_note.speech.pipeline": every pipeline module logs under this name

__all__ = ["Chunk", "Processor", "build_chunks", "drop_faint_turns", "write_day_markdown", "segment_heading", "transcribe_file",
           "tokenize", "load_glossary", "system_share", "channel_pairs", "system_turn_share",
           "SPLIT_MIN_PIECE_S", "SPLIT_EDGE_S", "SPLIT_MIN_TURN_S",
           "Stopped", "StoreContext", "register_after_store", "stage_timer", "write_days_markdown"]
