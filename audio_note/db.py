"""SQLite data layer. One file: data/audio_note.sqlite3, WAL mode.

Tables
------
segments           one row per stored audio file (speech-gated recording or imported chunk)
transcripts        raw + corrected ASR output per segment
utterances         speaker-attributed sentences with absolute timestamps (the unit of memory)
utterance_tokens   jieba-segmented text, mirrored into the FTS5 index utterances_fts
utterance_vectors  sentence embeddings (float32 blobs) for vector / hybrid retrieval
speakers           enrolled people (+ is_self / relation columns added by _migrate)
speaker_embeddings enrollment embeddings (float32 blobs)
speaker_negatives  negative samples ("this voice is NOT that person") + utterance_rejections (per-sentence vetoes)
cluster_embeddings one embedding per diarization cluster of a processed segment (for naming unknown voices)
daily_summaries    per-day LLM digest
memory_notes       long-lived notes of 0.3.x (kept, no longer written: copied into memory_items by migration v2)
kv                 misc state

1.0 (schema version 2, see docs/SPEC_v1.md section 3)
-----------------------------------------------------
lexicon / lexicon_aliases     canonical spellings + their misheard / alternative forms
utterance_revisions           every change of utterances.text (who, why, undo)
inbox_items                   questions the system wants to ask the user
utterance_voice               one voice embedding per utterance (computed once, re-scored with numpy)
speaker_centroids / voices    adaptive per-condition speaker centroids, persistent anonymous voices
sessions                      conversations cut by time gaps (episodes)
memory_items (+ tokens / fts) long-term memory entries with provenance and status; memory_vectors = generic vectors
chat_conversations / chat_messages   stored memory chats
usage_local / usage_llm / job_runs   per-day CPU / LLM accounting and the job history

Schema changes
--------------
* new tables: append a CREATE TABLE IF NOT EXISTS to SCHEMA;
* new columns on existing tables: append to MIGRATION_COLUMNS (never index them inside SCHEMA: SCHEMA runs first);
* data changes / indexes on migrated columns: a numbered step in MIGRATIONS (PRAGMA user_version). Before the
  first step runs on a database that already holds recordings, the file is backed up to <data_dir>/backups.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import sqlite3
import threading
import time
import unicodedata
import weakref
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    started_at TEXT NOT NULL,            -- ISO-8601 local time of first sample
    duration_s REAL NOT NULL,
    sample_rate INTEGER NOT NULL DEFAULT 16000,
    source TEXT NOT NULL DEFAULT 'mic',  -- mic | import
    origin TEXT,                          -- original file path for imports
    status TEXT NOT NULL DEFAULT 'new',  -- new | processing | done | failed | no_speech | deleted
    speech_s REAL,                       -- seconds of speech found by VAD
    error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_segments_status ON segments(status);
CREATE INDEX IF NOT EXISTS idx_segments_started ON segments(started_at);

CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY,
    segment_id INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    asr_model TEXT,
    raw_json TEXT,                       -- full ASR response (segments with timestamps)
    text_raw TEXT,
    text_corrected TEXT,
    correction_json TEXT,                -- list of {from,to,reason}
    language TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_transcripts_segment ON transcripts(segment_id);

CREATE TABLE IF NOT EXISTS speakers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS speaker_embeddings (
    id INTEGER PRIMARY KEY,
    speaker_id INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL,             -- float32 little-endian
    dim INTEGER NOT NULL,
    source_path TEXT,
    duration_s REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_spk_emb_speaker ON speaker_embeddings(speaker_id);

CREATE TABLE IF NOT EXISTS utterances (
    id INTEGER PRIMARY KEY,
    segment_id INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    start_s REAL NOT NULL,               -- offset inside the segment file
    end_s REAL NOT NULL,
    abs_start TEXT NOT NULL,             -- ISO local datetime
    day TEXT NOT NULL,                   -- YYYY-MM-DD (local)
    cluster_label INTEGER,               -- diarization cluster id within the segment
    speaker_id INTEGER REFERENCES speakers(id) ON DELETE SET NULL,
    speaker_name TEXT,                   -- denormalised for display
    speaker_score REAL,
    text_raw TEXT NOT NULL,
    text TEXT NOT NULL,                  -- corrected text
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_utt_day ON utterances(day);
CREATE INDEX IF NOT EXISTS idx_utt_abs ON utterances(abs_start);
CREATE INDEX IF NOT EXISTS idx_utt_speaker ON utterances(speaker_id);

CREATE TABLE IF NOT EXISTS utterance_tokens (
    utterance_id INTEGER PRIMARY KEY REFERENCES utterances(id) ON DELETE CASCADE,
    text_seg TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS utterances_fts USING fts5(
    text_seg, content='utterance_tokens', content_rowid='utterance_id', tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS utt_tok_ai AFTER INSERT ON utterance_tokens BEGIN
  INSERT INTO utterances_fts(rowid, text_seg) VALUES (new.utterance_id, new.text_seg);
END;
CREATE TRIGGER IF NOT EXISTS utt_tok_ad AFTER DELETE ON utterance_tokens BEGIN
  INSERT INTO utterances_fts(utterances_fts, rowid, text_seg) VALUES ('delete', old.utterance_id, old.text_seg);
END;
CREATE TRIGGER IF NOT EXISTS utt_tok_au AFTER UPDATE ON utterance_tokens BEGIN
  INSERT INTO utterances_fts(utterances_fts, rowid, text_seg) VALUES ('delete', old.utterance_id, old.text_seg);
  INSERT INTO utterances_fts(rowid, text_seg) VALUES (new.utterance_id, new.text_seg);
END;

CREATE TABLE IF NOT EXISTS utterance_vectors (
    utterance_id INTEGER PRIMARY KEY REFERENCES utterances(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL                    -- float32, L2-normalised
);
CREATE INDEX IF NOT EXISTS idx_utt_vec_model ON utterance_vectors(model);

CREATE TABLE IF NOT EXISTS daily_summaries (
    day TEXT PRIMARY KEY,
    summary_md TEXT NOT NULL,
    summary_json TEXT,
    model TEXT,
    utterance_count INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    pushed_at TEXT
);

CREATE TABLE IF NOT EXISTS memory_notes (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,                  -- person | topic | todo | decision | fact
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    day_from TEXT,
    day_to TEXT,
    source_utterance_ids TEXT,           -- JSON list
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_notes_kind ON memory_notes(kind);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- "this is NOT that person": voices the user rejected for a speaker (negative samples)
CREATE TABLE IF NOT EXISTS speaker_negatives (
    id INTEGER PRIMARY KEY,
    speaker_id INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL,             -- float32, L2-normalised
    dim INTEGER NOT NULL,
    duration_s REAL,
    source_path TEXT,                    -- "<segment file>#a-b" or a speech-only copy
    utterance_id INTEGER,                -- the sentence it came from (no FK: the sentence may be split / deleted later)
    segment_id INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_spk_neg_speaker ON speaker_negatives(speaker_id);

-- per-sentence veto: automatic identification never assigns this speaker to this sentence again
CREATE TABLE IF NOT EXISTS utterance_rejections (
    utterance_id INTEGER NOT NULL REFERENCES utterances(id) ON DELETE CASCADE,
    speaker_id INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    PRIMARY KEY (utterance_id, speaker_id)
);

CREATE TABLE IF NOT EXISTS cluster_embeddings (
    segment_id INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    cluster_label INTEGER NOT NULL,
    embedding BLOB NOT NULL,             -- float32, L2-normalised
    dim INTEGER NOT NULL,
    speech_s REAL NOT NULL,              -- total duration of this cluster's turns in the segment
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    PRIMARY KEY (segment_id, cluster_label)
);

-- ===== 1.0 (schema version 2) ======================================================================

-- lexicon: canonical spellings (terms / names / projects ...) and their aliases (misheard forms, nicknames)
CREATE TABLE IF NOT EXISTS lexicon (
    id INTEGER PRIMARY KEY,
    term TEXT NOT NULL UNIQUE,            -- canonical spelling
    kind TEXT NOT NULL DEFAULT 'term',    -- person | term | project | org | place | product | other
    note TEXT,                            -- one-line explanation (for the model and for people)
    pinyin TEXT,                          -- toneless pinyin key, syllables separated by spaces; latin words lower-cased
    speaker_id INTEGER REFERENCES speakers(id) ON DELETE SET NULL,
    source TEXT NOT NULL DEFAULT 'user',  -- user | answer | edit | glossary | speaker | auto
    status TEXT NOT NULL DEFAULT 'active',-- active | candidate (found automatically, unconfirmed) | disabled
    hotword INTEGER NOT NULL DEFAULT 1,   -- goes into the ASR hot-word prompt
    hits INTEGER NOT NULL DEFAULT 0,
    last_hit_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE TABLE IF NOT EXISTS lexicon_aliases (
    id INTEGER PRIMARY KEY,
    lexicon_id INTEGER NOT NULL REFERENCES lexicon(id) ON DELETE CASCADE,
    alias TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'mishear', -- mishear (rewritten to the term) | aka (kept; retrieval / entity merging only)
    scope TEXT NOT NULL DEFAULT 'auto',   -- auto (rewrite on sight) | hint (only shown to the model)
    pinyin TEXT,
    source TEXT NOT NULL DEFAULT 'user',  -- user | answer | edit | llm
    hits INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    UNIQUE(alias, lexicon_id)
);
CREATE INDEX IF NOT EXISTS idx_lex_alias ON lexicon_aliases(alias);

-- text revisions: every change of utterances.text
CREATE TABLE IF NOT EXISTS utterance_revisions (
    id INTEGER PRIMARY KEY,
    utterance_id INTEGER NOT NULL REFERENCES utterances(id) ON DELETE CASCADE,
    old_text TEXT NOT NULL,
    new_text TEXT NOT NULL,
    source TEXT NOT NULL,                 -- lexicon | llm | refine | answer | user | guard | undo
    detail TEXT,                          -- JSON: {"edits":[{"from","to","reason","alias_id"?,"lexicon_id"?}],"inbox_id"?,"model"?}
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    undone_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_rev_utt ON utterance_revisions(utterance_id);

-- inbox: things the system wants to ask the user
CREATE TABLE IF NOT EXISTS inbox_items (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,                   -- term | voice | hint | conflict | rule
    status TEXT NOT NULL DEFAULT 'open',  -- open | answered | dismissed | snoozed | expired
    dedupe_key TEXT,                      -- unique per kind among open items
    priority REAL NOT NULL DEFAULT 0,
    title TEXT NOT NULL,                  -- the question in one sentence
    payload TEXT NOT NULL DEFAULT '{}',   -- JSON
    utterance_id INTEGER,                 -- main evidence (no FK: the sentence may be split / deleted)
    session_id INTEGER,
    speaker_id INTEGER,
    lexicon_id INTEGER,
    occurrences INTEGER NOT NULL DEFAULT 1,
    answer TEXT,                          -- JSON
    snooze_until TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    answered_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_inbox_open_key ON inbox_items(kind, dedupe_key)
    WHERE status='open' AND dedupe_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_inbox_status ON inbox_items(status, priority);

-- one voice embedding per utterance: computed once, every later re-scoring uses it
CREATE TABLE IF NOT EXISTS utterance_voice (
    utterance_id INTEGER PRIMARY KEY REFERENCES utterances(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    embedding BLOB NOT NULL,              -- float32, L2-normalised
    speech_s REAL
);

-- adaptive centroid of a speaker per recording condition
CREATE TABLE IF NOT EXISTS speaker_centroids (
    id INTEGER PRIMARY KEY,
    speaker_id INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    condition TEXT NOT NULL,              -- near | far | system | all
    embedding BLOB NOT NULL,
    dim INTEGER NOT NULL,
    model TEXT,
    n INTEGER NOT NULL DEFAULT 0,
    speech_s REAL NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    UNIQUE(speaker_id, condition)
);

-- persistent anonymous voices (the same person / programme across segments); can be named or ignored
CREATE TABLE IF NOT EXISTS voices (
    id INTEGER PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'unknown', -- unknown | named | ignored
    speaker_id INTEGER REFERENCES speakers(id) ON DELETE SET NULL,
    label TEXT,                             -- user remark ("TV", "downstairs")
    centroid BLOB,
    dim INTEGER,
    model TEXT,
    n_clusters INTEGER NOT NULL DEFAULT 0,
    speech_s REAL NOT NULL DEFAULT 0,
    first_seen TEXT,
    last_seen TEXT,
    days INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);

-- sessions (episodes): a stretch of continuous conversation cut by time gaps
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    day TEXT NOT NULL,                    -- the day it started
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    utterance_count INTEGER NOT NULL DEFAULT 0,
    speech_s REAL NOT NULL DEFAULT 0,
    chars INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'open',  -- open | closed
    kind TEXT,                            -- conversation | meeting | monologue | media | noise
    title TEXT,
    summary TEXT,
    topics TEXT,                          -- JSON list
    participants TEXT,                    -- JSON list
    raw_hash TEXT,                        -- fingerprint of utterance ids + text_raw (re-refine only when the composition changed)
    refined_hash TEXT,
    refined_at TEXT,
    text_hash TEXT,                       -- fingerprint of utterance ids + text (re-digest only when the text changed)
    digested_hash TEXT,
    digested_at TEXT,
    meta TEXT,                            -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_sessions_day ON sessions(day);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);

-- memory items: provenance, status, editable (replaces memory_notes)
CREATE TABLE IF NOT EXISTS memory_items (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,                   -- fact | todo | decision | person | preference | project | event | topic | note
    title TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    subject TEXT,                         -- who / what it is about (a name or a lexicon spelling)
    lexicon_id INTEGER REFERENCES lexicon(id) ON DELETE SET NULL,
    speaker_id INTEGER REFERENCES speakers(id) ON DELETE SET NULL,
    status TEXT NOT NULL DEFAULT 'active',-- active | done | superseded | archived | rejected (tombstone)
    source TEXT NOT NULL DEFAULT 'auto',  -- auto | digest | user | chat | voice | legacy
    confidence REAL,
    importance REAL NOT NULL DEFAULT 0.5,
    pinned INTEGER NOT NULL DEFAULT 0,
    day TEXT,
    valid_from TEXT,
    valid_to TEXT,
    due TEXT,                             -- todo deadline YYYY-MM-DD
    owner TEXT,                           -- who the todo belongs to
    session_id INTEGER,
    evidence TEXT,                        -- JSON list of utterance ids
    superseded_by INTEGER,
    fingerprint TEXT,                     -- memory_fingerprint(kind, title): de-duplication / tombstones
    hits INTEGER NOT NULL DEFAULT 0,
    last_used_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_items_kind ON memory_items(kind, status);
CREATE INDEX IF NOT EXISTS idx_items_subject ON memory_items(subject);
CREATE INDEX IF NOT EXISTS idx_items_fp ON memory_items(fingerprint);
CREATE TABLE IF NOT EXISTS memory_item_tokens (
    item_id INTEGER PRIMARY KEY REFERENCES memory_items(id) ON DELETE CASCADE,
    text_seg TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS memory_items_fts USING fts5(
    text_seg, content='memory_item_tokens', content_rowid='item_id', tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS item_tok_ai AFTER INSERT ON memory_item_tokens BEGIN
  INSERT INTO memory_items_fts(rowid, text_seg) VALUES (new.item_id, new.text_seg);
END;
CREATE TRIGGER IF NOT EXISTS item_tok_ad AFTER DELETE ON memory_item_tokens BEGIN
  INSERT INTO memory_items_fts(memory_items_fts, rowid, text_seg) VALUES ('delete', old.item_id, old.text_seg);
END;
CREATE TRIGGER IF NOT EXISTS item_tok_au AFTER UPDATE ON memory_item_tokens BEGIN
  INSERT INTO memory_items_fts(memory_items_fts, rowid, text_seg) VALUES ('delete', old.item_id, old.text_seg);
  INSERT INTO memory_items_fts(rowid, text_seg) VALUES (new.item_id, new.text_seg);
END;

-- generic vectors: memory items, session summaries, extra views of an utterance
CREATE TABLE IF NOT EXISTS memory_vectors (
    kind TEXT NOT NULL,                   -- item | session | utt
    ref_id INTEGER NOT NULL,
    view TEXT NOT NULL DEFAULT '',        -- view of an utterance: speaker / window; '' otherwise
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL,                    -- float32, L2-normalised
    PRIMARY KEY (kind, ref_id, view)
);

-- stored chats
CREATE TABLE IF NOT EXISTS chat_conversations (
    id INTEGER PRIMARY KEY,
    title TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES chat_conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,                   -- user | assistant
    content TEXT NOT NULL,
    meta TEXT,                            -- JSON: citations / stats / tools
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);

-- usage accounting (aggregated per day: tiny tables)
CREATE TABLE IF NOT EXISTS usage_local (
    day TEXT NOT NULL,
    stage TEXT NOT NULL,                  -- vad | gate | diarize | embed | identify | rescore | text_embed | ...
    n INTEGER NOT NULL DEFAULT 0,
    audio_s REAL NOT NULL DEFAULT 0,
    wall_s REAL NOT NULL DEFAULT 0,
    cpu_s REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (day, stage)
);
CREATE TABLE IF NOT EXISTS usage_llm (
    day TEXT NOT NULL,
    purpose TEXT NOT NULL,                -- asr | listen | correct | refine | episode | consolidate | daily | chat | relations | ...
    model TEXT NOT NULL,
    calls INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    audio_s REAL NOT NULL DEFAULT 0,
    wall_s REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (day, purpose, model)
);
CREATE TABLE IF NOT EXISTS job_runs (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'ui',    -- ui | scheduler | cli | chain
    status TEXT NOT NULL,                 -- running | done | failed
    started_at TEXT NOT NULL,
    finished_at TEXT,
    result TEXT,
    error TEXT
);
"""

#: columns added after the first release: (table, column, declaration). Applied by DB._migrate with
#: ALTER TABLE ... ADD COLUMN when PRAGMA table_info does not list them (old databases and new ones alike).
MIGRATION_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("speakers", "is_self", "INTEGER NOT NULL DEFAULT 0"),          # at most one row is 1
    ("speakers", "relation", "TEXT"),                                # 朋友 / 同事 / 领导 ... (free text allowed)
    ("speakers", "relation_source", "TEXT"),                         # 'user' | 'auto' | NULL
    ("speakers", "relation_note", "TEXT"),                           # inference reason / user remark
    ("speakers", "relation_updated_at", "TEXT"),
    ("utterances", "speaker_source", "TEXT"),                        # NULL = automatic; 'user' = hand-labelled (relabel keeps it)
    ("speaker_embeddings", "origin", "TEXT"),                        # 'mic' | 'file' | 'label' | 'voice' | 'cluster' | NULL (old rows)
    ("segments", "faint_s", "REAL"),                                 # seconds of turns dropped as too quiet (audio.min_speech_dbfs)
    ("utterances", "loudness_db", "REAL"),                           # speech level of the utterance's audio in dBFS (NULL = not computed)
    ("segments", "channel_rms", "BLOB"),                             # float16 pairs (mic_rms, system_rms) per capture block; NULL = mic only / unknown
    ("segments", "channel_block_s", "REAL"),                         # seconds per channel_rms pair
    ("utterances", "system_share", "REAL"),                          # share of the sentence's energy that came from system audio (0..1; NULL = unknown)
    # --- 1.0 (SPEC_v1 section 3.3) ---
    ("utterances", "session_id", "INTEGER"),                         # the session (sessions.id) the sentence belongs to
    ("utterances", "text_source", "TEXT"),                           # who decided the current text: NULL = ASR as is | lexicon | llm | refine | answer | user | guard
    ("utterances", "edited_at", "TEXT"),                             # time of the last revision
    ("utterances", "flags", "TEXT"),                                 # JSON: {"guard": "loop"}, {"media": true} ...
    ("utterances", "speaker_guess_id", "INTEGER"),                   # "probably X": not enough evidence to assign, worth asking
    ("utterances", "speaker_guess_score", "REAL"),                   # (normalised) score of the guess
    ("utterances", "speaker_meta", "TEXT"),                          # JSON: how the speaker was decided (method, raw / normalised score, condition)
    ("speaker_embeddings", "model", "TEXT"),                         # voice model id; NULL = the default CAM++ zh-cn
    ("speaker_negatives", "model", "TEXT"),
    ("cluster_embeddings", "model", "TEXT"),
    ("cluster_embeddings", "voice_id", "INTEGER"),                   # the persistent anonymous voice (voices.id) it belongs to
    ("cluster_embeddings", "loudness_db", "REAL"),                   # recording condition of the cluster (adaptive centroids split by it)
    ("cluster_embeddings", "system_share", "REAL"),
    ("segments", "realtime", "INTEGER"),                             # 1 = processed by the "as soon as it is recorded" path (statistics)
)

#: speaker_embeddings.origin values whose voice cannot contain anybody else (the user recorded / uploaded it)
TRUSTED_SAMPLE_ORIGINS = frozenset({"mic", "file"})


#: kv key prefix holding the utterance count at the last relation inference of a speaker (analysis.relations)
RELATION_KV_PREFIX = "relation_inferred_n:"


def relation_cache_key(sid: int) -> str:
    return f"{RELATION_KV_PREFIX}{int(sid)}"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --- versioned migrations (PRAGMA user_version) ------------------------------------------------------------

class SchemaTooNew(RuntimeError):
    """The database file was written by a newer audio-note: this build refuses to open (and change) it."""


class MigrationError(RuntimeError):
    """The database could not be upgraded (the backup failed, a data step failed). Nothing was changed."""


#: sub-directory of the data directory that receives the pre-upgrade copies
BACKUP_DIR_NAME = "backups"
#: how many pre-upgrade copies are kept (oldest removed first)
BACKUP_KEEP = 3
#: segments.error of a segment whose transcription was interrupted by a service stop (speech pipeline:
#: RuntimeError('stopped')); such rows go back to 'new' (DB.requeue_stopped), they are not failures
STOPPED_ERROR = "RuntimeError: stopped"


def memory_fingerprint(kind: str, title: str) -> str:
    """Identity of a memory item for de-duplication and tombstones: sha1(kind + normalised title)[:16].
    The title is NFKC-folded and lower-cased; whitespace, control characters and punctuation are dropped,
    so "买 牛奶！" and "买牛奶" are the same item. (memory.items.fingerprint must stay identical to this.)"""
    norm = unicodedata.normalize("NFKC", str(title or "")).lower()
    norm = "".join(ch for ch in norm if not ch.isspace() and unicodedata.category(ch)[0] not in "PZC")
    return hashlib.sha1((str(kind or "").strip().lower() + norm).encode("utf-8")).hexdigest()[:16]


def _evidence_json(raw: Any) -> str | None:
    """memory_notes.source_utterance_ids (a JSON list, or junk) -> a clean JSON list of ints, or None."""
    if raw is None or raw == "":
        return None
    try:
        val = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except (TypeError, ValueError):
        return None
    if not isinstance(val, list):
        return None
    ids: list[int] = []
    for v in val:
        try:
            ids.append(int(v))
        except (TypeError, ValueError):
            continue
    return json.dumps(ids) if ids else None


def content_hash(pairs: Iterable[tuple[int, str | None]]) -> str:
    """Fingerprint of an ordered list of (utterance id, text): what sessions.raw_hash / text_hash hold.
    One definition for everyone who compares or refreshes those columns (sessions, refine, revisions)."""
    h = hashlib.sha1()
    for uid, text in pairs:
        h.update(f"{int(uid)}{text or ''}".encode("utf-8"))
    return h.hexdigest()[:16]


def copy_notes_to_items(c: sqlite3.Connection) -> int:
    """memory_notes -> memory_items: kind / title / content as they are, source='legacy', evidence = the note's
    source_utterance_ids, fingerprint = memory_fingerprint(kind, title), day = the last day the note was seen
    (day_to, else day_from), valid_from = the first one (day_from). The old table is left alone.
    While memory_items is still empty (the upgrade itself) every note is copied and keeps its id, so ids the old
    notes API handed out keep pointing at the same entry. Run again later it is idempotent and respects what
    happened since: a note whose fingerprint already exists in memory_items - in any state, tombstones
    (status='rejected') included - is skipped.
    The caller provides the transaction. Returns the number of items inserted."""
    first = c.execute("SELECT 1 FROM memory_items LIMIT 1").fetchone() is None
    known = set() if first else {r[0] for r in c.execute("SELECT fingerprint FROM memory_items").fetchall()}
    seen: set[tuple[str, str]] = set()
    notes = c.execute("SELECT id, kind, title, content, day_from, day_to, source_utterance_ids, created_at, updated_at"
                      " FROM memory_notes ORDER BY id").fetchall()
    n = 0
    for r in notes:
        nid, kind, title, content, day_from, day_to, src, created, updated = (r[i] for i in range(9))
        fp = memory_fingerprint(kind, title)
        if (kind, title) in seen or fp in known:
            continue
        seen.add((kind, title))
        c.execute(
            "INSERT INTO memory_items(id, kind, title, content, source, status, day, valid_from, evidence, fingerprint,"
            " created_at, updated_at) VALUES (?,?,?,?, 'legacy', 'active', ?,?,?,?,?,?)",
            (int(nid) if first else None, kind, title, content or "", day_to or day_from, day_from, _evidence_json(src),
             fp, created or now_iso(), updated or created or now_iso()))
        if not first:
            known.add(fp)
        n += 1
    return n


def _migration_v2(c: sqlite3.Connection) -> None:
    """1.0: one timestamp format for segments.updated_at, memory_notes copied into memory_items, indexes on the
    columns MIGRATION_COLUMNS added. Runs inside one BEGIN IMMEDIATE transaction; every statement is idempotent."""
    # 1. SQLite's column default writes 'YYYY-MM-DD HH:MM:SS', set_segment_status writes now_iso() ('T'): one format
    c.execute("UPDATE segments SET updated_at = replace(updated_at, ' ', 'T') WHERE updated_at LIKE '____-__-__ __:__:__%'")
    # 2. memory_notes -> memory_items (the old table stays; nothing should write it any more)
    copy_notes_to_items(c)
    # 3. (the lexicon import happens at run time: lexicon.sync_sources needs the config)
    # 4. indexes on migrated columns (SCHEMA cannot hold them: it runs before the columns exist on an old file)
    c.execute("CREATE INDEX IF NOT EXISTS idx_utt_session ON utterances(session_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_cluster_voice ON cluster_embeddings(voice_id)")


#: the schema version this build reads and writes (PRAGMA user_version). 0 = the 0.3.x layout (never stamped).
SCHEMA_VERSION = 2
#: (version, description, data step). A step runs once, inside its own BEGIN IMMEDIATE transaction, when the file's
#: user_version is lower; steps must be idempotent and may only use the connection they are given.
MIGRATIONS: list[tuple[int, str, Callable[[sqlite3.Connection], None]]] = [
    (2, "1.0: segments.updated_at in one format, memory_notes -> memory_items, indexes on migrated columns", _migration_v2),
]


def _user_version(c: sqlite3.Connection) -> int:
    return int(c.execute("PRAGMA user_version").fetchone()[0])


class _ConnHolder:
    """Owns one thread's connection. It lives in the DB's threading.local, so it is released (and the
    connection closed) when its thread exits or the DB object itself is garbage collected; the DB
    keeps only weak references to holders, which is what lets that happen."""

    __slots__ = ("conn", "closed", "__weakref__")

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.closed = False

    def close(self) -> bool:
        if self.closed:
            return False
        self.closed = True
        try:
            self.conn.close()
        except sqlite3.Error:
            pass
        return True

    def __del__(self) -> None:   # pragma: no cover - timing depends on the GC
        self.close()


class DB:
    """Thin wrapper; one connection per thread, autocommit unless inside transaction().
    Connections are closed when their thread ends, when the DB object is collected, or by close_all()."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._holders: list[weakref.ref[_ConnHolder]] = []    # weak: a dead thread's holder may be collected
        self._conns_lock = threading.Lock()
        #: pre-upgrade copy written by this open (None: nothing to back up / already current)
        self.last_backup: Path | None = None
        #: versions whose data step ran during this open
        self.migrated: tuple[int, ...] = ()
        try:
            c = self.conn()
            found = _user_version(c)
            if found > SCHEMA_VERSION:
                # checked before anything is written: a newer build's file is left exactly as it is
                raise SchemaTooNew(f"数据库来自更新的版本（schema v{found}，本程序只支持到 v{SCHEMA_VERSION}）："
                                   f"请升级 audio-note 后再打开 {self.path}")
            self._ensure_wal(c)
            c.execute("PRAGMA synchronous=NORMAL")
            if found < SCHEMA_VERSION:
                self.last_backup = self._backup_before_upgrade(c, found)
            c.executescript(SCHEMA)
            self._migrate(c)
            self.migrated = self._upgrade(c)
        except BaseException:
            self.close_all()     # never keep a handle on a file we refused / failed to open (Windows locks it)
            raise

    def schema_version(self) -> int:
        """PRAGMA user_version of the file (== SCHEMA_VERSION once it is open)."""
        return _user_version(self.conn())

    @staticmethod
    def _ensure_wal(c: sqlite3.Connection) -> None:
        """PRAGMA journal_mode=WAL. On a file that is already in WAL mode this is free. The one-time switch of
        a file that is not needs the database to itself and does not wait for the busy timeout, so several
        connections opening such a file at the same moment see 'database is locked': retried for a few seconds."""
        for attempt in range(60):
            try:
                c.execute("PRAGMA journal_mode=WAL").fetchone()
                return
            except sqlite3.OperationalError as e:
                if "locked" not in str(e).lower() or attempt == 59:
                    raise
                time.sleep(min(0.25, 0.02 * (attempt + 1)))

    def _backup_before_upgrade(self, c: sqlite3.Connection, old_version: int) -> Path | None:
        """Copy the database to <data_dir>/backups/<stem>-v<old>-<YYYYmmdd-HHMMSS>.sqlite3 before an upgrade
        touches it. Only a file that already holds recordings (segments not empty) is copied; the newest
        BACKUP_KEEP copies are kept. Raises MigrationError when the copy cannot be made (the caller then
        does not upgrade)."""
        try:
            if c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='segments'").fetchone() is None:
                return None
            n_src = int(c.execute("SELECT COUNT(*) FROM segments").fetchone()[0])
        except sqlite3.Error as e:
            raise MigrationError(f"无法读取数据库 {self.path}，没有升级：{e}") from e
        if n_src == 0:
            return None
        bdir = self.path.parent / BACKUP_DIR_NAME
        prefix = f"{self.path.stem}-v{int(old_version)}-"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = bdir / f"{prefix}{stamp}.sqlite3"
        tmp = bdir / f"{prefix}{stamp}.{os.getpid()}-{threading.get_ident()}.partial"
        try:
            bdir.mkdir(parents=True, exist_ok=True)
            dst = sqlite3.connect(tmp)
            try:
                c.backup(dst)
                dst.execute("PRAGMA journal_mode=DELETE")      # a single self-contained file (no -wal / -shm next to it)
                n_dst = int(dst.execute("SELECT COUNT(*) FROM segments").fetchone()[0])
            finally:
                dst.close()
            if n_dst < n_src:
                raise MigrationError(f"备份不完整（{n_dst} / {n_src} 个片段）")
            # never write over an existing copy: two processes (or threads) upgrading in the same second each
            # keep their own file, the later one under a numbered name
            for k in range(1, 1000):
                if k > 1:
                    target = bdir / f"{prefix}{stamp}-{k}.sqlite3"
                if target.exists():
                    continue
                try:
                    os.rename(tmp, target)
                    break
                except FileExistsError:
                    continue
                except PermissionError:
                    if not target.exists():
                        raise
            else:
                raise MigrationError("备份文件名用尽")
        except Exception as e:  # noqa: BLE001 - whatever went wrong, the upgrade must not start
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise MigrationError(f"升级数据库前的备份失败，没有做任何改动（{type(e).__name__}: {e}）。"
                                 f"请检查 {bdir} 是否可写、磁盘是否已满，然后重试") from e
        log.info("database backed up before the upgrade from schema v%d: %s", old_version, target)
        try:
            found: list[tuple[float, str, Path]] = []
            for f in bdir.glob(f"{self.path.stem}-v*-*.sqlite3"):
                try:
                    found.append((f.stat().st_mtime, f.name, f))
                except OSError:
                    continue                       # another opener is pruning the same directory right now
            for _mtime, _name, f in sorted(found, reverse=True)[BACKUP_KEEP:]:
                if f != target:
                    f.unlink(missing_ok=True)
        except OSError as e:
            log.warning("could not prune old database backups in %s: %s", bdir, e)
        return target

    @staticmethod
    def _upgrade(c: sqlite3.Connection) -> tuple[int, ...]:
        """Run the data steps of MIGRATIONS whose version is above the file's user_version, each inside its own
        BEGIN IMMEDIATE transaction (the version is read again under the write lock, so two processes upgrading
        at once run every step exactly once), then stamp SCHEMA_VERSION. Returns the versions applied here."""
        if _user_version(c) >= SCHEMA_VERSION:
            return ()
        applied: list[int] = []
        for ver, desc, step in sorted(MIGRATIONS, key=lambda m: m[0]):
            if _user_version(c) >= ver:
                continue
            c.execute("BEGIN IMMEDIATE")
            try:
                if _user_version(c) >= ver:      # another process got here first
                    c.execute("ROLLBACK")
                    continue
                step(c)
                c.execute(f"PRAGMA user_version={int(ver)}")
                c.execute("COMMIT")
            except BaseException as e:
                try:
                    c.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                if isinstance(e, Exception):
                    raise MigrationError(f"数据库升级到 v{ver} 失败，已回滚（{type(e).__name__}: {e}）") from e
                raise
            applied.append(int(ver))
            log.info("database upgraded to schema v%d: %s", ver, desc)
        if _user_version(c) < SCHEMA_VERSION:      # SCHEMA_VERSION bumped without a data step
            c.execute(f"PRAGMA user_version={int(SCHEMA_VERSION)}")
        return tuple(applied)

    @staticmethod
    def _migrate(c: sqlite3.Connection) -> list[str]:
        """Add the columns of MIGRATION_COLUMNS that are missing (idempotent). Returns the ones added.
        Two processes upgrading the same old database at once both see a column missing; the loser's
        ALTER TABLE fails with 'duplicate column name', which is the wanted end state and is ignored."""
        added: list[str] = []
        for table, col, decl in MIGRATION_COLUMNS:
            have = {r[1] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}
            if col in have:
                continue
            try:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
                continue
            added.append(f"{table}.{col}")
        if "speaker_embeddings.origin" in added:
            # one-time guess for rows written before the column existed: a "<file>#a-b" source is a transcript
            # label, an "enroll_*" copy came from the page's microphone / file upload (trusted anchor);
            # everything else (CLI record, voice-group copies) stays NULL
            c.execute("UPDATE speaker_embeddings SET origin = CASE"
                      " WHEN source_path LIKE '%#%' THEN 'label'"
                      " WHEN source_path LIKE '%enroll\\_%' ESCAPE '\\' THEN 'file' END"
                      " WHERE origin IS NULL AND source_path IS NOT NULL")
        if added:
            import logging

            logging.getLogger(__name__).info("database upgraded: added %s", ", ".join(added))
        return added

    def conn(self) -> sqlite3.Connection:
        h: _ConnHolder | None = getattr(self._local, "holder", None)
        if h is None:
            c = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA foreign_keys=ON")
            self._load_vec(c)
            h = _ConnHolder(c)
            self._local.holder = h
            with self._conns_lock:
                self._holders = [r for r in self._holders if r() is not None]
                self._holders.append(weakref.ref(h))
        return h.conn

    def open_connections(self) -> int:
        """Connections currently open on any thread (for diagnostics / tests)."""
        with self._conns_lock:
            return sum(1 for r in self._holders if (h := r()) is not None and not h.closed)

    def close_all(self) -> int:
        """Close every connection this DB object opened, on any thread (sqlite allows it because the
        connections are created with check_same_thread=False). Used when a Runtime is discarded (profile
        switch): Windows refuses to delete or move a database whose file is still open somewhere.
        Returns the number of connections closed. Further use of this DB object raises ProgrammingError."""
        with self._conns_lock:
            holders, self._holders = [h for r in self._holders if (h := r()) is not None], []
        return sum(1 for h in holders if h.close())

    #: True once the sqlite-vec extension loaded on this process (vector search in SQL); None = untried
    vec_available: bool | None = None

    @classmethod
    def _load_vec(cls, c: sqlite3.Connection) -> None:
        """Load sqlite-vec (vec_distance_cosine etc.) into the connection. Optional: without it the
        vector store falls back to numpy, so failures only log once."""
        if cls.vec_available is False:
            return
        try:
            import sqlite_vec

            c.enable_load_extension(True)
            sqlite_vec.load(c)
            c.enable_load_extension(False)
            cls.vec_available = True
        except Exception as e:  # noqa: BLE001 - missing wheel / extension loading disabled
            if cls.vec_available is None:
                import logging

                logging.getLogger(__name__).info("sqlite-vec unavailable (%s); vector search uses numpy", e)
            cls.vec_available = False

    # --- generic helpers -------------------------------------------------
    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        return self.conn().execute(sql, tuple(params))

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return self.conn().execute(sql, tuple(params)).fetchall()

    def one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        return self.conn().execute(sql, tuple(params)).fetchone()

    def transaction(self) -> "_Tx":
        return _Tx(self.conn())

    # --- sessions ------------------------------------------------------------
    def session_hashes(self, session_id: int) -> tuple[str, str]:
        """(raw_hash, text_hash) of a session as its sentences are now: ids + text_raw (changes when sentences
        join or leave: refine again) and ids + text (changes with every edit: digest again)."""
        rows = self.query("SELECT id, text_raw, text FROM utterances WHERE session_id=? ORDER BY abs_start, id",
                          (int(session_id),))
        return (content_hash((r["id"], r["text_raw"]) for r in rows), content_hash((r["id"], r["text"]) for r in rows))

    def refresh_session_hashes(self, session_id: int | None) -> tuple[str, str] | None:
        """Store session_hashes() on the row (call after sentences were added, removed or edited). Returns
        the pair, or None when there is no such session. Never touches refined_hash / digested_hash."""
        if session_id is None:
            return None
        raw, text = self.session_hashes(int(session_id))
        cur = self.execute("UPDATE sessions SET raw_hash=?, text_hash=?, updated_at=? WHERE id=?",
                           (raw, text, now_iso(), int(session_id)))
        return (raw, text) if cur.rowcount else None

    # --- segments ----------------------------------------------------------
    def add_segment(self, path: Path | str, started_at: datetime, duration_s: float, sample_rate: int = 16000,
                    source: str = "mic", origin: str | None = None, speech_s: float | None = None) -> int:
        cur = self.execute(
            "INSERT INTO segments(path, started_at, duration_s, sample_rate, source, origin, speech_s, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (str(path), started_at.isoformat(timespec="milliseconds"), duration_s, sample_rate, source, origin, speech_s, now_iso()),
        )
        return int(cur.lastrowid)

    def get_segment(self, seg_id: int) -> sqlite3.Row | None:
        return self.one("SELECT * FROM segments WHERE id=?", (seg_id,))

    def set_segment_status(self, seg_id: int, status: str, error: str | None = None) -> None:
        self.execute("UPDATE segments SET status=?, error=?, updated_at=? WHERE id=?", (status, error, now_iso(), seg_id))

    def pending_segments(self, min_age_s: int = 0, limit: int = 50) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM segments WHERE status='new' AND created_at <= datetime('now','localtime', ?) ORDER BY started_at LIMIT ?",
            (f"-{int(min_age_s)} seconds", limit),
        )

    def reset_stale_processing(self, max_age_s: int = 600) -> int:
        """Put segments stuck in 'processing' (process killed mid-segment) back to 'new'.
        Only rows whose updated_at is older than max_age_s are touched, so a processor that is
        genuinely working in another thread/process is left alone. Returns the number reset."""
        # updated_at is written by now_iso() ('T' separator); rows from before schema v2 and SQLite's column
        # default use a space. Comparing those two as strings never matched on the same day ('T' > ' '), so
        # the threshold is computed here in the 'T' format and the column is normalised inside the comparison.
        cutoff = (datetime.now() - timedelta(seconds=int(max_age_s))).isoformat(timespec="seconds")
        cur = self.execute(
            "UPDATE segments SET status='new', error=NULL, updated_at=? WHERE status='processing'"
            " AND replace(updated_at, ' ', 'T') <= ?",
            (now_iso(), cutoff),
        )
        return int(cur.rowcount or 0)

    def requeue_stopped(self) -> int:
        """Segments whose transcription was interrupted by a service stop (stored as failed with
        STOPPED_ERROR by the speech pipeline) go back to 'new': the audio is fine, the next run retries
        them. Returns the number re-queued."""
        cur = self.execute("UPDATE segments SET status='new', error=NULL, updated_at=? WHERE status='failed' AND error=?",
                           (now_iso(), STOPPED_ERROR))
        return int(cur.rowcount or 0)

    def unfinished_segments_for_day(self, day: str, failed_within_s: int = 86400) -> int:
        """Segments of `day` (by started_at) that have not produced utterances yet: new / processing,
        plus failed ones that failed recently (older failures need a manual --retry-failed)."""
        cutoff = (datetime.now() - timedelta(seconds=int(failed_within_s))).isoformat(timespec="seconds")
        row = self.one(
            "SELECT COUNT(*) AS n FROM segments WHERE started_at >= ? AND started_at < ? AND ("
            " status IN ('new','processing') OR (status='failed' AND replace(updated_at, ' ', 'T') >= ?))",
            (f"{day}T00:00:00", f"{day}T23:59:59.999", cutoff),
        )
        return int(row["n"]) if row else 0

    # --- speakers ----------------------------------------------------------
    def upsert_speaker(self, name: str, notes: str | None = None) -> int:
        row = self.one("SELECT id FROM speakers WHERE name=?", (name,))
        if row:
            if notes is not None:
                self.execute("UPDATE speakers SET notes=? WHERE id=?", (notes, row["id"]))
            return int(row["id"])
        return int(self.execute("INSERT INTO speakers(name, notes) VALUES (?,?)", (name, notes)).lastrowid)

    def add_speaker_embedding(self, speaker_id: int, emb: np.ndarray, source_path: str | None, duration_s: float | None,
                              origin: str | None = None) -> int:
        """Store one enrolment embedding. `origin` says where the sample came from ('mic' = speakers record,
        'file' = enroll upload / file, 'label' = transcript label, 'voice' = naming a voice group, 'cluster' =
        stored cluster embedding of an expired recording); the consistency check anchors on 'mic' / 'file'."""
        emb = np.asarray(emb, dtype=np.float32)
        return int(self.execute(
            "INSERT INTO speaker_embeddings(speaker_id, embedding, dim, source_path, duration_s, origin) VALUES (?,?,?,?,?,?)",
            (speaker_id, emb.tobytes(), int(emb.shape[0]), source_path, duration_s, origin),
        ).lastrowid)

    def speaker_embeddings(self) -> list[tuple[int, str, np.ndarray]]:
        rows = self.query(
            "SELECT e.speaker_id, s.name, e.embedding FROM speaker_embeddings e JOIN speakers s ON s.id=e.speaker_id"
        )
        return [(int(r["speaker_id"]), r["name"], np.frombuffer(r["embedding"], dtype=np.float32)) for r in rows]

    # --- negative samples / per-sentence vetoes (speakers v4) ----------------------
    def add_speaker_negative(self, speaker_id: int, emb: np.ndarray, source_path: str | None, duration_s: float | None,
                             utterance_id: int | None = None, segment_id: int | None = None) -> int:
        """Store one negative sample ("this voice is NOT speaker_id"); the embedding is stored L2-normalised."""
        emb = np.asarray(emb, dtype=np.float32).ravel()
        n = float(np.linalg.norm(emb))
        if n > 0:
            emb = emb / n
        return int(self.execute(
            "INSERT INTO speaker_negatives(speaker_id, embedding, dim, duration_s, source_path, utterance_id, segment_id)"
            " VALUES (?,?,?,?,?,?,?)",
            (int(speaker_id), emb.astype(np.float32).tobytes(), int(emb.shape[0]), duration_s, source_path,
             int(utterance_id) if utterance_id is not None else None, int(segment_id) if segment_id is not None else None),
        ).lastrowid)

    def speaker_negatives(self) -> list[tuple[int, np.ndarray]]:
        """[(speaker_id, L2-normalised embedding)] of every negative sample."""
        out: list[tuple[int, np.ndarray]] = []
        for r in self.query("SELECT speaker_id, embedding FROM speaker_negatives ORDER BY id"):
            e = np.frombuffer(r["embedding"], dtype=np.float32)
            n = float(np.linalg.norm(e))
            out.append((int(r["speaker_id"]), e / n if n > 0 else e))
        return out

    def delete_speaker_negative(self, neg_id: int) -> int:
        cur = self.execute("DELETE FROM speaker_negatives WHERE id=?", (int(neg_id),))
        return int(cur.rowcount or 0)

    def reject_utterances(self, speaker_id: int, utterance_ids: Iterable[int], unknown_name: str) -> int:
        """Veto `speaker_id` for these sentences (utterance_rejections, INSERT OR IGNORE) and release the ones currently
        attributed to that speaker (speaker_id NULL, name = unknown_name, no score, automatic again).
        Returns the number of sentences released."""
        sid = int(speaker_id)
        ids = sorted({int(i) for i in utterance_ids})
        if not ids:
            return 0
        released = 0
        c = self.conn()
        with _Tx(c):
            for k in range(0, len(ids), 500):
                chunk = ids[k:k + 500]
                marks = ",".join("?" * len(chunk))
                c.execute(f"INSERT OR IGNORE INTO utterance_rejections(utterance_id, speaker_id)"
                          f" SELECT id, ? FROM utterances WHERE id IN ({marks})", [sid, *chunk])
                cur = c.execute(f"UPDATE utterances SET speaker_id=NULL, speaker_name=?, speaker_score=NULL, speaker_source=NULL"
                                f" WHERE speaker_id=? AND id IN ({marks})", [unknown_name, sid, *chunk])
                # "not X" also answers "maybe X": the guess goes with the attribution
                c.execute(f"UPDATE utterances SET speaker_guess_id=NULL, speaker_guess_score=NULL"
                          f" WHERE speaker_guess_id=? AND id IN ({marks})", [sid, *chunk])
                released += int(cur.rowcount or 0)
        return released

    def rejected_speakers(self, utterance_ids: Iterable[int]) -> dict[int, set[int]]:
        """{utterance_id: {vetoed speaker ids}} for the given sentences (only the ones with a veto appear)."""
        ids = sorted({int(i) for i in utterance_ids})
        out: dict[int, set[int]] = {}
        for k in range(0, len(ids), 500):
            chunk = ids[k:k + 500]
            for r in self.query(f"SELECT utterance_id, speaker_id FROM utterance_rejections WHERE utterance_id IN ({','.join('?' * len(chunk))})",
                                chunk):
                out.setdefault(int(r["utterance_id"]), set()).add(int(r["speaker_id"]))
        return out

    # --- self / relations ----------------------------------------------------
    def self_speaker(self) -> sqlite3.Row | None:
        """The speaker marked as the user themselves (at most one), or None."""
        return self.one("SELECT * FROM speakers WHERE is_self=1 ORDER BY id LIMIT 1")

    def set_self_speaker(self, sid: int | None) -> None:
        """Make `sid` the one and only self speaker; None only clears the flag everywhere."""
        c = self.conn()
        with _Tx(c):
            c.execute("UPDATE speakers SET is_self=0 WHERE is_self<>0")
            if sid is not None:
                c.execute("UPDATE speakers SET is_self=1 WHERE id=?", (int(sid),))

    def set_relation(self, sid: int, relation: str | None, source: str | None, note: str | None = None) -> bool:
        """Store the speaker's relation to the user (source 'user' | 'auto'). An empty relation clears all three
        columns and forgets the inference cache (so the next relation run asks the model again).
        An 'auto' write never replaces a relation the user set (they may have labelled it while the LLM
        call was in flight). Returns True when the row was written."""
        relation = (relation or "").strip()
        sid = int(sid)
        if not relation:
            c = self.conn()
            with _Tx(c):
                cur = c.execute("UPDATE speakers SET relation=NULL, relation_source=NULL, relation_note=NULL, relation_updated_at=NULL"
                                " WHERE id=?", (sid,))
                c.execute("DELETE FROM kv WHERE key=?", (relation_cache_key(sid),))
            return int(cur.rowcount or 0) == 1
        guard = " AND (relation_source IS NULL OR relation_source<>'user')" if source == "auto" else ""
        cur = self.execute(f"UPDATE speakers SET relation=?, relation_source=?, relation_note=?, relation_updated_at=? WHERE id=?{guard}",
                           (relation, source, note, now_iso(), sid))
        return int(cur.rowcount or 0) == 1

    def delete_speaker(self, sid: int, unknown_name: str) -> int:
        """Remove a speaker: its utterances go back to unidentified (name = unknown_name, no score and no
        'user' mark, so relabel may re-identify them) and the relation inference cache is dropped.
        Returns the number of utterances released (0 when the speaker had none or did not exist)."""
        sid = int(sid)
        c = self.conn()
        with _Tx(c):
            cur = c.execute("UPDATE utterances SET speaker_name=?, speaker_score=NULL, speaker_source=NULL WHERE speaker_id=?",
                            (unknown_name, sid))
            c.execute("DELETE FROM kv WHERE key=?", (relation_cache_key(sid),))
            c.execute("DELETE FROM speakers WHERE id=?", (sid,))
        return int(cur.rowcount or 0)

    def people_table(self) -> list[dict[str, Any]]:
        """Every enrolled speaker with their relation and utterance count: self first, then by utterances desc."""
        rows = self.query(
            "SELECT s.id, s.name, s.is_self, s.relation, s.relation_source, s.relation_note,"
            " (SELECT COUNT(*) FROM utterances u WHERE u.speaker_id=s.id) AS n"
            " FROM speakers s ORDER BY s.is_self DESC, n DESC, s.name, s.id"
        )
        return [{"id": int(r["id"]), "name": r["name"], "is_self": bool(r["is_self"]), "relation": r["relation"],
                 "relation_source": r["relation_source"], "relation_note": r["relation_note"], "utterances": int(r["n"])}
                for r in rows]

    # --- cluster embeddings ------------------------------------------------------
    def upsert_cluster_embedding(self, segment_id: int, cluster_label: int, emb: np.ndarray, speech_s: float) -> None:
        emb = np.asarray(emb, dtype=np.float32).ravel()
        self.execute(
            "INSERT INTO cluster_embeddings(segment_id, cluster_label, embedding, dim, speech_s) VALUES (?,?,?,?,?)"
            " ON CONFLICT(segment_id, cluster_label) DO UPDATE SET embedding=excluded.embedding, dim=excluded.dim,"
            " speech_s=excluded.speech_s, created_at=datetime('now','localtime')",
            (int(segment_id), int(cluster_label), emb.tobytes(), int(emb.shape[0]), float(speech_s)),
        )

    def cluster_embeddings(self, segment_ids: list[int] | None = None) -> list[tuple[int, int, np.ndarray, float]]:
        """[(segment_id, cluster_label, embedding, speech_s)] for all segments or the given ones.
        Sentinel rows (cluster_label < 0: scanned, nothing to embed) are left out."""
        if segment_ids is None:
            rows = self.query("SELECT segment_id, cluster_label, embedding, speech_s FROM cluster_embeddings WHERE cluster_label >= 0"
                              " ORDER BY segment_id, cluster_label")
        else:
            ids = [int(i) for i in segment_ids]
            if not ids:
                return []
            rows = self.query(
                f"SELECT segment_id, cluster_label, embedding, speech_s FROM cluster_embeddings WHERE segment_id IN ({','.join('?' * len(ids))})"
                " AND cluster_label >= 0 ORDER BY segment_id, cluster_label", ids)
        return [(int(r["segment_id"]), int(r["cluster_label"]), np.frombuffer(r["embedding"], dtype=np.float32), float(r["speech_s"]))
                for r in rows]

    def delete_cluster_embeddings(self, segment_id: int) -> int:
        cur = self.execute("DELETE FROM cluster_embeddings WHERE segment_id=?", (int(segment_id),))
        return int(cur.rowcount or 0)

    def mark_cluster_embeddings_empty(self, segment_id: int) -> None:
        """Record that `segment_id` was scanned for cluster embeddings but none could be made (unreadable
        file, utterances outside the audio ...): a sentinel row with cluster_label=-1 and an empty blob, so
        the segment is not counted as missing and re-scanned every session. delete_cluster_embeddings
        (process_segment) removes it like any other row."""
        self.execute(
            "INSERT INTO cluster_embeddings(segment_id, cluster_label, embedding, dim, speech_s) VALUES (?,-1,X'',0,0)"
            " ON CONFLICT(segment_id, cluster_label) DO UPDATE SET created_at=datetime('now','localtime')",
            (int(segment_id),))

    # --- utterances --------------------------------------------------------
    def add_utterances(self, rows: list[dict[str, Any]]) -> list[int]:
        """rows: dicts with keys matching the utterances columns plus optional 'text_seg' / 'speaker_source' / 'loudness_db' /
        'system_share' / 'session_id' / 'text_source' / 'flags' / 'speaker_guess_id' / 'speaker_guess_score' / 'speaker_meta'
        ('flags' and 'speaker_meta' are JSON columns: a dict / list is serialised here, a string is stored as is)."""
        ids: list[int] = []
        c = self.conn()
        with _Tx(c):
            for r in rows:
                cur = c.execute(
                    "INSERT INTO utterances(segment_id,start_s,end_s,abs_start,day,cluster_label,speaker_id,speaker_name,speaker_score,"
                    "text_raw,text,speaker_source,loudness_db,system_share,session_id,text_source,flags,speaker_guess_id,"
                    "speaker_guess_score,speaker_meta) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (r["segment_id"], r["start_s"], r["end_s"], r["abs_start"], r["day"], r.get("cluster_label"),
                     r.get("speaker_id"), r.get("speaker_name"), r.get("speaker_score"), r["text_raw"], r["text"],
                     r.get("speaker_source"), r.get("loudness_db"), r.get("system_share"), r.get("session_id"),
                     r.get("text_source"), _json_col(r.get("flags")), r.get("speaker_guess_id"), r.get("speaker_guess_score"),
                     _json_col(r.get("speaker_meta"))),
                )
                uid = int(cur.lastrowid)
                ids.append(uid)
                if r.get("text_seg"):
                    c.execute("INSERT INTO utterance_tokens(utterance_id, text_seg) VALUES (?,?)", (uid, r["text_seg"]))
        return ids

    def utterances_for_day(self, day: str) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM utterances WHERE day=? ORDER BY abs_start, start_s", (day,))

    def delete_utterances_for_segment(self, seg_id: int) -> None:
        self.execute("DELETE FROM utterances WHERE segment_id=?", (seg_id,))

    # --- utterance vectors ---------------------------------------------------
    def add_vectors(self, model: str, pairs: Iterable[tuple[int, np.ndarray]]) -> int:
        """Upsert (utterance_id, vector) pairs for `model`. Vectors are stored L2-normalised float32."""
        n = 0
        c = self.conn()
        with _Tx(c):
            for uid, v in pairs:
                v = np.asarray(v, dtype=np.float32).ravel()
                norm = float(np.linalg.norm(v))
                if norm > 0:
                    v = v / norm
                c.execute(
                    "INSERT INTO utterance_vectors(utterance_id, model, dim, vec) VALUES (?,?,?,?)"
                    " ON CONFLICT(utterance_id) DO UPDATE SET model=excluded.model, dim=excluded.dim, vec=excluded.vec",
                    (int(uid), model, int(v.shape[0]), v.tobytes()),
                )
                n += 1
        return n

    def utterances_without_vectors(self, model: str, limit: int = 1000) -> list[sqlite3.Row]:
        """Utterances that have no vector for `model` yet (rows embedded by another model count as missing)."""
        return self.query(
            "SELECT u.id, u.text FROM utterances u LEFT JOIN utterance_vectors v ON v.utterance_id = u.id"
            " WHERE v.utterance_id IS NULL OR v.model != ? ORDER BY u.id LIMIT ?",
            (model, limit),
        )

    def vector_count(self, model: str | None = None) -> int:
        if model is None:
            return int(self.one("SELECT COUNT(*) AS n FROM utterance_vectors")["n"])
        return int(self.one("SELECT COUNT(*) AS n FROM utterance_vectors WHERE model=?", (model,))["n"])

    # --- kv ----------------------------------------------------------------
    def get_kv(self, key: str, default: Any = None) -> Any:
        row = self.one("SELECT value FROM kv WHERE key=?", (key,))
        return json.loads(row["value"]) if row else default

    def set_kv(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO kv(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, ensure_ascii=False)),
        )

    # --- usage accounting (metrics.py) ---------------------------------------------
    def add_usage_local(self, stage: str, *, n: int = 1, audio_s: float = 0.0, wall_s: float = 0.0, cpu_s: float = 0.0,
                        day: str | None = None) -> None:
        """Add one measurement of a local stage (diarize / embed / ...) to its per-day row."""
        self.execute(
            "INSERT INTO usage_local(day, stage, n, audio_s, wall_s, cpu_s) VALUES (?,?,?,?,?,?)"
            " ON CONFLICT(day, stage) DO UPDATE SET n=n+excluded.n, audio_s=audio_s+excluded.audio_s,"
            " wall_s=wall_s+excluded.wall_s, cpu_s=cpu_s+excluded.cpu_s",
            (day or datetime.now().strftime("%Y-%m-%d"), str(stage), int(n), float(audio_s), float(wall_s), float(cpu_s)))

    def add_usage_llm(self, purpose: str, model: str, *, calls: int = 1, errors: int = 0, prompt_tokens: int = 0,
                      completion_tokens: int = 0, audio_s: float = 0.0, wall_s: float = 0.0, day: str | None = None) -> None:
        """Add LLM / ASR calls to the per-day row of (purpose, model)."""
        self.execute(
            "INSERT INTO usage_llm(day, purpose, model, calls, errors, prompt_tokens, completion_tokens, audio_s, wall_s)"
            " VALUES (?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(day, purpose, model) DO UPDATE SET calls=calls+excluded.calls, errors=errors+excluded.errors,"
            " prompt_tokens=prompt_tokens+excluded.prompt_tokens, completion_tokens=completion_tokens+excluded.completion_tokens,"
            " audio_s=audio_s+excluded.audio_s, wall_s=wall_s+excluded.wall_s",
            (day or datetime.now().strftime("%Y-%m-%d"), str(purpose or "other"), str(model or "?"), int(calls), int(errors),
             int(prompt_tokens), int(completion_tokens), float(audio_s), float(wall_s)))

    # --- job history ------------------------------------------------------------------
    def start_job_run(self, name: str, source: str = "ui", started_at: str | None = None) -> int:
        """Insert a 'running' row into job_runs; returns its id (finish it with finish_job_run)."""
        return int(self.execute("INSERT INTO job_runs(name, source, status, started_at) VALUES (?,?, 'running', ?)",
                                (str(name), str(source), started_at or now_iso())).lastrowid)

    def finish_job_run(self, run_id: int, status: str, result: Any = None, error: str | None = None) -> None:
        self.execute("UPDATE job_runs SET status=?, finished_at=?, result=?, error=? WHERE id=?",
                     (str(status), now_iso(), _json_col(result), error, int(run_id)))

    def add_job_run(self, name: str, source: str, status: str, started_at: str, result: Any = None,
                    error: str | None = None) -> int:
        """Record a run that is already over in one statement (scheduler-driven jobs)."""
        return int(self.execute(
            "INSERT INTO job_runs(name, source, status, started_at, finished_at, result, error) VALUES (?,?,?,?,?,?,?)",
            (str(name), str(source), str(status), started_at, now_iso(), _json_col(result), error)).lastrowid)

    def close_orphan_job_runs(self) -> int:
        """Rows still 'running' when a service starts belong to a run that died with the previous process."""
        cur = self.execute("UPDATE job_runs SET status='failed', finished_at=?, error=COALESCE(error, 'interrupted')"
                           " WHERE status='running'", (now_iso(),))
        return int(cur.rowcount or 0)

    # --- maintenance ----------------------------------------------------------------------
    def maintenance(self, vacuum: bool = False, keep_job_runs_days: int = 30) -> dict[str, Any]:
        """Housekeeping: prune the job history, optimise the FTS indexes and the query planner statistics,
        fold the WAL back into the main file; VACUUM only on request (it rewrites the whole file)."""
        out: dict[str, Any] = {}
        cutoff = (datetime.now() - timedelta(days=max(1, int(keep_job_runs_days)))).isoformat(timespec="seconds")
        out["job_runs_pruned"] = int(self.execute("DELETE FROM job_runs WHERE status<>'running' AND started_at < ?",
                                                  (cutoff,)).rowcount or 0)
        for fts in ("utterances_fts", "memory_items_fts"):
            try:
                self.execute(f"INSERT INTO {fts}({fts}) VALUES ('optimize')")
            except sqlite3.Error as e:
                log.warning("maintenance: optimize %s failed: %s", fts, e)
        try:
            self.execute("PRAGMA optimize")
        except sqlite3.Error as e:
            log.warning("maintenance: PRAGMA optimize failed: %s", e)
        if vacuum:
            self.execute("VACUUM")
        try:
            row = self.one("PRAGMA wal_checkpoint(TRUNCATE)")
            out["wal_checkpoint"] = "busy" if row is not None and int(row[0]) else "ok"
        except sqlite3.Error as e:
            out["wal_checkpoint"] = f"failed: {e}"
        out["vacuum"] = bool(vacuum)
        try:
            out["db_bytes"] = int(self.path.stat().st_size)
        except OSError:
            out["db_bytes"] = None
        return out


def _json_col(value: Any) -> Any:
    """Value for a JSON TEXT column: None and strings pass through, anything else is serialised."""
    if value is None or isinstance(value, (str, bytes)):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


class _Tx:
    """`with db.transaction() as c:` = BEGIN IMMEDIATE ... COMMIT (ROLLBACK when the block raises).

    IMMEDIATE takes the write lock up front: a read-then-write block cannot fail half-way with SQLITE_BUSY
    because another connection wrote in between; it waits (busy timeout) at BEGIN instead.
    Re-entrant: when the connection is already inside a transaction (a helper that opens its own _Tx is
    called from inside `with db.transaction():`) a SAVEPOINT is used, so the inner block can still roll
    back on its own and the outer one decides about the whole."""

    _seq = itertools.count(1)

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._savepoint: str | None = None

    def __enter__(self) -> sqlite3.Connection:
        if getattr(self.conn, "in_transaction", False):
            self._savepoint = f"tx_{next(_Tx._seq)}"
            self.conn.execute(f"SAVEPOINT {self._savepoint}")
        else:
            self.conn.execute("BEGIN IMMEDIATE")
        return self.conn

    def __exit__(self, et, ev, tb) -> bool:
        sp = self._savepoint
        if sp is not None:
            self._savepoint = None
            if et is not None:
                self.conn.execute(f"ROLLBACK TO {sp}")
            self.conn.execute(f"RELEASE {sp}")
            return False
        if et is None:
            try:
                self.conn.execute("COMMIT")
            except BaseException:
                try:
                    self.conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
        else:
            try:
                self.conn.execute("ROLLBACK")
            except sqlite3.Error:      # the connection was closed under us (profile switch): keep the original error
                pass
        return False
