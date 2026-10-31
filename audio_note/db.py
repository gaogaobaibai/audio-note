"""SQLite data layer. One file: data/audio_note.sqlite3, WAL mode.

Tables
------
segments           one row per stored audio file (speech-gated recording or imported chunk)
transcripts        raw + corrected ASR output per segment
utterances         speaker-attributed sentences with absolute timestamps (the unit of memory)
utterance_tokens   jieba-segmented text, mirrored into the FTS5 index utterances_fts
speakers           enrolled people
speaker_embeddings enrollment embeddings (float32 blobs)
daily_summaries    per-day LLM digest
memory_notes       long-lived notes (entities, topics, todos) maintained by the daily analysis
kv                 misc state
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

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
"""


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class DB:
    """Thin wrapper; one connection per thread, autocommit unless inside transaction()."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        c = self.conn()
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.executescript(SCHEMA)

    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA foreign_keys=ON")
            self._local.conn = c
        return c

    # --- generic helpers -------------------------------------------------
    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        return self.conn().execute(sql, tuple(params))

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return self.conn().execute(sql, tuple(params)).fetchall()

    def one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        return self.conn().execute(sql, tuple(params)).fetchone()

    def transaction(self) -> "_Tx":
        return _Tx(self.conn())

    # --- segments ----------------------------------------------------------
    def add_segment(self, path: Path | str, started_at: datetime, duration_s: float, sample_rate: int = 16000,
                    source: str = "mic", origin: str | None = None, speech_s: float | None = None) -> int:
        cur = self.execute(
            "INSERT INTO segments(path, started_at, duration_s, sample_rate, source, origin, speech_s) VALUES (?,?,?,?,?,?,?)",
            (str(path), started_at.isoformat(timespec="milliseconds"), duration_s, sample_rate, source, origin, speech_s),
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
        cur = self.execute(
            "UPDATE segments SET status='new', error=NULL, updated_at=? WHERE status='processing'"
            " AND updated_at <= datetime('now','localtime', ?)",
            (now_iso(), f"-{int(max_age_s)} seconds"),
        )
        return int(cur.rowcount or 0)

    def unfinished_segments_for_day(self, day: str, failed_within_s: int = 86400) -> int:
        """Segments of `day` (by started_at) that have not produced utterances yet: new / processing,
        plus failed ones that failed recently (older failures need a manual --retry-failed)."""
        row = self.one(
            "SELECT COUNT(*) AS n FROM segments WHERE started_at >= ? AND started_at < ? AND ("
            " status IN ('new','processing') OR (status='failed' AND updated_at >= datetime('now','localtime', ?)))",
            (f"{day}T00:00:00", f"{day}T23:59:59.999", f"-{int(failed_within_s)} seconds"),
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

    def add_speaker_embedding(self, speaker_id: int, emb: np.ndarray, source_path: str | None, duration_s: float | None) -> int:
        emb = np.asarray(emb, dtype=np.float32)
        return int(self.execute(
            "INSERT INTO speaker_embeddings(speaker_id, embedding, dim, source_path, duration_s) VALUES (?,?,?,?,?)",
            (speaker_id, emb.tobytes(), int(emb.shape[0]), source_path, duration_s),
        ).lastrowid)

    def speaker_embeddings(self) -> list[tuple[int, str, np.ndarray]]:
        rows = self.query(
            "SELECT e.speaker_id, s.name, e.embedding FROM speaker_embeddings e JOIN speakers s ON s.id=e.speaker_id"
        )
        return [(int(r["speaker_id"]), r["name"], np.frombuffer(r["embedding"], dtype=np.float32)) for r in rows]

    # --- utterances --------------------------------------------------------
    def add_utterances(self, rows: list[dict[str, Any]]) -> list[int]:
        """rows: dicts with keys matching the utterances columns plus optional 'text_seg'."""
        ids: list[int] = []
        c = self.conn()
        with _Tx(c):
            for r in rows:
                cur = c.execute(
                    "INSERT INTO utterances(segment_id,start_s,end_s,abs_start,day,cluster_label,speaker_id,speaker_name,speaker_score,text_raw,text)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (r["segment_id"], r["start_s"], r["end_s"], r["abs_start"], r["day"], r.get("cluster_label"),
                     r.get("speaker_id"), r.get("speaker_name"), r.get("speaker_score"), r["text_raw"], r["text"]),
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

    # --- kv ----------------------------------------------------------------
    def get_kv(self, key: str, default: Any = None) -> Any:
        row = self.one("SELECT value FROM kv WHERE key=?", (key,))
        return json.loads(row["value"]) if row else default

    def set_kv(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO kv(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, ensure_ascii=False)),
        )


class _Tx:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def __enter__(self) -> sqlite3.Connection:
        self.conn.execute("BEGIN")
        return self.conn

    def __exit__(self, et, ev, tb) -> bool:
        if et is None:
            self.conn.execute("COMMIT")
        else:
            self.conn.execute("ROLLBACK")
        return False
