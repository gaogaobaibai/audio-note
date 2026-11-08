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
memory_notes       long-lived notes (entities, topics, todos) maintained by the daily analysis
kv                 misc state
"""
from __future__ import annotations

import json
import sqlite3
import threading
import weakref
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
)

#: speaker_embeddings.origin values whose voice cannot contain anybody else (the user recorded / uploaded it)
TRUSTED_SAMPLE_ORIGINS = frozenset({"mic", "file"})


#: kv key prefix holding the utterance count at the last relation inference of a speaker (analysis.relations)
RELATION_KV_PREFIX = "relation_inferred_n:"


def relation_cache_key(sid: int) -> str:
    return f"{RELATION_KV_PREFIX}{int(sid)}"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


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
        c = self.conn()
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.executescript(SCHEMA)
        self._migrate(c)

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
        'system_share'."""
        ids: list[int] = []
        c = self.conn()
        with _Tx(c):
            for r in rows:
                cur = c.execute(
                    "INSERT INTO utterances(segment_id,start_s,end_s,abs_start,day,cluster_label,speaker_id,speaker_name,speaker_score,"
                    "text_raw,text,speaker_source,loudness_db,system_share) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (r["segment_id"], r["start_s"], r["end_s"], r["abs_start"], r["day"], r.get("cluster_label"),
                     r.get("speaker_id"), r.get("speaker_name"), r.get("speaker_score"), r["text_raw"], r["text"],
                     r.get("speaker_source"), r.get("loudness_db"), r.get("system_share")),
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
