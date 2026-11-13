"""Processor: stored segments -> speaker-attributed, corrected utterances.

    Processor(VoiceStage, TextStage)
        process_pending() / process_segment(seg_id)        the pipeline proper (this file)
        relabel() / relabel_segment(seg_id)                relabel.py
        preview_split(utt_id) / split_utterance(utt_id)    split.py

One segment (`_process`) is a plain sequence:

    audio  = read_segment(path)
    turns  = self._voice(seg, audio)                   # stage_voice.py: diarize, faint gate, identify
    chunks = build_chunks(turns, ...)                  # chunking.py; none -> 'no_speech', audio file deleted
    spoken, billed, filtered = self._transcribe(seg, audio, chunks)      # stage_text.py: ASR
    texts, edits_json = self._correct(spoken)          # stage_text.py: correction
    rows   = self._rows(seg, audio, spoken, texts)
    ids    = self._store(seg, spoken, rows, edits_json)        # transcripts + utterances, ONE transaction
    self._after_store(seg_id, ids, days)               # hooks (below)
    embeddings for retrieval, status 'done', day Markdown

Hooks for the stages
--------------------
Row hooks: while `_rows` builds the utterance rows (for a segment and for the pieces of a split) each stage may add
columns it knows about; every key that is a column of `utterances` is stored with the row (insert_utterance):

    self._voice_row(row, chunk)          stage_voice.py   (e.g. speaker_guess_id, speaker_meta)
    self._text_row(row, chunk, raw)      stage_text.py    (e.g. text_source, flags)

After-store hooks: `_after_store(seg_id, utterance_ids, days)` runs once the utterances of a segment are committed,
and again for the pieces of a split (split.py). utterance_ids[i] is the row of spoken[i]; days are the YYYY-MM-DD
days touched. While it runs, `self.store_ctx` (StoreContext) holds what the hooks may need beyond the ids: the
segment row, the audio, the chunks with their raw text, the rows as stored and the applied correction edits.
It runs, in order:

    1. self._voice_after_store(seg_id, utterance_ids, days)      stage_voice.py   (voice embeddings per utterance)
    2. self._text_after_store(seg_id, utterance_ids, days)       stage_text.py    (text revisions)
    3. fn(processor, seg_id, utterance_ids, days) for fn in AFTER_STORE_HOOKS     (register_after_store: session
                                                                                   sync, events, ...)

`self.scratch` is a dict that lives for one segment (or one split): emptied before the stages run, handed to the
hooks as `store_ctx.extra`, emptied again after them. A stage can leave there what its hooks need later.

All hooks are no-ops today. A hook that raises is logged and skipped: a row hook only loses its extra columns, and
when the after-store hooks run the rows are already in the database, so a hook can neither fail nor requeue the
segment. After-store hooks are not called when nothing was stored.

Stop requests: a stage that sees `stop_event` set raises chunking.Stopped; process_segment then puts the segment
back to 'new' (not 'failed') and process_pending ends its run.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..audio.vad import SAMPLE_RATE
from ..config import Config
from ..db import DB
from ..llm.client import LLMError
from . import relabel as _relabel
from . import split as _split
from .chunking import Chunk, F, Stopped, log, stage_timer
from .stage_text import TextStage
from .stage_voice import VoiceStage

STALE_PROCESSING_S = 600   # a 'processing' row untouched for this long belongs to a dead run
EMBED_MAX_FAILURES = 3     # consecutive failed embedding passes after which a Processor stops trying
EMBED_PASS_LIMIT = 128     # sentences (per vector view) one embedding pass after a segment may fill


# --------------------------------------------------------------------------- tokenizer
def tokenize(text: str) -> str:
    """Space-joined jieba tokens for the FTS index; prefers memory.index.tokenize when available."""
    try:
        from ..memory.index import tokenize as _tok  # may still be in progress / absent
        return _tok(text)
    except Exception:  # noqa: BLE001 - ImportError, SyntaxError while the module is being written, ...
        pass
    try:
        import jieba
        return " ".join(w for w in jieba.lcut(text) if w.strip())
    except Exception:  # noqa: BLE001
        return " ".join(text.split())


# --------------------------------------------------------------------------- storage helpers
def transcript_item(i: int, chunk: Chunk, raw: str, text: str) -> dict[str, Any]:
    """One entry of transcripts.raw_json: what one chunk was when it was transcribed."""
    return {
        "i": i, "start_s": round(chunk.start_s, 3), "end_s": round(chunk.end_s, 3), "cluster": chunk.cluster,
        "speaker_id": chunk.speaker_id, "speaker_name": chunk.speaker_name, "score": chunk.score,
        "text_raw": raw, "text": text,
    }


def utterance_columns(con: sqlite3.Connection) -> list[str]:
    """Column names of the utterances table as it is now (migrations add columns), without the rowid alias."""
    return [str(r[1]) for r in con.execute("PRAGMA table_info(utterances)").fetchall() if str(r[1]) != "id"]


def insert_utterance(con: sqlite3.Connection, row: dict[str, Any], columns: Sequence[str] | None = None,
                     above_id: int | None = None) -> int:
    """INSERT one utterances row (and its utterance_tokens row when row["text_seg"] is not empty) on `con`,
    inside the caller's transaction. Returns the new id.

    Every key of `row` that is a column of the table is written, so a column added by a migration only needs the
    key in the row dict (the same keys DB.add_utterances takes); other keys (text_seg, ...) are ignored. A dict /
    list value is stored as JSON (the JSON columns: flags, speaker_meta). `above_id`: take an id above every
    existing id AND above this one (split: the id of the row being replaced must never come back for a piece)."""
    cols = [c for c in (columns if columns is not None else F.utterance_columns(con)) if c in row]
    names, marks = list(cols), ["?"] * len(cols)
    params = [json.dumps(row[c], ensure_ascii=False, default=str) if isinstance(row[c], (dict, list, tuple)) else row[c]
              for c in cols]
    if above_id is not None:
        names.insert(0, "id")
        marks.insert(0, "(SELECT MAX(COALESCE(MAX(id),0), ?)+1 FROM utterances)")
        params.insert(0, int(above_id))
    cur = con.execute(f"INSERT INTO utterances({','.join(names)}) VALUES ({','.join(marks)})", params)
    uid = int(cur.lastrowid)
    if row.get("text_seg"):
        con.execute("INSERT INTO utterance_tokens(utterance_id, text_seg) VALUES (?,?)", (uid, row["text_seg"]))
    return uid


def edits_list(edits_json: str | None) -> list[dict[str, Any]]:
    """The applied edits of TextStage._correct as a list ([] for None / anything unusable)."""
    if not edits_json:
        return []
    try:
        data = json.loads(edits_json)
    except (TypeError, ValueError):
        return []
    return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else []


# --------------------------------------------------------------------------- after-store hooks
@dataclass
class StoreContext:
    """What the after-store hooks may read (Processor.store_ctx, only set while Processor._after_store runs).
    utterance_ids[i], rows[i] and spoken[i] describe the same utterance; edits[k]["i"] indexes them."""
    origin: str                                # "process" (a whole segment) | "split" (the pieces of one utterance)
    seg: Any                                   # the segments row
    audio: np.ndarray                          # the whole segment: 16 kHz mono float32
    spoken: list[tuple[Chunk, str]]            # (chunk, raw ASR text)
    rows: list[dict[str, Any]]                 # the utterance rows as stored (text = after correction)
    edits: list[dict[str, Any]]                # applied correction edits: {"i", "from", "to", "reason", ...}
    utterance_ids: list[int]
    days: list[str]
    replaced_utterance_id: int | None = None   # split only: the row the pieces replaced
    extra: dict[str, Any] = field(default_factory=dict)   # Processor.scratch: what the stages left for their hooks


AfterStoreHook = Callable[["Processor", int, list[int], list[str]], None]

#: Extra after-store hooks, called as fn(processor, seg_id, utterance_ids, days) after the two stage hooks.
AFTER_STORE_HOOKS: list[AfterStoreHook] = []


def register_after_store(fn: AfterStoreHook) -> AfterStoreHook:
    """Add `fn(processor, seg_id, utterance_ids, days)` to AFTER_STORE_HOOKS (once; usable as a decorator)."""
    if fn not in AFTER_STORE_HOOKS:
        AFTER_STORE_HOOKS.append(fn)
    return fn


# --------------------------------------------------------------------------- processor
class Processor(VoiceStage, TextStage):
    """Turns stored segments into speaker-attributed, corrected utterances.

    Heavy objects (speaker engine, ASR, corrector, MiMo client) are built lazily so that
    importing / constructing the Processor never needs an API key or model files.
    """

    def __init__(self, cfg: Config, db: DB, client: Any = None, engine: Any = None, asr: Any = None,
                 corrector: Any = None, stop_event: threading.Event | None = None):
        self.cfg = cfg
        self.db = db
        self._client = client
        self._engine = engine
        self._asr = asr
        self._corrector = corrector
        self._corrector_tried = corrector is not None
        self._terms_registered = False
        self._embedder: Any = None       # memory.embed.BaseEmbedder | None (not loaded) | False (gave up)
        self._embed_failures = 0         # consecutive failed embedding passes (see _embed_new)
        self._lock = threading.Lock()
        self.stop_event = stop_event
        self.skipped_chunks = 0   # ASR chunks skipped because the content filter (421) rejected them
        self.store_ctx: StoreContext | None = None   # set while _after_store runs
        self.scratch: dict[str, Any] = {}            # per segment / per split: from the stages to their hooks

    # --- lazy components ---------------------------------------------------
    # engine: VoiceStage; asr, corrector: TextStage
    @property
    def client(self) -> Any:
        with self._lock:
            if self._client is None:
                from ..llm.client import MimoClient, MissingAPIKey
                try:
                    self._client = MimoClient(self.cfg, rpm=int(self.cfg.mimo.rpm))
                except MissingAPIKey as e:
                    raise LLMError("ASR 需要 MiMo API key：请在 config.yaml 的 mimo.api_key 填写或设置环境变量 MIMO_API_KEY") from e
                from .. import metrics

                metrics.install(self._client, self.db)   # a client built here (CLI runs) is accounted like the scheduler's
            return self._client

    def _embed_new(self, ids: list[int]) -> None:
        """Sentence embeddings for vector / hybrid retrieval: embeds every utterance that has no vector yet (the
        new rows, plus whatever an earlier pass left behind). Best effort, never raises.

        - The model cannot be built or loaded (missing files, missing package): permanent for this Processor,
          `_embedder` becomes False and no later segment tries again (the chat's Retriever embeds whatever is
          missing on its first search; `audio-note reindex` does it by hand).
        - The embedding pass itself fails (a transient error): only logged; the next segment's pass picks up the
          rows left behind. After EMBED_MAX_FAILURES failed passes in a row the Processor gives up as above."""
        if not ids or self.cfg.memory.retrieval not in ("vector", "hybrid") or self._embedder is False:
            return
        if self._embedder is None:
            try:
                from ..memory import vectors  # noqa: F401 - a missing module is as permanent as a missing model
                from ..memory.embed import get_embedder

                embedder = get_embedder(self.cfg)
                load = getattr(embedder, "load", None)
                if callable(load):
                    load()
            except Exception as e:  # noqa: BLE001
                log.warning("embedding skipped (%s); run `audio-note reindex` after fixing", e)
                self._embedder = False
                return
            self._embedder = embedder
        try:
            from ..memory import vectors

            # bounded: right after the 1.0 upgrade the whole history lacks its new 'window' vectors; filling them
            # in one pass took ~80 s per 2k sentences and kept fresh recordings waiting. Each pass catches up a bit.
            vectors.index_missing(self.db, self._embedder, limit=F.EMBED_PASS_LIMIT)
            self._embed_failures = 0
        except Exception as e:  # noqa: BLE001
            self._embed_failures += 1
            if self._embed_failures >= int(F.EMBED_MAX_FAILURES):
                log.warning("embedding skipped (%s, %d passes in a row); run `audio-note reindex` after fixing",
                            e, self._embed_failures)
                self._embedder = False
            else:
                log.warning("embedding failed this time (%s); the next segment retries", e)

    def _register_terms(self) -> None:
        """Teach jieba the enrolled speaker names + glossary terms once, so text_seg cuts them
        as single tokens (the memory chat registers the same set at query time)."""
        if self._terms_registered:
            return
        self._terms_registered = True
        try:
            from ..memory.index import register_from
            register_from(self.cfg, self.db)
        except Exception as e:  # noqa: BLE001 - memory module optional for the pipeline
            log.debug("register_from skipped: %s", e)

    # --- batch ---------------------------------------------------------------
    def process_pending(self, limit: int | None = None, retry_failed: bool = False) -> dict[str, int]:
        """Process every 'new' segment older than pipeline.min_segment_age_s (plus 'failed' ones if asked)."""
        stats = {"processed": 0, "done": 0, "no_speech": 0, "failed": 0, "utterances": 0}
        lim = int(limit) if limit else 10_000
        # a previous run killed mid-segment leaves rows in 'processing' forever: reclaim the stale ones
        n_reset = self.db.reset_stale_processing(F.STALE_PROCESSING_S)
        if n_reset:
            log.warning("reset %d segment(s) stuck in 'processing' back to 'new'", n_reset)
        rows = list(self.db.pending_segments(int(self.cfg.pipeline.min_segment_age_s), lim))
        if retry_failed and len(rows) < lim:
            rows.extend(self.db.query("SELECT * FROM segments WHERE status='failed' ORDER BY started_at LIMIT ?", (lim - len(rows),)))
        for row in rows:
            if self.stop_event is not None and self.stop_event.is_set():
                log.info("process_pending: stop requested, %d segment(s) left for the next run", len(rows) - stats["processed"])
                break
            seg_id = int(row["id"])
            stats["processed"] += 1
            try:
                n = self.process_segment(seg_id)
            except Stopped:
                # cut short by a stop request: the segment is 'new' again, so it was not processed by this run
                stats["processed"] -= 1
                log.info("process_pending: stop requested during segment #%d, %d segment(s) left for the next run",
                         seg_id, len(rows) - stats["processed"])
                break
            except Exception as e:  # noqa: BLE001 - already marked failed + logged
                stats["failed"] += 1
                log.error("segment #%d failed: %s", seg_id, e)
                continue
            seg = self.db.get_segment(seg_id)
            status = seg["status"] if seg else "done"
            if status == "no_speech":
                stats["no_speech"] += 1
            else:
                stats["done"] += 1
            stats["utterances"] += n
        log.info("process_pending: %s", stats)
        return stats

    # --- single segment ----------------------------------------------------
    def process_segment(self, seg_id: int) -> int:
        """Run the full pipeline on one segment. Returns utterances written.
        On any error the segment is marked 'failed' (error text stored) and the exception re-raised. A stop
        request in the middle of the segment (Stopped) is not an error: the segment goes back to 'new', so the
        next run does it again, and Stopped is re-raised for the caller to end its loop."""
        seg = self.db.get_segment(seg_id)
        if seg is None:
            raise ValueError(f"segment {seg_id} not found")
        self.db.set_segment_status(seg_id, "processing")
        try:
            n = self._process(seg)
        except Stopped:
            log.info("segment #%d: stop requested, back to 'new' for the next run", seg_id)
            self.db.set_segment_status(seg_id, "new")
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("segment #%d failed", seg_id)
            self.db.set_segment_status(seg_id, "failed", f"{type(e).__name__}: {e}"[:1000])
            raise
        finally:
            self.store_ctx, self.scratch = None, {}      # whatever happened, nothing of this segment is kept alive
        return n

    def _process(self, seg: Any) -> int:
        seg_id = int(seg["id"])
        path = Path(seg["path"])
        F._parse_iso(seg["started_at"])      # a broken row fails here, before any model or API call
        if not path.exists():
            raise FileNotFoundError(f"segment file missing: {path}")
        audio = F.read_segment(path)
        audio_len_s = len(audio) / SAMPLE_RATE
        self.scratch = {}
        self._register_terms()
        self._forget_attempt(seg_id)

        # who spoke when -> what to send to ASR
        turns = self._voice(seg, audio)
        chunks = F.build_chunks(turns, audio_len_s, float(self.cfg.mimo.asr_max_chunk_s)) if turns else []
        if not chunks:
            return self._no_speech(seg_id, path)

        # what was said
        with stage_timer(self.db, "asr", sum(c.audio_end_s - c.audio_start_s for c in chunks)):
            spoken, billed, filtered = self._transcribe(seg, audio, chunks)
        log.info("segment #%d: %d turns -> %d chunks -> %d with text, %d filtered (%.0fs billed)",
                 seg_id, len(turns), len(chunks), len(spoken), filtered, billed)
        with stage_timer(self.db, "correct"):
            texts, edits_json = self._correct(spoken)

        # transcripts + utterances, then everything that hangs off the new rows
        with stage_timer(self.db, "store"):
            rows = self._rows(seg, audio, spoken, texts)
            ids = self._store(seg, spoken, rows, edits_json)
        days = sorted({r["day"] for r in rows})
        self.store_ctx = StoreContext("process", seg, audio, spoken, rows, F.edits_list(edits_json), ids, days,
                                      extra=self.scratch)
        self._after_store(seg_id, ids, days)
        self._embed_new(ids)
        self.db.set_segment_status(seg_id, "done")
        F.write_days_markdown(self.db, self.cfg, days)
        return len(rows)

    def _forget_attempt(self, seg_id: int) -> None:
        """Idempotency for retries: drop anything a previous attempt at this segment wrote."""
        self.db.delete_utterances_for_segment(seg_id)
        self.db.execute("DELETE FROM transcripts WHERE segment_id=?", (seg_id,))
        self.db.delete_cluster_embeddings(seg_id)

    def _no_speech(self, seg_id: int, path: Path) -> int:
        """Nothing to transcribe: the audio file is deleted and the segment closed as 'no_speech'."""
        log.info("segment #%d: no speech turns -> no_speech, deleting %s", seg_id, path.name)
        try:
            path.unlink()
        except OSError as e:
            log.warning("cannot delete %s: %s", path, e)
        self.db.set_segment_status(seg_id, "no_speech")
        return 0

    def _rows(self, seg: Any, audio: np.ndarray, spoken: list[tuple[Chunk, str]], texts: list[str]) -> list[dict[str, Any]]:
        """The utterances rows for `spoken` (one per chunk with text), `texts[i]` being the text to store. The
        stages' row hooks (_voice_row, _text_row) may add columns of their own to each row."""
        seg_id = int(seg["id"])
        started_at = F._parse_iso(seg["started_at"])
        unknown = self.cfg.speakers.unknown_label
        pairs, block_s = F._segment_channels(seg)
        rows: list[dict[str, Any]] = []
        for i, (c, raw) in enumerate(spoken):
            text = texts[i]
            abs_start = started_at + timedelta(seconds=c.start_s)
            row = {
                "segment_id": seg_id, "start_s": c.start_s, "end_s": c.end_s,
                "abs_start": abs_start.isoformat(timespec="milliseconds"), "day": abs_start.strftime("%Y-%m-%d"),
                "cluster_label": c.cluster, "speaker_id": c.speaker_id,
                "speaker_name": c.speaker_name or unknown, "speaker_score": c.score,
                "text_raw": raw, "text": text, "text_seg": F.tokenize(text),
                "loudness_db": F.speech_dbfs(F._clip(audio, c.start_s, c.end_s)),
                "system_share": F.system_share(pairs, block_s, c.start_s, c.end_s),
            }
            try:
                self._voice_row(row, c)
            except Exception as e:  # noqa: BLE001 - the sentence is stored without the voice stage's extra columns
                log.warning("row hook voice failed for segment #%d: %s", seg_id, e, exc_info=True)
            try:
                self._text_row(row, c, raw)
            except Exception as e:  # noqa: BLE001 - the sentence is stored without the text stage's extra columns
                log.warning("row hook text failed for segment #%d: %s", seg_id, e, exc_info=True)
            rows.append(row)
        return rows

    def _store(self, seg: Any, spoken: list[tuple[Chunk, str]], rows: list[dict[str, Any]],
               edits_json: str | None) -> list[int]:
        """Write the segment's transcripts row and its utterances (with their FTS tokens) in ONE transaction, so a
        segment never has one without the other. Returns the new utterance ids: ids[i] is the row of spoken[i].
        (An empty `spoken` still leaves the transcripts row: the segment was transcribed, to nothing.)"""
        seg_id = int(seg["id"])
        raw_items = [F.transcript_item(i, c, raw, rows[i]["text"]) for i, (c, raw) in enumerate(spoken)]
        with self.db.transaction() as con:
            con.execute(
                "INSERT INTO transcripts(segment_id, asr_model, raw_json, text_raw, text_corrected, correction_json, language)"
                " VALUES (?,?,?,?,?,?,?)",
                (seg_id, self.cfg.mimo.asr_model, json.dumps(raw_items, ensure_ascii=False),
                 "\n".join(r for _, r in spoken), "\n".join(r["text"] for r in rows), edits_json,
                 self.cfg.mimo.asr_language),
            )
            columns = F.utterance_columns(con)
            return [F.insert_utterance(con, r, columns) for r in rows]

    def _after_store(self, seg_id: int, utterance_ids: list[int], days: list[str]) -> None:
        """Hook point after the utterances of a segment (or the pieces of a split) were committed; see the module
        docstring for the list of hooks, their order and what `self.store_ctx` offers them. `store_ctx` and
        `scratch` are cleared when the hooks are done (they hold the segment's audio and whatever the stages kept)."""
        try:
            if not utterance_ids:
                return
            hooks: list[tuple[str, Callable[[], None]]] = [
                ("voice", lambda: self._voice_after_store(seg_id, utterance_ids, days)),
                ("text", lambda: self._text_after_store(seg_id, utterance_ids, days)),
            ]
            for fn in list(F.AFTER_STORE_HOOKS):
                hooks.append((getattr(fn, "__name__", "hook"), lambda fn=fn: fn(self, seg_id, utterance_ids, days)))
            for name, call in hooks:
                try:
                    call()
                except Exception as e:  # noqa: BLE001 - the rows are in; a hook never fails the segment
                    log.warning("after-store hook %s failed for segment #%d: %s", name, seg_id, e, exc_info=True)
        finally:
            self.store_ctx = None
            self.scratch = {}

    # --- re-identification without ASR (relabel.py) --------------------------
    def relabel_segment(self, seg_id: int) -> int:
        """Re-run speaker identification on a 'done' segment using the stored utterances as turns (no
        diarization, no ASR, no billing). Returns the number of utterances whose speaker changed.
        See relabel.relabel_segment."""
        return _relabel.relabel_segment(self, seg_id)

    def relabel(self, day: str | None = None, segment_id: int | None = None) -> dict[str, int]:
        """relabel_segment over every 'done' segment whose audio still exists (optionally one day / one id)."""
        return _relabel.relabel(self, day=day, segment_id=segment_id)

    # --- splitting a mixed utterance (split.py) ------------------------------
    def preview_split(self, utt_id: int, at: list[float] | None = None, auto: bool = True) -> dict[str, Any]:
        """The pieces split_utterance would produce (bounds + identification), without ASR or DB writes."""
        return _split.preview_split(self, utt_id, at=at, auto=auto)

    def split_utterance(self, utt_id: int, at: list[float] | None = None, auto: bool = False) -> dict[str, Any]:
        """Split one (mixed) utterance into several, each transcribed again; see split.split_utterance.
        Returns {"removed", "pieces": [{id, start_s, end_s, speaker, speaker_id, score, text}], "billed_s"}."""
        return _split.split_utterance(self, utt_id, at=at, auto=auto)

    # the split helpers were private methods of the old Processor: kept as thin delegates
    def _utterance_for_split(self, utt_id: int) -> tuple[Any, Any, np.ndarray]:
        return _split._utterance_for_split(self, utt_id)

    def _pieces_manual(self, clip: np.ndarray, at: list[float], seg: Any = None, offset_s: float = 0.0,
                       veto: frozenset[int] = frozenset()) -> list[dict[str, Any]]:
        return _split._pieces_manual(self, clip, at, seg, offset_s, veto)

    def _pieces_auto(self, clip: np.ndarray, seg: Any = None, offset_s: float = 0.0,
                     veto: frozenset[int] = frozenset()) -> list[dict[str, Any]]:
        return _split._pieces_auto(self, clip, seg, offset_s, veto)

    @staticmethod
    def _merge_short_pieces(pieces: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return _split._merge_short_pieces(pieces)

    def _plan_split(self, utt_id: int, at: list[float] | None, auto: bool) -> tuple[Any, Any, np.ndarray, list[dict[str, Any]]]:
        return _split._plan_split(self, utt_id, at, auto)


# --------------------------------------------------------------------------- one-off helper
def transcribe_file(cfg: Config, client: Any, path: Path | str, db: DB | None = None,
                    engine: Any = None, asr: Any = None, corrector: Any = None,
                    correct: bool = True) -> list[dict[str, Any]]:
    """Diarize + identify + ASR (+ correction) one audio file without writing to the DB.

    Returns [{"start_s","end_s","cluster","speaker","speaker_id","score","text_raw","text"}].
    Speaker identification uses the enrolled speakers from `db` (or the configured DB if it exists).
    """
    from ..audio.importer import decode_any

    path = Path(path)
    audio = decode_any(path)
    audio_len_s = len(audio) / SAMPLE_RATE
    if engine is None:
        from .speakers import SpeakerEngine
        engine = SpeakerEngine(cfg)
    if asr is None:
        from .asr import ASR
        asr = ASR(cfg, client, vad=getattr(engine, "vad", None))
    if db is None and cfg.db_path.exists():
        db = DB(cfg.db_path)

    turns = list(engine.diarize(audio))
    if turns and db is not None:
        turns = list(engine.identify_turns(audio, turns, db))
    unknown = cfg.speakers.unknown_label
    chunks = F.build_chunks(turns, audio_len_s, float(cfg.mimo.asr_max_chunk_s)) if turns else []
    spoken, _billed, _filtered = F.transcribe_chunks(asr, audio, chunks, F._asr_concurrency(cfg))

    texts = [raw for _, raw in spoken]
    if correct and spoken and bool(cfg.correction.enabled):
        if corrector is None:
            try:
                from .correct import default_corrector
                corrector = default_corrector(cfg, db, client)      # by correction.mode, like Processor.corrector
            except Exception as e:  # noqa: BLE001
                log.warning("correction unavailable: %s", e)
                corrector = None
        texts, _edits = F.correct_spoken(corrector, spoken, cfg, db)

    return [{
        "start_s": round(c.start_s, 3), "end_s": round(c.end_s, 3), "cluster": c.cluster,
        "speaker": c.speaker_name or unknown, "speaker_id": c.speaker_id, "score": c.score,
        "text_raw": raw, "text": texts[i],
    } for i, (c, raw) in enumerate(spoken)]


__all__ = ["AFTER_STORE_HOOKS", "Processor", "StoreContext", "register_after_store", "tokenize", "transcribe_file"]
