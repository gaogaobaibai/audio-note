"""Processing pipeline: segment file -> diarize -> identify -> ASR chunks -> correction -> DB -> daily Markdown.

See docs/DESIGN.md section 5. The Processor is meant to run in the scheduler thread while the
Recorder thread keeps writing new segments; the DB hands out one connection per thread.

Public pieces
-------------
Chunk                 one speaker-homogeneous stretch of audio sent to ASR as a unit
build_chunks(...)     pure function: diarization turns -> ASR chunks (merge / split / pad / drop)
Processor             process_pending() / process_segment(seg_id) / relabel_segment(seg_id) / split_utterance(utt_id)
write_day_markdown()  regenerate transcripts/YYYY-MM-DD.md from the DB
transcribe_file()     one-off diarize + identify + ASR of any audio file (no DB writes)
system_share()        share of a time range's energy that came from system audio (segments.channel_rms)

Speakers v4: a turn whose system-audio share (segments.channel_rms, see system_share) reaches
speakers.system_turn_share is never identified as the self speaker (the other side of a meeting / call), and a
sentence is never re-identified as a speaker the user vetoed for it (utterance_rejections).
"""
from __future__ import annotations

import inspect
import json
import logging
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..audio.loudness import is_faint, min_speech_dbfs, speech_dbfs
from ..audio.storage import read_segment
from ..audio.vad import SAMPLE_RATE
from ..config import Config
from ..db import DB
from ..llm.client import ContentFiltered, LLMError
from .speakers import Turn

log = logging.getLogger(__name__)

STALE_PROCESSING_S = 600   # a 'processing' row untouched for this long belongs to a dead run
SPLIT_MIN_PIECE_S = 0.6    # a split piece shorter than this is merged into the previous one
SPLIT_EDGE_S = 0.3         # a manual split point must lie inside (SPLIT_EDGE_S, duration - SPLIT_EDGE_S)
SPLIT_MIN_TURN_S = 1.0     # auto split: diarization turns shorter than this are ignored
SPLIT_PAD_S = 0.15         # ASR padding around a piece (same as build_chunks)
DEFAULT_SYSTEM_TURN_SHARE = 0.6


# --------------------------------------------------------------------------- system audio share
def channel_pairs(blob: bytes | None) -> np.ndarray | None:
    """segments.channel_rms (float16, interleaved mic, sys, mic, sys, ...) as an (n, 2) float64 array; None when
    there is no usable data."""
    if not blob:
        return None
    try:
        a = np.frombuffer(bytes(blob), dtype=np.float16)
    except (TypeError, ValueError):
        return None
    n = a.shape[0] // 2
    if n == 0:
        return None
    a = np.nan_to_num(a[: 2 * n].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0).reshape(n, 2)
    return np.abs(a)


def system_share(pairs: np.ndarray | None, block_s: float | None, start_s: float, end_s: float) -> float | None:
    """Σsys² / (Σsys² + Σmic²) over the capture blocks the range [start_s, end_s) covers (weighted by overlap; one
    (mic_rms, system_rms) pair per `block_s` seconds from the segment start). None when the segment has no
    channel data, the range lies outside it, or the covered blocks are silent."""
    if pairs is None or not block_s or float(block_s) <= 0:
        return None
    bs = float(block_s)
    a, b = max(0.0, float(start_s)), float(end_s)
    if b <= a:
        return None
    i0 = int(math.floor(a / bs))
    i1 = min(int(math.ceil(b / bs)), pairs.shape[0])
    if i0 >= i1:
        return None
    idx = np.arange(i0, i1)
    w = np.minimum((idx + 1) * bs, b) - np.maximum(idx * bs, a)
    w = np.clip(w, 0.0, None)
    blk = pairs[i0:i1]
    mic = float(np.sum(w * blk[:, 0] ** 2))
    sys_ = float(np.sum(w * blk[:, 1] ** 2))
    tot = mic + sys_
    if not math.isfinite(tot) or tot <= 0:
        return None
    return round(sys_ / tot, 4)


def system_turn_share(cfg: Config) -> float:
    try:
        return float(cfg.speakers.get("system_turn_share", DEFAULT_SYSTEM_TURN_SHARE))
    except (TypeError, ValueError, AttributeError):
        return DEFAULT_SYSTEM_TURN_SHARE


def _segment_channels(seg: Any) -> tuple[np.ndarray | None, float | None]:
    """(channel pairs, block seconds) of a segment row; (None, None) when it has none (mic only, imported, old row)."""
    try:
        blob, bs = seg["channel_rms"], seg["channel_block_s"]
    except (IndexError, KeyError, TypeError):
        return None, None
    pairs = channel_pairs(blob)
    try:
        bs_f = float(bs) if bs is not None else None
    except (TypeError, ValueError):
        bs_f = None
    if pairs is None or not bs_f or bs_f <= 0:
        return None, None
    return pairs, bs_f


# --------------------------------------------------------------------------- chunks
@dataclass
class Chunk:
    """A stretch of audio from one diarization cluster, bounded for ASR.

    start_s / end_s are the speech bounds (used for timestamps); audio_start_s / audio_end_s
    include the padding actually sent to ASR (clamped to the file).
    """
    start_s: float
    end_s: float
    cluster: int
    speaker_id: int | None = None
    speaker_name: str | None = None
    score: float | None = None
    audio_start_s: float = 0.0
    audio_end_s: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def slice(self, audio: np.ndarray) -> np.ndarray:
        a = max(0, int(round(self.audio_start_s * SAMPLE_RATE)))
        b = min(len(audio), int(round(self.audio_end_s * SAMPLE_RATE)))
        return audio[a:b]


def _clip(audio: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    a = max(0, int(float(start_s) * SAMPLE_RATE))
    b = min(len(audio), int(float(end_s) * SAMPLE_RATE))
    return audio[a:b] if b > a else audio[:0]


def drop_faint_turns(audio: np.ndarray, turns: Sequence[Turn], threshold: float | None) -> tuple[list[Turn], int, float]:
    """Remove the diarization turns whose speech level (audio.loudness.speech_dbfs) is below `threshold` dBFS:
    far-away / unclear voices are neither transcribed nor used for speaker identification. Returns
    (kept turns in order, dropped count, dropped seconds); threshold None keeps everything."""
    if threshold is None:
        return list(turns), 0, 0.0
    kept: list[Turn] = []
    dropped, faint_s = 0, 0.0
    for t in turns:
        if speech_dbfs(_clip(audio, t.start, t.end)) < float(threshold):
            dropped += 1
            faint_s += max(0.0, t.duration)
        else:
            kept.append(t)
    return kept, dropped, faint_s


def build_chunks(turns: Sequence[Turn], audio_len_s: float, max_chunk_s: float, gap_s: float = 1.0,
                 pad_s: float = 0.15, min_chunk_s: float = 0.6) -> list[Chunk]:
    """Turn diarization turns into ASR chunks (pure function).

    1. Sort turns by start; merge consecutive turns of the same cluster whose gap is <= gap_s,
       as long as the merged span stays <= max_chunk_s (otherwise a new chunk starts).
    2. A single turn longer than max_chunk_s is cut into equal pieces <= max_chunk_s.
    3. Chunks shorter than min_chunk_s are dropped.
    4. pad_s is added on both sides for the audio slice, clamped to [0, audio_len_s].
    """
    max_chunk_s = float(max_chunk_s) if max_chunk_s and max_chunk_s > 0 else float("inf")
    pieces: list[Turn] = []
    for t in sorted(turns, key=lambda t: (t.start, t.end)):
        start, end = float(t.start), min(float(t.end), float(audio_len_s))
        if end <= start:
            continue
        dur = end - start
        if dur > max_chunk_s:
            n = int(math.ceil(dur / max_chunk_s))
            step = dur / n
            for k in range(n):
                pieces.append(Turn(start + k * step, start + (k + 1) * step, t.cluster, t.speaker_id, t.speaker_name, t.score))
        else:
            pieces.append(Turn(start, end, t.cluster, t.speaker_id, t.speaker_name, t.score))

    merged: list[Chunk] = []
    for p in pieces:
        last = merged[-1] if merged else None
        if (last is not None and last.cluster == p.cluster and p.start - last.end_s <= gap_s
                and p.end - last.start_s <= max_chunk_s):
            last.end_s = max(last.end_s, p.end)
        else:
            merged.append(Chunk(p.start, p.end, p.cluster, p.speaker_id, p.speaker_name, p.score))

    out: list[Chunk] = []
    for c in merged:
        if c.duration_s < min_chunk_s:
            continue
        c.audio_start_s = max(0.0, c.start_s - pad_s)
        c.audio_end_s = min(float(audio_len_s), c.end_s + pad_s)
        out.append(c)
    return out


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


# --------------------------------------------------------------------------- markdown
def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def segment_heading(seg_id: int, seg_path: str, started_at: datetime, duration_s: float,
                    source: str | None = None, origin: str | None = None) -> str:
    """`## HH:MM:SS–HH:MM:SS (#id file.opus)[（导入：name）]` — the segment reference lets the user run
    `speakers label <id> ... --from/--to` or `speakers label --at` straight from the transcript."""
    end = started_at + timedelta(seconds=float(duration_s or 0.0))
    label = ""
    if source == "import" and origin:
        label = f"（导入：{Path(origin).name}）"
    return f"## {started_at.strftime('%H:%M:%S')}–{end.strftime('%H:%M:%S')} (#{seg_id} {Path(seg_path).name}){label}"


def write_day_markdown(db: DB, cfg: Config, day: str) -> Path:
    """Regenerate transcripts/YYYY-MM-DD.md from the DB (grouped by segment)."""
    unknown = cfg.speakers.unknown_label
    rows = db.query(
        "SELECT u.*, s.path AS seg_path, s.started_at AS seg_started_at, s.duration_s AS seg_duration_s,"
        " s.source AS seg_source, s.origin AS seg_origin"
        " FROM utterances u JOIN segments s ON s.id=u.segment_id WHERE u.day=? ORDER BY u.abs_start, u.start_s, u.id",
        (day,),
    )
    lines: list[str] = [f"# {day} 记录", ""]
    cur_seg: int | None = None
    for r in rows:
        if r["segment_id"] != cur_seg:
            cur_seg = r["segment_id"]
            if lines[-1] != "":
                lines.append("")
            lines.append(segment_heading(int(r["segment_id"]), r["seg_path"], _parse_iso(r["seg_started_at"]),
                                         r["seg_duration_s"], r["seg_source"], r["seg_origin"]))
        ts = _parse_iso(r["abs_start"]).strftime("%H:%M:%S")
        name = r["speaker_name"] or unknown
        text = (r["text"] or "").replace("\r", "").replace("\n", " ").strip()
        lines.append(f"- **{ts} {name}**：{text}")
    out_dir = cfg.transcripts_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{day}.md"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines).rstrip("\n") + "\n")
    return path


def load_glossary(cfg: Config, db: DB | None = None) -> list[str]:
    """glossary.txt terms (one per line, '#' comments) plus all enrolled speaker names."""
    terms: list[str] = []
    try:
        p = cfg.glossary_path
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip()
                    if w and not w.startswith("#"):
                        terms.append(w)
    except Exception as e:  # noqa: BLE001
        log.warning("cannot read glossary: %s", e)
    if db is not None:
        try:
            terms.extend(r["name"] for r in db.query("SELECT name FROM speakers ORDER BY id"))
        except Exception as e:  # noqa: BLE001
            log.warning("cannot list speakers for glossary: %s", e)
    seen: set[str] = set()
    return [t for t in terms if not (t in seen or seen.add(t))]


# --------------------------------------------------------------------------- processor
class Processor:
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
        self._embedder: Any = None       # memory.embed.BaseEmbedder | None (not loaded) | False (failed)
        self._lock = threading.Lock()
        self.stop_event = stop_event
        self.skipped_chunks = 0   # ASR chunks skipped because the content filter (421) rejected them

    # --- lazy components ---------------------------------------------------
    @property
    def client(self) -> Any:
        with self._lock:
            if self._client is None:
                from ..llm.client import MimoClient, MissingAPIKey
                try:
                    self._client = MimoClient(self.cfg, rpm=int(self.cfg.mimo.rpm))
                except MissingAPIKey as e:
                    raise LLMError("ASR 需要 MiMo API key：请在 config.yaml 的 mimo.api_key 填写或设置环境变量 MIMO_API_KEY") from e
            return self._client

    @property
    def engine(self) -> Any:
        with self._lock:
            if self._engine is None:
                from .speakers import SpeakerEngine
                self._engine = SpeakerEngine(self.cfg)
            return self._engine

    @property
    def asr(self) -> Any:
        if self._asr is None:
            client = self.client  # may raise a clear LLMError
            with self._lock:
                if self._asr is None:
                    from .asr import ASR
                    # share the engine's Silero VAD when it is already built (saves one onnx session)
                    vad = getattr(self._engine, "vad", None) if self._engine is not None else None
                    self._asr = ASR(self.cfg, client, vad=vad)
        return self._asr

    @property
    def corrector(self) -> Any:
        """Corrector instance or None (disabled, unavailable, or no API key)."""
        if self._corrector is not None or self._corrector_tried:
            return self._corrector
        with self._lock:
            if self._corrector_tried:
                return self._corrector
            self._corrector_tried = True
            if not bool(self.cfg.correction.enabled):
                log.info("correction disabled by config")
                return None
            try:
                from .correct import Corrector
            except Exception as e:  # noqa: BLE001 - module written in parallel / missing
                log.warning("speech.correct unavailable (%s); skipping correction", e)
                return None
        try:
            client = self.client
        except LLMError as e:
            log.warning("no MiMo client for correction: %s", e)
            return None
        try:
            with self._lock:
                self._corrector = Corrector(self.cfg, client)
        except Exception as e:  # noqa: BLE001
            log.warning("cannot build Corrector (%s); skipping correction", e)
            self._corrector = None
        return self._corrector

    def _embed_new(self, ids: list[int]) -> None:
        """Sentence embeddings for vector / hybrid retrieval. Best effort: a missing model only logs
        (the chat's Retriever embeds whatever is missing on its first search)."""
        if not ids or self.cfg.memory.retrieval not in ("vector", "hybrid") or self._embedder is False:
            return
        try:
            if self._embedder is None:
                from ..memory.embed import get_embedder

                self._embedder = get_embedder(self.cfg)
            from ..memory import vectors

            vectors.index_missing(self.db, self._embedder)
        except Exception as e:  # noqa: BLE001
            log.warning("embedding skipped (%s); run `audio-note reindex` after fixing", e)
            self._embedder = False

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
        n_reset = self.db.reset_stale_processing(STALE_PROCESSING_S)
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
        On any error the segment is marked 'failed' (error text stored) and the exception re-raised."""
        seg = self.db.get_segment(seg_id)
        if seg is None:
            raise ValueError(f"segment {seg_id} not found")
        self.db.set_segment_status(seg_id, "processing")
        try:
            n = self._process(seg)
        except Exception as e:  # noqa: BLE001
            log.exception("segment #%d failed", seg_id)
            self.db.set_segment_status(seg_id, "failed", f"{type(e).__name__}: {e}"[:1000])
            raise
        return n

    def _process(self, seg: Any) -> int:
        seg_id = int(seg["id"])
        path = Path(seg["path"])
        started_at = _parse_iso(seg["started_at"])
        if not path.exists():
            raise FileNotFoundError(f"segment file missing: {path}")
        audio = read_segment(path)
        audio_len_s = len(audio) / SAMPLE_RATE
        self._register_terms()

        # idempotency for retries: drop anything a previous attempt wrote
        self.db.delete_utterances_for_segment(seg_id)
        self.db.execute("DELETE FROM transcripts WHERE segment_id=?", (seg_id,))
        self.db.delete_cluster_embeddings(seg_id)

        # 2. diarize, drop the faint (far-away) turns, identify the rest
        turns: list[Turn] = list(self.engine.diarize(audio))
        min_db = min_speech_dbfs(self.cfg)
        turns, n_faint, faint_s = drop_faint_turns(audio, turns, min_db)
        if n_faint:
            log.info("segment #%d: dropped %d faint turns (%.1f s, < %g dBFS)", seg_id, n_faint, faint_s, min_db)
        self.db.execute("UPDATE segments SET faint_s=? WHERE id=?", (float(faint_s), seg_id))
        pairs, block_s = _segment_channels(seg)
        if turns:
            self._mark_system_turns(turns, pairs, block_s)
            turns = self._identify(audio, turns, seg_id)
            turns = self._enforce_system_turns(turns, pairs, block_s)
        chunks = build_chunks(turns, audio_len_s, float(self.cfg.mimo.asr_max_chunk_s)) if turns else []
        if not chunks:
            log.info("segment #%d: no speech turns -> no_speech, deleting %s", seg_id, path.name)
            try:
                path.unlink()
            except OSError as e:
                log.warning("cannot delete %s: %s", path, e)
            self.db.set_segment_status(seg_id, "no_speech")
            return 0

        # 4. ASR per chunk (concurrent: API latency is ~0.8x realtime per request, independent of length)
        unknown = self.cfg.speakers.unknown_label
        spoken, billed, filtered = transcribe_chunks(self.asr, audio, chunks, _asr_concurrency(self.cfg),
                                                     stop_event=self.stop_event)
        self.skipped_chunks += filtered
        log.info("segment #%d: %d turns -> %d chunks -> %d with text, %d filtered (%.0fs billed)",
                 seg_id, len(turns), len(chunks), len(spoken), filtered, billed)

        # 5. correction
        corrected, edits_json = self._correct(spoken)

        # 6. transcripts + utterances
        raw_items: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        days: set[str] = set()
        for i, (c, raw) in enumerate(spoken):
            text = corrected[i]
            abs_start = started_at + timedelta(seconds=c.start_s)
            day = abs_start.strftime("%Y-%m-%d")
            days.add(day)
            raw_items.append({
                "i": i, "start_s": round(c.start_s, 3), "end_s": round(c.end_s, 3), "cluster": c.cluster,
                "speaker_id": c.speaker_id, "speaker_name": c.speaker_name, "score": c.score,
                "text_raw": raw, "text": text,
            })
            rows.append({
                "segment_id": seg_id, "start_s": c.start_s, "end_s": c.end_s,
                "abs_start": abs_start.isoformat(timespec="milliseconds"), "day": day,
                "cluster_label": c.cluster, "speaker_id": c.speaker_id,
                "speaker_name": c.speaker_name or unknown, "speaker_score": c.score,
                "text_raw": raw, "text": text, "text_seg": tokenize(text),
                "loudness_db": speech_dbfs(_clip(audio, c.start_s, c.end_s)),
                "system_share": system_share(pairs, block_s, c.start_s, c.end_s),
            })
        self.db.execute(
            "INSERT INTO transcripts(segment_id, asr_model, raw_json, text_raw, text_corrected, correction_json, language)"
            " VALUES (?,?,?,?,?,?,?)",
            (seg_id, self.cfg.mimo.asr_model, json.dumps(raw_items, ensure_ascii=False),
             "\n".join(r for _, r in spoken), "\n".join(corrected), edits_json, self.cfg.mimo.asr_language),
        )
        ids = self.db.add_utterances(rows)
        self._embed_new(ids)

        # 7. done + markdown
        self.db.set_segment_status(seg_id, "done")
        for day in sorted(days):
            try:
                write_day_markdown(self.db, self.cfg, day)
            except Exception as e:  # noqa: BLE001 - the DB is the source of truth
                log.warning("cannot write markdown for %s: %s", day, e)
        return len(rows)

    def _self_id(self) -> int | None:
        try:
            me = self.db.self_speaker()
        except Exception:  # noqa: BLE001 - DB stand-ins without the method
            return None
        return int(me["id"]) if me is not None else None

    def _mark_system_turns(self, turns: list[Turn], pairs: np.ndarray | None, block_s: float | None) -> int:
        """Add the self speaker to Turn.exclude for every turn that is mostly system audio. Returns how many."""
        if pairs is None:
            return 0
        me = self._self_id()
        if me is None:
            return 0
        thr = system_turn_share(self.cfg)
        n = 0
        for t in turns:
            sh = system_share(pairs, block_s, t.start, t.end)
            if sh is not None and sh >= thr:
                t.exclude = frozenset(getattr(t, "exclude", ()) or ()) | {me}
                n += 1
        return n

    def _enforce_system_turns(self, turns: list[Turn], pairs: np.ndarray | None, block_s: float | None) -> list[Turn]:
        """Safety net after identification (engines that ignore Turn.exclude): a system-audio turn identified as the
        self speaker becomes unknown, on a fresh cluster id per original cluster so build_chunks does not merge it
        back into the user's sentences."""
        if pairs is None or not turns:
            return turns
        me = self._self_id()
        if me is None:
            return turns
        thr = system_turn_share(self.cfg)
        unknown = self.cfg.speakers.unknown_label
        next_id = max(int(t.cluster) for t in turns) + 1
        new_ids: dict[int, int] = {}
        n = 0
        for t in turns:
            if t.speaker_id != me:
                continue
            sh = system_share(pairs, block_s, t.start, t.end)
            if sh is None or sh < thr:
                continue
            if t.cluster not in new_ids:
                new_ids[t.cluster] = next_id
                next_id += 1
            t.cluster = new_ids[t.cluster]
            t.speaker_id, t.speaker_name = None, unknown
            n += 1
        if n:
            log.info("%d system-audio turn(s) were not attributed to the self speaker", n)
        return turns

    def _identify(self, audio: np.ndarray, turns: list[Turn], seg_id: int | None, per_utterance: bool = False) -> list[Turn]:
        """engine.identify_turns with the segment id (so cluster embeddings get stored); engines that do
        not take `segment_id` (fakes in tests, older implementations) are called the old way.
        `per_utterance` (relabel) prefers engine.identify_utterances: every utterance long enough is matched on
        its own audio and keeps its own score, the short ones follow their cluster."""
        fn = getattr(self.engine, "identify_utterances", None) if per_utterance else None
        if not callable(fn):
            fn = self.engine.identify_turns
        if seg_id is not None and _accepts_segment_id(fn):
            return list(fn(audio, turns, self.db, segment_id=seg_id))
        return list(fn(audio, turns, self.db))

    # --- re-identification without ASR ------------------------------------
    def relabel_segment(self, seg_id: int) -> int:
        """Re-run speaker identification on a 'done' segment using the stored utterances as turns
        (no diarization, no ASR, no billing) and update speaker_id/name/score + the day Markdown.
        Utterances the user labelled by hand (speaker_source='user' with a speaker_id; a deleted speaker
        leaves speaker_id NULL and the row is automatic again) are left untouched. Returns the number of
        utterances whose speaker changed. Used after `speakers enroll/label` so history picks up a newly
        enrolled voice; the segment's cluster_embeddings are refreshed on the way.

        A sentence is never identified as a speaker the user vetoed for it (utterance_rejections), and a mostly
        system-audio sentence (segments.channel_rms, speakers.system_turn_share) never as the self speaker; the
        engine tries the next candidate, and the write-back enforces it (unknown) whatever the engine did.
        utterances.system_share is (re)computed when the segment has channel data.

        The write is conditional: between reading the rows and the UPDATE (the embedding takes seconds)
        a request thread may hand-label one of them (name_voice / label); such a row is skipped by the
        WHERE clause and not counted as changed."""
        seg = self.db.get_segment(seg_id)
        if seg is None:
            raise ValueError(f"segment {seg_id} not found")
        path = Path(seg["path"])
        if not path.exists():
            raise FileNotFoundError(f"segment file missing (expired?): {path}")
        utts = self.db.query("SELECT * FROM utterances WHERE segment_id=? ORDER BY start_s, id", (seg_id,))
        if not utts:
            return 0
        audio = read_segment(path)
        # loudness: fill in what older rows lack, and keep the faint (far-away) utterances out of identification
        min_db = min_speech_dbfs(self.cfg)
        levels: list[float] = []
        for u in utts:
            lv = u["loudness_db"]
            if lv is None:
                lv = speech_dbfs(_clip(audio, u["start_s"], u["end_s"]))
                self.db.execute("UPDATE utterances SET loudness_db=? WHERE id=? AND loudness_db IS NULL", (lv, u["id"]))
            levels.append(float(lv))
        loud = [i for i, lv in enumerate(levels) if not is_faint(lv, min_db)]
        # vetoes (utterance_rejections) and system-audio sentences (never the self speaker)
        pairs, block_s = _segment_channels(seg)
        shares: list[float | None] = [None] * len(utts)
        if pairs is not None:
            for i, u in enumerate(utts):
                sh = system_share(pairs, block_s, float(u["start_s"]), float(u["end_s"]))
                shares[i] = sh
                if sh is not None and _row_get(u, "system_share") != sh:
                    self.db.execute("UPDATE utterances SET system_share=? WHERE id=?", (sh, u["id"]))
        rejected = self.db.rejected_speakers([int(u["id"]) for u in utts]) if hasattr(self.db, "rejected_speakers") else {}
        me = self._self_id() if pairs is not None else None
        sys_thr = system_turn_share(self.cfg)
        excludes: list[frozenset[int]] = []
        for i, u in enumerate(utts):
            ex = set(rejected.get(int(u["id"]), ()))
            if me is not None and shares[i] is not None and shares[i] >= sys_thr:
                ex.add(me)
            excludes.append(frozenset(ex))
        turns = [Turn(float(utts[i]["start_s"]), float(utts[i]["end_s"]),
                      int(utts[i]["cluster_label"] if utts[i]["cluster_label"] is not None else i), exclude=excludes[i]) for i in loud]
        if turns:
            turns = self._identify(audio, turns, seg_id, per_utterance=True)
        by_row = dict(zip(loud, turns))
        unknown = self.cfg.speakers.unknown_label
        changed = 0
        days: set[str] = set()
        for i, u in enumerate(utts):
            if u["speaker_source"] == "user" and u["speaker_id"] is not None:
                continue
            t = by_row.get(i)
            if t is None:
                # too quiet to identify: left as it is, just without a score -- unless it is attributed to a speaker it
                # may not be (a veto, or the self speaker on system audio): then it is released to unknown
                if u["speaker_id"] is not None and int(u["speaker_id"]) in excludes[i]:
                    cur = self.db.execute(
                        "UPDATE utterances SET speaker_id=NULL, speaker_name=?, speaker_score=NULL, speaker_source=NULL"
                        " WHERE id=? AND speaker_id=? AND (speaker_source IS NULL OR speaker_source<>'user')",
                        (unknown, u["id"], int(u["speaker_id"])))
                    if int(cur.rowcount or 0) == 1:
                        changed += 1
                        days.add(u["day"])
                    continue
                self.db.execute("UPDATE utterances SET speaker_score=NULL WHERE id=?", (u["id"],))
                continue
            sid_new, name, score_new = t.speaker_id, (t.speaker_name or unknown), t.score
            if sid_new is not None and sid_new in excludes[i]:
                sid_new, name = None, unknown        # the engine ignored the veto: enforce it
            cur = self.db.execute(
                "UPDATE utterances SET speaker_id=?, speaker_name=?, speaker_score=?, speaker_source=NULL WHERE id=?"
                " AND (speaker_source IS NULL OR speaker_source<>'user' OR speaker_id IS NULL)"
                " AND NOT EXISTS (SELECT 1 FROM utterance_rejections r WHERE r.utterance_id=utterances.id AND r.speaker_id=?)",
                (sid_new, name, score_new, u["id"], sid_new))
            if int(cur.rowcount or 0) != 1:
                continue                        # hand-labelled meanwhile: leave it alone
            if u["speaker_id"] != sid_new or (u["speaker_name"] or unknown) != name:
                changed += 1
                days.add(u["day"])
        if changed:
            for day in sorted(days):
                try:
                    write_day_markdown(self.db, self.cfg, day)
                except Exception as e:  # noqa: BLE001
                    log.warning("cannot write markdown for %s: %s", day, e)
        log.info("relabel segment #%d: %d/%d utterances changed", seg_id, changed, len(utts))
        return changed

    # --- splitting a mixed utterance -------------------------------------
    def _utterance_for_split(self, utt_id: int) -> tuple[Any, Any, np.ndarray]:
        """(utterance row, segment row, segment audio) or ValueError / FileNotFoundError."""
        u = self.db.one("SELECT * FROM utterances WHERE id=?", (int(utt_id),))
        if u is None:
            raise ValueError("没有这条记录")
        seg = self.db.get_segment(int(u["segment_id"]))
        if seg is None:
            raise ValueError(f"语句所属的录音段 #{u['segment_id']} 已不存在")
        path = Path(seg["path"])
        if not path.exists():
            raise FileNotFoundError("这段音频已不存在（超过保留期），无法拆分")
        return u, seg, read_segment(path)

    def _match_clip(self, clip: np.ndarray, enrolled: list[tuple[int, str, np.ndarray]], thr: float,
                    exclude: frozenset[int] = frozenset()) -> tuple[int | None, str | None, float | None]:
        """Identify one piece on its own audio (at least SPLIT_MIN_TURN_S of it); unknown otherwise.
        Negative samples and `exclude` are passed on only when there are any (engines / fakes without the
        keywords keep working)."""
        if not enrolled or len(clip) < int(SPLIT_MIN_TURN_S * SAMPLE_RATE):
            return None, None, None
        embed = getattr(self.engine, "embed", None)
        match = getattr(self.engine, "match", None)
        if not callable(embed) or not callable(match):
            return None, None, None
        from .speakers import load_negatives, negative_margin

        kw: dict[str, Any] = {}
        negatives = load_negatives(self.db)
        if negatives:
            kw["negatives"], kw["margin"] = negatives, negative_margin(self.cfg)
        if exclude:
            kw["exclude"] = exclude
        sid, name, score = match(embed(clip), enrolled, thr, **kw)
        if sid is not None and sid in exclude:
            return None, None, score
        return sid, name, score

    def _system_exclude(self, seg: Any, start_s: float, end_s: float) -> frozenset[int]:
        """{self speaker id} when [start_s, end_s) of the segment is mostly system audio, else empty."""
        pairs, block_s = _segment_channels(seg)
        if pairs is None:
            return frozenset()
        me = self._self_id()
        sh = system_share(pairs, block_s, start_s, end_s)
        if me is None or sh is None or sh < system_turn_share(self.cfg):
            return frozenset()
        return frozenset({me})

    def _pieces_manual(self, clip: np.ndarray, at: list[float], seg: Any = None, offset_s: float = 0.0,
                       veto: frozenset[int] = frozenset()) -> list[dict[str, Any]]:
        """Cut at `at` and identify every piece on its own audio; `veto` (the speakers the user rejected for the
        sentence being split) is excluded for every piece."""
        dur = len(clip) / SAMPLE_RATE
        pts: list[float] = []
        for x in at:
            try:
                v = float(x)
            except (TypeError, ValueError):
                raise ValueError(f"拆分点应为秒数: {x!r}")
            if not (SPLIT_EDGE_S < v < dur - SPLIT_EDGE_S):
                raise ValueError(f"拆分点 {v:.1f}s 必须在 {SPLIT_EDGE_S:.1f}s 和 {dur - SPLIT_EDGE_S:.1f}s 之间")
            if pts and v < pts[-1]:
                raise ValueError("拆分点必须按时间升序")
            if not pts or v - pts[-1] > 1e-6:
                pts.append(v)
        if not pts:
            raise ValueError("需要至少一个拆分点")
        enrolled = self.db.speaker_embeddings()
        thr = float(self.cfg.speakers.identify_threshold)
        bounds = [0.0, *pts, dur]
        pieces: list[dict[str, Any]] = []
        for a, b in zip(bounds, bounds[1:]):
            excl = (self._system_exclude(seg, offset_s + a, offset_s + b) if seg is not None else frozenset()) | veto
            sid, name, score = self._match_clip(clip[int(a * SAMPLE_RATE): int(b * SAMPLE_RATE)], enrolled, thr, excl)
            pieces.append({"rel_start": a, "rel_end": b, "speaker_id": sid, "speaker_name": name, "score": score})
        return pieces

    def _pieces_auto(self, clip: np.ndarray, seg: Any = None, offset_s: float = 0.0,
                     veto: frozenset[int] = frozenset()) -> list[dict[str, Any]]:
        """Diarize the utterance's audio, identify (cluster + turn level), merge neighbouring turns of the
        same speaker / cluster into pieces. Mostly system-audio turns are never the self speaker, and no turn is
        ever a speaker in `veto` (rejected by the user for the sentence being split)."""
        turns = [t for t in self.engine.diarize(clip) if t.duration >= SPLIT_MIN_TURN_S]
        if not turns:
            raise ValueError("没有分出第二个声音，请手动指定拆分点")
        if veto:
            for t in turns:
                t.exclude = frozenset(getattr(t, "exclude", ()) or ()) | veto
        pairs, block_s = _segment_channels(seg) if seg is not None else (None, None)
        if pairs is not None:
            me = self._self_id()
            for t in turns:
                sh = system_share(pairs, block_s, offset_s + t.start, offset_s + t.end)
                if me is not None and sh is not None and sh >= system_turn_share(self.cfg):
                    t.exclude = frozenset(getattr(t, "exclude", ()) or ()) | {me}
        turns = sorted(self.engine.identify_turns(clip, turns, self.db), key=lambda t: (t.start, t.end))
        unknown = self.cfg.speakers.unknown_label
        for t in turns:
            if t.speaker_id is not None and t.speaker_id in (getattr(t, "exclude", ()) or ()):
                t.speaker_id, t.speaker_name = None, unknown
        pieces: list[dict[str, Any]] = []
        for t in turns:
            key = ("s", t.speaker_id) if t.speaker_id is not None else ("c", t.cluster)
            if pieces and pieces[-1]["_key"] == key:
                pieces[-1]["rel_end"] = max(pieces[-1]["rel_end"], float(t.end))
                continue
            pieces.append({"_key": key, "rel_start": float(t.start), "rel_end": float(t.end), "speaker_id": t.speaker_id,
                           "speaker_name": t.speaker_name if t.speaker_id is not None else None, "score": t.score})
        for pc in pieces:
            pc.pop("_key", None)
        if len(pieces) < 2:
            raise ValueError("没有分出第二个声音，请手动指定拆分点")
        return pieces

    @staticmethod
    def _merge_short_pieces(pieces: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for pc in pieces:
            if out and pc["rel_end"] - pc["rel_start"] < SPLIT_MIN_PIECE_S:
                out[-1]["rel_end"] = pc["rel_end"]
                continue
            out.append(dict(pc))
        if len(out) > 1 and out[0]["rel_end"] - out[0]["rel_start"] < SPLIT_MIN_PIECE_S:
            nxt = out[1]
            nxt["rel_start"] = out[0]["rel_start"]
            out = out[1:]
        return out

    def _plan_split(self, utt_id: int, at: list[float] | None, auto: bool) -> tuple[Any, Any, np.ndarray, list[dict[str, Any]]]:
        u, seg, audio = self._utterance_for_split(utt_id)
        a, b = float(u["start_s"]), float(u["end_s"])
        clip = audio[int(a * SAMPLE_RATE): int(b * SAMPLE_RATE)]
        if len(clip) < int(2 * SPLIT_MIN_PIECE_S * SAMPLE_RATE):
            raise ValueError("这句太短，无法拆分")
        # the speakers the user vetoed for this sentence are never one of its pieces either
        veto = (frozenset(self.db.rejected_speakers([int(u["id"])]).get(int(u["id"]), ()))
                if hasattr(self.db, "rejected_speakers") else frozenset())
        if auto:
            pieces = self._pieces_auto(clip, seg, a, veto)
        elif at:
            pieces = self._pieces_manual(clip, list(at), seg, a, veto)
        else:
            raise ValueError("需要给出拆分点（at）或选择自动拆分（auto）")
        pieces = self._merge_short_pieces(pieces)
        if len(pieces) < 2:
            raise ValueError("拆分点太靠近两端，只剩一段，没有拆分")
        unknown = self.cfg.speakers.unknown_label
        for pc in pieces:
            if pc.get("speaker_id") is not None and int(pc["speaker_id"]) in veto:
                pc["speaker_id"], pc["speaker_name"] = None, None     # a veto always wins
            pc["start_s"] = round(a + float(pc["rel_start"]), 3)
            pc["end_s"] = round(a + float(pc["rel_end"]), 3)
            pc["rel_start"] = round(float(pc["rel_start"]), 3)
            pc["rel_end"] = round(float(pc["rel_end"]), 3)
            pc["speaker"] = pc["speaker_name"] or unknown
            pc["score"] = round(float(pc["score"]), 3) if pc.get("score") is not None else None
        return u, seg, audio, pieces

    def preview_split(self, utt_id: int, at: list[float] | None = None, auto: bool = True) -> dict[str, Any]:
        """The pieces split_utterance would produce (bounds + identification), without ASR or DB writes."""
        u, _seg, _audio, pieces = self._plan_split(utt_id, at, auto)
        return {"utterance_id": int(u["id"]), "segment_id": int(u["segment_id"]), "start_s": float(u["start_s"]),
                "end_s": float(u["end_s"]), "pieces": [{k: pc[k] for k in ("start_s", "end_s", "rel_start", "rel_end",
                                                                             "speaker", "speaker_id", "score")} for pc in pieces]}

    def split_utterance(self, utt_id: int, at: list[float] | None = None, auto: bool = False) -> dict[str, Any]:
        """Split one (mixed) utterance into several: `auto` diarizes + identifies its audio, `at` cuts at the
        given seconds (relative to the utterance start). Every piece is transcribed again (ASR + correction) and
        replaces the original row inside one transaction (tokens / vectors cascade); piece 0 keeps the cluster
        label, the others get fresh labels; cluster_embeddings and the day Markdown are refreshed.
        Returns {"removed", "pieces": [{id, start_s, end_s, speaker, speaker_id, score, text}], "billed_s"}.
        An ASR failure (LLMError) leaves the database untouched."""
        u, seg, audio, pieces = self._plan_split(utt_id, at, auto)
        seg_id = int(u["segment_id"])
        audio_len_s = len(audio) / SAMPLE_RATE
        started_at = _parse_iso(seg["started_at"])
        base_cluster = int(u["cluster_label"]) if u["cluster_label"] is not None else 0
        top = self.db.one("SELECT MAX(cluster_label) AS m FROM utterances WHERE segment_id=?", (seg_id,))
        next_cluster = int(top["m"]) + 1 if top and top["m"] is not None else base_cluster + 1
        chunks: list[Chunk] = []
        for k, pc in enumerate(pieces):
            cl = base_cluster if k == 0 else next_cluster + k - 1
            c = Chunk(float(pc["start_s"]), float(pc["end_s"]), cl, pc["speaker_id"], pc["speaker_name"], pc["score"])
            c.audio_start_s = max(0.0, c.start_s - SPLIT_PAD_S)
            c.audio_end_s = min(audio_len_s, c.end_s + SPLIT_PAD_S)
            chunks.append(c)
        self._register_terms()
        spoken, billed, filtered = transcribe_chunks(self.asr, audio, chunks, _asr_concurrency(self.cfg), stop_event=self.stop_event)
        self.skipped_chunks += filtered
        if len(spoken) != len(chunks):
            # every piece must come back with text, or the sentence would lose part of what was said
            raise ValueError("有一段没有转写出文字（或被内容过滤），没有修改这句")
        corrected, _edits = self._correct(spoken)
        unknown = self.cfg.speakers.unknown_label
        pairs, block_s = _segment_channels(seg)
        rows: list[dict[str, Any]] = []
        days: set[str] = {str(u["day"])}
        for i, (c, raw) in enumerate(spoken):
            text = corrected[i]
            abs_start = started_at + timedelta(seconds=c.start_s)
            day = abs_start.strftime("%Y-%m-%d")
            days.add(day)
            rows.append({"segment_id": seg_id, "start_s": c.start_s, "end_s": c.end_s,
                         "abs_start": abs_start.isoformat(timespec="milliseconds"), "day": day,
                         "cluster_label": c.cluster, "speaker_id": c.speaker_id, "speaker_name": c.speaker_name or unknown,
                         "speaker_score": c.score, "text_raw": raw, "text": text, "text_seg": tokenize(text), "speaker_source": None,
                         "loudness_db": speech_dbfs(_clip(audio, c.start_s, c.end_s)),
                         "system_share": system_share(pairs, block_s, c.start_s, c.end_s)})
        ids: list[int] = []
        with self.db.transaction() as con:
            # the original sentence's vetoes go to every piece (read before the DELETE: they cascade with the row)
            vetoed = [int(r[0]) for r in con.execute(
                "SELECT speaker_id FROM utterance_rejections WHERE utterance_id=?", (int(utt_id),)).fetchall()]
            # the highest id so far (the row being replaced may be it: its id must not come back for a piece)
            top_id = int(con.execute("SELECT COALESCE(MAX(id),0) FROM utterances").fetchone()[0])
            # the row must still be the one we planned on (ASR took seconds: another split / relabel may have replaced it)
            cur = con.execute("DELETE FROM utterances WHERE id=? AND start_s=? AND end_s=?",
                              (int(utt_id), float(u["start_s"]), float(u["end_s"])))
            if int(cur.rowcount or 0) != 1:
                raise ValueError("这句已经被修改或拆分过了，请刷新")
            for r in rows:
                # new ids above every id ever used, so the page never confuses a piece with the row it replaced
                cur = con.execute(
                    "INSERT INTO utterances(id,segment_id,start_s,end_s,abs_start,day,cluster_label,speaker_id,speaker_name,speaker_score,"
                    "text_raw,text,speaker_source,loudness_db,system_share)"
                    " VALUES ((SELECT MAX(COALESCE(MAX(id),0), ?)+1 FROM utterances),?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (top_id, r["segment_id"], r["start_s"], r["end_s"], r["abs_start"], r["day"], r["cluster_label"], r["speaker_id"],
                     r["speaker_name"], r["speaker_score"], r["text_raw"], r["text"], r["speaker_source"], r["loudness_db"],
                     r["system_share"]))
                uid = int(cur.lastrowid)
                ids.append(uid)
                for vs in vetoed:
                    con.execute("INSERT OR IGNORE INTO utterance_rejections(utterance_id, speaker_id) VALUES (?,?)", (uid, vs))
                if r["text_seg"]:
                    con.execute("INSERT INTO utterance_tokens(utterance_id, text_seg) VALUES (?,?)", (uid, r["text_seg"]))
        self._embed_new(ids)
        cluster_embedding = getattr(self.engine, "cluster_embedding", None)
        if callable(cluster_embedding):
            for cl in sorted({c.cluster for c, _ in spoken}):
                ts = [Turn(float(r["start_s"]), float(r["end_s"]), cl) for r in self.db.query(
                    "SELECT start_s, end_s FROM utterances WHERE segment_id=? AND cluster_label=? ORDER BY start_s, id", (seg_id, cl))]
                try:
                    emb = cluster_embedding(audio, ts)
                    if emb is not None:
                        self.db.upsert_cluster_embedding(seg_id, cl, emb, sum(t.duration for t in ts))
                except Exception as e:  # noqa: BLE001 - the rows are in; the voice grouping just lacks this cluster
                    log.warning("split: cluster embedding for #%d/%d failed: %s", seg_id, cl, e)
        for day in sorted(days):
            try:
                write_day_markdown(self.db, self.cfg, day)
            except Exception as e:  # noqa: BLE001 - the DB is the source of truth
                log.warning("cannot write markdown for %s: %s", day, e)
        out_pieces = [{"id": uid, "start_s": round(c.start_s, 3), "end_s": round(c.end_s, 3), "speaker": c.speaker_name or unknown,
                       "speaker_id": c.speaker_id, "score": round(float(c.score), 3) if c.score is not None else None, "text": corrected[i]}
                      for i, (uid, (c, _raw)) in enumerate(zip(ids, spoken))]
        log.info("split utterance #%d (segment #%d) into %d pieces (%.1fs billed)", int(utt_id), seg_id, len(ids), billed)
        return {"removed": int(utt_id), "pieces": out_pieces, "billed_s": round(float(billed), 1)}

    def relabel(self, day: str | None = None, segment_id: int | None = None) -> dict[str, int]:
        """relabel_segment over every 'done' segment whose audio still exists (optionally one day / one id)."""
        stats = {"segments": 0, "changed": 0, "skipped": 0}
        if segment_id is not None:
            rows = [self.db.get_segment(int(segment_id))]
        elif day:
            rows = self.db.query("SELECT * FROM segments WHERE status='done' AND started_at >= ? AND started_at < ? ORDER BY started_at",
                                 (f"{day}T00:00:00", f"{day}T23:59:59.999"))
        else:
            rows = self.db.query("SELECT * FROM segments WHERE status='done' ORDER BY started_at")
        for row in rows:
            if row is None:
                continue
            if self.stop_event is not None and self.stop_event.is_set():
                break
            if not Path(row["path"]).exists():
                stats["skipped"] += 1
                continue
            try:
                stats["changed"] += self.relabel_segment(int(row["id"]))
                stats["segments"] += 1
            except Exception as e:  # noqa: BLE001
                stats["skipped"] += 1
                log.warning("relabel segment #%d failed: %s", row["id"], e)
        log.info("relabel: %s", stats)
        return stats

    def _correct(self, spoken: list[tuple[Chunk, str]]) -> tuple[list[str], str | None]:
        """Returns (corrected texts aligned with `spoken`, edits JSON or None)."""
        texts = [raw for _, raw in spoken]
        if not spoken:
            return texts, None
        corrector = self.corrector
        if corrector is None:
            return texts, None
        unknown = self.cfg.speakers.unknown_label
        items = [{"i": i, "speaker": c.speaker_name or unknown, "text": raw} for i, (c, raw) in enumerate(spoken)]
        try:
            result = corrector.correct(items, load_glossary(self.cfg, self.db))
        except LLMError as e:
            log.warning("correction failed, keeping raw text: %s", e)
            return texts, None
        except Exception as e:  # noqa: BLE001 - never block the pipeline on the corrector
            log.warning("correction crashed (%s), keeping raw text", e)
            return texts, None
        out = list(texts)
        all_edits: list[dict[str, Any]] = []
        try:
            by_i = _index_correction(result)
            for i in range(len(out)):
                r = by_i.get(i)
                if not r:
                    continue
                t = r.get("text")
                if isinstance(t, str) and t.strip():
                    out[i] = t.strip()
                for e in r.get("edits") or []:
                    if isinstance(e, dict):
                        all_edits.append({"i": i, **e})
        except Exception as e:  # noqa: BLE001
            log.warning("unusable correction result (%s), keeping raw text", e)
            return texts, None
        return out, (json.dumps(all_edits, ensure_ascii=False) if all_edits else None)


def _row_get(r: Any, key: str) -> Any:
    try:
        return r[key]
    except (IndexError, KeyError):
        return None


def _accepts_segment_id(fn: Any) -> bool:
    """True when `fn(audio, turns, db, segment_id=...)` is a valid call (keyword or **kwargs)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if "segment_id" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _index_correction(result: Any) -> dict[int, dict[str, Any]]:
    """Accept [{"i":0,"text":..,"edits":[..]}, ...] or {0: {...}} / {"0": {...}}."""
    by_i: dict[int, dict[str, Any]] = {}
    if isinstance(result, dict):
        items = result.get("items") if "items" in result else list(result.items())
        if isinstance(items, list) and items and isinstance(items[0], tuple):
            for k, v in items:
                if isinstance(v, dict):
                    by_i[int(k)] = v
            return by_i
        result = items
    for pos, r in enumerate(result or []):
        if isinstance(r, dict):
            by_i[int(r.get("i", pos))] = r
    return by_i


# --------------------------------------------------------------------------- one-off helper
def _asr_concurrency(cfg: Config) -> int:
    try:
        return max(1, int(cfg.mimo.get("asr_concurrency", 4)))
    except (TypeError, ValueError):
        return 4


def transcribe_chunks(asr: Any, audio: np.ndarray, chunks: list[Chunk], concurrency: int = 4,
                      stop_event: Any = None) -> tuple[list[tuple[Chunk, str]], float, int]:
    """Run ASR over chunks with a small thread pool; results keep chunk order.

    Returns (spoken=[(chunk, text)], billed_seconds, filtered_count). ContentFiltered chunks are skipped;
    any other exception is re-raised (the segment is retried later). Honors stop_event between submissions.
    """
    from concurrent.futures import ThreadPoolExecutor

    if not chunks:
        return [], 0.0, 0
    workers = max(1, min(int(concurrency), len(chunks)))

    def one(c: Chunk) -> tuple[Any, Exception | None]:
        if stop_event is not None and stop_event.is_set():
            return None, RuntimeError("stopped")
        try:
            return asr.transcribe(c.slice(audio)), None
        except ContentFiltered as e:
            return None, e

    if workers == 1:
        results = [one(c) for c in chunks]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="asr") as ex:
            results = list(ex.map(one, chunks))

    spoken: list[tuple[Chunk, str]] = []
    billed = 0.0
    filtered = 0
    for c, (res, err) in zip(chunks, results):
        if err is not None:
            if isinstance(err, ContentFiltered):
                filtered += 1
                log.warning("chunk %.1f-%.1fs skipped: %s", c.start_s, c.end_s, err)
                continue
            raise err
        billed += float(getattr(res, "seconds", 0.0) or 0.0)
        text = (getattr(res, "text", "") or "").strip()
        if text:
            spoken.append((c, text))
    return spoken, billed, filtered


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
    chunks = build_chunks(turns, audio_len_s, float(cfg.mimo.asr_max_chunk_s)) if turns else []
    spoken, _billed, _filtered = transcribe_chunks(asr, audio, chunks, _asr_concurrency(cfg))

    texts = [raw for _, raw in spoken]
    if correct and spoken and bool(cfg.correction.enabled):
        if corrector is None:
            try:
                from .correct import Corrector
                corrector = Corrector(cfg, client)
            except Exception as e:  # noqa: BLE001
                log.warning("correction unavailable: %s", e)
                corrector = None
        if corrector is not None:
            items = [{"i": i, "speaker": c.speaker_name or unknown, "text": raw} for i, (c, raw) in enumerate(spoken)]
            try:
                by_i = _index_correction(corrector.correct(items, load_glossary(cfg, db)))
                for i in range(len(texts)):
                    t = (by_i.get(i) or {}).get("text")
                    if isinstance(t, str) and t.strip():
                        texts[i] = t.strip()
            except Exception as e:  # noqa: BLE001
                log.warning("correction failed, keeping raw text: %s", e)

    return [{
        "start_s": round(c.start_s, 3), "end_s": round(c.end_s, 3), "cluster": c.cluster,
        "speaker": c.speaker_name or unknown, "speaker_id": c.speaker_id, "score": c.score,
        "text_raw": raw, "text": texts[i],
    } for i, (c, raw) in enumerate(spoken)]


__all__ = ["Chunk", "Processor", "build_chunks", "drop_faint_turns", "write_day_markdown", "segment_heading", "transcribe_file",
           "tokenize", "load_glossary", "system_share", "channel_pairs", "system_turn_share",
           "SPLIT_MIN_PIECE_S", "SPLIT_EDGE_S", "SPLIT_MIN_TURN_S"]
