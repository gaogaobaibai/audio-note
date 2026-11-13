"""Text stage of the pipeline: ASR chunks -> raw text -> corrected text.

`TextStage` is a mixin for `processor.Processor`. The rest of the pipeline only uses two methods:

    _transcribe(seg, audio, chunks) -> (spoken, billed_seconds, filtered_count)
        spoken = [(chunk, raw text)] in chunk order; chunks that came back empty or were rejected by the content
        filter are left out. One entry = one utterance row.
    _correct(spoken) -> (texts, edits_json)
        texts[i] is the text stored for spoken[i] (the raw text when nothing changed or correction is unavailable);
        edits_json is a JSON list [{"i", "from", "to", "reason", ...}] of the applied edits (i indexes `spoken`)
        or None. It never raises: a failing corrector keeps the raw text.

Contract with the rest of the pipeline
--------------------------------------
- `_transcribe` raises chunking.Stopped("stopped") when `self.stop_event` was set before a chunk was sent (the
  segment goes back to 'new'), and lets LLMError through when the API gave up (the segment is 'failed' and the
  scheduler retries transient failures). It adds the content-filtered chunks to `self.skipped_chunks`.
- All three correction call sites (process_segment, split_utterance, transcribe_file) go through
  `correct_spoken(corrector, spoken, cfg, db)`; `corrector.correct(items, glossary)` is the seam the tests fake.
- Which corrector: `TextStage.corrector` follows correction.mode (correct.py): session = the lexicon's
  deterministic pass, no LLM and no API key needed; segment (the default until speech.refine ships) = lexicon
  pass + the legacy LLM corrector; off /
  correction.enabled false = None. An injected corrector always wins.
- `_text_row(row, chunk, raw)` adds columns to an utterances row before it is stored: text_source ('lexicon' |
  'llm': who made the last change) and edited_at for a sentence whose text is not the raw ASR text.
- `_text_after_store(seg_id, utterance_ids, days)` runs after the utterances of a segment (or the pieces of a
  split) were committed; `self.store_ctx` (processor.StoreContext) then holds the applied edits next to the ids.
  It writes one row of utterance_revisions per applied edit (raw text -> ... -> stored text, see
  `revision_steps`), counts the lexicon hits, and after a split moves the open questions of the replaced
  sentence to the piece that now contains the word.
- `self.scratch` is a dict that lives for one segment / one split (emptied before the stages run and after the
  hooks): what `_transcribe` / `_correct` leave there is still there in the two hooks. This stage does not use
  it: what `_correct` learned about who changed which sentence is kept in `self._text_ctx` until the next call.
- Glossary (the word list handed to `corrector.correct`): `load_glossary(cfg, db)` = glossary.txt
  (glossary_file.read_glossary) + enrolled speaker names + what the user taught the lexicon, minus entries the
  user switched off.

The host class provides: cfg, db, client (property), stop_event, skipped_chunks, scratch, store_ctx, _lock, _engine,
_asr, _corrector, _corrector_tried. Names that were globals of the old pipeline.py are reached through `F` (see chunking.py).
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..config import Config
from ..db import DB, now_iso
from ..llm.client import ContentFiltered, LLMError
from . import glossary_file
from .chunking import Chunk, F, Stopped, log


# --------------------------------------------------------------------------- ASR
def _asr_concurrency(cfg: Config) -> int:
    try:
        return max(1, int(cfg.mimo.get("asr_concurrency", 4)))
    except (TypeError, ValueError):
        return 4


def transcribe_chunks(asr: Any, audio: np.ndarray, chunks: list[Chunk], concurrency: int = 4,
                      stop_event: Any = None) -> tuple[list[tuple[Chunk, str]], float, int]:
    """Run ASR over chunks with a small thread pool; results keep chunk order.

    Returns (spoken=[(chunk, text)], billed_seconds, filtered_count). ContentFiltered chunks are skipped;
    any other exception is re-raised (the segment is retried later). Honors stop_event between submissions:
    a chunk that was not sent yet raises Stopped("stopped"), which is not a failure of the segment.
    """
    from concurrent.futures import ThreadPoolExecutor

    if not chunks:
        return [], 0.0, 0
    workers = max(1, min(int(concurrency), len(chunks)))

    def one(c: Chunk) -> tuple[Any, Exception | None]:
        if stop_event is not None and stop_event.is_set():
            return None, Stopped("stopped")
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


# --------------------------------------------------------------------------- glossary
def load_glossary(cfg: Config, db: DB | None = None) -> list[str]:
    """The word list for `corrector.correct(items, glossary)`: glossary.txt terms (one per line; '#' lines and
    inline ' #' comments dropped, see glossary_file), all enrolled speaker names, then the entries the user
    taught the lexicon (lexicon page, answers, hand edits) - without duplicates and without entries the user
    switched off."""
    terms: list[str] = []
    try:
        terms.extend(glossary_file.read_glossary(cfg.glossary_path))
    except Exception as e:  # noqa: BLE001
        log.warning("cannot read glossary: %s", e)
    if db is not None:
        try:
            terms.extend(r["name"] for r in db.query("SELECT name FROM speakers ORDER BY id"))
        except Exception as e:  # noqa: BLE001
            log.warning("cannot list speakers for glossary: %s", e)
        try:
            from . import lexicon

            lex = lexicon.Lexicon.load(db)
            off = {t.term for t in lex.terms if t.status == "disabled"}
            terms = [t for t in terms if t not in off]
            terms.extend(t.term for t in lex.active if t.source in ("user", "answer", "edit"))
        except Exception as e:  # noqa: BLE001 - the lexicon is an addition here, never a reason to lose the glossary
            log.debug("lexicon terms not added to the glossary: %s", e)
    seen: set[str] = set()
    return [t for t in terms if not (t in seen or seen.add(t))]


# --------------------------------------------------------------------------- correction
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


def correct_spoken(corrector: Any, spoken: list[tuple[Chunk, str]], cfg: Config,
                   db: DB | None = None) -> tuple[list[str], str | None]:
    """The one correction entry point (process_segment, split_utterance and transcribe_file all end up here).

    Returns (texts aligned with `spoken`, edits JSON or None). `corrector` is anything with
    `correct(items, glossary)` (items = [{"i", "speaker", "text"}]) or None (correction off / unavailable).
    Never raises: whatever goes wrong, the raw texts are kept."""
    texts = [raw for _, raw in spoken]
    if not spoken or corrector is None:
        return texts, None
    unknown = cfg.speakers.unknown_label
    items = [{"i": i, "speaker": c.speaker_name or unknown, "text": raw} for i, (c, raw) in enumerate(spoken)]
    try:
        result = corrector.correct(items, F.load_glossary(cfg, db))
    except LLMError as e:
        log.warning("correction failed, keeping raw text: %s", e)
        return texts, None
    except Exception as e:  # noqa: BLE001 - never block the pipeline on the corrector
        log.warning("correction crashed (%s), keeping raw text", e)
        return texts, None
    out = list(texts)
    all_edits: list[dict[str, Any]] = []
    try:
        by_i = F._index_correction(result)
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


# --------------------------------------------------------------------------- revisions of the first pass
def revision_steps(raw: str, final: str, edits: list[dict[str, Any]], default_source: str = "llm") -> list[dict[str, Any]]:
    """The path from the raw ASR text to the stored text as one step per applied edit:
    [{"old", "new", "source", "detail"}], old of the first step = raw, new of the last = final.

    Each edit {"from", "to", "source"?, "pos"?, ...} is replayed on the running text: at the first occurrence
    of `from` at or after the place the previous edit ended (edits come left to right), else at its first
    occurrence anywhere. An edit that cannot be replayed is skipped, and whatever difference is left at the end
    becomes one closing step without edit details - so the chain always ends at the stored text, also for a
    corrector that changed text without saying how."""
    steps: list[dict[str, Any]] = []
    cur = raw
    floor = 0
    for e in edits:
        frm, to = e.get("from"), e.get("to")
        if not isinstance(frm, str) or not isinstance(to, str) or not frm or frm == to:
            continue
        at = cur.find(frm, floor)
        if at < 0:
            at = cur.find(frm)
        if at < 0:
            continue
        nxt = cur[:at] + to + cur[at + len(frm):]
        detail = {k: v for k, v in e.items() if k not in ("i", "source", "pos")}
        detail["start"], detail["end"] = at, at + len(to)
        steps.append({"old": cur, "new": nxt, "source": str(e.get("source") or default_source), "detail": {"edits": [detail]}})
        cur = nxt
        floor = at + len(to)
    if cur != final:
        steps.append({"old": cur, "new": final, "source": steps[-1]["source"] if steps else default_source, "detail": {"edits": []}})
    return steps


class TextStage:
    """Mixin: what was said (see the module docstring for the contract)."""

    # --- lazy components -----------------------------------------------------
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
        """The first-pass corrector for correction.mode (see correct.py), or None (mode off, correction
        disabled, module unavailable). session mode needs neither an API key nor the network."""
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
                from .correct import default_corrector
            except Exception as e:  # noqa: BLE001 - module written in parallel / missing
                log.warning("speech.correct unavailable (%s); skipping correction", e)
                return None
        try:
            # the client is only built for segment mode (default_corrector calls the factory then)
            corrector = default_corrector(self.cfg, self.db, lambda: self.client)
        except Exception as e:  # noqa: BLE001
            log.warning("cannot build the corrector (%s); skipping correction", e)
            corrector = None
        with self._lock:
            self._corrector = corrector
        return self._corrector

    # --- the stage -----------------------------------------------------------
    def _transcribe(self, seg: Any, audio: np.ndarray,
                    chunks: list[Chunk]) -> tuple[list[tuple[Chunk, str]], float, int]:
        """ASR for `chunks` of the segment `seg` (`audio` is the whole segment; ASR API latency is ~0.8x realtime
        per request whatever the length, hence the small thread pool)."""
        spoken, billed, filtered = F.transcribe_chunks(self.asr, audio, chunks, F._asr_concurrency(self.cfg),
                                                       stop_event=self.stop_event)
        self.skipped_chunks += filtered
        return spoken, billed, filtered

    def _correct(self, spoken: list[tuple[Chunk, str]]) -> tuple[list[str], str | None]:
        """Returns (corrected texts aligned with `spoken`, edits JSON or None)."""
        if not spoken:
            return [], None
        corrector = self.corrector
        texts, edits_json = F.correct_spoken(corrector, spoken, self.cfg, self.db)
        # who changed what, for the row hook (text_source) and the after-store hook (revisions): an edit says so
        # itself ("source"), anything else is attributed to the corrector ('lexicon' for the lexicon pass, 'llm'
        # for the legacy corrector and for injected ones)
        default = str(getattr(corrector, "source", None) or "llm")
        last: dict[int, str] = {}
        for e in F.edits_list(edits_json):
            try:
                last[int(e.get("i"))] = str(e.get("source") or default)
            except (TypeError, ValueError):
                continue
        by_chunk = {id(c): last.get(i, default) for i, (c, raw) in enumerate(spoken) if texts[i] != raw}
        # `spoken` is kept so the chunk objects (and with them the id() keys) stay valid until the next call
        self._text_ctx = {"default_source": default, "source_by_chunk": by_chunk, "spoken": spoken}
        return texts, edits_json

    def _text_row(self, row: dict[str, Any], chunk: Chunk, raw: str) -> None:
        """Row hook of the text stage: add to `row` (the utterances row about to be stored for `chunk`, whose ASR
        text was `raw`; row["text"] is the text after correction) the columns this stage knows about. Called for
        a segment's rows and for the pieces of a split. A sentence the first pass changed gets text_source
        (who made the last change) and edited_at; an untouched one keeps both NULL (= ASR as it came)."""
        if row.get("text") == raw:
            return
        mine = getattr(self, "_text_ctx", None) or {}
        row["text_source"] = (mine.get("source_by_chunk") or {}).get(id(chunk)) or "llm"
        row["edited_at"] = now_iso()

    def _text_after_store(self, seg_id: int, utterance_ids: list[int], days: list[str]) -> None:
        """After-store hook of the text stage (see Processor._after_store): the history of what the first pass
        did. For every stored sentence whose text is not the raw ASR text, one utterance_revisions row per
        applied edit (revision_steps), all in one transaction; lexicon entries that were used or said are
        counted (they move up in the hot words); after a split, open questions about the replaced sentence
        follow the word to its new sentence."""
        ctx = self.store_ctx
        if ctx is None:
            return
        from . import revisions

        mine = getattr(self, "_text_ctx", None) or {}
        default = str(mine.get("default_source") or "llm") if mine.get("spoken") is ctx.spoken else "llm"
        by_i: dict[int, list[dict[str, Any]]] = {}
        for e in ctx.edits:
            try:
                by_i.setdefault(int(e.get("i")), []).append(e)
            except (TypeError, ValueError):
                continue
        todo: list[tuple[int, list[dict[str, Any]]]] = []
        for i, uid in enumerate(utterance_ids):
            if i >= len(ctx.spoken) or i >= len(ctx.rows):
                break
            raw, final = ctx.spoken[i][1], ctx.rows[i].get("text") or ""
            if raw != final:
                todo.append((int(uid), revision_steps(raw, final, by_i.get(i, []), default)))
        if todo:
            now = now_iso()
            with self.db.transaction() as con:
                for uid, steps in todo:
                    for st in steps:
                        revisions.record(con, uid, st["old"], st["new"], st["source"], st["detail"], now)
        self._count_lexicon_hits(ctx)
        if ctx.origin == "split" and ctx.replaced_utterance_id is not None:
            self._move_questions(int(ctx.replaced_utterance_id), utterance_ids, [r.get("text") or "" for r in ctx.rows])

    def _count_lexicon_hits(self, ctx: Any) -> None:
        """Entries spelled out in the new sentences and aliases the first pass rewrote: hits for hotwords()."""
        try:
            from . import lexicon

            lex = lexicon.Lexicon.load(self.db)
            if not lex.active:
                return
            lids = [h.term.id for r in ctx.rows for h in lex.find_in(r.get("text") or "") if h.alias is None]
            aids = [int(e["alias_id"]) for e in ctx.edits if isinstance(e.get("alias_id"), int)]
            lexicon.touch(self.db, lexicon_ids=lids, alias_ids=aids)
        except Exception as e:  # noqa: BLE001 - statistics only
            log.debug("lexicon hits not counted: %s", e)

    def _move_questions(self, old_id: int, new_ids: list[int], texts: list[str]) -> None:
        """A split replaced sentence `old_id` by `new_ids`: an open 'term' question that pointed at it now points
        at the first piece that contains its span (so the page can still show and answer it in place)."""
        try:
            rows = self.db.query("SELECT id, payload FROM inbox_items WHERE utterance_id=? AND status IN ('open','queued','snoozed')",
                                 (old_id,))
            for r in rows:
                try:
                    p = json.loads(r["payload"] or "{}")
                except ValueError:
                    continue
                span = str(p.get("span") or "")
                target = next(((uid, t) for uid, t in zip(new_ids, texts) if span and span in t), None)
                if target is None:
                    continue
                p["utterance_id"], p["sentence"] = int(target[0]), target[1]
                p.pop("start", None)
                p.pop("end", None)
                self.db.execute("UPDATE inbox_items SET utterance_id=?, payload=?, updated_at=? WHERE id=?",
                                (int(target[0]), json.dumps(p, ensure_ascii=False), now_iso(), int(r["id"])))
        except Exception as e:  # noqa: BLE001 - a question pointing at a sentence that is gone is harmless
            log.debug("questions of utterance #%d not moved: %s", old_id, e)


__all__ = ["TextStage", "correct_spoken", "load_glossary", "revision_steps", "transcribe_chunks"]
