"""Voice stage of the pipeline: segment audio -> identified diarization turns.

`VoiceStage` is a mixin for `processor.Processor`. Everything between reading a segment's audio and build_chunks
sits behind ONE method:

    _voice(seg, audio) -> list[Turn]        # [] = nothing worth transcribing (the segment becomes 'no_speech')

Today that is: diarize -> drop the faint (far-away) turns -> mark the mostly-system-audio turns -> identify ->
enforce "system audio is never the self speaker". The internals may change freely (pre-gate, fast path) as long
as the contract below holds.

Speakers v5 (SPEC_v1 5.1): WHO spoke is decided by scoring.py. The engine leaves a scoring.Decision and the voice
vectors it computed on every turn; this stage keeps them in `self.scratch["voice"]`, writes "maybe X" and the
decision details with the row (`_voice_row`: speaker_guess_id / speaker_guess_score / speaker_meta), and once the
rows are stored (`_voice_after_store`) writes one voice vector per sentence (utterance_voice, reusing the engine's
vectors where a sentence is exactly one embedded turn or one whole cluster) and lets scoring.rescore look at the
new sentences together with their neighbours in time (tracks). Engines that leave no Decision (the fakes of the
tests, older implementations) get none of this: their answers are stored as they are.

Contract with the rest of the pipeline
--------------------------------------
- `_voice` returns Turn objects with start / end / cluster and, when identified, speaker_id / speaker_name / score.
  Turns of one cluster that are close in time are merged into one ASR chunk (= one utterance) by build_chunks,
  which takes the speaker of the first turn: give turns that must not be merged different cluster ids.
- `_voice` stores segments.faint_s (seconds dropped by the loudness gate; 0.0 when none) for its segment.
- `_voice` may raise chunking.Stopped("stopped") when `self.stop_event` is set: the segment goes back to 'new'.
- relabel.py and split.py identify through `_identify`, `_match_clip` and `_system_exclude` (and call
  `self.engine.diarize` for an auto split), so a better identification here reaches them too.
- `_voice_row(row, chunk)` may add columns to an utterances row before it is stored (any key that is a column of
  the table is written; dict / list values as JSON).
- `_voice_after_store(seg_id, utterance_ids, days)` runs after the utterances of a segment (or the pieces of a
  split) were committed; `self.store_ctx` (processor.StoreContext) then holds the audio, the chunks and the rows.
- `self.scratch` is a dict that lives for one segment / one split (emptied before the stages run and after the
  hooks): what `_voice` leaves there (say, a decision per cluster) is still there in the two hooks.

The host class provides: cfg, db, stop_event, scratch, store_ctx, _lock, _engine (None until first use, or the
injected engine).
Names that were globals of the old pipeline.py are reached through `F` (see chunking.py).
"""
from __future__ import annotations

import inspect
import time
from typing import Any, Sequence

import numpy as np

from ..audio.vad import SAMPLE_RATE
from .chunking import F, log, stage_timer
from .speakers import Turn

VOICE_MIN_S = 0.5     # a sentence with less audio than this gets no voice vector of its own
VOICE_MAX_S = 60.0    # at most this much audio goes into one sentence vector


def drop_faint_turns(audio: np.ndarray, turns: Sequence[Turn], threshold: float | None) -> tuple[list[Turn], int, float]:
    """Remove the diarization turns whose speech level (audio.loudness.speech_dbfs) is below `threshold` dBFS:
    far-away / unclear voices are neither transcribed nor used for speaker identification. Returns
    (kept turns in order, dropped count, dropped seconds); threshold None keeps everything."""
    if threshold is None:
        return list(turns), 0, 0.0
    kept: list[Turn] = []
    dropped, faint_s = 0, 0.0
    for t in turns:
        if F.speech_dbfs(F._clip(audio, t.start, t.end)) < float(threshold):
            dropped += 1
            faint_s += max(0.0, t.duration)
        else:
            kept.append(t)
    return kept, dropped, faint_s


def _accepts_segment_id(fn: Any) -> bool:
    """True when `fn(audio, turns, db, segment_id=...)` is a valid call (keyword or **kwargs)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if "segment_id" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


class VoiceStage:
    """Mixin: who spoke when (see the module docstring for the contract)."""

    # --- lazy component ------------------------------------------------------
    @property
    def engine(self) -> Any:
        with self._lock:
            if self._engine is None:
                from .speakers import SpeakerEngine
                self._engine = SpeakerEngine(self.cfg)
            return self._engine

    # --- the stage -----------------------------------------------------------
    def _voice(self, seg: Any, audio: np.ndarray) -> list[Turn]:
        """Diarize `audio` (the whole segment `seg`), drop the faint turns, identify the rest."""
        seg_id = int(seg["id"])
        audio_s = len(audio) / SAMPLE_RATE
        with stage_timer(self.db, "diarize", audio_s):
            turns: list[Turn] = list(self.engine.diarize(audio))
        with stage_timer(self.db, "gate", audio_s):
            min_db = F.min_speech_dbfs(self.cfg)
            turns, n_faint, faint_s = F.drop_faint_turns(audio, turns, min_db)
            if n_faint:
                log.info("segment #%d: dropped %d faint turns (%.1f s, < %g dBFS)", seg_id, n_faint, faint_s, min_db)
            self.db.execute("UPDATE segments SET faint_s=? WHERE id=?", (float(faint_s), seg_id))
        if not turns:
            return []
        with stage_timer(self.db, "identify", audio_s):
            pairs, block_s = F._segment_channels(seg)
            self._mark_system_turns(turns, pairs, block_s)
            turns = self._identify(audio, turns, seg_id)
            turns = self._enforce_system_turns(turns, pairs, block_s)
        # what the hooks need later: the turns as decided, and whether the scoring layer decided them
        self.scratch["voice"] = {"turns": list(turns),
                                 "scored": any(getattr(t, "decision", None) is not None for t in turns)}
        return turns

    def _chunk_turns(self, chunk: Any) -> tuple[list[Turn], list[Turn]]:
        """(the turns that lie inside `chunk`, all turns of its cluster) from what `_voice` kept."""
        kept = (self.scratch.get("voice") or {}).get("turns") or []
        mates = [t for t in kept if t.cluster == chunk.cluster]
        inside = [t for t in mates if t.start >= float(chunk.start_s) - 1e-3 and t.end <= float(chunk.end_s) + 1e-3]
        return sorted(inside, key=lambda t: (t.start, t.end)), mates

    def _voice_row(self, row: dict[str, Any], chunk: Any) -> None:
        """Row hook of the voice stage: the decision details of the sentence (speaker_meta) and, for a sentence
        nobody is assigned to, who it may be (speaker_guess_id / speaker_guess_score: shown as "maybe X", asked
        in the review queue, never written to speaker_id). Only when the scoring layer decided the turns."""
        inside, _mates = self._chunk_turns(chunk)
        if not inside:
            return
        t = inside[0]                                    # build_chunks gives a chunk the speaker of its first turn
        d = getattr(t, "decision", None)
        if d is None:
            return
        if row.get("speaker_id") != d.speaker_id:
            # the stage overruled the engine afterwards (system audio is never the self speaker)
            row["speaker_meta"] = {"m": "veto"}
            return
        row["speaker_meta"] = d.meta()
        excl = frozenset(getattr(t, "exclude", ()) or ())
        if d.speaker_id is None and d.guess_id is not None and d.guess_id not in excl:
            row["speaker_guess_id"] = int(d.guess_id)
            row["speaker_guess_score"] = round(float(d.guess_score), 4) if d.guess_score is not None else None

    def _chunk_vector(self, audio: np.ndarray, chunk: Any, embed: Any) -> tuple[np.ndarray | None, float, bool]:
        """(voice vector of the sentence, seconds of speech behind it, True when it had to be embedded now).
        A sentence that is exactly one turn the engine embedded, or one whole cluster, reuses that vector."""
        inside, mates = self._chunk_turns(chunk)
        if len(inside) == 1 and getattr(inside[0], "emb", None) is not None:
            return inside[0].emb, float(inside[0].duration), False
        if inside and len(inside) == len(mates) and getattr(inside[0], "cluster_emb", None) is not None:
            return inside[0].cluster_emb, float(sum(t.duration for t in inside)), False
        parts = [F._clip(audio, t.start, t.end) for t in inside] or [F._clip(audio, chunk.start_s, chunk.end_s)]
        clip = np.concatenate(parts) if len(parts) > 1 else parts[0]
        if len(clip) < int(VOICE_MIN_S * SAMPLE_RATE):
            return None, 0.0, False
        return embed(clip[: int(VOICE_MAX_S * SAMPLE_RATE)]), len(clip) / SAMPLE_RATE, True

    def _voice_after_store(self, seg_id: int, utterance_ids: list[int], days: list[str]) -> None:
        """After-store hook of the voice stage: one voice vector per stored sentence (utterance_voice), then the
        scoring layer looks at the new sentences together with their neighbours in time (scoring.rescore on this
        segment: short sentences follow their track, a "maybe" may become sure). Runs for a segment and for the
        pieces of a split, and only with an engine that decides through the scoring layer."""
        ctx = self.store_ctx
        engine = self._engine
        embed = getattr(engine, "embed", None)
        if ctx is None or not utterance_ids or not callable(embed):
            return
        scored = bool((self.scratch.get("voice") or {}).get("scored"))
        if ctx.origin == "split":
            scored = callable(getattr(engine, "score", None))
        if not scored:
            return
        from . import scoring

        model = scoring.model_id(self.cfg)
        rows: list[tuple[int, str, int, bytes, float]] = []
        extra_n, extra_s, wall, cpu = 0, 0.0, 0.0, 0.0
        for uid, (chunk, _raw) in zip(utterance_ids, ctx.spoken):
            t0, c0 = time.perf_counter(), time.process_time()
            vec, speech_s, fresh = self._chunk_vector(ctx.audio, chunk, embed)
            if fresh:
                # an embedding that could not be reused: what per-sentence vectors cost on top of identification
                extra_n += 1
                extra_s += float(speech_s)
                wall += time.perf_counter() - t0
                cpu += time.process_time() - c0
            if vec is None:
                continue
            v = np.asarray(vec, dtype=np.float32).ravel()
            norm = float(np.linalg.norm(v))
            if v.shape[0] < 2 or not np.isfinite(norm) or norm <= 0:
                continue
            rows.append((int(uid), model, int(v.shape[0]), (v / norm).astype(np.float32).tobytes(), round(float(speech_s), 3)))
        if extra_n:
            try:
                self.db.add_usage_local("embed", n=extra_n, audio_s=extra_s, wall_s=wall, cpu_s=cpu)
            except Exception as e:  # noqa: BLE001 - accounting only
                log.debug("usage of voice vectors not recorded: %s", e)
        if rows:
            with self.db.transaction() as con:
                con.executemany("INSERT OR REPLACE INTO utterance_voice(utterance_id, model, dim, embedding, speech_s)"
                                " VALUES (?,?,?,?,?)", rows)
        with stage_timer(self.db, "identify", 0.0, n=0):
            res = scoring.rescore(self.db, self.cfg, segment_ids=[int(seg_id)])
        if res.get("changed"):
            log.info("segment #%d: %d sentence(s) re-decided with their neighbours (%s)", seg_id, res["changed"], res["moves"])

    # --- helpers (also used by relabel.py / split.py) ------------------------
    def _self_id(self) -> int | None:
        try:
            me = self.db.self_speaker()
        except Exception:  # noqa: BLE001 - DB stand-ins without the method
            return None
        return int(me["id"]) if me is not None else None

    def _mark_system_turns(self, turns: list[Turn], pairs: np.ndarray | None, block_s: float | None) -> int:
        """Add the self speaker to Turn.exclude for every turn that is mostly system audio (returns how many), and
        note every turn's system-audio share on it."""
        if pairs is None:
            return 0
        me = self._self_id()
        if me is None:
            for t in turns:
                try:
                    t.system_share = F.system_share(pairs, block_s, t.start, t.end)
                except AttributeError:
                    pass
            return 0
        thr = F.system_turn_share(self.cfg)
        n = 0
        for t in turns:
            sh = F.system_share(pairs, block_s, t.start, t.end)
            try:
                t.system_share = sh                       # the scoring layer picks the condition centroid by it
            except AttributeError:                        # turn stand-ins with __slots__
                pass
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
        thr = F.system_turn_share(self.cfg)
        unknown = self.cfg.speakers.unknown_label
        next_id = max(int(t.cluster) for t in turns) + 1
        new_ids: dict[int, int] = {}
        n = 0
        for t in turns:
            if t.speaker_id != me:
                continue
            sh = F.system_share(pairs, block_s, t.start, t.end)
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
        not take `segment_id` (fakes in tests, older implementations) are called the old way, and so is every
        engine when `seg_id` is None (a clip that is not a whole segment: nothing is stored for it).
        `per_utterance` (relabel) prefers engine.identify_utterances: every utterance long enough is matched on
        its own audio and keeps its own score, the short ones follow their cluster."""
        fn = getattr(self.engine, "identify_utterances", None) if per_utterance else None
        if not callable(fn):
            fn = self.engine.identify_turns
        if seg_id is not None and F._accepts_segment_id(fn):
            return list(fn(audio, turns, self.db, segment_id=seg_id))
        return list(fn(audio, turns, self.db))

    def _match_clip(self, clip: np.ndarray, enrolled: list[tuple[int, str, np.ndarray]], thr: float,
                    exclude: frozenset[int] = frozenset()) -> tuple[int | None, str | None, float | None]:
        """Identify one piece on its own audio (at least SPLIT_MIN_TURN_S of it); unknown otherwise.
        An engine with `score` (SpeakerEngine) decides through the scoring layer - the same rules as the first
        pass; `enrolled` / `thr` then only say whether anybody is enrolled. Other engines (fakes) go through
        embed + match as before: negative samples and `exclude` are passed on only when there are any."""
        if not enrolled or len(clip) < int(F.SPLIT_MIN_TURN_S * SAMPLE_RATE):
            return None, None, None
        embed = getattr(self.engine, "embed", None)
        match = getattr(self.engine, "match", None)
        score = getattr(self.engine, "score", None)
        if callable(embed) and callable(score):
            d = score(self.db, embed(clip), speech_s=len(clip) / SAMPLE_RATE, loudness_db=F.speech_dbfs(clip),
                      exclude=frozenset(exclude or ()))
            if d.speaker_id is not None and d.speaker_id in exclude:
                return None, None, d.score
            return d.speaker_id, d.name, d.score
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
        pairs, block_s = F._segment_channels(seg)
        if pairs is None:
            return frozenset()
        me = self._self_id()
        sh = F.system_share(pairs, block_s, start_s, end_s)
        if me is None or sh is None or sh < F.system_turn_share(self.cfg):
            return frozenset()
        return frozenset({me})


__all__ = ["VOICE_MAX_S", "VOICE_MIN_S", "VoiceStage", "drop_faint_turns"]
