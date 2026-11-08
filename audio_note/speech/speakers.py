"""Speaker diarization (who spoke when) + enrollment-based identification (which known person).

Models (sherpa-onnx, CPU, no torch):
  segmentation : pyannote segmentation-3.0 (ONNX)
  embedding    : 3D-Speaker CAM++ zh-cn 16k (192-d)

Enrollment: `enroll(name, wav_paths)` extracts one embedding per sample (speech-gated) and stores
them in speaker_embeddings; a 16 kHz copy of the speech-only part of every accepted sample is kept
under data/speakers/<name>/ so voices can be re-embedded later. Identification: cosine similarity
between a cluster's mean embedding and each enrolled speaker's embeddings (max over samples);
accepted if >= identify_threshold.

v3 (mixed sentences): a cluster decision is re-checked turn by turn (`identify_turns`, speakers.turn_check_min_s),
a new enrollment sample must resemble the person's existing samples (`check_sample` / `SampleMismatch`,
speakers.sample_consistency_min, `force=True` overrides) and `audit_samples` flags the outliers already stored.
v3b: the reference voice is anchored on samples the user recorded / uploaded themselves (speaker_embeddings.origin
'mic' / 'file'); without such an anchor two rival factions of samples are reported as `ambiguous` instead of one
being deleted (see _analyse_samples).
v4 (negatives): `match` also takes the negative samples ("this voice is NOT X", speaker_negatives) and a set of
excluded speaker ids: a candidate X wins only when its best positive score p >= threshold and p - (best negative
of X) >= speakers.negative_margin, otherwise the next candidate (by p) is tried. Turns carry `exclude` (speakers
vetoed for that sentence, or the self speaker for system-audio turns) which identify_turns / identify_utterances
honour. audit_samples reports each sample's `influence` (automatic sentences only this sample lets through).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import sherpa_onnx
import soundfile as sf

from ..config import Config
from ..db import DB
from ..audio.storage import read_segment
from ..audio.vad import SAMPLE_RATE, GateConfig, SileroVAD, segment_audio

log = logging.getLogger(__name__)

MIN_ENROLL_SPEECH_S = 1.0   # samples with less speech than this are rejected (enroll and label alike)
MIN_CLUSTER_SPEECH_S = 0.5  # a diarization cluster shorter than this is neither identified nor stored
DEFAULT_SAMPLE_CONSISTENCY_MIN = 0.5
DEFAULT_TURN_CHECK_MIN_S = 1.0
DEFAULT_TURN_REJECT_MARGIN = 0.15
DEFAULT_MIN_LABEL_SAMPLE_S = 2.5   # samples cut out of recordings (label / name a voice) need this much speech
DEFAULT_NEGATIVE_MARGIN = 0.0      # p - n must reach this for a speaker with negative samples
DEFAULT_IDENTIFY_THRESHOLD = 0.45


class SampleMismatch(ValueError):
    """A new sample does not resemble the person's existing samples (best cosine `score` < `threshold`).
    Raised by the enroll functions unless `force=True`; the API answers 400 with "mismatch": true."""

    def __init__(self, name: str, score: float, threshold: float):
        self.name = str(name)
        self.score = float(score)
        self.threshold = float(threshold)
        super().__init__(f"这段声音和 {self.name} 已有的样本不像（相似度 {self.score:.2f} < {self.threshold:.2f}），没有加入；"
                         "确定是同一个人可以用“仍然加入”")


class TooShortSample(ValueError):
    """The clip holds less speech than a voice sample needs (`speech_s` < `min_s`). The label flows catch it
    and still assign the utterance to the person, just without adding a sample."""

    def __init__(self, speech_s: float, min_s: float):
        self.speech_s = float(speech_s)
        self.min_s = float(min_s)
        super().__init__(f"选中的范围只有 {self.speech_s:.1f}s 语音（至少需要 {self.min_s:g}s），没有加入声纹样本")


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def sample_consistency_min(cfg: Config) -> float:
    try:
        return float(cfg.speakers.get("sample_consistency_min", DEFAULT_SAMPLE_CONSISTENCY_MIN))
    except (TypeError, ValueError, AttributeError):
        return DEFAULT_SAMPLE_CONSISTENCY_MIN


def min_label_sample_s(cfg: Config) -> float:
    """Minimum speech (s) for a sample taken out of a recording (transcript label / naming a voice group).
    Short far-field snippets make unreliable embeddings that later attract the wrong voice."""
    try:
        return float(cfg.speakers.get("min_label_sample_s", DEFAULT_MIN_LABEL_SAMPLE_S))
    except (TypeError, ValueError, AttributeError):
        return DEFAULT_MIN_LABEL_SAMPLE_S


def negative_margin(cfg: Config) -> float:
    """speakers.negative_margin: how much more a voice must resemble X's samples than X's negative samples."""
    try:
        return float(cfg.speakers.get("negative_margin", DEFAULT_NEGATIVE_MARGIN))
    except (TypeError, ValueError, AttributeError):
        return DEFAULT_NEGATIVE_MARGIN


def load_negatives(db: Any) -> list[tuple[int, np.ndarray]]:
    """db.speaker_negatives(), tolerant of DB stand-ins without the method (older fakes)."""
    fn = getattr(db, "speaker_negatives", None)
    if not callable(fn):
        return []
    try:
        return list(fn())
    except Exception as e:  # noqa: BLE001 - a missing table on a foreign DB must not stop identification
        log.debug("no negative samples: %s", e)
        return []


def negative_scores(emb: np.ndarray, negatives: list[tuple[int, np.ndarray]] | None) -> dict[int, float]:
    """{speaker_id: best cosine between `emb` and that speaker's negative samples} (same dimension only)."""
    out: dict[int, float] = {}
    if not negatives:
        return out
    e = np.asarray(emb, dtype=np.float32).ravel()
    for sid, n in negatives:
        n = np.asarray(n, dtype=np.float32).ravel()
        if n.shape[0] != e.shape[0]:
            continue
        nn = float(np.linalg.norm(n)) or 1.0
        s = float(np.dot(e, n)) / nn
        if s > out.get(int(sid), -1.0):
            out[int(sid)] = s
    return out


ANCHOR_DURATION_CAP_S = 10.0   # duration weight cap when no trusted (mic / file) sample can anchor the core


@dataclass
class SampleAnalysis:
    """Result of _analyse_samples: the reference voice, which samples form the core and which are ambiguous."""
    ref: np.ndarray | None
    core: frozenset[int] = frozenset()        # indices of the core faction (medoid + its neighbours)
    ambiguous: frozenset[int] = frozenset()   # indices of samples in a two-faction stand-off (core + the rival faction)


def _analyse_samples(embs: list[np.ndarray], durs: list[float], min_consistency: float,
                     origins: list[str | None] | None = None) -> SampleAnalysis:
    """Reference voice of a speaker from their stored samples (v3b anchor rules):

    * The medoid (the sample most similar to all the others, duration-weighted, ties to the longer one) is chosen
      among the samples whose origin is 'mic' or 'file' (the user recorded / uploaded them: they cannot contain
      somebody else). Only when there is none is it chosen among all samples, with the duration weight capped at
      ANCHOR_DURATION_CAP_S so a long mixed clip cannot buy its way in.
    * Core = medoid + every sample at least `min_consistency` similar to it; the reference is their duration-
      weighted mean, normalised. Consistency is always measured against the core, so two stray samples that
      merely resemble each other cannot vouch for one another.
    * Without a trusted anchor, non-core samples that form a faction of their own (>= 2 samples resembling each
      other) make the speaker *ambiguous*: nobody can tell which faction is the real voice, so both factions are
      flagged `ambiguous` and neither is an outlier (the user decides by ear). A lone stranger stays an outlier.
    """
    if not embs:
        return SampleAnalysis(None)
    if len(embs) == 1:
        return SampleAnalysis(embs[0], frozenset({0}))
    n = len(embs)
    origins = list(origins) if origins is not None else [None] * n
    from ..db import TRUSTED_SAMPLE_ORIGINS

    trusted = [i for i in range(n) if origins[i] in TRUSTED_SAMPLE_ORIGINS]
    E = np.stack(embs)
    w = np.asarray([max(float(d or 0.0), 0.1) for d in durs], dtype=np.float64)
    if not trusted:
        w = np.minimum(w, ANCHOR_DURATION_CAP_S)
    S = E @ E.T
    score = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        score[i] = float(np.dot(S[i, others], w[others]) / w[others].sum())
    candidates = trusted or list(range(n))
    # tie-break on duration so a long clean recording beats a short snippet
    medoid = int(max(candidates, key=lambda i: (round(score[i], 6), w[i])))
    thr = float(min_consistency)
    core = [j for j in range(n) if j == medoid or S[medoid, j] >= thr]
    ref = _unit((E[core] * w[core][:, None]).sum(axis=0))
    ambiguous: set[int] = set()
    if not trusted:
        rest = [j for j in range(n) if j not in core]
        faction = {i for i in rest if any(j != i and S[i, j] >= thr for j in rest)}
        if len(faction) >= 2:
            ambiguous = faction | set(core)
    return SampleAnalysis(ref, frozenset(core), frozenset(ambiguous))


def _core_reference(embs: list[np.ndarray], durs: list[float], min_consistency: float,
                    origins: list[str | None] | None = None) -> np.ndarray | None:
    """The reference voice alone (see _analyse_samples)."""
    return _analyse_samples(embs, durs, min_consistency, origins).ref


def _speaker_sample_rows(db: DB, speaker_id: int) -> list:
    return db.query("SELECT id, embedding, dim, source_path, duration_s, origin FROM speaker_embeddings WHERE speaker_id=? ORDER BY id",
                    (int(speaker_id),))


def check_sample(db: DB, speaker_id: int, emb: np.ndarray, min_consistency: float) -> tuple[float | None, bool]:
    """(cosine between `emb` and the speaker's reference voice or None when there is no sample, ok).
    Samples of another embedding dimension (older model) are not comparable and ignored. When the stored
    samples are ambiguous (two factions, no trusted anchor) the best cosine to any sample is returned and the
    sample is never refused. Pure numpy."""
    e = _unit(emb)
    embs, durs, origins = [], [], []
    for r in _speaker_sample_rows(db, speaker_id):
        if int(r["dim"]) != e.shape[0]:
            continue
        embs.append(_unit(np.frombuffer(r["embedding"], dtype=np.float32)))
        durs.append(r["duration_s"])
        origins.append(r["origin"])
    a = _analyse_samples(embs, durs, min_consistency, origins)
    if a.ref is None:
        return None, True
    if a.ambiguous:
        return max(float(np.dot(e, x)) for x in embs), True
    s = float(np.dot(e, a.ref))
    return s, s >= float(min_consistency)


def sample_influence(db: DB, speaker_id: int, embs: list[np.ndarray | None], threshold: float) -> list[int]:
    """Per sample (aligned with `embs`): number of the speaker's automatic sentences (speaker_source IS NULL) that only
    this sample lets through. Approximated with the stored cluster embeddings: the sentences are grouped by
    (segment_id, cluster_label); a cluster whose embedding reaches `threshold` against exactly one sample (removing
    it drops the best score below the threshold) adds its sentence count to that sample. Sentences without a
    cluster embedding are not counted. Pure numpy."""
    out = [0] * len(embs)
    if not embs:
        return out
    groups = db.query("SELECT segment_id, cluster_label, COUNT(*) AS n FROM utterances WHERE speaker_id=? AND speaker_source IS NULL"
                      " AND cluster_label IS NOT NULL GROUP BY segment_id, cluster_label", (int(speaker_id),))
    if not groups:
        return out
    counts = {(int(g["segment_id"]), int(g["cluster_label"])): int(g["n"]) for g in groups}
    seg_ids = sorted({k[0] for k in counts})
    thr = float(threshold)
    for k in range(0, len(seg_ids), 500):
        for seg, cl, emb, _s in db.cluster_embeddings(seg_ids[k:k + 500]):
            n = counts.get((seg, cl))
            if not n or emb.size == 0:
                continue
            c = _unit(emb)
            hits = [i for i, e in enumerate(embs) if e is not None and e.size and e.shape[0] == c.shape[0]
                    and float(np.dot(c, e)) >= thr]
            if len(hits) == 1:
                out[hits[0]] += n
    return out


def audit_samples(db: DB, speaker_id: int, min_consistency: float, identify_threshold: float | None = None) -> list[dict]:
    """Every stored sample of a speaker with its `origin`, its `consistency` (cosine to the core reference; None
    when it is the only comparable one), `ambiguous` (part of a two-faction stand-off, see _analyse_samples),
    `outlier` (consistency < min_consistency and not ambiguous) and `influence` (automatic sentences only this
    sample lets through at `identify_threshold`, default 0.45; see sample_influence). Pure numpy, no model."""
    rows = _speaker_sample_rows(db, speaker_id)
    embs = [(_unit(np.frombuffer(r["embedding"], dtype=np.float32)) if r["dim"] else None) for r in rows]
    # the reference is built per embedding dimension (samples of an older model are compared among themselves)
    analyses: dict[int, tuple[SampleAnalysis, list[int]]] = {}
    for dim in {e.shape[0] for e in embs if e is not None and e.size}:
        idx = [i for i, e in enumerate(embs) if e is not None and e.shape[0] == dim]
        if len(idx) < 2:
            continue
        a = _analyse_samples([embs[i] for i in idx], [rows[i]["duration_s"] for i in idx], min_consistency,
                             [rows[i]["origin"] for i in idx])
        analyses[dim] = (a, idx)
    try:
        influence = sample_influence(db, speaker_id, embs,
                                     DEFAULT_IDENTIFY_THRESHOLD if identify_threshold is None else float(identify_threshold))
    except Exception as e:  # noqa: BLE001 - diagnostics only
        log.debug("sample influence of speaker #%s failed: %s", speaker_id, e)
        influence = [0] * len(rows)
    out: list[dict] = []
    for i, (r, ei) in enumerate(zip(rows, embs)):
        best: float | None = None
        amb = False
        if ei is not None and ei.size and ei.shape[0] in analyses:
            a, idx = analyses[ei.shape[0]]
            if a.ref is not None:
                best = float(np.dot(ei, a.ref))
                amb = idx.index(i) in a.ambiguous
        out.append({"id": int(r["id"]), "source_path": r["source_path"], "duration_s": r["duration_s"], "origin": r["origin"],
                    "consistency": round(best, 3) if best is not None else None, "ambiguous": amb,
                    "outlier": best is not None and best < float(min_consistency) and not amb,
                    "influence": int(influence[i])})
    return out


def speaker_is_ambiguous(db: DB, speaker_id: int, min_consistency: float) -> bool:
    """True when the speaker's samples split into two factions nobody can arbitrate (see _analyse_samples)."""
    return any(a["ambiguous"] for a in audit_samples(db, speaker_id, min_consistency))


@dataclass
class Turn:
    start: float
    end: float
    cluster: int                 # diarization cluster within this audio
    speaker_id: int | None = None
    speaker_name: str | None = None
    score: float | None = None
    exclude: frozenset[int] = field(default_factory=frozenset)   # speaker ids this turn must never be identified as

    @property
    def duration(self) -> float:
        return self.end - self.start


class SpeakerEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        sp = cfg.speakers
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(cfg.model_path("embedding_model")), num_threads=int(sp.num_threads))
        )
        self.dim = self.extractor.dim
        self._diar: sherpa_onnx.OfflineSpeakerDiarization | None = None
        self._vad: SileroVAD | None = None

    # --- lazy heavy objects ---------------------------------------------
    @property
    def diarizer(self) -> sherpa_onnx.OfflineSpeakerDiarization:
        if self._diar is None:
            sp = self.cfg.speakers
            dcfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(self.cfg.model_path("segmentation_model"))),
                    num_threads=int(sp.num_threads)),
                embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(self.cfg.model_path("embedding_model")), num_threads=int(sp.num_threads)),
                clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=float(sp.cluster_threshold)),
                min_duration_on=float(sp.min_turn_s), min_duration_off=0.5,
            )
            self._diar = sherpa_onnx.OfflineSpeakerDiarization(dcfg)
        return self._diar

    @property
    def vad(self) -> SileroVAD:
        if self._vad is None:
            self._vad = SileroVAD(self.cfg.model_path("vad_model"))
        return self._vad

    # --- embeddings --------------------------------------------------------
    def embed(self, audio: np.ndarray) -> np.ndarray:
        st = self.extractor.create_stream()
        st.accept_waveform(SAMPLE_RATE, audio.astype(np.float32, copy=False))
        st.input_finished()
        e = np.asarray(self.extractor.compute(st), dtype=np.float32)
        n = np.linalg.norm(e)
        return e / n if n > 0 else e

    def speech_only(self, audio: np.ndarray, max_s: float = 60.0) -> np.ndarray:
        """VAD speech parts of `audio` concatenated (up to max_s); the whole clip when VAD finds nothing."""
        segs = segment_audio(audio, self.vad, GateConfig(min_silence_s=0.3, pre_roll_s=0.1, post_roll_s=0.1, max_segment_s=max_s))
        parts = [s.audio for s in segs]
        speech = np.concatenate(parts) if parts else audio
        return speech[: int(max_s * SAMPLE_RATE)]

    def embed_speech_only(self, audio: np.ndarray, max_s: float = 60.0) -> tuple[np.ndarray, float]:
        """Concatenate VAD speech parts (up to max_s) before embedding. Returns (embedding, speech_seconds)."""
        speech = self.speech_only(audio, max_s)
        return self.embed(speech), len(speech) / SAMPLE_RATE

    def _save_sample(self, name: str, speech: np.ndarray, stem: str) -> Path | None:
        """Keep a 16 kHz mono copy of the accepted (speech-only) sample under data/speakers/<name>/ so
        the voice can be re-embedded later (new model / threshold) even if the original file is gone."""
        try:
            d = self.cfg.speakers_dir / _safe_name(name)
            d.mkdir(parents=True, exist_ok=True)
            base = _safe_name(stem) or "sample"
            dest = d / f"{base}.wav"
            k = 0
            while dest.exists():
                k += 1
                dest = d / f"{base}_{k}.wav"
            sf.write(str(dest), speech.astype(np.float32, copy=False), SAMPLE_RATE, format="WAV", subtype="PCM_16")
            return dest
        except Exception as e:  # noqa: BLE001 - a missing copy must not block enrollment
            log.warning("cannot save speaker sample for %s: %s", name, e)
            return None

    # --- diarization -------------------------------------------------------
    def diarize(self, audio: np.ndarray) -> list[Turn]:
        if len(audio) < SAMPLE_RATE:  # < 1 s: single turn
            return [Turn(0.0, len(audio) / SAMPLE_RATE, 0)]
        res = self.diarizer.process(audio.astype(np.float32, copy=False)).sort_by_start_time()
        return [Turn(float(r.start), float(r.end), int(r.speaker)) for r in res]

    # --- identification ----------------------------------------------------
    def cluster_embedding(self, audio: np.ndarray, turns: list[Turn]) -> np.ndarray | None:
        """Embedding of one diarization cluster: its turns concatenated (at most 60 s of them).
        None when they hold less than 0.5 s of audio (too short to embed meaningfully)."""
        parts = [audio[int(t.start * SAMPLE_RATE): int(t.end * SAMPLE_RATE)] for t in turns]
        speech = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        if len(speech) < int(MIN_CLUSTER_SPEECH_S * SAMPLE_RATE):
            return None
        return self.embed(speech[: 60 * SAMPLE_RATE])

    def _turn_check_params(self) -> tuple[float, float]:
        sp = self.cfg.speakers
        try:
            min_s = float(sp.get("turn_check_min_s", DEFAULT_TURN_CHECK_MIN_S))
        except (TypeError, ValueError):
            min_s = DEFAULT_TURN_CHECK_MIN_S
        try:
            margin = float(sp.get("turn_reject_margin", DEFAULT_TURN_REJECT_MARGIN))
        except (TypeError, ValueError):
            margin = DEFAULT_TURN_REJECT_MARGIN
        return min_s, margin

    def _cluster_decisions(self, audio: np.ndarray, clusters: dict[int, list[Turn]], enrolled: list[tuple[int, str, np.ndarray]],
                           thr: float, negatives: list[tuple[int, np.ndarray]] | None = None, margin: float = 0.0
                           ) -> dict[int, tuple[int | None, str | None, float | None, np.ndarray | None]]:
        """Per cluster: (speaker_id, speaker_name, score, embedding) - the decision as before v3. A speaker every turn
        of the cluster excludes (Turn.exclude) is left out of the cluster's candidates."""
        out: dict[int, tuple[int | None, str | None, float | None, np.ndarray | None]] = {}
        for cid, ts in clusters.items():
            emb = self.cluster_embedding(audio, ts)
            if emb is None:
                out[cid] = (None, None, None, None)
                continue
            common = frozenset.intersection(*[frozenset(getattr(t, "exclude", ()) or ()) for t in ts]) if ts else frozenset()
            sid, name, score = self.match(emb, enrolled, thr, negatives, margin=margin, exclude=common)
            out[cid] = (sid, name, score, emb)
        return out

    @staticmethod
    def _sim_to_speaker(emb: np.ndarray, enrolled: list[tuple[int, str, np.ndarray]], sid: int) -> float:
        best = -1.0
        for s, _n, e in enrolled:
            if s == sid and e.shape[0] == emb.shape[0]:
                best = max(best, float(np.dot(emb, _unit(e))))
        return best

    def _turn_audio(self, audio: np.ndarray, t: Turn) -> np.ndarray | None:
        seg = audio[int(t.start * SAMPLE_RATE): int(t.end * SAMPLE_RATE)]
        return seg if len(seg) >= int(MIN_CLUSTER_SPEECH_S * SAMPLE_RATE) else None

    def identify_turns(self, audio: np.ndarray, turns: list[Turn], db: DB, segment_id: int | None = None) -> list[Turn]:
        """Embed each cluster (all its turns concatenated, speech up to 60 s) and match against enrolled speakers.
        With `segment_id` every cluster embedding is also stored in cluster_embeddings (speech_s = sum of its
        turn durations) so the unknown voices can later be grouped and named without re-reading the audio.

        Turn-level correction (only with enrolled speakers and speakers.turn_check_min_s > 0): every turn at
        least that long is embedded on its own; a turn that clearly belongs to another enrolled speaker, or
        does not resemble the cluster's speaker at all (or resembles that speaker's negative samples more), is
        moved to a new cluster id (one per (original cluster, speaker) so build_chunks still merges neighbours).
        Shorter turns keep the cluster decision - unless it is a speaker the turn excludes (Turn.exclude: a
        vetoed speaker, or the user themselves for system audio), then the cluster embedding is matched again
        without that speaker. Negative samples (db.speaker_negatives) apply to every decision (see `match`).
        The list is returned in its original order; cluster ids of moved turns change in place."""
        enrolled = db.speaker_embeddings()
        negatives = load_negatives(db)
        nmargin = negative_margin(self.cfg)
        thr = float(self.cfg.speakers.identify_threshold)
        unknown = self.cfg.speakers.unknown_label
        clusters: dict[int, list[Turn]] = {}
        for t in turns:
            clusters.setdefault(t.cluster, []).append(t)
        decisions = self._cluster_decisions(audio, clusters, enrolled, thr, negatives, nmargin)
        for cid, ts in clusters.items():
            sid, name, score, _emb = decisions[cid]
            for t in ts:
                t.speaker_id, t.speaker_name, t.score = sid, (name if sid is not None else unknown), score

        moved = 0
        moved_s = 0.0
        min_s, margin = self._turn_check_params()
        if enrolled and turns:
            next_id = max(clusters) + 1
            new_ids: dict[tuple[int, int | None], int] = {}
            for cid, ts in list(clusters.items()):
                s_c, _n_c, _sc_c, emb_c = decisions[cid]
                for t in ts:
                    excl = frozenset(getattr(t, "exclude", ()) or ())
                    seg = self._turn_audio(audio, t) if (min_s > 0 and t.duration >= min_s) else None
                    target: tuple[int | None, str | None, float | None] | None = None
                    if seg is not None:
                        e_t = self.embed(seg)
                        s_t, name_t, score_t = self.match(e_t, enrolled, thr, negatives, margin=nmargin, exclude=excl)
                        if s_c is not None:
                            sim_c = self._sim_to_speaker(e_t, enrolled, s_c)
                            neg_c = negative_scores(e_t, negatives).get(s_c, -1.0)
                            holds = s_c not in excl and sim_c >= thr and sim_c - neg_c >= nmargin
                            if s_t is not None and s_t != s_c and not holds:
                                target = (s_t, name_t, score_t)
                            elif s_t is None and (s_c in excl or sim_c < thr - margin or sim_c - neg_c < nmargin):
                                target = (None, None, score_t)
                        elif s_t is not None:
                            target = (s_t, name_t, score_t)
                    elif s_c is not None and s_c in excl:
                        # too short to check on its own: the cluster's voice, minus the speaker this turn excludes
                        target = (self.match(emb_c, enrolled, thr, negatives, margin=nmargin, exclude=excl)
                                  if emb_c is not None else (None, None, None))
                    if target is None:
                        continue
                    key = (cid, target[0])
                    if key not in new_ids:
                        new_ids[key] = next_id
                        next_id += 1
                    t.cluster = new_ids[key]
                    t.speaker_id, t.speaker_name, t.score = target[0], (target[1] if target[0] is not None else unknown), target[2]
                    moved += 1
                    moved_s += t.duration
            if moved:
                log.info("segment #%s: turn-level check moved %d turns (%.1f s)",
                         segment_id if segment_id is not None else "?", moved, moved_s)

        if segment_id is not None:
            final: dict[int, list[Turn]] = {}
            for t in turns:
                final.setdefault(t.cluster, []).append(t)
            for cid, ts in final.items():
                emb = self.cluster_embedding(audio, ts) if (moved or cid not in decisions) else decisions[cid][3]
                if emb is None:
                    continue
                db.upsert_cluster_embedding(int(segment_id), int(cid), emb, sum(t.duration for t in ts))
        return turns

    def identify_utterances(self, audio: np.ndarray, turns: list[Turn], db: DB, segment_id: int | None = None) -> list[Turn]:
        """Per-utterance identification for relabel: cluster decision as a baseline (cluster embeddings stored
        with `segment_id`), then every turn at least speakers.turn_check_min_s long is matched on its own audio
        and carries its own score; shorter turns keep the cluster result (matched again without the speakers the
        turn excludes, when the cluster's speaker is one of them). Negative samples apply. Cluster ids never change."""
        enrolled = db.speaker_embeddings()
        negatives = load_negatives(db)
        nmargin = negative_margin(self.cfg)
        thr = float(self.cfg.speakers.identify_threshold)
        unknown = self.cfg.speakers.unknown_label
        clusters: dict[int, list[Turn]] = {}
        for t in turns:
            clusters.setdefault(t.cluster, []).append(t)
        decisions = self._cluster_decisions(audio, clusters, enrolled, thr, negatives, nmargin)
        min_s, _margin = self._turn_check_params()
        own = 0
        for cid, ts in clusters.items():
            sid, name, score, emb = decisions[cid]
            if emb is not None and segment_id is not None:
                db.upsert_cluster_embedding(int(segment_id), int(cid), emb, sum(t.duration for t in ts))
            for t in ts:
                excl = frozenset(getattr(t, "exclude", ()) or ())
                t.speaker_id, t.speaker_name, t.score = sid, (name if sid is not None else unknown), score
                seg = None
                if enrolled and min_s > 0 and t.duration >= min_s:
                    seg = self._turn_audio(audio, t)
                if seg is None:
                    if sid is not None and sid in excl:
                        s2, n2, sc2 = (self.match(emb, enrolled, thr, negatives, margin=nmargin, exclude=excl)
                                       if emb is not None else (None, None, None))
                        t.speaker_id, t.speaker_name, t.score = s2, (n2 if s2 is not None else unknown), sc2
                    continue
                s_t, name_t, score_t = self.match(self.embed(seg), enrolled, thr, negatives, margin=nmargin, exclude=excl)
                t.speaker_id, t.speaker_name, t.score = s_t, (name_t if s_t is not None else unknown), score_t
                own += 1
        log.debug("segment #%s: %d/%d utterances identified on their own audio", segment_id, own, len(turns))
        return turns

    @staticmethod
    def match(emb: np.ndarray, enrolled: list[tuple[int, str, np.ndarray]], threshold: float,
              negatives: list[tuple[int, np.ndarray]] | None = None, margin: float = 0.0,
              exclude: Any = None) -> tuple[int | None, str | None, float | None]:
        """(speaker_id, name, score) of the enrolled speaker `emb` belongs to, or (None, None, best score).

        Candidate p = best cosine to the speaker's samples. Without `negatives` the best candidate wins when
        p >= threshold (the rule before v4). With negatives (from db.speaker_negatives) a candidate X also needs
        p - n >= `margin`, n = best cosine to X's negative samples (-1 without any); a candidate that fails is
        skipped and the next one by p is tried. Speakers in `exclude` are never candidates. The score returned is
        always p (of the winner, or the best remaining candidate when nobody qualifies); samples of another
        embedding dimension are ignored."""
        if not enrolled:
            return None, None, None
        emb = np.asarray(emb, dtype=np.float32).ravel()
        excl = {int(x) for x in (exclude or ())}
        best: dict[int, tuple[str, float]] = {}
        for sid, name, e in enrolled:
            if sid in excl or e.shape[0] != emb.shape[0]:
                continue
            e = e / (np.linalg.norm(e) or 1.0)
            s = float(np.dot(emb, e))
            if sid not in best or s > best[sid][1]:
                best[sid] = (name, s)
        if not best:
            return None, None, None
        ranked = sorted(best.items(), key=lambda kv: -kv[1][1])
        neg = negative_scores(emb, negatives)
        for sid, (name, score) in ranked:
            if score < threshold:
                break
            if score - neg.get(sid, -1.0) >= float(margin):
                return sid, name, score
        return None, None, ranked[0][1][1]

    # --- sample consistency ----------------------------------------------------
    def check_sample(self, db: DB, speaker_id: int, emb: np.ndarray) -> tuple[float | None, bool]:
        """(best cosine to the speaker's existing samples or None when there is none, ok = not below
        speakers.sample_consistency_min)."""
        return check_sample(db, speaker_id, emb, sample_consistency_min(self.cfg))

    def audit_samples(self, db: DB, speaker_id: int) -> list[dict]:
        """Stored samples with `consistency` / `outlier` (see module-level audit_samples); no model involved."""
        try:
            thr: float | None = float(self.cfg.speakers.identify_threshold)
        except (TypeError, ValueError, AttributeError):
            thr = None
        return audit_samples(db, speaker_id, sample_consistency_min(self.cfg), thr)

    def _guard_sample(self, db: DB, name: str, emb: np.ndarray, force: bool) -> float | None:
        """Raise SampleMismatch when `emb` does not resemble `name`'s existing samples (unless force).
        Returns the best score (None without samples / without the speaker)."""
        row = db.one("SELECT id FROM speakers WHERE name=?", (name,))
        if row is None:
            return None
        score, ok = self.check_sample(db, int(row["id"]), emb)
        if not ok and not force:
            raise SampleMismatch(name, float(score), sample_consistency_min(self.cfg))
        return score

    # --- enrollment ----------------------------------------------------------
    def enroll(self, db: DB, name: str, sources: list[Path | str], notes: str | None = None, force: bool = False) -> int:
        """Register `name` with one embedding per source file. Files are decoded with libsndfile/ffmpeg.
        A file whose voice does not resemble the person's existing samples is skipped (logged) unless `force`."""
        from ..audio.importer import decode_any

        sid = db.upsert_speaker(name)
        n = 0
        for src in sources:
            src = Path(src)
            audio = decode_any(src)
            speech = self.speech_only(audio)
            speech_s = len(speech) / SAMPLE_RATE
            if speech_s < MIN_ENROLL_SPEECH_S:
                log.warning("skip %s: only %.1fs speech", src, speech_s)
                continue
            emb = self.embed(speech)
            score, ok = self.check_sample(db, sid, emb)
            if not ok and not force:
                log.warning("skip %s: does not resemble %s's existing samples (cosine %.2f < %.2f); use force to add it anyway",
                            src, name, score, sample_consistency_min(self.cfg))
                continue
            saved = self._save_sample(name, speech, src.stem)
            db.add_speaker_embedding(sid, emb, str(saved or src), speech_s, origin="file")
            n += 1
        if n and notes is not None:
            db.upsert_speaker(name, notes)      # notes only once at least one sample made it in
        log.info("enrolled %s with %d samples", name, n)
        return sid

    def enroll_from_audio(self, db: DB, name: str, audio: np.ndarray, source: str | None = None,
                          copy_as_source: bool = False, force: bool = False, min_speech_s: float | None = None,
                          origin: str | None = "label") -> int:
        """Add one embedding for `name` from an in-memory clip. Raises ValueError when the clip holds
        less than MIN_ENROLL_SPEECH_S of speech (a noise embedding would later attract false matches) and
        SampleMismatch when the voice does not resemble the person's existing samples (unless `force`).
        `source` ("<file>#a-b") is stored as the sample's source_path and names the saved copy; with
        `copy_as_source` the saved speech-only copy is stored instead (for clips stitched from several
        ranges, where a "#a-b" slice of the original would not play back the right audio). `origin` is the
        sample's provenance stored next to it ('label' by default, 'voice' when naming a voice group)."""
        speech = self.speech_only(audio)
        speech_s = len(speech) / SAMPLE_RATE
        need = MIN_ENROLL_SPEECH_S if min_speech_s is None else max(float(min_speech_s), MIN_ENROLL_SPEECH_S)
        if speech_s < need:
            raise TooShortSample(speech_s, need)
        emb = self.embed(speech)
        self._guard_sample(db, name, emb, force)
        sid = db.upsert_speaker(name)
        saved = self._save_sample(name, speech, Path(str(source).split("#")[0]).stem if source else "label")
        stored = str(saved) if saved and (copy_as_source or source is None) else source
        db.add_speaker_embedding(sid, emb, stored, speech_s, origin=origin)
        return sid

    def enroll_from_segment_turns(self, db: DB, name: str, seg_path: Path | str, turns: list[tuple[float, float]],
                                  force: bool = False, min_speech_s: float | None = None) -> int:
        """Label an existing recording's time ranges as `name` (used by `audio-note speakers label`)."""
        audio = read_segment(seg_path)
        parts = [audio[int(a * SAMPLE_RATE): int(b * SAMPLE_RATE)] for a, b in turns]
        return self.enroll_from_audio(db, name, np.concatenate(parts), f"{seg_path}#{turns[0][0]:.1f}-{turns[-1][1]:.1f}",
                                      force=force, min_speech_s=min_speech_s, origin="label")


def _safe_name(s: str) -> str:
    """File-system friendly version of a speaker name / stem."""
    return "".join(ch if ch.isalnum() or ch in "-_ ." else "_" for ch in str(s)).strip(" .") or "_"
