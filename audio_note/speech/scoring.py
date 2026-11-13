"""Scoring layer of speaker identification (SPEC_v1 section 5.1): who a voice vector belongs to. Pure numpy.

Everything that decides "this voice is X" goes through here - the first pass of the pipeline (SpeakerEngine
.identify_turns), relabel, split and `rescore` - so a sentence gets the same answer whenever it is asked.

    Gallery.load(db, cfg)          the reference voices: core samples, per-condition centroids, negatives, cohort
    Gallery.score(emb, ...)        one vector -> Decision (sure / guess / unknown)
    Gallery.score_many(X, ...)     the same for a matrix (rescore, bench)
    condition_of(loudness, share)  near | far | system
    adapt(db, cfg)                 rebuild speaker_centroids from trusted evidence
    rescore(db, cfg, ...)          re-decide stored sentences from stored vectors (no audio, no model)
    run_pending(cfg, db, ...)      chain step: adapt + rescore when the reference voices changed
    job_rescore(rt, **kw)          the `rescore` job

How a decision is made
----------------------
1. Raw score of candidate S = cosine to S's *anchor* (the duration-weighted mean of the core samples: the samples
   that pass the consistency check of `analyse_samples`) or to S's centroid for the recording condition of the
   vector, whichever supports S more. Samples outside the core never take part: one stray sample can no longer
   recognise hundreds of sentences on its own.
2. Centroids come from `adapt`: the anchor pulled towards what S really sounds like near the microphone, far from
   it and over system audio, learned from sentences the user confirmed and from confident automatic decisions.
   A centroid that drifts away from the anchor is refused.
3. With enough impostor voices stored (the *cohort*: stored cluster vectors that are nobody enrolled, system
   audio, other people's voices, vetoed voices) the raw score is normalised against them:

       norm = raw - (mean score of S's closest impostors against S
                     + mean score of S's closest impostors against this very vector) / 2

   i.e. how far, in cosine, the vector stands above what impostors reach - against the speaker and against
   itself (a faint far-away vector scores low against everybody; a look-alike scores high against the cohort
   members that sound like it). "Closest" is a share of the cohort (cohort_top_share), so the scale does not
   move while the history grows. This is adaptive score normalisation without the division by the spread of
   the impostor scores: with that division the scale differed by a factor of two between the synthetic bench
   and the real data (the spread of the top third of a small cohort is not a stable number), so no threshold
   fitted both; the mean-only form keeps the unit in cosine and one threshold serves both.
   Without a cohort the raw cosine and the old threshold decide (and a track must clear it by a margin).
4. Short vectors are weak evidence: under `follow_s` seconds a vector never decides on its own, under `short_s`
   it needs a higher score. `tracks.py` lets short sentences follow their cluster / track instead.
5. Three states: sure (speaker_id), guess (guess_id: "maybe X", shown to the user, never written to speaker_id),
   unknown. Excluded speakers (vetoes, the self speaker on system audio) and negative samples always win.

Keeping history in step: `refresh` (chain step `run_pending`, and after every change of samples / negatives /
labels made in the UI or the CLI) runs `adapt` when its inputs changed and `rescore` when the gallery version
differs from the one history was last decided with. Both are deterministic functions of what is stored, so
running them twice changes nothing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta
from typing import Any, Iterable, Sequence

import numpy as np

log = logging.getLogger(__name__)

CONDITIONS = ("near", "far", "system")
ALL = "all"                      # the centroid over every condition (used when a condition has none of its own)
ANCHOR_DURATION_CAP_S = 10.0     # duration weight cap when no trusted (mic / file) sample can anchor the core
SELF_MATCH_COS = 0.985           # a cohort vector this close to the scored vector is the vector itself
KV_RESCORE = "speakers_rescore"  # {"version", "at", "summary"} of the last rescore
KV_ADAPT = "speakers_adapt"      # {"inputs", "clusters", "at"} of the last adapt
EVIDENCE_CAP_S = 10.0            # one cluster weighs at most this many seconds in a centroid


# ================================================================================================ parameters
@dataclass(frozen=True)
class Params:
    """Every number the scoring layer uses, read once from cfg.speakers (missing keys fall back to these
    defaults, so old config files and test stand-ins keep working). The defaults come from
    scripts/bench/bench_speakers.py; config.DEFAULTS carries the same values."""
    scoring: str = "v5"                 # v5 | legacy (best single sample >= identify_threshold, nothing else)
    identify_threshold: float = 0.45    # raw cosine for "sure" when there is no cohort to normalise with
    guess_raw_margin: float = 0.08      # no cohort: raw >= threshold - this is worth a "maybe"
    negative_margin: float = 0.0
    system_turn_share: float = 0.6
    sample_consistency_min: float = 0.5
    sure_norm: float = 0.25             # normalised score for "sure"
    guess_norm: float = 0.15            # normalised score for "maybe"
    margin_norm: float = 0.05           # the winner must beat the runner-up by this (normalised)
    raw_floor: float = 0.30             # normalised decisions never accept a raw cosine below this
    cohort_min: int = 20                # fewer impostor vectors than this: no normalisation
    cohort_top_share: float = 0.33      # adaptive part: this share of the impostors, the closest ones (>= 10)
    cohort_max: int = 300               # impostors kept per speaker (the most recent ones)
    cohort_scan: int = 3000             # most recent stored clusters looked at for the cohort
    cohort_max_raw: float = 0.35        # a stored cluster closer than this to a speaker is not their impostor
    cohort_min_s: float = 3.0
    short_s: float = 2.0                # below: soft evidence (stricter on its own, follows its track)
    follow_s: float = 1.0               # below: never decides on its own
    short_extra_norm: float = 0.05      # extra normalised score a short_s vector needs to be sure on its own
    short_extra_raw: float = 0.05       # the same without a cohort (raw cosine)
    follow_slack: float = 0.05          # a short vector "leans" towards X when its score is within this of a guess
    turn_reject_margin: float = 0.15    # no cohort: a turn below threshold - this does not belong to its cluster's speaker
    track_extra_raw: float = 0.10       # no cohort: a track hands its speaker on only this far above the threshold
    far_dbfs: float = -30.0             # speech level below this = "far" condition
    adapt: bool = True
    adapt_min_anchor_cos: float = 0.5   # a centroid further than this from the anchor is refused (drift guard)
    adapt_min_clusters: int = 3         # evidence clusters a condition needs for a centroid of its own
    adapt_min_s: float = 3.0            # automatic evidence needs this much speech
    adapt_extra_norm: float = 0.05      # ... and a score this far above sure_norm
    adapt_extra_raw: float = 0.10       # the same without a cohort
    adapt_anchor_weight_s: float = 20.0 # the anchor counts like this many seconds of evidence (shrinkage)
    adapt_scan: int = 5000              # most recent stored clusters looked at for evidence
    min_speech_dbfs: float | None = -40.0

    @classmethod
    def from_cfg(cls, cfg: Any, **overrides: Any) -> "Params":
        sp = _section(cfg, "speakers")
        vals: dict[str, Any] = {}
        for f in fields(cls):
            if f.name == "min_speech_dbfs":
                continue
            raw = sp.get(f.name, f.default) if hasattr(sp, "get") else f.default
            vals[f.name] = _coerce(raw, f.default)
        vals["scoring"] = str(vals["scoring"] or "v5").strip().lower()
        try:
            from ..audio.loudness import min_speech_dbfs

            vals["min_speech_dbfs"] = min_speech_dbfs(cfg)
        except Exception:  # noqa: BLE001 - config stand-ins without an audio section
            vals["min_speech_dbfs"] = None
        vals.update(overrides)
        return cls(**vals)

    def key(self) -> str:
        """The values a stored decision depends on (part of the gallery version)."""
        skip = {"cohort_scan", "adapt_scan"}
        return json.dumps({f.name: getattr(self, f.name) for f in fields(self) if f.name not in skip}, sort_keys=True)


def _section(cfg: Any, name: str) -> Any:
    try:
        return getattr(cfg, name)
    except Exception:  # noqa: BLE001
        try:
            return cfg[name]
        except Exception:  # noqa: BLE001
            return {}


def _coerce(value: Any, default: Any) -> Any:
    """`value` as the type of `default`; the default itself when it cannot be read that way."""
    if value is None:
        return default
    try:
        if isinstance(default, bool):
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if isinstance(default, int):
            return int(value)
        if isinstance(default, float):
            v = float(value)
            return v if math.isfinite(v) else default
        return type(default)(value)
    except (TypeError, ValueError):
        return default


def model_id(cfg: Any) -> str:
    """Identifier of the voice model the vectors come from (utterance_voice.model): the model file's stem."""
    try:
        name = str(_section(cfg, "speakers").get("embedding_model") or "")
    except Exception:  # noqa: BLE001
        name = ""
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0] if name else "default"


def unit(v: Any) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _blob_unit(blob: Any, dim: int | None = None) -> np.ndarray | None:
    if not blob:
        return None
    v = np.frombuffer(bytes(blob), dtype=np.float32)
    if v.size == 0 or (dim is not None and v.shape[0] != dim):
        return None
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 and math.isfinite(n) else None


# ================================================================================================ samples
@dataclass
class SampleAnalysis:
    """Result of analyse_samples: the reference voice, which samples form the core and which are ambiguous."""
    ref: np.ndarray | None
    core: frozenset[int] = frozenset()        # indices of the core faction (medoid + its neighbours)
    ambiguous: frozenset[int] = frozenset()   # indices of samples in a two-faction stand-off (core + the rival faction)


def analyse_samples(embs: list[np.ndarray], durs: list[float], min_consistency: float,
                    origins: list[str | None] | None = None) -> SampleAnalysis:
    """Reference voice of a speaker from their stored samples (the anchor rules of speakers v3b):

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
    ref = unit((E[core] * w[core][:, None]).sum(axis=0))
    ambiguous: set[int] = set()
    if not trusted:
        rest = [j for j in range(n) if j not in core]
        faction = {i for i in rest if any(j != i and S[i, j] >= thr for j in rest)}
        if len(faction) >= 2:
            ambiguous = faction | set(core)
    return SampleAnalysis(ref, frozenset(core), frozenset(ambiguous))


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


# ================================================================================================ decisions
@dataclass
class Decision:
    speaker_id: int | None        # sure: the automatic attribution
    name: str | None
    score: float | None           # raw cosine shown to the user (of the winner, else of the best candidate)
    norm: float | None            # normalised score: cosine above the closest impostors (None without a cohort)
    guess_id: int | None          # "maybe X": never together with speaker_id
    guess_score: float | None     # normalised score of the guess (raw cosine without a cohort)
    method: str                   # anchor | centroid:<condition> | track | smooth | veto | none
    runner_up: float | None       # score of the second candidate, in the unit of the decision (norm, else raw)
    condition: str | None = None  # near | far | system: the recording condition the vector was scored under
    guess_name: str | None = None

    @property
    def state(self) -> str:
        return "sure" if self.speaker_id is not None else "guess" if self.guess_id is not None else "unknown"

    def meta(self) -> dict[str, Any]:
        """What utterances.speaker_meta stores about this decision (small, JSON)."""
        out: dict[str, Any] = {"m": self.method}
        if self.score is not None:
            out["raw"] = round(float(self.score), 3)
        if self.norm is not None:
            out["norm"] = round(float(self.norm), 2)
        if self.runner_up is not None:
            out["next"] = round(float(self.runner_up), 2)
        if self.condition:
            out["cond"] = self.condition
        return out


NO_DECISION = Decision(None, None, None, None, None, None, "none", None)


def condition_of(loudness_db: float | None, system_share: float | None, cfg: Any) -> str:
    """Recording condition of a voice vector: 'system' when most of its energy came from system audio, 'far'
    when its speech level is below speakers.far_dbfs, else 'near' (also when the level is unknown)."""
    p = cfg if isinstance(cfg, Params) else Params.from_cfg(cfg)
    if system_share is not None and float(system_share) >= p.system_turn_share:
        return "system"
    if loudness_db is not None and float(loudness_db) < p.far_dbfs:
        return "far"
    return "near"


# ================================================================================================ gallery
@dataclass
class _Speaker:
    sid: int
    name: str
    anchor: np.ndarray                         # unit: weighted mean of the core samples
    core: np.ndarray                           # [k, d] the core samples
    every: np.ndarray                          # [n, d] all samples of this dimension (legacy mode only)
    outliers: int = 0
    centroids: dict[str, np.ndarray] = field(default_factory=dict)      # condition (or ALL) -> unit vector
    negatives: np.ndarray | None = None        # [k, d]
    stats: dict[str, float] = field(default_factory=dict)   # model key -> mean score of its closest impostors


@dataclass
class Evaluation:
    """Scores of one vector against every speaker of the gallery (what a caller needs beyond the Decision)."""
    decision: Decision
    sids: list[int]
    raw: np.ndarray               # [S] raw cosine per speaker
    norm: np.ndarray | None       # [S] normalised score per speaker (None without a cohort)
    allowed: np.ndarray           # [S] False for excluded speakers and for speakers the negatives veto
    sure_at: float                # the threshold "sure" needed for this vector (unit of `norm`, else of `raw`)
    guess_at: float
    reject_at: float = -math.inf  # below this the vector does not resemble the speaker at all
    slack: float = 0.05

    def _col(self, sid: int | None) -> int | None:
        try:
            return self.sids.index(int(sid)) if sid is not None else None
        except ValueError:
            return None

    def value(self, sid: int | None) -> float | None:
        """Score of speaker `sid` in the unit the thresholds are in (norm when available, else raw)."""
        j = self._col(sid)
        if j is None:
            return None
        return float(self.norm[j]) if self.norm is not None else float(self.raw[j])

    def holds(self, sid: int | None) -> bool:
        """True when this vector alone would be accepted as `sid` (were `sid` its best candidate)."""
        j = self._col(sid)
        return bool(j is not None and self.allowed[j] and self.value(sid) >= self.sure_at)

    def plausible(self, sid: int | None) -> bool:
        """True when this vector does not speak against `sid`: allowed and at least worth a guess."""
        j = self._col(sid)
        return bool(j is not None and self.allowed[j] and self.value(sid) >= self.guess_at)

    def leans(self, sid: int | None) -> bool:
        """Weaker than `plausible`, for short vectors: allowed and within `slack` of a guess."""
        j = self._col(sid)
        return bool(j is not None and self.allowed[j] and self.value(sid) >= self.guess_at - self.slack)

    def rejects(self, sid: int | None) -> bool:
        """True when this vector cannot be `sid`: excluded, vetoed by a negative sample, or far below a guess."""
        j = self._col(sid)
        return bool(j is None or not self.allowed[j] or self.value(sid) < self.reject_at)

    def following(self, sid: int, name: str | None, method: str, runner_up: float | None = None) -> Decision:
        """The Decision of a vector that takes speaker `sid` from its cluster / track / neighbours: the speaker is
        theirs, the scores are this vector's own."""
        j = self.sids.index(int(sid))
        return Decision(int(sid), name, float(self.raw[j]), float(self.norm[j]) if self.norm is not None else None,
                        None, None, method, runner_up, self.decision.condition)


class Gallery:
    """The reference voices of one database at one moment. Build with `Gallery.load(db, cfg)`."""

    _cache: dict[str, tuple[Any, "Gallery"]] = {}

    def __init__(self, params: Params, speakers: list[_Speaker], *, self_id: int | None = None, dim: int = 0,
                 cohort: np.ndarray | None = None, cohort_masks: np.ndarray | None = None,
                 cohort_segments: np.ndarray | None = None, model: str = "default", inputs: str = ""):
        self.params = params
        self.speakers = speakers
        self.sids = [s.sid for s in speakers]
        self.names = {s.sid: s.name for s in speakers}
        self.self_id = self_id
        self.dim = int(dim)
        self.model = model
        self.cohort = cohort if cohort is not None and len(cohort) else None
        self.cohort_masks = cohort_masks
        self.cohort_segments = cohort_segments
        self.normalised = False
        self._A = np.stack([s.anchor for s in speakers]) if speakers else np.zeros((0, self.dim), dtype=np.float32)
        self._prepare_cohort()
        self.inputs = inputs                   # fingerprint of samples / negatives / parameters (not the centroids)
        self.version = self._version()

    # --- construction ---------------------------------------------------------
    @classmethod
    def load(cls, db: Any, cfg: Any, *, centroids: bool = True, params: Params | None = None) -> "Gallery":
        """Read samples (with origin), centroids, negatives and the impostor cohort. Cached per database until
        any of them (or a scoring setting) changes. `centroids=False` gives the anchors-only gallery `adapt`
        starts from."""
        p = params or Params.from_cfg(cfg)
        stamp = _stamp(db, p, centroids)
        key = f"{getattr(db, 'path', id(db))}|{int(bool(centroids))}"
        hit = cls._cache.get(key)
        if stamp is not None and hit is not None and hit[0] == stamp:
            return hit[1]
        g = _load_gallery(db, cfg, p, centroids)
        if stamp is not None:
            if len(cls._cache) > 16:
                cls._cache.clear()
            cls._cache[key] = (stamp, g)
        return g

    @classmethod
    def forget(cls, db: Any = None) -> None:
        """Drop cached galleries (of one database, or all)."""
        if db is None:
            cls._cache.clear()
            return
        prefix = f"{getattr(db, 'path', id(db))}|"
        for k in [k for k in cls._cache if k.startswith(prefix)]:
            cls._cache.pop(k, None)

    def _version(self) -> str:
        h = hashlib.sha1(self.inputs.encode("utf-8"))
        for s in self.speakers:
            for cond in sorted(s.centroids):
                h.update(f"{s.sid}:{cond}:".encode("utf-8"))
                h.update(np.round(s.centroids[cond], 3).astype(np.float32).tobytes())
        return h.hexdigest()[:16]

    def _prepare_cohort(self) -> None:
        """Per speaker and per model (anchor, each centroid): the mean score of its closest impostors.
        Normalisation is on only when every speaker has at least cohort_min impostors (one unit for the whole
        ranking)."""
        p = self.params
        self.normalised = False
        if self.cohort is None or self.cohort_masks is None or not self.speakers or p.scoring == "legacy":
            return
        counts = self.cohort_masks.sum(axis=1)
        if int(counts.min()) < max(2, int(p.cohort_min)):
            return
        for j, s in enumerate(self.speakers):
            rows = self.cohort[self.cohort_masks[j]]
            top = self._top(len(rows))
            s.stats = {}
            for key, vec in [("anchor", s.anchor)] + [(f"centroid:{c}", v) for c, v in s.centroids.items()]:
                s.stats[key] = float(np.sort(rows @ vec)[-top:].mean())
        self.normalised = True

    def _top(self, m: int) -> int:
        """How many of a speaker's `m` impostors count as "the closest ones". A share, not a number: the scale
        of the normalised score then stays the same while the history (and with it the cohort) grows."""
        return max(1, min(int(m), max(10, int(round(float(self.params.cohort_top_share) * int(m))))))

    @property
    def has_speakers(self) -> bool:
        return bool(self.speakers)

    def describe(self) -> dict[str, Any]:
        """Aggregate facts for status pages and job results (no vectors)."""
        return {"version": self.version, "speakers": len(self.speakers), "normalised": self.normalised,
                "cohort": int(len(self.cohort)) if self.cohort is not None else 0,
                "centroids": {str(s.sid): sorted(s.centroids) for s in self.speakers if s.centroids},
                "outlier_samples": int(sum(s.outliers for s in self.speakers)), "model": self.model}

    # --- scoring --------------------------------------------------------------
    def score(self, emb: Any, *, speech_s: float | None = None, loudness_db: float | None = None,
              system_share: float | None = None, exclude: frozenset[int] = frozenset(),
              segment_id: Any = None) -> Decision:
        """Decide one voice vector. `speech_s` is how much speech it was computed from (short vectors are weak
        evidence), `loudness_db` / `system_share` select the condition centroid, `exclude` are speakers it must
        never be (vetoes). The self speaker is excluded on system audio by itself. `segment_id` keeps the
        vector's own segment out of the impostor statistics (re-scoring stored vectors)."""
        return self.evaluate(emb, speech_s=speech_s, loudness_db=loudness_db, system_share=system_share,
                             exclude=exclude, segment_id=segment_id).decision

    def evaluate(self, emb: Any, *, speech_s: float | None = None, loudness_db: float | None = None,
                 system_share: float | None = None, exclude: Iterable[int] = frozenset(),
                 segment_id: Any = None) -> Evaluation:
        e = np.asarray(emb, dtype=np.float32).ravel()
        if not self.speakers or e.shape[0] != self.dim:
            return Evaluation(NO_DECISION, [], np.zeros(0), None, np.zeros(0, dtype=bool), math.inf, math.inf)
        return self.evaluate_many(e[None, :], speech_s=[speech_s], loudness_db=[loudness_db], system_share=[system_share],
                                  excludes=[frozenset(int(x) for x in (exclude or ()))], segments=[segment_id])[0]

    def score_many(self, X: Any, **kw: Any) -> list[Decision]:
        return [ev.decision for ev in self.evaluate_many(X, **kw)]

    def evaluate_many(self, X: Any, *, speech_s: Sequence[float | None] | None = None,
                      loudness_db: Sequence[float | None] | None = None,
                      system_share: Sequence[float | None] | None = None,
                      excludes: Sequence[Iterable[int]] | None = None,
                      segments: Sequence[Any] | None = None) -> list[Evaluation]:
        """`evaluate` for the rows of X ([n, dim], unit vectors or not)."""
        p = self.params
        X = np.asarray(X, dtype=np.float32)
        n = int(X.shape[0]) if X.ndim == 2 else 0
        if n == 0:
            return []
        if not self.speakers or X.shape[1] != self.dim:
            empty = Evaluation(NO_DECISION, [], np.zeros(0), None, np.zeros(0, dtype=bool), math.inf, math.inf)
            return [empty] * n
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.where(norms > 0, norms, 1.0)
        S = len(self.speakers)
        conds = [condition_of(loudness_db[i] if loudness_db is not None else None,
                              system_share[i] if system_share is not None else None, p) for i in range(n)]
        legacy = p.scoring == "legacy"
        raw = np.empty((n, S), dtype=np.float64)
        model = np.full((n, S), "anchor", dtype=object)
        if legacy:
            for j, s in enumerate(self.speakers):
                raw[:, j] = (X @ s.every.T).max(axis=1)
        else:
            raw[:] = X @ self._A.T
        nrm: np.ndarray | None = None
        if self.normalised:
            T = X @ self.cohort.T
            T[T >= SELF_MATCH_COS] = -1.0            # the vector itself (re-scoring what the cohort was built from)
            if segments is not None and self.cohort_segments is not None:
                for i in range(n):
                    if segments[i] is not None:
                        T[i, self.cohort_segments == segments[i]] = -1.0
            nrm = np.empty((n, S), dtype=np.float64)
            t_mu = np.empty((n, S))
            for j, s in enumerate(self.speakers):
                Tj = T[:, self.cohort_masks[j]]
                top = self._top(Tj.shape[1])
                t_mu[:, j] = np.partition(Tj, Tj.shape[1] - top, axis=1)[:, -top:].mean(axis=1)
                nrm[:, j] = raw[:, j] - 0.5 * (s.stats["anchor"] + t_mu[:, j])
        if not legacy:
            # the centroid of the vector's condition (else the all-conditions one) may support S more than the anchor
            for cond in set(conds):
                rows = np.array([i for i in range(n) if conds[i] == cond], dtype=np.int64)
                for j, s in enumerate(self.speakers):
                    key = cond if cond in s.centroids else ALL if ALL in s.centroids else None
                    if key is None:
                        continue
                    rc = X[rows] @ s.centroids[key]
                    if nrm is not None:
                        zc = rc - 0.5 * (s.stats[f"centroid:{key}"] + t_mu[rows, j])
                        better = zc > nrm[rows, j]
                        nrm[rows[better], j] = zc[better]
                    else:
                        better = rc > raw[rows, j]
                    raw[rows[better], j] = rc[better]
                    model[rows[better], j] = f"centroid:{key}"
        out: list[Evaluation] = []
        self_col = self.sids.index(self.self_id) if self.self_id in self.sids else None
        for i in range(n):
            excl = {int(x) for x in (excludes[i] if excludes is not None and excludes[i] else ())}
            allowed = np.array([sid not in excl for sid in self.sids], dtype=bool)
            if self_col is not None and conds[i] == "system":
                allowed[self_col] = False                      # system audio is never the user themselves
            dur = speech_s[i] if speech_s is not None else None
            out.append(self._decide(X[i], raw[i], nrm[i] if nrm is not None else None, model[i], allowed, conds[i], dur))
        return out

    def _thresholds(self, speech_s: float | None) -> tuple[float, float, float, float]:
        """(sure, guess, minimum raw cosine, reject) for a vector computed from `speech_s` seconds of speech."""
        p = self.params
        short = speech_s is not None and float(speech_s) < p.short_s
        tiny = speech_s is not None and float(speech_s) < p.follow_s
        if self.normalised:
            sure = math.inf if tiny else p.sure_norm + (p.short_extra_norm if short else 0.0)
            return sure, p.guess_norm, p.raw_floor, p.guess_norm - p.follow_slack
        if p.scoring == "legacy":
            return p.identify_threshold, math.inf, -1.0, p.identify_threshold - p.turn_reject_margin
        sure = math.inf if tiny else p.identify_threshold + (p.short_extra_raw if short else 0.0)
        return sure, p.identify_threshold - p.guess_raw_margin, -1.0, p.identify_threshold - p.turn_reject_margin

    def _decide(self, x: np.ndarray, raw: np.ndarray, nrm: np.ndarray | None, model: np.ndarray, allowed: np.ndarray,
                cond: str, speech_s: float | None) -> Evaluation:
        p = self.params
        sure_at, guess_at, floor, reject_at = self._thresholds(speech_s)
        allowed = allowed.copy()
        val = nrm if nrm is not None else raw
        base = allowed.copy()                      # before the negatives: what the displayed score is taken from
        vetoed = not base.all()
        for j, s in enumerate(self.speakers):
            if allowed[j] and s.negatives is not None and len(s.negatives):
                if raw[j] - float((s.negatives @ x).max()) < p.negative_margin:
                    allowed[j] = False             # resembles "not S" at least as much as S
                    vetoed = vetoed or val[j] >= guess_at
        ev = Evaluation(NO_DECISION, self.sids, raw, nrm, allowed, sure_at, guess_at, reject_at, p.follow_slack)
        if not base.any():
            ev.decision = Decision(None, None, None, None, None, None, "veto", None, cond)
            return ev
        shown = float(raw[base].max())
        if not allowed.any():
            ev.decision = Decision(None, None, shown, None, None, None, "veto", None, cond)
            return ev
        order = [j for j in np.argsort(-val) if allowed[j]]
        j1 = int(order[0])
        v1, r1 = float(val[j1]), float(raw[j1])
        v2 = float(val[order[1]]) if len(order) > 1 else None
        s1 = self.speakers[j1]
        clear = nrm is None or v2 is None or (v1 - v2) >= p.margin_norm
        if v1 >= sure_at and r1 >= floor and clear:
            ev.decision = Decision(s1.sid, s1.name, r1, v1 if nrm is not None else None, None, None, str(model[j1]), v2, cond)
        elif v1 >= guess_at and r1 >= floor:
            ev.decision = Decision(None, None, shown, v1 if nrm is not None else None, s1.sid, v1, str(model[j1]), v2, cond,
                                   guess_name=s1.name)
        else:
            ev.decision = Decision(None, None, shown, v1 if nrm is not None else None, None, None,
                                   "veto" if vetoed else "none", v2, cond)
        return ev


# ================================================================================================ loading
def _stamp(db: Any, p: Params, centroids: bool) -> Any:
    """Cheap fingerprint of everything a Gallery is built from; None when the database cannot say (stand-ins)."""
    try:
        a = db.one("SELECT COUNT(*) AS n, COALESCE(MAX(id),0) AS m, COALESCE(SUM(dim),0) AS d FROM speaker_embeddings")
        b = db.one("SELECT COUNT(*) AS n, COALESCE(MAX(id),0) AS m FROM speaker_negatives")
        c = db.one("SELECT COUNT(*) AS n, COALESCE(MAX(updated_at),'') AS m, COALESCE(SUM(n),0) AS s FROM speaker_centroids")
        d = db.one("SELECT COUNT(*) AS n, COALESCE(MAX(segment_id),0) AS m FROM cluster_embeddings")
        e = db.one("SELECT COUNT(*) AS n, COALESCE(MAX(id),0) AS m, COALESCE(SUM(is_self * id),0) AS s FROM speakers")
        f = db.one("SELECT COUNT(*) AS n FROM utterance_rejections")
        g = db.one("SELECT COUNT(*) AS n, COALESCE(MAX(id),0) AS m FROM utterances WHERE speaker_source='user'")
        return (tuple(a), tuple(b), tuple(c) if centroids else None, tuple(d), tuple(e), tuple(f), tuple(g), p.key())
    except Exception:  # noqa: BLE001
        return None


def _sample_rows(db: Any) -> list[dict[str, Any]]:
    """Every enrolment sample as a dict (tolerant of older schemas and of DB stand-ins that only offer
    speaker_embeddings())."""
    for sql in ("SELECT e.id, e.speaker_id, s.name, e.embedding, e.dim, e.duration_s, e.origin FROM speaker_embeddings e"
                " JOIN speakers s ON s.id=e.speaker_id ORDER BY e.id",
                "SELECT e.id, e.speaker_id, s.name, e.embedding, e.dim, e.duration_s, NULL AS origin FROM speaker_embeddings e"
                " JOIN speakers s ON s.id=e.speaker_id ORDER BY e.id"):
        try:
            return [dict(r) for r in db.query(sql)]
        except Exception:  # noqa: BLE001
            continue
    try:
        return [{"id": k, "speaker_id": int(sid), "name": name, "embedding": np.asarray(e, dtype=np.float32).tobytes(),
                 "dim": int(np.asarray(e).shape[0]), "duration_s": None, "origin": None}
                for k, (sid, name, e) in enumerate(db.speaker_embeddings())]
    except Exception:  # noqa: BLE001
        return []


def _self_id(db: Any) -> int | None:
    try:
        me = db.self_speaker()
        return int(me["id"]) if me is not None else None
    except Exception:  # noqa: BLE001
        return None


def _negatives(db: Any, dim: int) -> dict[int, np.ndarray]:
    out: dict[int, list[np.ndarray]] = {}
    fn = getattr(db, "speaker_negatives", None)
    if not callable(fn):
        return {}
    try:
        for sid, v in fn():
            v = np.asarray(v, dtype=np.float32).ravel()
            if v.shape[0] == dim:
                out.setdefault(int(sid), []).append(unit(v))
    except Exception as e:  # noqa: BLE001 - a missing table on a foreign DB must not stop identification
        log.debug("no negative samples: %s", e)
        return {}
    return {sid: np.stack(vs) for sid, vs in out.items()}


def _cluster_rows(db: Any, dim: int, limit: int, min_s: float = 0.0) -> list[dict[str, Any]]:
    """The most recent stored cluster vectors with their recording condition and what the user said about them:
    {seg, cl, emb, speech_s, loudness_db, system_share, total_s (seconds of its sentences),
    labels: {speaker_id: seconds hand-labelled}, vetoes: {speaker_id: seconds vetoed}}."""
    try:
        rows = db.query(
            "SELECT segment_id, cluster_label, embedding, speech_s, loudness_db, system_share FROM cluster_embeddings"
            " WHERE cluster_label >= 0 AND dim=? AND speech_s >= ? ORDER BY segment_id DESC, cluster_label LIMIT ?",
            (int(dim), float(min_s), int(limit)))
    except Exception:  # noqa: BLE001 - stand-ins / databases older than the 1.0 columns
        try:
            rows = db.query(
                "SELECT segment_id, cluster_label, embedding, speech_s, NULL AS loudness_db, NULL AS system_share"
                " FROM cluster_embeddings WHERE cluster_label >= 0 AND dim=? AND speech_s >= ?"
                " ORDER BY segment_id DESC, cluster_label LIMIT ?", (int(dim), float(min_s), int(limit)))
        except Exception:  # noqa: BLE001
            return []
    out: list[dict[str, Any]] = []
    for r in rows:
        v = _blob_unit(r["embedding"], dim)
        if v is not None:
            out.append({"seg": int(r["segment_id"]), "cl": int(r["cluster_label"]), "emb": v, "speech_s": float(r["speech_s"] or 0.0),
                        "loudness_db": r["loudness_db"], "system_share": r["system_share"], "labels": {}, "vetoes": {},
                        "total_s": 0.0})
    if not out:
        return out
    by_key = {(c["seg"], c["cl"]): c for c in out}
    lo, hi = min(c["seg"] for c in out), max(c["seg"] for c in out)
    try:
        # condition of clusters stored before the 1.0 columns existed, and how much of each cluster the user labelled
        for r in db.query(
                "SELECT segment_id, cluster_label,"
                " SUM(end_s - start_s) AS total,"
                " SUM(CASE WHEN loudness_db IS NOT NULL THEN (end_s - start_s) * loudness_db END) AS lw,"
                " SUM(CASE WHEN loudness_db IS NOT NULL THEN (end_s - start_s) END) AS ld,"
                " SUM(CASE WHEN system_share IS NOT NULL THEN (end_s - start_s) * system_share END) AS sw,"
                " SUM(CASE WHEN system_share IS NOT NULL THEN (end_s - start_s) END) AS sd"
                " FROM utterances WHERE segment_id BETWEEN ? AND ? AND cluster_label IS NOT NULL"
                " GROUP BY segment_id, cluster_label", (lo, hi)):
            c = by_key.get((int(r["segment_id"]), int(r["cluster_label"])))
            if c is None:
                continue
            c["total_s"] = float(r["total"] or 0.0)
            if c["loudness_db"] is None and r["ld"]:
                c["loudness_db"] = float(r["lw"]) / float(r["ld"])
            if c["system_share"] is None and r["sd"]:
                c["system_share"] = float(r["sw"]) / float(r["sd"])
        for r in db.query(
                "SELECT segment_id, cluster_label, speaker_id, SUM(end_s - start_s) AS s FROM utterances"
                " WHERE speaker_source='user' AND speaker_id IS NOT NULL AND cluster_label IS NOT NULL"
                " AND segment_id BETWEEN ? AND ? GROUP BY segment_id, cluster_label, speaker_id", (lo, hi)):
            c = by_key.get((int(r["segment_id"]), int(r["cluster_label"])))
            if c is not None:
                c["labels"][int(r["speaker_id"])] = float(r["s"] or 0.0)
        for r in db.query(
                "SELECT u.segment_id, u.cluster_label, r.speaker_id, SUM(u.end_s - u.start_s) AS s FROM utterance_rejections r"
                " JOIN utterances u ON u.id = r.utterance_id WHERE u.cluster_label IS NOT NULL"
                " AND u.segment_id BETWEEN ? AND ? GROUP BY u.segment_id, u.cluster_label, r.speaker_id", (lo, hi)):
            c = by_key.get((int(r["segment_id"]), int(r["cluster_label"])))
            if c is not None:
                c["vetoes"][int(r["speaker_id"])] = float(r["s"] or 0.0)
    except Exception as e:  # noqa: BLE001 - the vectors alone still make a cohort
        log.debug("cluster facts unavailable: %s", e)
    return out


def _centroid_rows(db: Any, dim: int) -> dict[int, dict[str, np.ndarray]]:
    out: dict[int, dict[str, np.ndarray]] = {}
    try:
        rows = db.query("SELECT speaker_id, condition, embedding FROM speaker_centroids WHERE dim=?", (int(dim),))
    except Exception:  # noqa: BLE001
        return out
    for r in rows:
        v = _blob_unit(r["embedding"], dim)
        if v is not None:
            out.setdefault(int(r["speaker_id"]), {})[str(r["condition"])] = v
    return out


def _build_speakers(samples: list[dict[str, Any]], dim: int, p: Params) -> list[_Speaker]:
    per: dict[int, list[dict[str, Any]]] = {}
    for r in samples:
        if int(r["dim"] or 0) == dim:
            per.setdefault(int(r["speaker_id"]), []).append(r)
    out: list[_Speaker] = []
    for sid, rows in per.items():
        embs = [v for v in (_blob_unit(r["embedding"], dim) for r in rows)]
        keep = [k for k, v in enumerate(embs) if v is not None]
        if not keep:
            continue
        embs = [embs[k] for k in keep]
        rows = [rows[k] for k in keep]
        a = analyse_samples(embs, [r["duration_s"] for r in rows], p.sample_consistency_min, [r["origin"] for r in rows])
        core = sorted(a.core) or list(range(len(embs)))
        E = np.stack(embs)
        out.append(_Speaker(sid, str(rows[0]["name"]), unit(a.ref), E[core], E, outliers=len(embs) - len(core)))
    return out


def _told_not(c: dict[str, Any], sid: int) -> bool:
    """True when the user said this cluster is not speaker `sid`: most of it was hand-labelled as somebody else or
    vetoed for `sid` (one vetoed sentence in a long cluster says nothing about the rest)."""
    total = float(c.get("total_s") or 0.0) or float(c["speech_s"]) or 1e-9
    other = sum(sec for t, sec in c["labels"].items() if t != sid)
    return (other + float(c["vetoes"].get(sid, 0.0))) >= 0.5 * total


def _cohort(speakers: list[_Speaker], clusters: list[dict[str, Any]], p: Params
            ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """(vectors [m, d], masks [S, m], segment ids [m]): for every speaker the stored clusters that are certainly
    somebody else (`clusters` newest first; at most cohort_max per speaker).

    A cluster is an impostor of S when it is far from every model of S (cohort_max_raw) AND one of: far from
    everybody enrolled, mostly system audio, said by the user not to be S, clearly another enrolled speaker.
    Anything close to S stays out, even a voice the user vetoed: speech of S nobody confirmed yet must not leak
    in, and a handful of look-alikes among the closest impostors would move the whole scale (measured: a dozen
    of them halve every score of S). Look-alikes the user rejected act through the negative samples instead,
    which only touch vectors that resemble them."""
    rows = [c for c in clusters if c["speech_s"] >= p.cohort_min_s]
    if not rows or not speakers:
        return None, None, None
    V = np.stack([c["emb"] for c in rows])
    S = len(speakers)
    sim = np.empty((S, len(rows)), dtype=np.float32)
    for j, s in enumerate(speakers):
        sim[j] = V @ s.anchor
        for vec in s.centroids.values():
            sim[j] = np.maximum(sim[j], V @ vec)
    far_all = sim.max(axis=0) < p.cohort_max_raw
    system = np.array([c["system_share"] is not None and float(c["system_share"]) >= p.system_turn_share for c in rows])
    masks = np.zeros((S, len(rows)), dtype=bool)
    for j, s in enumerate(speakers):
        others = np.delete(sim, j, axis=0).max(axis=0) if S > 1 else np.full(len(rows), -1.0, dtype=np.float32)
        told = np.array([_told_not(c, s.sid) for c in rows])
        mine = np.array([s.sid in c["labels"] for c in rows])
        clearly_other = (others >= 0.5) & (others - sim[j] >= 0.15)
        m = (sim[j] < p.cohort_max_raw) & (far_all | system | told | clearly_other) & ~mine
        masks[j, np.flatnonzero(m)[: max(1, int(p.cohort_max))]] = True
    used = masks.any(axis=0)
    if not used.any():
        return None, None, None
    return V[used], masks[:, used], np.array([c["seg"] for c, u in zip(rows, used) if u], dtype=np.int64)


def _load_gallery(db: Any, cfg: Any, p: Params, centroids: bool = True,
                  override: dict[int, dict[str, np.ndarray]] | None = None,
                  clusters: list[dict[str, Any]] | None = None) -> Gallery:
    samples = _sample_rows(db)
    dims: dict[int, int] = {}
    for r in samples:
        dims[int(r["dim"] or 0)] = dims.get(int(r["dim"] or 0), 0) + 1
    dim = max(dims.items(), key=lambda kv: kv[1])[0] if dims else 0
    speakers = _build_speakers(samples, dim, p) if dim else []
    if speakers and p.scoring != "legacy":
        cents = override if override is not None else (_centroid_rows(db, dim) if (centroids and p.adapt) else {})
        negs = _negatives(db, dim)
        for s in speakers:
            s.centroids = dict(cents.get(s.sid, {}))
            s.negatives = negs.get(s.sid)
    elif speakers:
        negs = _negatives(db, dim)
        for s in speakers:
            s.negatives = negs.get(s.sid)
    cohort = masks = segs = None
    if speakers and p.scoring != "legacy":
        if clusters is None:
            clusters = _cluster_rows(db, dim, p.cohort_scan, p.cohort_min_s)
        cohort, masks, segs = _cohort(speakers, clusters, p)
    h = hashlib.sha1(p.key().encode("utf-8"))
    for r in samples:
        h.update(f"|{r['id']}:{r['speaker_id']}:{r['dim']}:{r['origin']}".encode("utf-8"))
    for s in speakers:
        if s.negatives is not None:
            h.update(f"|n{s.sid}:{len(s.negatives)}".encode("utf-8"))
            h.update(np.round(s.negatives, 3).astype(np.float32).tobytes())
    me = _self_id(db)
    h.update(f"|self:{me}".encode("utf-8"))
    return Gallery(p, speakers, self_id=me, dim=dim, cohort=cohort, cohort_masks=masks, cohort_segments=segs,
                   model=model_id(cfg), inputs=h.hexdigest())


# ================================================================================================ adapt
def _weighted_centroid(vectors: list[np.ndarray], weights: list[float], anchor: np.ndarray, anchor_weight: float) -> np.ndarray:
    """The evidence pulled towards the anchor: with little evidence the centroid stays near the enrolled voice,
    with a lot it is what the person really sounds like."""
    acc = anchor.astype(np.float64) * float(anchor_weight)
    for v, w in zip(vectors, weights):
        acc = acc + v.astype(np.float64) * float(w)
    return unit(acc)


def _evidence(g: Gallery, clusters: list[dict[str, Any]], p: Params) -> dict[int, list[tuple[np.ndarray, float, str, str]]]:
    """Per speaker: (vector, weight, condition, kind) of the stored clusters that are trusted to be that speaker.
    kind 'user': most of the cluster was hand-labelled as the speaker; 'auto': the gallery `g` is confident
    (score well above "sure", enough speech). Vetoed clusters, clusters the user gave to somebody else and
    clusters resembling the speaker's negative samples are never evidence."""
    out: dict[int, list[tuple[np.ndarray, float, str, str]]] = {s.sid: [] for s in g.speakers}
    if not clusters or not g.speakers:
        return out
    X = np.stack([c["emb"] for c in clusters])
    evs = g.evaluate_many(X, speech_s=[c["speech_s"] for c in clusters], loudness_db=[c["loudness_db"] for c in clusters],
                          system_share=[c["system_share"] for c in clusters], excludes=[set(c["vetoes"]) for c in clusters],
                          segments=[c["seg"] for c in clusters])
    extra = p.adapt_extra_norm if g.normalised else p.adapt_extra_raw
    for c, ev in zip(clusters, evs):
        cond = ev.decision.condition or "near"
        weight = min(float(c["speech_s"]), EVIDENCE_CAP_S)
        total = float(c.get("total_s") or 0.0) or float(c["speech_s"])
        labelled = {sid: s for sid, s in c["labels"].items() if sid in out}
        owner = max(labelled, key=lambda k: labelled[k]) if labelled else None
        if owner is not None:
            # the user's word counts when it covers most of the cluster and nobody else was named in it
            j = g.sids.index(owner)
            if len(c["labels"]) == 1 and labelled[owner] >= 0.5 * total and owner not in c["vetoes"] and ev.allowed[j] \
                    and not (cond == "system" and owner == g.self_id):
                out[owner].append((c["emb"], weight, cond, "user"))
            continue
        if c["labels"]:
            continue                                   # labelled as somebody who is no longer enrolled
        d = ev.decision
        if d.speaker_id is None or c["speech_s"] < p.adapt_min_s:
            continue
        if ev.value(d.speaker_id) < ev.sure_at + extra:
            continue
        out[d.speaker_id].append((c["emb"], weight, cond, "auto"))
    return out


def _outlier_seeds(db: Any, g: Gallery, p: Params) -> dict[int, list[tuple[np.ndarray, float, str, str]]]:
    """Samples outside the core that the user cut out of real recordings ('label' / 'voice'): they no longer
    recognise anything on their own, but they are a hint of how the person sounds far from the microphone."""
    out: dict[int, list[tuple[np.ndarray, float, str, str]]] = {}
    by_sid = {s.sid: s for s in g.speakers}
    for r in _sample_rows(db):
        s = by_sid.get(int(r["speaker_id"]))
        if s is None or r.get("origin") not in ("label", "voice"):
            continue
        v = _blob_unit(r["embedding"], g.dim)
        if v is None or float((s.core @ v).max()) >= 0.999:
            continue                                   # a core sample
        if float(s.anchor @ v) >= p.sample_consistency_min:
            continue                                   # consistent: already part of the anchor's neighbourhood
        out.setdefault(s.sid, []).append((v, min(float(r["duration_s"] or 3.0), EVIDENCE_CAP_S), "far", "sample"))
    return out


def _centroids_from(g: Gallery, evidence: dict[int, list[tuple[np.ndarray, float, str, str]]], p: Params,
                    conditions: bool) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict[str, dict[str, Any]]]]:
    cents: dict[int, dict[str, np.ndarray]] = {}
    report: dict[int, dict[str, dict[str, Any]]] = {}
    for s in g.speakers:
        ev = evidence.get(s.sid) or []
        for cond in ((ALL,) + CONDITIONS) if conditions else (ALL,):
            rows = ev if cond == ALL else [e for e in ev if e[2] == cond]
            rows = [e for e in rows if not (cond == "system" and s.sid == g.self_id)]
            info: dict[str, Any] = {"n": len(rows), "speech_s": round(sum(e[1] for e in rows), 1),
                                    "user": sum(1 for e in rows if e[3] == "user")}
            report.setdefault(s.sid, {})[cond] = info
            if len(rows) < max(1, int(p.adapt_min_clusters)):
                info["kept"] = False
                info["why"] = "too_few"
                continue
            c = _weighted_centroid([e[0] for e in rows], [e[1] for e in rows], s.anchor, p.adapt_anchor_weight_s)
            cos = float(c @ s.anchor)
            info["anchor_cos"] = round(cos, 3)
            if cos < p.adapt_min_anchor_cos:
                info["kept"] = False
                info["why"] = "drift"
                continue
            info["kept"] = True
            cents.setdefault(s.sid, {})[cond] = c
    return cents, report


def adapt(db: Any, cfg: Any, *, dry_run: bool = False, params: Params | None = None) -> dict[str, Any]:
    """Rebuild speaker_centroids from trusted evidence, deterministically from what is stored (never from earlier
    automatic decisions): pass 1 scores the stored cluster vectors against the anchors alone and makes an
    all-conditions centroid per speaker out of the confident ones plus what the user confirmed; pass 2 scores
    again with those and makes the per-condition centroids. Drift guard: a centroid further than
    speakers.adapt_min_anchor_cos from the anchor is refused; a condition needs speakers.adapt_min_clusters
    clusters. Returns {"speakers": {id: {condition: {n, speech_s, user, anchor_cos?, kept, why?}}}, "written",
    "changed", "dry_run"}."""
    p = params or Params.from_cfg(cfg)
    out: dict[str, Any] = {"speakers": {}, "written": 0, "changed": False, "dry_run": bool(dry_run), "enabled": bool(p.adapt)}
    if not p.adapt or p.scoring == "legacy":
        if not dry_run:
            try:
                n = db.one("SELECT COUNT(*) AS n FROM speaker_centroids")
                if n and int(n["n"]):
                    db.execute("DELETE FROM speaker_centroids")
                    out["changed"] = True
                    Gallery.forget(db)
            except Exception as e:  # noqa: BLE001
                log.debug("clearing centroids failed: %s", e)
        return out
    g0 = _load_gallery(db, cfg, p, centroids=False)
    if not g0.speakers:
        return out
    clusters = _cluster_rows(db, g0.dim, p.adapt_scan)
    seeds = _outlier_seeds(db, g0, p)

    def with_seeds(ev: dict[int, list]) -> dict[int, list]:
        for sid, rows in seeds.items():
            ev.setdefault(sid, []).extend(rows)
        return ev

    c1, _ = _centroids_from(g0, with_seeds(_evidence(g0, clusters, p)), p, conditions=False)
    g1 = _load_gallery(db, cfg, p, override=c1, clusters=[c for c in clusters if c["speech_s"] >= p.cohort_min_s][: p.cohort_scan])
    cents, report = _centroids_from(g1, with_seeds(_evidence(g1, clusters, p)), p, conditions=True)
    out["speakers"] = {str(sid): r for sid, r in report.items()}
    old = _centroid_rows(db, g0.dim)
    changed = set(old) != set(cents)
    for sid, per in cents.items():
        if set(per) != set(old.get(sid, {})):
            changed = True
            continue
        for cond, v in per.items():
            if float(v @ old[sid][cond]) < 0.999:
                changed = True
    out["changed"] = bool(changed)
    out["written"] = sum(len(v) for v in cents.values())
    if dry_run or not changed:
        return out
    now = datetime.now().isoformat(timespec="seconds")
    with db.transaction() as con:
        con.execute("DELETE FROM speaker_centroids WHERE dim=? OR dim IS NULL", (int(g0.dim),))
        for sid, per in cents.items():
            for cond, v in per.items():
                info = report[sid][cond]
                con.execute(
                    "INSERT INTO speaker_centroids(speaker_id, condition, embedding, dim, model, n, speech_s, updated_at)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (int(sid), cond, np.asarray(v, dtype=np.float32).tobytes(), int(g0.dim), g0.model, int(info["n"]),
                     float(info["speech_s"]), now))
    Gallery.forget(db)
    return out


# ================================================================================================ rescore
_ROW_COLS = ("id, segment_id, cluster_label, start_s, end_s, abs_start, day, speaker_id, speaker_name, speaker_score,"
             " speaker_source, loudness_db, system_share, session_id, speaker_guess_id, speaker_guess_score, speaker_meta")


def _ts(iso: Any) -> float:
    try:
        return datetime.fromisoformat(str(iso)).timestamp()
    except (TypeError, ValueError):
        return 0.0


def _state_of(speaker_id: Any, guess_id: Any) -> str:
    return "sure" if speaker_id is not None else "guess" if guess_id is not None else "unknown"


def _chunks(items: Sequence[Any], size: int = 500) -> Iterable[Sequence[Any]]:
    for k in range(0, len(items), size):
        yield items[k:k + size]


def _unknown_label(cfg: Any) -> str:
    try:
        return str(_section(cfg, "speakers").get("unknown_label") or "未知")
    except Exception:  # noqa: BLE001
        return "未知"


def _select_rows(db: Any, day: str | None, segment_ids: Iterable[int] | None, gap_s: float) -> tuple[list[Any], list[Any]]:
    """(the rows to re-decide, context rows): context = the other sentences close enough in time to share a
    conversation with a selected one; they take part in the tracks but are never written."""
    if day is None and segment_ids is None:
        return list(db.query(f"SELECT {_ROW_COLS} FROM utterances ORDER BY abs_start, id")), []
    sel: list[Any] = []
    if segment_ids is not None:
        ids = sorted({int(i) for i in segment_ids})
        for chunk in _chunks(ids):
            sel.extend(db.query(f"SELECT {_ROW_COLS} FROM utterances WHERE segment_id IN ({','.join('?' * len(chunk))})", chunk))
        if day is not None:
            sel = [r for r in sel if r["day"] == day]
    else:
        sel = list(db.query(f"SELECT {_ROW_COLS} FROM utterances WHERE day=?", (str(day),)))
    if not sel:
        return [], []
    sel.sort(key=lambda r: (str(r["abs_start"]), int(r["id"])))
    try:
        lo = (datetime.fromisoformat(str(sel[0]["abs_start"])) - timedelta(seconds=gap_s)).isoformat(timespec="milliseconds")
        hi = (datetime.fromisoformat(str(sel[-1]["abs_start"])) + timedelta(seconds=gap_s + 600)).isoformat(timespec="milliseconds")
    except (TypeError, ValueError):
        return sel, []
    have = {int(r["id"]) for r in sel}
    ctx = [r for r in db.query(f"SELECT {_ROW_COLS} FROM utterances WHERE abs_start >= ? AND abs_start <= ? ORDER BY abs_start, id", (lo, hi))
           if int(r["id"]) not in have]
    return sel, ctx


def _own_vectors(db: Any, ids: list[int], dim: int | None, model: str) -> dict[int, tuple[np.ndarray, float | None]]:
    """{utterance id: (unit vector, speech seconds)} from utterance_voice; `dim` None takes every dimension."""
    out: dict[int, tuple[np.ndarray, float | None]] = {}
    for chunk in _chunks(ids):
        try:
            rows = db.query("SELECT utterance_id, embedding, speech_s, model, dim FROM utterance_voice WHERE utterance_id IN"
                            f" ({','.join('?' * len(chunk))})", list(chunk))
        except Exception:  # noqa: BLE001 - databases / stand-ins without the table
            return out
        for r in rows:
            if dim is not None and (int(r["dim"] or 0) != int(dim) or r["model"] not in (None, "", model)):
                continue                               # a vector of another voice model cannot be compared
            v = _blob_unit(r["embedding"], dim)
            if v is not None:
                out[int(r["utterance_id"])] = (v, float(r["speech_s"]) if r["speech_s"] is not None else None)
    return out


def rescore(db: Any, cfg: Any, *, day: str | None = None, segment_ids: Iterable[int] | None = None, dry_run: bool = False,
            events: Any = None, stop_event: Any = None) -> dict[str, Any]:
    """Re-decide stored sentences from stored vectors: the sentence's own vector (utterance_voice) when there is
    one, else the vector of its diarization cluster. No audio is read and no model runs, so the whole history
    takes seconds. The rules are the ones of the first pass (Gallery + tracks.decide).

    Never changed: sentences the user labelled (speaker_source='user'); a sentence is never given to a speaker it
    was vetoed for, nor to the self speaker when it is system audio. Sentences without any stored vector and
    sentences below the loudness gate keep what they have (only an attribution they may not have is released).

    Returns {"utterances", "scored", "no_vector", "faint", "user", "before": {sure, guess, unknown},
    "after": {...}, "changed", "speaker_changed", "moves": {"unknown>sure": n, ...}, "days", "dry_run", "version",
    "normalised", "wall_s"}; with `dry_run` nothing is written."""
    from ..audio.loudness import is_faint
    from . import tracks as _tracks

    t_start = time.perf_counter()
    p = Params.from_cfg(cfg)
    g = Gallery.load(db, cfg, params=p)
    tp = _tracks.TrackParams.from_cfg(cfg)
    unknown = _unknown_label(cfg)
    out: dict[str, Any] = {"utterances": 0, "scored": 0, "no_vector": 0, "faint": 0, "user": 0,
                           "before": {"sure": 0, "guess": 0, "unknown": 0}, "after": {"sure": 0, "guess": 0, "unknown": 0},
                           "changed": 0, "speaker_changed": 0, "moves": {}, "days": [], "dry_run": bool(dry_run),
                           "version": g.version, "normalised": bool(g.normalised), "wall_s": 0.0}
    sel, ctx = _select_rows(db, day, segment_ids, tp.gap_s)
    out["utterances"] = len(sel)
    if not sel or (stop_event is not None and stop_event.is_set()):
        return out
    rows = sorted(list(sel) + list(ctx), key=lambda r: (str(r["abs_start"]), int(r["id"])))
    ids = [int(r["id"]) for r in rows]
    # With nobody enrolled there is no dimension to match: every stored vector then counts as "judged" (and nobody
    # is found), so attributions nothing vouches for any more are released like any other.
    dim = g.dim or None
    own = _own_vectors(db, ids, dim, g.model)
    clusters: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
    seg_ids = sorted({int(r["segment_id"]) for r in rows})
    for chunk in _chunks(seg_ids):
        for seg, cl, emb, speech_s in db.cluster_embeddings(list(chunk)):
            if emb.size == 0 or (dim is not None and emb.shape[0] != dim):
                continue
            v = unit(emb)
            if float(np.linalg.norm(v)) > 0:
                clusters[(seg, cl)] = (v, float(speech_s))
    vetoes = db.rejected_speakers(ids) if hasattr(db, "rejected_speakers") else {}
    me = g.self_id if g.self_id is not None else _self_id(db)

    # --- observations: one per sentence with its own vector, one per cluster for the rest
    obs: list[Any] = []
    obs_of: dict[int, int] = {}                 # utterance id -> index into obs
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    facts: dict[int, dict[str, Any]] = {}
    for r in rows:
        uid = int(r["id"])
        dur = max(0.0, float(r["end_s"]) - float(r["start_s"]))
        excl = set(vetoes.get(uid, ()))
        share = r["system_share"]
        if me is not None and share is not None and float(share) >= p.system_turn_share:
            excl.add(int(me))
        user = r["speaker_source"] == "user" and r["speaker_id"] is not None
        faint = is_faint(r["loudness_db"], p.min_speech_dbfs)
        t0 = _ts(r["abs_start"])
        key = (int(r["segment_id"]), int(r["cluster_label"])) if r["cluster_label"] is not None else None
        f = {"excl": frozenset(excl), "user": user, "faint": faint, "dur": dur, "t0": t0, "key": key, "vec": None}
        facts[uid] = f
        if faint and not user:
            continue                                   # too quiet to identify: neither decided nor evidence
        fixed = int(r["speaker_id"]) if user else None
        if uid in own:
            v, sp = own[uid]
            f["vec"] = "utt"
            obs_of[uid] = len(obs)
            obs.append(_tracks.Obs(v, t0, t0 + dur, sp if sp is not None else dur, r["loudness_db"], share, f["excl"],
                                   r["session_id"], int(r["segment_id"]), fixed, uid))
        elif key is not None and key in clusters:
            v, sp = clusters[key]
            f["vec"] = "cluster"
            if user:
                obs_of[uid] = len(obs)
                obs.append(_tracks.Obs(v, t0, t0 + dur, sp, r["loudness_db"], share, f["excl"], r["session_id"],
                                       int(r["segment_id"]), fixed, uid))
            else:
                gr = grouped.setdefault(key, {"rows": [], "t0": t0, "t1": t0 + dur, "session": r["session_id"]})
                gr["rows"].append(r)
                gr["t0"], gr["t1"] = min(gr["t0"], t0), max(gr["t1"], t0 + dur)
    for key, gr in grouped.items():
        v, sp = clusters[key]
        members = gr["rows"]
        w = [max(facts[int(m["id"])]["dur"], 1e-3) for m in members]
        lv = [(float(m["loudness_db"]), ww) for m, ww in zip(members, w) if m["loudness_db"] is not None]
        sh = [(float(m["system_share"]), ww) for m, ww in zip(members, w) if m["system_share"] is not None]
        loud = sum(a * b for a, b in lv) / sum(b for _, b in lv) if lv else None
        # a member without channel data is microphone audio: one system-audio sentence does not make the cluster one
        share = sum(a * b for a, b in sh) / sum(w) if sh else None
        common = frozenset.intersection(*[facts[int(m["id"])]["excl"] for m in members])
        for m in members:
            obs_of[int(m["id"])] = len(obs)
        obs.append(_tracks.Obs(v, gr["t0"], gr["t1"], sp, loud, share, common, gr["session"], key[0], None, key))
    decisions = _tracks.decide(obs, g, tp) if obs else []

    # --- new values per selected automatic row
    updates: list[tuple[Any, ...]] = []
    days: set[str] = set()
    moves: dict[str, int] = {}
    for r in sel:
        uid = int(r["id"])
        f = facts[uid]
        if f["user"]:
            out["user"] += 1
            continue
        old_sid = int(r["speaker_id"]) if r["speaker_id"] is not None else None
        old_gid = int(r["speaker_guess_id"]) if r["speaker_guess_id"] is not None else None
        before = _state_of(old_sid, old_gid)
        out["before"][before] += 1
        k = obs_of.get(uid)
        if k is None:
            out["faint" if f["faint"] else "no_vector"] += 1
            # nothing to decide with: keep it, except what it may not be (a veto, the self speaker on system audio)
            new_sid = None if (old_sid is not None and old_sid in f["excl"]) else old_sid
            new_gid = None if (old_gid is not None and (old_gid in f["excl"] or new_sid is not None)) else old_gid
            if new_sid == old_sid and new_gid == old_gid:
                out["after"][before] += 1
                continue
            new = (new_sid, r["speaker_name"] if new_sid is not None else unknown,
                   r["speaker_score"] if new_sid is not None else None, new_gid,
                   r["speaker_guess_score"] if new_gid is not None else None, r["speaker_meta"])
        else:
            out["scored"] += 1
            d = decisions[k]
            hit = d.speaker_id if d.speaker_id is not None else d.guess_id
            if f["vec"] == "cluster" and hit is not None and hit in f["excl"]:
                # the cluster as a whole may be X, this sentence may not: decide it again without X
                v, sp = clusters[f["key"]]
                d = g.score(v, speech_s=sp, loudness_db=r["loudness_db"], system_share=r["system_share"],
                            exclude=f["excl"], segment_id=int(r["segment_id"]))
            new_sid = d.speaker_id if (d.speaker_id is None or d.speaker_id not in f["excl"]) else None
            new_gid = d.guess_id if (new_sid is None and d.guess_id is not None and d.guess_id not in f["excl"]) else None
            meta = d.meta()
            meta["vec"] = f["vec"]
            new = (new_sid, (d.name or g.names.get(new_sid) or unknown) if new_sid is not None else unknown,
                   round(float(d.score), 4) if d.score is not None else None, new_gid,
                   round(float(d.guess_score), 4) if (new_gid is not None and d.guess_score is not None) else None,
                   json.dumps(meta, ensure_ascii=False, sort_keys=True))
        after = _state_of(new[0], new[3])
        out["after"][after] += 1
        same_speaker = new[0] == old_sid
        old_score = r["speaker_score"]
        unchanged = (same_speaker and new[3] == old_gid and (r["speaker_meta"] or None) == (new[5] or None)
                     and (old_score is None) == (new[2] is None)
                     and (old_score is None or abs(float(old_score) - float(new[2])) < 5e-4))
        if unchanged:
            continue
        if not same_speaker or new[3] != old_gid:
            out["changed"] += 1
            mk = "sure>sure(other)" if (before == after == "sure") else f"{before}>{after}"
            moves[mk] = moves.get(mk, 0) + 1
        if not same_speaker:
            out["speaker_changed"] += 1
            days.add(str(r["day"]))
        updates.append((*new, uid, new[0] if new[0] is not None else -1))
    out["moves"] = dict(sorted(moves.items()))
    out["days"] = sorted(days)
    if updates and not dry_run:
        with db.transaction() as con:
            for u in updates:
                # a request thread may have hand-labelled or vetoed the row meanwhile: the WHERE clause skips it
                con.execute(
                    "UPDATE utterances SET speaker_id=?, speaker_name=?, speaker_score=?, speaker_source=NULL,"
                    " speaker_guess_id=?, speaker_guess_score=?, speaker_meta=? WHERE id=?"
                    " AND (speaker_source IS NULL OR speaker_source<>'user' OR speaker_id IS NULL)"
                    " AND NOT EXISTS (SELECT 1 FROM utterance_rejections r WHERE r.utterance_id=utterances.id AND r.speaker_id=?)",
                    u)
        if days:
            try:
                from .pipeline import write_days_markdown

                write_days_markdown(db, cfg, days)
            except Exception as e:  # noqa: BLE001 - the DB is the source of truth
                log.warning("rescore: cannot rewrite markdown: %s", e)
        if events is not None and out["changed"]:
            try:
                events.publish("speakers", change="rescored", changed=int(out["changed"]), days=out["days"])
            except Exception as e:  # noqa: BLE001
                log.debug("rescore: event not published: %s", e)
    out["wall_s"] = round(time.perf_counter() - t_start, 3)
    if out["changed"]:
        log.info("rescore%s: %d sentences, %d changed (%s), sure %d -> %d, guess %d -> %d, unknown %d -> %d, %.2fs",
                 " (dry run)" if dry_run else "", out["utterances"], out["changed"], out["moves"],
                 out["before"]["sure"], out["after"]["sure"], out["before"]["guess"], out["after"]["guess"],
                 out["before"]["unknown"], out["after"]["unknown"], out["wall_s"])
    return out


# ================================================================================================ keeping history in step
def _adapt_inputs(db: Any, p: Params) -> str | None:
    """Fingerprint of what `adapt` learns from apart from the growing pile of cluster vectors: samples, negatives,
    hand labels, vetoes, the self speaker and the settings."""
    st = _stamp(db, p, centroids=False)
    if st is None:
        return None
    a, b, _c, _d, e, f, g, key = st
    return hashlib.sha1(repr((a, b, e, f, g, key)).encode("utf-8")).hexdigest()[:16]


def _brief(summary: dict[str, Any]) -> dict[str, Any]:
    keys = ("utterances", "scored", "no_vector", "faint", "user", "before", "after", "changed", "speaker_changed", "moves",
            "days", "dry_run", "version", "normalised", "wall_s")
    return {k: summary[k] for k in keys if k in summary}


def refresh(db: Any, cfg: Any, *, events: Any = None, stop_event: Any = None, force: bool = False,
            source: str = "chain") -> dict[str, Any]:
    """Bring the centroids and the stored decisions in step with the reference voices. Cheap when nothing changed.

    1. `adapt` runs when samples / negatives / hand labels / vetoes / settings changed since the last time, or when
       the stored cluster vectors grew by a tenth (new evidence).
    2. `rescore` of the whole history runs when the gallery version differs from the one of the last rescore
       (which includes the very first start after the upgrade). It leaves a job_runs row ('rescore') with the
       before / after counts.
    Returns {"adapted": bool, "rescored": bool, "version"?, "adapt"?, "rescore"?}."""
    out: dict[str, Any] = {"adapted": False, "rescored": False}
    p = Params.from_cfg(cfg)
    try:
        n_samples = int(db.one("SELECT COUNT(*) AS n FROM speaker_embeddings")["n"])
        n_clusters = int(db.one("SELECT COUNT(*) AS n FROM cluster_embeddings WHERE cluster_label >= 0")["n"])
        n_centroids = int(db.one("SELECT COUNT(*) AS n FROM speaker_centroids")["n"])
        last = db.get_kv(KV_RESCORE) or {}
        state = db.get_kv(KV_ADAPT) or {}
    except Exception as e:  # noqa: BLE001 - a foreign / half-built database: nothing to keep in step
        log.debug("refresh skipped: %s", e)
        return out
    if n_samples == 0 and n_centroids == 0 and not last:
        return out                                     # nobody enrolled, nothing ever decided
    if stop_event is not None and stop_event.is_set():
        return out
    inputs = _adapt_inputs(db, p)
    grown = n_clusters >= int(state.get("clusters") or 0) * 1.1 + 10
    if force or inputs != state.get("inputs") or grown:
        rep = adapt(db, cfg, params=p)
        db.set_kv(KV_ADAPT, {"inputs": inputs, "clusters": n_clusters, "at": datetime.now().isoformat(timespec="seconds")})
        out["adapted"] = bool(rep.get("changed"))
        out["adapt"] = {"written": rep.get("written", 0), "changed": bool(rep.get("changed"))}
    g = Gallery.load(db, cfg, params=p)
    out["version"] = g.version
    if not force and last.get("version") == g.version:
        return out
    if stop_event is not None and stop_event.is_set():
        return out
    started = datetime.now().isoformat(timespec="seconds")
    try:
        summary = rescore(db, cfg, events=events, stop_event=stop_event)
    except Exception as e:  # noqa: BLE001 - recorded, and tried again next time
        log.exception("rescore failed: %s", e)
        try:
            db.add_job_run("rescore", source, "failed", started, error=f"{type(e).__name__}: {e}"[:500])
        except Exception:  # noqa: BLE001
            pass
        raise
    db.set_kv(KV_RESCORE, {"version": g.version, "at": started, "summary": _brief(summary)})
    out["rescored"] = True
    out["rescore"] = _brief(summary)
    try:
        db.add_job_run("rescore", source, "done", started, result=_brief(summary))
    except Exception as e:  # noqa: BLE001 - the history row is a convenience
        log.debug("rescore: job_runs row not written: %s", e)
    return out


def run_pending(cfg: Any, db: Any, client: Any = None, *, stop_event: Any = None, events: Any = None,
                limit: int | None = None) -> dict[str, Any]:
    """Chain step (after `sessions`): `refresh`. Needs no LLM client; a no-op while nothing changed."""
    return refresh(db, cfg, events=events, stop_event=stop_event, source="chain")


def job_rescore(rt: Any, day: str | None = None, segment_ids: Any = None, segment_id: Any = None, dry_run: Any = False,
                **_: Any) -> dict[str, Any]:
    """The `rescore` job (重判说话人): rebuild the centroids, then re-decide the history (or one day / some
    segments) from stored vectors. {"dry_run": true} only reports what would change."""
    db, cfg = rt.db, rt.cfg
    dry = _coerce(dry_run, False)
    ids: list[int] | None = None
    raw = segment_ids if segment_ids is not None else ([segment_id] if segment_id is not None else None)
    if raw is not None:
        if isinstance(raw, (str, bytes)):
            raw = [x for x in str(raw).replace("，", ",").split(",") if x.strip()]
        try:
            ids = [int(x) for x in (raw if isinstance(raw, (list, tuple, set)) else [raw])]
        except (TypeError, ValueError):
            raise ValueError("segment_ids 应为片段 id 列表")
    day = str(day).strip() if day else None
    stop = None
    fn = getattr(rt, "job_stop_event", None)
    if callable(fn):
        stop = fn()
    events = getattr(rt, "events", None)
    out: dict[str, Any] = {}
    if not dry:
        rep = adapt(db, cfg)
        out["adapt"] = {"written": rep.get("written", 0), "changed": bool(rep.get("changed"))}
        try:
            n = int(db.one("SELECT COUNT(*) AS n FROM cluster_embeddings WHERE cluster_label >= 0")["n"])
            db.set_kv(KV_ADAPT, {"inputs": _adapt_inputs(db, Params.from_cfg(cfg)), "clusters": n,
                                 "at": datetime.now().isoformat(timespec="seconds")})
        except Exception as e:  # noqa: BLE001
            log.debug("adapt state not stored: %s", e)
    started = datetime.now().isoformat(timespec="seconds")
    summary = rescore(db, cfg, day=day, segment_ids=ids, dry_run=dry, events=events, stop_event=stop)
    out.update(_brief(summary))
    if not dry and day is None and ids is None:
        db.set_kv(KV_RESCORE, {"version": summary["version"], "at": started, "summary": _brief(summary)})
    return out


__all__ = ["ALL", "CONDITIONS", "Decision", "Evaluation", "Gallery", "KV_ADAPT", "KV_RESCORE", "NO_DECISION", "Params",
           "SampleAnalysis", "adapt", "analyse_samples", "condition_of", "job_rescore", "model_id", "negative_scores", "refresh",
           "rescore", "run_pending", "unit"]
