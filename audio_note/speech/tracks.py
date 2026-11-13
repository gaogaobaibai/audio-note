"""Voices over time: what the neighbours say about a sentence (SPEC_v1 section 5.1, rules 4 and 6). Pure numpy.

A single short sentence is weak evidence of who spoke, but a conversation is not a bag of unrelated sentences:
the same voice keeps coming back within a few minutes. This module pools that evidence.

    decide(observations, gallery)    -> one Decision per observation

1. *Blocks*: observations of one session (sessions.id, when the memory package has already cut sessions), else a
   time-gap rule: a pause longer than `gap_s` starts a new block. Nothing is ever linked across blocks.
2. *Tracks*: inside a block the voice vectors are linked into tracks (greedy centroid linkage, cosine >=
   speakers.track_link_cos). A track is one voice in one conversation.
3. Each track is scored as a whole (its pooled vector, its total speech) and the answer flows back to its members:
     - a member with enough speech of its own (>= speakers.short_s) keeps its own decision; the track only lifts
       it from "maybe X" to X when the track is sure of the same X;
     - a shorter member follows the track unless its own vector speaks against it;
     - a member under speakers.follow_s seconds only ever follows.
   A sentence the user labelled anchors its track: the track is that person as long as its pooled voice does not
   contradict it.
4. *Flips*: an isolated short sentence left undecided between two sentences of the same person A (close in time)
   is given to A when its own voice at least leans towards A ("A ? A" -> "A A A").

Vetoes (Obs.exclude), negative samples and the system-audio rule are enforced for every member separately: a
track never hands a sentence to somebody that sentence must not be.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Hashable, Sequence

import numpy as np

from .scoring import NO_DECISION, Decision, Evaluation, Gallery, Params

POOL_CAP_S = 10.0          # one member weighs at most this many seconds in its track's pooled vector
DEFAULT_GAP_S = 300.0      # pause that separates two conversations when no session ids are known
DEFAULT_LINK_COS = 0.55
DEFAULT_SMOOTH_GAP_S = 3.0
MAX_LINK = 400             # observations linked at once: a longer block is linked in consecutive stretches of this size


@dataclass
class Obs:
    """One voice observation: a sentence (its own vector) or a diarization cluster (the vector of all its turns)."""
    emb: np.ndarray | None
    t0: float                              # seconds on a clock shared by all observations (absolute time)
    t1: float
    speech_s: float
    loudness_db: float | None = None
    system_share: float | None = None
    exclude: frozenset[int] = frozenset()  # speakers this observation must never be (vetoes)
    session: Hashable | None = None        # sessions.id when known
    segment: Hashable | None = None        # kept out of the impostor statistics of its own vector
    fixed: int | None = None               # the user's label: never changed, anchors the track
    key: Any = None


@dataclass(frozen=True)
class TrackParams:
    gap_s: float = DEFAULT_GAP_S
    link_cos: float = DEFAULT_LINK_COS
    smooth: bool = True
    smooth_gap_s: float = DEFAULT_SMOOTH_GAP_S

    @classmethod
    def from_cfg(cls, cfg: Any, **overrides: Any) -> "TrackParams":
        def read(section: str, key: str, default: Any) -> Any:
            try:
                v = getattr(cfg, section).get(key, default)
            except Exception:  # noqa: BLE001 - config stand-ins without the section
                return default
            if v is None:
                return default
            try:
                return bool(v) if isinstance(default, bool) else float(v)
            except (TypeError, ValueError):
                return default

        gap = read("memory", "session_gap_min", DEFAULT_GAP_S / 60.0) * 60.0
        vals = {"gap_s": gap if gap > 0 else DEFAULT_GAP_S,
                "link_cos": read("speakers", "track_link_cos", DEFAULT_LINK_COS),
                "smooth": read("speakers", "smooth", True),
                "smooth_gap_s": read("speakers", "smooth_gap_s", DEFAULT_SMOOTH_GAP_S)}
        vals.update(overrides)
        return cls(**vals)


# ================================================================================================ grouping
def blocks(obs: Sequence[Obs], gap_s: float = DEFAULT_GAP_S) -> list[list[int]]:
    """Indices grouped into conversations, each in time order. Observations that carry a session id are grouped
    by it; the rest by the time-gap rule (a pause longer than `gap_s` between one observation's end and the
    next one's start begins a new block), so this works before any session exists."""
    by_session: dict[Hashable, list[int]] = {}
    loose: list[int] = []
    for i, o in enumerate(obs):
        if o.session is not None:
            by_session.setdefault(o.session, []).append(i)
        else:
            loose.append(i)
    out = [sorted(idx, key=lambda i: (obs[i].t0, obs[i].t1, i)) for idx in by_session.values()]
    loose.sort(key=lambda i: (obs[i].t0, obs[i].t1, i))
    cur: list[int] = []
    end = 0.0
    for i in loose:
        if cur and obs[i].t0 - end > float(gap_s):
            out.append(cur)
            cur = []
        end = obs[i].t1 if not cur else max(end, obs[i].t1)
        cur.append(i)
    if cur:
        out.append(cur)
    out.sort(key=lambda idx: obs[idx[0]].t0)
    return out


def link(obs: Sequence[Obs], idx: Sequence[int], link_cos: float = DEFAULT_LINK_COS) -> list[list[int]]:
    """Tracks among the observations `idx` of one block: greedy agglomerative linking of the voice vectors
    (centroid linkage, weights = speech seconds capped at POOL_CAP_S) while the closest pair of tracks is at
    least `link_cos` similar. Observations without a vector stay alone. Returns lists of indices into `obs`."""
    have = [i for i in idx if obs[i].emb is not None]
    alone = [[i] for i in idx if obs[i].emb is None]
    n = len(have)
    if n <= 1:
        return [[i] for i in have] + alone
    E = np.stack([_unit(obs[i].emb) for i in have])
    w = np.array([max(min(float(obs[i].speech_s or 0.0), POOL_CAP_S), 0.1) for i in have], dtype=np.float64)
    groups: list[list[int]] = [[k] for k in range(n)]
    cent = E.astype(np.float64).copy()
    active = np.ones(n, dtype=bool)
    sims = cent @ cent.T
    np.fill_diagonal(sims, -np.inf)
    while True:
        flat = int(np.argmax(sims))
        a, b = divmod(flat, n)
        if not np.isfinite(sims[a, b]) or sims[a, b] < float(link_cos):
            break
        groups[a].extend(groups[b])
        groups[b] = []
        active[b] = False
        sims[b, :] = -np.inf
        sims[:, b] = -np.inf
        members = groups[a]
        c = (E[members] * w[members][:, None]).sum(axis=0)
        cent[a] = c / (np.linalg.norm(c) or 1.0)
        row = cent @ cent[a]
        row[~active] = -np.inf
        row[a] = -np.inf
        sims[a, :] = row
        sims[:, a] = row
    return [[have[k] for k in g] for g in groups if g] + alone


def _unit(v: Any) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _pooled(obs: Sequence[Obs], members: Sequence[int]) -> tuple[np.ndarray, float, float | None, float | None]:
    """(pooled unit vector, total speech seconds, weighted loudness, weighted system share) of a track. A member
    without channel data counts as microphone audio (share 0) once any member has a share: one system-audio
    sentence must not turn a whole microphone track into "system audio"."""
    acc = None
    total = 0.0
    lw = ls = sw = ss = wt = 0.0
    for i in members:
        o = obs[i]
        w = max(min(float(o.speech_s or 0.0), POOL_CAP_S), 0.1)
        v = _unit(o.emb).astype(np.float64) * w
        acc = v if acc is None else acc + v
        total += float(o.speech_s or 0.0)
        wt += w
        if o.loudness_db is not None:
            lw += w * float(o.loudness_db)
            ls += w
        if o.system_share is not None:
            sw += w * float(o.system_share)
            ss += w
    return _unit(acc), total, (lw / ls if ls else None), (sw / wt if ss else None)


# ================================================================================================ decisions
def decide(obs: Sequence[Obs], gallery: Gallery, params: TrackParams | None = None, *, cfg: Any = None) -> list[Decision]:
    """One Decision per observation: its own evidence combined with its track's and its neighbours' (see the module
    docstring). Observations with `fixed` set get that speaker back (method 'user'); observations without a
    vector get NO_DECISION unless a flip gives them to their neighbours' speaker."""
    tp = params or (TrackParams.from_cfg(cfg) if cfg is not None else TrackParams())
    p: Params = gallery.params
    n = len(obs)
    out: list[Decision] = [NO_DECISION] * n
    if n == 0:
        return out
    have = [i for i in range(n) if obs[i].emb is not None and np.asarray(obs[i].emb).shape[0] == gallery.dim]
    evs: dict[int, Evaluation] = {}
    if have and gallery.has_speakers:
        res = gallery.evaluate_many(np.stack([np.asarray(obs[i].emb, dtype=np.float32) for i in have]),
                                    speech_s=[obs[i].speech_s for i in have], loudness_db=[obs[i].loudness_db for i in have],
                                    system_share=[obs[i].system_share for i in have], excludes=[obs[i].exclude for i in have],
                                    segments=[obs[i].segment for i in have])
        evs = dict(zip(have, res))
        for i in have:
            out[i] = evs[i].decision
    if not gallery.has_speakers:
        return out
    groups = blocks(obs, tp.gap_s)
    for block in groups:
        scored = [i for i in block if i in evs]
        # linking is cubic in the block size: an all-day "conversation" (hours without a pause) is linked in
        # stretches, which costs nothing in practice - a voice that matters comes back within minutes
        for k in range(0, len(scored), MAX_LINK):
            for members in link(obs, scored[k:k + MAX_LINK], tp.link_cos):
                if len(members) < 2:
                    continue
                _decide_track(obs, members, evs, gallery, p, tp, out)
    for i in range(n):
        if obs[i].fixed is not None:
            sid = int(obs[i].fixed)
            out[i] = Decision(sid, gallery.names.get(sid), None, None, None, None, "user", None,
                              out[i].condition if out[i] is not NO_DECISION else None)
    if tp.smooth:
        for block in groups:
            _flatten(obs, block, evs, gallery, p, tp, out)
    return out


def _decide_track(obs: Sequence[Obs], members: Sequence[int], evs: dict[int, Evaluation], gallery: Gallery, p: Params,
                  tp: TrackParams, out: list[Decision]) -> None:
    vec, total, loud, share = _pooled(obs, members)
    common = frozenset.intersection(*[frozenset(obs[i].exclude) for i in members])
    track_ev = gallery.evaluate(vec, speech_s=total, loudness_db=loud, system_share=share, exclude=common)
    track = track_ev.decision
    fixed = {int(obs[i].fixed) for i in members if obs[i].fixed is not None}
    if len(fixed) == 1:
        # the user named one sentence of this voice: the voice is that person unless its pooled vector disagrees
        sid = next(iter(fixed))
        if sid in gallery.names and track_ev.plausible(sid):
            j = track_ev.sids.index(sid)
            track = Decision(sid, gallery.names[sid], float(track_ev.raw[j]),
                             float(track_ev.norm[j]) if track_ev.norm is not None else None, None, None, "track", None,
                             track.condition)
    elif len(fixed) > 1:
        return                                          # two people named in one "voice": the link is not to be trusted
    sid = track.speaker_id
    gid = track.guess_id
    if sid is not None and not fixed and not gallery.normalised and track_ev.value(sid) < track_ev.sure_at + p.track_extra_raw:
        # Raw cosines (no cohort yet): the pooled vector of a look-alike is cleaner than any of its sentences and
        # creeps over the plain threshold (measured: a friend at 0.47 against a threshold of 0.45, which took 6 of
        # his 14 sentences with it). A track must clear the threshold by a margin before others may follow it.
        sid, gid = None, sid
    for i in members:
        if obs[i].fixed is not None:
            continue
        own = evs[i]
        d = own.decision
        dur = float(obs[i].speech_s or 0.0)
        if sid is not None:
            if d.speaker_id is not None:
                continue                                # its own vector decided (the same person, or clearly another)
            j = own._col(sid)
            if j is None or not own.allowed[j]:
                continue                                # vetoed / negative / system audio: never through the track
            if dur < p.follow_s or (dur < p.short_s and own.leans(sid)) or own.plausible(sid):
                out[i] = own.following(sid, track.name, "track", track.runner_up)
        elif gid is not None and d.speaker_id is None and d.guess_id is None:
            j = own._col(gid)
            if j is None or not own.allowed[j]:
                continue
            if dur < p.short_s or own.leans(gid):
                score = track.guess_score if track.guess_score is not None else track_ev.value(gid)
                out[i] = replace(d, guess_id=gid, guess_score=score, guess_name=track.guess_name or gallery.names.get(gid),
                                 method="track")


def _flatten(obs: Sequence[Obs], block: Sequence[int], evs: dict[int, Evaluation], gallery: Gallery, p: Params,
             tp: TrackParams, out: list[Decision]) -> None:
    """"A ? A": a short undecided sentence between two close sentences of the same person, whose own voice leans
    towards that person, is theirs. Decided sentences are never flipped."""
    order = sorted(block, key=lambda i: (obs[i].t0, obs[i].t1, i))
    for k in range(1, len(order) - 1):
        i, a, b = order[k], order[k - 1], order[k + 1]
        if obs[i].fixed is not None or out[i].speaker_id is not None:
            continue
        sid = out[a].speaker_id
        if sid is None or out[b].speaker_id != sid or out[a].method == "smooth":
            continue
        if float(obs[i].speech_s or 0.0) >= p.short_s:
            continue
        if obs[i].t0 - obs[a].t1 > tp.smooth_gap_s or obs[b].t0 - obs[i].t1 > tp.smooth_gap_s:
            continue
        if sid in obs[i].exclude:
            continue
        own = evs.get(i)
        if own is None:
            continue
        if not own.leans(sid):
            continue
        out[i] = own.following(sid, gallery.names.get(sid), "smooth")


__all__ = ["DEFAULT_GAP_S", "DEFAULT_LINK_COS", "Obs", "TrackParams", "blocks", "decide", "link"]
