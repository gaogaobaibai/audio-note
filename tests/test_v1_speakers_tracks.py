"""Speakers v5, voices over time (audio_note/speech/tracks.py): blocks, tracks, following, "A ? A" flips.

Pure numpy: a Gallery built from a database stand-in, observations made by hand.
"""
from __future__ import annotations

import math

import numpy as np

from audio_note.speech import tracks
from audio_note.speech.scoring import Gallery
from audio_note.speech.tracks import Obs, TrackParams

DIM = 64


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _voice(seed: int) -> np.ndarray:
    return _unit(np.random.default_rng(seed).standard_normal(DIM))


def _at(base: np.ndarray, seed: int, cos: float) -> np.ndarray:
    o = np.random.default_rng(seed).standard_normal(base.shape[0])
    o = o - float(np.dot(o, base)) * base
    o = o / np.linalg.norm(o)
    return _unit(cos * base + math.sqrt(max(0.0, 1.0 - cos * cos)) * o)


def _mix(base: np.ndarray, like: np.ndarray, seed: int, cos_base: float, along: float) -> np.ndarray:
    """A unit vector with cosine `cos_base` to `base` that also carries `along` of the part of `like` that is not
    `base`: a voice that sounds like `like` (same track) while being as close to `base` as the test needs."""
    u = like - float(like @ base) * base
    u = u / np.linalg.norm(u)
    w = np.random.default_rng(seed).standard_normal(base.shape[0])
    w = w - float(w @ base) * base - float(w @ u) * u
    w = w / np.linalg.norm(w)
    return _unit(cos_base * base + along * u + math.sqrt(max(0.0, 1.0 - cos_base ** 2 - along ** 2)) * w)


A, B = _voice(1), _voice(2)
JIA, YI = 1, 2


class FakeDB:
    def __init__(self, me: int | None = None):
        self.me = me

    def speaker_embeddings(self):
        return [(JIA, "甲", A), (YI, "乙", B)]

    def self_speaker(self):
        return {"id": self.me} if self.me is not None else None


def gallery(me: int | None = None, **speakers) -> Gallery:
    return Gallery.load(FakeDB(me), {"speakers": speakers})


def obs(emb, t0: float, dur: float, **kw) -> Obs:
    return Obs(emb, t0, t0 + dur, kw.pop("speech_s", dur), **kw)


# ================================================================================================ grouping
def test_blocks_use_session_ids_and_fall_back_to_the_time_gap():
    o = [obs(A, 0, 2, session=7), obs(A, 1000, 2, session=7),            # one session, however far apart
         obs(A, 10, 2), obs(A, 100, 2), obs(A, 500, 2),                  # no session: 300 s rule -> [10, 100] and [500]
         obs(A, 20, 2, session=8)]
    got = tracks.blocks(o, gap_s=300.0)
    assert sorted(sorted(b) for b in got) == [[0, 1], [2, 3], [4], [5]]
    assert tracks.blocks([], 300.0) == []
    # the gap is measured from the END of what was said so far
    long_ = [obs(A, 0, 400), obs(A, 500, 2), obs(A, 900, 2)]
    assert tracks.blocks(long_, gap_s=300.0) == [[0, 1], [2]]
    assert tracks.blocks(long_, gap_s=50.0) == [[0], [1], [2]]
    # blocks come back in time order, members too
    shuffled = [obs(A, 50, 1), obs(A, 0, 1), obs(A, 5000, 1)]
    assert tracks.blocks(shuffled, gap_s=300.0) == [[1, 0], [2]]


def test_link_groups_one_voice_and_keeps_strangers_apart():
    o = [obs(_at(A, 10, 0.9), 0, 3), obs(_at(B, 11, 0.9), 4, 3), obs(_at(A, 12, 0.9), 8, 3), obs(None, 12, 1),
         obs(_at(A, 13, 0.85), 14, 0.8)]
    got = sorted(sorted(g) for g in tracks.link(o, range(len(o)), link_cos=0.55))
    assert got == [[0, 2, 4], [1], [3]]                                  # the one without a vector stays alone
    assert sorted(sorted(g) for g in tracks.link(o, range(len(o)), link_cos=0.99)) == [[0], [1], [2], [3], [4]]
    assert tracks.link(o, [1], 0.55) == [[1]] and tracks.link(o, [], 0.55) == []


def test_track_params_come_from_config():
    class Cfg:
        memory = {"session_gap_min": 2}
        speakers = {"track_link_cos": 0.7, "smooth": False, "smooth_gap_s": "x"}

    tp = TrackParams.from_cfg(Cfg())
    assert (tp.gap_s, tp.link_cos, tp.smooth, tp.smooth_gap_s) == (120.0, 0.7, False, 3.0)
    assert TrackParams.from_cfg(object()) == TrackParams()
    assert TrackParams.from_cfg(object(), link_cos=0.9).link_cos == 0.9


# ================================================================================================ decisions
def test_short_sentences_follow_their_track():
    g = gallery()
    strong = _at(A, 20, 0.8)
    weak_short = _at(strong, 21, 0.75)                                   # same voice, little speech: raw to 甲 about 0.6
    tiny = _at(strong, 22, 0.70)
    o = [obs(strong, 0, 4), obs(weak_short, 5, 1.5), obs(tiny, 7, 0.6), obs(_at(B, 23, 0.8), 9, 4)]
    alone = [g.score(x.emb, speech_s=x.speech_s) for x in o]
    assert alone[0].speaker_id == JIA and alone[2].speaker_id is None    # under follow_s a vector never decides alone
    ds = tracks.decide(o, g, TrackParams())
    assert [d.speaker_id for d in ds] == [JIA, JIA, JIA, YI]
    assert ds[0].method == "anchor" and ds[2].method == "track" and ds[2].name == "甲"
    assert ds[2].score is not None and abs(ds[2].score - float(tiny @ A)) < 1e-4       # the score is the sentence's own
    # nothing is linked across conversations
    apart = [obs(strong, 0, 4), obs(tiny, 5000, 0.6)]
    assert [d.speaker_id for d in tracks.decide(apart, g, TrackParams())] == [JIA, None]


def test_a_long_sentence_keeps_its_own_answer_and_a_maybe_is_lifted():
    g = gallery()
    strong = _at(A, 30, 0.85)
    maybe = _mix(A, strong, 31, 0.40, 0.7)                               # raw to 甲 0.40: "maybe" on its own
    unlike = _mix(A, strong, 32, 0.02, 0.7)                              # same track, but its own voice is far from 甲
    o = [obs(strong, 0, 6), obs(maybe, 7, 3.0), obs(unlike, 11, 3.0)]
    own = [g.evaluate(x.emb, speech_s=x.speech_s) for x in o]
    assert own[1].decision.guess_id == JIA and own[1].decision.speaker_id is None
    assert not own[2].plausible(JIA)
    ds = tracks.decide(o, g, TrackParams(link_cos=0.3))
    assert ds[1].speaker_id == JIA and ds[1].method == "track"           # the track is sure, the sentence leaned that way
    assert ds[2].speaker_id is None                                      # long enough to speak for itself: it says no


def test_a_track_never_overrides_vetoes_or_the_system_audio_rule():
    g = gallery(me=JIA)
    strong = _at(A, 40, 0.85)
    tiny = _at(strong, 41, 0.8)
    o = [obs(strong, 0, 5), obs(tiny, 6, 0.6, exclude=frozenset({JIA})),          # vetoed for 甲
         obs(_at(strong, 42, 0.8), 8, 0.6, system_share=0.9),                     # system audio: never the user
         obs(_at(strong, 43, 0.8), 10, 0.6)]
    ds = tracks.decide(o, g, TrackParams())
    assert [d.speaker_id for d in ds] == [JIA, None, None, JIA]
    assert ds[1].guess_id is None and ds[2].guess_id is None


def test_a_hand_label_anchors_its_track_and_is_never_changed():
    g = gallery()
    quiet = _at(A, 50, 0.42)                                             # every sentence alone: only "maybe 甲"
    o = [obs(_at(quiet, 51, 0.95), 0, 3.0, fixed=JIA), obs(_at(quiet, 52, 0.95), 4, 3.0), obs(_at(quiet, 53, 0.95), 8, 0.7)]
    assert g.score(o[1].emb, speech_s=3.0).speaker_id is None
    ds = tracks.decide(o, g, TrackParams())
    assert [d.speaker_id for d in ds] == [JIA, JIA, JIA] and ds[0].method == "user" and ds[1].method == "track"
    # a label the voice contradicts anchors nothing; two different labels in one voice neither
    wrong = [obs(_at(quiet, 54, 0.95), 0, 3.0, fixed=YI), obs(_at(quiet, 55, 0.95), 4, 3.0)]
    ds = tracks.decide(wrong, g, TrackParams())
    assert ds[0].speaker_id == YI and ds[0].method == "user" and ds[1].speaker_id is None
    both = [obs(_at(quiet, 56, 0.95), 0, 3.0, fixed=YI), obs(_at(quiet, 57, 0.95), 4, 3.0, fixed=JIA), obs(_at(quiet, 58, 0.95), 8, 3.0)]
    assert tracks.decide(both, g, TrackParams())[2].speaker_id is None


def test_a_maybe_spreads_over_its_track():
    g = gallery()
    quiet = _at(A, 60, 0.42)
    stray = _mix(A, quiet, 62, 0.25, 0.8)                                # nothing on its own, but the same voice
    o = [obs(_at(quiet, 61, 0.97), 0, 4.0), obs(stray, 5, 0.6)]
    ds = tracks.decide(o, g, TrackParams())
    assert ds[0].speaker_id is None and ds[0].guess_id == JIA
    assert ds[1].speaker_id is None and ds[1].guess_id == JIA and ds[1].method == "track"


def test_flips_between_two_sentences_of_the_same_person_are_smoothed():
    g = gallery()
    a1, a2 = _at(A, 70, 0.8), _at(A, 71, 0.8)
    leaning = _at(A, 72, 0.36)                                           # a short "hm": leans to 甲, linked to nobody
    hostile = _voice(73)
    seq = [obs(a1, 0, 4), obs(leaning, 4.5, 0.9), obs(a2, 6, 4)]
    off = tracks.decide(seq, g, TrackParams(link_cos=2.0, smooth=False))
    assert [d.speaker_id for d in off] == [JIA, None, JIA]
    on = tracks.decide(seq, g, TrackParams(link_cos=2.0))
    assert [d.speaker_id for d in on] == [JIA, JIA, JIA] and on[1].method == "smooth"
    # not when the neighbours are far away in time, differ, or the sentence does not lean that way / is long / is vetoed
    late = [obs(a1, 0, 4), obs(leaning, 10, 0.9), obs(a2, 12, 4)]
    assert tracks.decide(late, g, TrackParams(link_cos=2.0))[1].speaker_id is None
    mixed = [obs(a1, 0, 4), obs(leaning, 4.5, 0.9), obs(_at(B, 74, 0.8), 6, 4)]
    assert tracks.decide(mixed, g, TrackParams(link_cos=2.0))[1].speaker_id is None
    other = [obs(a1, 0, 4), obs(hostile, 4.5, 0.9), obs(a2, 6, 4)]
    assert tracks.decide(other, g, TrackParams(link_cos=2.0))[1].speaker_id is None
    long_one = [obs(a1, 0, 4), obs(leaning, 4.5, 2.5), obs(a2, 7.5, 4)]
    assert tracks.decide(long_one, g, TrackParams(link_cos=2.0))[1].speaker_id is None
    vetoed = [obs(a1, 0, 4), obs(leaning, 4.5, 0.9, exclude=frozenset({JIA})), obs(a2, 6, 4)]
    assert tracks.decide(vetoed, g, TrackParams(link_cos=2.0))[1].speaker_id is None


def test_decide_without_speakers_or_vectors():
    empty = Gallery.load(object(), {})
    ds = tracks.decide([obs(A, 0, 3), obs(None, 4, 1)], empty)
    assert [d.state for d in ds] == ["unknown", "unknown"]
    assert tracks.decide([], gallery()) == []
    ds = tracks.decide([obs(None, 0, 1), obs(np.ones(3, dtype=np.float32), 2, 3)], gallery())      # no / foreign vectors
    assert [d.state for d in ds] == ["unknown", "unknown"]


def test_without_a_cohort_a_track_must_clear_the_threshold_by_a_margin():
    """Raw cosines: the pooled vector of a look-alike is cleaner than its sentences and creeps over the threshold."""
    look = _at(A, 80, 0.47)
    members = [_mix(A, look, 81 + k, 0.42, 0.8) for k in range(4)]          # each alone: 0.42 to 甲, "maybe"
    o = [obs(m, 5.0 * k, 4.0) for k, m in enumerate(members[:3])] + [obs(members[3], 15.0, 0.6)]
    g = gallery()
    pooled = _unit(sum(members[:3]) * 4.0 + members[3] * 0.6)
    assert 0.45 <= float(pooled @ A) < 0.55                                 # over the plain threshold, under the margin
    ds = tracks.decide(o, g, TrackParams())
    assert [d.speaker_id for d in ds] == [None] * 4 and all(d.guess_id == JIA for d in ds)
    # the same data with the margin switched off shows what it prevents
    loose = tracks.decide(o, gallery(track_extra_raw=0.0), TrackParams())
    assert loose[3].speaker_id == JIA and loose[3].method == "track"
    # a real track of 甲 is far above the margin and still carries its short sentences
    real = [obs(_at(A, 90, 0.8), 0, 4.0), obs(_mix(A, _at(A, 90, 0.8), 91, 0.5, 0.5), 5, 0.6)]
    assert [d.speaker_id for d in tracks.decide(real, g, TrackParams())] == [JIA, JIA]


def test_a_very_long_block_is_linked_in_stretches(monkeypatch):
    """Linking is cubic in the block size: hours without a pause must not stall a rescore."""
    monkeypatch.setattr(tracks, "MAX_LINK", 4)
    seen: list[int] = []
    real = tracks.link

    def spy(o, idx, link_cos=0.55):
        seen.append(len(list(idx)))
        return real(o, idx, link_cos)

    monkeypatch.setattr(tracks, "link", spy)
    strong = _at(A, 100, 0.85)
    o = [obs(_at(strong, 101 + k, 0.9), 2.0 * k, 0.6 if k % 2 else 3.0) for k in range(10)]     # one block of ten
    ds = tracks.decide(o, gallery(), TrackParams())
    assert seen == [4, 4, 2] and all(d.speaker_id == JIA for d in ds)
