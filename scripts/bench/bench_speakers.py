"""Speaker identification quality: what the pipeline stored, what a scoring rule WOULD decide, and a proxy on
real data that has no ground truth.

    # 1. what the pipeline decided (bench profile + truth)
    python scripts/bench/bench_speakers.py profile --profile baseline --root R --out spk.json [--compare old.json]

    # 2. replay stored embeddings through scoring rules, no model is run (cluster_embeddings; utterance_voice when
    #    the database has it; or "lines": one embedding per truth line and per 1 / 2 / 4 s crop, computed once)
    python scripts/bench/bench_speakers.py replay --profile baseline --root R --scorers legacy,centroid,margin,asnorm
    python scripts/bench/bench_speakers.py replay --profile baseline --source lines --scorers legacy --scorer mypkg.mod:fn
    #    the 1.0 scoring layer step by step, as a rescore of everything and as a first pass, plus the second-model cascade
    python scripts/bench/bench_speakers.py replay --source lines --root R \
        --scorers legacy,v5_centroid,v5_adapt,v5_norm,v5_duration,v5,v5_first,v5_cascade \
        --cascade-model 3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx

    # 3. real-data proxy: a database COPY without truth; prints aggregates only (no names, no text)
    python scripts/bench/bench_speakers.py proxy --db copy.sqlite3 --scorers legacy,centroid
    #    spec_gallery = what the 1.0 scoring layer says about every stored vector on its own; v5_db = what the
    #    database looks like after adapt + rescore (constraints and hand labels enforced)
    python scripts/bench/bench_speakers.py proxy --db copy.sqlite3 --scorers legacy,asnorm,spec_gallery,v5_db

Vocabulary. A *trial* is one decision: an utterance (profile mode) or one embedding (replay). Its truth is the
speaker of the story line it overlaps most; speakers that are not enrolled (the narrator, the friend) are
*non-targets* and must come out as unknown.
    accuracy           decisions that are exactly right (enrolled -> that person, not enrolled -> unknown)
    FRR                enrolled speaker NOT recognised as themselves (= rejected as unknown + confused with somebody)
    FAR                non-target accepted as some enrolled person;  unknown rejection = 1 - FAR
    per speaker S      FRR_S: S's trials not labelled S;  FAR_S: other people's trials labelled S
Numbers are given per trial and weighted by seconds, by duration bucket and by acoustic condition.

A scorer is ``fn(trials, gallery, **params) -> [decision]`` with one decision per trial:
    {"name": decided speaker or None, "best": best candidate or None, "score": that candidate's score or None,
     "guess": optional "maybe" name}
It may read trial["emb"], ["speech_s"], ["loudness_db"], ["system_share"], ["segment"], ["t"] - never the truth
fields. ``gallery`` has .names {id: name}, .samples [(speaker_id, unit vector, origin, seconds)], .negatives
[(speaker_id, unit vector)], .self_id and .db_path. "best"/"score" let the bench sweep a threshold without
calling the scorer again. Load your own with --scorer module:function or --scorer path/to/file.py:function.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

try:
    from . import align, audiofx, bench_cer, common
except ImportError:  # run as a script
    import align
    import audiofx
    import bench_cer
    import common

BUCKETS: tuple[tuple[float, float, str], ...] = ((0, 1, "<1s"), (1, 2, "1-2s"), (2, 4, "2-4s"), (4, 8, "4-8s"), (8, math.inf, ">=8s"))
CONDITIONS = ("near", "far", "overlap", "system")
MIXED_MIN_S = 0.5            # an utterance is "mixed" when a second speaker's line overlaps it this long ...
MIXED_MIN_SHARE = 0.25       # ... or for this share of its duration
DEFAULT_THRESHOLD = 0.45
DEFAULT_SYSTEM_SHARE = 0.6
SWEEP = tuple(round(0.20 + 0.05 * k, 2) for k in range(13))      # 0.20 .. 0.80


def bucket_of(seconds: float | None) -> str:
    s = float(seconds or 0.0)
    for lo, hi, name in BUCKETS:
        if lo <= s < hi:
            return name
    return BUCKETS[-1][2]


# ================================================================================================ tallies (pure)
class Tally:
    """Counts for a set of trials, per trial and weighted by seconds."""

    __slots__ = ("c", "w")
    KEYS = ("n", "target", "correct", "rejected", "confused", "nontarget", "accepted")

    def __init__(self) -> None:
        self.c = dict.fromkeys(self.KEYS, 0)
        self.w = dict.fromkeys(self.KEYS, 0.0)

    def add(self, expected: str | None, got: str | None, seconds: float = 0.0) -> None:
        def bump(k: str) -> None:
            self.c[k] += 1
            self.w[k] += float(seconds or 0.0)

        bump("n")
        if expected is not None:
            bump("target")
            bump("correct" if got == expected else "rejected" if got is None else "confused")
        else:
            bump("nontarget")
            if got is not None:
                bump("accepted")

    @staticmethod
    def _rates(c: dict[str, float]) -> dict[str, Any]:
        right = c["correct"] + (c["nontarget"] - c["accepted"])
        return {"accuracy": common.rate(right, c["n"]),
                "frr": common.rate(c["rejected"] + c["confused"], c["target"]),
                "reject_rate": common.rate(c["rejected"], c["target"]),
                "confusion_rate": common.rate(c["confused"], c["target"]),
                "far": common.rate(c["accepted"], c["nontarget"]),
                "unknown_rejection": common.rate(c["nontarget"] - c["accepted"], c["nontarget"])}

    def result(self) -> dict[str, Any]:
        out: dict[str, Any] = dict(self.c)
        out.update(self._rates(self.c))
        out["seconds"] = round(self.w["n"], 1)
        out["by_time"] = self._rates(self.w)
        return out


def evaluate(trials: Sequence[dict[str, Any]], decided: Sequence[str | None], enrolled: Sequence[str]) -> dict[str, Any]:
    """Score decisions against the truth carried by the trials (pure). `decided[k]` is a speaker name or None."""
    enrolled_set = set(enrolled)
    overall, pure = Tally(), Tally()
    by_cond: dict[str, Tally] = {}
    by_dur: dict[str, Tally] = {}
    by_session: dict[str, Tally] = {}
    per_target: dict[str, list[int]] = {s: [0, 0] for s in enrolled}        # [trials of S, labelled S]
    per_other: dict[str, list[int]] = {s: [0, 0] for s in enrolled}         # [trials of others, labelled S]
    unknown_by_speaker: dict[str, list[int]] = {}
    confusion: dict[str, dict[str, int]] = {}
    for t, got in zip(trials, decided):
        truth = t["speaker"]
        expected = truth if truth in enrolled_set else None
        got = got if got in enrolled_set else None
        sec = float(t.get("dur") or 0.0)
        overall.add(expected, got, sec)
        if not t.get("mixed"):
            pure.add(expected, got, sec)
        by_cond.setdefault(str(t.get("condition")), Tally()).add(expected, got, sec)
        by_dur.setdefault(bucket_of(sec), Tally()).add(expected, got, sec)
        if t.get("session") is not None:
            by_session.setdefault(str(t["session"]), Tally()).add(expected, got, sec)
        for s in enrolled:
            if truth == s:
                per_target[s][0] += 1
                per_target[s][1] += got == s
            else:
                per_other[s][0] += 1
                per_other[s][1] += got == s
        if expected is None:
            u = unknown_by_speaker.setdefault(str(truth), [0, 0])
            u[0] += 1
            u[1] += got is None
        row = confusion.setdefault(str(truth), {})
        key = got if got is not None else "(unknown)"
        row[key] = row.get(key, 0) + 1
    order = {name: k for k, (_lo, _hi, name) in enumerate(BUCKETS)}
    return {
        "overall": overall.result(), "unmixed_only": pure.result(),
        "by_condition": {k: v.result() for k, v in sorted(by_cond.items())},
        "by_duration": {k: v.result() for k, v in sorted(by_dur.items(), key=lambda kv: order.get(kv[0], 99))},
        "by_session": {k: v.result() for k, v in sorted(by_session.items())},
        "per_speaker": {s: {"trials": per_target[s][0], "frr": common.rate(per_target[s][0] - per_target[s][1], per_target[s][0]),
                            "other_trials": per_other[s][0], "far": common.rate(per_other[s][1], per_other[s][0])}
                        for s in enrolled},
        "unknown_speakers": {s: {"trials": v[0], "rejected": v[1], "rejection_rate": common.rate(v[1], v[0])}
                             for s, v in sorted(unknown_by_speaker.items())},
        "confusion": confusion,
    }


def headline(ev: dict[str, Any]) -> dict[str, Any]:
    o = ev["overall"]
    out = {"accuracy": o["accuracy"], "frr": o["frr"], "far": o["far"], "unknown_rejection": o["unknown_rejection"],
           "confusion_rate": o["confusion_rate"], "trials": o["n"]}
    for c in CONDITIONS:
        b = ev["by_condition"].get(c)
        if b:
            out[f"accuracy_{c}"] = b["accuracy"]
            out[f"frr_{c}"] = b["frr"]
            out[f"far_{c}"] = b["far"]
    return out


# ================================================================================================ truth for utterances
def utterance_trials(truth: dict[str, Any], utts_by_session: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One trial per stored utterance with the truth of the line it overlaps most. Also returns coverage facts:
    per condition how much of the truth lines' time was transcribed at all (far lines may be dropped as faint)."""
    trials: list[dict[str, Any]] = []
    stray = 0
    cover: dict[str, list[float]] = {}
    for s in truth["sessions"]:
        lines = sorted(s["lines"], key=lambda ln: (ln["t0"], ln["i"]))
        spans = [(float(ln["t0"]), float(ln["t1"])) for ln in lines]
        utts = utts_by_session.get(s["id"], [])
        uspans = [(float(u["u0"]), float(u["u1"])) for u in utts]
        for ln, cov in zip(lines, align.line_coverage(uspans, spans)):
            c = cover.setdefault(ln["condition"], [0.0, 0.0, 0, 0])
            d = float(ln["t1"]) - float(ln["t0"])
            c[0] += d
            c[1] += d * cov
            c[2] += 1
            c[3] += cov < 0.1
        for u, a in zip(utts, align.assign_lines(uspans, spans)):
            if a["line"] is None:
                stray += 1
                continue
            ln = lines[a["line"]]
            dur = float(u["u1"]) - float(u["u0"])
            mixed = any(lines[k]["speaker"] != ln["speaker"] and (ov >= MIXED_MIN_S or ov >= MIXED_MIN_SHARE * dur) for k, ov in a["others"])
            trials.append({"id": u.get("id"), "session": s["id"], "line": ln["i"], "speaker": ln["speaker"], "condition": ln["condition"],
                           "dur": dur, "mixed": mixed, "t": float(u["u0"]), "segment": u.get("segment_id"),
                           "cluster": u.get("cluster_label"), "loudness_db": u.get("loudness_db"), "system_share": u.get("system_share"),
                           "stored_name": u.get("speaker_name"), "stored_id": u.get("speaker_id"), "stored_score": u.get("speaker_score"),
                           "guess_id": u.get("speaker_guess_id")})
    coverage = {k: {"lines": int(v[2]), "missed_lines": int(v[3]), "seconds": round(v[0], 1), "covered_share": common.rate(v[1], v[0])}
                for k, v in sorted(cover.items())}
    return trials, {"stray_utterances": stray, "line_coverage": coverage}


# ================================================================================================ gallery + embeddings
def _vec(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    v = np.frombuffer(bytes(blob), dtype=np.float32).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else None


@dataclass
class Gallery:
    names: dict[int, str]
    samples: list[tuple[int, np.ndarray, str | None, float | None]]
    negatives: list[tuple[int, np.ndarray]] = field(default_factory=list)
    self_id: int | None = None
    db_path: str | None = None
    samples2: list[tuple[int, np.ndarray, str | None, float | None]] = field(default_factory=list)   # a second model's

    @property
    def speaker_ids(self) -> list[int]:
        return sorted({sid for sid, *_ in self.samples})

    def matrix(self, dim: int) -> tuple[np.ndarray, np.ndarray]:
        """(sample matrix [n, dim], speaker id per row) for samples of dimension `dim`."""
        rows = [(sid, e) for sid, e, _o, _d in self.samples if e.shape[0] == dim]
        if not rows:
            return np.zeros((0, dim), dtype=np.float32), np.zeros(0, dtype=np.int64)
        return np.stack([e for _, e in rows]), np.array([sid for sid, _ in rows], dtype=np.int64)


def load_gallery(db_path: Path | str) -> Gallery:
    conn = common.open_ro(db_path)
    try:
        scols = common.columns(conn, "speakers")
        names = {int(r["id"]): str(r["name"]) for r in conn.execute("SELECT id, name FROM speakers")}
        self_id = None
        if "is_self" in scols:
            r = conn.execute("SELECT id FROM speakers WHERE is_self=1").fetchone()
            self_id = int(r[0]) if r else None
        ecols = common.columns(conn, "speaker_embeddings")
        origin = "origin" if "origin" in ecols else "NULL"
        samples = []
        for r in conn.execute(f"SELECT speaker_id, embedding, {origin} AS origin, duration_s FROM speaker_embeddings ORDER BY id"):
            v = _vec(r["embedding"])
            if v is not None:
                samples.append((int(r["speaker_id"]), v, r["origin"], r["duration_s"]))
        negatives = []
        if common.table_exists(conn, "speaker_negatives"):
            for r in conn.execute("SELECT speaker_id, embedding FROM speaker_negatives ORDER BY id"):
                v = _vec(r["embedding"])
                if v is not None:
                    negatives.append((int(r["speaker_id"]), v))
    finally:
        conn.close()
    return Gallery(names, samples, negatives, self_id, str(db_path))


def cluster_trials(db_path: Path | str, utt_trials: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """One trial per stored cluster embedding; its truth is the seconds-weighted majority of its utterances."""
    conn = common.open_ro(db_path)
    try:
        ccols = common.columns(conn, "cluster_embeddings")
        extra = [c for c in ("loudness_db", "system_share") if c in ccols]
        rows = conn.execute("SELECT segment_id, cluster_label, embedding, speech_s" + "".join(f", {c}" for c in extra)
                            + " FROM cluster_embeddings ORDER BY segment_id, cluster_label").fetchall()
    finally:
        conn.close()
    members: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for t in utt_trials:
        members.setdefault((t["segment"], t["cluster"]), []).append(t)
    out: list[dict[str, Any]] = []
    stats = {"clusters": len(rows), "without_utterances": 0, "impure": 0}
    for r in rows:
        emb = _vec(r["embedding"])
        ms = members.get((r["segment_id"], r["cluster_label"]))
        if emb is None or not ms:
            stats["without_utterances"] += 1
            continue
        tot = sum(m["dur"] for m in ms) or 1e-9

        def top(key: str) -> tuple[Any, float]:
            w: dict[Any, float] = {}
            for m in ms:
                w[m[key]] = w.get(m[key], 0.0) + m["dur"]
            k = max(w, key=lambda x: w[x])
            return k, w[k] / tot

        speaker, purity = top("speaker")
        cond, _ = top("condition")

        def wmean(key: str) -> float | None:
            vals = [(m[key], m["dur"]) for m in ms if m.get(key) is not None]
            return sum(v * d for v, d in vals) / sum(d for _, d in vals) if vals else None

        keys = r.keys()
        if purity < 0.8:
            stats["impure"] += 1
        out.append({"id": f"{r['segment_id']}:{r['cluster_label']}", "emb": emb, "speech_s": float(r["speech_s"] or 0.0),
                    "dur": float(r["speech_s"] or 0.0), "speaker": speaker, "condition": cond, "mixed": purity < 0.8,
                    "purity": round(purity, 3), "session": ms[0]["session"], "segment": r["segment_id"], "t": min(m["t"] for m in ms),
                    "loudness_db": r["loudness_db"] if "loudness_db" in keys and r["loudness_db"] is not None else wmean("loudness_db"),
                    "system_share": r["system_share"] if "system_share" in keys and r["system_share"] is not None else wmean("system_share")})
    return out, stats


def voice_trials(db_path: Path | str, utt_trials: Sequence[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """One trial per utterance_voice row (1.0 databases); None when the table does not exist."""
    conn = common.open_ro(db_path)
    try:
        if not common.table_exists(conn, "utterance_voice"):
            return None
        rows = {int(r["utterance_id"]): r for r in conn.execute("SELECT utterance_id, embedding, speech_s FROM utterance_voice")}
    finally:
        conn.close()
    out = []
    for t in utt_trials:
        r = rows.get(int(t["id"])) if t.get("id") is not None else None
        emb = _vec(r["embedding"]) if r is not None else None
        if emb is None:
            continue
        sp = float(r["speech_s"]) if r["speech_s"] is not None else t["dur"]
        out.append({**t, "emb": emb, "speech_s": sp, "dur": sp})
    return out


def line_trials(truth: dict[str, Any], audio_dir: Path, cache: Path, crops: Sequence[float] = (1.0, 2.0, 4.0),
                models_dir: Path | None = None, threads: int = 2, model: str | None = None
                ) -> tuple[list[dict[str, Any]], Gallery]:
    """Oracle segmentation: one embedding per truth line and per centred crop of 1 / 2 / 4 s, straight from the
    session audio, plus a gallery from the enrollment files. Runs the embedding model of the code under test
    once and caches the vectors in `cache` (npz). `model`: another embedding model file under the models
    directory (the cascade's second model; give it a cache file of its own)."""
    import soundfile as sf

    keys: list[tuple[str, int, float]] = []
    for s in truth["sessions"]:
        for ln in s["lines"]:
            d = float(ln["t1"]) - float(ln["t0"])
            keys.append((s["id"], int(ln["i"]), 0.0))
            keys.extend((s["id"], int(ln["i"]), c) for c in crops if d >= c + 0.5)
    enroll_keys = [(n, f) for n in truth.get("enrolled") or [k for k, v in truth["speakers"].items() if v["enrolled"]]
                   for f in truth["enroll"][n]]
    import hashlib

    digest = hashlib.sha1()                 # the audio itself: re-synthesised or re-rendered clips invalidate the cache
    for f in [audio_dir / s["file"] for s in truth["sessions"]] + [audio_dir.parent.parent / "enroll" / f for _n, f in enroll_keys]:
        digest.update(f.read_bytes())
    tag = json.dumps([truth.get("story"), truth.get("variant"), truth.get("render"), [list(k) for k in keys], enroll_keys,
                      digest.hexdigest()], ensure_ascii=False)
    stamp = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:16]
    data = None
    if cache.exists():
        z = np.load(str(cache), allow_pickle=False)
        if str(z["stamp"]) == stamp:
            data = z
    if data is None:
        try:
            from . import build
        except ImportError:
            import build
        cfg = build.bare_config(models_dir=models_dir)
        cfg["speakers"]["num_threads"] = int(threads)
        if model:
            cfg["speakers"]["embedding_model"] = str(model)
        from audio_note.speech.speakers import SpeakerEngine

        eng = SpeakerEngine(cfg)
        embs, meta = [], []
        for s in truth["sessions"]:
            audio = sf.read(str(audio_dir / s["file"]), dtype="float32")[0]
            pairs = np.load(str(audio_dir / s["channels_file"])) if s.get("channels_file") else None
            for ln in s["lines"]:
                t0, t1 = float(ln["t0"]), float(ln["t1"])
                for crop in [0.0] + [c for c in crops if (t1 - t0) >= c + 0.5]:
                    a, b = (t0, t1) if not crop else ((t0 + t1 - crop) / 2, (t0 + t1 + crop) / 2)
                    clip = audio[int(a * common.SR): int(b * common.SR)]
                    share = None
                    if pairs is not None:
                        blk = pairs[int(a * common.SR) // audiofx.BLOCK: max(int(a * common.SR) // audiofx.BLOCK + 1, int(b * common.SR) // audiofx.BLOCK)]
                        e_m, e_s = float(np.sum(blk[:, 0].astype(np.float64) ** 2)), float(np.sum(blk[:, 1].astype(np.float64) ** 2))
                        share = e_s / (e_m + e_s) if (e_m + e_s) > 0 else None
                    embs.append(eng.embed(clip))
                    meta.append([s["id"], int(ln["i"]), float(crop), float(b - a), audiofx.active_level_db(clip),
                                 -1.0 if share is None else share])
        gal = []
        for n, f in enroll_keys:
            a = sf.read(str(audio_dir.parent.parent / "enroll" / f), dtype="float32")[0]
            gal.append(eng.embed(eng.speech_only(a)))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(cache), stamp=np.array(stamp), embs=np.stack(embs).astype(np.float32),
                 meta_sid=np.array([m[0] for m in meta]), meta_num=np.array([m[1:] for m in meta], dtype=np.float64),
                 gallery=np.stack(gal).astype(np.float32))
        data = np.load(str(cache), allow_pickle=False)
    by_line = {(s["id"], int(ln["i"])): (s, ln) for s in truth["sessions"] for ln in s["lines"]}
    trials = []
    for emb, sid, num in zip(data["embs"], data["meta_sid"], data["meta_num"]):
        s, ln = by_line[(str(sid), int(num[0]))]
        crop = float(num[1])
        trials.append({"id": f"{sid}:{int(num[0])}:{crop:g}", "emb": emb / (np.linalg.norm(emb) or 1.0), "speech_s": float(num[2]),
                       "dur": float(num[2]), "speaker": ln["speaker"], "condition": ln["condition"], "mixed": False,
                       "session": str(sid), "segment": str(sid), "t": float(ln["t0"]), "loudness_db": float(num[3]),
                       "system_share": None if num[4] < 0 else float(num[4]), "crop": crop})
    names = {k + 1: n for k, n in enumerate(dict.fromkeys(n for n, _ in enroll_keys))}
    ids = {n: k for k, n in names.items()}
    samples = [(ids[n], g / (np.linalg.norm(g) or 1.0), "file", None) for (n, _f), g in zip(enroll_keys, data["gallery"])]
    self_name = next((n for n, v in truth["speakers"].items() if v.get("self")), None)
    return trials, Gallery(names, samples, [], ids.get(self_name), None)


# ================================================================================================ scorers
def _score_tables(trials: Sequence[dict[str, Any]], gallery: Gallery, centroid: bool) -> tuple[np.ndarray, list[int]]:
    """[n_trials, n_speakers] best cosine per speaker (max over samples, or cosine to the mean sample)."""
    sids = gallery.speaker_ids
    if not trials or not sids:
        return np.full((len(trials), len(sids)), -1.0, dtype=np.float32), sids
    dim = int(trials[0]["emb"].shape[0])
    x = np.stack([t["emb"] for t in trials]).astype(np.float32)
    m, owner = gallery.matrix(dim)
    out = np.full((len(trials), len(sids)), -1.0, dtype=np.float32)
    for k, sid in enumerate(sids):
        rows = m[owner == sid]
        if not len(rows):
            continue
        if centroid:
            c = rows.mean(axis=0)
            c = c / (np.linalg.norm(c) or 1.0)
            out[:, k] = x @ c
        else:
            out[:, k] = (x @ rows.T).max(axis=1)
    return out, sids


def _negative_tables(trials: Sequence[dict[str, Any]], gallery: Gallery, sids: list[int]) -> np.ndarray:
    out = np.full((len(trials), len(sids)), -1.0, dtype=np.float32)
    if not trials or not gallery.negatives:
        return out
    x = np.stack([t["emb"] for t in trials]).astype(np.float32)
    for k, sid in enumerate(sids):
        rows = [n for s, n in gallery.negatives if s == sid and n.shape[0] == x.shape[1]]
        if rows:
            out[:, k] = (x @ np.stack(rows).T).max(axis=1)
    return out


def _decide(scores: np.ndarray, sids: list[int], gallery: Gallery, trials: Sequence[dict[str, Any]], threshold: float,
            system_turn_share: float | None, margin: float = 0.0, negatives: np.ndarray | None = None,
            negative_margin: float = 0.0) -> list[dict[str, Any]]:
    """Per trial: the best candidate that is allowed to win ("best", its "score") and whether it does ("name").
    A system-audio trial never has the self speaker as a candidate; a candidate closer to its own negative
    samples than `negative_margin` allows is vetoed and the next one is tried (SpeakerEngine.match). When every
    candidate is vetoed "best" is None. "raw_top" is the best cosine before vetoes: the number match() returns
    (and the pipeline stores as speaker_score) when nobody is accepted."""
    out = []
    self_col = sids.index(gallery.self_id) if gallery.self_id in sids else None
    for k, t in enumerate(trials):
        row = scores[k].astype(np.float64).copy()
        allowed = np.ones(len(row), dtype=bool)
        if (self_col is not None and system_turn_share is not None and t.get("system_share") is not None
                and float(t["system_share"]) >= system_turn_share):
            allowed[self_col] = False               # system audio is never the user themselves
        if not allowed.any():
            out.append({"name": None, "best": None, "score": None})
            continue
        raw_top = float(row[allowed].max())
        if negatives is not None:
            allowed &= (row - negatives[k]) >= negative_margin
        if not allowed.any():
            out.append({"name": None, "best": None, "score": None, "raw_top": raw_top})
            continue
        ranked = [j for j in np.argsort(-row) if allowed[j]]
        best, score = int(ranked[0]), float(row[ranked[0]])
        second = float(row[ranked[1]]) if len(ranked) > 1 else -1.0
        ok = score >= threshold and (score - second) >= margin
        name = gallery.names.get(sids[best])
        out.append({"name": name if ok else None, "best": name, "score": score, "runner_up": second, "raw_top": raw_top})
    return out


def scorer_legacy(trials: Sequence[dict[str, Any]], gallery: Gallery, threshold: float = DEFAULT_THRESHOLD,
                  system_turn_share: float | None = DEFAULT_SYSTEM_SHARE, negative_margin: float = 0.0, **_: Any) -> list[dict[str, Any]]:
    """The pre-1.0 rule (SpeakerEngine.match): best cosine over a person's samples >= threshold; negative
    samples veto; a system-audio trial is never the self speaker."""
    scores, sids = _score_tables(trials, gallery, centroid=False)
    neg = _negative_tables(trials, gallery, sids) if gallery.negatives else None
    return _decide(scores, sids, gallery, trials, threshold, system_turn_share, negatives=neg, negative_margin=negative_margin)


def scorer_centroid(trials: Sequence[dict[str, Any]], gallery: Gallery, threshold: float = DEFAULT_THRESHOLD,
                    system_turn_share: float | None = DEFAULT_SYSTEM_SHARE, **_: Any) -> list[dict[str, Any]]:
    """Cosine to the mean of a person's samples instead of the best single sample."""
    scores, sids = _score_tables(trials, gallery, centroid=True)
    return _decide(scores, sids, gallery, trials, threshold, system_turn_share)


def scorer_margin(trials: Sequence[dict[str, Any]], gallery: Gallery, threshold: float = DEFAULT_THRESHOLD, margin: float = 0.08,
                  system_turn_share: float | None = DEFAULT_SYSTEM_SHARE, **_: Any) -> list[dict[str, Any]]:
    """legacy + the winner must beat the runner-up by `margin` (two similar enrolled voices -> unknown)."""
    scores, sids = _score_tables(trials, gallery, centroid=False)
    return _decide(scores, sids, gallery, trials, threshold, system_turn_share, margin=margin)


def scorer_asnorm(trials: Sequence[dict[str, Any]], gallery: Gallery, threshold: float = 3.0, top_n: int = 30,
                  cohort_max_raw: float = 0.35, cohort_min_s: float = 2.0, raw_floor: float = 0.30,
                  system_turn_share: float | None = DEFAULT_SYSTEM_SHARE, **_: Any) -> list[dict[str, Any]]:
    """Reference adaptive score normalisation: cohort = trials that resemble nobody enrolled (best raw cosine
    < cohort_max_raw, >= cohort_min_s of speech). score' = mean of the z-scores of the raw cosine against the
    top-N cohort scores of the speaker centroid and of the trial. Falls back to `legacy` with < 10 cohort
    vectors. `threshold` is in normalised units; a raw cosine below raw_floor is never accepted."""
    raw, sids = _score_tables(trials, gallery, centroid=True)
    if not trials or not sids:
        return _decide(raw, sids, gallery, trials, threshold, system_turn_share)
    x = np.stack([t["emb"] for t in trials]).astype(np.float32)
    keep = [k for k, t in enumerate(trials) if float(raw[k].max()) < cohort_max_raw and float(t.get("speech_s") or 0) >= cohort_min_s]
    if len(keep) < 10:
        return scorer_legacy(trials, gallery, system_turn_share=system_turn_share)
    cohort = x[keep]
    n = min(top_n, len(keep))
    m, owner = gallery.matrix(x.shape[1])
    norm = np.full_like(raw, -9.0)
    t_scores = x @ cohort.T                                  # trial vs cohort
    for k in range(len(trials)):                              # a cohort member must not score against itself
        if k in keep:
            t_scores[k, keep.index(k)] = -1.0
    t_top = -np.sort(-t_scores, axis=1)[:, :n]
    t_mu, t_sd = t_top.mean(axis=1), t_top.std(axis=1) + 1e-6
    for j, sid in enumerate(sids):
        rows = m[owner == sid]
        if not len(rows):
            continue
        c = rows.mean(axis=0)
        c = c / (np.linalg.norm(c) or 1.0)
        e_top = -np.sort(-(cohort @ c))[:n]
        e_mu, e_sd = float(e_top.mean()), float(e_top.std()) + 1e-6
        z = 0.5 * ((raw[:, j] - e_mu) / e_sd + (raw[:, j] - t_mu) / t_sd)
        norm[:, j] = np.where(raw[:, j] >= raw_floor, z, -9.0)
    return _decide(norm, sids, gallery, trials, threshold, system_turn_share)


def scorer_spec_gallery(trials: Sequence[dict[str, Any]], gallery: Gallery, **_: Any) -> list[dict[str, Any]]:
    """Adapter for the 1.0 scoring layer (SPEC 5.1): audio_note.speech.scoring.Gallery.load(db, cfg).score(emb,
    speech_s=, loudness_db=, system_share=). The profile database is COPIED first, so the code under test may
    migrate the copy; the profile itself stays untouched."""
    import shutil
    import tempfile

    common.use_project()
    from audio_note.config import load_config
    from audio_note.db import DB
    from audio_note.speech import scoring

    if not gallery.db_path:
        raise RuntimeError("spec_gallery needs a profile database (not available with --source lines)")
    src = Path(gallery.db_path)
    with tempfile.TemporaryDirectory(prefix="bench_gallery_") as td:
        dst = Path(td) / src.name
        shutil.copy2(src, dst)
        for suffix in ("-wal", "-shm"):
            if Path(str(src) + suffix).exists():
                shutil.copy2(str(src) + suffix, str(dst) + suffix)
        cfg = load_config(src.parent.parent / "config.yaml")
        db = DB(dst)
        try:
            adapt = getattr(scoring, "adapt", None)
            if callable(adapt):
                adapt(db, cfg)                      # the copy learns its centroids from what it has stored
            g = scoring.Gallery.load(db, cfg)
            out = []
            for t in trials:
                d = common.call_supported(g.score, t["emb"], speech_s=t.get("speech_s"), loudness_db=t.get("loudness_db"),
                                          system_share=t.get("system_share"))
                name = getattr(d, "name", None) if getattr(d, "speaker_id", None) is not None else None
                guess_id = getattr(d, "guess_id", None)
                best = name or (gallery.names.get(int(guess_id)) if guess_id is not None else None)
                score = getattr(d, "norm", None)
                out.append({"name": name, "best": best, "score": getattr(d, "score", None) if score is None else score,
                            "guess": gallery.names.get(int(guess_id)) if guess_id is not None else None,
                            "method": getattr(d, "method", None)})
            return out
        finally:
            try:
                db.close_all()
            except Exception:  # noqa: BLE001
                pass


# ================================================================================================ 1.0 scoring layer (v5)
# These scorers replay the REAL scoring module (audio_note.speech.scoring + tracks) on a throw-away database built
# from the trials, so the numbers are those of the code that ships. A step can be switched off with a parameter:
#   v5_centroid   anchors only (centroid of the core samples), raw cosine + identify_threshold
#   v5_adapt      + adaptive per-condition centroids
#   v5_norm       + cohort normalisation
#   v5_duration   + duration rule (short vectors are soft evidence)
#   v5            + tracks and smoothing (the defaults)
#   v5_first      the defaults as a FIRST PASS: a session only knows the sessions before it (and, inside the
#                 session, only what was said so far); `v5` is what a rescore of the whole history gives
#   v5_cascade    v5 + a second voice model asked about the "maybe" vectors only (replay --cascade-model)
# Any speakers.* setting can be overridden with --param key=value.
V5_STEPS: dict[str, dict[str, Any]] = {
    "v5_centroid": {"adapt": False, "cohort_min": 10 ** 9, "short_s": 0.0, "follow_s": 0.0, "_tracks": False},
    "v5_adapt": {"cohort_min": 10 ** 9, "short_s": 0.0, "follow_s": 0.0, "_tracks": False},
    "v5_norm": {"short_s": 0.0, "follow_s": 0.0, "_tracks": False},
    "v5_duration": {"_tracks": False},
    "v5": {},
}


def _trial_key(t: dict[str, Any]) -> tuple[str, str, float]:
    """(session, line, crop seconds) of a trial. Oracle line trials are named "session:line:crop"; any other
    trial is its own line with crop 0."""
    parts = str(t.get("id")).split(":")
    if len(parts) == 3:
        try:
            return parts[0], parts[1], float(parts[2])
        except ValueError:
            pass
    return str(t.get("session") or t.get("segment") or ""), str(t.get("id")), 0.0


def _v5_cfg(data_dir: Path, overrides: dict[str, Any]) -> Any:
    import copy

    common.use_project()
    from audio_note.config import DEFAULTS, Config

    c = copy.deepcopy(DEFAULTS)
    c["paths"]["data_dir"] = str(data_dir)
    c["speakers"].update({k: v for k, v in overrides.items() if not k.startswith("_")})
    return Config(c)


def _v5_db(path: Path, trials: Sequence[dict[str, Any]], gallery: Gallery, sessions: Sequence[str] | None = None,
           skip_speakers: Sequence[str] = (), truth_of: dict[Any, str] | None = None) -> tuple[Any, dict[tuple[str, str], int]]:
    """A database holding the enrolled samples and one stored cluster vector per full line (crop 0) of the given
    sessions: what the pipeline would have stored. Returns (db, {(session, line): segment id})."""
    common.use_project()
    from datetime import timedelta

    from audio_note.db import DB

    db = DB(path)
    for sid, name in gallery.names.items():
        db.execute("INSERT INTO speakers(id, name, is_self) VALUES (?,?,?)", (int(sid), name, int(sid == gallery.self_id)))
    for sid, e, origin, dur in gallery.samples:
        db.add_speaker_embedding(int(sid), e, None, float(dur or 10.0), origin=origin or "file")
    for sid, e in gallery.negatives:
        db.add_speaker_negative(int(sid), e, None, 2.0)
    order: dict[str, int] = {}
    for t in trials:
        order.setdefault(_trial_key(t)[0], len(order))
    base = datetime(2026, 1, 1, 8, 0, 0)
    seg_of: dict[tuple[str, str], int] = {}
    for t in trials:
        s, line, crop = _trial_key(t)
        if crop != 0 or (sessions is not None and s not in sessions):
            continue
        if truth_of is not None and truth_of.get(t.get("id")) in skip_speakers:
            continue
        seg = db.add_segment(f"{s}_{line}.wav", base + timedelta(hours=order[s], seconds=float(t.get("t") or 0.0)),
                             float(t.get("dur") or t.get("speech_s") or 0.0))
        db.execute("UPDATE segments SET status='done' WHERE id=?", (seg,))
        db.upsert_cluster_embedding(seg, 0, t["emb"], float(t.get("speech_s") or 0.0))
        db.execute("UPDATE cluster_embeddings SET loudness_db=?, system_share=? WHERE segment_id=?",
                   (t.get("loudness_db"), t.get("system_share"), seg))
        seg_of[(s, line)] = seg
    return db, seg_of


def _v5_views(trials: Sequence[dict[str, Any]], seg_of: dict[tuple[str, str], int]) -> list[tuple[list[int], list[Any]]]:
    """Trials as observations in time, one list per crop length: "what if every sentence were only this long"."""
    common.use_project()
    from audio_note.speech import tracks

    order: dict[str, int] = {}
    for t in trials:
        order.setdefault(_trial_key(t)[0], len(order))
    views: dict[float, tuple[list[int], list[Any]]] = {}
    for i, t in enumerate(trials):
        s, line, crop = _trial_key(t)
        t0 = order[s] * 3600.0 + float(t.get("t") or 0.0)
        idx, obs = views.setdefault(crop, ([], []))
        idx.append(i)
        obs.append(tracks.Obs(t["emb"], t0, t0 + float(t.get("dur") or 0.0), float(t.get("speech_s") or 0.0), t.get("loudness_db"),
                              t.get("system_share"), session=s, segment=seg_of.get((s, line))))
    return list(views.values())


def _v5_decide(db: Any, cfg: Any, trials: Sequence[dict[str, Any]], seg_of: dict[tuple[str, str], int], use_tracks: bool,
               prefix_only: bool = False) -> list[Any]:
    """adapt + Gallery.load + (tracks.decide | Gallery.score_many) on `trials`. With `prefix_only` every trial is
    decided knowing only the trials of its session up to itself (the first pass of a live recording)."""
    from audio_note.speech import scoring, tracks

    scoring.adapt(db, cfg)
    scoring.Gallery.forget(db)
    g = scoring.Gallery.load(db, cfg)
    out: list[Any] = [None] * len(trials)
    if not use_tracks:
        segs = [seg_of.get(_trial_key(t)[:2]) for t in trials]
        ds = g.score_many(np.stack([t["emb"] for t in trials]), speech_s=[t.get("speech_s") for t in trials],
                          loudness_db=[t.get("loudness_db") for t in trials], system_share=[t.get("system_share") for t in trials],
                          segments=segs) if len(trials) else []
        return list(ds)
    tp = tracks.TrackParams.from_cfg(cfg)
    for idx, obs in _v5_views(trials, seg_of):
        if not prefix_only:
            for i, d in zip(idx, tracks.decide(obs, g, tp)):
                out[i] = d
            continue
        by_session: dict[Any, list[int]] = {}
        for k, o in enumerate(obs):
            by_session.setdefault(o.session, []).append(k)
        for ks in by_session.values():
            ks.sort(key=lambda k: obs[k].t0)
            for n in range(len(ks)):
                out[idx[ks[n]]] = tracks.decide([obs[k] for k in ks[: n + 1]], g, tp)[-1]
    return out


def _v5_out(gallery: Gallery, ds: Sequence[Any]) -> list[dict[str, Any]]:
    out = []
    for d in ds:
        name = d.name if d.speaker_id is not None else None
        guess = d.guess_name or (gallery.names.get(int(d.guess_id)) if d.guess_id is not None else None)
        out.append({"name": name, "best": name or guess, "score": d.norm if d.norm is not None else d.score,
                    "guess": guess if name is None else None, "method": d.method, "raw_top": d.score})
    return out


def _v5_run(trials: Sequence[dict[str, Any]], gallery: Gallery, step: dict[str, Any], params: dict[str, Any],
            first_pass: bool = False) -> list[Any]:
    import tempfile

    overrides = {**step, **params}
    use_tracks = bool(overrides.pop("_tracks", True))
    with tempfile.TemporaryDirectory(prefix="bench_v5_") as td:
        cfg = _v5_cfg(Path(td), overrides)
        if not first_pass:
            db, seg_of = _v5_db(Path(td) / "all.sqlite3", trials, gallery)
            try:
                return _v5_decide(db, cfg, trials, seg_of, use_tracks)
            finally:
                db.close_all()
        sessions: list[str] = []
        for t in trials:
            if _trial_key(t)[0] not in sessions:
                sessions.append(_trial_key(t)[0])
        out: list[Any] = [None] * len(trials)
        for k, s in enumerate(sessions):
            db, seg_of = _v5_db(Path(td) / f"before_{k}.sqlite3", trials, gallery, sessions=sessions[:k])
            try:
                idx = [i for i, t in enumerate(trials) if _trial_key(t)[0] == s]
                for i, d in zip(idx, _v5_decide(db, cfg, [trials[i] for i in idx], seg_of, use_tracks, prefix_only=True)):
                    out[i] = d
            finally:
                db.close_all()
        return out


def _make_v5(step_name: str) -> Callable[..., list[dict[str, Any]]]:
    def scorer(trials: Sequence[dict[str, Any]], gallery: Gallery, **params: Any) -> list[dict[str, Any]]:
        params.pop("threshold", None)                      # the proxy passes the legacy threshold to every scorer
        return _v5_out(gallery, _v5_run(trials, gallery, dict(V5_STEPS[step_name]), dict(params)))

    scorer.__name__ = f"scorer_{step_name}"
    scorer.__doc__ = f"The 1.0 scoring layer, step '{step_name}' (see V5_STEPS)."
    return scorer


def scorer_v5_first(trials: Sequence[dict[str, Any]], gallery: Gallery, **params: Any) -> list[dict[str, Any]]:
    """The 1.0 defaults as a first pass: every session is decided with the history before it only."""
    params.pop("threshold", None)
    return _v5_out(gallery, _v5_run(trials, gallery, {}, dict(params), first_pass=True))


def scorer_v5_cascade(trials: Sequence[dict[str, Any]], gallery: Gallery, cascade_min: float = 0.45, **params: Any) -> list[dict[str, Any]]:
    """v5, then a second voice model is asked about the vectors v5 only calls "maybe X": X is accepted when the
    second model's best enrolled speaker is X too with a raw cosine >= cascade_min. The second model knows the
    enrolled samples only (trial["emb2"], gallery.samples2: replay --cascade-model). decision["asked2"] marks the
    vectors that needed the second model."""
    params.pop("threshold", None)
    ds = _v5_run(trials, gallery, {}, dict(params))
    out = _v5_out(gallery, ds)
    samples2 = getattr(gallery, "samples2", None) or []
    anchors: dict[int, np.ndarray] = {}
    for sid in {s for s, *_ in samples2}:
        m = np.stack([e for s, e, _o, _d in samples2 if s == sid]).mean(axis=0)
        anchors[sid] = m / (np.linalg.norm(m) or 1.0)
    for t, d, o in zip(trials, ds, out):
        if d.speaker_id is not None or d.guess_id is None or t.get("emb2") is None or not anchors:
            continue
        o["asked2"] = True
        sims = {sid: float(np.dot(t["emb2"], a)) for sid, a in anchors.items()}
        best = max(sims, key=lambda k: sims[k])
        if best == d.guess_id and sims[best] >= cascade_min:
            o["name"], o["guess"], o["method"] = gallery.names.get(best), None, "cascade"
    return out


def scorer_v5_db(trials: Sequence[dict[str, Any]], gallery: Gallery, **params: Any) -> list[dict[str, Any]]:
    """What a database looks like after the 1.0 code brought it in step: a COPY of gallery.db_path is migrated,
    `scoring.adapt` and `scoring.rescore` run on it, and the stored decision of every trial (an utterance id) is
    read back. Hand labels, vetoes and the system-audio rule are enforced by rescore itself."""
    import shutil
    import tempfile

    common.use_project()
    from audio_note.db import DB
    from audio_note.speech import scoring

    if not gallery.db_path:
        raise RuntimeError("v5_db needs a database (proxy mode, or replay with a profile)")
    params.pop("threshold", None)
    src = Path(gallery.db_path)
    with tempfile.TemporaryDirectory(prefix="bench_v5db_") as td:
        dst = Path(td) / "audio_note.sqlite3"
        shutil.copy2(src, dst)
        for suffix in ("-wal", "-shm"):
            if Path(str(src) + suffix).exists():
                shutil.copy2(str(src) + suffix, str(dst) + suffix)
        cfg = _v5_cfg(Path(td), dict(params))
        db = DB(dst)
        try:
            scoring.adapt(db, cfg)
            scoring.Gallery.forget(db)
            summary = scoring.rescore(db, cfg)
            rows = {int(r["id"]): r for r in db.query("SELECT id, speaker_id, speaker_guess_id, speaker_score, speaker_meta FROM utterances")}
        finally:
            db.close_all()
    out = []
    for t in trials:
        r = rows.get(int(t["id"])) if str(t.get("id")).isdigit() else None
        if r is None:
            out.append({"name": None, "best": None, "score": None})
            continue
        name = gallery.names.get(int(r["speaker_id"])) if r["speaker_id"] is not None else None
        guess = gallery.names.get(int(r["speaker_guess_id"])) if r["speaker_guess_id"] is not None else None
        try:
            meta = json.loads(r["speaker_meta"]) if r["speaker_meta"] else {}
        except ValueError:
            meta = {}
        out.append({"name": name, "best": name or guess, "score": meta.get("norm", r["speaker_score"]), "guess": guess,
                    "method": meta.get("m"), "raw_top": r["speaker_score"]})
    scorer_v5_db.last_summary = summary            # type: ignore[attr-defined]
    return out


SCORERS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "legacy": scorer_legacy, "centroid": scorer_centroid, "margin": scorer_margin, "asnorm": scorer_asnorm,
    "spec_gallery": scorer_spec_gallery,
    **{name: _make_v5(name) for name in V5_STEPS},
    "v5_first": scorer_v5_first, "v5_cascade": scorer_v5_cascade, "v5_db": scorer_v5_db,
}


def load_scorer(spec: str) -> tuple[str, Callable[..., list[dict[str, Any]]]]:
    """"name" (built in), "package.module:function" or "path/to/file.py:function"."""
    if spec in SCORERS:
        return spec, SCORERS[spec]
    if ":" not in spec:
        raise SystemExit(f"unknown scorer {spec!r}; built in: {', '.join(SCORERS)}")
    mod_name, fn_name = spec.rsplit(":", 1)
    if mod_name.endswith(".py") or "/" in mod_name or "\\" in mod_name:
        path = Path(mod_name)
        sp = importlib.util.spec_from_file_location(f"bench_scorer_{path.stem}", path)
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
    else:
        common.use_project()
        mod = importlib.import_module(mod_name)
    return f"{Path(mod_name).stem if mod_name.endswith('.py') else mod_name.rsplit('.', 1)[-1]}:{fn_name}", getattr(mod, fn_name)


def sweep(trials: Sequence[dict[str, Any]], decisions: Sequence[dict[str, Any]], enrolled: Sequence[str],
          thresholds: Sequence[float] = SWEEP) -> list[dict[str, Any]]:
    """FRR / FAR / accuracy when the scorer's best candidate is accepted at each threshold (pure)."""
    out = []
    for thr in thresholds:
        decided = [d.get("best") if (d.get("best") is not None and d.get("score") is not None and float(d["score"]) >= thr) else None
                   for d in decisions]
        o = evaluate(trials, decided, enrolled)["overall"]
        out.append({"threshold": thr, "accuracy": o["accuracy"], "frr": o["frr"], "far": o["far"], "confusion_rate": o["confusion_rate"]})
    return out


def equal_error(points: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """The sweep point where FRR and FAR are closest."""
    ok = [p for p in points if p["frr"] is not None and p["far"] is not None]
    if not ok:
        return None
    p = min(ok, key=lambda p: abs(p["frr"] - p["far"]))
    return {"threshold": p["threshold"], "frr": p["frr"], "far": p["far"]}


def replay(trials: list[dict[str, Any]], gallery: Gallery, scorers: Sequence[tuple[str, Callable[..., Any]]], enrolled: Sequence[str],
           params: dict[str, Any] | None = None, sweep_scores: bool = True) -> dict[str, Any]:
    """Run every scorer over the trials and score its decisions."""
    public = [{k: v for k, v in t.items() if k not in ("speaker", "condition", "mixed", "purity", "line", "stored_name", "stored_id",
                                                        "stored_score", "guess_id", "crop")} for t in trials]
    out: dict[str, Any] = {}
    for name, fn in scorers:
        decisions = fn(public, gallery, **(params or {}))
        if len(decisions) != len(trials):
            raise RuntimeError(f"scorer {name} returned {len(decisions)} decisions for {len(trials)} trials")
        ev = evaluate(trials, [d.get("name") for d in decisions], enrolled)
        ev["headline"] = headline(ev)
        guesses = [d.get("guess") for d in decisions]
        if any(g is not None for g in guesses):
            tgt = [(t, g) for t, g, d in zip(trials, guesses, decisions) if d.get("name") is None and g is not None]
            ev["guesses"] = {"n": len(tgt), "right": sum(1 for t, g in tgt if g == t["speaker"]),
                             "wrong": sum(1 for t, g in tgt if g != t["speaker"])}
        if any(d.get("asked2") for d in decisions):
            asked = sum(1 for d in decisions if d.get("asked2"))
            ev["second_model"] = {"asked": asked, "share": common.rate(asked, len(decisions)),
                                  "accepted": sum(1 for d in decisions if d.get("method") == "cascade")}
        methods: dict[str, int] = {}
        for d in decisions:
            if d.get("name") is not None and d.get("method"):
                k = str(d["method"]).split(":")[0]
                methods[k] = methods.get(k, 0) + 1
        if methods:
            ev["methods"] = dict(sorted(methods.items()))
        numeric = [d for d in decisions if d.get("score") is not None]
        if sweep_scores and numeric and name != "asnorm" and all(-1.01 <= float(d["score"]) <= 1.01 for d in numeric):
            ev["sweep"] = sweep(trials, decisions, enrolled)
            ev["equal_error"] = equal_error(ev["sweep"])
        out[name] = ev
    return out


# ================================================================================================ profile mode
def run_profile(pdir: Path, truth_path: Path | None = None, label: str | None = None) -> dict[str, Any]:
    truth = common.load_truth(truth_path or pdir)
    db_path = common.profile_db_path(pdir)
    utts = bench_cer.load_utterances(db_path, truth)
    trials, facts = utterance_trials(truth, utts)
    enrolled = list(truth.get("enrolled") or [n for n, v in truth["speakers"].items() if v["enrolled"]])
    unknown = truth.get("unknown_label") or "未知"
    decided = [None if (t["stored_id"] is None or t["stored_name"] == unknown) else t["stored_name"] for t in trials]
    ev = evaluate(trials, decided, enrolled)
    out: dict[str, Any] = {"kind": "speakers", "label": label or pdir.name, "profile": str(pdir), "variant": truth.get("variant"),
                           "enrolled": enrolled, "created_at": datetime.now().isoformat(timespec="seconds"),
                           "headline": headline(ev), **ev, **facts}
    self_name = next((n for n, v in truth["speakers"].items() if v.get("self")), None)
    sys_trials = [(t, d) for t, d in zip(trials, decided) if t["condition"] == "system"]
    out["system_audio"] = {"trials": len(sys_trials), "labelled_self": sum(1 for _t, d in sys_trials if d == self_name)}
    gallery = load_gallery(db_path)
    if any(t.get("guess_id") is not None for t in trials):
        g = [(t, gallery.names.get(int(t["guess_id"]))) for t, d in zip(trials, decided) if d is None and t.get("guess_id") is not None]
        out["guesses"] = {"n": len(g), "right": sum(1 for t, n in g if n == t["speaker"]), "wrong": sum(1 for t, n in g if n != t["speaker"])}
    scores = [float(t["stored_score"]) for t in trials if t.get("stored_score") is not None]
    if scores:
        thr = DEFAULT_THRESHOLD
        out["near_threshold"] = {"threshold": thr, "band": 0.10, "scored": len(scores),
                                 "share": common.rate(sum(1 for s in scores if abs(s - thr) <= 0.10), len(scores))}
    imp = pdir / "bench_import.json"
    if imp.exists():
        info = common.read_json(imp)
        out["import"] = {k: info.get(k) for k in ("code", "asr", "variant", "threads", "overrides", "segments", "segment_status", "audio_s")}
    return out


# ================================================================================================ proxy mode
def proxy(db_path: Path | str, threshold: float = DEFAULT_THRESHOLD, band: float = 0.10,
          scorers: Sequence[tuple[str, Callable[..., Any]]] = (), system_turn_share: float = DEFAULT_SYSTEM_SHARE) -> dict[str, Any]:
    """Aggregates over a database without ground truth (a COPY of real data): the mix of decisions, how many sit
    near the threshold, and - per scorer - how often a replayed decision contradicts the few things that are
    known for sure (hand labels, vetoes, system audio is never the user). No names and no text are returned."""
    conn = common.open_ro(db_path)
    try:
        ucols = common.columns(conn, "utterances")
        has = lambda c: c in ucols                                             # noqa: E731
        sel = ["id", "segment_id", "cluster_label", "start_s", "end_s", "speaker_id", "speaker_score"]
        sel += [c for c in ("speaker_source", "loudness_db", "system_share", "speaker_guess_id") if has(c)]
        utts = [dict(r) for r in conn.execute(f"SELECT {', '.join(sel)} FROM utterances")]
        vetoes: dict[int, set[int]] = {}
        if common.table_exists(conn, "utterance_rejections"):
            for r in conn.execute("SELECT utterance_id, speaker_id FROM utterance_rejections"):
                vetoes.setdefault(int(r[0]), set()).add(int(r[1]))
        clusters = conn.execute("SELECT segment_id, cluster_label, embedding, speech_s FROM cluster_embeddings").fetchall() \
            if common.table_exists(conn, "cluster_embeddings") else []
        voice = None
        if common.table_exists(conn, "utterance_voice"):
            voice = {int(r[0]): (r[1], r[2]) for r in conn.execute("SELECT utterance_id, embedding, speech_s FROM utterance_voice")}
        seg_status = {r[0]: r[1] for r in conn.execute("SELECT status, COUNT(*) FROM segments GROUP BY status")}
    finally:
        conn.close()
    gallery = load_gallery(db_path)

    def state(u: dict[str, Any]) -> str:
        if u.get("speaker_source") == "user" and u["speaker_id"] is not None:
            return "user"
        if u["speaker_id"] is not None:
            return "sure"
        return "maybe" if u.get("speaker_guess_id") is not None else "unknown"

    n = len(utts)
    states: dict[str, int] = {}
    by_dur: dict[str, dict[str, int]] = {}
    by_loud: dict[str, dict[str, int]] = {}
    hist: dict[str, int] = {}
    near = scored = 0
    for u in utts:
        st = state(u)
        states[st] = states.get(st, 0) + 1
        dur = float(u["end_s"]) - float(u["start_s"])
        by_dur.setdefault(bucket_of(dur), {}).setdefault(st, 0)
        by_dur[bucket_of(dur)][st] += 1
        lv = u.get("loudness_db")
        lb = "unknown" if lv is None else "<-40" if lv < -40 else "-40..-32" if lv < -32 else "-32..-26" if lv < -26 else ">=-26"
        if u.get("system_share") is not None and float(u["system_share"]) >= system_turn_share:
            lb = "system"
        by_loud.setdefault(lb, {}).setdefault(st, 0)
        by_loud[lb][st] += 1
        sc = u.get("speaker_score")
        if sc is not None and st != "user":
            scored += 1
            near += abs(float(sc) - threshold) <= band
            k = f"{math.floor(float(sc) * 20) / 20:.2f}"
            hist[k] = hist.get(k, 0) + 1
    order = {name: k for k, (_lo, _hi, name) in enumerate(BUCKETS)}
    out: dict[str, Any] = {
        "kind": "speakers_proxy", "utterances": n, "segments": seg_status, "enrolled_speakers": len(gallery.names),
        "samples": len(gallery.samples), "negatives": len(gallery.negatives), "threshold": threshold, "band": band,
        "decisions": {k: {"n": v, "share": common.rate(v, n)} for k, v in sorted(states.items())},
        "near_threshold": {"scored": scored, "n": int(near), "share": common.rate(near, scored)},
        "by_duration": {k: v for k, v in sorted(by_dur.items(), key=lambda kv: order.get(kv[0], 99))},
        "by_loudness": dict(sorted(by_loud.items())),
        "score_histogram": dict(sorted(hist.items())),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    # --- replay with constraints instead of truth
    emb_by_cluster = {(r["segment_id"], r["cluster_label"]): (_vec(r["embedding"]), float(r["speech_s"] or 0.0)) for r in clusters}
    trials: list[dict[str, Any]] = []
    for u in utts:
        emb, sp = None, None
        if voice is not None and int(u["id"]) in voice:
            emb, sp = _vec(voice[int(u["id"])][0]), voice[int(u["id"])][1]
        if emb is None:
            emb, sp = emb_by_cluster.get((u["segment_id"], u["cluster_label"]), (None, None))
        if emb is None:
            continue
        dur = float(u["end_s"]) - float(u["start_s"])
        must = int(u["speaker_id"]) if (u.get("speaker_source") == "user" and u["speaker_id"] is not None) else None
        must_not = set(vetoes.get(int(u["id"]), ()))
        if gallery.self_id is not None and u.get("system_share") is not None and float(u["system_share"]) >= system_turn_share:
            must_not.add(int(gallery.self_id))
        trials.append({"id": u["id"], "emb": emb, "speech_s": float(sp if sp is not None else dur), "dur": dur, "segment": u["segment_id"],
                       "t": float(u["start_s"]), "loudness_db": u.get("loudness_db"), "system_share": u.get("system_share"),
                       "_must": must, "_must_not": must_not, "_stored": u["speaker_id"]})
    out["replay_trials"] = len(trials)
    out["embedding_source"] = "utterance_voice" if voice else "cluster_embeddings"
    ids = {v: k for k, v in gallery.names.items()}
    out["replay"] = {}
    for name, fn in scorers:
        public = [{k: v for k, v in t.items() if not k.startswith("_")} for t in trials]
        try:
            decisions = fn(public, gallery, threshold=threshold) if name in ("legacy", "centroid", "margin") else fn(public, gallery)
        except Exception as e:  # noqa: BLE001 - one scorer failing must not hide the others
            out["replay"][name] = {"error": f"{type(e).__name__}: {e}"[:200]}
            continue
        sure = maybe = unknown = near_s = scored_s = 0
        must_n = must_bad = not_n = not_bad = changed = 0
        for t, d in zip(trials, decisions):
            sid = ids.get(d.get("name")) if d.get("name") is not None else None
            sure += sid is not None
            maybe += sid is None and d.get("guess") is not None
            unknown += sid is None and d.get("guess") is None
            sc = d.get("raw_top") if d.get("raw_top") is not None else d.get("score")
            if sc is not None and -1.01 <= float(sc) <= 1.01:
                scored_s += 1
                near_s += abs(float(sc) - threshold) <= band
            if t["_must"] is not None:
                must_n += 1
                must_bad += sid != t["_must"]
            if t["_must_not"]:
                not_n += 1
                not_bad += sid in t["_must_not"]
            changed += (sid != t["_stored"]) and t["_must"] is None
        m = len(trials)
        out["replay"][name] = {
            "sure": {"n": int(sure), "share": common.rate(sure, m)}, "maybe": {"n": int(maybe), "share": common.rate(maybe, m)},
            "unknown": {"n": int(unknown), "share": common.rate(unknown, m)},
            "near_threshold": {"scored": int(scored_s), "share": common.rate(near_s, scored_s)},
            "hand_labels": {"n": int(must_n), "contradicted": int(must_bad)},
            "must_not": {"n": int(not_n), "violated": int(not_bad)},
            "differs_from_stored": {"n": int(changed), "share": common.rate(changed, m)},
        }
    return out


# ================================================================================================ output
def _p(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def format_eval(title: str, ev: dict[str, Any]) -> str:
    rows = [title]

    def line(name: str, b: dict[str, Any]) -> str:
        return (f"  {name:<10} n={b['n']:<4} acc {_p(b['accuracy']):>6}  FRR {_p(b['frr']):>6} (rejected {_p(b['reject_rate'])}, confused "
                f"{_p(b['confusion_rate'])})  FAR {_p(b['far']):>6}   by time: acc {_p(b['by_time']['accuracy'])} FRR {_p(b['by_time']['frr'])} "
                f"FAR {_p(b['by_time']['far'])}")

    rows.append(line("overall", ev["overall"]))
    for k, b in ev["by_condition"].items():
        rows.append(line(k, b))
    for k, b in ev["by_duration"].items():
        rows.append(line(k, b))
    rows.append("  per speaker: " + "; ".join(f"{s} FRR {_p(v['frr'])} FAR {_p(v['far'])}" for s, v in ev["per_speaker"].items()))
    if ev.get("unknown_speakers"):
        rows.append("  not enrolled: " + "; ".join(f"{s} rejected {v['rejected']}/{v['trials']}" for s, v in ev["unknown_speakers"].items()))
    if ev.get("equal_error"):
        e = ev["equal_error"]
        rows.append(f"  sweep: FRR = FAR near threshold {e['threshold']:.2f} (FRR {_p(e['frr'])}, FAR {_p(e['far'])})")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["profile", "replay", "proxy"])
    ap.add_argument("--profile", default=None, help="bench profile directory or name under <root>/profiles")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--truth", type=Path, default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--source", default="auto", choices=["auto", "clusters", "utterances", "lines"], help="replay: where the embeddings come from")
    ap.add_argument("--variant", default=None, help="replay --source lines: rendered variant (default: the profile's, else mixed)")
    ap.add_argument("--scorers", default="legacy,centroid,margin", help="comma-separated built-in scorers")
    ap.add_argument("--scorer", action="append", default=[], help="extra scorer: module:function or file.py:function (repeatable)")
    ap.add_argument("--param", action="append", default=[], metavar="key=value", help="replay: keyword passed to every scorer")
    ap.add_argument("--db", type=Path, default=None, help="proxy: database copy to read")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--band", type=float, default=0.10)
    ap.add_argument("--models-dir", type=Path, default=None)
    ap.add_argument("--cascade-model", default=None, metavar="FILE.onnx",
                    help="replay --source lines: a second embedding model under the models directory (scorer v5_cascade)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--compare", type=Path, default=None)
    args = ap.parse_args(argv)
    scorers = [load_scorer(s) for s in [x for x in args.scorers.split(",") if x] + list(args.scorer)]

    if args.mode == "proxy":
        if not args.db:
            ap.error("proxy needs --db")
        res = proxy(args.db, args.threshold, args.band, scorers)
        res["label"] = args.label or "proxy"
        print(json.dumps({k: v for k, v in res.items() if k != "score_histogram"}, ensure_ascii=False, indent=1))
    else:
        root = args.root or common.default_root()
        pdir = common.resolve_profile(args.profile, root) if args.profile else None
        if args.mode == "profile":
            if pdir is None:
                ap.error("profile mode needs --profile")
            res = run_profile(pdir, args.truth, args.label)
            print(format_eval(f"== speakers [{res['label']}] variant={res.get('variant')} enrolled={','.join(res['enrolled'])}", res))
            print("  line coverage: " + "; ".join(f"{k} {_p(v['covered_share'])} of {v['seconds']:.0f}s ({v['missed_lines']} of {v['lines']} lines missed)"
                                                  for k, v in res["line_coverage"].items()))
        else:
            import yaml

            params = {}
            for item in args.param:
                k, v = item.split("=", 1)
                params[k.strip()] = yaml.safe_load(v)
            source = args.source
            stats: dict[str, Any] = {}
            if source == "lines":
                variant = args.variant or (common.load_truth(pdir).get("variant") if pdir else "mixed")
                audio_dir = root / "sessions" / variant
                truth = common.read_json(audio_dir / "truth.json")
                trials, gallery = line_trials(truth, audio_dir, root / "emb_cache" / f"lines_{variant}.npz", models_dir=args.models_dir)
                enrolled = [n for n, v in truth["speakers"].items() if v["enrolled"]]
                if args.cascade_model:
                    stem = Path(args.cascade_model).stem
                    t2, g2 = line_trials(truth, audio_dir, root / "emb_cache" / f"lines_{variant}__{stem}.npz",
                                         models_dir=args.models_dir, model=args.cascade_model)
                    by_id = {t["id"]: t["emb"] for t in t2}
                    for t in trials:
                        t["emb2"] = by_id.get(t["id"])
                    gallery.samples2 = list(g2.samples)
            else:
                if pdir is None:
                    ap.error("replay needs --profile (or --source lines)")
                truth = common.load_truth(args.truth or pdir)
                db_path = common.profile_db_path(pdir)
                utt_trials, _facts = utterance_trials(truth, bench_cer.load_utterances(db_path, truth))
                enrolled = list(truth.get("enrolled") or [n for n, v in truth["speakers"].items() if v["enrolled"]])
                gallery = load_gallery(db_path)
                trials = voice_trials(db_path, utt_trials) if source in ("auto", "utterances") else None
                if trials:
                    source = "utterances"
                elif source == "utterances":
                    raise SystemExit("this database has no utterance_voice rows; use --source clusters or lines")
                else:
                    trials, stats = cluster_trials(db_path, utt_trials)
                    source = "clusters"
            results = replay(trials, gallery, scorers, enrolled, params)
            res = {"kind": "speakers_replay", "label": args.label or (pdir.name if pdir else source), "source": source,
                   "trials": len(trials), "trial_stats": stats, "params": params, "enrolled": enrolled, "scorers": results,
                   "headline": {name: r["headline"] for name, r in results.items()},
                   "created_at": datetime.now().isoformat(timespec="seconds")}
            for name, r in results.items():
                print(format_eval(f"== replay [{res['label']}] source={source} scorer={name} trials={len(trials)}", r))
                if r.get("guesses"):
                    print(f"  maybe: {r['guesses']['n']} (right {r['guesses']['right']}, wrong {r['guesses']['wrong']})")
                if r.get("second_model"):
                    m = r["second_model"]
                    print(f"  second model asked for {m['asked']} of {len(trials)} vectors ({_p(m['share'])}), accepted {m['accepted']}")
    if args.compare and res.get("headline"):
        old = common.read_json(args.compare)
        print(f"-- compared with [{old.get('label')}]")
        for k, v in res["headline"].items():
            ov = (old.get("headline") or {}).get(k)
            if isinstance(v, float) and isinstance(ov, float):
                print(f"  {k:<22} {_p(ov):>7} -> {_p(v):>7}  ({100 * (v - ov):+.1f} pt)")
    if args.out:
        common.write_json(args.out, res)
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
