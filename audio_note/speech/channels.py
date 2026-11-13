"""System-audio share of a time range, from segments.channel_rms (one (mic_rms, system_rms) pair per capture block).

Speakers v4: a turn whose system-audio share reaches speakers.system_turn_share is never identified as the self
speaker (it is the other side of a meeting / call). Part of the audio_note.speech.pipeline facade.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..config import Config
from .chunking import F

DEFAULT_SYSTEM_TURN_SHARE = 0.6


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
        return float(cfg.speakers.get("system_turn_share", F.DEFAULT_SYSTEM_TURN_SHARE))
    except (TypeError, ValueError, AttributeError):
        return F.DEFAULT_SYSTEM_TURN_SHARE


def _segment_channels(seg: Any) -> tuple[np.ndarray | None, float | None]:
    """(channel pairs, block seconds) of a segment row; (None, None) when it has none (mic only, imported, old row)."""
    try:
        blob, bs = seg["channel_rms"], seg["channel_block_s"]
    except (IndexError, KeyError, TypeError):
        return None, None
    pairs = F.channel_pairs(blob)
    try:
        bs_f = float(bs) if bs is not None else None
    except (TypeError, ValueError):
        bs_f = None
    if pairs is None or not bs_f or bs_f <= 0:
        return None, None
    return pairs, bs_f


__all__ = ["DEFAULT_SYSTEM_TURN_SHARE", "channel_pairs", "system_share", "system_turn_share"]
