"""Threshold calibration from the enrolled voice samples (pure numpy, no model, no I/O).

Used by `audio-note speakers calibrate` and by the web API; lives here (not in the CLI) so neither layer has
to import the other.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def calibrate_threshold(embeddings: list[tuple[int, str, np.ndarray]], lo: float = 0.3, hi: float = 0.7) -> dict[str, Any]:
    """Cosine scores between enrolled embeddings: same-speaker vs cross-speaker distributions and a
    suggested identify_threshold = midpoint(max cross, min same) clamped to [lo, hi]."""
    embs = []
    for sid, name, e in embeddings:
        e = np.asarray(e, dtype=np.float32)
        n = float(np.linalg.norm(e))
        embs.append((int(sid), name, e / n if n > 0 else e))
    same: list[float] = []
    cross: list[float] = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            s = float(np.dot(embs[i][2], embs[j][2]))
            (same if embs[i][0] == embs[j][0] else cross).append(s)

    def stats(xs: list[float]) -> dict[str, float] | None:
        if not xs:
            return None
        a = np.asarray(xs)
        return {"n": int(a.size), "min": float(a.min()), "mean": float(a.mean()), "max": float(a.max())}

    out: dict[str, Any] = {
        "n_embeddings": len(embs),
        "n_speakers": len({e[0] for e in embs}),
        "same": stats(same),
        "cross": stats(cross),
        "suggested": None,
        "overlap": False,
    }
    if same and cross:
        mid = (max(cross) + min(same)) / 2.0
        out["overlap"] = max(cross) >= min(same)
        out["suggested"] = float(min(hi, max(lo, mid)))
    elif same:
        out["suggested"] = float(min(hi, max(lo, min(same) - 0.05)))
    elif cross:
        out["suggested"] = float(min(hi, max(lo, max(cross) + 0.05)))
    return out
