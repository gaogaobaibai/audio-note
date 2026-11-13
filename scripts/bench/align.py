"""Character alignment and the metrics built on it (pure functions: no audio, no database, no network).

    ops = align(ref, hyp)                      # [("=", i, j) | ("s", i, j) | ("d", i, None) | ("i", None, j)]
    counts(ops)                                # {"n", "hits", "sub", "del", "ins", "errors"}
    cer(ref, hyp)                              # errors / len(ref)
    term_stats(ref, ops, ["青禾", "t30"])       # per term: occurrences in the reference and how many came out wrong
    assign_lines(utterances, lines)            # which truth line each utterance belongs to (time overlap)

Strings are compared character by character; normalise them first (textnorm.normalize). The alignment is the
standard minimum edit distance one (substitution, deletion, insertion all cost 1); ties prefer a diagonal step,
then a deletion, then an insertion, so results are deterministic.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

Op = tuple[str, "int | None", "int | None"]


def _codes(s: Sequence[Any]) -> np.ndarray:
    if isinstance(s, str):
        return np.frombuffer(s.encode("utf-32-le"), dtype=np.uint32).astype(np.int64)
    table: dict[Any, int] = {}
    return np.array([table.setdefault(x, len(table)) for x in s], dtype=np.int64)


def _token_codes(ref: Sequence[Any], hyp: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(ref, str) and isinstance(hyp, str):
        return _codes(ref), _codes(hyp)
    table: dict[Any, int] = {}
    a = np.array([table.setdefault(x, len(table)) for x in ref], dtype=np.int64)
    b = np.array([table.setdefault(x, len(table)) for x in hyp], dtype=np.int64)
    return a, b


def _matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full edit-distance matrix, (len(a) + 1) x (len(b) + 1), computed row by row with numpy."""
    n, m = len(a), len(b)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    idx = np.arange(m + 1, dtype=np.int32)
    d[0] = idx
    for i in range(1, n + 1):
        prev = d[i - 1]
        v = np.empty(m + 1, dtype=np.int32)
        v[0] = i
        if m:
            v[1:] = np.minimum(prev[:-1] + (b != a[i - 1]), prev[1:] + 1)
        # insertions: cur[j] = min_k<=j (v[k] + j - k)
        d[i] = np.minimum.accumulate(v - idx) + idx
    return d


def distance(ref: Sequence[Any], hyp: Sequence[Any]) -> int:
    """Edit distance between two strings / token sequences."""
    a, b = _token_codes(ref, hyp)
    if len(a) == 0 or len(b) == 0:
        return int(len(a) + len(b))
    return int(_matrix(a, b)[len(a), len(b)])


def align(ref: Sequence[Any], hyp: Sequence[Any]) -> list[Op]:
    """Minimum-edit alignment of `hyp` against `ref` as a list of operations in reading order."""
    a, b = _token_codes(ref, hyp)
    n, m = len(a), len(b)
    if n == 0:
        return [("i", None, j) for j in range(m)]
    if m == 0:
        return [("d", i, None) for i in range(n)]
    d = _matrix(a, b)
    ops: list[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            same = a[i - 1] == b[j - 1]
            if d[i, j] == d[i - 1, j - 1] + (0 if same else 1):
                ops.append(("=" if same else "s", i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("d", i - 1, None))
            i -= 1
            continue
        ops.append(("i", None, j - 1))
        j -= 1
    ops.reverse()
    return ops


def counts(ops: Iterable[Op]) -> dict[str, int]:
    c = {"n": 0, "hits": 0, "sub": 0, "del": 0, "ins": 0}
    for op, _i, _j in ops:
        if op == "=":
            c["hits"] += 1
            c["n"] += 1
        elif op == "s":
            c["sub"] += 1
            c["n"] += 1
        elif op == "d":
            c["del"] += 1
            c["n"] += 1
        else:
            c["ins"] += 1
    c["errors"] = c["sub"] + c["del"] + c["ins"]
    return c


def cer(ref: Sequence[Any], hyp: Sequence[Any]) -> float | None:
    """(substitutions + deletions + insertions) / len(ref); None for an empty reference."""
    if len(ref) == 0:
        return None
    return distance(ref, hyp) / len(ref)


# ------------------------------------------------------------------------------------------------ ownership
def concat(texts: Sequence[str]) -> tuple[str, list[int]]:
    """Join the texts and say, for every character of the result, which text it came from."""
    owners: list[int] = []
    for k, t in enumerate(texts):
        owners.extend([k] * len(t))
    return "".join(texts), owners


def attribute(ops: Sequence[Op], ref_owner: Sequence[int], hyp_owner: Sequence[int]) -> tuple[dict[int, dict[str, int]], dict[int, dict[str, int]]]:
    """Split the error counts of one alignment by the owner of the characters involved.

    Returns (by reference owner, by hypothesis owner). Reference side: every reference character counts once
    as n and as hit / sub / del for its owner; an insertion goes to the owner of the nearest reference
    character before it (the first owner for a leading insertion). Hypothesis side: hit / sub / ins for the
    owner of the hypothesis character; a deletion goes to the owner of the nearest hypothesis character
    before it (the first one for a leading deletion)."""
    by_ref: dict[int, dict[str, int]] = {}
    by_hyp: dict[int, dict[str, int]] = {}

    def slot(table: dict[int, dict[str, int]], k: int) -> dict[str, int]:
        return table.setdefault(k, {"n": 0, "hits": 0, "sub": 0, "del": 0, "ins": 0})

    first_ref = ref_owner[0] if len(ref_owner) else None
    first_hyp = hyp_owner[0] if len(hyp_owner) else None
    last_ref, last_hyp = first_ref, first_hyp
    for op, i, j in ops:
        if op in ("=", "s"):
            r, h = slot(by_ref, ref_owner[i]), slot(by_hyp, hyp_owner[j])
            key = "hits" if op == "=" else "sub"
            r["n"] += 1
            r[key] += 1
            h["n"] += 1
            h[key] += 1
            last_ref, last_hyp = ref_owner[i], hyp_owner[j]
        elif op == "d":
            r = slot(by_ref, ref_owner[i])
            r["n"] += 1
            r["del"] += 1
            last_ref = ref_owner[i]
            if last_hyp is not None:
                slot(by_hyp, last_hyp)["del"] += 1
        else:
            h = slot(by_hyp, hyp_owner[j])
            h["n"] += 1
            h["ins"] += 1
            last_hyp = hyp_owner[j]
            if last_ref is not None:
                slot(by_ref, last_ref)["ins"] += 1
    return by_ref, by_hyp


def hyp_to_ref(ops: Sequence[Op], n_hyp: int) -> list[int | None]:
    """For each hypothesis position the reference position it is aligned to (None for an insertion)."""
    out: list[int | None] = [None] * n_hyp
    for op, i, j in ops:
        if j is not None and op in ("=", "s"):
            out[j] = i
    return out


def ref_window(h2r: Sequence[int | None], n_ref: int, a: int, b: int) -> tuple[int, int]:
    """Reference range [lo, hi) a hypothesis range [a, b) has to answer for, given hyp_to_ref(): everything
    after the reference character aligned to the last aligned hypothesis character before `a` and before the
    one aligned to the first aligned hypothesis character at or after `b`. Deletions next to the range are
    therefore inside the window (generous on purpose: an edit that restores dropped characters must be able
    to score), and for an empty range (a == b) the window holds exactly the deletions at that point."""
    lo = 0
    for j in range(min(a, len(h2r)) - 1, -1, -1):
        if h2r[j] is not None:
            lo = h2r[j] + 1
            break
    hi = n_ref
    for j in range(max(b, 0), len(h2r)):
        if h2r[j] is not None:
            hi = h2r[j]
            break
    return lo, max(lo, hi)


def diff_regions(old: str, new: str) -> list[tuple[int, int, str]]:
    """Changed regions between two strings as [(start, end, replacement)]: old[start:end] became `replacement`.
    Adjacent substitutions / deletions / insertions are merged into one region."""
    regions: list[tuple[int, int, str]] = []
    cur: list[Any] | None = None
    pos = 0                       # next index of `old` not yet consumed
    for op, i, j in align(old, new):
        if op == "=":
            if cur is not None:
                regions.append((cur[0], cur[1], "".join(cur[2])))
                cur = None
            pos = i + 1
            continue
        if cur is None:
            cur = [pos, pos, []]
        if op in ("s", "d"):
            cur[1] = i + 1
            pos = i + 1
        if op in ("s", "i"):
            cur[2].append(new[j])
    if cur is not None:
        regions.append((cur[0], cur[1], "".join(cur[2])))
    return regions


# ------------------------------------------------------------------------------------------------ terms
def term_occurrences(text: str, term: str) -> list[tuple[int, int]]:
    """[start, end) of every occurrence of `term` in `text` (both already normalised). An occurrence of a term
    that starts / ends with a digit must not continue a longer number ("3500" is not found in "13500",
    "2.4" not in "2.4.1")."""
    out: list[tuple[int, int]] = []
    if not term:
        return out
    start = 0
    n = len(term)
    while True:
        k = text.find(term, start)
        if k < 0:
            break
        end = k + n
        ok = True
        if term[0].isdigit() and k > 0 and (text[k - 1].isdigit() or (text[k - 1] == "." and k > 1 and text[k - 2].isdigit())):
            ok = False
        if ok and term[-1].isdigit() and end < len(text) and (
                text[end].isdigit() or (text[end] == "." and end + 1 < len(text) and text[end + 1].isdigit())):
            ok = False
        if ok:
            out.append((k, end))
        start = k + 1
    return out


def _owner_runs(n: int, owner: Sequence[int] | None) -> list[tuple[int, int]]:
    """[start, end) of every maximal stretch of characters with the same owner (one stretch without owners)."""
    if owner is None or n == 0:
        return [(0, n)]
    runs: list[tuple[int, int]] = []
    start = 0
    for k in range(1, n + 1):
        if k == n or owner[k] != owner[start]:
            runs.append((start, k))
            start = k
    return runs


def term_hits(ref: str, ops: Sequence[Op], terms: Sequence[str], ref_owner: Sequence[int] | None = None) -> list[dict[str, Any]]:
    """One record per occurrence of a term in the reference: {"term", "start", "end", "owner", "ok"}.

    An occurrence is right ("ok") when every one of its reference characters is matched and nothing was
    inserted between them. With `ref_owner` (see concat) terms are searched inside each owner's text only, so
    a number at the end of one line never fuses with a number that opens the next line."""
    ok = [False] * len(ref)
    ins_before = [0] * (len(ref) + 1)      # insertions between reference characters i-1 and i
    pos = 0
    for op, i, _j in ops:
        if op == "=":
            ok[i] = True
            pos = i + 1
        elif op in ("s", "d"):
            pos = i + 1
        else:
            ins_before[pos] += 1
    out: list[dict[str, Any]] = []
    for lo, hi in _owner_runs(len(ref), ref_owner):
        piece = ref[lo:hi]
        for term in terms:
            for a, b in term_occurrences(piece, term):
                a, b = a + lo, b + lo
                good = all(ok[a:b]) and not any(ins_before[a + 1:b])
                out.append({"term": term, "start": a, "end": b, "owner": ref_owner[a] if ref_owner is not None else None,
                            "ok": good})
    return out


def term_stats(ref: str, ops: Sequence[Op], terms: Sequence[str], ref_owner: Sequence[int] | None = None) -> dict[str, dict[str, int]]:
    """{term: {"n": occurrences in the reference, "wrong": occurrences not reproduced exactly}} (see term_hits)."""
    out: dict[str, dict[str, int]] = {t: {"n": 0, "wrong": 0} for t in terms}
    for h in term_hits(ref, ops, terms, ref_owner):
        out[h["term"]]["n"] += 1
        out[h["term"]]["wrong"] += 0 if h["ok"] else 1
    return out


# ------------------------------------------------------------------------------------------------ time matching
def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_lines(utterances: Sequence[tuple[float, float]], lines: Sequence[tuple[float, float]]) -> list[dict[str, Any]]:
    """For each utterance (start, end) the truth line it overlaps most: {"line": index or None, "overlap": s,
    "share": overlap / utterance duration, "others": [(line index, overlap s), ...] sorted by overlap}."""
    out: list[dict[str, Any]] = []
    for u0, u1 in utterances:
        hits = [(k, overlap(u0, u1, l0, l1)) for k, (l0, l1) in enumerate(lines)]
        hits = sorted((h for h in hits if h[1] > 0), key=lambda h: (-h[1], h[0]))
        dur = max(u1 - u0, 1e-9)
        if not hits:
            out.append({"line": None, "overlap": 0.0, "share": 0.0, "others": []})
        else:
            out.append({"line": hits[0][0], "overlap": hits[0][1], "share": min(1.0, hits[0][1] / dur), "others": hits[1:]})
    return out


def line_coverage(utterances: Sequence[tuple[float, float]], lines: Sequence[tuple[float, float]]) -> list[float]:
    """Share of each truth line's duration that some utterance covers (union of the utterance intervals)."""
    merged: list[list[float]] = []
    for u0, u1 in sorted(utterances):
        if merged and u0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], u1)
        else:
            merged.append([u0, u1])
    out = []
    for l0, l1 in lines:
        cov = sum(overlap(l0, l1, m0, m1) for m0, m1 in merged)
        out.append(min(1.0, cov / max(l1 - l0, 1e-9)))
    return out
