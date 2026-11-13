"""Splitting one (mixed) utterance into several: Processor.preview_split / Processor.split_utterance.

The functions take the Processor as `proc` (db / cfg / engine / stop_event, the voice-stage helpers `_identify`,
`_match_clip`, `_system_exclude`, `_self_id`, the text-stage methods `_transcribe` / `_correct`, and `_rows` /
`_after_store` / `_embed_new` / `scratch`); Processor keeps the public methods and delegates here. The planning
helpers call each other through `proc` (`proc._pieces_auto(...)`), as they did when they were methods, so a
subclass or a test can still replace one of them. Names that were globals of the old pipeline.py are reached
through `F` (see chunking.py).

What a split stores
-------------------
One transaction replaces the utterance row by its pieces (fresh ids above every id ever used; the original's
vetoes are copied to every piece; FTS tokens per piece) AND brings the segment's `transcripts` row in line with
the utterances that exist afterwards (`refresh_transcript`):

- `raw_json`: one item per current utterance of the segment, in time order, `i` renumbered from 0. The item of an
  untouched utterance is kept as it was written when the segment was processed; the item of the removed row is
  gone; every piece gets a new item {"i", "start_s", "end_s", "cluster", "speaker_id", "speaker_name", "score",
  "text_raw", "text"}. Every item written here also carries "utterance_id" (items written by process_segment do
  not have it: they are matched to their rows by start_s / end_s).
- `correction_json`: the applied edits [{"i", "from", "to", "reason", ...}] of the untouched utterances
  (re-indexed to the new `i`) plus the edits the corrector made on the pieces; edits of the removed row are gone.
  NULL when no edit is left.
- `text_raw` / `text_corrected`: the items' texts, newline-joined.

So a split neither loses the pieces' correction edits nor leaves the transcript describing a row that no longer
exists. After the transaction the after-store hooks run for the pieces (Processor._after_store with
store_ctx.origin == "split"), exactly as they do for a freshly processed segment.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from ..audio.vad import SAMPLE_RATE
from .chunking import Chunk, F, log, stage_timer
from .speakers import Turn

SPLIT_MIN_PIECE_S = 0.6    # a split piece shorter than this is merged into the previous one
SPLIT_EDGE_S = 0.3         # a manual split point must lie inside (SPLIT_EDGE_S, duration - SPLIT_EDGE_S)
SPLIT_MIN_TURN_S = 1.0     # auto split: diarization turns shorter than this are ignored
SPLIT_PAD_S = 0.15         # ASR padding around a piece (same as build_chunks)


# --------------------------------------------------------------------------- planning (no ASR, no writes)
def _utterance_for_split(proc: Any, utt_id: int) -> tuple[Any, Any, np.ndarray]:
    """(utterance row, segment row, segment audio) or ValueError / FileNotFoundError."""
    u = proc.db.one("SELECT * FROM utterances WHERE id=?", (int(utt_id),))
    if u is None:
        raise ValueError("没有这条记录")
    seg = proc.db.get_segment(int(u["segment_id"]))
    if seg is None:
        raise ValueError(f"语句所属的录音段 #{u['segment_id']} 已不存在")
    path = Path(seg["path"])
    if not path.exists():
        raise FileNotFoundError("这段音频已不存在（超过保留期），无法拆分")
    return u, seg, F.read_segment(path)


def _pieces_manual(proc: Any, clip: np.ndarray, at: list[float], seg: Any = None, offset_s: float = 0.0,
                   veto: frozenset[int] = frozenset()) -> list[dict[str, Any]]:
    """Cut at `at` and identify every piece on its own audio; `veto` (the speakers the user rejected for the
    sentence being split) is excluded for every piece."""
    dur = len(clip) / SAMPLE_RATE
    edge = F.SPLIT_EDGE_S
    pts: list[float] = []
    for x in at:
        try:
            v = float(x)
        except (TypeError, ValueError):
            raise ValueError(f"拆分点应为秒数: {x!r}")
        if not (edge < v < dur - edge):
            raise ValueError(f"拆分点 {v:.1f}s 必须在 {edge:.1f}s 和 {dur - edge:.1f}s 之间")
        if pts and v < pts[-1]:
            raise ValueError("拆分点必须按时间升序")
        if not pts or v - pts[-1] > 1e-6:
            pts.append(v)
    if not pts:
        raise ValueError("需要至少一个拆分点")
    enrolled = proc.db.speaker_embeddings()
    thr = float(proc.cfg.speakers.identify_threshold)
    bounds = [0.0, *pts, dur]
    pieces: list[dict[str, Any]] = []
    for a, b in zip(bounds, bounds[1:]):
        excl = (proc._system_exclude(seg, offset_s + a, offset_s + b) if seg is not None else frozenset()) | veto
        sid, name, score = proc._match_clip(clip[int(a * SAMPLE_RATE): int(b * SAMPLE_RATE)], enrolled, thr, excl)
        pieces.append({"rel_start": a, "rel_end": b, "speaker_id": sid, "speaker_name": name, "score": score})
    return pieces


def _pieces_auto(proc: Any, clip: np.ndarray, seg: Any = None, offset_s: float = 0.0,
                 veto: frozenset[int] = frozenset()) -> list[dict[str, Any]]:
    """Diarize the utterance's audio, identify (cluster + turn level), merge neighbouring turns of the
    same speaker / cluster into pieces. Mostly system-audio turns are never the self speaker, and no turn is
    ever a speaker in `veto` (rejected by the user for the sentence being split)."""
    clip_s = len(clip) / SAMPLE_RATE
    with stage_timer(proc.db, "diarize", clip_s):
        turns = [t for t in proc.engine.diarize(clip) if t.duration >= F.SPLIT_MIN_TURN_S]
    if not turns:
        raise ValueError("没有分出第二个声音，请手动指定拆分点")
    if veto:
        for t in turns:
            t.exclude = frozenset(getattr(t, "exclude", ()) or ()) | veto
    pairs, block_s = F._segment_channels(seg) if seg is not None else (None, None)
    if pairs is not None:
        me = proc._self_id()
        for t in turns:
            sh = F.system_share(pairs, block_s, offset_s + t.start, offset_s + t.end)
            if me is not None and sh is not None and sh >= F.system_turn_share(proc.cfg):
                t.exclude = frozenset(getattr(t, "exclude", ()) or ()) | {me}
    with stage_timer(proc.db, "identify", clip_s):
        # seg_id None: the clip is not a whole segment, nothing is stored for it
        turns = sorted(proc._identify(clip, turns, None), key=lambda t: (t.start, t.end))
    unknown = proc.cfg.speakers.unknown_label
    for t in turns:
        if t.speaker_id is not None and t.speaker_id in (getattr(t, "exclude", ()) or ()):
            t.speaker_id, t.speaker_name = None, unknown
    pieces: list[dict[str, Any]] = []
    for t in turns:
        key = ("s", t.speaker_id) if t.speaker_id is not None else ("c", t.cluster)
        if pieces and pieces[-1]["_key"] == key:
            pieces[-1]["rel_end"] = max(pieces[-1]["rel_end"], float(t.end))
            continue
        pieces.append({"_key": key, "rel_start": float(t.start), "rel_end": float(t.end), "speaker_id": t.speaker_id,
                       "speaker_name": t.speaker_name if t.speaker_id is not None else None, "score": t.score})
    for pc in pieces:
        pc.pop("_key", None)
    if len(pieces) < 2:
        raise ValueError("没有分出第二个声音，请手动指定拆分点")
    return pieces


def _merge_short_pieces(pieces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    min_piece = F.SPLIT_MIN_PIECE_S
    out: list[dict[str, Any]] = []
    for pc in pieces:
        if out and pc["rel_end"] - pc["rel_start"] < min_piece:
            out[-1]["rel_end"] = pc["rel_end"]
            continue
        out.append(dict(pc))
    if len(out) > 1 and out[0]["rel_end"] - out[0]["rel_start"] < min_piece:
        nxt = out[1]
        nxt["rel_start"] = out[0]["rel_start"]
        out = out[1:]
    return out


def _plan_split(proc: Any, utt_id: int, at: list[float] | None, auto: bool) -> tuple[Any, Any, np.ndarray, list[dict[str, Any]]]:
    u, seg, audio = proc._utterance_for_split(utt_id)
    a, b = float(u["start_s"]), float(u["end_s"])
    clip = audio[int(a * SAMPLE_RATE): int(b * SAMPLE_RATE)]
    if len(clip) < int(2 * F.SPLIT_MIN_PIECE_S * SAMPLE_RATE):
        raise ValueError("这句太短，无法拆分")
    # the speakers the user vetoed for this sentence are never one of its pieces either
    veto = (frozenset(proc.db.rejected_speakers([int(u["id"])]).get(int(u["id"]), ()))
            if hasattr(proc.db, "rejected_speakers") else frozenset())
    if auto:
        pieces = proc._pieces_auto(clip, seg, a, veto)
    elif at:
        pieces = proc._pieces_manual(clip, list(at), seg, a, veto)
    else:
        raise ValueError("需要给出拆分点（at）或选择自动拆分（auto）")
    pieces = proc._merge_short_pieces(pieces)
    if len(pieces) < 2:
        raise ValueError("拆分点太靠近两端，只剩一段，没有拆分")
    unknown = proc.cfg.speakers.unknown_label
    for pc in pieces:
        if pc.get("speaker_id") is not None and int(pc["speaker_id"]) in veto:
            pc["speaker_id"], pc["speaker_name"] = None, None     # a veto always wins
        pc["start_s"] = round(a + float(pc["rel_start"]), 3)
        pc["end_s"] = round(a + float(pc["rel_end"]), 3)
        pc["rel_start"] = round(float(pc["rel_start"]), 3)
        pc["rel_end"] = round(float(pc["rel_end"]), 3)
        pc["speaker"] = pc["speaker_name"] or unknown
        pc["score"] = round(float(pc["score"]), 3) if pc.get("score") is not None else None
    return u, seg, audio, pieces


def preview_split(proc: Any, utt_id: int, at: list[float] | None = None, auto: bool = True) -> dict[str, Any]:
    """The pieces split_utterance would produce (bounds + identification), without ASR or DB writes."""
    proc.scratch = {}
    try:
        u, _seg, _audio, pieces = proc._plan_split(utt_id, at, auto)
    finally:
        proc.scratch = {}       # nothing is stored: whatever the voice helpers kept for a hook is dropped
    return {"utterance_id": int(u["id"]), "segment_id": int(u["segment_id"]), "start_s": float(u["start_s"]),
            "end_s": float(u["end_s"]), "pieces": [{k: pc[k] for k in ("start_s", "end_s", "rel_start", "rel_end",
                                                                         "speaker", "speaker_id", "score")} for pc in pieces]}


# --------------------------------------------------------------------------- the transcript of a split segment
def _json_list(text: Any) -> list[Any]:
    if not text:
        return []
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _bounds_key(start_s: Any, end_s: Any) -> tuple[float, float] | None:
    try:
        return round(float(start_s), 3), round(float(end_s), 3)
    except (TypeError, ValueError):
        return None


def refresh_transcript(con: sqlite3.Connection, cfg: Any, seg_id: int,
                       pieces: dict[int, tuple[Chunk, str, str]],
                       piece_edits: dict[int, list[dict[str, Any]]]) -> None:
    """Rewrite the segment's transcripts row so that it describes the utterances that exist NOW (see the module
    docstring for the stored shape). Runs on `con` inside the split transaction, after the row was replaced.

    pieces: {utterance id: (chunk, raw text, stored text)} of the rows just inserted;
    piece_edits: {utterance id: [applied edit, ...]} for them (an edit's own "i" is ignored).
    Old items are matched to the current rows by "utterance_id" when they have one, else by (start_s, end_s)."""
    tr = con.execute("SELECT id, raw_json, correction_json FROM transcripts WHERE segment_id=? ORDER BY id DESC LIMIT 1",
                     (int(seg_id),)).fetchone()
    old_items = [it for it in _json_list(tr[1]) if isinstance(it, dict)] if tr is not None else []
    old_edits: dict[int, list[dict[str, Any]]] = {}
    for e in (_json_list(tr[2]) if tr is not None else []):
        if isinstance(e, dict):
            try:
                old_edits.setdefault(int(e.get("i")), []).append(e)
            except (TypeError, ValueError):
                continue
    by_uid: dict[int, dict[str, Any]] = {}
    by_bounds: dict[tuple[float, float], dict[str, Any]] = {}
    for pos, it in enumerate(old_items):
        it.setdefault("i", pos)
        key = _bounds_key(it.get("start_s"), it.get("end_s"))
        if isinstance(it.get("utterance_id"), int):
            by_uid[int(it["utterance_id"])] = it
        elif key is not None:
            by_bounds.setdefault(key, it)

    items: list[dict[str, Any]] = []
    edits: list[dict[str, Any]] = []
    rows = con.execute(
        "SELECT id, start_s, end_s, cluster_label, speaker_id, speaker_name, speaker_score, text_raw, text"
        " FROM utterances WHERE segment_id=? ORDER BY start_s, id", (int(seg_id),)).fetchall()
    for j, r in enumerate(rows):
        uid = int(r[0])
        if uid in pieces:
            chunk, raw, text = pieces[uid]
            item = F.transcript_item(j, chunk, raw, text)
            carried = piece_edits.get(uid, [])
        else:
            old = by_uid.pop(uid, None)
            if old is None:
                key = _bounds_key(r[1], r[2])
                old = by_bounds.pop(key, None) if key is not None else None
            if old is not None:
                carried = old_edits.get(int(old["i"]), []) if isinstance(old.get("i"), int) else []
                item = dict(old)
                item["i"] = j
            else:       # a row the transcript never described (older data): described from the row itself
                carried = []
                item = {"i": j, "start_s": round(float(r[1]), 3), "end_s": round(float(r[2]), 3), "cluster": r[3],
                        "speaker_id": r[4], "speaker_name": r[5], "score": r[6], "text_raw": r[7], "text": r[8]}
        item["utterance_id"] = uid
        items.append(item)
        for e in carried:
            edits.append({**e, "i": j})
    raw_json = json.dumps(items, ensure_ascii=False)
    text_raw = "\n".join(str(it.get("text_raw") or "") for it in items)
    text_corrected = "\n".join(str(it.get("text") or "") for it in items)
    correction_json = json.dumps(edits, ensure_ascii=False) if edits else None
    if tr is not None:
        con.execute("UPDATE transcripts SET raw_json=?, text_raw=?, text_corrected=?, correction_json=? WHERE id=?",
                    (raw_json, text_raw, text_corrected, correction_json, int(tr[0])))
    else:
        con.execute(
            "INSERT INTO transcripts(segment_id, asr_model, raw_json, text_raw, text_corrected, correction_json, language)"
            " VALUES (?,?,?,?,?,?,?)",
            (int(seg_id), cfg.mimo.asr_model, raw_json, text_raw, text_corrected, correction_json, cfg.mimo.asr_language))


# --------------------------------------------------------------------------- the split
def split_utterance(proc: Any, utt_id: int, at: list[float] | None = None, auto: bool = False) -> dict[str, Any]:
    """Split one (mixed) utterance into several: `auto` diarizes + identifies its audio, `at` cuts at the
    given seconds (relative to the utterance start). Every piece is transcribed again (ASR + correction) and
    replaces the original row inside one transaction (tokens / vectors cascade); piece 0 keeps the cluster
    label, the others get fresh labels; the segment's transcripts row, cluster_embeddings and the day Markdown
    are refreshed (see the module docstring for what the transcript holds afterwards).
    Returns {"removed", "pieces": [{id, start_s, end_s, speaker, speaker_id, score, text}], "billed_s"}.
    An ASR failure (LLMError) leaves the database untouched."""
    proc.scratch = {}
    try:
        return _split_utterance(proc, utt_id, at, auto)
    finally:
        proc.store_ctx, proc.scratch = None, {}


def _split_utterance(proc: Any, utt_id: int, at: list[float] | None, auto: bool) -> dict[str, Any]:
    db, cfg = proc.db, proc.cfg
    u, seg, audio, pieces = proc._plan_split(utt_id, at, auto)
    seg_id = int(u["segment_id"])
    audio_len_s = len(audio) / SAMPLE_RATE
    F._parse_iso(seg["started_at"])     # a broken row fails here, before ASR is paid for
    base_cluster = int(u["cluster_label"]) if u["cluster_label"] is not None else 0
    top = db.one("SELECT MAX(cluster_label) AS m FROM utterances WHERE segment_id=?", (seg_id,))
    next_cluster = int(top["m"]) + 1 if top and top["m"] is not None else base_cluster + 1
    chunks: list[Chunk] = []
    for k, pc in enumerate(pieces):
        cl = base_cluster if k == 0 else next_cluster + k - 1
        c = Chunk(float(pc["start_s"]), float(pc["end_s"]), cl, pc["speaker_id"], pc["speaker_name"], pc["score"])
        c.audio_start_s = max(0.0, c.start_s - F.SPLIT_PAD_S)
        c.audio_end_s = min(audio_len_s, c.end_s + F.SPLIT_PAD_S)
        chunks.append(c)
    proc._register_terms()
    with stage_timer(db, "asr", sum(c.audio_end_s - c.audio_start_s for c in chunks)):
        spoken, billed, _filtered = proc._transcribe(seg, audio, chunks)
    if len(spoken) != len(chunks):
        # every piece must come back with text, or the sentence would lose part of what was said
        raise ValueError("有一段没有转写出文字（或被内容过滤），没有修改这句")
    with stage_timer(db, "correct"):
        corrected, edits_json = proc._correct(spoken)
    edits = F.edits_list(edits_json)
    unknown = cfg.speakers.unknown_label
    rows = proc._rows(seg, audio, spoken, corrected)
    days: set[str] = {str(u["day"])} | {str(r["day"]) for r in rows}
    ids: list[int] = []
    with stage_timer(db, "store"):
        with db.transaction() as con:
            # the original sentence's vetoes go to every piece (read before the DELETE: they cascade with the row)
            vetoed = [int(r[0]) for r in con.execute(
                "SELECT speaker_id FROM utterance_rejections WHERE utterance_id=?", (int(utt_id),)).fetchall()]
            # the highest id so far (the row being replaced may be it: its id must not come back for a piece)
            top_id = int(con.execute("SELECT COALESCE(MAX(id),0) FROM utterances").fetchone()[0])
            # the row must still be the one we planned on (ASR took seconds: another split / relabel may have replaced it)
            cur = con.execute("DELETE FROM utterances WHERE id=? AND start_s=? AND end_s=?",
                              (int(utt_id), float(u["start_s"]), float(u["end_s"])))
            if int(cur.rowcount or 0) != 1:
                raise ValueError("这句已经被修改或拆分过了，请刷新")
            columns = F.utterance_columns(con)
            for r in rows:
                # new ids above every id ever used, so the page never confuses a piece with the row it replaced
                uid = F.insert_utterance(con, r, columns, above_id=top_id)
                ids.append(uid)
                for vs in vetoed:
                    con.execute("INSERT OR IGNORE INTO utterance_rejections(utterance_id, speaker_id) VALUES (?,?)", (uid, vs))
            # the transcript follows the rows: the pieces' edits are kept, the removed row is no longer described
            piece_edits: dict[int, list[dict[str, Any]]] = {}
            for e in edits:
                try:
                    piece_edits.setdefault(ids[int(e.get("i"))], []).append(e)
                except (TypeError, ValueError, IndexError):
                    continue
            F.refresh_transcript(con, cfg, seg_id,
                                 {uid: (c, raw, rows[i]["text"]) for i, (uid, (c, raw)) in enumerate(zip(ids, spoken))},
                                 piece_edits)
    day_list = sorted(days)
    proc.store_ctx = F.StoreContext("split", seg, audio, spoken, rows, edits, ids, day_list,
                                    replaced_utterance_id=int(utt_id), extra=proc.scratch)
    proc._after_store(seg_id, ids, day_list)
    proc._embed_new(ids)
    cluster_embedding = getattr(proc.engine, "cluster_embedding", None)
    if callable(cluster_embedding):
        for cl in sorted({c.cluster for c, _ in spoken}):
            ts = [Turn(float(r["start_s"]), float(r["end_s"]), cl) for r in db.query(
                "SELECT start_s, end_s FROM utterances WHERE segment_id=? AND cluster_label=? ORDER BY start_s, id", (seg_id, cl))]
            try:
                emb = cluster_embedding(audio, ts)
                if emb is not None:
                    db.upsert_cluster_embedding(seg_id, cl, emb, sum(t.duration for t in ts))
            except Exception as e:  # noqa: BLE001 - the rows are in; the voice grouping just lacks this cluster
                log.warning("split: cluster embedding for #%d/%d failed: %s", seg_id, cl, e)
    F.write_days_markdown(db, cfg, day_list)
    out_pieces = [{"id": uid, "start_s": round(c.start_s, 3), "end_s": round(c.end_s, 3), "speaker": c.speaker_name or unknown,
                   "speaker_id": c.speaker_id, "score": round(float(c.score), 3) if c.score is not None else None,
                   "text": rows[i]["text"]}
                  for i, (uid, (c, _raw)) in enumerate(zip(ids, spoken))]
    log.info("split utterance #%d (segment #%d) into %d pieces (%.1fs billed)", int(utt_id), seg_id, len(ids), billed)
    return {"removed": int(utt_id), "pieces": out_pieces, "billed_s": round(float(billed), 1)}


__all__ = ["SPLIT_EDGE_S", "SPLIT_MIN_PIECE_S", "SPLIT_MIN_TURN_S", "SPLIT_PAD_S", "preview_split", "refresh_transcript",
           "split_utterance"]
