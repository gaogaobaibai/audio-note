"""Relation inference: how does each enrolled speaker relate to the user (朋友 / 同事 / 领导 ...)?

    infer_relations(cfg, db, client)                      # every candidate speaker
    infer_relations(cfg, db, client, speaker_ids=[3], force=True)
    maybe_infer_after_daily(cfg, db, client)              # called after the daily digest; [] when disabled / no key

Candidates: speakers that are not the user themselves, whose relation was not set by hand
(relation_source != 'user' — a user label is never overwritten, not even with force), and that have
at least cfg.speakers.relation_min_utterances utterances. The utterance count at the last inference
is cached in kv `relation_inferred_n:<sid>`; a speaker is only re-asked once that count has grown
by 50 % (or with force=True). A cached count above the current one (utterances re-identified away,
a speaker deleted and re-created) is stale and re-asked; clearing a relation or deleting the speaker
drops the cache (db.set_relation / db.delete_speaker).

The 'auto' write is conditional (db.set_relation): a relation the user set while the LLM call was in
flight wins, and the result then reports applied=False.

Evidence per speaker: up to MAX_EVIDENCE of their utterances, sampled evenly across days, each with
the previous and next utterance of the same recording (speakers by name, the user as 「本人」), plus
the daily analysis's person notes for that name. One JSON-mode LLM call per speaker; an LLMError
for one speaker is logged and does not stop the others.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from ..config import is_placeholder_key
from ..db import RELATION_KV_PREFIX
from ..llm.client import LLMError

log = logging.getLogger(__name__)

RELATION_CHOICES = ("家人", "伴侣", "朋友", "好朋友", "同事", "领导", "下属", "导师", "学生", "客户", "合作伙伴", "同学", "其他")
UNSURE = "不确定"

MAX_EVIDENCE = 60          # utterances of the speaker handed to the model
MAX_LINE_CHARS = 200       # one utterance line in the evidence
MAX_NOTE_CHARS = 600       # person notes from memory_notes
KV_PREFIX = RELATION_KV_PREFIX     # "relation_inferred_n:"
CACHE_GROWTH = 1.5         # re-infer only once the utterance count grew this much
DEFAULT_MIN_UTTERANCES = 15
DEFAULT_MIN_CONFIDENCE = 0.6

SELF_LABEL = "本人"

SYSTEM_TEMPLATE = (
    "你是用户的私人记录助手。用户全天佩戴录音设备，下面是从转写记录里抽出的若干对话片段，"
    "每行格式为「HH:MM 说话人：内容」。{self_note}\n"
    "请判断「{name}」和用户本人是什么关系。只能根据给出的对话内容判断（称呼、谈话内容、语气、一起做的事），"
    "不要凭常识或名字臆测；证据不足就回答“{unsure}”。\n"
    "候选关系（relation 只能取其中之一，或“{unsure}”）：{choices}。\n"
    "必须只输出一个 JSON 对象，不要输出其他内容。JSON 结构（键名固定）：\n"
    '{{"relation": "候选之一或“{unsure}”", "confidence": 0到1之间的小数, '
    '"reason": "不超过60字，引用对话里的称呼或事件作为依据", "address": "{name} 对用户本人的称呼，没有就留空"}}'
)

SELF_NOTE_KNOWN = "记录里标为「本人」的发言就是用户本人（{self_name}）说的。"
SELF_NOTE_UNKNOWN = "用户本人没有登记，记录里没有标出哪位是本人；对话里的第一人称（我）可能是本人，未识别的说话人标为「{unknown}」。"

USER_TEMPLATE = (
    "说话人：{name}（记录里共 {total} 条语句，下面抽取了 {shown} 条及其前后文，按时间排列；"
    "「>>」标出的是 {name} 的话）\n"
    "\n"
    "=== 对话片段开始 ===\n"
    "{evidence}\n"
    "=== 对话片段结束 ===\n"
    "{notes}"
    "\n"
    "请按约定的 JSON 结构判断「{name}」和用户本人的关系。"
)


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _cfg_int(cfg: Any, key: str, default: int) -> int:
    try:
        return int(cfg.speakers.get(key, default))
    except (TypeError, ValueError, AttributeError):
        return default


def _cfg_float(cfg: Any, key: str, default: float) -> float:
    try:
        return float(cfg.speakers.get(key, default))
    except (TypeError, ValueError, AttributeError):
        return default


def _evenly(items: list[Any], k: int) -> list[Any]:
    """k items spread evenly over the list (all of them when k >= len)."""
    n = len(items)
    if k <= 0 or n == 0:
        return []
    if k >= n:
        return list(items)
    if k == 1:
        return [items[n // 2]]
    return [items[round(i * (n - 1) / (k - 1))] for i in range(k)]


def _sample_across_days(rows: list[sqlite3.Row], k: int) -> list[sqlite3.Row]:
    """Up to k rows: every day gets a share (round-robin), evenly spaced within the day, time order kept."""
    if len(rows) <= k:
        return list(rows)
    by_day: dict[str, list[sqlite3.Row]] = {}
    for r in rows:
        by_day.setdefault(str(r["day"] or ""), []).append(r)
    days = sorted(by_day)
    quota = {d: 0 for d in days}
    remaining = k
    while remaining > 0:
        progressed = False
        for d in days:
            if remaining <= 0:
                break
            if quota[d] < len(by_day[d]):
                quota[d] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    out: list[sqlite3.Row] = []
    for d in days:
        out += _evenly(by_day[d], quota[d])
    return out


def _clip(text: Any, limit: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _hhmm(abs_start: Any) -> str:
    s = str(abs_start or "").replace("T", " ")
    return s[11:16] if len(s) >= 16 else s


class _Labeler:
    def __init__(self, self_id: int | None, unknown: str):
        self.self_id = self_id
        self.unknown = unknown

    def __call__(self, row: sqlite3.Row) -> str:
        sid = row["speaker_id"]
        if sid is not None and self.self_id is not None and int(sid) == self.self_id:
            return SELF_LABEL
        return str(row["speaker_name"] or self.unknown)


def build_evidence(db: Any, sid: int, *, unknown: str, max_items: int = MAX_EVIDENCE) -> tuple[str, int, int]:
    """(evidence text, total utterances of the speaker, number shown). Each shown utterance comes with the
    previous and next utterance of the same recording segment."""
    rows = db.query(
        "SELECT id, segment_id, day, abs_start, start_s, speaker_id, speaker_name, text, text_raw"
        " FROM utterances WHERE speaker_id=? ORDER BY abs_start, id", (int(sid),))
    total = len(rows)
    if not rows:
        return "", 0, 0
    picked = _sample_across_days(rows, max_items)
    seg_ids = sorted({int(r["segment_id"]) for r in picked})
    marks = ",".join("?" for _ in seg_ids)
    seg_rows = db.query(
        "SELECT id, segment_id, day, abs_start, start_s, speaker_id, speaker_name, text, text_raw"
        f" FROM utterances WHERE segment_id IN ({marks}) ORDER BY segment_id, start_s, id", seg_ids)
    by_seg: dict[int, list[sqlite3.Row]] = {}
    pos: dict[int, int] = {}
    for r in seg_rows:
        lst = by_seg.setdefault(int(r["segment_id"]), [])
        pos[int(r["id"])] = len(lst)
        lst.append(r)
    self_row = db.self_speaker()
    label = _Labeler(int(self_row["id"]) if self_row else None, unknown)

    def line(r: sqlite3.Row, mark: str) -> str:
        text = _clip(r["text"] or r["text_raw"], MAX_LINE_CHARS)
        return f"{mark}{_hhmm(r['abs_start'])} {label(r)}：{text}"

    out: list[str] = []
    last_day: str | None = None
    for r in picked:
        day = str(r["day"] or "")
        if day != last_day:
            out.append(f"【{day}】")
            last_day = day
        lst = by_seg.get(int(r["segment_id"]), [])
        i = pos.get(int(r["id"]))
        if i is not None and i > 0:
            out.append(line(lst[i - 1], "   "))
        out.append(line(r, ">> "))
        if i is not None and i + 1 < len(lst):
            out.append(line(lst[i + 1], "   "))
        out.append("")
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out), total, len(picked)


def person_notes(db: Any, name: str) -> str:
    """Content of the daily analysis's person notes for this name ('' when none)."""
    rows = db.query("SELECT content, day_from, day_to FROM memory_notes WHERE kind='person' AND title=? ORDER BY updated_at DESC, id DESC",
                    (name,))
    parts: list[str] = []
    for r in rows:
        span = str(r["day_from"] or "")
        if r["day_to"] and r["day_to"] != r["day_from"]:
            span += f"~{r['day_to']}"
        content = " ".join(str(r["content"] or "").split())
        if content:
            parts.append(f"{content}" + (f"（{span}）" if span and span not in content else ""))
    if not parts:
        return ""
    return _clip("；".join(parts), MAX_NOTE_CHARS)


def build_prompt(name: str, *, self_name: str | None, unknown: str, evidence: str, total: int, shown: int,
                 notes: str = "") -> tuple[str, str]:
    """(system, user) for one speaker."""
    self_note = SELF_NOTE_KNOWN.format(self_name=self_name) if self_name else SELF_NOTE_UNKNOWN.format(unknown=unknown)
    system = SYSTEM_TEMPLATE.format(self_note=self_note, name=name, unsure=UNSURE, choices="、".join(RELATION_CHOICES))
    notes_block = f"\n日报里关于「{name}」的笔记（可能过时，仅供参考）：{notes}\n" if notes else ""
    user = USER_TEMPLATE.format(name=name, total=total, shown=shown, evidence=evidence, notes=notes_block)
    return system, user


def parse_result(raw: Any) -> dict[str, Any]:
    """Normalise the model's JSON: relation (str), confidence (0..1 float), reason, address."""
    d = raw if isinstance(raw, dict) else {}
    relation = " ".join(str(d.get("relation") or "").split()).strip("「」“”\"' ")
    conf_raw = d.get("confidence")
    try:
        conf = float(conf_raw) if conf_raw not in (None, "") else 0.0
    except (TypeError, ValueError):
        conf = 0.0
    if conf > 1.0 and conf <= 100.0:   # "80" / 80 -> 0.8
        conf = conf / 100.0
    conf = max(0.0, min(1.0, conf))
    reason = _clip(d.get("reason"), 120)
    address = _clip(d.get("address"), 40)
    return {"relation": relation, "confidence": conf, "reason": reason, "address": address}


# ----------------------------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------------------------
def _result(sid: int, name: str, *, relation: str | None = None, confidence: float = 0.0, reason: str = "",
            applied: bool = False, skipped: str | None = None, address: str = "") -> dict[str, Any]:
    return {"id": int(sid), "name": name, "relation": relation, "confidence": float(confidence), "reason": reason,
            "address": address, "applied": bool(applied), "skipped": skipped}


def infer_relations(cfg: Any, db: Any, client: Any, speaker_ids: list[int] | None = None, force: bool = False) -> list[dict[str, Any]]:
    """Ask the LLM how each candidate speaker relates to the user and store confident answers
    (`speakers.relation`, source 'auto'). Returns one dict per examined speaker:
    {id, name, relation, confidence, reason, address, applied, skipped}; `skipped` is the reason a
    speaker was not sent to the model (self / labelled by the user / too few utterances / cached / LLM error)."""
    min_utts = max(1, _cfg_int(cfg, "relation_min_utterances", DEFAULT_MIN_UTTERANCES))
    min_conf = _cfg_float(cfg, "relation_min_confidence", DEFAULT_MIN_CONFIDENCE)
    unknown = str(cfg.speakers.get("unknown_label") or "未知")
    model = str(cfg.mimo.get("analysis_model") or "") if hasattr(cfg, "mimo") else ""

    people = db.people_table()
    if speaker_ids is not None:
        wanted = {int(s) for s in speaker_ids}
        people = [p for p in people if p["id"] in wanted]
    self_row = db.self_speaker()
    self_name = str(self_row["name"]) if self_row else None

    out: list[dict[str, Any]] = []
    for p in people:
        sid, name, n = int(p["id"]), str(p["name"]), int(p["utterances"])
        if p["is_self"]:
            out.append(_result(sid, name, skipped="本人"))
            continue
        if p["relation_source"] == "user":
            out.append(_result(sid, name, relation=p["relation"], skipped="用户已标注，不覆盖"))
            continue
        if n < min_utts:
            out.append(_result(sid, name, relation=p["relation"], skipped=f"语句太少（{n} < {min_utts}）"))
            continue
        key = f"{KV_PREFIX}{sid}"
        prev = db.get_kv(key)
        if not force and prev is not None:
            try:
                prev_n = int(prev)
            except (TypeError, ValueError):
                prev_n = None
            if prev_n is not None and prev_n <= n < prev_n * CACHE_GROWTH:
                out.append(_result(sid, name, relation=p["relation"], skipped=f"已推断过（当时 {prev_n} 条，现在 {n} 条）"))
                continue
            if prev_n is not None and prev_n > n:
                log.info("relation cache for %s is stale (%d > %d utterances); asking again", name, prev_n, n)

        evidence, total, shown = build_evidence(db, sid, unknown=unknown)
        if not evidence:
            out.append(_result(sid, name, relation=p["relation"], skipped="没有可用的语句"))
            continue
        system, user = build_prompt(name, self_name=self_name, unknown=unknown, evidence=evidence, total=total, shown=shown,
                                    notes=person_notes(db, name))
        kw: dict[str, Any] = {"max_tokens": 512}
        if model:
            kw["model"] = model
        try:
            res = parse_result(client.json(system, user, **kw))
        except LLMError as e:
            log.warning("relation inference for %s (id %d) failed: %s", name, sid, e)
            out.append(_result(sid, name, relation=p["relation"], skipped=f"LLM 调用失败：{e}"))
            continue
        applied = res["confidence"] >= min_conf and res["relation"] in RELATION_CHOICES
        if applied:
            # conditional: a relation the user set while the model was thinking is kept
            applied = bool(db.set_relation(sid, res["relation"], "auto", note=res["reason"] or None))
            if applied:
                log.info("relation inferred: %s -> %s (%.2f) %s", name, res["relation"], res["confidence"], res["reason"])
            else:
                log.info("relation for %s inferred as %s but the user labelled it meanwhile; kept theirs", name, res["relation"])
        else:
            log.info("relation for %s not applied: %r (%.2f) %s", name, res["relation"], res["confidence"], res["reason"])
        db.set_kv(key, total)
        out.append(_result(sid, name, relation=res["relation"] or UNSURE, confidence=res["confidence"], reason=res["reason"],
                           address=res["address"], applied=applied))
    return out


def maybe_infer_after_daily(cfg: Any, db: Any, client: Any) -> list[dict[str, Any]]:
    """Run infer_relations after the daily digest unless speakers.infer_relations is off or there is no
    usable LLM (no client / no API key). Never raises: a failure is logged and [] returned."""
    try:
        enabled = bool(cfg.speakers.get("infer_relations", True))
    except AttributeError:
        enabled = True
    if not enabled:
        log.debug("relation inference disabled (speakers.infer_relations)")
        return []
    if client is None:
        return []
    try:
        key = cfg.mimo.get("api_key")
    except AttributeError:
        key = None
    if is_placeholder_key(key):
        log.debug("relation inference skipped: no API key")
        return []
    try:
        results = infer_relations(cfg, db, client)
    except Exception as e:  # noqa: BLE001 - best effort after the digest
        log.warning("relation inference after daily failed: %s", e)
        return []
    n_applied = sum(1 for r in results if r.get("applied"))
    if n_applied:
        log.info("relations inferred after daily: %d applied", n_applied)
    return results


__all__ = ["RELATION_CHOICES", "UNSURE", "infer_relations", "maybe_infer_after_daily", "build_evidence", "build_prompt",
           "parse_result", "person_notes"]
