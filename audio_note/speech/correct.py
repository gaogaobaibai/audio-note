"""LLM correction of ASR text as *edit pairs* with hard guardrails (DESIGN.md section 6).

The model never rewrites the whole text; it returns ``{"edits": [{"i", "from", "to", "reason"}]}``
and this module applies them locally under code-enforced rules:

* ``from`` must be a substring of the item's text; multiple occurrences replace the first
  not-yet-edited one.
* Digits must not change (the sequence of digit runs in ``from`` and ``to`` must be equal).
* If an item's accumulated character edit distance exceeds ``MAX_EDIT_RATIO`` of its length,
  all of that item's edits are dropped.
* Empty ``from`` and ``from == to`` are ignored.

Items are batched by ``correction.max_chars_per_call``; any failure in a batch (LLMError,
bad JSON, anything else) returns the originals for that batch and logs a warning.

Pure helpers usable without an LLM: :func:`apply_edits`, :func:`levenshtein`, :func:`load_glossary`.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from ..config import Config

log = logging.getLogger(__name__)

MAX_EDIT_RATIO = 0.30
"""Drop all edits of an item once their summed edit distance exceeds this fraction of its length."""

_DIGIT_RUN = re.compile(r"\d+")


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------
def levenshtein(a: str, b: str) -> int:
    """Character-level Levenshtein distance (two-row DP, no dependencies)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def digit_runs(s: str) -> list[str]:
    """Sequence of ASCII digit runs in ``s`` (``"第3季度2024年"`` -> ``["3", "2024"]``)."""
    return _DIGIT_RUN.findall(s)


def _norm_edit(e: Any) -> dict[str, str] | None:
    if not isinstance(e, dict):
        return None
    frm = e.get("from")
    to = e.get("to")
    if frm is None or to is None:
        return None
    return {"from": str(frm), "to": str(to), "reason": str(e.get("reason") or "")}


def apply_edits(text: str, edits: list[dict[str, Any]]) -> tuple[str, list[dict[str, str]]]:
    """Apply model-proposed edits to ``text`` under the guardrails.

    Returns ``(new_text, applied_edits)``. ``applied_edits`` are normalized dicts with
    ``from``/``to``/``reason`` in the order they were applied. When the accumulated edit
    distance of the applied edits exceeds ``MAX_EDIT_RATIO * len(text)``, the original text
    and an empty list are returned instead.
    """
    if not text or not edits:
        return text, []
    cur = text
    # spans (start, end) in *current* coordinates that were produced by an earlier edit
    replaced: list[tuple[int, int]] = []
    applied: list[dict[str, str]] = []
    total_dist = 0
    for raw in edits:
        e = _norm_edit(raw)
        if e is None:
            continue
        frm, to = e["from"], e["to"]
        if frm == "" or frm == to:
            continue
        if digit_runs(frm) != digit_runs(to):
            log.debug("edit dropped (digits change): %r -> %r", frm, to)
            continue
        idx = _find_unreplaced(cur, frm, replaced)
        if idx < 0:
            log.debug("edit dropped (not found / already edited): %r", frm)
            continue
        cur = cur[:idx] + to + cur[idx + len(frm):]
        delta = len(to) - len(frm)
        replaced = [(s + delta, t + delta) if s >= idx + len(frm) else (s, t) for s, t in replaced]
        replaced.append((idx, idx + len(to)))
        total_dist += levenshtein(frm, to)
        applied.append(e)
    if not applied:
        return text, []
    if total_dist > MAX_EDIT_RATIO * len(text):
        log.info("all %d edits dropped: edit distance %d > %.0f%% of %d chars",
                 len(applied), total_dist, MAX_EDIT_RATIO * 100, len(text))
        return text, []
    return cur, applied


def _find_unreplaced(cur: str, frm: str, replaced: list[tuple[int, int]]) -> int:
    """Index of the first occurrence of ``frm`` in ``cur`` that does not overlap an edited span."""
    pos = 0
    while True:
        idx = cur.find(frm, pos)
        if idx < 0:
            return -1
        end = idx + len(frm)
        if not any(idx < t and s < end for s, t in replaced):
            return idx
        pos = idx + 1


def load_glossary(path: str | os.PathLike | None) -> list[str]:
    """Read one term per line; blank lines and ``#`` comments are ignored. Missing file -> ``[]``."""
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    out: list[str] = []
    seen: set[str] = set()
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if " #" in s:
                s = s.split(" #", 1)[0].strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return out


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是中文语音识别（ASR）结果的校对助手。输入是一段对话的识别文本（中文为主，可能中英混合），按时间顺序分成若干条，每条有编号 i 和说话人。你的任务是找出**明显的识别错误**，以"编辑对"形式返回，绝不重写全文。

只允许三类修改：
1. 同音/近音错别字：例如"在见"→"再见"、"以经"→"已经"、"权利"（实为"权力"）。
2. 人名、专有名词、术语：优先参照下方词表；听起来像词表某个词的同音/近音片段，改成词表写法。说话人名字也在词表里。
3. 明显的分词/断句错误：标点放错位置、一个词被标点截断。

绝对禁止：
- 改动任何数字、日期、金额、时间、编号（包括汉字数字与阿拉伯数字互转）。
- 增加或删除句子，调整语序，翻译，把口语改成书面语，删除语气词（"嗯""啊""那个""就是说"）。
- 修改你不确定的内容。拿不准就不要改，宁可漏改也不要错改。

输出格式（JSON）：{"edits":[{"i":0,"from":"原文片段","to":"修改后","reason":"简短理由"}]}
- i 为该条的编号；from 必须是该条原文中**连续出现**的子串，尽量短（一个词或几个字），但要足够定位。
- 同一条里有多处错误就给多个编辑；没有需要修改的地方返回 {"edits":[]}。

示例 1（词表：李四）
输入：[{"i":0,"speaker":"张三","text":"我们明天在见，记得把报告发给李思。"}]
输出：{"edits":[{"i":0,"from":"在见","to":"再见","reason":"同音错别字"},{"i":0,"from":"李思","to":"李四","reason":"词表人名"}]}

示例 2
输入：[{"i":1,"speaker":"未知","text":"这个方案大概要三十万预算吧，嗯，我觉得可以，就是说要跟老板在确认一下。"}]
输出：{"edits":[{"i":1,"from":"在确认","to":"再确认","reason":"同音错别字"}]}
（数字"三十万"和口语词"嗯""就是说"都不动）

示例 3
输入：[{"i":2,"speaker":"王五","text":"下午三点开会，地点在302。"}]
输出：{"edits":[]}
"""


def build_system_prompt(glossary: list[str], speakers: list[str]) -> str:
    parts = [SYSTEM_PROMPT.rstrip()]
    terms = [t for t in glossary if t]
    if speakers:
        parts.append("说话人：" + "、".join(speakers))
    parts.append("词表：" + ("、".join(terms) if terms else "（空）"))
    return "\n\n".join(parts) + "\n"


def build_user_prompt(items: list[dict[str, Any]]) -> str:
    payload = [{"i": it["i"], "speaker": str(it.get("speaker") or ""), "text": str(it.get("text") or "")}
               for it in items]
    return "请校对以下语音识别结果，只返回 JSON：\n" + json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# corrector
# ---------------------------------------------------------------------------
class Corrector:
    """Edit-pair correction of ASR chunks via ``client.json`` (JSON mode)."""

    def __init__(self, cfg: Config, client: Any):
        self.cfg = cfg
        self.client = client
        c = cfg.correction
        self.enabled = bool(c.get("enabled", True))
        self.max_chars = max(1, int(c.get("max_chars_per_call", 3000)))
        self.unknown_label = str(cfg.get("speakers", {}).get("unknown_label", "未知"))

    # -- public --------------------------------------------------------------
    def correct(self, items: list[dict[str, Any]], glossary: list[str] | None = None) -> list[dict[str, Any]]:
        """``items = [{"i", "speaker", "text"}]`` -> ``[{"i", "text", "edits"}]`` in the same order."""
        glossary = list(glossary or [])
        if not items:
            return []
        if not self.enabled:
            return [self._original(it) for it in items]
        out: list[dict[str, Any]] = []
        for batch in self.batches(items):
            out.extend(self._correct_batch(batch, glossary))
        return out

    def batches(self, items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Group consecutive items so each group's total text length stays <= max_chars.
        An item longer than max_chars on its own forms a batch by itself."""
        groups: list[list[dict[str, Any]]] = []
        cur: list[dict[str, Any]] = []
        n = 0
        for it in items:
            ln = len(str(it.get("text") or ""))
            if cur and n + ln > self.max_chars:
                groups.append(cur)
                cur, n = [], 0
            cur.append(it)
            n += ln
        if cur:
            groups.append(cur)
        return groups

    # -- internals -----------------------------------------------------------
    @staticmethod
    def _original(it: dict[str, Any]) -> dict[str, Any]:
        return {"i": it["i"], "text": str(it.get("text") or ""), "edits": []}

    def _speakers(self, items: list[dict[str, Any]]) -> list[str]:
        names: list[str] = []
        for it in items:
            s = str(it.get("speaker") or "").strip()
            if s and s != self.unknown_label and s not in names:
                names.append(s)
        return names

    def _correct_batch(self, batch: list[dict[str, Any]], glossary: list[str]) -> list[dict[str, Any]]:
        originals = [self._original(it) for it in batch]
        if not any(o["text"].strip() for o in originals):
            return originals
        system = build_system_prompt(glossary, self._speakers(batch))
        user = build_user_prompt(batch)
        total = sum(len(o["text"]) for o in originals)
        max_tokens = int(min(8192, max(1024, total * 2)))
        try:
            resp = self.client.json(system, user, max_tokens=max_tokens)
            by_item = parse_edits(resp, [o["i"] for o in originals])
        except Exception as e:  # LLMError, ValueError, anything: never block the pipeline
            log.warning("correction batch failed (%d items, %d chars): %s", len(batch), total, e)
            return originals
        out: list[dict[str, Any]] = []
        for o in originals:
            edits = by_item.get(o["i"], [])
            new_text, applied = apply_edits(o["text"], edits)
            out.append({"i": o["i"], "text": new_text, "edits": applied})
        n_applied = sum(len(o["edits"]) for o in out)
        log.debug("correction batch: %d items, %d proposed, %d applied",
                  len(batch), sum(len(v) for v in by_item.values()), n_applied)
        return out


def parse_edits(resp: Any, item_ids: list[Any]) -> dict[Any, list[dict[str, Any]]]:
    """Tolerantly map a model response to ``{i: [edit, ...]}`` restricted to ``item_ids``.

    Accepts ``{"edits": [...]}``, a bare list, or a single edit dict. ``i`` may be a numeric
    string; a missing ``i`` is allowed only when there is exactly one item.
    """
    if isinstance(resp, dict):
        edits = resp.get("edits")
        if edits is None and "from" in resp:
            edits = [resp]
    elif isinstance(resp, list):
        edits = resp
    else:
        edits = None
    if edits is None:
        raise ValueError(f"no 'edits' in model response: {str(resp)[:200]!r}")
    if not isinstance(edits, list):
        raise ValueError("'edits' is not a list")
    valid = {}
    for i in item_ids:
        valid[i] = i
        valid[str(i)] = i
    by_item: dict[Any, list[dict[str, Any]]] = {}
    for e in edits:
        if not isinstance(e, dict):
            continue
        i = e.get("i")
        if i is None:
            if len(item_ids) != 1:
                continue
            key = item_ids[0]
        else:
            key = valid.get(i) if not isinstance(i, float) else valid.get(int(i))
            if key is None:
                key = valid.get(str(i).strip())
        if key is None:
            continue
        by_item.setdefault(key, []).append(e)
    return by_item
