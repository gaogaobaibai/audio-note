"""LLM query expansion: synonyms / brand names / colloquial variants of a search query.

Keyword search misses paraphrases ("扫地机" vs the transcript's "科沃斯"); one cheap JSON call
produces alternative terms that are OR-ed into the FTS query. Results are cached per process.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from ..llm.client import LLMError

log = logging.getLogger(__name__)

SYSTEM = (
    "你是检索助手。用户在自己日常对话的录音转写（口语、中文为主）里搜索。给定一个查询，列出转写里可能出现的其他说法："
    "同义词、口语说法、具体的品牌/产品/地点名、上位或下位概念、相关动作或结果。"
    '只输出 JSON：{"terms": [...]}，最多 8 个词，每个 1–6 个字，不要重复查询里已有的词，不要解释。'
)

MAX_TERMS = 8
_WS = re.compile(r"\s+")
_STRIP = "\"'“”‘’,.;:，。；：、"


class QueryExpander:
    def __init__(self, client: Any, model: str | None = None, max_terms: int = MAX_TERMS):
        self.client = client
        self.model = model
        self.max_terms = max(1, int(max_terms))
        self.cache: dict[str, list[str]] = {}

    def expand(self, query: str) -> list[str]:
        q = _WS.sub(" ", (query or "").strip())
        if not q:
            return []
        if q in self.cache:
            return self.cache[q]
        terms: list[str] = []
        try:
            data = self.client.json(SYSTEM, f"查询：{q}", model=self.model, max_tokens=300)
            terms = clean_terms(data, q, self.max_terms)
        except LLMError as e:
            log.warning("query expansion failed for %r: %s", q, e)
        except Exception as e:  # noqa: BLE001 - never break a search because of expansion
            log.warning("query expansion error for %r: %s", q, e)
        self.cache[q] = terms
        return terms


def clean_terms(data: Any, query: str, max_terms: int = MAX_TERMS) -> list[str]:
    """Normalise the model's JSON: accept {"terms": [...]} or a bare list; drop junk, dupes and
    anything already contained in the query."""
    raw = data.get("terms") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for t in raw:
        if not isinstance(t, str):
            continue
        t = _WS.sub("", t.strip().strip(_STRIP))
        if not t or len(t) > 12 or t in seen or t in query:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out
