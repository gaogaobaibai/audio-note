"""Function-calling tools the memory chat exposes to the LLM, and their executor.

TOOLS is an OpenAI-style tool list (passed as `tools=` to chat.completions). `run_tool()` dispatches
a call by name, always returns a compact string (<= MAX_CHARS) and never raises: any failure is
rendered as "ERROR: ...", so the model can recover.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, timedelta
from typing import Any, Callable

from ..db import DB
from . import index

log = logging.getLogger(__name__)

MAX_CHARS = 4000
_TRUNC = "\n…(输出已截断，请缩小范围或换更具体的关键词)"

_DAY = {"type": "string", "description": "日期 YYYY-MM-DD（也接受 today/yesterday/今天/昨天）"}

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_transcripts",
            "description": "在全部录音转写记录中检索（关键词全文索引 + 语义向量，自动融合），返回匹配的句子及其时间、说话人、utterance_id。"
                           "可用自然语言短语或关键词；可多次调用、换说法；可按日期范围和说话人过滤。找不到时会自动做子串匹配。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "关键词或短语，多个词用空格分开"},
                    "day_from": {**_DAY, "description": "起始日期（含），可选"},
                    "day_to": {**_DAY, "description": "结束日期（含），可选"},
                    "speaker": {"type": "string", "description": "只看某个说话人的话（精确姓名），可选"},
                    "limit": {"type": "integer", "description": "最多返回条数，默认 20"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_transcript",
            "description": "按时间顺序读取某一天（可限定小时范围）的完整转写记录。内容太长会被截断，请用 hour_from/hour_to 缩小范围。",
            "parameters": {
                "type": "object",
                "properties": {
                    "day": _DAY,
                    "hour_from": {"type": "number", "description": "起始小时 0-24（含），可用小数如 8.5，默认 0"},
                    "hour_to": {"type": "number", "description": "结束小时 0-24（不含），默认 24"},
                },
                "required": ["day"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_summary",
            "description": "读取某一天由 LLM 生成的日报（概览/话题/决定/待办/人物）。适合先了解当天整体情况。",
            "parameters": {"type": "object", "properties": {"day": _DAY}, "required": ["day"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_days",
            "description": "列出有记录的日期（最新在前，最多 60 天）及每天的句子数、时间跨度、是否有日报。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_speakers",
            "description": "列出已登记的说话人（标出哪位是用户本人、与本人的关系）及其备注、记录条数，以及未识别说话人的条数。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "检索长期记忆条目（系统从对话里整理出来、用户可以改）：人物 person、话题 topic、待办 todo、决定 decision、"
                           "事实 fact、偏好 preference、项目 project、事件 event、备注 note。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "标题/内容中的关键词，可选（为空则列出最近的笔记）"},
                    "kind": {"type": "string", "enum": ["person", "topic", "todo", "decision", "fact", "preference",
                                                        "project", "event", "note"],
                             "description": "笔记类型，可选"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_context",
            "description": "查看某条记录（utterance_id，来自检索结果中的 #编号）前后的对话上下文。",
            "parameters": {
                "type": "object",
                "properties": {
                    "utterance_id": {"type": "integer", "description": "记录编号"},
                    "before": {"type": "integer", "description": "向前取几条，默认 5"},
                    "after": {"type": "integer", "description": "向后取几条，默认 5"},
                },
                "required": ["utterance_id"],
            },
        },
    },
]

TOOL_NAMES = [t["function"]["name"] for t in TOOLS]


# --- helpers ---------------------------------------------------------------
def _cap(s: str, limit: int = MAX_CHARS) -> str:
    if len(s) <= limit:
        return s
    return s[: max(0, limit - len(_TRUNC))] + _TRUNC


_DAY_RE = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$")


def norm_day(v: Any) -> str | None:
    """'2026-10-31' | '2026/10/3' | today | yesterday | 今天 | 昨天 | 前天 -> 'YYYY-MM-DD'; None if invalid/empty."""
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    low = s.lower()
    rel = {"today": 0, "今天": 0, "yesterday": 1, "昨天": 1, "前天": 2}
    if low in rel:
        return (date.today() - timedelta(days=rel[low])).isoformat()
    m = _DAY_RE.match(s)
    if not m:
        return None
    try:
        return date(int(m[1]), int(m[2]), int(m[3])).isoformat()
    except ValueError:
        return None


def _day_arg(args: dict, key: str, required: bool = False) -> str | None:
    raw = args.get(key)
    if raw in (None, ""):
        if required:
            raise ValueError(f"缺少参数 {key}")
        return None
    d = norm_day(raw)
    if d is None:
        raise ValueError(f"参数 {key} 不是有效日期: {raw!r}，请用 YYYY-MM-DD")
    return d


def _int_arg(args: dict, key: str, default: int, lo: int = 0, hi: int = 1000) -> int:
    v = args.get(key)
    if v in (None, ""):
        return default
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        raise ValueError(f"参数 {key} 应为整数: {v!r}")


def _hour_of(abs_start: str) -> float:
    try:
        h, m, s = abs_start[11:13], abs_start[14:16], abs_start[17:19]
        return int(h) + int(m or 0) / 60 + int(s or 0) / 3600
    except ValueError:
        return 0.0


# --- tool implementations ----------------------------------------------------
def search_transcripts(args: dict, db: DB, cfg, retriever: Any = None) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ValueError("缺少参数 query")
    day_from, day_to = _day_arg(args, "day_from"), _day_arg(args, "day_to")
    speaker = (args.get("speaker") or None) or None
    limit = _int_arg(args, "limit", int(cfg.memory.top_k), 1, 100)
    if retriever is None:
        from .retrieval import Retriever

        retriever = Retriever(cfg, db, mode="fts")
    res = retriever.search(query, day_from, day_to, speaker, limit)
    hits, how = res.hits, res.how_label
    if res.expansions:
        how += "，已扩展：" + "、".join(res.expansions)
    if not hits:
        scope = ""
        if day_from or day_to:
            scope += f" 日期 {day_from or '…'}~{day_to or '…'}"
        if speaker:
            scope += f" 说话人 {speaker}"
        return f"没有找到与「{query}」相关的记录{('（' + scope.strip() + '）') if scope else ''}。可换关键词、放宽日期或用 list_days 查看有记录的日期。"
    lines = [f"找到 {len(hits)} 条（{how}，按相关度）："]
    lines += [index.fmt_row(h) for h in hits]
    return _cap("\n".join(lines))


def get_transcript(args: dict, db: DB, cfg) -> str:
    day = _day_arg(args, "day", required=True)
    try:
        h0 = float(args.get("hour_from") if args.get("hour_from") not in (None, "") else 0)
        h1 = float(args.get("hour_to") if args.get("hour_to") not in (None, "") else 24)
    except (TypeError, ValueError):
        raise ValueError("hour_from/hour_to 应为数字")
    h0, h1 = max(0.0, h0), min(24.0, h1)
    if h1 <= h0:
        raise ValueError("hour_to 必须大于 hour_from")
    rows = [r for r in db.utterances_for_day(day) if h0 <= _hour_of(r["abs_start"]) < h1]
    if not rows:
        total = db.one("SELECT COUNT(*) AS n FROM utterances WHERE day=?", (day,))["n"]
        if total == 0:
            return f"{day} 没有任何记录。可用 list_days 查看有记录的日期。"
        return f"{day} {h0:g}:00–{h1:g}:00 没有记录（当天共 {total} 条，请调整小时范围）。"
    header = f"{day} {h0:g}:00–{h1:g}:00 共 {len(rows)} 条："
    body = "\n".join(index.fmt_row(r, with_date=False) for r in rows)
    return _cap(header + "\n" + body)


def get_daily_summary(args: dict, db: DB, cfg) -> str:
    day = _day_arg(args, "day", required=True)
    row = db.one("SELECT * FROM daily_summaries WHERE day=?", (day,))
    if row is None:
        n = db.one("SELECT COUNT(*) AS n FROM utterances WHERE day=?", (day,))["n"]
        hint = f"当天有 {n} 条记录，可用 get_transcript 或 search_transcripts 直接查看。" if n else "当天也没有录音记录。"
        return f"{day} 还没有生成日报。{hint}"
    head = f"{day} 日报（模型 {row['model'] or '?'}，基于 {row['utterance_count'] or 0} 条记录）：\n"
    return _cap(head + (row["summary_md"] or "").strip())


def list_days(args: dict, db: DB, cfg) -> str:
    rows = db.query(
        "SELECT u.day, COUNT(*) AS n, MIN(u.abs_start) AS first, MAX(u.abs_start) AS last,"
        " COUNT(DISTINCT u.speaker_name) AS spk,"
        " EXISTS(SELECT 1 FROM daily_summaries d WHERE d.day=u.day) AS has_summary"
        " FROM utterances u GROUP BY u.day ORDER BY u.day DESC LIMIT 60"
    )
    if not rows:
        return "目前还没有任何转写记录。"
    lines = [f"有记录的日期（最新在前，共 {len(rows)} 天）："]
    for r in rows:
        lines.append(
            f"{r['day']}: {r['n']} 条, {index.fmt_time(r['first'], False)[:5]}–{index.fmt_time(r['last'], False)[:5]},"
            f" {r['spk']} 位说话人{'，有日报' if r['has_summary'] else ''}"
        )
    return _cap("\n".join(lines))


def list_speakers(args: dict, db: DB, cfg) -> str:
    unknown = cfg.speakers.unknown_label
    rows = db.query(
        "SELECT s.id, s.name, s.notes, s.is_self, s.relation, s.relation_source,"
        " (SELECT COUNT(*) FROM utterances u WHERE u.speaker_id = s.id) AS n,"
        " (SELECT COUNT(*) FROM speaker_embeddings e WHERE e.speaker_id = s.id) AS samples"
        " FROM speakers s ORDER BY s.is_self DESC, n DESC, s.name"
    )
    unk = db.one("SELECT COUNT(*) AS n FROM utterances WHERE speaker_id IS NULL")["n"]
    if not rows:
        return f"还没有登记任何说话人（未识别的记录 {unk} 条，标为「{unknown}」）。可用 `audio-note speakers enroll <名字> <wav>` 登记。"
    lines = [f"已登记 {len(rows)} 位说话人："]
    has_self = False
    for r in rows:
        tags: list[str] = []
        if r["is_self"]:
            has_self = True
            name = f"{r['name']}（本人）"
        else:
            name = str(r["name"])
            if r["relation"]:
                tags.append(f"关系：{r['relation']}" + ("（自动推断）" if r["relation_source"] == "auto" else ""))
        if r["notes"]:
            tags.append(f"备注：{r['notes']}")
        extra = ("，" + "，".join(tags)) if tags else ""
        lines.append(f"- {name}: {r['n']} 条记录，{r['samples']} 个登记样本{extra}")
    if not has_self:
        lines.append("- 用户本人：未登记（记录里没有标出哪位是本人）")
    lines.append(f"- {unknown}（未识别）: {unk} 条记录")
    return _cap("\n".join(lines))


def search_notes(args: dict, db: DB, cfg, retriever: Any = None) -> str:
    """Memory items (the successor of the 0.3 notes; the tool keeps its name for the model's sake). Open items
    and finished todos are shown, deleted ones never. With a retriever whose embedding model is already
    loaded the search is semantic as well as by keyword."""
    from . import items as mem_items

    query = str(args.get("query") or "").strip()
    kind = (args.get("kind") or "").strip() or None
    if kind and kind not in mem_items.KINDS:
        raise ValueError(f"kind 必须是 {', '.join(mem_items.KINDS)} 之一")
    loaded = getattr(retriever, "loaded_embedder", None)
    embedder = loaded() if callable(loaded) else None
    rows = mem_items.search(db, cfg, query, kinds=[kind] if kind else None, status=("active", "done"), limit=30,
                            embedder=embedder)
    if not rows:
        what = f"「{query}」" if query else ""
        return f"没有找到{what}{('类型 ' + kind + ' 的') if kind else ''}笔记。笔记由每日分析生成，可改用 search_transcripts 直接检索记录。"
    lines = [f"找到 {len(rows)} 条笔记："]
    for r in rows:
        done = "（已完成）" if r["status"] == "done" else ""
        lines.append(mem_items.format_line(r, content_chars=400) + done)
    return _cap("\n".join(lines))


def get_context(args: dict, db: DB, cfg) -> str:
    uid = _int_arg(args, "utterance_id", -1, lo=-1, hi=2**62)
    if uid < 0:
        raise ValueError("缺少参数 utterance_id")
    before = _int_arg(args, "before", 5, 0, 50)
    after = _int_arg(args, "after", 5, 0, 50)
    rows = index.window(db, uid, before, after)
    if not rows:
        return f"ERROR: 没有编号为 {uid} 的记录"
    lines = [f"记录 #{uid} 的上下文（>> 为该条）："]
    for r in rows:
        mark = ">> " if int(r["id"]) == uid else "   "
        lines.append(mark + index.fmt_row(r))
    return _cap("\n".join(lines))


_DISPATCH: dict[str, Callable[..., str]] = {
    "search_transcripts": search_transcripts,
    "get_transcript": get_transcript,
    "get_daily_summary": get_daily_summary,
    "list_days": list_days,
    "list_speakers": list_speakers,
    "search_notes": search_notes,
    "get_context": get_context,
}


def run_tool(name: str, args: dict | str | None, db: DB, cfg, retriever: Any = None) -> str:
    """Execute a tool call. Always returns a string <= MAX_CHARS; errors become 'ERROR: ...'.
    `retriever` (memory.retrieval.Retriever) backs search_transcripts; None = plain keyword search."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return f"ERROR: 未知工具 {name!r}，可用工具: {', '.join(TOOL_NAMES)}"
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError as e:
            return f"ERROR: 参数不是合法 JSON: {e}"
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return "ERROR: 参数必须是 JSON 对象"
    try:
        out = fn(args, db, cfg, retriever) if name in ("search_transcripts", "search_notes") else fn(args, db, cfg)
    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:  # noqa: BLE001
        log.exception("tool %s failed", name)
        return f"ERROR: {type(e).__name__}: {e}"
    return _cap(out if isinstance(out, str) else str(out))
