"""Daily digest: utterances of one day -> sessions -> text blocks -> LLM map/reduce -> Markdown + DB rows.

    analyzer = DailyAnalyzer(cfg, db)              # client is created lazily from cfg (needs mimo.api_key)
    md, js = analyzer.run("2026-10-31")            # writes data/daily/2026-10-31.md, daily_summaries, memory_notes
    analyzer.push("2026-10-31")                    # sends the digest through push.Notifier, sets pushed_at

Pipeline
  sessions(day)   utterances split where the silence between consecutive rows exceeds `session_gap_min`
  blocks          each session rendered as "HH:MM 说话人：text" lines and cut into <= `max_block_chars` pieces
  map             one JSON call per block  (prompts.MAP_*)
  reduce          one JSON call merging the block results into the day's digest (prompts.REDUCE_*)
  render_markdown pure function: digest JSON -> Markdown (概览/主要话题/决定/待办/提到的人/情绪/未闭环/时间线)
  memory_notes    todos / decisions / people upserted by (kind, title); day_to and content refreshed on re-run

Model: cfg.mimo.analysis_model if set, else cfg.mimo.chat_model.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from ..config import Config
from ..db import DB, now_iso
from ..llm.client import LLMError, MissingAPIKey
from . import prompts

log = logging.getLogger(__name__)

DEFAULT_SESSION_GAP_MIN = 20
DEFAULT_MAX_BLOCK_CHARS = 6000
NOTE_TITLE_MAX = 80

SECTION_TITLES = ("概览", "主要话题", "决定", "待办", "提到的人", "情绪", "未闭环", "时间线")

EMPTY_DIGEST: dict[str, Any] = {
    "overview": "", "topics": [], "decisions": [], "todos": [], "people": [],
    "mood": "", "open_loops": [], "timeline": [],
}


@dataclass
class Block:
    """One chunk of transcript handed to the map prompt."""
    index: int                 # 1-based position within the day
    session: int               # 1-based session number
    start: datetime
    end: datetime
    text: str
    utterance_ids: list[int] = field(default_factory=list)

    @property
    def time_range(self) -> str:
        return f"{self.start:%H:%M}–{self.end:%H:%M}"


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _day_str(day: str | date | datetime) -> str:
    if isinstance(day, datetime):
        return day.date().isoformat()
    if isinstance(day, date):
        return day.isoformat()
    s = str(day).strip()
    return date.fromisoformat(s).isoformat()  # validates YYYY-MM-DD


def _parse_dt(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")


def _row_start(row: sqlite3.Row | dict) -> datetime:
    return _parse_dt(str(row["abs_start"]))


def _row_end(row: sqlite3.Row | dict) -> datetime:
    dur = max(0.0, float(row["end_s"] or 0.0) - float(row["start_s"] or 0.0))
    return _row_start(row) + timedelta(seconds=dur)


def _speaker(row: sqlite3.Row | dict, unknown: str) -> str:
    name = row["speaker_name"] if "speaker_name" in row.keys() else None
    return str(name) if name else unknown


def _str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v).strip()


def _str_list(v: Any) -> list[str]:
    if v is None or v == "":
        return []
    if isinstance(v, str):
        items = [v]
    elif isinstance(v, dict):
        items = [v]
    else:
        try:
            items = list(v)
        except TypeError:
            items = [v]
    out: list[str] = []
    for it in items:
        if isinstance(it, dict):
            s = _str(it.get("text") or it.get("event") or it.get("name") or it.get("title") or it)
        else:
            s = _str(it)
        if s and s not in out:
            out.append(s)
    return out


def _todo_list(v: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for it in (v or []) if isinstance(v, list) else ([v] if v else []):
        if isinstance(it, dict):
            text = _str(it.get("text") or it.get("title") or it.get("todo"))
            d = {"text": text}
            if _str(it.get("owner")):
                d["owner"] = _str(it.get("owner"))
            if _str(it.get("due")):
                d["due"] = _str(it.get("due"))
        else:
            d = {"text": _str(it)}
        if d["text"] and d["text"] not in seen:
            seen.add(d["text"])
            out.append(d)
    return out


def _people_list(v: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for it in (v or []) if isinstance(v, list) else ([v] if v else []):
        if isinstance(it, dict):
            name = _str(it.get("name") or it.get("title"))
            ctx = _str(it.get("context") or it.get("content") or it.get("summary"))
        else:
            name, ctx = _str(it), ""
        if name and name not in seen:
            seen.add(name)
            out.append({"name": name, "context": ctx})
    return out


def _timeline_list(v: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for it in (v or []) if isinstance(v, list) else ([v] if v else []):
        if isinstance(it, dict):
            t, ev = _str(it.get("time")), _str(it.get("event") or it.get("text"))
        else:
            s = _str(it)
            t, ev = (s.split(" ", 1) + [""])[:2] if " " in s else ("", s)
        if ev:
            out.append({"time": t, "event": ev})
    return out


def normalize_map_result(d: Any) -> dict[str, Any]:
    d = d if isinstance(d, dict) else {}
    return {
        "title": _str(d.get("title")),
        "summary": _str(d.get("summary")),
        "topics": _str_list(d.get("topics")),
        "decisions": _str_list(d.get("decisions")),
        "todos": _str_list(d.get("todos")),
        "people": _str_list(d.get("people")),
        "mood": _str(d.get("mood")),
        "open_questions": _str_list(d.get("open_questions") or d.get("open_loops")),
    }


def normalize_digest(d: Any) -> dict[str, Any]:
    d = d if isinstance(d, dict) else {}
    return {
        "overview": _str(d.get("overview") or d.get("summary")),
        "topics": _str_list(d.get("topics")),
        "decisions": _str_list(d.get("decisions")),
        "todos": _todo_list(d.get("todos")),
        "people": _people_list(d.get("people")),
        "mood": _str(d.get("mood")),
        "open_loops": _str_list(d.get("open_loops") or d.get("open_questions")),
        "timeline": _timeline_list(d.get("timeline")),
    }


def _fallback_digest(blocks: list[Block], results: list[dict[str, Any]]) -> dict[str, Any]:
    """Digest built without the reduce call (used when reduce fails): plain concatenation."""
    out = json.loads(json.dumps(EMPTY_DIGEST))
    out["overview"] = "\n".join(r["summary"] for r in results if r.get("summary"))
    for r, b in zip(results, blocks):
        out["topics"] += [t for t in r["topics"] if t not in out["topics"]]
        out["decisions"] += [t for t in r["decisions"] if t not in out["decisions"]]
        out["todos"] += [{"text": t} for t in r["todos"] if t not in [x["text"] for x in out["todos"]]]
        out["people"] += [{"name": p, "context": ""} for p in r["people"] if p not in [x["name"] for x in out["people"]]]
        out["open_loops"] += [t for t in r["open_questions"] if t not in out["open_loops"]]
        if r.get("title"):
            out["timeline"].append({"time": f"{b.start:%H:%M}", "event": r["title"]})
    moods = [r["mood"] for r in results if r.get("mood")]
    out["mood"] = "；".join(moods)
    return out


# ----------------------------------------------------------------------------------------------
# Markdown
# ----------------------------------------------------------------------------------------------
def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {s}" for s in items) if items else "（无）"


def render_markdown(day: str, summary_json: dict[str, Any]) -> str:
    """Pure: digest dict (reduce output, optionally with the meta keys run() adds) -> Markdown."""
    d = normalize_digest(summary_json or {})
    meta = summary_json or {}
    lines: list[str] = [f"# {day} 每日记录", ""]

    stats: list[str] = []
    n = meta.get("utterance_count")
    if isinstance(n, int):
        stats.append(f"共 {n} 条记录")
    if isinstance(meta.get("session_count"), int):
        stats.append(f"{meta['session_count']} 个时段")
    if meta.get("time_range"):
        stats.append(str(meta["time_range"]))
    if meta.get("speakers"):
        stats.append("说话人：" + "、".join(str(s) for s in meta["speakers"]))
    if stats:
        lines += ["> " + " · ".join(stats), ""]

    lines += ["## 概览", d["overview"] or "（无）", ""]
    lines += ["## 主要话题", _bullets(d["topics"]), ""]
    lines += ["## 决定", _bullets(d["decisions"]), ""]

    todos: list[str] = []
    for t in d["todos"]:
        extra = []
        if t.get("owner"):
            extra.append(f"负责人：{t['owner']}")
        if t.get("due"):
            extra.append(f"截止：{t['due']}")
        todos.append(f"[ ] {t['text']}" + (f"（{'，'.join(extra)}）" if extra else ""))
    lines += ["## 待办", _bullets(todos), ""]

    people = [f"**{p['name']}**：{p['context']}" if p.get("context") else f"**{p['name']}**" for p in d["people"]]
    lines += ["## 提到的人", _bullets(people), ""]
    lines += ["## 情绪", d["mood"] or "（无）", ""]
    lines += ["## 未闭环", _bullets(d["open_loops"]), ""]
    timeline = [f"{t['time']} {t['event']}".strip() for t in d["timeline"]]
    lines += ["## 时间线", _bullets(timeline), ""]
    return "\n".join(lines)


def empty_day_markdown(day: str) -> str:
    return f"# {day} 每日记录\n\n当天没有记录到任何语音内容。\n"


# ----------------------------------------------------------------------------------------------
# analyzer
# ----------------------------------------------------------------------------------------------
class DailyAnalyzer:
    def __init__(self, cfg: Config, db: DB, client: Any = None):
        self.cfg = cfg
        self.db = db
        self._client = client
        m = cfg.mimo
        self.model: str = str(m.get("analysis_model") or m.get("chat_model") or "")
        a = cfg.get("analysis") or {}
        self.session_gap_min = float(a.get("session_gap_min", DEFAULT_SESSION_GAP_MIN))
        self.max_block_chars = int(a.get("max_block_chars", DEFAULT_MAX_BLOCK_CHARS))
        self.unknown_label = str(cfg.speakers.get("unknown_label") or "未知")

    @property
    def client(self) -> Any:
        if self._client is None:
            from ..llm.client import MimoClient

            self._client = MimoClient(self.cfg, rpm=int(self.cfg.mimo.rpm))
        return self._client

    # --- grouping -------------------------------------------------------------------------
    def sessions(self, day: str | date) -> list[list[sqlite3.Row]]:
        """Utterances of `day` grouped into sessions; a new session starts after a gap > session_gap_min."""
        rows = self.db.utterances_for_day(_day_str(day))
        gap = timedelta(minutes=self.session_gap_min)
        out: list[list[sqlite3.Row]] = []
        cur: list[sqlite3.Row] = []
        prev_end: datetime | None = None
        for r in rows:
            start = _row_start(r)
            if cur and prev_end is not None and start - prev_end > gap:
                out.append(cur)
                cur = []
            cur.append(r)
            end = _row_end(r)
            prev_end = end if prev_end is None or end > prev_end else prev_end
        if cur:
            out.append(cur)
        return out

    def _line(self, r: sqlite3.Row) -> str:
        text = " ".join(str(r["text"] or r["text_raw"] or "").split())
        return f"{_row_start(r):%H:%M} {_speaker(r, self.unknown_label)}：{text}"

    def blocks(self, sessions: list[list[sqlite3.Row]]) -> list[Block]:
        """Render sessions as lines and cut into blocks of at most max_block_chars characters."""
        limit = max(200, self.max_block_chars)
        blocks: list[Block] = []
        for si, rows in enumerate(sessions, 1):
            lines: list[str] = []
            ids: list[int] = []
            start: datetime | None = None
            end: datetime | None = None
            size = 0

            def flush() -> None:
                nonlocal lines, ids, start, end, size
                if lines and start is not None and end is not None:
                    blocks.append(Block(len(blocks) + 1, si, start, end, "\n".join(lines), ids))
                lines, ids, start, end, size = [], [], None, None, 0

            for r in rows:
                line = self._line(r)
                pieces = [line] if len(line) <= limit else [line[i:i + limit] for i in range(0, len(line), limit)]
                for piece in pieces:
                    if lines and size + len(piece) + 1 > limit:
                        flush()
                    rs, re_ = _row_start(r), _row_end(r)
                    start = rs if start is None else start
                    end = re_ if end is None or re_ > end else end
                    lines.append(piece)
                    ids.append(int(r["id"]))
                    size += len(piece) + 1
            flush()
        return blocks

    # --- LLM stages -----------------------------------------------------------------------
    def _json(self, system: str, user: str, **kw: Any) -> Any:
        if self.model:
            kw.setdefault("model", self.model)
        return self.client.json(system, user, **kw)

    def map_block(self, day: str, block: Block, total: int) -> dict[str, Any]:
        system, user = prompts.map_prompt(day, block.index, total, f"{block.start:%H:%M}", f"{block.end:%H:%M}", block.text)
        res = normalize_map_result(self._json(system, user, max_tokens=2048))
        res["time_from"] = f"{block.start:%H:%M}"
        res["time_to"] = f"{block.end:%H:%M}"
        res["session"] = block.session
        return res

    def reduce_blocks(self, day: str, results: list[dict[str, Any]], *, utterance_count: int, session_count: int,
                      speakers: list[str]) -> dict[str, Any]:
        system, user = prompts.reduce_prompt(day, results, utterance_count=utterance_count,
                                             session_count=session_count, speakers=speakers)
        return normalize_digest(self._json(system, user, max_tokens=4096))

    # --- main entry -----------------------------------------------------------------------
    def run(self, day: str | date, force: bool = False) -> tuple[str, dict[str, Any]]:
        """Analyse one day. Returns (summary_md, summary_json). Without utterances no LLM call is made."""
        day = _day_str(day)
        sessions = self.sessions(day)
        rows = [r for s in sessions for r in s]
        if not rows:
            md = empty_day_markdown(day)
            js = dict(EMPTY_DIGEST, day=day, utterance_count=0, session_count=0, overview="当天没有记录到任何语音内容。")
            if force:
                self._write_summary(day, md, js, 0)
            log.info("daily %s: no utterances", day)
            return md, js

        existing = self.db.one("SELECT summary_md, summary_json, utterance_count FROM daily_summaries WHERE day=?", (day,))
        if existing and not force and int(existing["utterance_count"] or 0) == len(rows) and existing["summary_json"]:
            try:
                return existing["summary_md"], json.loads(existing["summary_json"])
            except (TypeError, ValueError):
                pass

        blocks = self.blocks(sessions)
        speakers = sorted({_speaker(r, self.unknown_label) for r in rows} - {self.unknown_label})
        results: list[dict[str, Any]] = []
        kept_blocks: list[Block] = []
        for b in blocks:
            try:
                results.append(self.map_block(day, b, len(blocks)))
                kept_blocks.append(b)
            except MissingAPIKey:
                raise  # configuration problem, not a transient failure of this block
            except LLMError as e:
                log.warning("daily %s: block %d/%d failed: %s", day, b.index, len(blocks), e)
        if not results:
            raise LLMError(f"daily {day}: all {len(blocks)} blocks failed")

        try:
            digest = self.reduce_blocks(day, results, utterance_count=len(rows), session_count=len(sessions), speakers=speakers)
        except LLMError as e:
            log.warning("daily %s: reduce failed (%s); using concatenated block summaries", day, e)
            digest = _fallback_digest(kept_blocks, results)

        js: dict[str, Any] = dict(digest)
        js.update({
            "day": day,
            "utterance_count": len(rows),
            "session_count": len(sessions),
            "block_count": len(blocks),
            "time_range": f"{_row_start(rows[0]):%H:%M}–{max(_row_end(r) for r in rows):%H:%M}",
            "speakers": speakers,
            "model": self.model,
            "blocks": results,
        })
        md = render_markdown(day, js)
        self._write_summary(day, md, js, len(rows))
        self._upsert_notes(day, js, rows)
        log.info("daily %s: %d utterances, %d sessions, %d blocks -> %d chars", day, len(rows), len(sessions), len(blocks), len(md))
        return md, js

    # --- persistence ----------------------------------------------------------------------
    def _write_summary(self, day: str, md: str, js: dict[str, Any], count: int) -> None:
        self.cfg.daily_dir.mkdir(parents=True, exist_ok=True)
        path = self.cfg.daily_dir / f"{day}.md"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(md)
        self.db.execute(
            "INSERT INTO daily_summaries(day, summary_md, summary_json, model, utterance_count, created_at, pushed_at)"
            " VALUES (?,?,?,?,?,?,NULL)"
            " ON CONFLICT(day) DO UPDATE SET summary_md=excluded.summary_md, summary_json=excluded.summary_json,"
            " model=excluded.model, utterance_count=excluded.utterance_count, created_at=excluded.created_at, pushed_at=NULL",
            (day, md, json.dumps(js, ensure_ascii=False), self.model or None, count, now_iso()),
        )

    def _upsert_notes(self, day: str, js: dict[str, Any], rows: list[sqlite3.Row]) -> None:
        notes: list[tuple[str, str, str, list[int]]] = []  # kind, title, content, source ids
        for t in js.get("todos", []):
            text = t.get("text", "") if isinstance(t, dict) else str(t)
            if not text:
                continue
            extra = []
            if isinstance(t, dict):
                if t.get("owner"):
                    extra.append(f"负责人：{t['owner']}")
                if t.get("due"):
                    extra.append(f"截止：{t['due']}")
            notes.append(("todo", text[:NOTE_TITLE_MAX], text + (f"（{'，'.join(extra)}）" if extra else ""), []))
        for d in js.get("decisions", []):
            d = str(d)
            if d:
                notes.append(("decision", d[:NOTE_TITLE_MAX], d, []))
        for p in js.get("people", []):
            name = p.get("name", "") if isinstance(p, dict) else str(p)
            if not name:
                continue
            ctx = p.get("context", "") if isinstance(p, dict) else ""
            ids = [int(r["id"]) for r in rows
                   if _speaker(r, self.unknown_label) == name or name in str(r["text"] or "")][:50]
            notes.append(("person", name[:NOTE_TITLE_MAX], f"{day}：{ctx}" if ctx else day, ids))

        with self.db.transaction() as c:
            for kind, title, content, ids in notes:
                row = c.execute("SELECT id, day_from, day_to FROM memory_notes WHERE kind=? AND title=?", (kind, title)).fetchone()
                src = json.dumps(ids) if ids else None
                if row:
                    day_from = min(row["day_from"] or day, day)
                    day_to = max(row["day_to"] or day, day)
                    c.execute("UPDATE memory_notes SET content=?, day_from=?, day_to=?, source_utterance_ids=?, updated_at=? WHERE id=?",
                              (content, day_from, day_to, src, now_iso(), row["id"]))
                else:
                    c.execute("INSERT INTO memory_notes(kind, title, content, day_from, day_to, source_utterance_ids) VALUES (?,?,?,?,?,?)",
                              (kind, title, content, day, day, src))
        log.debug("daily %s: upserted %d memory notes", day, len(notes))

    # --- push -----------------------------------------------------------------------------
    def push(self, day: str | date) -> dict[str, str | None]:
        """Send the digest through push.Notifier. Returns {channel: error_or_None}; sets pushed_at on any success."""
        from ..push.base import Message, Notifier  # written by another module; imported lazily

        day = _day_str(day)
        row = self.db.one("SELECT summary_md FROM daily_summaries WHERE day=?", (day,))
        md = row["summary_md"] if row else self.run(day)[0]
        results = Notifier(self.cfg).send(Message(title=f"每日记录 {day}", body_md=md))
        ok = [k for k, v in (results or {}).items() if v is None]
        if ok:
            self.db.execute("UPDATE daily_summaries SET pushed_at=? WHERE day=?", (now_iso(), day))
            log.info("daily %s pushed via %s", day, ", ".join(ok))
        else:
            log.warning("daily %s: push failed on every channel: %s", day, results)
        return dict(results or {})
