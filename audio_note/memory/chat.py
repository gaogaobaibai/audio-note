"""Memory chat: iterative retrieval (function calling) over the transcript database.

MemoryChat.ask() runs an agentic loop: the model sees a system prompt with today's date and the
tools in memory/tools.py, calls tools as it likes (up to cfg.memory.max_tool_rounds rounds), each
result is appended as a role=tool message, and the final text answer is returned. repl() wraps it
in a rich console loop that keeps conversation history.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Callable

from ..config import Config
from ..db import DB
from ..llm.client import LLMError, parse_json_loose
from . import index
from .tools import TOOLS, run_tool

log = logging.getLogger(__name__)

_WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
_FINAL_NUDGE = "工具调用次数已用完。请根据目前已经获得的信息直接回答用户的问题；信息不足就如实说明。"
_TRIM_KEEP = 200  # chars kept from an old tool result when the context budget is exceeded


class MemoryChat:
    def __init__(self, cfg: Config, db: DB, client: Any):
        self.cfg = cfg
        self.db = db
        self.client = client
        self.model: str = cfg.memory.chat_model or cfg.mimo.chat_model
        self.max_rounds = max(0, int(cfg.memory.max_tool_rounds))
        self.max_context_chars = int(cfg.memory.max_context_chars)
        self.on_tool: Callable[[str, dict, str], None] | None = None  # (name, args, result) observer
        try:
            index.register_from(cfg, db)
        except Exception as e:  # noqa: BLE001
            log.warning("register speaker/glossary terms failed: %s", e)

    # --- prompt ---------------------------------------------------------------
    def system_prompt(self, now: datetime | None = None) -> str:
        now = now or datetime.now()
        unknown = self.cfg.speakers.unknown_label
        return (
            "你是用户的个人记忆助手。用户全天录音，录音已被转写成带时间戳和说话人的文字记录（utterance），"
            "存在本地数据库里。你只能通过提供的工具查找这些记录，不能凭空编造内容。\n"
            f"现在是 {now.strftime('%Y-%m-%d %H:%M')}（{_WEEKDAYS[now.weekday()]}）。\n\n"
            "工作方式：\n"
            "1. 先判断问题涉及的时间范围和人物；把“昨天 / 上周 / 周一”等相对日期换算成 YYYY-MM-DD。\n"
            "2. 用 search_transcripts 按关键词检索（可以多次调用、换同义词、加日期或说话人过滤）；"
            "需要前后文用 get_context；要看某段时间的完整内容用 get_transcript（配合 hour_from/hour_to）；"
            "想了解某一天整体情况先看 get_daily_summary；不确定哪些天有记录用 list_days；"
            "人名不确定用 list_speakers；待办、决定、人物笔记用 search_notes。\n"
            "3. 回答时引用依据：写明时间（月-日 时:分）和说话人，例如「10-31 08:19 张三说：……」。"
            f"说话人是「{unknown}」时如实说明是未识别的说话人。转写可能有错别字，引用时可按上下文理解但不要臆造。\n"
            "4. 检索不到就明确说没有找到，不要猜测；可以建议用户换个说法或时间范围。\n"
            "5. 用用户提问所用的语言回答（中文提问用中文）。简洁、直接、有条理；长答案用要点。"
        )

    # --- agentic loop -----------------------------------------------------------
    def ask(self, question: str, history: list[dict[str, Any]] | None = None) -> str:
        """Answer one question. `history` = prior user/assistant messages (not mutated)."""
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt()}]
        messages += list(history or [])
        messages.append({"role": "user", "content": question})

        for rnd in range(self.max_rounds):
            msg = self._call(messages, tools=TOOLS)
            tool_calls = list(getattr(msg, "tool_calls", None) or [])
            if not tool_calls:
                return _content(msg)
            # one id per call, shared by the assistant turn and its tool result (servers that omit
            # ids would otherwise get mismatched call_{i} / call_{round} pairs)
            ids = [getattr(tc, "id", None) or f"call_{rnd}_{i}" for i, tc in enumerate(tool_calls)]
            messages.append(_assistant_message(msg, tool_calls, ids))
            for tc, tc_id in zip(tool_calls, ids):
                name, args = _parse_call(tc)
                result = run_tool(name, args, self.db, self.cfg)
                log.info("tool round %d: %s(%s) -> %d chars", rnd + 1, name, json.dumps(args, ensure_ascii=False)[:200], len(result))
                if self.on_tool:
                    try:
                        self.on_tool(name, args, result)
                    except Exception:  # noqa: BLE001
                        log.debug("on_tool observer failed", exc_info=True)
                messages.append({"role": "tool", "tool_call_id": tc_id, "content": result})
            self._trim_tool_results(messages)

        # budget exhausted: force a final answer without tools
        messages.append({"role": "user", "content": _FINAL_NUDGE})
        return _content(self._call(messages, tools=None))

    def _call(self, messages: list[dict[str, Any]], tools: list[dict] | None) -> Any:
        resp = self.client.chat(messages, model=self.model, tools=tools, max_tokens=4096)
        return resp.choices[0].message

    def _trim_tool_results(self, messages: list[dict[str, Any]]) -> None:
        """Keep the sum of tool outputs under memory.max_context_chars by shortening the oldest ones."""
        budget = self.max_context_chars
        if budget <= 0:
            return
        total = sum(len(m.get("content") or "") for m in messages if m.get("role") == "tool")
        for m in messages:
            if total <= budget:
                break
            if m.get("role") != "tool":
                continue
            c = m.get("content") or ""
            if len(c) > _TRIM_KEEP + 50:
                m["content"] = c[:_TRIM_KEEP] + "\n…(较早的工具结果已省略以节省上下文)"
                total -= len(c) - len(m["content"])

    # --- REPL -------------------------------------------------------------------
    def repl(self, max_history: int = 20) -> None:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel

        console = Console()
        history: list[dict[str, Any]] = []
        console.print(Panel.fit(
            f"记忆问答  模型 {self.model}  最多 {self.max_rounds} 轮检索\n"
            "输入问题回车；/clear 清空对话；/quit 退出", title="audio-note chat"))

        def show_tool(name: str, args: dict, result: str) -> None:
            arg_s = json.dumps(args, ensure_ascii=False)
            first = result.splitlines()[0] if result else ""
            console.print(f"[dim]  ⟶ {name}({arg_s[:120]}) → {first[:100]}[/dim]")

        self.on_tool = show_tool
        try:
            while True:
                try:
                    q = console.input("[bold cyan]你[/] > ").strip()
                except (EOFError, KeyboardInterrupt):
                    console.print()
                    break
                if not q:
                    continue
                if q in ("/quit", "/exit", "/q"):
                    break
                if q == "/clear":
                    history.clear()
                    console.print("[dim]对话已清空[/dim]")
                    continue
                if q == "/help":
                    console.print("命令：/clear 清空对话  /quit 退出")
                    continue
                try:
                    with console.status("[dim]检索中…[/dim]"):
                        answer = self.ask(q, history)
                except LLMError as e:
                    console.print(f"[red]LLM 调用失败：{e}[/red]")
                    continue
                console.print(Panel(Markdown(answer or "(空回答)"), title="助手", border_style="green"))
                history.append({"role": "user", "content": q})
                history.append({"role": "assistant", "content": answer})
                if len(history) > max_history:
                    del history[: len(history) - max_history]
        finally:
            self.on_tool = None


# --- message helpers ------------------------------------------------------------
def _content(msg: Any) -> str:
    return (getattr(msg, "content", None) or "").strip()


def _parse_call(tc: Any) -> tuple[str, dict]:
    fn = getattr(tc, "function", None) or tc
    name = getattr(fn, "name", None) or ""
    raw = getattr(fn, "arguments", None)
    if isinstance(raw, dict):
        return name, raw
    if not raw:
        return name, {}
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        try:
            args = parse_json_loose(raw)
        except LLMError:
            log.warning("tool %s: unparsable arguments %r", name, raw[:200])
            args = {}
    return name, args if isinstance(args, dict) else {}


def _assistant_message(msg: Any, tool_calls: list[Any], ids: list[str] | None = None) -> dict[str, Any]:
    """Serialise the model's tool-calling turn so it can be sent back as context. `ids` are the
    tool_call ids to use (fallbacks already applied); they must match the tool messages that follow."""
    calls = []
    for i, tc in enumerate(tool_calls):
        fn = getattr(tc, "function", None) or tc
        raw = getattr(fn, "arguments", None)
        if not isinstance(raw, str):
            raw = json.dumps(raw or {}, ensure_ascii=False)
        calls.append({
            "id": (ids[i] if ids and i < len(ids) else None) or getattr(tc, "id", None) or f"call_0_{i}",
            "type": "function",
            "function": {"name": getattr(fn, "name", None) or "", "arguments": raw},
        })
    return {"role": "assistant", "content": getattr(msg, "content", None) or None, "tool_calls": calls}
