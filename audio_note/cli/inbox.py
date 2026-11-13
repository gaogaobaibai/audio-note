"""`audio-note inbox ...`: the questions the system wants to ask ("待确认"), from the terminal.

    inbox list [--status S] [--kind K] [--limit N] [--json]
    inbox answer ID [--choice N | --text WORD | --correct] [--no-history]
    inbox dismiss ID [--never]

The logic is in audio_note.inbox; the web UI's 待确认 page does the same with more context (audio, sentences).
"""
from __future__ import annotations

import argparse
import logging
from typing import Any

from rich.markup import escape as _e      # user text may contain [brackets]: never markup
from rich.table import Table

from ..config import Config
from ._late import Late, handler
from .helpers import CLIError, _db, _lazy_module, add_json_flag, emit_json, wants_json

log = logging.getLogger("audio_note.cli")
console = Late("console")

KIND_LABELS = {"term": "词", "rule": "规则", "voice": "声音", "hint": "说话人", "conflict": "记忆"}
STATUS_LABELS = {"open": "待回答", "answered": "已回答", "dismissed": "已忽略", "snoozed": "稍后再问", "queued": "排队中",
                 "expired": "已失效"}


def _inbox() -> Any:
    return _lazy_module("inbox")


def cmd_inbox_list(cfg: Config, args: argparse.Namespace) -> int:
    inbox = _inbox()
    db = _db(cfg)
    items = inbox.list_items(db, status=args.status or "open", kind=args.kind or None, limit=int(args.limit), cfg=cfg)
    counts = inbox.counts(db, cfg)
    if wants_json(args):
        emit_json({"items": items, "counts": counts})
        return 0
    if not items:
        console.print("没有要确认的事。" if (args.status or "open") == "open" else "没有符合条件的问题。")
        return 0
    t = Table(title=f"待确认（{counts['open']} 个在等你回答" + (f"，{counts['queued']} 个排队中" if counts.get("queued") else "") + "）")
    for col in ("#", "类型", "问题", "选项", "出现", "状态"):
        t.add_column(col)
    for it in items:
        p = it["payload"]
        cands = p.get("candidates") or []
        options = _e("  ".join(f"[{i}] {c}" for i, c in enumerate(cands)))
        title = _e(it["title"] + (f"\n  {p['sentence']}" if p.get("sentence") else ""))
        t.add_row(str(it["id"]), KIND_LABELS.get(it["kind"], it["kind"]), title, options, str(it["occurrences"]),
                  STATUS_LABELS.get(it["status"], it["status"]))
    console.print(t)
    console.print("回答：audio-note inbox answer <#> --choice <选项序号> | --text <正确写法> | --correct（原文没错）")
    return 0


def cmd_inbox_answer(cfg: Config, args: argparse.Namespace) -> int:
    inbox = _inbox()
    db = _db(cfg)
    given = sum(1 for v in (args.choice is not None, args.text is not None, args.correct) if v)
    if given > 1:
        raise CLIError("--choice / --text / --correct 只能用一个")
    try:
        res = inbox.answer(db, cfg, int(args.id), choice=args.choice, text=args.text, as_correct=bool(args.correct),
                           apply_history=not args.no_history)
    except LookupError as e:
        raise CLIError(str(e)) from e
    except ValueError as e:
        raise CLIError(str(e)) from e
    if res["kind"] == "term":
        if res.get("as_correct"):
            console.print(f"好的，「{_e(res['term'])}」没写错，以后不再问。")
        else:
            n = len(res.get("changed") or [])
            how = {"auto": "以后听成这样会自动改", "hint": "它是常用词 / 有歧义，以后只作提示，不自动改"}.get(res.get("scope"), "")
            console.print(f"已记下：「{_e(res['alias'])}」→「{_e(res['term'])}」" + (f"（{how}）" if how else "") + f"；改了 {n} 句记录。")
    elif res["kind"] == "rule":
        console.print(f"改了 {len(res.get('changed') or [])} 句记录。" if res.get("applied") else "好的，以前的记录不动。")
    else:
        console.print("已回答。")
    return 0


def cmd_inbox_dismiss(cfg: Config, args: argparse.Namespace) -> int:
    inbox = _inbox()
    try:
        inbox.dismiss(_db(cfg), int(args.id), never=bool(args.never))
    except (LookupError, ValueError) as e:
        raise CLIError(str(e)) from e
    console.print("已忽略，以后不再问这个。" if args.never else "已忽略。")
    return 0


def register(sub: Any) -> None:
    p = sub.add_parser("inbox", help="待确认：系统拿不准、想问你的事（听不准的词、要不要批量修改……）")
    isub = p.add_subparsers(dest="inbox_cmd", metavar="<子命令>")
    isub.required = True

    s = isub.add_parser("list", help="列出问题")
    s.add_argument("--status", default="open", choices=["open", "answered", "dismissed", "snoozed", "queued", "expired", "all"],
                   help="默认只看还没回答的")
    s.add_argument("--kind", default=None, choices=["term", "rule", "voice", "hint", "conflict"], help="只看某一类")
    s.add_argument("--limit", type=int, default=50)
    add_json_flag(s)
    s.set_defaults(func=handler(cmd_inbox_list))

    s = isub.add_parser("answer", help="回答一个问题")
    s.add_argument("id", type=int, help="问题编号（inbox list 的 # 列）")
    s.add_argument("--choice", type=int, default=None, metavar="N", help="选第 N 个选项（从 0 开始）")
    s.add_argument("--text", default=None, metavar="WORD", help="自己写出正确的写法")
    s.add_argument("--correct", action="store_true", help="原文没错（以后不再问）")
    s.add_argument("--no-history", action="store_true", help="只改被问到的这一句，不动其他记录")
    s.set_defaults(func=handler(cmd_inbox_answer))

    s = isub.add_parser("dismiss", help="忽略一个问题")
    s.add_argument("id", type=int, help="问题编号")
    s.add_argument("--never", action="store_true", help="以后也别再问这个")
    s.set_defaults(func=handler(cmd_inbox_dismiss))
