"""`audio-note memory ...`: look at and tidy what the system remembers.

    memory items   [--kind K] [--status S] [-q TEXT] [--subject NAME] [--limit N]      list / search memory items
    memory todos   [--all | --done] [--due-before YYYY-MM-DD]                          todo items
    memory people  [NAME]                                                              people profiles / one person
    memory sessions [--day YYYY-MM-DD] [--sync]                                        sessions of a day
    memory forget  (--day D [--from HH:MM] [--to HH:MM] | --session ID | --utterance ID ...) [--yes]
                                                                                       forget across every layer

Every listing takes --json. `forget` only reports what it would remove (a dry run, also with --dry-run) unless
--yes is given. Kept in its own module (not cli/memory.py, which holds daily / chat / reindex) so that the
`memory` command lands after the commands whose order tests pin.
"""
from __future__ import annotations

import argparse
from typing import Any

from ..config import Config
from ._late import Late, handler
from .helpers import CLIError, _db, _lazy_module, _parse_day, add_json_flag, emit_json, wants_json

console = Late("console")

_STATUS_LABELS = {"active": "有效", "done": "已完成", "archived": "已归档", "superseded": "已合并", "rejected": "已删除"}


def _status(value: str | None) -> Any:
    if not value or value == "active":
        return ("active",)
    return "all" if value == "all" else tuple(s.strip() for s in value.split(",") if s.strip())


def _item_line(it: dict[str, Any]) -> str:
    extra = []
    if it.get("status") and it["status"] != "active":
        extra.append(_STATUS_LABELS.get(it["status"], it["status"]))
    if it.get("pinned"):
        extra.append("置顶")
    if it.get("due"):
        extra.append(f"截止 {it['due']}")
    if it.get("owner"):
        extra.append(f"负责人 {it['owner']}")
    tail = f"  [dim]（{'，'.join(extra)}）[/dim]" if extra else ""
    content = " ".join(str(it.get("content") or "").split())
    title = str(it.get("title") or "")
    if not content or content == title:
        body = title
    elif content.startswith(title):          # the content usually repeats the title: say it once
        body = content[:120]
    else:
        body = f"{title}：{content[:80]}"
    from rich.markup import escape

    body = escape(body)                      # model-written text, never markup
    return f"[dim]#{it['id']:<5} {it.get('day') or '':<10}[/dim] [cyan]{it['kind']:<10}[/cyan] {body}{tail}"


def cmd_memory_items(cfg: Config, args: argparse.Namespace) -> int:
    items = _lazy_module("memory.items")
    db = _db(cfg)
    kinds = [k.strip() for k in (args.kind or "").split(",") if k.strip()] or None
    for k in kinds or []:
        if k not in items.KINDS:
            raise CLIError(f"--kind 应为 {' / '.join(items.KINDS)}")
    if args.q:
        rows = items.search(db, cfg, args.q, kinds=kinds, status=_status(args.status), subject=args.subject, limit=args.limit)
    else:
        rows = items.list_items(db, kinds=kinds, status=_status(args.status), subject=args.subject, limit=args.limit)
    if wants_json(args):
        emit_json({"items": rows, "counts": items.counts(db)})
        return 0
    if not rows:
        console.print("没有符合条件的记忆。")
        return 0
    for it in rows:
        console.print(_item_line(it))
    c = items.counts(db)
    console.print(f"[dim]共显示 {len(rows)} 条；库里有效的记忆 {c['total']} 条（" +
                  "，".join(f"{k} {n}" for k, n in sorted(c["by_kind"].items())) + "）[/dim]")
    return 0


def cmd_memory_todos(cfg: Config, args: argparse.Namespace) -> int:
    items = _lazy_module("memory.items")
    db = _db(cfg)
    status = "all" if args.all else ("done" if args.done else "active")
    try:
        rows = items.todos(db, status=_status(status), due_before=args.due_before)
    except ValueError as e:
        raise CLIError(str(e)) from e
    if wants_json(args):
        emit_json({"todos": rows})
        return 0
    if not rows:
        console.print("没有待办。")
        return 0
    for it in rows:
        mark = "[green]✓[/green]" if it["status"] == "done" else "[ ]"
        due = f" [yellow]{it['due']}[/yellow]" if it.get("due") else ""
        owner = f" [dim]{it['owner']}[/dim]" if it.get("owner") else ""
        console.print(f"{mark} [dim]#{it['id']}[/dim]{due} {it['title']}{owner}")
    return 0


def cmd_memory_people(cfg: Config, args: argparse.Namespace) -> int:
    items = _lazy_module("memory.items")
    db = _db(cfg)
    if args.name:
        try:
            p = items.person(db, cfg, args.name)
        except LookupError as e:
            raise CLIError(str(e)) from e
        if wants_json(args):
            emit_json(p)
            return 0
        head = p["name"] + ("（本人）" if p.get("is_self") else "")
        if p.get("aliases"):
            head += "，也叫 " + "、".join(p["aliases"])
        if p.get("relation"):
            head += f"，{p['relation']}"
        console.print(f"[bold]{head}[/bold]")
        console.print(f"[dim]{p.get('utterances', 0)} 句话，{p.get('days', 0)} 天，最近 {p.get('last_seen') or '—'}[/dim]")
        for it in p["facts"]:
            console.print("  " + _item_line(it))
        for it in p["todos"]:
            console.print("  " + _item_line(it))
        for s in p["recent_sessions"]:
            console.print(f"  [dim]{s['day']} {s['start']}–{s['end']}[/dim] {s.get('title') or '（还没有摘要）'}")
        return 0
    rows = items.people(db, cfg)
    if wants_json(args):
        emit_json({"people": rows})
        return 0
    if not rows:
        console.print("还没有任何人物。")
        return 0
    for p in rows:
        name = p["name"] + ("（本人）" if p["is_self"] else "")
        aka = f" [dim]也叫 {'、'.join(p['aliases'])}[/dim]" if p["aliases"] else ""
        rel = f" {p['relation']}" if p.get("relation") else ""
        console.print(f"{name}{rel}{aka}  [dim]{p['utterances']} 句话，{p['facts']} 条记忆，{p['todos']} 个待办，"
                      f"最近 {p.get('last_seen') or '—'}[/dim]")
    return 0


def cmd_memory_sessions(cfg: Config, args: argparse.Namespace) -> int:
    sessions = _lazy_module("memory.sessions")
    db = _db(cfg)
    synced = sessions.sync(cfg, db) if args.sync else None
    day = _parse_day(args.day)
    rows = sessions.for_day(db, day, unknown=str(cfg.speakers.unknown_label or "未知"))
    if wants_json(args):
        emit_json({"day": day, "sessions": rows, **({"sync": synced} if synced is not None else {})})
        return 0
    if synced is not None and synced.get("changed"):
        console.print(f"[dim]已整理：{synced['assigned']} 句归入会话，新建 {synced['created']} 个，关闭 {synced['closed']} 个[/dim]")
    if not rows:
        console.print(f"{day} 没有会话（没有记录，或还没整理：加 --sync）。")
        return 0
    for s in rows:
        who = "、".join(f"{sp['name']}×{sp['n']}" for sp in s["speakers"][:4])
        state = "" if s["status"] == "closed" else " [yellow]进行中[/yellow]"
        console.print(f"[dim]#{s['id']:<4}[/dim] {s['start']}–{s['end']}  {s['utterance_count']:>3} 句 {s['chars']:>5} 字  "
                      f"{s.get('title') or '[dim]（还没有摘要）[/dim]'}{state}  [dim]{who}[/dim]")
    return 0


def cmd_memory_forget(cfg: Config, args: argparse.Namespace) -> int:
    mod = _lazy_module("memory.forget")
    if not (args.day or args.session or args.utterance or args.time_from or args.time_to):
        raise CLIError("请指定要忘记的范围：--day（可加 --from / --to）、--session 或 --utterance")
    db = _db(cfg)
    dry = bool(args.dry_run) or not args.yes
    try:
        rep = mod.forget(db, cfg, day=args.day, time_from=args.time_from, time_to=args.time_to,
                         utterance_ids=args.utterance or None, session_id=args.session, dry_run=dry)
    except ValueError as e:
        raise CLIError(str(e)) from e
    if wants_json(args):
        emit_json(rep)
        return 0
    verb = "将会" if rep["dry_run"] else "已经"
    console.print(f"{verb}忘记 [bold]{rep['utterances']}[/bold] 句话（{rep['chars']} 字，{'、'.join(rep['days']) or '—'}）：")
    console.print(f"  录音：整段删除 {rep['segments']['whole']} 段（{rep['audio_files']} 个文件），"
                  f"部分保留 {rep['segments']['partial']} 段；转写原文 {rep['transcripts']} 处；声音向量 {rep['voice_vectors']}，"
                  f"文本向量 {rep['vectors']}，修订记录 {rep['revisions']}")
    it = rep["items"]
    console.print(f"  会话：删除 {rep['sessions']['removed']}，更新 {rep['sessions']['updated']}；记忆：清除 {it['tombstoned']}，"
                  f"去掉出处 {it['trimmed']}，待重建 {it['removed']}；日报 {len(rep['digests'])} 天；待确认 {rep['inbox']}；"
                  f"聊天记录 {rep['chat_messages']}；旧备份 {len(rep['backups'])} 个")
    for n in rep["notes"]:
        console.print(f"  [yellow]注意[/yellow] {n}")
    for e in rep["errors"]:
        console.print(f"  [red]没有完成[/red] {e}")
    if rep["dry_run"]:
        console.print("[dim]这只是预览，什么都没有改。确认无误后加 --yes 真正执行（无法撤销）。[/dim]")
    return 1 if rep["errors"] and not rep["dry_run"] else 0


def register(sub: Any) -> None:
    m = sub.add_parser("memory", help="查看 / 整理记忆：条目、待办、人物、会话、遗忘")
    ms = m.add_subparsers(dest="memory_cmd", metavar="<子命令>")
    ms.required = True

    s = ms.add_parser("items", help="列出或搜索记忆条目")
    s.add_argument("--kind", help="类型，逗号分隔：fact,todo,decision,person,preference,project,event,topic,note")
    s.add_argument("--status", default="active", help="active（默认）| done | archived | superseded | rejected | all，可逗号分隔")
    s.add_argument("-q", "--query", dest="q", help="搜索标题和内容")
    s.add_argument("--subject", help="只看关于某个人 / 某件事的")
    s.add_argument("--limit", type=int, default=50)
    add_json_flag(s)
    s.set_defaults(func=handler(cmd_memory_items))

    s = ms.add_parser("todos", help="待办")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="包括已完成的")
    g.add_argument("--done", action="store_true", help="只看已完成的")
    s.add_argument("--due-before", metavar="YYYY-MM-DD", help="只看这一天（含）之前到期的")
    add_json_flag(s)
    s.set_defaults(func=handler(cmd_memory_todos))

    s = ms.add_parser("people", help="人物档案（给出名字看某个人）")
    s.add_argument("name", nargs="?", help="名字或别称")
    add_json_flag(s)
    s.set_defaults(func=handler(cmd_memory_people))

    s = ms.add_parser("sessions", help="某天的会话")
    s.add_argument("--day", default=None, help="YYYY-MM-DD，默认今天")
    s.add_argument("--sync", action="store_true", help="先把还没归入会话的句子整理一遍")
    add_json_flag(s)
    s.set_defaults(func=handler(cmd_memory_sessions))

    s = ms.add_parser("forget", help="跨所有层忘记一段内容（默认只预览，加 --yes 才执行）")
    s.add_argument("--day", metavar="YYYY-MM-DD", help="忘记这一整天；或与 --from / --to 一起限定时间段")
    s.add_argument("--from", dest="time_from", metavar="HH:MM", help="起始时间（HH:MM 需要 --day；也可写完整的 YYYY-MM-DDTHH:MM）")
    s.add_argument("--to", dest="time_to", metavar="HH:MM", help="结束时间（不含）")
    s.add_argument("--session", type=int, metavar="ID", help="忘记这个会话")
    s.add_argument("--utterance", type=int, action="append", metavar="ID", help="忘记这句话（可重复）")
    s.add_argument("--dry-run", action="store_true", help="只预览（默认就是预览）")
    s.add_argument("--yes", action="store_true", help="真正执行，无法撤销")
    add_json_flag(s)
    s.set_defaults(func=handler(cmd_memory_forget))
