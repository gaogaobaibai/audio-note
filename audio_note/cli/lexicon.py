"""`audio-note lexicon ...`: the vocabulary of the recordings (names, projects, terms) and how they get mis-heard.

    lexicon list [-q TEXT] [--kind K] [--status S] [--json]
    lexicon add TERM [--kind K] [--note TEXT] [--alias WRONG]... [--no-hotword]
    lexicon alias TERM ALIAS [--aka] [--scope auto|hint] [--apply]
    lexicon import FILE [--replace]            one term per line (glossary.txt format)
    lexicon export [--out FILE] [--with-speakers]
    lexicon apply [--term TERM [--alias ALIAS]] [--day D] [--yes]     fix stored sentences (dry run without --yes)

The rules are in audio_note.speech.lexicon; everything `apply` changes goes through revisions and can be undone
sentence by sentence in the web UI.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from rich.markup import escape as _e      # user text may contain [brackets]: never markup
from rich.table import Table

from ..config import Config
from ._late import Late, handler
from .helpers import CLIError, _db, _lazy_module, _parse_day, add_json_flag, emit_json, wants_json

log = logging.getLogger("audio_note.cli")
console = Late("console")

SCOPE_LABELS = {"auto": "自动替换", "hint": "仅提示"}
STATUS_LABELS = {"active": "", "candidate": "待确认", "disabled": "已停用"}


def _lex() -> Any:
    return _lazy_module("speech.lexicon")


def _find(lexicon: Any, db: Any, term: str) -> dict[str, Any]:
    """The entry spelled `term` (any status), or a CLIError naming what to do."""
    term = " ".join((term or "").split())
    for t in lexicon.list_terms(db, status="all"):
        if t["term"] == term or (term.isascii() and t["term"].lower() == term.lower()):
            return t
    raise CLIError(f"词库里没有「{_e(term)}」；先用 audio-note lexicon add 加上")


def cmd_lexicon_list(cfg: Config, args: argparse.Namespace) -> int:
    lexicon = _lex()
    db = _db(cfg)
    lexicon.sync_sources(db, cfg)
    status = args.status or None
    terms = lexicon.list_terms(db, q=args.query or None, kind=args.kind or None, status=status)
    if wants_json(args):
        emit_json({"terms": terms, "counts": lexicon.counts(db)})
        return 0
    if not terms:
        console.print("词库是空的。用 audio-note lexicon add <词> 添加，或者在网页的「待确认」里回答问题，词库会自己长出来。")
        return 0
    t = Table(title=f"词库（{len(terms)} 个词条）")
    for col in ("#", "写法", "类别", "听错成（→ 自动替换 / ? 仅提示）", "别称", "热词", "命中", "说明"):
        t.add_column(col)
    for e in terms:
        wrong = "、".join(("" if a["scope"] == "auto" else "?") + a["alias"] for a in e["aliases"] if a["kind"] == "mishear")
        aka = "、".join(a["alias"] for a in e["aliases"] if a["kind"] == "aka")
        name = e["term"] + (f"（{STATUS_LABELS[e['status']]}）" if STATUS_LABELS.get(e["status"]) else "")
        t.add_row(str(e["id"]), _e(name), lexicon.KIND_LABELS.get(e["kind"], e["kind"]), _e(wrong), _e(aka),
                  "是" if e["hotword"] else "", str(e["hits"] or ""), _e(e["note"] or ""))
    console.print(t)
    return 0


def cmd_lexicon_add(cfg: Config, args: argparse.Namespace) -> int:
    lexicon = _lex()
    db = _db(cfg)
    try:
        tid = lexicon.upsert_term(db, args.term, kind=args.kind or "term", note=args.note, source="user",
                                  hotword=False if args.no_hotword else None)
        for a in args.alias or []:
            lexicon.add_alias(db, tid, a, source="user", cfg=cfg)
    except (ValueError, LookupError) as e:
        raise CLIError(str(e)) from e
    e = lexicon.get_term(db, tid)
    console.print(f"已记下「{_e(e['term'])}」（{lexicon.KIND_LABELS.get(e['kind'], e['kind'])}）。")
    for a in e["aliases"]:
        console.print(f"  听成「{_e(a['alias'])}」时：{SCOPE_LABELS.get(a['scope'], a['scope'])}")
    return 0


def cmd_lexicon_alias(cfg: Config, args: argparse.Namespace) -> int:
    lexicon = _lex()
    db = _db(cfg)
    e = _find(lexicon, db, args.term)
    try:
        aid = lexicon.add_alias(db, e["id"], args.alias, kind="aka" if args.aka else "mishear", scope=args.scope or None,
                                source="user", cfg=cfg)
    except (ValueError, LookupError) as err:
        raise CLIError(str(err)) from err
    a = lexicon.get_alias(db, aid)
    e, a = {**e, "term": _e(e["term"])}, {**a, "alias": _e(a["alias"])}
    if a["kind"] == "aka":
        console.print(f"已记下：「{a['alias']}」是「{e['term']}」的另一个叫法（不会改写文字）。")
        return 0
    n = lexicon.count_alias_matches(db, aid)
    if a["scope"] == "auto":
        console.print(f"已记下：以后听成「{a['alias']}」会自动改成「{e['term']}」。")
    else:
        console.print(f"已记下：「{a['alias']}」可能是「{e['term']}」。它本身是常用词 / 太短 / 有歧义，所以不会自动替换，只作为提示。")
    if n and args.apply:
        res = lexicon.apply_alias_to_history(db, cfg, aid)
        console.print(f"已把以前记录里的 {res['applied']} 处改了过来（网页里可以逐句撤销）。")
    elif n:
        console.print(f"以前的记录里有 {n} 句包含「{a['alias']}」。要一起改：audio-note lexicon apply --term {e['term']} --alias {a['alias']} --yes")
    return 0


def cmd_lexicon_import(cfg: Config, args: argparse.Namespace) -> int:
    lexicon = _lex()
    glossary_file = _lazy_module("speech.glossary_file")
    path = Path(args.file)
    if not path.is_file():
        raise CLIError(f"文件不存在: {path}")
    try:
        terms = glossary_file.read_glossary(path)
    except (OSError, UnicodeDecodeError) as e:
        raise CLIError(f"读不了这个文件（需要 UTF-8 文本，每行一个词）: {e}") from e
    res = lexicon.import_terms(_db(cfg), terms, source="glossary", replace=bool(args.replace))
    console.print(f"读到 {res['total']} 个词：新增 {res['added']}，重新启用 {res['enabled']}" +
                  (f"，停用了不在文件里的 {res['disabled']} 个" if args.replace else "") + "。")
    return 0


def cmd_lexicon_export(cfg: Config, args: argparse.Namespace) -> int:
    lexicon = _lex()
    db = _db(cfg)
    lexicon.sync_sources(db, cfg)
    terms = lexicon.export_terms(db, include_speakers=bool(args.with_speakers))
    text = "\n".join(terms) + ("\n" if terms else "")
    if args.out:
        out = Path(args.out)
        try:
            out.write_text(text, encoding="utf-8", newline="\n")
        except OSError as e:
            raise CLIError(f"写不了 {out}: {e}") from e
        console.print(f"已导出 {len(terms)} 个词到 {out}")
    else:
        console.out(text, end="", highlight=False)
    return 0


def cmd_lexicon_apply(cfg: Config, args: argparse.Namespace) -> int:
    """Fix stored sentences with the lexicon: one alias (--term + --alias), one entry (--term) or every
    automatic rule. Lists what would change; --yes does it."""
    lexicon = _lex()
    db = _db(cfg)
    day = _parse_day(args.day) if args.day else None
    lexicon.sync_sources(db, cfg)
    if args.alias and not args.term:
        raise CLIError("--alias 需要和 --term 一起用")
    try:
        if args.term and args.alias:
            e = _find(lexicon, db, args.term)
            alias = next((a for a in e["aliases"] if a["alias"] == " ".join(args.alias.split())), None)
            if alias is None:
                raise CLIError(f"「{_e(e['term'])}」没有别名「{_e(args.alias)}」")
            res = lexicon.apply_alias_to_history(db, cfg, alias["id"], dry_run=not args.yes, day=day)
            for m in res["matches"][:20]:
                console.print(f"  #{m['utterance_id']} {m['day']} {m['time']}  {m['before']}  →  {m['after']}", markup=False)
            if args.yes:
                console.print(f"改了 {res['applied']} 句（网页里可以逐句撤销）。")
            else:
                console.print(f"共 {res['total']} 句会被修改。确认请加 --yes。" if res["total"] else "没有需要修改的句子。")
            return 0
        lexicon_id = _find(lexicon, db, args.term)["id"] if args.term else None
        res = lexicon.recorrect(db, cfg, day=day, lexicon_id=lexicon_id, dry_run=not args.yes)
    except (ValueError, LookupError) as err:
        raise CLIError(str(err)) from err
    if args.yes:
        console.print(f"检查了 {res['scanned']} 句，改了 {res['changed']} 句（{res['edits']} 处；网页里可以逐句撤销）。")
    elif res["changed"]:
        console.print(f"检查了 {res['scanned']} 句，其中 {res['changed']} 句（{res['edits']} 处）会被修改。确认请加 --yes。")
    else:
        console.print(f"检查了 {res['scanned']} 句，没有需要修改的。")
    return 0


def register(sub: Any) -> None:
    p = sub.add_parser("lexicon", help="词库：人名 / 项目 / 术语的规范写法，以及它们常被听错成什么")
    lsub = p.add_subparsers(dest="lexicon_cmd", metavar="<子命令>")
    lsub.required = True

    s = lsub.add_parser("list", help="列出词条")
    s.add_argument("-q", "--query", default=None, help="只看包含这段文字（或拼音开头相同）的词条")
    s.add_argument("--kind", default=None, choices=["person", "term", "project", "org", "place", "product", "other"], help="只看某一类")
    s.add_argument("--status", default=None, choices=["active", "candidate", "disabled", "all"], help="默认不显示已停用的")
    add_json_flag(s)
    s.set_defaults(func=handler(cmd_lexicon_list))

    s = lsub.add_parser("add", help="添加一个词条（规范写法）")
    s.add_argument("term", help="规范写法，例如 星河计划")
    s.add_argument("--kind", default="term", choices=["person", "term", "project", "org", "place", "product", "other"],
                   help="类别：person 人名 / term 术语 / project 项目 / org 机构 / place 地点 / product 产品 / other 其他")
    s.add_argument("--note", default=None, help="一句话说明（校对时会给模型看）")
    s.add_argument("--alias", action="append", metavar="WRONG", help="常被听错成的写法，可以写多次")
    s.add_argument("--no-hotword", action="store_true", help="不作为转写时的热词")
    s.set_defaults(func=handler(cmd_lexicon_add))

    s = lsub.add_parser("alias", help="给词条加一个听错的写法（或别称）")
    s.add_argument("term", help="词条的规范写法")
    s.add_argument("alias", help="听错的写法（加 --aka 则是别称）")
    s.add_argument("--aka", action="store_true", help="这是别称 / 昵称：不改写文字，只用于检索和认人")
    s.add_argument("--scope", default=None, choices=["auto", "hint"], help="auto 出现就替换 / hint 只作提示；不写则自动判断")
    s.add_argument("--apply", action="store_true", help="同时把以前记录里的这个写法改过来")
    s.set_defaults(func=handler(cmd_lexicon_alias))

    s = lsub.add_parser("import", help="从文本文件导入（每行一个词，# 开头是注释）")
    s.add_argument("file", help="词表文件（UTF-8）")
    s.add_argument("--replace", action="store_true", help="以这个文件为准：不在文件里的词条（说话人名字除外）停用")
    s.set_defaults(func=handler(cmd_lexicon_import))

    s = lsub.add_parser("export", help="导出规范写法（每行一个）")
    s.add_argument("--out", default=None, metavar="FILE", help="写到文件；不写则打印出来")
    s.add_argument("--with-speakers", action="store_true", help="把登记的说话人名字也导出")
    s.set_defaults(func=handler(cmd_lexicon_export))

    s = lsub.add_parser("apply", help="按词库修正以前的记录（不加 --yes 只预览）")
    s.add_argument("--term", default=None, help="只处理这个词条")
    s.add_argument("--alias", default=None, help="配合 --term：只处理这一个听错的写法（哪怕它只是提示）")
    s.add_argument("--day", default=None, help="只处理某天 YYYY-MM-DD")
    s.add_argument("--yes", "-y", action="store_true", help="确认修改")
    s.set_defaults(func=handler(cmd_lexicon_apply))
