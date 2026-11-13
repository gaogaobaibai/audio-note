"""Daily digest, push and memory commands: `daily`, `push-test`, `chat`, `reindex`."""
from __future__ import annotations

import argparse
from typing import Any

from ..config import Config
from ._late import Late, handler
from .helpers import _db, _invoke, _lazy, _lazy_module, _meter, _parse_day, _print_result

console = Late("console")
_client = Late("_client")


def cmd_daily(cfg: Config, args: argparse.Namespace) -> int:
    from rich.markdown import Markdown

    day = _parse_day(args.day)
    db = _db(cfg)
    existing = db.one("SELECT summary_md, created_at, pushed_at FROM daily_summaries WHERE day=?", (day,))
    DailyAnalyzer = _lazy("analysis.daily", "DailyAnalyzer")
    analyzer = None
    if existing and not args.force:
        console.print(f"{day} 的日报已存在（{existing['created_at']}），--force 可重新生成。")
        console.print(Markdown(existing["summary_md"]))
    else:
        client = _meter(_client(cfg), db)
        analyzer = DailyAnalyzer(cfg, db, client)
        result = _invoke(analyzer.run, day=day, force=bool(args.force))
        md = result[0] if isinstance(result, tuple) else result
        if isinstance(md, str):
            console.print(Markdown(md))
        else:
            _print_result(result)
    if args.push:
        if analyzer is None:
            analyzer = DailyAnalyzer(cfg, db, _meter(_client(cfg), db))
        analyzer.push(day)
        console.print(f"已推送 {day} 的日报。")
    return 0


def cmd_push_test(cfg: Config, args: argparse.Namespace) -> int:
    Notifier = _lazy("push.base", "Notifier")
    Message = _lazy("push.base", "Message")
    text = " ".join(args.text) if isinstance(args.text, list) else str(args.text)
    msg = Message("audio-note 测试", text)
    results = Notifier(cfg).send(msg)
    ok = True
    if isinstance(results, dict):
        for ch, err in results.items():
            if err:
                ok = False
                console.print(f"[red]{ch}[/red]: {err}")
            else:
                console.print(f"[green]{ch}[/green]: 已发送")
    else:
        console.print(str(results))
    return 0 if ok else 1


def cmd_chat(cfg: Config, args: argparse.Namespace) -> int:
    MemoryChat = _lazy("memory.chat", "MemoryChat")
    Retriever = _lazy("memory.retrieval", "Retriever")
    db = _db(cfg)
    client = _meter(_client(cfg), db)
    retriever = Retriever(cfg, db, client, mode=args.retrieval, expand=args.expand)
    chat = MemoryChat(cfg, db, client, retriever=retriever, strategy=args.strategy)
    question = " ".join(args.question).strip() if args.question else ""
    if question:
        answer = _invoke(chat.ask, question=question, history=[])
        _print_result(answer)
        return 0
    chat.repl()
    return 0


def cmd_reindex(cfg: Config, args: argparse.Namespace) -> int:
    """Rebuild the keyword index (after glossary / speaker changes) and embed utterances without a vector."""
    mem_index = _lazy_module("memory.index")
    db = _db(cfg)
    mem_index.register_from(cfg, db)
    n = mem_index.reindex(db)
    console.print(f"关键词索引已重建：{n} 条。")
    if args.no_vectors or cfg.memory.retrieval == "fts":
        console.print("向量索引跳过（memory.retrieval = fts 或 --no-vectors）。")
        return 0
    vectors = _lazy_module("memory.vectors")
    get_embedder = _lazy("memory.embed", "get_embedder")
    emb = get_embedder(cfg)
    # the configured sentence views (memory.vector_views) count as missing too; memory items are embedded along
    try:
        missing = vectors.missing_count(db, emb.name, getattr(emb, "views", None))
    except TypeError:                       # a stand-in vectors module that only knows the plain view
        missing = vectors.missing_count(db, emb.name)
    if not missing:
        vectors.index_missing(db, emb)      # nothing to do for sentences; still picks up memory items without a vector
        console.print(f"向量索引已是最新（模型 {emb.name}，{db.vector_count(emb.name)} 条）。")
        return 0
    console.print(f"正在用 {emb.name} 嵌入 {missing} 条记录…")
    k = vectors.index_missing(db, emb)
    console.print(f"向量索引完成：新增 {k} 条，共 {db.vector_count(emb.name)} 条。")
    return 0


def register(sub: Any) -> None:
    s = sub.add_parser("daily", help="生成（并推送）某天的日报")
    s.add_argument("--day", default=None, help="YYYY-MM-DD，默认今天")
    s.add_argument("--push", action="store_true")
    s.add_argument("--force", action="store_true", help="已存在也重新生成")
    s.set_defaults(func=handler(cmd_daily))

    s = sub.add_parser("push-test", help="向配置的推送渠道发测试消息")
    s.add_argument("text", nargs="+")
    s.set_defaults(func=handler(cmd_push_test))

    s = sub.add_parser("chat", help="记忆问答（无参数进入交互模式）")
    s.add_argument("question", nargs="*")
    s.add_argument("--strategy", choices=["agentic", "prefetch", "rag"], default=None, help="覆盖 memory.strategy")
    s.add_argument("--retrieval", choices=["fts", "vector", "hybrid"], default=None, help="覆盖 memory.retrieval")
    s.add_argument("--expand", dest="expand", action="store_true", default=None, help="开启 LLM 查询扩展")
    s.add_argument("--no-expand", dest="expand", action="store_false", help="关闭 LLM 查询扩展")
    s.set_defaults(func=handler(cmd_chat))

    s = sub.add_parser("reindex", help="重建关键词索引并补齐向量索引")
    s.add_argument("--no-vectors", action="store_true", help="只重建关键词索引")
    s.set_defaults(func=handler(cmd_reindex))
