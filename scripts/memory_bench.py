"""Benchmark the memory layer: retrieval modes (fts / vector / hybrid, ± LLM query expansion) and
chat strategies (agentic / prefetch / rag) on a question set with known evidence.

Two levels:
  retrieval  deterministic, no LLM except for expansion: recall@k of the evidence utterances
  chat       end-to-end answers scored by keyword groups; also llm/tool call counts and latency

Usage (against the fake dataset built by scripts/fake_days.py):
  set AUDIO_NOTE_CONFIG=...\\fake_config.yaml
  python scripts/memory_bench.py retrieval                       # all modes, k = 5/10/20
  python scripts/memory_bench.py retrieval --modes fts,hybrid --expand
  python scripts/memory_bench.py chat --strategies rag,prefetch,agentic --retrieval hybrid
  python scripts/memory_bench.py chat --only 3,7,12 --out bench.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_note.config import load_config  # noqa: E402
from audio_note.db import DB  # noqa: E402
from audio_note.memory import index  # noqa: E402
from audio_note.memory.retrieval import MODES, Retriever  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = HERE / "memory_questions.yaml"


# --- question set ------------------------------------------------------------------
def load_questions(path: Path, only: set[int] | None = None) -> tuple[list[dict], str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    qs = []
    for i, q in enumerate(data["questions"], start=1):
        if only and i not in only:
            continue
        q = dict(q)
        q["n"] = i
        q.setdefault("evidence", [])
        q.setdefault("expect", [])
        q.setdefault("tags", [])
        q.setdefault("min_evidence", len(q["evidence"]) if len(q["evidence"]) <= 2 else 2)
        qs.append(q)
    return qs, str(data.get("default_now") or "")


def resolve_evidence(db: DB, snippets: list[str]) -> list[set[int]]:
    """For each evidence snippet the set of utterance ids whose text contains it."""
    out = []
    for s in snippets:
        pat = "%" + s.replace("%", "\\%").replace("_", "\\_") + "%"
        rows = db.query("SELECT id FROM utterances WHERE text LIKE ? ESCAPE '\\'", (pat,))
        ids = {int(r["id"]) for r in rows}
        if not ids:
            print(f"  ! evidence not found in DB: {s!r}", file=sys.stderr)
        out.append(ids)
    return out


def evidence_found(hit_ids: list[int], evidence: list[set[int]], k: int) -> int:
    top = set(hit_ids[:k])
    return sum(1 for ids in evidence if ids & top)


def answer_score(answer: str, expect: list[list[str]]) -> tuple[int, int]:
    """(#groups matched, #groups)."""
    a = answer or ""
    ok = sum(1 for group in expect if any(alt in a for alt in group))
    return ok, len(expect)


# --- retrieval benchmark ---------------------------------------------------------------
def bench_retrieval(cfg, db: DB, client, qs: list[dict], modes: list[str], expand: bool, ks: list[int]) -> dict:
    results: dict = {}
    for mode in modes:
        r = Retriever(cfg, db, client if expand else None, mode=mode, expand=expand)
        if mode != "fts":
            r.ensure_indexed()
        label = mode + ("+expand" if expand else "")
        rows = []
        t0 = time.perf_counter()
        for q in qs:
            if not q["evidence"]:
                continue
            ev = resolve_evidence(db, q["evidence"])
            res = r.search(q["q"], limit=max(ks))
            ids = [h.id for h in res.hits]
            found = {k: evidence_found(ids, ev, k) for k in ks}
            need = min(q["min_evidence"], len(ev))
            rows.append({
                "n": q["n"], "q": q["q"], "tags": q["tags"], "need": need, "total": len(ev),
                "found": found, "ok": {k: found[k] >= need for k in ks},
                "how": res.how, "expansions": res.expansions, "top": ids[:5],
            })
        dt = time.perf_counter() - t0
        summary = {}
        for k in ks:
            summary[f"q_ok@{k}"] = sum(1 for x in rows if x["ok"][k])
            summary[f"ev_recall@{k}"] = round(sum(x["found"][k] for x in rows) / max(1, sum(x["total"] for x in rows)), 3)
        summary["questions"] = len(rows)
        summary["seconds"] = round(dt, 2)
        results[label] = {"summary": summary, "rows": rows}
        print(f"\n== retrieval {label}: " + ", ".join(f"{k}={v}" for k, v in summary.items()))
        for x in rows:
            marks = "".join("✓" if x["ok"][k] else "✗" for k in ks)
            exp = (" +" + "/".join(x["expansions"])) if x["expansions"] else ""
            print(f"  {x['n']:>2} {marks} {x['found'][max(ks)]}/{x['total']} {x['q']}{exp}")
    return results


# --- chat benchmark ---------------------------------------------------------------------
def bench_chat(cfg, db: DB, client, qs: list[dict], strategies: list[str], retrieval: str, expand: bool,
               default_now: str, verbose: bool) -> dict:
    from audio_note.memory.chat import MemoryChat

    results: dict = {}
    for strat in strategies:
        retriever = Retriever(cfg, db, client, mode=retrieval, expand=expand)
        chat = MemoryChat(cfg, db, client, retriever=retriever, strategy=strat)
        label = f"{strat}/{retrieval}" + ("+expand" if expand else "")
        rows = []
        usage0 = dict(getattr(client, "usage", {}) or {})
        for q in qs:
            now = q.get("now") or default_now
            chat.now = datetime.fromisoformat(now) if now else None
            calls: list[str] = []
            chat.on_tool = lambda n, a, r, calls=calls: calls.append(f"{n}({json.dumps(a, ensure_ascii=False)[:60]})")
            t0 = time.perf_counter()
            try:
                answer = chat.ask(q["q"])
                err = None
            except Exception as e:  # noqa: BLE001
                answer, err = "", f"{type(e).__name__}: {e}"
            dt = time.perf_counter() - t0
            ok, total = answer_score(answer, q["expect"])
            row = {"n": q["n"], "q": q["q"], "tags": q["tags"], "ok": ok, "total": total, "pass": ok == total,
                   "seconds": round(dt, 1), "stats": dict(chat.last_stats), "tools": calls, "answer": answer, "error": err}
            rows.append(row)
            mark = "✓" if row["pass"] else "✗"
            print(f"  {q['n']:>2} {mark} {ok}/{total} {dt:5.1f}s llm={row['stats'].get('llm_calls')} tools={row['stats'].get('tool_calls')} {q['q']}")
            if verbose or not row["pass"]:
                print("       " + " ".join((answer or err or "").split())[:300])
        usage1 = dict(getattr(client, "usage", {}) or {})
        tokens = {k: usage1.get(k, 0) - usage0.get(k, 0) for k in ("prompt_tokens", "completion_tokens", "requests")}
        summary = {
            "pass": sum(1 for x in rows if x["pass"]), "questions": len(rows),
            "groups": f"{sum(x['ok'] for x in rows)}/{sum(x['total'] for x in rows)}",
            "seconds": round(sum(x["seconds"] for x in rows), 1),
            "llm_calls": sum(x["stats"].get("llm_calls", 0) for x in rows),
            "tool_calls": sum(x["stats"].get("tool_calls", 0) for x in rows),
            **{k: v for k, v in tokens.items() if v},
        }
        results[label] = {"summary": summary, "rows": rows}
        print(f"== chat {label}: " + ", ".join(f"{k}={v}" for k, v in summary.items()))
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("level", choices=["retrieval", "chat"])
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--only", help="comma-separated question numbers")
    ap.add_argument("--modes", default=",".join(MODES), help="retrieval modes (retrieval level)")
    ap.add_argument("--retrieval", default=None, help="retrieval mode for the chat level (default: config)")
    ap.add_argument("--strategies", default="prefetch", help="chat strategies, comma-separated")
    ap.add_argument("--expand", action="store_true", help="LLM query expansion on")
    ap.add_argument("--ks", default="5,10,20")
    ap.add_argument("--out", type=Path, help="write full results as JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    cfg = load_config()
    db = DB(cfg.db_path)
    index.register_from(cfg, db)
    only = {int(x) for x in args.only.split(",")} if args.only else None
    qs, default_now = load_questions(args.questions, only)
    n = db.one("SELECT COUNT(*) AS n FROM utterances")["n"]
    print(f"db {cfg.db_path}: {n} utterances; {len(qs)} questions")

    client = None
    if args.level == "chat" or args.expand:
        from audio_note.llm.client import MimoClient

        client = MimoClient(cfg)

    if args.level == "retrieval":
        ks = [int(k) for k in args.ks.split(",")]
        out = bench_retrieval(cfg, db, client, qs, args.modes.split(","), args.expand, ks)
    else:
        retrieval = args.retrieval or cfg.memory.retrieval
        out = bench_chat(cfg, db, client, qs, args.strategies.split(","), retrieval, args.expand, default_now, args.verbose)

    if args.out:
        args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
