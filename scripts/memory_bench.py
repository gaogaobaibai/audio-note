"""Benchmark the memory layer. Levels without an LLM first (they decide the defaults), the chat level last.

  retrieval  recall@k of the evidence sentences per retrieval mode; `--grid` compares the hybrid variants
             (keyword list x sentence views x keyword weight); summaries per tag; ms per query
  prefetch   what the chat model is handed before its first turn: share of the answer keyword groups that are
             literally inside MemoryChat.prefetch(question), and the size of that context
  items      memory items on a throw-away copy: re-running a digest must not duplicate, tombstones must not
             come back, user edits must survive, merges must stick; near-duplicate pairs left for consolidation
  forget     plant canary sentences in every layer of a throw-away copy, forget them, scan every table and
             every file for leaks (target: 0), time it
  sessions   memory.sessions.sync on a throw-away copy: wall time, time when nothing changed, sessions per day
  chat       end-to-end answers scored by keyword groups; llm / tool calls, tokens, latency (needs the API key)

Datasets
  (default)        the profile of AUDIO_NOTE_CONFIG / --profile and scripts/memory_questions.yaml (written for the
                   3-day demo data of scripts/fake_days.py)
  --story DIR      the fixed benchmark story of scripts/bench/story.py as a text-only profile in DIR (built on
                   first use: 12 sessions, 157 sentences, no audio, no API) with the built-in STORY_QUESTIONS,
                   which lean on names, product codes and figures - what the demo questions barely probe

Usage
  set AUDIO_NOTE_CONFIG=...\\fake_config.yaml
  python scripts/memory_bench.py retrieval                       # all modes, k = 5/10/20
  python scripts/memory_bench.py retrieval --grid --story DIR    # decide hybrid_keywords / vector_views
  python scripts/memory_bench.py prefetch
  python scripts/memory_bench.py items --work DIR ; python scripts/memory_bench.py forget --work DIR
  python scripts/memory_bench.py chat --strategies rag,prefetch,agentic --retrieval hybrid --key-config config.yaml
  python scripts/memory_bench.py chat --only 3,7,12 --out bench.json

items / forget / sessions never touch the profile itself: they copy its database into --work (default: a
temporary directory) first. retrieval / prefetch add missing vectors to the profile they read.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_note.config import PROJECT_ROOT, Config, load_config  # noqa: E402
from audio_note.db import DB  # noqa: E402
from audio_note.memory import index  # noqa: E402
from audio_note.memory.retrieval import KEYWORD_MODES, MODES, Retriever  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = HERE / "memory_questions.yaml"
LEVELS = ("retrieval", "prefetch", "items", "forget", "sessions", "chat")

#: questions on the benchmark story (scripts/bench/story.py). Same fields as memory_questions.yaml.
STORY_NOW = "2026-11-05T10:00"
STORY_QUESTIONS: list[dict[str, Any]] = [
    {"q": "朗月T30到手价多少钱？", "tags": ["entity", "number"], "evidence": ["到手两千六百九十九"],
     "expect": [["两千六百九十九", "2699"]]},
    {"q": "鹿鸣咖啡的新店开在哪条路？", "tags": ["entity", "conflict"], "evidence": ["新店开在栖霞路", "开在南湖路"],
     "expect": [["栖霞路"], ["南湖路"]]},
    {"q": "CDN合同首期最后定的是多少？", "tags": ["entity", "update"], "evidence": ["首期十万吧", "按首期十万重新走合同"],
     "expect": [["十万", "10万"]]},
    {"q": "青禾灰度最后定在哪天，比例多少？", "tags": ["entity", "update"],
     "evidence": ["十一月九号，比例提到百分之十", "青禾灰度还是十一月九号"], "expect": [["十一月九号", "11月9"], ["百分之十", "10%"]]},
    {"q": "压测的目标QPS是多少？实际跑到多少？", "tags": ["entity", "number", "multi-hop"],
     "evidence": ["目标QPS三千五", "QPS最高到四千二"], "expect": [["三千五", "3500"], ["四千二", "4200"]]},
    {"q": "埋点SDK升到哪个版本解决了数据丢失？", "tags": ["entity", "number"], "evidence": ["埋点SDK升到二点四就好了"],
     "expect": [["二点四", "2.4"]]},
    {"q": "岚姐买的显示器是什么型号？多少钱？", "tags": ["alias", "entity"], "evidence": ["极影X7，二十七寸，三千二百块"],
     "expect": [["X7"], ["三千二百", "3200"]]},
    {"q": "小满是谁？多大了？", "tags": ["entity"], "evidence": ["小满就是你家那只橘猫", "两岁半"], "expect": [["猫"], ["两岁半"]]},
    {"q": "白鹭网关一开始还剩多少个bug？", "tags": ["entity", "number"], "evidence": ["目前还剩二十三个bug"],
     "expect": [["二十三", "23"]]},
    {"q": "P0的bug有几个，跟什么有关？", "tags": ["entity"], "evidence": ["P0的两个，都跟支付回调有关"],
     "expect": [["两个", "2个"], ["支付回调"]]},
    {"q": "青禾的DAU目标是多少？", "tags": ["entity", "number"], "evidence": ["灰度期间DAU目标两万，全量以后到十五万"],
     "expect": [["两万"], ["十五万"]]},
    {"q": "OKR要在什么时候之前交？", "tags": ["entity"], "evidence": ["陈默说周四之前交"], "expect": [["周四"]]},
    {"q": "压测以后平均延迟是多少？", "tags": ["number"], "evidence": ["平均延迟九十毫秒"], "expect": [["九十毫秒", "90"]]},
    {"q": "T30 Pro卖多少钱？", "tags": ["entity", "number"], "evidence": ["T30 Pro，售价三千四百九十九"],
     "expect": [["三千四百九十九", "3499"]]},
    {"q": "青禾什么时候全量？", "tags": ["entity"], "evidence": ["十一月十八号全量"], "expect": [["十一月十八号", "11月18"]]},
    {"q": "给阿梨妈妈订的生日蛋糕是什么口味？", "tags": ["paraphrase"], "evidence": ["生日蛋糕我订好了，栗子口味"],
     "expect": [["栗子"]]},
    {"q": "什么时候带猫去打疫苗？", "tags": ["paraphrase", "alias"], "evidence": ["周六上午你带它去", "周六我还要带它去打疫苗"],
     "min_evidence": 1, "expect": [["周六"]]},
    {"q": "CDN供应商一年的报价是多少？", "tags": ["entity", "number"], "evidence": ["供应商报价一年十二万五"],
     "expect": [["十二万五", "12.5万", "125000"]]},
    {"q": "压测环境加了几台机器，每个月多少钱？", "tags": ["multi-hop", "number"],
     "evidence": ["压测环境要加两台机器", "一个月大概三千块"], "expect": [["两台"], ["三千"]]},
    {"q": "周五谁请大家喝咖啡？", "tags": ["entity"], "evidence": ["周五我请大家喝鹿鸣咖啡"], "expect": [["陈默"]]},
    {"q": "老秦推荐了什么东西？", "tags": ["entity"], "evidence": ["朗月T30你听过没有", "他推荐了朗月T30扫地机"],
     "min_evidence": 1, "expect": [["朗月", "T30"]]},
    {"q": "前天中午吃饭的时候聊了什么？", "tags": ["temporal"], "min_evidence": 1,
     "evidence": ["极影X7，二十七寸，三千二百块", "陈默说周四之前交"], "expect": [["显示器", "极影", "X7", "OKR"]]},
    {"q": "昨天晚上谁来家里了？", "tags": ["temporal"], "evidence": ["打扰了啊，路过你们家"], "expect": [["老秦", "未知", "朋友"]]},
    {"q": "白鹭网关的文档什么时候补完？", "tags": ["entity"], "evidence": ["周五前一定补完", "让岚姐周五前补完"],
     "min_evidence": 1, "expect": [["周五"]]},
    {"q": "上次单机压测QPS到了多少？", "tags": ["number"], "evidence": ["上次单机压到一千八"], "expect": [["一千八", "1800"]]},
    {"q": "极影要出什么新款显示器？", "tags": ["entity"], "evidence": ["X7的升级款X7S"], "expect": [["X7S"]]},
    {"q": "周岚的bug修得怎么样了？", "tags": ["alias", "temporal", "multi-hop"],
     "evidence": ["昨天修了十六个，还剩七个", "七个全清了，现在是零bug"], "expect": [["七"], ["零", "清"]]},
    {"q": "我们有没有聊过换工作的事？", "tags": ["negative"], "evidence": [], "expect": [["没有", "未找到", "没找到", "无"]]},
]


# --- question set ------------------------------------------------------------------
def _prepare(items: list[dict], only: set[int] | None = None) -> list[dict]:
    qs = []
    for i, q in enumerate(items, start=1):
        if only and i not in only:
            continue
        q = dict(q)
        q["n"] = i
        q.setdefault("evidence", [])
        q.setdefault("expect", [])
        q.setdefault("tags", [])
        q.setdefault("min_evidence", len(q["evidence"]) if len(q["evidence"]) <= 2 else 2)
        qs.append(q)
    return qs


def load_questions(path: Path, only: set[int] | None = None) -> tuple[list[dict], str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _prepare(data["questions"], only), str(data.get("default_now") or "")


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


def first_rank(hit_ids: list[int], evidence: list[set[int]]) -> int | None:
    """1-based rank of the first hit that is evidence (for MRR); None when there is none."""
    wanted = set().union(*evidence) if evidence else set()
    for rank, uid in enumerate(hit_ids, start=1):
        if uid in wanted:
            return rank
    return None


def answer_score(answer: str, expect: list[list[str]]) -> tuple[int, int]:
    """(#groups matched, #groups)."""
    a = answer or ""
    ok = sum(1 for group in expect if any(alt in a for alt in group))
    return ok, len(expect)


def by_tag(rows: list[dict], good: Any) -> dict[str, str]:
    """{"tag": "passed/total"} over the rows' tags (a row counts under each of its tags)."""
    tot: dict[str, list[int]] = {}
    for x in rows:
        for t in x.get("tags") or ["untagged"]:
            a = tot.setdefault(t, [0, 0])
            a[0] += 1 if good(x) else 0
            a[1] += 1
    return {t: f"{a}/{b}" for t, (a, b) in sorted(tot.items())}


# --- the story dataset -----------------------------------------------------------------
def build_story(root: Path, lexicon: bool = True) -> Path:
    """Write the benchmark story as a text-only profile under `root` (config.yaml, data/, glossary.txt) and
    return the config path. One segment per run of six lines (a recording is cut at every pause), fixed sentence
    timing, enrolled speakers by name, the unenrolled friend and the narrator unidentified. With `lexicon` the
    gold terms are in the lexicon (as after a few answered questions), including one nickname alias."""
    sys.path.insert(0, str(HERE / "bench"))
    import story  # noqa: E402

    root = Path(root)
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    (root / "glossary.txt").write_text("", encoding="utf-8")
    cfg_path = root / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "paths": {"data_dir": str(data), "models_dir": str(PROJECT_ROOT / "models")},
        "correction": {"glossary_file": str(root / "glossary.txt")}}, allow_unicode=True), encoding="utf-8")
    cfg = load_config(cfg_path)
    cfg.ensure_dirs()
    db = DB(cfg.db_path)
    if db.one("SELECT COUNT(*) AS n FROM utterances")["n"]:
        db.close_all()
        return cfg_path
    speaker_ids = {}
    for name, c in story.CAST.items():
        if c.get("enrolled"):
            speaker_ids[name] = db.upsert_speaker(name)
            if c.get("self"):
                db.set_self_speaker(speaker_ids[name])
    if lexicon:
        for t in story.LEXICON:
            db.execute("INSERT OR IGNORE INTO lexicon(term, kind, note, source) VALUES (?,?,?, 'glossary')",
                       (t["term"], t["kind"], t.get("note")))
        zl = db.one("SELECT id FROM lexicon WHERE term='周岚'")
        if zl is not None:       # what answering "岚姐是谁" once would leave behind
            db.execute("DELETE FROM lexicon WHERE term='岚姐'")
            db.execute("INSERT OR IGNORE INTO lexicon_aliases(lexicon_id, alias, kind, source) VALUES (?, '岚姐', 'aka', 'answer')",
                       (int(zl["id"]),))
            db.execute("UPDATE lexicon SET speaker_id=? WHERE id=?", (speaker_ids.get("周岚"), int(zl["id"])))
    index.register_from(cfg, db)
    for s in story.sessions():
        t = datetime.fromisoformat(s["start"])
        seg, seg_start, offset = None, t, 0.0
        for i, ln in enumerate(s["lines"]):
            if i % 6 == 0:
                seg_start, offset = t, 0.0
                seg = db.add_segment(cfg.audio_dir / f"{s['id']}_{i // 6}.opus", seg_start, 60.0)
                db.set_segment_status(seg, "done")
            dur = max(1.5, len(ln["text"]) / 5.0)
            name = ln["speaker"] if ln["speaker"] in speaker_ids else None
            db.add_utterances([{
                "segment_id": seg, "start_s": round(offset, 2), "end_s": round(offset + dur, 2),
                "abs_start": t.isoformat(timespec="milliseconds"), "day": t.strftime("%Y-%m-%d"), "cluster_label": 0,
                "speaker_id": speaker_ids.get(ln["speaker"]), "speaker_name": name, "text_raw": ln["text"], "text": ln["text"],
                "text_seg": index.tokenize(ln["text"])}])
            offset += dur + 0.8
            t += timedelta(seconds=dur + 0.8)
    from audio_note.memory import sessions

    sessions.sync(cfg, db, now=datetime.fromisoformat(STORY_NOW))
    db.close_all()
    return cfg_path


def story_gold_items() -> list[dict[str, Any]]:
    sys.path.insert(0, str(HERE / "bench"))
    import story  # noqa: E402

    return story.memory_items()


# --- retrieval benchmark ---------------------------------------------------------------
def run_retrieval(r: Retriever, db: DB, qs: list[dict], ks: list[int], default_now: str = "", temporal: bool = False) -> dict:
    """Evidence recall of one configured Retriever. `temporal`: go through search_question (the time a question
    names narrows the search, as the chat prefetch does) instead of the plain search."""
    rows = []
    t0 = time.perf_counter()
    for q in qs:
        if not q["evidence"]:
            continue
        ev = resolve_evidence(db, q["evidence"])
        if temporal:
            now = q.get("now") or default_now
            res = r.search_question(q["q"], now=datetime.fromisoformat(now) if now else None, limit=max(ks))
        else:
            res = r.search(q["q"], limit=max(ks))
        ids = [h.id for h in res.hits]
        found = {k: evidence_found(ids, ev, k) for k in ks}
        need = min(q["min_evidence"], len(ev))
        rows.append({
            "n": q["n"], "q": q["q"], "tags": q["tags"], "need": need, "total": len(ev),
            "found": found, "ok": {k: found[k] >= need for k in ks}, "rank": first_rank(ids, ev),
            "how": res.how, "expansions": res.expansions, "top": ids[:5],
        })
    dt = time.perf_counter() - t0
    summary: dict[str, Any] = {}
    for k in ks:
        summary[f"q_ok@{k}"] = sum(1 for x in rows if x["ok"][k])
        summary[f"ev_recall@{k}"] = round(sum(x["found"][k] for x in rows) / max(1, sum(x["total"] for x in rows)), 3)
    summary["mrr"] = round(sum(1.0 / x["rank"] for x in rows if x["rank"]) / max(1, len(rows)), 3)
    summary["questions"] = len(rows)
    summary["seconds"] = round(dt, 2)
    summary["ms_per_query"] = round(1000.0 * dt / max(1, len(rows)), 1)
    summary["by_tag"] = by_tag(rows, lambda x: x["ok"][ks[min(1, len(ks) - 1)]])
    return {"summary": summary, "rows": rows}


def run_lookup(r: Retriever, db: DB, n: int = 40) -> dict:
    """Exact look-ups: `n` rare words of the archive (found in one or two sentences; chosen by a fixed rule, no
    randomness), each used as the whole query. Reports how often a sentence containing the word is the first
    hit / among the first three / ten. This is what a search box is used for; questions do not probe it."""
    counts: dict[str, list[int]] = {}
    for row in db.query("SELECT utterance_id, text_seg FROM utterance_tokens ORDER BY utterance_id"):
        for tok in set(str(row["text_seg"] or "").split()):
            if len(tok) >= 2 and tok.lower() not in index.STOPWORDS and tok.lower() not in index.QUESTION_WORDS:
                counts.setdefault(tok, []).append(int(row["utterance_id"]))
    rare = sorted((t for t, ids in counts.items() if len(ids) <= 2), key=lambda t: (-len(t), t))
    step = max(1, len(rare) // max(1, n))
    words = rare[::step][:n]
    at = {1: 0, 3: 0, 10: 0}
    for w in words:
        ids = [h.id for h in r.search(w, limit=10).hits]
        for k in at:
            at[k] += 1 if set(ids[:k]) & set(counts[w]) else 0
    return {"words": len(words), **{f"hit@{k}": v for k, v in at.items()}}


def bench_retrieval(cfg, db: DB, client, qs: list[dict], modes: list[str], expand: bool, ks: list[int],
                    default_now: str = "", temporal: bool = False) -> dict:
    results: dict = {}
    for mode in modes:
        r = Retriever(cfg, db, client if expand else None, mode=mode, expand=expand)
        if mode != "fts":
            r.ensure_indexed(limit=None)       # everything, in the foreground: the numbers must not depend on a backlog
        label = mode + ("+expand" if expand else "") + ("+time" if temporal else "")
        results[label] = run_retrieval(r, db, qs, ks, default_now, temporal)
        results[label]["summary"]["lookup"] = run_lookup(r, db)
        summary, rows = results[label]["summary"], results[label]["rows"]
        print(f"\n== retrieval {label}: " + ", ".join(f"{k}={v}" for k, v in summary.items()))
        for x in rows:
            marks = "".join("✓" if x["ok"][k] else "✗" for k in ks)
            exp = (" +" + "/".join(x["expansions"])) if x["expansions"] else ""
            print(f"  {x['n']:>2} {marks} {x['found'][max(ks)]}/{x['total']} {x['q']}{exp}")
    return results


def _variant_cfg(cfg, **memory: Any):
    data = copy.deepcopy(dict(cfg))
    data["memory"] = {**data["memory"], **memory}
    return Config(data)


def bench_grid(cfg, db: DB, qs: list[dict], ks: list[int], default_now: str = "", temporal: bool = False) -> dict:
    """Hybrid variants side by side: keyword list (off / strict / loose / content, content at several weights)
    x sentence views. Every view is indexed once up front; the embedding model is shared."""
    from audio_note.memory import vectors

    base = Retriever(_variant_cfg(cfg, vector_views=list(vectors.VIEWS)), db, None, mode="hybrid")
    emb = base.embedder()
    if emb is None:
        raise SystemExit("the embedding model is not available: the grid needs it")
    t0 = time.perf_counter()
    base.ensure_indexed(limit=None)
    print(f"indexed all views in {time.perf_counter() - t0:.1f}s; vector storage: {vectors.storage(db)}")
    view_sets = [["plain"], ["plain", "window"], ["plain", "speaker"], ["plain", "speaker", "window"]]
    kw_sets: list[tuple[str, float]] = [("off", 1.0), ("strict", 1.0), ("loose", 1.0), ("content", 0.5), ("content", 1.0),
                                        ("content", 1.5)]
    results: dict = {}
    head = f"{'keywords':<14}{'views':<24}" + "".join(f"{'ok@' + str(k):>7}" for k in ks) + \
        "".join(f"{'rec@' + str(k):>8}" for k in ks) + f"{'mrr':>7}{'ms/q':>7}   lookup hit@1/3/10"
    print("\n" + head)
    for kw, w in kw_sets:
        for views in view_sets:
            r = Retriever(_variant_cfg(cfg, hybrid_keywords=kw, fts_weight=w, vector_views=views), db, None, embedder=emb,
                          mode="hybrid")
            res = run_retrieval(r, db, qs, ks, default_now, temporal)
            label = f"{kw}@{w:g}|{'+'.join(views)}"
            results[label] = res
            s = res["summary"]
            s["lookup"] = lk = run_lookup(r, db)
            print(f"{kw + '@' + format(w, 'g'):<14}{'+'.join(views):<24}" + "".join(f"{s[f'q_ok@{k}']:>7}" for k in ks) +
                  "".join(f"{s[f'ev_recall@{k}']:>8.3f}" for k in ks) + f"{s['mrr']:>7.3f}{s['ms_per_query']:>7.1f}"
                  f"   {lk['hit@1']}/{lk['hit@3']}/{lk['hit@10']} of {lk['words']}")
    results["storage"] = vectors.storage(db)
    return results


# --- prefetch coverage -------------------------------------------------------------------
def bench_prefetch(cfg, db: DB, qs: list[dict], default_now: str, k: int | None = None, variants: dict | None = None) -> dict:
    """Share of the answer keyword groups that are literally in the context the chat gets before its first turn
    (MemoryChat.prefetch: sentence hits + related memory items), per variant of the memory settings."""
    from audio_note.memory.chat import MemoryChat

    variants = variants or {"configured": {}}
    results: dict = {}
    shared = None
    for name, overrides in variants.items():
        vcfg = _variant_cfg(cfg, **overrides)
        retriever = Retriever(vcfg, db, None, embedder=shared)
        shared = shared or retriever.embedder()
        retriever.ensure_indexed(limit=None)
        chat = MemoryChat(vcfg, db, None, retriever=retriever, strategy="prefetch")
        rows = []
        t0 = time.perf_counter()
        for q in qs:
            if "negative" in q["tags"] or not q["expect"]:
                continue
            now = q.get("now") or default_now
            chat.now = datetime.fromisoformat(now) if now else None
            ctx = chat.prefetch(q["q"], k)
            ok, total = answer_score(ctx, q["expect"])
            rows.append({"n": q["n"], "q": q["q"], "tags": q["tags"], "ok": ok, "total": total, "full": ok == total,
                         "chars": len(ctx)})
        dt = time.perf_counter() - t0
        summary = {
            "groups": f"{sum(x['ok'] for x in rows)}/{sum(x['total'] for x in rows)}",
            "questions_fully_covered": f"{sum(1 for x in rows if x['full'])}/{len(rows)}",
            "avg_chars": round(sum(x["chars"] for x in rows) / max(1, len(rows))),
            "ms_per_question": round(1000.0 * dt / max(1, len(rows)), 1),
            "misses": [x["n"] for x in rows if not x["full"]],
            "by_tag": by_tag(rows, lambda x: x["full"]),
        }
        results[name] = {"summary": summary, "rows": rows}
        print(f"== prefetch {name}: " + ", ".join(f"{a}={b}" for a, b in summary.items()))
    return results


# --- throw-away copies ---------------------------------------------------------------------
def scratch_copy(cfg, work: Path | None, name: str):
    """(config, DB) of a copy of the profile's database under `work`/<name> (a temporary directory when `work`
    is None). The copy is taken with SQLite's backup API, so a database in use is copied consistently."""
    import sqlite3

    root = Path(work) if work else Path(tempfile.mkdtemp(prefix="memory_bench_"))
    target = root / name
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    data = copy.deepcopy(dict(cfg))
    data["paths"] = {**data["paths"], "data_dir": str(target)}
    vcfg = Config(data)
    vcfg.ensure_dirs()
    src = sqlite3.connect(f"file:{Path(cfg.db_path).as_posix()}?mode=ro", uri=True)
    dst = sqlite3.connect(vcfg.db_path)
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()
    return vcfg, DB(vcfg.db_path)


# --- memory items ----------------------------------------------------------------------------
def bench_items(cfg, work: Path | None, gold: list[dict] | None = None, replays: int = 3) -> dict:
    """De-duplication of memory items. `gold`: items to play (default: the items the copy already holds, read
    back as if a digest produced them again). Reports duplicates after `replays` re-runs with cosmetic title
    changes, tombstones that came back, user edits that were overwritten, merges that were undone, and the
    near-duplicate pairs (same kind, cosine >= 0.80 / 0.85 / 0.90) that only consolidation can catch."""
    from audio_note.memory import items, vectors
    from audio_note.memory.embed import get_embedder

    vcfg, db = scratch_copy(cfg, work, "items")
    out: dict[str, Any] = {}
    try:
        index.register_from(vcfg, db)
        existing = db.one("SELECT COUNT(*) AS n FROM memory_items")["n"]
        fp_dups = db.one("SELECT COUNT(*) AS n FROM (SELECT fingerprint FROM memory_items WHERE status<>'rejected'"
                         " GROUP BY fingerprint HAVING COUNT(*) > 1)")["n"]
        out["existing_items"] = int(existing)
        out["existing_fingerprint_duplicate_groups"] = int(fp_dups)
        if gold is None:
            gold = [{"kind": r["kind"], "title": r["title"], "content": r["content"], "day": r["day"]}
                    for r in db.query("SELECT kind, title, content, day FROM memory_items WHERE status='active' ORDER BY id")]
        t0 = time.perf_counter()
        ids = [items.upsert(db, vcfg, {**g, "content": g.get("content") or g["title"]}, source="digest")[0] for g in gold]
        out["played"] = len(gold)
        out["upsert_ms_per_item"] = round(1000.0 * (time.perf_counter() - t0) / max(1, len(gold)), 2)
        base = db.one("SELECT COUNT(*) AS n FROM memory_items")["n"]
        distinct = len(set(ids))
        tomb, edited, merged = ids[0:5], ids[5:10], list(zip(ids[10:14], ids[14:18]))
        for i in tomb:
            items.reject(db, i)
        for i in edited:
            items.update(db, i, content="用户改过的内容")
        merged = [(a, b) for a, b in merged if a != b and items.get(db, a)["status"] == "active" and items.get(db, b)["status"] == "active"]
        for keep, drop in merged:
            items.merge(db, keep, drop)
        cosmetic = [lambda t: t, lambda t: " " + t + "。", lambda t: t.replace("，", ", ").upper(), lambda t: "　".join(t)]
        actions: dict[str, int] = {}
        for rnd in range(replays):
            for g in gold:
                _id, action = items.upsert(db, vcfg, {**g, "title": cosmetic[(rnd + 1) % len(cosmetic)](g["title"]),
                                                      "content": (g.get("content") or g["title"]) + f"（第{rnd + 2}次）"},
                                           source="digest")
                actions[action] = actions.get(action, 0) + 1
        after = db.one("SELECT COUNT(*) AS n FROM memory_items")["n"]
        out["distinct_items"] = distinct
        out["duplicates_after_replays"] = int(after) - int(base)
        out["replay_actions"] = actions
        out["tombstones_resurrected"] = sum(1 for i in tomb if items.get(db, i)["status"] != "rejected")
        out["user_edits_overwritten"] = sum(1 for i in edited if items.get(db, i)["content"] != "用户改过的内容")
        out["merges_undone"] = sum(1 for _keep, drop in merged if items.get(db, drop)["status"] != "superseded")
        out["tested"] = {"tombstones": len(tomb), "user_edits": len(edited), "merges": len(merged), "replays": replays}
        try:
            emb = get_embedder(vcfg)
            t0 = time.perf_counter()
            vectors.index_items(db, emb)
            out["embed_ms_per_item"] = round(1000.0 * (time.perf_counter() - t0) / max(1, distinct), 2)
            import numpy as np

            rows = db.query("SELECT i.id, i.kind, v.vec FROM memory_items i JOIN memory_vectors v ON v.kind='item' AND v.ref_id=i.id"
                            " WHERE i.status='active'")
            pairs = {0.8: 0, 0.85: 0, 0.9: 0}
            by_kind: dict[str, list[Any]] = {}
            for r in rows:
                by_kind.setdefault(r["kind"], []).append(np.frombuffer(r["vec"], dtype=np.float32))
            for vecs in by_kind.values():
                if len(vecs) < 2:
                    continue
                sims = np.stack(vecs) @ np.stack(vecs).T
                iu = np.triu_indices(len(vecs), k=1)
                for thr in pairs:
                    pairs[thr] += int((sims[iu] >= thr).sum())
            out["near_duplicate_pairs"] = {f">={k}": v for k, v in pairs.items()}
            out["active_items"] = len(rows)
        except Exception as e:  # noqa: BLE001 - no embedding model: the rest still stands
            out["near_duplicate_pairs"] = f"skipped: {e}"
    finally:
        db.close_all()
    print("== items: " + ", ".join(f"{a}={b}" for a, b in out.items()))
    return out


# --- forget ------------------------------------------------------------------------------------
CANARIES = ("WQX9173", "琥珀暗语", "ZKV5528")


def _leaks(db: DB, root: Path, needles: tuple[str, ...]) -> dict[str, list[str]]:
    import sqlite3

    tables: list[str] = []
    for t in [r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type='table'")]:
        try:
            cols = [r["name"] for r in db.query(f"PRAGMA table_xinfo('{t}')")]
        except sqlite3.Error:
            continue
        for col in cols:
            for n in needles:
                try:
                    if db.one(f'SELECT 1 FROM "{t}" WHERE CAST("{col}" AS TEXT) LIKE ? LIMIT 1', (f"%{n}%",)) is not None:
                        tables.append(f"{t}.{col}")
                        break
                except sqlite3.Error:
                    continue
    db.close_all()
    files = []
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        blob = f.read_bytes()
        if any(n.encode("utf-8") in blob or n.lower().encode("utf-8") in blob or n.encode("utf-16-le") in blob for n in needles):
            files.append(str(f.relative_to(root)))
    return {"tables": sorted(set(tables)), "files": files}


def bench_forget(cfg, work: Path | None) -> dict:
    """Plant canary sentences in every layer of a copy, forget them, count what is left (tables, files).
    Also times a dry run and the real run, and runs the same search before and after."""
    import numpy as np

    from audio_note.memory import items, sessions, vectors
    from audio_note.memory.embed import get_embedder
    from audio_note.memory.forget import forget
    from audio_note.speech.markdown import write_day_markdown

    vcfg, db = scratch_copy(cfg, work, "forget")
    day = "2031-01-15"                                  # a day of its own: nothing of the profile is forgotten
    out: dict[str, Any] = {}
    start = datetime.fromisoformat(f"{day}T10:00:00")
    index.register_from(vcfg, db)
    seg_a = db.add_segment(vcfg.audio_dir / "canary_a.opus", start, 20.0)
    seg_b = db.add_segment(vcfg.audio_dir / "canary_b.opus", start + timedelta(seconds=30), 20.0)
    for p in ("canary_a.opus", "canary_b.opus"):
        (vcfg.audio_dir / p).write_bytes(b"OggS" + b"\0" * 512)
    texts_a = [f"今天说到{CANARIES[0]}这个代号", f"{CANARIES[1]}就是晚上八点见"]
    texts_b = ["这句话要留下来", f"中间这句提到{CANARIES[2]}", "这句也留下来"]

    def add(seg: int, base: datetime, texts: list[str]) -> list[int]:
        return db.add_utterances([{
            "segment_id": seg, "start_s": 4.0 * i, "end_s": 4.0 * i + 3, "abs_start": (base + timedelta(seconds=4 * i)).isoformat(
                timespec="milliseconds"), "day": day, "cluster_label": i % 2, "text_raw": t, "text": t,
            "text_seg": index.tokenize(t)} for i, t in enumerate(texts)])

    ua, ub = add(seg_a, start, texts_a), add(seg_b, start + timedelta(seconds=30), texts_b)
    for seg, ids, texts in ((seg_a, ua, texts_a), (seg_b, ub, texts_b)):
        db.set_segment_status(seg, "done")
        its = [{"i": i, "start_s": 4.0 * i, "end_s": 4.0 * i + 3, "text_raw": t, "text": t} for i, t in enumerate(texts)]
        db.execute("INSERT INTO transcripts(segment_id, raw_json, text_raw, text_corrected) VALUES (?,?,?,?)",
                   (seg, json.dumps(its, ensure_ascii=False), "\n".join(texts), "\n".join(texts)))
        for label in (0, 1):
            db.upsert_cluster_embedding(seg, label, np.ones(8, dtype=np.float32), 3.0)
    target = [*ua, ub[1]]
    emb = None
    try:
        emb = get_embedder(vcfg)
        vectors.index_missing(db, emb)
    except Exception as e:  # noqa: BLE001
        out["vectors"] = f"skipped: {e}"
    db.execute("INSERT INTO utterance_revisions(utterance_id, old_text, new_text, source) VALUES (?,?,?, 'user')",
               (ua[0], texts_a[0].replace(CANARIES[0], CANARIES[2]), texts_a[0]))
    sessions.sync(vcfg, db)
    items.upsert(db, vcfg, {"kind": "fact", "title": f"{CANARIES[1]}是晚上八点", "evidence": ua, "day": day}, embedder=emb)
    items.upsert(db, vcfg, {"kind": "todo", "title": f"确认{CANARIES[2]}", "day": day}, source="digest")
    db.execute("INSERT INTO daily_summaries(day, summary_md, summary_json, utterance_count) VALUES (?,?,?,?)",
               (day, f"# 日报\n{CANARIES[0]} {CANARIES[1]}", json.dumps({"overview": CANARIES[2]}), 5))
    (vcfg.daily_dir / f"{day}.md").write_text(f"{CANARIES[0]} {CANARIES[1]}\n", encoding="utf-8")
    write_day_markdown(db, vcfg, day)
    db.execute("INSERT INTO inbox_items(kind, title, payload, utterance_id) VALUES ('term', ?, ?, ?)",
               (f"{CANARIES[0]} 是什么？", json.dumps({"span": CANARIES[0], "sentence": texts_a[0]}, ensure_ascii=False), ua[0]))

    def found(q: str) -> int:
        r = Retriever(vcfg, db, None, embedder=emb)
        return sum(1 for h in r.search(q, limit=20).hits if h.id in target)

    out["found_before"] = {c: found(c) for c in CANARIES}
    n_before = db.one("SELECT COUNT(*) AS n FROM utterances")["n"]
    t0 = time.perf_counter()
    dry = forget(db, vcfg, utterance_ids=target)
    out["dry_run_ms"] = round(1000.0 * (time.perf_counter() - t0), 1)
    t0 = time.perf_counter()
    rep = forget(db, vcfg, utterance_ids=target, dry_run=False)
    out["forget_ms"] = round(1000.0 * (time.perf_counter() - t0), 1)
    out["db_bytes"] = Path(vcfg.db_path).stat().st_size
    out["report"] = {k: rep[k] for k in ("utterances", "segments", "audio_files", "transcripts", "cluster_embeddings", "vectors",
                                         "revisions", "sessions", "items", "digests", "inbox", "physical", "errors")}
    out["dry_run_matches"] = all(dry[k] == rep[k] for k in ("utterances", "segments", "audio_files", "transcripts", "items", "inbox"))
    out["found_after"] = {c: found(c) for c in CANARIES}
    out["other_utterances_untouched"] = db.one("SELECT COUNT(*) AS n FROM utterances")["n"] == n_before - len(target)
    leaks = _leaks(db, Path(vcfg.data_dir), CANARIES)          # closes the database
    out["leaks"] = leaks
    out["leak_count"] = len(leaks["tables"]) + len(leaks["files"])
    print("== forget: " + ", ".join(f"{a}={b}" for a, b in out.items()))
    return out


# --- sessions ------------------------------------------------------------------------------------
def bench_sessions(cfg, work: Path | None) -> dict:
    """memory.sessions.sync on a copy with every assignment cleared first: wall time of the full cut, of a call
    that finds nothing to do, and the shape of the result (sessions per day; no content)."""
    import statistics

    from audio_note.memory import sessions

    vcfg, db = scratch_copy(cfg, work, "sessions")
    try:
        db.execute("UPDATE utterances SET session_id=NULL")
        db.execute("DELETE FROM sessions")
        n = int(db.one("SELECT COUNT(*) AS n FROM utterances")["n"])
        last = db.one("SELECT MAX(abs_start) AS t FROM utterances")["t"]
        now = datetime.fromisoformat(str(last).replace(" ", "T")) + timedelta(days=1) if last else datetime.now()
        t0 = time.perf_counter()
        first = sessions.sync(vcfg, db, now=now)
        wall = time.perf_counter() - t0
        idle = []
        for _ in range(50):
            t0 = time.perf_counter()
            sessions.sync(vcfg, db, now=now)
            idle.append(1000.0 * (time.perf_counter() - t0))
        per_day = [int(r["n"]) for r in db.query("SELECT COUNT(*) AS n FROM sessions GROUP BY day ORDER BY day")]
        sizes = [int(r["utterance_count"]) for r in db.query("SELECT utterance_count FROM sessions")]
        out = {
            "utterances": n, "sessions": first["created"], "gap_min": sessions.gap_minutes(vcfg),
            "first_sync_ms": round(1000.0 * wall, 1), "ms_per_1000_utterances": round(1e6 * wall / max(1, n), 1),
            "idle_sync_ms_median": round(statistics.median(idle), 3), "idle_sync_ms_max": round(max(idle), 3),
            "sessions_per_day": per_day,
            "median_utterances_per_session": statistics.median(sizes) if sizes else 0,
            "single_utterance_sessions": sum(1 for s in sizes if s == 1),
            "unassigned_after": int(db.one("SELECT COUNT(*) AS n FROM utterances WHERE session_id IS NULL")["n"]),
        }
    finally:
        db.close_all()
    print("== sessions: " + ", ".join(f"{a}={b}" for a, b in out.items()))
    return out


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
        # MimoClient counts requests under "calls" (older clients said "requests"): report it as api_calls
        tokens = {k: usage1.get(k, 0) - usage0.get(k, 0) for k in ("prompt_tokens", "completion_tokens")}
        tokens["api_calls"] = (usage1.get("calls", usage1.get("requests", 0)) or 0) - (usage0.get("calls", usage0.get("requests", 0)) or 0)
        summary = {
            "pass": sum(1 for x in rows if x["pass"]), "questions": len(rows),
            "groups": f"{sum(x['ok'] for x in rows)}/{sum(x['total'] for x in rows)}",
            "seconds": round(sum(x["seconds"] for x in rows), 1),
            "llm_calls": sum(x["stats"].get("llm_calls", 0) for x in rows),
            "tool_calls": sum(x["stats"].get("tool_calls", 0) for x in rows),
            **{k: v for k, v in tokens.items() if v},
            "by_tag": by_tag(rows, lambda x: x["pass"]),
        }
        results[label] = {"summary": summary, "rows": rows}
        print(f"== chat {label}: " + ", ".join(f"{k}={v}" for k, v in summary.items()))
    return results


def _client(cfg, key_config: Path | None):
    """MimoClient for the chat level. `key_config`: a config.yaml to take ONLY mimo.api_key and the base_url
    that key belongs to from, in memory (the benchmark profile itself usually has no key)."""
    from audio_note.llm.client import MimoClient

    if key_config:
        data = yaml.safe_load(Path(key_config).read_text(encoding="utf-8")) or {}
        mimo = data.get("mimo") or {}
        key = str(mimo.get("api_key") or "").strip()
        if key:
            cfg["mimo"]["api_key"] = key
            if mimo.get("base_url"):
                cfg["mimo"]["base_url"] = str(mimo["base_url"])
    return MimoClient(cfg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("level", choices=LEVELS)
    ap.add_argument("--questions", type=Path, default=None, help="question file (default: memory_questions.yaml, or the story's own set)")
    ap.add_argument("--only", help="comma-separated question numbers")
    ap.add_argument("--modes", default=",".join(MODES), help="retrieval modes (retrieval level)")
    ap.add_argument("--grid", action="store_true", help="retrieval: compare the hybrid variants (keyword list x views x weight)")
    ap.add_argument("--temporal", action="store_true", help="retrieval: search through search_question (time-aware)")
    ap.add_argument("--retrieval", default=None, help="retrieval mode for the chat level (default: config)")
    ap.add_argument("--strategies", default="prefetch", help="chat strategies, comma-separated")
    ap.add_argument("--expand", action="store_true", help="LLM query expansion on")
    ap.add_argument("--ks", default="5,10,20")
    ap.add_argument("--k", type=int, default=None, help="prefetch: hits per question (default: memory.prefetch_k)")
    ap.add_argument("--variants", action="store_true", help="prefetch: compare keyword / view settings instead of the configured ones")
    ap.add_argument("--story", type=Path, default=None, help="use the benchmark story as a text-only profile in this directory (built on first use)")
    ap.add_argument("--no-lexicon", action="store_true", help="--story: build it with an empty lexicon")
    ap.add_argument("--gold-items", action="store_true", help="items: play the story's gold memory items (with --story)")
    ap.add_argument("--work", type=Path, default=None, help="items / forget / sessions: directory for the throw-away copy")
    ap.add_argument("--key-config", type=Path, default=None, help="chat: config.yaml to read mimo.api_key from (in memory only)")
    ap.add_argument("--out", type=Path, help="write full results as JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--profile", default=None, help="profile whose data to benchmark (default: AUDIO_NOTE_CONFIG / current profile)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    only = {int(x) for x in args.only.split(",")} if args.only else None
    if args.story:
        cfg = load_config(build_story(args.story, lexicon=not args.no_lexicon))
        qs, default_now = (load_questions(args.questions, only) if args.questions else (_prepare(STORY_QUESTIONS, only), STORY_NOW))
    else:
        cfg = load_config(profile=args.profile)
        if not Path(cfg.db_path).is_file():
            # never create an empty database behind a mistyped AUDIO_NOTE_CONFIG / --profile
            print(f"no database at {cfg.db_path}: set AUDIO_NOTE_CONFIG / --profile, or use --story DIR", file=sys.stderr)
            return 2
        qs, default_now = load_questions(args.questions or DEFAULT_QUESTIONS, only)
    ks = [int(k) for k in args.ks.split(",")]

    if args.level in ("items", "forget", "sessions"):
        if args.level == "items":
            out: Any = bench_items(cfg, args.work, story_gold_items() if (args.gold_items and args.story) else None)
        elif args.level == "forget":
            out = bench_forget(cfg, args.work)
        else:
            out = bench_sessions(cfg, args.work)
    else:
        db = DB(cfg.db_path)
        index.register_from(cfg, db)
        n = db.one("SELECT COUNT(*) AS n FROM utterances")["n"]
        print(f"db {cfg.db_path}: {n} utterances; {len(qs)} questions")
        client = _client(cfg, args.key_config) if (args.level == "chat" or args.expand) else None
        if args.level == "retrieval" and args.grid:
            out = bench_grid(cfg, db, qs, ks, default_now, args.temporal)
        elif args.level == "retrieval":
            out = bench_retrieval(cfg, db, client, qs, args.modes.split(","), args.expand, ks, default_now, args.temporal)
        elif args.level == "prefetch":
            variants = None
            if args.variants:
                variants = {
                    "0.3 behaviour (strict keywords, plain vectors)": {"hybrid_keywords": "strict", "vector_views": ["plain"]},
                    "content keywords, plain": {"hybrid_keywords": "content", "vector_views": ["plain"]},
                    "content keywords, plain+window": {"hybrid_keywords": "content", "vector_views": ["plain", "window"]},
                    "content keywords, plain+speaker+window": {"hybrid_keywords": "content", "vector_views": ["plain", "speaker", "window"]},
                    "no keywords, plain+window": {"hybrid_keywords": "off", "vector_views": ["plain", "window"]},
                }
            out = bench_prefetch(cfg, db, qs, default_now, args.k, variants)
        else:
            retrieval = args.retrieval or cfg.memory.retrieval
            out = bench_chat(cfg, db, client, qs, args.strategies.split(","), retrieval, args.expand, default_now, args.verbose)

    if args.out:
        args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
