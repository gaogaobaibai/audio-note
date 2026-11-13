"""Merge bench result files into one Markdown report: one table per kind, one column per result.

    python scripts/bench/report.py baseline/cer.json baseline/speakers.json baseline/cost.json \
                                   v1/cer.json v1/speakers.json v1/cost.json --out report.md

Every input is a JSON written by bench_cer.py / bench_speakers.py (profile, replay, proxy) / bench_cost.py;
the file's "kind" decides the table it goes into and its "label" names the column. Rates are percentages,
costs are CPU-seconds per second of audio. Without --out the Markdown goes to stdout.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    from . import common
except ImportError:  # run as a script
    import common

CONDITIONS = (("near", "近讲"), ("far", "远讲"), ("overlap", "重叠"), ("system", "系统声音"))
BUCKETS = ("<1s", "1-2s", "2-4s", "4-8s", ">=8s")


def pct(x: Any, digits: int = 1) -> str:
    return "—" if not isinstance(x, (int, float)) else f"{100 * x:.{digits}f}%"


def num(x: Any, digits: int = 3) -> str:
    if x is None:
        return "—"
    return f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def get(d: Any, *path: str) -> Any:
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def table(header: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def section_cer(results: list[dict[str, Any]]) -> str:
    labels = [r["label"] for r in results]
    rows: list[list[str]] = []

    def row(name: str, fn: Any) -> None:
        rows.append([name] + [fn(r) for r in results])

    row("CER 原始（全部台词）", lambda r: pct(get(r, "overall", "raw", "cer"), 2))
    row("CER 校对后（全部台词）", lambda r: pct(get(r, "overall", "corrected", "cer"), 2))
    row("CER 原始（只算转写到的台词）", lambda r: pct(get(r, "covered_lines_only", "raw", "cer"), 2))
    row("CER 校对后（只算转写到的台词）", lambda r: pct(get(r, "covered_lines_only", "corrected", "cer"), 2))
    for key, zh in CONDITIONS:
        row(f"CER {zh}：原始 → 校对后", lambda r, k=key: f"{pct(get(r, 'by_condition', k, 'raw', 'cer'), 2)} → {pct(get(r, 'by_condition', k, 'corrected', 'cer'), 2)}")
    row("没被转写的台词（句）", lambda r: f"{num(get(r, 'overall', 'missed_lines'))} / {num(get(r, 'overall', 'lines'))}")
    row("专有名词错误率：原始 → 校对后", lambda r: f"{pct(get(r, 'terms', 'overall', 'rate_raw'))} → {pct(get(r, 'terms', 'overall', 'rate_corrected'))}")
    for key, zh in (("name", "人名 / 项目 / 产品"), ("term", "术语 / 缩写"), ("number", "数字金额")):
        row(f"　{zh}", lambda r, k=key: f"{pct(get(r, 'terms', 'by_group', k, 'rate_raw'))} → {pct(get(r, 'terms', 'by_group', k, 'rate_corrected'))}")
    row("校对改动：改对 / 改坏 / 无关", lambda r: f"{num(get(r, 'edits', 'helpful'))} / {num(get(r, 'edits', 'harmful'))} / {num(get(r, 'edits', 'neutral'))}")
    row("　其中完全改对 / 凭空改成专有名词", lambda r: f"{num(get(r, 'edits', 'exact'))} / {num(get(r, 'edits', 'wrong_term_injections'))}")
    row("校对改动：修掉的字 / 改坏的字", lambda r: f"{num(get(r, 'edits', 'chars_fixed'))} / {num(get(r, 'edits', 'chars_broken'))}")
    if any(r.get("questions") for r in results):
        row("提问数 / 问对了（确实有错）", lambda r: f"{num(get(r, 'questions', 'term_questions'))} / {num(get(r, 'questions', 'pointing_at_errors'))}")
    parts = ["## 转写（CER）", table(["指标"] + labels, rows)]
    days = sorted({d for r in results for d in (r.get("terms", {}).get("by_day") or {})})
    if days:
        drows = [[d] + [f"{pct(get(r, 'terms', 'by_day', d, 'rate_raw'))} → {pct(get(r, 'terms', 'by_day', d, 'rate_corrected'))}" for r in results]
                 for d in days]
        parts += ["专有名词错误率按天（原始 → 校对后）：", table(["日期"] + labels, drows)]
    return "\n\n".join(p for p in parts if p)


def _speaker_rows(results: list[dict[str, Any]], ev_of: Any) -> list[list[str]]:
    rows: list[list[str]] = []

    def row(name: str, fn: Any) -> None:
        rows.append([name] + [fn(ev_of(r)) for r in results])

    def triple(b: Any) -> str:
        return "—" if not b else f"{pct(b.get('accuracy'))} / {pct(b.get('frr'))} / {pct(b.get('far'))}"

    row("总体：准确率 / FRR / FAR", lambda e: triple(get(e, "overall")))
    row("总体（按时长加权）", lambda e: triple(get(e, "overall", "by_time")))
    row("　其中认成别人（混淆率）", lambda e: pct(get(e, "overall", "confusion_rate")))
    for key, zh in CONDITIONS:
        row(f"{zh}（n）", lambda e, k=key: f"{triple(get(e, 'by_condition', k))} ({num(get(e, 'by_condition', k, 'n'))})")
    for b in BUCKETS:
        row(f"时长 {b}（n）", lambda e, k=b: f"{triple(get(e, 'by_duration', k))} ({num(get(e, 'by_duration', k, 'n'))})")
    speakers = sorted({s for r in results for s in (ev_of(r).get("per_speaker") or {})})
    for s in speakers:
        row(f"{s}：FRR / FAR", lambda e, k=s: f"{pct(get(e, 'per_speaker', k, 'frr'))} / {pct(get(e, 'per_speaker', k, 'far'))}")
    unknown = sorted({s for r in results for s in (ev_of(r).get("unknown_speakers") or {})})
    for s in unknown:
        row(f"未登记的 {s}：被正确拒识", lambda e, k=s: f"{num(get(e, 'unknown_speakers', k, 'rejected'))} / {num(get(e, 'unknown_speakers', k, 'trials'))}")
    return rows


def section_speakers(results: list[dict[str, Any]]) -> str:
    labels = [r["label"] for r in results]
    rows = _speaker_rows(results, lambda r: r)
    for key, zh in CONDITIONS:
        rows.append([f"{zh}台词被转写的时长占比"] + [pct(get(r, "line_coverage", key, "covered_share")) for r in results])
    rows.append(["阈值 ±0.10 内的判定占比"] + [pct(get(r, "near_threshold", "share")) for r in results])
    return "\n\n".join(["## 说话人（流水线的实际判定）", "每格：准确率 / FRR（本人没认出）/ FAR（未登记的人被认成某人）。",
                        table(["指标"] + labels, rows)])


def section_replay(results: list[dict[str, Any]]) -> str:
    parts = ["## 说话人（向量回放，不跑模型）"]
    for r in results:
        scorers = r.get("scorers") or {}
        names = list(scorers)
        fake = [{"label": n, **scorers[n]} for n in names]
        rows = _speaker_rows(fake, lambda x: x)
        rows.append(["FRR = FAR 处的阈值"] + [f"{num(get(scorers[n], 'equal_error', 'threshold'), 2)}（{pct(get(scorers[n], 'equal_error', 'frr'))}）"
                                               if scorers[n].get("equal_error") else "—" for n in names])
        parts.append(f"### {r['label']}（来源：{r.get('source')}，{r.get('trials')} 个向量）")
        parts.append(table(["指标"] + names, rows))
    return "\n\n".join(parts)


def section_cost(results: list[dict[str, Any]]) -> str:
    header = ["线程数 / 阶段"] + [r["label"] for r in results]
    rows: list[list[str]] = []
    threads = sorted({t for r in results for t in (r.get("threads") or {})}, key=int)
    for t in threads:
        for key, zh in (("diarize", "分离"), ("embed", "声纹向量"), ("identify", "判定"), ("voice_total", "声音阶段合计"), ("pipeline", "整条本机流水线")):
            rows.append([f"{t} 线程 · {zh}"] + [f"{num(get(r, 'threads', t, key, 'cpu_per_audio_s'))} / {num(get(r, 'threads', t, key, 'wall_per_audio_s'))}"
                                                 for r in results])
    return "\n\n".join(["## 本机开销", "每格：每秒音频的 CPU 秒 / 墙钟秒（CPU 为整个进程、所有线程之和）。", table(header, rows)])


def section_proxy(results: list[dict[str, Any]]) -> str:
    labels = [r.get("label", "proxy") for r in results]
    rows = [["句子数"] + [num(r.get("utterances")) for r in results]]
    for key, zh in (("sure", "确定（自动）"), ("maybe", "可能"), ("unknown", "未知"), ("user", "手标")):
        rows.append([f"判定：{zh}"] + [pct(get(r, "decisions", key, "share")) for r in results])
    rows.append(["阈值 ±带宽内的自动判定"] + [pct(get(r, "near_threshold", "share")) for r in results])
    parts = ["## 真实数据代理指标（无真值，只有汇总）", table(["指标"] + labels, rows)]
    for r in results:
        rep = r.get("replay") or {}
        if not rep:
            continue
        names = list(rep)
        rrows = []
        for key, zh in (("sure", "确定"), ("maybe", "可能"), ("unknown", "未知")):
            rrows.append([zh] + [pct(get(rep[n], key, "share")) for n in names])
        rrows.append(["阈值附近"] + [pct(get(rep[n], "near_threshold", "share")) for n in names])
        rrows.append(["与手标矛盾 / 手标数"] + [f"{num(get(rep[n], 'hand_labels', 'contradicted'))} / {num(get(rep[n], 'hand_labels', 'n'))}" for n in names])
        rrows.append(["违反否决或系统声音规则 / 约束数"] + [f"{num(get(rep[n], 'must_not', 'violated'))} / {num(get(rep[n], 'must_not', 'n'))}" for n in names])
        rrows.append(["与库里现有判定不同"] + [pct(get(rep[n], "differs_from_stored", "share")) for n in names])
        parts.append(f"回放（{r.get('label', 'proxy')}，{r.get('replay_trials')} 句，向量来自 {r.get('embedding_source')}）：")
        parts.append(table(["指标"] + names, rrows))
    return "\n\n".join(parts)


SECTIONS = (("cer", section_cer), ("speakers", section_speakers), ("speakers_replay", section_replay), ("cost", section_cost),
            ("speakers_proxy", section_proxy))


def build_report(results: list[dict[str, Any]], title: str = "audio-note 评测汇总") -> str:
    parts = [f"# {title}"]
    codes = []
    for r in results:
        code = r.get("code") or get(r, "import", "code")
        if code:
            codes.append(f"- `{r.get('label')}`（{r.get('kind')}）：版本 {code.get('version') or '—'}，提交 {code.get('commit') or '—'}")
    if codes:
        parts.append("\n".join(dict.fromkeys(codes)))
    for kind, fn in SECTIONS:
        chosen = [r for r in results if r.get("kind") == kind]
        if chosen:
            parts.append(fn(chosen))
    return "\n\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path, help="result JSON files")
    ap.add_argument("--title", default="audio-note 评测汇总")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    results = []
    for f in args.files:
        r = common.read_json(f)
        if not isinstance(r, dict) or "kind" not in r:
            print(f"skip {f}: not a bench result", file=sys.stderr)
            continue
        r.setdefault("label", f.stem)
        results.append(r)
    md = build_report(results, args.title)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8", newline="\n")
        print(f"-> {args.out}")
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
