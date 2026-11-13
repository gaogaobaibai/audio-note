"""Transcription quality of a bench profile against the story's ground truth.

    python scripts/bench/bench_cer.py --profile baseline --root R --out cer.json
    python scripts/bench/bench_cer.py --profile R/profiles/v1 --out v1.json --compare cer.json

Reports, overall and per acoustic condition / session / day / speaker:
  * CER of the raw ASR text (utterances.text_raw) and of the stored text after correction (utterances.text);
  * proper-noun error rate: share of the occurrences of the story's hard terms (names, jargon, figures) that did
    not come out exactly right, raw vs corrected;
  * every change the correction made, classified against the truth as helpful (closer to what was said; "exact"
    when the changed span now equals the truth), harmful (further away) or neutral, how many changes brought in
    a lexicon term the speaker never said, plus purely cosmetic rewrites (punctuation / spacing / numeral style);
  * line coverage: how many truth lines were transcribed at all (far-field lines may be dropped as too quiet);
  * (1.0 databases) open questions and whether they point at real errors, revisions by source, LLM usage.

Method: per session the truth lines (time order) and the utterances (time order) are normalised
(textnorm.normalize: punctuation, width, case, numerals) and aligned once at character level; every error is
attributed to the truth line owning the reference character, so per-condition numbers do not depend on how the
pipeline cut the audio. The database is opened read-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from . import align, common, textnorm
except ImportError:  # run as a script
    import align
    import common
    import textnorm

CONDITIONS = ("near", "far", "overlap", "system")
COVERED_MIN = 0.5          # a line counts as transcribed when at least half of it lies inside utterances
MISSED_MAX = 0.1


# ------------------------------------------------------------------------------------------------ loading
def session_of_segment(origin: str | None, sessions: list[dict[str, Any]]) -> str | None:
    if not origin:
        return None
    name = Path(str(origin).replace("\\", "/")).name
    for s in sessions:
        if s["file"] == name:
            return s["id"]
    return None


def load_utterances(db_path: Path | str, truth: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """{session id: [utterance dicts in time order]} with times relative to the session start (seconds)."""
    conn = common.open_ro(db_path)
    try:
        ucols = common.columns(conn, "utterances")
        extra = [c for c in ("speaker_source", "loudness_db", "system_share", "text_source", "flags", "speaker_guess_id",
                             "speaker_guess_score", "session_id") if c in ucols]
        sel = ", ".join(["u.id", "u.segment_id", "u.start_s", "u.end_s", "u.abs_start", "u.day", "u.cluster_label",
                         "u.speaker_id", "u.speaker_name", "u.speaker_score", "u.text_raw", "u.text"] + [f"u.{c}" for c in extra])
        rows = conn.execute(f"SELECT {sel}, s.origin AS origin FROM utterances u JOIN segments s ON s.id = u.segment_id "
                            "ORDER BY u.abs_start, u.id").fetchall()
    finally:
        conn.close()
    sessions = truth["sessions"]
    starts = {s["id"]: datetime.fromisoformat(s["start"]) for s in sessions}
    out: dict[str, list[dict[str, Any]]] = {s["id"]: [] for s in sessions}
    for r in rows:
        d = dict(r)
        t = datetime.fromisoformat(d["abs_start"])
        sid = session_of_segment(d.pop("origin"), sessions)
        if sid is None:         # not imported by build.py: fall back to the time window
            for s in sessions:
                off = (t - starts[s["id"]]).total_seconds()
                if -1.0 <= off <= float(s["duration_s"]) + 1.0:
                    sid = s["id"]
                    break
        if sid is None:
            continue
        d["u0"] = (t - starts[sid]).total_seconds()
        d["u1"] = d["u0"] + max(0.0, float(d["end_s"]) - float(d["start_s"]))
        out[sid].append(d)
    for lst in out.values():
        lst.sort(key=lambda u: (u["u0"], u["id"]))
    return out


# ------------------------------------------------------------------------------------------------ scoring (pure)
def _zero() -> dict[str, int]:
    return {"n": 0, "hits": 0, "sub": 0, "del": 0, "ins": 0}


def _add(acc: dict[str, int], c: dict[str, int] | None) -> None:
    if c:
        for k in ("n", "hits", "sub", "del", "ins"):
            acc[k] += int(c.get(k, 0))


def _finish(c: dict[str, int]) -> dict[str, Any]:
    err = c["sub"] + c["del"] + c["ins"]
    return {**c, "errors": err, "cer": common.rate(err, c["n"])}


def _term_bucket() -> dict[str, int]:
    return {"n": 0, "wrong_raw": 0, "wrong_corrected": 0}


def _term_finish(b: dict[str, int]) -> dict[str, Any]:
    return {**b, "rate_raw": common.rate(b["wrong_raw"], b["n"]), "rate_corrected": common.rate(b["wrong_corrected"], b["n"])}


def score_session(sess: dict[str, Any], utts: list[dict[str, Any]], terms: list[dict[str, Any]], numerals: bool = True) -> dict[str, Any]:
    """Everything about one session: per-line error counts (raw / corrected), term occurrences, edits, coverage."""
    def norm(t: str | None) -> str:
        return textnorm.normalize(t, numerals=numerals)

    lines = sorted(sess["lines"], key=lambda ln: (ln["t0"], ln["i"]))
    spans = [(float(ln["t0"]), float(ln["t1"])) for ln in lines]
    utts = sorted(utts, key=lambda u: (u["u0"], u.get("id", 0)))
    uspans = [(float(u["u0"]), float(u["u1"])) for u in utts]
    ref_texts = [norm(ln["text"]) for ln in lines]
    ref, ref_owner = align.concat(ref_texts)
    raw_texts = [norm(u.get("text_raw")) for u in utts]
    cor_texts = [norm(u.get("text")) for u in utts]
    hyp_raw, raw_owner = align.concat(raw_texts)
    hyp_cor, cor_owner = align.concat(cor_texts)
    ops_raw = align.align(ref, hyp_raw)
    ops_cor = align.align(ref, hyp_cor)
    line_raw, utt_raw = align.attribute(ops_raw, ref_owner, raw_owner)
    line_cor, utt_cor = align.attribute(ops_cor, ref_owner, cor_owner)
    coverage = align.line_coverage(uspans, spans)
    assigned = align.assign_lines(uspans, spans)

    line_out = []
    for k, ln in enumerate(lines):
        line_out.append({"session": sess["id"], "i": ln["i"], "speaker": ln["speaker"], "condition": ln["condition"],
                         "chars": len(ref_texts[k]), "coverage": round(coverage[k], 3),
                         "raw": line_raw.get(k, _zero()), "corrected": line_cor.get(k, _zero())})

    norm_terms = {}
    for t in terms:
        nt = norm(t["term"])
        if nt:
            norm_terms[nt] = t
    hits_raw = align.term_hits(ref, ops_raw, list(norm_terms), ref_owner)
    hits_cor = align.term_hits(ref, ops_cor, list(norm_terms), ref_owner)
    cor_ok = {(h["term"], h["start"]): h["ok"] for h in hits_cor}
    term_out = []
    for h in hits_raw:
        t = norm_terms[h["term"]]
        ln = lines[h["owner"]]
        term_out.append({"term": t["term"], "group": t.get("group", "term"), "session": sess["id"], "line": ln["i"],
                         "condition": ln["condition"], "covered": coverage[h["owner"]] >= COVERED_MIN,
                         "ok_raw": bool(h["ok"]), "ok_corrected": bool(cor_ok.get((h["term"], h["start"]), False))})

    # --- edits: what the correction changed, judged against the truth
    h2r = align.hyp_to_ref(ops_raw, len(hyp_raw))
    line_start = [0] * (len(lines) + 1)
    for k, t in enumerate(ref_texts):
        line_start[k + 1] = line_start[k] + len(t)
    edits: list[dict[str, Any]] = []
    cosmetic = 0
    changed = 0
    utt_verdicts = {"helpful": 0, "harmful": 0, "neutral": 0}
    offset = 0
    for k, u in enumerate(utts):
        nr, nc = raw_texts[k], cor_texts[k]
        if (u.get("text_raw") or "") != (u.get("text") or ""):
            if nr == nc:
                cosmetic += 1
            else:
                changed += 1
                e_raw = utt_raw.get(k, _zero())
                e_cor = utt_cor.get(k, _zero())
                delta = (e_cor["sub"] + e_cor["del"] + e_cor["ins"]) - (e_raw["sub"] + e_raw["del"] + e_raw["ins"])
                utt_verdicts["helpful" if delta < 0 else "harmful" if delta > 0 else "neutral"] += 1
                touched = [assigned[k]["line"]] + [o[0] for o in assigned[k]["others"]] if assigned[k]["line"] is not None else []
                lo_lim = line_start[min(touched)] if touched else 0
                hi_lim = line_start[max(touched) + 1] if touched else 0
                for a, b, repl in align.diff_regions(nr, nc):
                    lo, hi = align.ref_window(h2r, len(ref), offset + a, offset + b)
                    lo, hi = max(lo, lo_lim), min(hi, hi_lim)
                    target = ref[lo:hi] if hi > lo else ""
                    before = align.distance(target, nr[a:b])
                    after = align.distance(target, repl)
                    verdict = "helpful" if after < before else "harmful" if after > before else "neutral"
                    owner = ref_owner[lo] if lo < len(ref_owner) and hi > lo else (assigned[k]["line"] if assigned[k]["line"] is not None else None)
                    injected = [t["term"] for nt, t in norm_terms.items() if nt in repl and nt not in target and nt not in nr[a:b]]
                    edits.append({"session": sess["id"], "utterance_id": u.get("id"), "from": nr[a:b], "to": repl,
                                  "truth": target, "before": before, "after": after, "verdict": verdict,
                                  "exact": after == 0 and before > 0, "injected_terms": injected,
                                  "condition": lines[owner]["condition"] if owner is not None else None})
        offset += len(nr)

    stray = [k for k, a in enumerate(assigned) if a["line"] is None]
    return {
        "id": sess["id"], "day": sess["day"], "kind": sess.get("kind"), "lines": line_out, "terms": term_out, "edits": edits,
        "utterances": len(utts), "changed_utterances": changed, "cosmetic_utterances": cosmetic,
        "utterance_verdicts": utt_verdicts,
        "stray_utterances": len(stray), "stray_chars": sum(len(cor_texts[k]) for k in stray),
        "_align": {"ref": ref, "ref_owner": ref_owner, "hyp_raw": hyp_raw, "hyp_cor": hyp_cor, "ops_raw": ops_raw, "ops_cor": ops_cor,
                   "raw_owner": raw_owner, "cor_owner": cor_owner, "utt_ids": [u.get("id") for u in utts],
                   "raw_texts": raw_texts, "cor_texts": cor_texts},
    }


def aggregate(per_session: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll the per-session records up: overall, covered lines only, by condition / session / day / speaker."""
    def bucket() -> dict[str, Any]:
        return {"lines": 0, "covered": 0, "missed": 0, "raw": _zero(), "corrected": _zero()}

    overall, covered_only = bucket(), bucket()
    by: dict[str, dict[str, dict[str, Any]]] = {"condition": {}, "session": {}, "day": {}, "speaker": {}}
    for s in per_session:
        for ln in s["lines"]:
            keys = {"condition": ln["condition"], "session": s["id"], "day": s["day"], "speaker": ln["speaker"]}
            targets = [overall] + [by[dim].setdefault(str(v), bucket()) for dim, v in keys.items()]
            if ln["coverage"] >= COVERED_MIN:
                targets.append(covered_only)
            for b in targets:
                b["lines"] += 1
                b["covered"] += ln["coverage"] >= COVERED_MIN
                b["missed"] += ln["coverage"] < MISSED_MAX
                _add(b["raw"], ln["raw"])
                _add(b["corrected"], ln["corrected"])

    def fin(b: dict[str, Any]) -> dict[str, Any]:
        return {"lines": b["lines"], "covered_lines": int(b["covered"]), "missed_lines": int(b["missed"]),
                "raw": _finish(b["raw"]), "corrected": _finish(b["corrected"])}

    terms_all = _term_bucket()
    terms_cov = _term_bucket()
    t_by: dict[str, dict[str, dict[str, int]]] = {"group": {}, "condition": {}, "day": {}, "term": {}}
    day_of = {s["id"]: s["day"] for s in per_session}
    for s in per_session:
        for t in s["terms"]:
            keys = {"group": t["group"], "condition": t["condition"], "day": day_of[t["session"]], "term": t["term"]}
            targets = [terms_all] + [t_by[dim].setdefault(str(v), _term_bucket()) for dim, v in keys.items()]
            if t["covered"]:
                targets.append(terms_cov)
            for b in targets:
                b["n"] += 1
                b["wrong_raw"] += not t["ok_raw"]
                b["wrong_corrected"] += not t["ok_corrected"]

    edits = [e for s in per_session for e in s["edits"]]
    e_tot = {"regions": len(edits), "helpful": 0, "harmful": 0, "neutral": 0, "exact": 0, "wrong_term_injections": 0,
             "chars_fixed": 0, "chars_broken": 0}
    e_by_cond: dict[str, dict[str, int]] = {}
    for e in edits:
        e_tot[e["verdict"]] += 1
        e_tot["exact"] += bool(e.get("exact"))
        e_tot["wrong_term_injections"] += bool(e.get("injected_terms"))
        gain = e["before"] - e["after"]
        e_tot["chars_fixed"] += max(0, gain)
        e_tot["chars_broken"] += max(0, -gain)
        c = e_by_cond.setdefault(str(e["condition"]), {"helpful": 0, "harmful": 0, "neutral": 0})
        c[e["verdict"]] += 1
    uv = {"helpful": 0, "harmful": 0, "neutral": 0}
    for s in per_session:
        for k, v in s["utterance_verdicts"].items():
            uv[k] += v
    return {
        "overall": fin(overall), "covered_lines_only": fin(covered_only),
        "by_condition": {k: fin(v) for k, v in sorted(by["condition"].items())},
        "by_session": {k: fin(v) for k, v in sorted(by["session"].items())},
        "by_day": {k: fin(v) for k, v in sorted(by["day"].items())},
        "by_speaker": {k: fin(v) for k, v in sorted(by["speaker"].items())},
        "terms": {"overall": _term_finish(terms_all), "covered_lines_only": _term_finish(terms_cov),
                  "by_group": {k: _term_finish(v) for k, v in sorted(t_by["group"].items())},
                  "by_condition": {k: _term_finish(v) for k, v in sorted(t_by["condition"].items())},
                  "by_day": {k: _term_finish(v) for k, v in sorted(t_by["day"].items())},
                  "by_term": {k: _term_finish(v) for k, v in sorted(t_by["term"].items())}},
        "edits": {**e_tot, "by_condition": e_by_cond,
                  "changed_utterances": sum(s["changed_utterances"] for s in per_session),
                  "cosmetic_utterances": sum(s["cosmetic_utterances"] for s in per_session),
                  "utterances_net": uv},
        "utterances": sum(s["utterances"] for s in per_session),
        "stray_utterances": sum(s["stray_utterances"] for s in per_session),
        "stray_chars": sum(s["stray_chars"] for s in per_session),
    }


def score(truth: dict[str, Any], utts_by_session: dict[str, list[dict[str, Any]]], numerals: bool = True,
          terms: list[dict[str, Any]] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(aggregated result, per-session details). Pure: truth + utterance dicts in, numbers out."""
    terms = terms if terms is not None else list(truth.get("terms") or [])
    per_session = [score_session(s, utts_by_session.get(s["id"], []), terms, numerals) for s in truth["sessions"]]
    return aggregate(per_session), per_session


# ------------------------------------------------------------------------------------------------ 1.0 extras
def question_stats(db_path: Path | str, per_session: list[dict[str, Any]], numerals: bool = True) -> dict[str, Any] | None:
    """Open / answered 'term' questions of a 1.0 database and whether each one points at a real error:
    the asked span is located in the utterance's text and compared with the truth it is aligned to.
    None when the database has no inbox_items table (pre-1.0 code)."""
    conn = common.open_ro(db_path)
    try:
        if not common.table_exists(conn, "inbox_items"):
            return None
        rows = [dict(r) for r in conn.execute("SELECT id, kind, status, payload, utterance_id FROM inbox_items")]
    finally:
        conn.close()
    where: dict[Any, tuple[dict[str, Any], int]] = {}
    for s in per_session:
        for k, uid in enumerate(s["_align"]["utt_ids"]):
            where[uid] = (s, k)
    by_kind: dict[str, int] = {}
    by_status: dict[str, int] = {}
    asked = located = hits = 0
    oracle: list[dict[str, Any]] = []
    for r in rows:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
        if r["kind"] != "term":
            continue
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        asked += 1
        try:
            payload = json.loads(r["payload"] or "{}")
        except ValueError:
            payload = {}
        span = textnorm.normalize(payload.get("span"), numerals=numerals)
        uid = payload.get("utterance_id") or r["utterance_id"]
        if not span or uid not in where:
            continue
        s, k = where[uid]
        al = s["_align"]
        found = None
        for texts, hyp, ops in ((al["cor_texts"], al["hyp_cor"], al["ops_cor"]), (al["raw_texts"], al["hyp_raw"], al["ops_raw"])):
            pos = texts[k].find(span)
            if pos >= 0:
                found = (sum(len(t) for t in texts[:k]) + pos, hyp, ops)
                break
        if found is None:
            continue
        located += 1
        a, hyp, ops = found
        b = a + len(span)
        h2r = align.hyp_to_ref(ops, len(hyp))
        ok = {j for op, _i, j in ops if op == "="}
        wrong = not all(j in ok for j in range(a, b))
        hits += wrong
        lo, hi = align.ref_window(h2r, len(al["ref"]), a, b)
        oracle.append({"inbox_id": r["id"], "status": r["status"], "span": payload.get("span"), "utterance_id": uid,
                       "really_wrong": bool(wrong), "truth": al["ref"][lo:hi]})
    return {"by_kind": by_kind, "term_by_status": by_status, "term_questions": asked, "located": located,
            "pointing_at_errors": hits, "hit_rate": common.rate(hits, located), "oracle": oracle}


def db_extras(db_path: Path | str) -> dict[str, Any]:
    """Revision counts by source and LLM usage, when the 1.0 tables exist."""
    out: dict[str, Any] = {}
    conn = common.open_ro(db_path)
    try:
        if common.table_exists(conn, "utterance_revisions"):
            out["revisions_by_source"] = {r[0]: r[1] for r in conn.execute(
                "SELECT source, COUNT(*) FROM utterance_revisions WHERE undone_at IS NULL GROUP BY source")}
        if common.table_exists(conn, "usage_llm"):
            out["usage_llm"] = [dict(r) for r in conn.execute(
                "SELECT purpose, model, SUM(calls) AS calls, SUM(errors) AS errors, SUM(prompt_tokens) AS prompt_tokens, "
                "SUM(completion_tokens) AS completion_tokens, ROUND(SUM(audio_s), 1) AS audio_s, ROUND(SUM(wall_s), 1) AS wall_s "
                "FROM usage_llm GROUP BY purpose, model ORDER BY purpose, model")]
        if common.table_exists(conn, "lexicon"):
            out["lexicon_terms"] = conn.execute("SELECT COUNT(*) FROM lexicon").fetchone()[0]
        if "text_source" in common.columns(conn, "utterances"):
            out["text_source"] = {str(r[0]): r[1] for r in conn.execute("SELECT text_source, COUNT(*) FROM utterances GROUP BY text_source")}
    finally:
        conn.close()
    return out


# ------------------------------------------------------------------------------------------------ report
def headline(result: dict[str, Any]) -> dict[str, Any]:
    o, t, e = result["overall"], result["terms"], result["edits"]
    return {
        "cer_raw": o["raw"]["cer"], "cer_corrected": o["corrected"]["cer"],
        "cer_raw_covered": result["covered_lines_only"]["raw"]["cer"],
        "cer_corrected_covered": result["covered_lines_only"]["corrected"]["cer"],
        "term_error_raw": t["overall"]["rate_raw"], "term_error_corrected": t["overall"]["rate_corrected"],
        "edits_helpful": e["helpful"], "edits_harmful": e["harmful"], "edits_neutral": e["neutral"],
        "edits_exact": e["exact"], "edits_wrong_term_injections": e["wrong_term_injections"],
        "lines": o["lines"], "missed_lines": o["missed_lines"],
    }


def run(profile: Path, truth_path: Path | None = None, label: str | None = None, numerals: bool = True,
        keep_details: bool = True) -> dict[str, Any]:
    truth = common.load_truth(truth_path or profile)
    db_path = common.profile_db_path(profile)
    utts = load_utterances(db_path, truth)
    result, per_session = score(truth, utts, numerals)
    out: dict[str, Any] = {"kind": "cer", "label": label or profile.name, "profile": str(profile),
                           "variant": truth.get("variant"), "story": truth.get("story"), "numerals": numerals,
                           "created_at": datetime.now().isoformat(timespec="seconds")}
    imp = profile / "bench_import.json"
    if imp.exists():
        info = common.read_json(imp)
        out["import"] = {k: info.get(k) for k in ("code", "asr", "glossary", "variant", "overrides", "asr_stats", "client_usage",
                                                  "process_wall_s", "audio_s", "segments", "segment_status")}
    out["headline"] = headline(result)
    out.update(result)
    q = question_stats(db_path, per_session, numerals)
    if q is not None:
        out["questions"] = q
    out["extras"] = db_extras(db_path)
    if keep_details:
        out["edit_list"] = [e for s in per_session for e in s["edits"]]
        out["line_list"] = [ln for s in per_session for ln in s["lines"]]
    return out


def _f(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.2f}%"


def format_text(res: dict[str, Any], old: dict[str, Any] | None = None) -> str:
    rows = [f"== CER  [{res['label']}]  variant={res.get('variant')}  utterances={res['utterances']}"]

    def line(name: str, b: dict[str, Any]) -> str:
        return (f"  {name:<14} lines {b['lines']:>3} (missed {b['missed_lines']:>2})  chars {b['raw']['n']:>5}  "
                f"CER raw {_f(b['raw']['cer']):>7}  corrected {_f(b['corrected']['cer']):>7}")

    rows.append(line("overall", res["overall"]))
    rows.append(line("covered lines", res["covered_lines_only"]))
    for c in CONDITIONS:
        if c in res["by_condition"]:
            rows.append(line(c, res["by_condition"][c]))
    t = res["terms"]
    rows.append(f"  proper nouns   {t['overall']['n']} occurrences: wrong raw {_f(t['overall']['rate_raw'])}, corrected {_f(t['overall']['rate_corrected'])}")
    for g, b in t["by_group"].items():
        rows.append(f"    {g:<8} n={b['n']:<4} raw {_f(b['rate_raw']):>7}  corrected {_f(b['rate_corrected']):>7}")
    e = res["edits"]
    rows.append(f"  edits          {e['regions']} changes in {e['changed_utterances']} utterances: helpful {e['helpful']} (exactly right {e['exact']}), "
                f"harmful {e['harmful']}, neutral {e['neutral']}; chars fixed {e['chars_fixed']}, broken {e['chars_broken']}; "
                f"wrongly introduced terms {e['wrong_term_injections']}; cosmetic-only utterances {e['cosmetic_utterances']}")
    if res.get("questions"):
        q = res["questions"]
        rows.append(f"  questions      {q['term_questions']} term questions, {q['located']} located, {q['pointing_at_errors']} point at real errors "
                    f"({_f(q['hit_rate'])})")
    if old:
        rows.append(f"-- compared with [{old.get('label')}]")
        for k, v in res["headline"].items():
            ov = (old.get("headline") or {}).get(k)
            if isinstance(v, float) and isinstance(ov, float):
                rows.append(f"  {k:<30} {_f(ov):>8} -> {_f(v):>8}  ({100 * (v - ov):+.2f} pt)")
            elif v is not None and ov is not None:
                rows.append(f"  {k:<30} {ov!s:>8} -> {v!s:>8}")
    return "\n".join(rows)


def compare(new: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"against": old.get("label"), "headline": {}}
    for k, v in new["headline"].items():
        ov = (old.get("headline") or {}).get(k)
        if isinstance(v, (int, float)) and isinstance(ov, (int, float)):
            out["headline"][k] = {"old": ov, "new": v, "delta": round(v - ov, 4)}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True, help="bench profile directory, or its name under <root>/profiles")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--truth", type=Path, default=None, help="truth.json (default: the copy inside the profile)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--no-numerals", action="store_true", help="do not unify Chinese / Arabic numerals")
    ap.add_argument("--out", type=Path, default=None, help="write the full result as JSON")
    ap.add_argument("--compare", type=Path, default=None, help="an earlier result JSON to compare with")
    args = ap.parse_args(argv)
    pdir = common.resolve_profile(args.profile, args.root)
    res = run(pdir, args.truth, args.label, numerals=not args.no_numerals)
    old = common.read_json(args.compare) if args.compare else None
    if old:
        res["compare"] = compare(res, old)
    print(format_text(res, old))
    if args.out:
        common.write_json(args.out, res)
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
