"""What the local stages cost: CPU-seconds and wall-seconds per second of audio at 1 / 2 / 4 model threads.

    python scripts/bench/bench_cost.py --root R --out cost.json                  # threads 1,2,4 on the mixed set
    python scripts/bench/bench_cost.py --root R --threads 1,2 --sessions d1s1,d2s4 --label v1 --compare cost.json

For every thread count a fresh Python process imports the rendered sessions into a throw-away profile
(speakers.num_threads = N, a stub instead of the ASR, correction off: no network, no key) and runs the
processing pipeline of the code under test with its speaker engine wrapped in timers:
    diarize    engine.diarize                          (segmentation + clustering)
    embed      engine.embed                            (speaker embeddings: clusters, turn checks, samples)
    identify   the matching logic around the embeddings (exclusive of embed)
    pipeline   the whole process_pending() run: the three above plus decoding, loudness, database, Markdown
    vad_gate   cutting the session files into segments (what the recorder does live)
CPU is the whole process (time.process_time), i.e. summed over threads - the number that drains a battery.
One process per thread count keeps onnxruntime's thread pools and the CPU accounting separate. When the
database of the code under test has a usage_local table (1.0), its rows are included for cross-checking.
Results depend on what else the machine is doing: run it on an otherwise idle machine and compare like with like.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from . import build, common
except ImportError:  # run as a script
    import build
    import common

MARK = "@@BENCH_COST@@"
STAGES = ("diarize", "embed", "identify")


def per_audio_second(report: dict[str, Any]) -> dict[str, Any]:
    """Turn one import report (build.import_profile) into per-audio-second costs (pure)."""
    audio = float(report.get("audio_s") or 0.0)
    stages = report.get("stages") or {}
    out: dict[str, Any] = {"audio_s": round(audio, 1), "segments": report.get("segments"), "utterances": report.get("utterances")}
    voice_cpu = voice_wall = 0.0
    for name in STAGES + ("vad_gate",):
        st = stages.get(name) or {"n": 0, "wall_s": 0.0, "cpu_s": 0.0}
        out[name] = {"calls": int(st["n"]), "cpu_s": round(st["cpu_s"], 2), "wall_s": round(st["wall_s"], 2),
                     "cpu_per_audio_s": common.rate(st["cpu_s"], audio), "wall_per_audio_s": common.rate(st["wall_s"], audio)}
        if name in STAGES:
            voice_cpu += st["cpu_s"]
            voice_wall += st["wall_s"]
    out["voice_total"] = {"cpu_s": round(voice_cpu, 2), "wall_s": round(voice_wall, 2),
                          "cpu_per_audio_s": common.rate(voice_cpu, audio), "wall_per_audio_s": common.rate(voice_wall, audio)}
    out["pipeline"] = {"cpu_s": report.get("process_cpu_s"), "wall_s": report.get("process_wall_s"),
                       "cpu_per_audio_s": common.rate(float(report.get("process_cpu_s") or 0.0), audio),
                       "wall_per_audio_s": common.rate(float(report.get("process_wall_s") or 0.0), audio)}
    return out


def usage_local(pdir: Path) -> list[dict[str, Any]] | None:
    """usage_local rows of a profile database, or None when the code under test does not keep them."""
    db = common.profile_db_path(pdir)
    if not db.exists():
        return None
    conn = common.open_ro(db)
    try:
        if not common.table_exists(conn, "usage_local"):
            return None
        return [dict(r) for r in conn.execute(
            "SELECT stage, SUM(n) AS n, ROUND(SUM(audio_s), 1) AS audio_s, ROUND(SUM(wall_s), 2) AS wall_s, "
            "ROUND(SUM(cpu_s), 2) AS cpu_s FROM usage_local GROUP BY stage ORDER BY stage")]
    finally:
        conn.close()


def worker(root: Path, name: str, variant: str, threads: int, sessions: set[str] | None, models_dir: Path | None,
           overrides: dict[str, Any] | None) -> int:
    rep = build.import_profile(root, name, variant=variant, asr="stub", threads=threads, force=True, sessions=sessions,
                               models_dir=models_dir, overrides=overrides, measure=True)
    rep["usage_local"] = usage_local(Path(rep["profile"]))
    print(MARK + json.dumps(rep, ensure_ascii=False))
    return 0


def run(root: Path, variant: str = "mixed", threads: list[int] | None = None, sessions: list[str] | None = None,
        label: str | None = None, models_dir: Path | None = None, keep: bool = False, sets: list[str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": "cost", "label": label or "cost", "variant": variant, "threads": {}, "headline": {},
                           "created_at": datetime.now().isoformat(timespec="seconds")}
    for n in threads or [1, 2, 4]:
        name = f"cost_t{n}"
        cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--root", str(root), "--variant", variant,
               "--threads", str(n), "--name", name]
        if sessions:
            cmd += ["--sessions", ",".join(sessions)]
        if models_dir:
            cmd += ["--models-dir", str(models_dir)]
        for s in sets or []:
            cmd += ["--set", s]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(MARK)), None)
        if proc.returncode != 0 or line is None:
            tail = "\n".join((proc.stderr or proc.stdout or "").splitlines()[-15:])
            raise RuntimeError(f"cost worker for {n} thread(s) failed (exit {proc.returncode}):\n{common.scrub(tail)}")
        rep = json.loads(line[len(MARK):])
        res = per_audio_second(rep)
        res["usage_local"] = rep.get("usage_local")
        res["failed_segments"] = (rep.get("process") or {}).get("failed")
        out["threads"][str(n)] = res
        out["code"] = rep.get("code")
        out["audio_s"] = res["audio_s"]
        for key in ("voice_total", "pipeline") + STAGES:
            out["headline"][f"{key}_cpu_per_audio_s@{n}"] = res[key]["cpu_per_audio_s"]
        out["headline"][f"pipeline_wall_per_audio_s@{n}"] = res["pipeline"]["wall_per_audio_s"]
        if not keep:
            try:
                build._safe_rmtree(common.profile_dir(root, name), root)
            except (OSError, RuntimeError):
                pass
    return out


def format_text(res: dict[str, Any]) -> str:
    rows = [f"== cost [{res['label']}] variant={res['variant']} audio={res.get('audio_s')} s  (CPU-s / wall-s per second of audio)"]
    rows.append(f"  {'threads':<8}{'diarize':>16}{'embed':>16}{'identify':>16}{'voice total':>16}{'pipeline':>16}")
    for n, r in res["threads"].items():
        def cell(k: str) -> str:
            return f"{r[k]['cpu_per_audio_s']:.3f} / {r[k]['wall_per_audio_s']:.3f}"

        rows.append(f"  {n:<8}{cell('diarize'):>16}{cell('embed'):>16}{cell('identify'):>16}{cell('voice_total'):>16}{cell('pipeline'):>16}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--variant", default="mixed", choices=build.VARIANTS)
    ap.add_argument("--threads", default="1,2,4", help="comma-separated thread counts (worker: a single number)")
    ap.add_argument("--sessions", default=None, help="comma-separated session ids (default: all)")
    ap.add_argument("--set", action="append", default=[], metavar="section.key=value", help="config override for the profile")
    ap.add_argument("--models-dir", type=Path, default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--keep", action="store_true", help="keep the temporary profiles (cost_t<N>)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--name", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    root = args.root or common.default_root()
    sessions = [s for s in args.sessions.split(",") if s] if args.sessions else None
    if args.worker:
        return worker(root, args.name or f"cost_t{args.threads}", args.variant, int(args.threads),
                      set(sessions) if sessions else None, args.models_dir, build._parse_overrides(args.set))
    res = run(root, args.variant, [int(x) for x in args.threads.split(",") if x], sessions, args.label, args.models_dir,
              args.keep, args.set)
    print(format_text(res))
    if args.compare:
        old = common.read_json(args.compare)
        print(f"-- compared with [{old.get('label')}]")
        for k, v in res["headline"].items():
            ov = (old.get("headline") or {}).get(k)
            if isinstance(v, (int, float)) and isinstance(ov, (int, float)) and ov:
                print(f"  {k:<36} {ov:.3f} -> {v:.3f}  ({100 * (v - ov) / ov:+.0f}%)")
    if args.out:
        common.write_json(args.out, res)
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
