"""Generate a fake multi-day, multi-speaker "life log" with MiMo TTS and run the whole pipeline on it.

    python scripts/fake_days.py --out <dir> --days 3 --start 2026-10-27          # write script + audio
    python scripts/fake_days.py --out <dir> --run                                 # enroll, import, process, daily, evaluate
    python scripts/fake_days.py --out <dir> --run --ask "星河项目什么时候上线？"   # plus memory-chat questions

Needs config.yaml (or AUDIO_NOTE_CONFIG) with a working MiMo key. The script is written by mimo-v2.6-flash,
each line is voiced by a fixed preset voice per character (mimo-v2.5-tts, free for now), and lines are
concatenated with pauses into one wav per session. truth.json keeps who-said-what-when so --run can score
speaker identification against ground truth.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio_note.config import load_config  # noqa: E402
from audio_note.llm.client import MimoClient  # noqa: E402

log = logging.getLogger("fake_days")
SR = 16000

# name -> (voice, description used in the script prompt). "我" is the recorder's owner.
CAST = {
    "我": ("苏打", "三十岁左右的男性产品经理，录音设备的主人，说话直接、偏口语"),
    "王敏": ("冰糖", "后端工程师，女，说话条理清晰，爱报数字"),
    "李强": ("白桦", "团队负责人/老板，男，语速快，喜欢下结论和派任务"),
    "小雨": ("茉莉", "“我”的伴侣，女，晚上在家聊天，关心生活琐事和周末安排"),
}
# appears once, NOT enrolled -> should come out as 未知
GUEST = {"快递员": "三十岁左右的男性，普通话带一点口音，语速快，嗓音略粗，说话干脆"}

ENROLL_TEXT = [
    "你好，这是一段用来注册声纹的样本。我会用平时说话的语速和音量，读几句普通的话。",
    "今天天气不错，下午大概三点左右开会，记得把上周的数据整理一下带过来。",
]


def _client() -> MimoClient:
    return MimoClient(load_config(), rpm=90)


# ----------------------------------------------------------------------------- script
SCRIPT_SYSTEM = (
    "你是一位编剧，要为一个“全天录音的个人记忆助手”生成测试用的真实感对话剧本。只输出 JSON。"
)


def script_prompt(days: list[str], cast: dict[str, tuple[str, str]]) -> str:
    cast_desc = "\n".join(f"- {n}：{d}" for n, (_, d) in cast.items())
    guest_desc = "\n".join(f"- {n}：{d}" for n, d in GUEST.items())
    return f"""请写一个连续 {len(days)} 天的对话剧本，日期依次为：{', '.join(days)}。
主人公是“我”（录音设备的主人）。固定人物：
{cast_desc}
偶发人物（只在第二天出现一次、两三句话，比如送快递）：
{guest_desc}

要求：
1. 每天 4 个时段(session)：晨会(约 09:30, 我/王敏/李强)、午饭闲聊(约 12:10, 我/王敏)、下午一对一或评审(约 15:00, 我/李强，第二天穿插快递员)、晚上在家(约 20:30, 我/小雨)。
2. 每个时段 10～14 句，每句 1～3 个自然口语短句（15～45 个汉字），不要旁白、不要舞台提示、不要括号。
3. 剧情要连续并且有可被提问的具体事实：一个叫“星河”的 App 要上线，上线日期在第一天讨论、第二天由李强拍板定下（给出具体日期）；有具体的 bug 数量、接口数量、预算金额、人名、地点；每天都有明确的待办和决定；晚上有周末计划、家人生日、买东西等生活细节；第三天要回顾前两天的事情。
4. 说话人名字只能用：{', '.join(list(cast) + list(GUEST))}。
5. 输出 JSON：{{"days":[{{"date":"YYYY-MM-DD","sessions":[{{"title":"晨会","start":"09:30","lines":[{{"speaker":"我","text":"..."}}]}}]}}]}}
"""


def gen_script(out: Path, days: list[str]) -> dict:
    p = out / "script.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    c = _client()
    log.info("writing script with %s ...", c.chat_model)
    js = c.json(SCRIPT_SYSTEM, script_prompt(days, CAST), max_tokens=16000, temperature=0.8)
    names = set(CAST) | set(GUEST)
    for d in js["days"]:
        for s in d["sessions"]:
            s["lines"] = [ln for ln in s["lines"] if ln.get("speaker") in names and ln.get("text", "").strip()]
    p.write_text(json.dumps(js, ensure_ascii=False, indent=1), encoding="utf-8")
    n = sum(len(s["lines"]) for d in js["days"] for s in d["sessions"])
    log.info("script: %d days, %d lines -> %s", len(js["days"]), n, p)
    return js


# ----------------------------------------------------------------------------- tts
def _resample(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    n = int(round(len(x) * sr_out / sr_in))
    return np.interp(np.arange(n) / sr_out, np.arange(len(x)) / sr_in, x).astype(np.float32)


def tts(client: MimoClient, speaker: str, text: str, cache: Path) -> np.ndarray:
    key = hashlib.sha1(f"{speaker}|{text}".encode()).hexdigest()[:16]
    f = cache / f"{key}.wav"
    if f.exists():
        a, sr = sf.read(str(f), dtype="float32")
        return a
    if speaker in CAST:
        model, kwargs = "mimo-v2.5-tts", {"format": "wav", "voice": CAST[speaker][0]}
        user = "平静自然的日常对话语气，正常语速"
    else:
        model, kwargs = "mimo-v2.5-tts-voicedesign", {"format": "wav"}
        user = GUEST[speaker]
    for attempt in range(4):
        try:
            client.limiter.wait()
            r = client.client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": user}, {"role": "assistant", "content": text}], audio=kwargs,
            )
            b = base64.b64decode(r.choices[0].message.audio.data)
            a, sr = sf.read(io.BytesIO(b), dtype="float32")
            if a.ndim > 1:
                a = a.mean(axis=1)
            a = _resample(a, sr)
            sf.write(str(f), a, SR, subtype="PCM_16")
            return a
        except Exception as e:  # noqa: BLE001
            log.warning("tts failed (%s) attempt %d: %s", speaker, attempt + 1, e)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"tts failed for {speaker}: {text[:30]}")


def render(out: Path, js: dict, seed: int = 7) -> dict:
    """Voice every line, build per-session wavs, write truth.json and enrollment wavs."""
    cache = out / "tts_cache"
    cache.mkdir(parents=True, exist_ok=True)
    sess_dir = out / "sessions"
    sess_dir.mkdir(exist_ok=True)
    enroll_dir = out / "enroll"
    enroll_dir.mkdir(exist_ok=True)
    client = _client()
    rng = random.Random(seed)

    jobs: list[tuple[str, str]] = [(ln["speaker"], ln["text"]) for d in js["days"] for s in d["sessions"] for ln in s["lines"]]
    jobs += [(n, t) for n in CAST for t in ENROLL_TEXT]
    uniq = list(dict.fromkeys(jobs))
    log.info("synthesising %d unique lines with 4 workers ...", len(uniq))
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda j: tts(client, j[0], j[1], cache), uniq))

    truth = {"sessions": []}
    for d in js["days"]:
        for s in d["sessions"]:
            start = datetime.strptime(f"{d['date']} {s['start']}", "%Y-%m-%d %H:%M")
            parts: list[np.ndarray] = [np.zeros(int(0.6 * SR), np.float32)]
            pos = 0.6
            lines = []
            for ln in s["lines"]:
                a = tts(client, ln["speaker"], ln["text"], cache)
                lines.append({"speaker": ln["speaker"], "text": ln["text"], "t0": round(pos, 2), "t1": round(pos + len(a) / SR, 2)})
                parts.append(a)
                gap = rng.uniform(0.7, 1.6)
                parts.append(np.zeros(int(gap * SR), np.float32))
                pos += len(a) / SR + gap
            audio = np.concatenate(parts)
            # a little room noise so files are not digitally silent between lines
            audio = audio + (rng.random() * 0 + 0.002) * np.random.default_rng(seed).standard_normal(len(audio)).astype(np.float32)
            safe = re.sub(r"[^\w一-鿿]+", "", s["title"])[:12]
            f = sess_dir / f"{start.strftime('%Y-%m-%d_%H%M')}_{safe}.wav"
            sf.write(str(f), audio, SR, subtype="PCM_16")
            truth["sessions"].append({"file": str(f), "start": start.isoformat(), "title": s["title"], "lines": lines})
            log.info("session %s: %d lines, %.0fs", f.name, len(lines), len(audio) / SR)
    for n in CAST:
        a = np.concatenate([tts(client, n, t, cache) for t in ENROLL_TEXT])
        sf.write(str(enroll_dir / f"{n}.wav"), a, SR, subtype="PCM_16")
    (out / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
    return truth


# ----------------------------------------------------------------------------- run
def run_pipeline(out: Path, questions: list[str]) -> None:
    from audio_note.analysis.daily import DailyAnalyzer
    from audio_note.audio.importer import import_file
    from audio_note.db import DB
    from audio_note.memory.chat import MemoryChat
    from audio_note.speech.pipeline import Processor
    from audio_note.speech.speakers import SpeakerEngine

    cfg = load_config()
    cfg.ensure_dirs()
    db = DB(cfg.db_path)
    client = MimoClient(cfg)
    truth = json.loads((out / "truth.json").read_text(encoding="utf-8"))
    log.info("data dir: %s", cfg.data_dir)

    engine = SpeakerEngine(cfg)
    for n in CAST:
        engine.enroll(db, n, [out / "enroll" / f"{n}.wav"])
    log.info("enrolled: %s", [r["name"] for r in db.query("SELECT name FROM speakers")])

    t = time.time()
    for s in truth["sessions"]:
        import_file(cfg, db, Path(s["file"]), start_time=datetime.fromisoformat(s["start"]))
    log.info("imported %d sessions in %.0fs", len(truth["sessions"]), time.time() - t)

    t = time.time()
    proc = Processor(cfg, db, client=client, engine=engine)
    stats = proc.process_pending()
    log.info("processed in %.0fs: %s | API usage %s", time.time() - t, stats, client.usage)

    # --- speaker accuracy vs truth --------------------------------------------------
    total = correct = unk_ok = unk_total = 0
    confusions: dict[str, int] = {}
    for s in truth["sessions"]:
        start = datetime.fromisoformat(s["start"])
        day = start.strftime("%Y-%m-%d")
        end = start + timedelta(seconds=s["lines"][-1]["t1"] + 5)
        rows = db.query("SELECT * FROM utterances WHERE day=? AND abs_start BETWEEN ? AND ? ORDER BY abs_start",
                        (day, start.isoformat(), end.isoformat()))
        for r in rows:
            u0 = (datetime.fromisoformat(r["abs_start"]) - start).total_seconds()
            u1 = u0 + (r["end_s"] - r["start_s"])
            best, ov = None, 0.0
            for ln in s["lines"]:
                o = min(u1, ln["t1"]) - max(u0, ln["t0"])
                if o > ov:
                    best, ov = ln, o
            if best is None:
                continue
            exp = best["speaker"] if best["speaker"] in CAST else cfg.speakers.unknown_label
            got = r["speaker_name"]
            total += 1
            if exp == cfg.speakers.unknown_label:
                unk_total += 1
                unk_ok += got == exp
            correct += got == exp
            if got != exp:
                confusions[f"{exp}->{got}"] = confusions.get(f"{exp}->{got}", 0) + 1
    print(f"\n说话人识别：{correct}/{total} 条正确 ({100 * correct / max(1, total):.1f}%)；未注册的快递员 {unk_ok}/{unk_total} 条正确标为未知")
    if confusions:
        print("混淆：", confusions)

    # --- transcripts sample ------------------------------------------------------------
    days = sorted({datetime.fromisoformat(s["start"]).strftime("%Y-%m-%d") for s in truth["sessions"]})
    md = (cfg.transcripts_dir / f"{days[0]}.md").read_text(encoding="utf-8")
    print("\n--- 转写记录节选 ---\n" + "\n".join(md.splitlines()[:16]))

    # --- daily digests -------------------------------------------------------------------
    an = DailyAnalyzer(cfg, db, client)
    for d in days:
        t = time.time()
        md, _ = an.run(d)
        print(f"\n--- 日报 {d}（{time.time() - t:.0f}s）---\n" + md[:1500])

    # --- chat ------------------------------------------------------------------------------
    if questions:
        chat = MemoryChat(cfg, db, client)
        chat.on_tool = lambda name, args, res: print(f"   [tool] {name}({json.dumps(args, ensure_ascii=False)}) -> {len(res)} chars")
        for q in questions:
            t = time.time()
            print(f"\n>>> {q}")
            print(chat.ask(q))
            print(f"   ({time.time() - t:.0f}s)")
    print("\nAPI usage:", client.usage)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--start", default="2026-10-27")
    ap.add_argument("--run", action="store_true", help="run enrollment/import/process/daily/chat on the generated data")
    ap.add_argument("--ask", action="append", default=[], help="memory chat question (repeatable)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for n in ("httpx", "httpx2", "httpcore", "httpcore2", "openai"):
        logging.getLogger(n).setLevel(logging.WARNING)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d0 = datetime.strptime(args.start, "%Y-%m-%d")
    days = [(d0 + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(args.days)]
    if not (out / "truth.json").exists():
        js = gen_script(out, days)
        render(out, js)
    if args.run:
        run_pipeline(out, args.ask)
    return 0


if __name__ == "__main__":
    sys.exit(main())
