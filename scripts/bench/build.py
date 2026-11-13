"""Build the benchmark audio from story.py and import it into a throw-away bench profile.

    python scripts/bench/build.py synth  --root R --key-config <config.yaml>   # MiMo TTS -> R/tts_cache (network)
    python scripts/bench/build.py voices --root R                              # does every character keep ONE voice?
    python scripts/bench/build.py render --root R [--variant mixed|clean|far]  # session wavs + truth.json (offline)
    python scripts/bench/build.py import --root R --name baseline --key-config <config.yaml>
    python scripts/bench/build.py import --root R --name spk --asr stub        # speakers / cost only, no API at all

R defaults to $BENCH_ROOT, else <repo>/tmp_bench (git-ignored). Layout under R:
    tts_cache/<key>.wav            one 16 kHz clip per (voice, text), reused forever
    refs/<name>.wav                the ONE designed reference clip of a designed voice (cloned for every line)
    enroll/<name>_<k>.wav          enrollment samples (clean, near-field)
    sessions/<variant>/<id>.wav    one file per session (+ <id>.channels.npy when it has system audio)
    sessions/<variant>/truth.json  per line: speaker, text, t0, t1, condition, session
    asr_cache/<key>.json           ASR answers keyed by the exact request (re-runs cost nothing)
    profiles/<name>/               config.yaml (no API key in it), data/, truth.json, bench_import.json

Voices: the owner and the two colleagues use MiMo preset voices; the partner, the unenrolled friend and the
narrator are *designed* once (mimo-v2.5-tts-voicedesign) and every line is cloned from that one reference
(mimo-v2.5-tts-voiceclone: audio.voice = "data:audio/wav;base64,<reference>"), so a character keeps one voice.
`voices` verifies that with the project's own speaker embedding model. (The two female preset voices are
near-twins for that model - centroid cosine 0.79 - which is why only one of them is used.)

Acoustic conditions (variant "mixed", the default, follows the flags in story.py):
    near     clean near-field speech at --near-db (speech level, dBFS)
    far      --far-db below that (default -20 dB), convolved with an exponentially decaying noise impulse response
    overlap  the line starts before the previous speaker has finished
    system   played on the system-audio channel: the session gets (mic_rms, system_rms) pairs per 32 ms block,
             stored in segments.channel_rms exactly like a live microphone + loopback recording
A noise floor (--noise-db) lies under the microphone channel. Variant "clean" renders every line near-field
on the microphone with no overlap; "far" renders every microphone line far-field.

`import` drives the code under test the way scripts/fake_days.py does (enroll -> VAD-gated segments ->
Processor.process_pending) with an explicit config file; see common.use_project for how the pre-upgrade
baseline is selected. The MiMo key never reaches a file or the console.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import soundfile as sf

try:
    from . import audiofx, common, story
except ImportError:  # run as a script
    import audiofx
    import common
    import story

log = logging.getLogger("bench.build")
SR = common.SR
TTS_VERSION = 1                      # bump to invalidate every cached clip
TTS_MODEL = "mimo-v2.5-tts"
TTS_DESIGN_MODEL = "mimo-v2.5-tts-voicedesign"
TTS_CLONE_MODEL = "mimo-v2.5-tts-voiceclone"
VARIANTS = ("mixed", "clean", "far")

NEAR_DB = -18.0                      # speech level of near-field lines (dBFS, audiofx.active_level_db)
FAR_REL_DB = -20.0                   # far-field lines relative to near-field
NOISE_DB = -58.0                     # RMS of the room noise under the microphone channel
SYSTEM_PEAK = 0.5                    # loopback capture is levelled to a peak of 0.5
LEAD_S = 0.6
TAIL_S = 1.0
#: silence before a line (s), picked by a fixed pattern; values >= 2 s make the recorder's gate close the segment
GAPS = (0.9, 0.6, 1.3, 0.5, 2.6, 0.8, 1.1, 0.4, 0.7, 3.4, 0.9, 0.6, 1.6, 0.5, 2.3, 1.0, 0.7)


# ================================================================================================ TTS
def voice_spec(speaker: str) -> tuple[str, str]:
    kind, value = story.CAST[speaker]["voice"]
    return str(kind), str(value)


def style_for(speaker: str) -> str:
    return story.STYLE_NARRATOR if speaker == story.NARRATOR else story.STYLE


def tts_key(speaker: str, text: str, ref_sha: str = "") -> str:
    kind, value = voice_spec(speaker)
    raw = json.dumps([TTS_VERSION, speaker, kind, value, style_for(speaker), text, ref_sha], ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def tts_jobs() -> list[tuple[str, str]]:
    """Unique (speaker, text) pairs: every story line plus the enrollment sentences of every on-site character."""
    jobs = [(ln["speaker"], ln["text"]) for ln in story.all_lines()]
    for name, c in story.CAST.items():
        if not c.get("system_only"):
            jobs.extend((name, t) for t in story.ENROLL_TEXT)
    return list(dict.fromkeys(jobs))


def _ref_path(root: Path, speaker: str) -> Path:
    return root / "refs" / f"{speaker}.wav"


def _ref_sha(root: Path, speaker: str) -> str:
    if voice_spec(speaker)[0] != "design":
        return ""
    p = _ref_path(root, speaker)
    return hashlib.sha1(p.read_bytes()).hexdigest()[:16] if p.exists() else ""


def clip_path(root: Path, speaker: str, text: str) -> Path:
    return root / "tts_cache" / f"{tts_key(speaker, text, _ref_sha(root, speaker))}.wav"


def _decode_wav(data: bytes) -> np.ndarray:
    a, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    return audiofx.resample(a, int(sr), SR)


def plausible_duration(text: str, seconds: float) -> bool:
    """Reject clips that are obviously truncated or run away (TTS models occasionally do both)."""
    n = max(1, len(re.sub(r"[\s，。！？、：；,.!?:;“”\"'（）()]", "", text)))
    return 0.07 * n <= seconds <= 0.75 * n + 1.5


class TTS:
    """MiMo speech synthesis through the project's client (rate limiter and credentials are shared with it)."""

    def __init__(self, client: Any, root: Path, timeout_s: float = 240.0):
        self.root = Path(root)
        self.client = client
        raw = getattr(client, "client", None)
        if raw is None:
            raise RuntimeError("the MiMo client exposes no OpenAI-compatible `client` attribute")
        try:
            raw = raw.with_options(timeout=timeout_s)
        except Exception:  # noqa: BLE001 - older SDKs: keep the client's own timeout
            pass
        self.raw = raw
        self.calls = 0
        self.cached = 0
        self._lock = threading.Lock()

    def _create(self, **kwargs: Any) -> bytes:
        limiter = getattr(self.client, "limiter", None)
        last: Exception | None = None
        for attempt in range(4):
            try:
                if limiter is not None:
                    limiter.wait()
                with self._lock:
                    self.calls += 1
                r = self.raw.chat.completions.create(**kwargs)
                audio = getattr(r.choices[0].message, "audio", None)
                data = getattr(audio, "data", None) if audio is not None else None
                if not data:
                    raise RuntimeError("no audio in the response")
                return base64.b64decode(data)
            except Exception as e:  # noqa: BLE001 - retried; the message is scrubbed before logging
                last = e
                log.warning("tts attempt %d failed: %s", attempt + 1, common.scrub(f"{type(e).__name__}: {e}")[:300])
                time.sleep(4 * (attempt + 1))
        raise RuntimeError(common.scrub(f"tts failed: {type(last).__name__}: {last}")[:300])

    def reference(self, speaker: str) -> Path:
        """The designed reference clip of a designed voice (made once, then only read)."""
        kind, value = voice_spec(speaker)
        if kind != "design":
            raise ValueError(f"{speaker} uses a preset voice")
        p = _ref_path(self.root, speaker)
        if p.exists():
            return p
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self._create(model=TTS_DESIGN_MODEL, audio={"format": "wav"},
                            messages=[{"role": "user", "content": value},
                                      {"role": "assistant", "content": story.DESIGN_REFERENCE_TEXT}])
        p.write_bytes(data)
        log.info("designed the voice of %s (%.1f s)", speaker, len(_decode_wav(data)) / SR)
        return p

    def clip(self, speaker: str, text: str) -> np.ndarray:
        kind, value = voice_spec(speaker)
        if kind == "design":
            self.reference(speaker)
        f = clip_path(self.root, speaker, text)
        if f.exists():
            with self._lock:
                self.cached += 1
            return sf.read(str(f), dtype="float32")[0]
        f.parent.mkdir(parents=True, exist_ok=True)
        messages = [{"role": "user", "content": style_for(speaker)}, {"role": "assistant", "content": text}]
        audio: np.ndarray | None = None
        for attempt in range(3):
            if kind == "preset":
                data = self._create(model=TTS_MODEL, messages=messages, audio={"format": "wav", "voice": value})
            else:
                ref = base64.b64encode(_ref_path(self.root, speaker).read_bytes()).decode()
                data = self._create(model=TTS_CLONE_MODEL, messages=messages,
                                    audio={"format": "wav", "voice": f"data:audio/wav;base64,{ref}"})
            audio = _decode_wav(data)
            if plausible_duration(text, len(audio) / SR):
                break
            log.warning("%s: implausible clip (%.1f s for %d chars), attempt %d", speaker, len(audio) / SR, len(text), attempt + 1)
        assert audio is not None
        tmp = f.with_suffix(".tmp.wav")
        sf.write(str(tmp), audio, SR, subtype="PCM_16")
        os.replace(tmp, f)
        return audio


def bare_config(extra_mimo: dict[str, Any] | None = None, models_dir: Path | None = None) -> Any:
    """A Config built from the code's defaults only (no file is read): key from the environment."""
    common.use_project()
    import copy

    from audio_note.config import DEFAULTS, Config

    merged = copy.deepcopy(DEFAULTS)
    merged["mimo"].update(extra_mimo or {})
    merged["mimo"]["api_key"] = os.environ.get("MIMO_API_KEY", "")
    merged["paths"]["models_dir"] = str(models_dir or default_models_dir())
    return Config(merged)


def default_models_dir() -> Path:
    env = os.environ.get("BENCH_MODELS_DIR")
    return Path(env) if env else common.REPO_ROOT / "models"


def synth(root: Path, key_config: str | None = None, workers: int = 4) -> dict[str, Any]:
    """Synthesize every missing clip. Preset voices and cloned voices run in separate small pools (a cloned
    line takes about a minute)."""
    root = Path(root)
    extra = common.load_key(key_config)
    common.use_project()
    from audio_note.llm.client import MimoClient

    client = MimoClient(bare_config(extra), rpm=60)
    tts = TTS(client, root)
    jobs = tts_jobs()
    for name in story.CAST:
        if voice_spec(name)[0] == "design":
            tts.reference(name)
    missing = [(s, t) for s, t in jobs if not clip_path(root, s, t).exists()]
    preset = [j for j in missing if voice_spec(j[0])[0] == "preset"]
    cloned = [j for j in missing if voice_spec(j[0])[0] == "design"]
    log.info("%d clips (%d cached); to synthesize: %d preset, %d cloned", len(jobs), len(jobs) - len(missing), len(preset), len(cloned))
    t0 = time.time()
    failed: list[str] = []

    def one(job: tuple[str, str]) -> None:
        try:
            tts.clip(*job)
        except Exception as e:  # noqa: BLE001 - reported at the end, the rest still gets done
            failed.append(f"{job[0]}: {job[1][:20]} ({common.scrub(str(e))[:120]})")

    with ThreadPoolExecutor(max(1, workers)) as ex_p, ThreadPoolExecutor(max(1, min(workers, 3))) as ex_c:
        futs = [ex_p.submit(one, j) for j in preset] + [ex_c.submit(one, j) for j in cloned]
        for f in futs:
            f.result()
    out = {"clips": len(jobs), "synthesized": len(missing) - len(failed), "failed": failed, "api_calls": tts.calls,
           "seconds": round(time.time() - t0, 1)}
    log.info("synth: %s", {k: v for k, v in out.items() if k != "failed"})
    for f in failed:
        log.error("failed: %s", f)
    return out


def load_clip(root: Path, speaker: str, text: str) -> np.ndarray:
    f = clip_path(Path(root), speaker, text)
    if not f.exists():
        raise FileNotFoundError(f"no cached clip for {speaker}: {text[:20]} - run `build.py synth` first")
    a = sf.read(str(f), dtype="float32")[0]
    return audiofx.fade(audiofx.trim_silence(a))


# ================================================================================================ voice check
def check_voices(root: Path, models_dir: Path | None = None, threads: int = 2) -> dict[str, Any]:
    """Embed every clip with the project's speaker model: per character the similarity of each line to the
    character's centroid, the centroid-to-centroid matrix and (designed voices) the similarity to the reference."""
    root = Path(root)
    cfg = bare_config(models_dir=models_dir)
    cfg["speakers"]["num_threads"] = int(threads)
    from audio_note.speech.speakers import SpeakerEngine

    eng = SpeakerEngine(cfg)
    embs: dict[str, list[tuple[str, np.ndarray, float]]] = {}
    for speaker, text in tts_jobs():
        clip = load_clip(root, speaker, text)
        embs.setdefault(speaker, []).append((text, eng.embed(clip), len(clip) / SR))
    cents = {}
    for name, rows in embs.items():
        c = np.mean([e for _, e, _ in rows], axis=0)
        cents[name] = c / (np.linalg.norm(c) or 1.0)
    report: dict[str, Any] = {"speakers": {}, "centroid_cosine": {}, "outliers": []}
    for name, rows in embs.items():
        sims = [float(e @ cents[name]) for _, e, _ in rows]
        long_sims = [s for s, (_, _, d) in zip(sims, rows) if d >= 2.0]
        info: dict[str, Any] = {"clips": len(rows), "voice": list(voice_spec(name)),
                                "to_centroid_mean": round(float(np.mean(sims)), 3), "to_centroid_min": round(float(np.min(sims)), 3),
                                "to_centroid_min_2s": round(float(np.min(long_sims)), 3) if long_sims else None}
        if voice_spec(name)[0] == "design" and _ref_path(root, name).exists():
            ref = eng.embed(audiofx.trim_silence(_decode_wav(_ref_path(root, name).read_bytes())))
            rs = [float(e @ ref) for _, e, _ in rows]
            info["to_reference_mean"] = round(float(np.mean(rs)), 3)
            info["to_reference_min"] = round(float(np.min(rs)), 3)
        report["speakers"][name] = info
        for (text, e, dur), s in zip(rows, sims):
            best_other = max(((float(e @ c), o) for o, c in cents.items() if o != name), default=(-1.0, ""))
            if dur >= 2.0 and (s < 0.5 or best_other[0] > s):
                report["outliers"].append({"speaker": name, "text": text, "seconds": round(dur, 2), "own": round(s, 3),
                                           "other": best_other[1], "other_cosine": round(best_other[0], 3)})
    names = list(cents)
    for a in names:
        report["centroid_cosine"][a] = {b: round(float(cents[a] @ cents[b]), 3) for b in names}
    common.write_json(root / "voices.json", report)
    return report


# ================================================================================================ render
def default_gap(session_index: int, line_index: int) -> float:
    return GAPS[(session_index * 5 + line_index * 3) % len(GAPS)]


def plan_lines(sess: dict[str, Any], variant: str) -> list[dict[str, Any]]:
    """Per line of a session: the flags that apply in this variant (pure)."""
    out = []
    lines = sess["lines"]
    for k, ln in enumerate(lines):
        sys_, far, overlap = bool(ln["sys"]), bool(ln["far"]), float(ln["overlap"] or 0.0)
        if variant == "clean":
            sys_, far, overlap = False, False, 0.0
        elif variant == "far":
            far, overlap = (not sys_), 0.0
        out.append({**ln, "sys": sys_, "far": far, "overlap": overlap})
    for k, ln in enumerate(out):
        nxt = out[k + 1] if k + 1 < len(out) else None
        ln["condition"] = story.base_condition(ln, nxt)
    return out


def render(root: Path, variant: str = "mixed", near_db: float = NEAR_DB, far_rel_db: float = FAR_REL_DB,
           noise_db: float = NOISE_DB, rt60_s: float = 0.45, drr_db: float = 0.0) -> dict[str, Any]:
    """Assemble session audio for one variant from the cached clips; writes sessions/<variant>/*.wav,
    enroll/*.wav and truth.json. Offline and deterministic."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    root = Path(root)
    out_dir = root / "sessions" / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    enroll_dir = root / "enroll"
    enroll_dir.mkdir(parents=True, exist_ok=True)
    truth: dict[str, Any] = {
        "version": 1, "story": story.STORY_ID, "variant": variant, "sample_rate": SR,
        "render": {"near_db": near_db, "far_rel_db": far_rel_db, "noise_db": noise_db, "rt60_s": rt60_s, "drr_db": drr_db,
                   "system_peak": SYSTEM_PEAK, "channel_block_s": audiofx.BLOCK / SR},
        "speakers": {n: {"enrolled": bool(c.get("enrolled")), "self": bool(c.get("self")),
                         "system_only": bool(c.get("system_only")), "voice": voice_spec(n)[0]} for n, c in story.CAST.items()},
        "terms": story.term_list(), "sessions": [], "lines": [],
    }
    for si, sess in enumerate(story.sessions()):
        plan = plan_lines(sess, variant)
        placed: list[tuple[int, np.ndarray, bool]] = []     # (start sample, audio, is system)
        entries: list[dict[str, Any]] = []
        pos = LEAD_S
        prev_end = 0.0
        prev_t0 = 0.0
        for k, ln in enumerate(plan):
            dry = load_clip(root, ln["speaker"], ln["text"])
            dur = len(dry) / SR
            jitter = (audiofx.stable_unit(story.STORY_ID, sess["id"], k, "level") - 0.5) * 3.0     # +-1.5 dB
            if ln["sys"]:
                wet = audiofx.set_peak(dry, SYSTEM_PEAK * 10.0 ** (jitter / 40.0))
            elif ln["far"]:
                wet = audiofx.far_field(dry, level_db=near_db + far_rel_db + jitter, rt60_s=rt60_s, drr_db=drr_db,
                                        seed=audiofx.stable_seed(story.STORY_ID, sess["id"], k, "ir"))
            else:
                wet = audiofx.set_active_level(dry, near_db + jitter)
            if k == 0:
                t0 = LEAD_S
            elif ln["overlap"] > 0:
                t0 = max(prev_t0 + 0.3, prev_end - min(ln["overlap"], 0.8 * dur))
            else:
                gap = ln["gap"] if ln.get("gap") is not None else default_gap(si, k)
                t0 = max(pos, prev_end) + float(gap)
            t1 = t0 + dur
            placed.append((int(round(t0 * SR)), wet, bool(ln["sys"])))
            entries.append({"i": k, "session": sess["id"], "speaker": ln["speaker"], "text": ln["text"],
                            "t0": round(t0, 3), "t1": round(t1, 3), "condition": ln["condition"],
                            "channel": "system" if ln["sys"] else "mic",
                            "level_db": round(audiofx.active_level_db(wet[: len(dry)]), 1)})
            prev_t0, prev_end = t0, max(prev_end, t1)
            pos = prev_end
        n = int(round((prev_end + TAIL_S) * SR)) + int(rt60_s * 1.5 * SR)
        mic = np.zeros(n, dtype=np.float32)
        sysb = np.zeros(n, dtype=np.float32)
        for start, wet, is_sys in placed:
            audiofx.mix_into(sysb if is_sys else mic, wet, start)
        mic += audiofx.noise_floor(n, noise_db, seed=audiofx.stable_seed(story.STORY_ID, sess["id"], "noise"))
        has_system = any(p[2] for p in placed)
        mix = audiofx.soft_limit(mic + sysb)
        start = datetime.fromisoformat(sess["start"])
        f = out_dir / f"{start.strftime('%Y-%m-%d_%H%M')}_{sess['id']}.wav"
        sf.write(str(f), mix, SR, subtype="PCM_16")
        ch_file = None
        if has_system:
            ch_file = f.with_suffix(".channels.npy")
            np.save(str(ch_file), audiofx.channel_pairs(mic, sysb))
        s_entry = {"id": sess["id"], "day": sess["day"], "start": sess["start"], "title": sess["title"], "kind": sess["kind"],
                   "file": f.name, "channels_file": ch_file.name if ch_file else None, "duration_s": round(n / SR, 3),
                   "has_system": has_system, "lines": entries}
        truth["sessions"].append(s_entry)
        truth["lines"].extend(entries)
        log.info("%s %s: %d lines, %.0f s%s", variant, f.name, len(entries), n / SR, " (system audio)" if has_system else "")
    enroll = {}
    for name, c in story.CAST.items():
        if c.get("system_only"):
            continue
        files = []
        for k, text in enumerate(story.ENROLL_TEXT, start=1):
            f = enroll_dir / f"{name}_{k}.wav"
            sf.write(str(f), audiofx.set_active_level(load_clip(root, name, text), near_db), SR, subtype="PCM_16")
            files.append(f.name)
        enroll[name] = files
    truth["enroll"] = enroll
    truth["memory"] = {"items": story.memory_items(), "conflicts": story.CONFLICTS, "media_distractors": story.MEDIA_DISTRACTORS}
    truth["lexicon"] = story.LEXICON
    truth["figures"] = story.FIGURES
    truth = json.loads(json.dumps(truth, ensure_ascii=False))      # plain JSON types (tuples -> lists), as stored
    common.write_json(out_dir / "truth.json", truth)
    return truth


# ================================================================================================ import
class StubASR:
    """Stands in for the ASR when only the local stages matter (speakers, cost): no network, no key."""

    text = "测试"

    def transcribe(self, audio: np.ndarray, **_kw: Any) -> Any:
        if len(audio) < int(0.3 * SR):
            return SimpleNamespace(text="", seconds=0.0, parts=0, flags=[])
        return SimpleNamespace(text=self.text, seconds=0.0, parts=1, flags=[])


class CachingClient:
    """Transparent proxy around the project's MiMo client that caches ASR answers on disk, keyed by the exact
    request (audio bytes, model, language and any further arguments such as a hot-word prompt). Everything
    else (chat, json, usage, attributes set by the application) goes straight to the wrapped client."""

    def __init__(self, inner: Any, cache_dir: Path, stats: dict[str, float] | None = None, max_live_s: float | None = None):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_dir", Path(cache_dir))
        object.__setattr__(self, "_stats", stats if stats is not None else
                           {"asr_requests": 0, "asr_cache_hits": 0, "asr_live_audio_s": 0.0, "asr_billed_s": 0.0})
        object.__setattr__(self, "_max_live_s", max_live_s)
        object.__setattr__(self, "_slock", threading.Lock())
        self._dir.mkdir(parents=True, exist_ok=True)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_inner"), name, value)

    @property
    def bench_stats(self) -> dict[str, float]:
        return dict(self._stats)

    def scoped(self, purpose: str) -> "CachingClient":
        inner = self._inner.scoped(purpose)
        return CachingClient(inner, self._dir, self._stats, self._max_live_s)

    def transcribe_bytes(self, wav_or_mp3: bytes, mime: str = "audio/wav", language: str = "auto",
                         model: str | None = None, **kw: Any) -> tuple[str, float]:
        inner = self._inner
        used_model = model or getattr(getattr(getattr(inner, "cfg", None), "mimo", None), "asr_model", None) or ""
        extra = {k: v for k, v in sorted(kw.items()) if v is not None}
        key = hashlib.sha1(json.dumps([hashlib.sha1(wav_or_mp3).hexdigest(), mime, language, used_model, extra],
                                      ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
        f = self._dir / f"{key}.json"
        audio_s = max(0.0, (len(wav_or_mp3) - 44) / (2.0 * SR)) if mime == "audio/wav" else 0.0
        with self._slock:
            self._stats["asr_requests"] += 1
        if f.exists():
            try:
                hit = json.loads(f.read_text(encoding="utf-8"))
                with self._slock:
                    self._stats["asr_cache_hits"] += 1
                return str(hit["text"]), float(hit.get("seconds") or 0.0)
            except (OSError, ValueError, KeyError):
                pass
        with self._slock:
            if self._max_live_s is not None and self._stats["asr_live_audio_s"] + audio_s > self._max_live_s:
                raise RuntimeError(f"bench ASR budget exhausted ({self._max_live_s:.0f} s of live audio)")
            self._stats["asr_live_audio_s"] += audio_s
        text, secs = inner.transcribe_bytes(wav_or_mp3, mime, language, model, **kw) if (model or kw) else \
            inner.transcribe_bytes(wav_or_mp3, mime, language)
        with self._slock:
            self._stats["asr_billed_s"] += float(secs or 0.0)
        tmp = f.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"text": text, "seconds": secs, "model": used_model, "language": language,
                                   "audio_s": round(audio_s, 2), "extra": extra}, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, f)
        return text, secs


def _parse_overrides(pairs: list[str] | None) -> dict[str, Any]:
    """["speakers.identify_threshold=0.5", "correction.mode=session"] -> nested dict (values parsed as YAML)."""
    import yaml

    out: dict[str, Any] = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--set expects section.key=value, got {item!r}")
        path, raw = item.split("=", 1)
        node = out
        keys = [k for k in path.strip().split(".") if k]
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = yaml.safe_load(raw)
    return out


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def write_profile_config(pdir: Path, *, mimo: dict[str, Any] | None = None, models_dir: Path | None = None,
                         threads: int | None = None, correction: bool = True, overrides: dict[str, Any] | None = None) -> Path:
    """config.yaml of a bench profile: absolute paths, no API key, nothing that pushes or serves."""
    import yaml

    pdir = Path(pdir)
    conf: dict[str, Any] = {
        "paths": {"data_dir": str(pdir / "data"), "models_dir": str(models_dir or default_models_dir())},
        "storage": {"cleanup_enabled": False, "keep_audio_days": 0},
        "correction": {"enabled": bool(correction), "glossary_file": str(pdir / "glossary.txt")},
        "pipeline": {"min_segment_age_s": 0},
        "analysis": {"push_daily": False},
        "push": {"channels": []},
        "ui": {"enabled": False},
        "log": {"file": str(pdir / "bench.log")},
    }
    if mimo:
        conf["mimo"] = dict(mimo)
    if threads:
        conf["speakers"] = {"num_threads": int(threads)}
    conf = _deep_merge(conf, overrides or {})
    if "api_key" in (conf.get("mimo") or {}):
        raise ValueError("a bench profile must not store an API key")
    pdir.mkdir(parents=True, exist_ok=True)
    p = pdir / "config.yaml"
    p.write_text("# audio-note bench profile (generated by scripts/bench/build.py). The API key comes from MIMO_API_KEY.\n"
                 + yaml.safe_dump(conf, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
    return p


def _safe_rmtree(pdir: Path, root: Path) -> None:
    pdir, parent = pdir.resolve(), (Path(root) / "profiles").resolve()
    if pdir.parent != parent or not re.fullmatch(r"[\w\-]{1,48}", pdir.name):
        raise RuntimeError(f"refusing to delete {pdir}: not a bench profile under {parent}")
    shutil.rmtree(pdir)


def import_sessions(cfg: Any, db: Any, truth: dict[str, Any], audio_dir: Path, only: set[str] | None = None) -> dict[str, list[int]]:
    """VAD-gate every session file into segments exactly like audio.importer.import_file, and attach the
    (mic, system) channel levels of sessions that have system audio. Returns {session id: [segment ids]}."""
    from audio_note.audio.recorder import gate_config_from
    from audio_note.audio.storage import finalize_segment
    from audio_note.audio.vad import SileroVAD, segment_audio

    vad = SileroVAD(cfg.model_path("vad_model"), threads=2)
    out: dict[str, list[int]] = {}
    block = audiofx.BLOCK
    for s in truth["sessions"]:
        if only and s["id"] not in only:
            continue
        path = Path(audio_dir) / s["file"]
        audio = sf.read(str(path), dtype="float32")[0]
        pairs = np.load(str(Path(audio_dir) / s["channels_file"])) if s.get("channels_file") else None
        start_time = datetime.fromisoformat(s["start"])
        ids: list[int] = []
        for seg in segment_audio(audio, vad, gate_config_from(cfg)):
            started = start_time + timedelta(seconds=seg.start_sample / SR)
            kw: dict[str, Any] = {"source": "import", "origin": str(path)}
            if pairs is not None:
                b0 = int(seg.start_sample // block)
                b1 = int(-(-(seg.start_sample + len(seg.audio)) // block))
                kw.update(channel_rms=pairs[b0:b1], channel_block_s=block / SR)
            sid = common.call_supported(finalize_segment, cfg, db, seg.audio, started, seg.speech_s, **kw)
            if sid is not None:
                ids.append(int(sid))
        out[s["id"]] = ids
    return out


def import_profile(root: Path, name: str, *, variant: str = "mixed", asr: str = "mimo", key_config: str | None = None,
                   enroll: list[str] | None = None, force: bool = False, threads: int | None = None,
                   glossary: str = "none", overrides: dict[str, Any] | None = None, models_dir: Path | None = None,
                   sessions: set[str] | None = None, chain: bool = False, max_asr_minutes: float = 30.0,
                   measure: bool = True) -> dict[str, Any]:
    """Create profiles/<name> under `root`, enroll the cast, import the rendered sessions and run the
    processing pipeline of the code under test. Returns (and stores as bench_import.json) what happened."""
    root = Path(root)
    audio_dir = root / "sessions" / variant
    truth = common.read_json(audio_dir / "truth.json")
    pdir = common.profile_dir(root, name)
    if pdir.exists():
        if not force:
            raise SystemExit(f"{pdir} exists; pass --force to rebuild it")
        _safe_rmtree(pdir, root)
    src = common.use_project()
    mimo_extra: dict[str, Any] = {}
    if asr == "mimo":
        mimo_extra = common.load_key(key_config)
    over = dict(overrides or {})
    if asr != "mimo":
        over = _deep_merge({"correction": {"mode": "off"}, "memory": {"retrieval": "fts"}}, over)
    cfg_path = write_profile_config(pdir, mimo=mimo_extra, models_dir=models_dir, threads=threads,
                                    correction=(asr == "mimo"), overrides=over)
    gl = pdir / "glossary.txt"
    terms = [t["term"] for t in story.LEXICON] if glossary == "gold" else []
    gl.write_text("# bench glossary\n" + "".join(t + "\n" for t in terms), encoding="utf-8", newline="\n")
    os.environ["AUDIO_NOTE_CONFIG"] = str(cfg_path)          # explicit: never the default config resolution
    os.environ.pop("AUDIO_NOTE_PROFILE", None)

    from audio_note.config import load_config
    from audio_note.db import DB
    from audio_note.speech.pipeline import Processor
    from audio_note.speech.speakers import SpeakerEngine

    cfg = load_config(cfg_path)
    cfg.ensure_dirs()
    db = DB(cfg.db_path)
    timer = common.StageTimer()
    t_all = time.time()

    engine = SpeakerEngine(cfg)
    enroll_names = list(enroll) if enroll is not None else [n for n, c in truth["speakers"].items() if c["enrolled"]]
    for n in enroll_names:
        files = [root / "enroll" / f for f in truth["enroll"][n]]
        sid = engine.enroll(db, n, files)
        if truth["speakers"][n].get("self") and callable(getattr(db, "set_self_speaker", None)):
            db.set_self_speaker(sid)
    wrapped = common.time_engine(engine, timer) if measure else []

    with timer.stage("vad_gate"):
        seg_ids = import_sessions(cfg, db, truth, audio_dir, sessions)
    n_segments = sum(len(v) for v in seg_ids.values())
    audio_s = float(db.one("SELECT COALESCE(SUM(duration_s), 0) AS s FROM segments")["s"])

    client: Any
    stats: dict[str, float] = {}
    kwargs: dict[str, Any] = {}
    if asr == "mimo":
        from audio_note.llm.client import MimoClient

        real = MimoClient(cfg, rpm=int(cfg.mimo.rpm))
        client = CachingClient(real, root / "asr_cache", stats or None, max_live_s=max_asr_minutes * 60.0)
        stats = client._stats
    else:
        client = object()
        kwargs["asr"] = StubASR()
    proc = Processor(cfg, db, client=client, engine=engine, **kwargs)
    w0, c0 = time.perf_counter(), time.process_time()
    result = proc.process_pending()
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0

    chain_result = None
    if chain:
        try:
            from audio_note import chain as chain_mod

            chain_result = chain_mod.run_chain(cfg, db, (lambda: client) if asr == "mimo" else (lambda: None))
        except ImportError:
            chain_result = {"skipped": "audio_note.chain is not available in this code"}
        except Exception as e:  # noqa: BLE001 - the import itself succeeded; report and carry on
            chain_result = {"error": common.scrub(f"{type(e).__name__}: {e}")[:300]}

    # the profile carries its own truth (absolute audio paths) so the scorers need nothing else
    truth_out = dict(truth)
    if sessions:                                   # a partial import is scored on what was imported
        truth_out["sessions"] = [s for s in truth["sessions"] if s["id"] in sessions]
        truth_out["lines"] = [ln for ln in truth["lines"] if ln["session"] in sessions]
    truth_out["audio_dir"] = str(audio_dir)
    truth_out["enrolled"] = enroll_names
    truth_out["unknown_label"] = str(cfg.speakers.unknown_label)
    common.write_json(pdir / "truth.json", truth_out)

    status = {r["status"]: r["n"] for r in db.query("SELECT status, COUNT(*) AS n FROM segments GROUP BY status")}
    usage = dict(getattr(client, "usage", {}) or {}) if asr == "mimo" else {}
    report = {
        "profile": str(pdir), "name": name, "variant": variant, "asr": asr, "glossary": glossary, "code": common.code_info(),
        "src": str(src), "enrolled": enroll_names, "threads": int(cfg.speakers.num_threads or 0),
        "sessions": len(seg_ids), "segments": n_segments, "segment_status": status, "audio_s": round(audio_s, 1),
        "utterances": int(db.one("SELECT COUNT(*) AS n FROM utterances")["n"]),
        "process": result, "process_wall_s": round(wall, 1), "process_cpu_s": round(cpu, 1),
        "stages": timer.snapshot(), "timed": wrapped, "asr_stats": {k: round(float(v), 1) for k, v in stats.items()},
        "client_usage": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in usage.items()},
        "chain": chain_result, "total_wall_s": round(time.time() - t_all, 1), "overrides": over,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    common.write_json(pdir / "bench_import.json", report)
    try:
        db.close_all()
    except Exception:  # noqa: BLE001 - older DB classes without close_all
        pass
    return report


# ================================================================================================ CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["synth", "voices", "render", "import"])
    ap.add_argument("--root", type=Path, default=None, help="bench root (default: $BENCH_ROOT or <repo>/tmp_bench)")
    ap.add_argument("--key-config", default=None, help="YAML file whose mimo.api_key / base_url to use (read in memory)")
    ap.add_argument("--variant", default="mixed", choices=VARIANTS)
    ap.add_argument("--workers", type=int, default=4, help="synth: parallel TTS requests")
    ap.add_argument("--name", default=None, help="import: profile name under <root>/profiles")
    ap.add_argument("--asr", default="mimo", choices=["mimo", "stub"], help="import: real ASR + correction, or a stub (no API)")
    ap.add_argument("--enroll", default=None, help="import: comma-separated names to enroll (default: the story's enrolled cast)")
    ap.add_argument("--glossary", default="none", choices=["none", "gold"], help="import: write the gold terms into glossary.txt")
    ap.add_argument("--threads", type=int, default=None, help="import: speakers.num_threads")
    ap.add_argument("--set", action="append", default=[], metavar="section.key=value", help="import: config override (repeatable)")
    ap.add_argument("--sessions", default=None, help="import: comma-separated session ids (default: all)")
    ap.add_argument("--chain", action="store_true", help="import: run the post-processing chain afterwards (1.0 code)")
    ap.add_argument("--max-asr-minutes", type=float, default=30.0, help="import: stop when live ASR audio exceeds this")
    ap.add_argument("--models-dir", type=Path, default=None, help="model files (default: $BENCH_MODELS_DIR or <repo>/models)")
    ap.add_argument("--force", action="store_true", help="import: delete an existing profile of that name first")
    ap.add_argument("--near-db", type=float, default=NEAR_DB)
    ap.add_argument("--far-db", type=float, default=FAR_REL_DB, help="render: far-field level relative to near-field")
    ap.add_argument("--noise-db", type=float, default=NOISE_DB)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for n in ("httpx", "httpx2", "httpcore", "httpcore2", "openai"):
        logging.getLogger(n).setLevel(logging.WARNING)
    root = args.root or common.default_root()
    root.mkdir(parents=True, exist_ok=True)
    if args.command == "synth":
        out = synth(root, args.key_config, args.workers)
        return 1 if out["failed"] else 0
    if args.command == "voices":
        rep = check_voices(root, args.models_dir)
        print(json.dumps(rep, ensure_ascii=False, indent=1))
        return 0
    if args.command == "render":
        truth = render(root, args.variant, args.near_db, args.far_db, args.noise_db)
        total = sum(s["duration_s"] for s in truth["sessions"])
        print(f"{args.variant}: {len(truth['sessions'])} sessions, {len(truth['lines'])} lines, {total / 60:.1f} min -> "
              f"{root / 'sessions' / args.variant}")
        return 0
    if not args.name:
        ap.error("import needs --name")
    rep = import_profile(root, args.name, variant=args.variant, asr=args.asr, key_config=args.key_config,
                         enroll=[x for x in args.enroll.split(",") if x] if args.enroll is not None else None,
                         force=args.force, threads=args.threads, glossary=args.glossary,
                         overrides=_parse_overrides(args.set), models_dir=args.models_dir,
                         sessions=set(args.sessions.split(",")) if args.sessions else None, chain=args.chain,
                         max_asr_minutes=args.max_asr_minutes)
    print(json.dumps({k: rep[k] for k in ("profile", "code", "segments", "segment_status", "utterances", "process",
                                           "process_wall_s", "process_cpu_s", "stages", "asr_stats", "client_usage")},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
