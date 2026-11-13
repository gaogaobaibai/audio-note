"""Speaker commands: `speakers enroll|record|list|label|voices|scan|set-self|relation|relations|relabel|
calibrate|audit|reject|negatives|rescore|scoring`."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from rich.table import Table

from ..config import Config
from ..db import DB
from ..speech.calibrate import calibrate_threshold
from ._late import Late, handler
from .helpers import (CLIError, _db, _expand_files, _force_kwargs, _lazy, _lazy_module, _meter, _parse_datetime, _parse_day,
                      _reindex_after_enroll, _speaker_by_name, add_json_flag, emit_json, resolve_label_target,
                      utterance_at, wants_json)

log = logging.getLogger("audio_note.cli")
console = Late("console")
_client = Late("_client")
_optional_engine = Late("_optional_engine")


def cmd_speakers_enroll(cfg: Config, args: argparse.Namespace) -> int:
    from ..speech.speakers import SpeakerEngine

    files = _expand_files(args.files)
    db = _db(cfg)
    engine = SpeakerEngine(cfg)
    before = db.one("SELECT COUNT(*) AS n FROM speaker_embeddings e JOIN speakers s ON s.id=e.speaker_id WHERE s.name=?", (args.name,))
    n0 = int(before["n"]) if before else 0
    sid = engine.enroll(db, args.name, files, notes=args.notes, **_force_kwargs(args))
    n = db.one("SELECT COUNT(*) AS n FROM speaker_embeddings WHERE speaker_id=?", (sid,))
    n1 = int(n["n"]) if n else 0
    console.print(f"已注册 {args.name} (id {sid})，共 {n1} 个样本（副本保存在 {cfg.speakers_dir}）。")
    skipped = len(files) - (n1 - n0)
    if skipped > 0:
        console.print(f"[yellow]{skipped} 个文件没有加入：语音太短，或和 {args.name} 已有的样本不像（确定是同一个人可加 --force）。[/yellow]")
    _reindex_after_enroll(cfg, db)
    _rescore_after(cfg, db)
    done = db.one("SELECT COUNT(*) AS n FROM segments WHERE status='done'")
    if done and done["n"]:
        console.print("已转写的记录可用 audio-note speakers relabel [--day YYYY-MM-DD] 按新声纹重新识别（不重复转写）。")
    return 0


def cmd_speakers_record(cfg: Config, args: argparse.Namespace) -> int:
    """Record an enrollment sample from the microphone, save it under data/speakers/<name>/, enroll it."""
    import time

    import numpy as np
    import soundfile as sf

    from ..audio.capture import MicCapture
    from ..audio.vad import SAMPLE_RATE
    from ..speech.speakers import SpeakerEngine

    seconds = max(5, int(args.seconds))
    out_dir = cfg.speakers_dir / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    console.print(f"将录制 {seconds} 秒作为 [bold]{args.name}[/bold] 的声纹样本，请用平时说话的音量自然地说话（念一段文字即可）。")
    for i in (3, 2, 1):
        console.print(f"  {i} ...")
        time.sleep(1)
    console.print("[green]开始录音[/green]")
    blocks: list[np.ndarray] = []
    got = 0
    with MicCapture(cfg.audio.device, int(cfg.audio.block_ms), float(cfg.audio.gain)) as mic:
        t0 = time.time()
        last_tick = 0
        while got < seconds * SAMPLE_RATE:
            b = mic.read(timeout=1.0)
            if b is None:
                if time.time() - t0 > seconds + 5:
                    console.print("[red]麦克风没有数据，检查设备（audio-note devices）[/red]")
                    return 2
                continue
            blocks.append(b)
            got += len(b)
            tick = int(got / SAMPLE_RATE)
            if tick != last_tick and tick % 5 == 0:
                last_tick = tick
                console.print(f"  已录 {tick}s")
    audio = np.concatenate(blocks)[: seconds * SAMPLE_RATE]
    sf.write(str(out), audio, SAMPLE_RATE, subtype="PCM_16")
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    console.print(f"样本已保存：{out}（峰值 {peak:.2f}）")
    if peak < 0.02:
        console.print("[yellow]音量很低，可能没录到声音；请检查麦克风或调大 audio.gain[/yellow]")
    db = _db(cfg)
    engine = SpeakerEngine(cfg)
    emb, speech_s = engine.embed_speech_only(audio)
    if speech_s < 3.0:
        console.print(f"[red]只检测到 {speech_s:.1f} 秒语音，样本不可用；请重录并持续说话[/red]")
        return 2
    sid = db.upsert_speaker(args.name, args.notes)
    db.add_speaker_embedding(sid, emb, str(out), speech_s, origin="mic")
    n = db.one("SELECT COUNT(*) AS n FROM speaker_embeddings WHERE speaker_id=?", (sid,))
    console.print(f"已注册 {args.name} (id {sid})，语音 {speech_s:.1f}s，共 {n['n'] if n else 0} 个样本。建议在不同场合再录 1–2 次以提高稳定性。")
    _reindex_after_enroll(cfg, db)
    _rescore_after(cfg, db)
    return 0


SAMPLE_ORIGIN_LABELS = {"mic": "麦克风", "file": "文件", "label": "标记", "voice": "声音组", "cluster": "簇"}


def cmd_speakers_audit(cfg: Config, args: argparse.Namespace) -> int:
    """Table of every enrolled sample with its consistency to the person's other samples; --fix deletes
    the outliers (samples that probably hold somebody else's voice)."""
    from ..speech.speakers import audit_samples, sample_consistency_min

    db = _db(cfg)
    min_cos = sample_consistency_min(cfg)
    speakers = db.query("SELECT id, name FROM speakers ORDER BY is_self DESC, name")
    t = Table(title=f"声纹样本一致性（阈值 {min_cos:.2f}）")
    for col in ("人", "样本", "来源", "时长", "一致性", "单独认出", "离群"):
        t.add_column(col)
    try:
        id_thr: float | None = float(cfg.speakers.identify_threshold)
    except (TypeError, ValueError, AttributeError):
        id_thr = None
    outliers: list[tuple[int, str, int]] = []
    ambiguous: list[str] = []
    total = 0
    for sp in speakers:
        rows = audit_samples(db, int(sp["id"]), min_cos, id_thr)
        if any(a["ambiguous"] for a in rows):
            ambiguous.append(sp["name"])
        for a in rows:
            total += 1
            cons = "-" if a["consistency"] is None else f"{a['consistency']:.2f}"
            flag = "[red]是[/red]" if a["outlier"] else ("[yellow]歧义[/yellow]" if a["ambiguous"] else "")
            if a["outlier"]:
                outliers.append((int(sp["id"]), sp["name"], int(a["id"])))
            src = Path(str(a["source_path"]).split("#")[0]).name if a["source_path"] else "（扫描时保存的声纹）"
            dur = f"{float(a['duration_s']):.1f}s" if a["duration_s"] is not None else "-"
            infl = int(a.get("influence") or 0)
            t.add_row(sp["name"], f"#{a['id']} {src}", SAMPLE_ORIGIN_LABELS.get(a.get("origin") or "", "-"), dur, cons,
                      f"{infl} 句" if infl else "-", flag)
    console.print(t)
    if not total:
        console.print("（没有样本）")
        return 0
    if ambiguous:
        console.print(f"[yellow]{'、'.join(ambiguous)} 的样本分成两组互不相像，没有麦克风 / 文件样本可以判断哪组才是本人："
                      "请在说话人页试听后删掉不是 TA 的（或用 speakers record 录一段作为基准）。[/yellow]")
    if not outliers:
        if not ambiguous:
            console.print("所有样本都和同一人的其他样本一致。")
        return 0
    names = sorted({n for _, n, _ in outliers})
    console.print(f"[yellow]{len(outliers)} 个样本和其他样本不像（可能混了别人的声音）：{'、'.join(names)}[/yellow]")
    if not getattr(args, "fix", False):
        console.print("删除它们：audio-note speakers audit --fix")
        return 0
    ids = [e for _, _, e in outliers]
    cur = db.execute(f"DELETE FROM speaker_embeddings WHERE id IN ({','.join('?' * len(ids))})", ids)
    console.print(f"已删除 {int(cur.rowcount or 0)} 个离群样本。")
    _rescore_after(cfg, db)
    for sp in speakers:
        left = db.one("SELECT COUNT(*) AS n FROM speaker_embeddings WHERE speaker_id=?", (int(sp["id"]),))
        if sp["name"] in names and left and int(left["n"]) == 0:
            console.print(f"[yellow]{sp['name']} 已没有样本：在说话人页重新标记一段，或 speakers enroll 补充。[/yellow]")
    return 0


def _relation_text(relation: str | None, source: str | None) -> str:
    """'同事' / '朋友（自动推断）' / '' for the speaker tables."""
    if not relation:
        return ""
    return f"{relation}（自动推断）" if source == "auto" else str(relation)


def cmd_speakers_list(cfg: Config, args: argparse.Namespace) -> int:
    db = _db(cfg)
    rows = db.query(
        "SELECT s.id, s.name, s.notes, s.created_at, s.is_self, s.relation, s.relation_source,"
        " COUNT(e.id) AS n, COALESCE(SUM(e.duration_s),0) AS dur"
        " FROM speakers s LEFT JOIN speaker_embeddings e ON e.speaker_id=s.id GROUP BY s.id ORDER BY s.is_self DESC, s.name"
    )
    if wants_json(args):
        emit_json({"speakers": [
            {"id": int(r["id"]), "name": r["name"], "is_self": bool(r["is_self"]), "relation": r["relation"],
             "relation_source": r["relation_source"], "samples": int(r["n"]), "speech_s": float(r["dur"]),
             "created_at": r["created_at"], "notes": r["notes"]} for r in rows]})
        return 0
    t = Table(title="已注册说话人")
    for col in ("id", "姓名", "本人", "关系", "样本数", "语音时长", "注册时间", "备注"):
        t.add_column(col)
    for r in rows:
        t.add_row(str(r["id"]), r["name"], "我" if r["is_self"] else "", _relation_text(r["relation"], r["relation_source"]),
                  str(r["n"]), f"{r['dur']:.0f}s", r["created_at"], r["notes"] or "")
    console.print(t)
    if not rows:
        console.print("（空）")
    elif not any(r["is_self"] for r in rows):
        console.print("还没有标出哪位是本人：audio-note speakers set-self <姓名>（日报 / 问答会站在本人视角）")
    return 0


def cmd_speakers_set_self(cfg: Config, args: argparse.Namespace) -> int:
    """Mark an enrolled speaker as the user ("我"): digests, chat and pushes take that person's point of view."""
    db = _db(cfg)
    row = _speaker_by_name(db, args.name)
    prev = db.self_speaker()
    db.set_self_speaker(int(row["id"]))
    if prev is not None and int(prev["id"]) != int(row["id"]):
        console.print(f"本人已从 {prev['name']} 改为 [bold]{row['name']}[/bold] (id {row['id']})。")
    else:
        console.print(f"已把 [bold]{row['name']}[/bold] (id {row['id']}) 设为本人。")
    return 0


def cmd_speakers_relation(cfg: Config, args: argparse.Namespace) -> int:
    """Store (or clear with '-') a speaker's relation to the user; `--note` keeps a remark next to it."""
    db = _db(cfg)
    row = _speaker_by_name(db, args.name)
    rel = str(args.relation or "").strip()
    if row["is_self"]:
        raise CLIError(f"{row['name']} 是本人，不需要标注关系")
    if rel in ("", "-"):
        db.set_relation(int(row["id"]), None, None)
        console.print(f"已清除 {row['name']} 的关系标注。")
        return 0
    db.set_relation(int(row["id"]), rel, "user", (args.note or "").strip() or None)
    console.print(f"已标注 {row['name']}：{rel}" + (f"（{args.note}）" if args.note else "") + "。")
    return 0


def cmd_speakers_relations(cfg: Config, args: argparse.Namespace) -> int:
    """Ask the LLM how each enrolled speaker relates to the user (hand-labelled ones are never touched)."""
    relations = _lazy_module("analysis.relations")
    db = _db(cfg)
    ids = None
    if args.speaker:
        ids = [int(_speaker_by_name(db, args.speaker)["id"])]
    client = _meter(_client(cfg), db)
    results = list(relations.infer_relations(cfg, db, client, speaker_ids=ids, force=bool(args.force)) or [])
    t = Table(title="关系推断")
    for col in ("id", "姓名", "关系", "置信度", "结果", "理由"):
        t.add_column(col)
    for r in results:
        conf = r.get("confidence")
        if r.get("skipped"):
            outcome = f"跳过：{r['skipped']}"
        elif r.get("applied"):
            outcome = "[green]已写入[/green]"
        else:
            outcome = "未写入（不确定或置信度不够）"
        t.add_row(str(r.get("id", "")), str(r.get("name", "")), str(r.get("relation") or ""),
                  f"{float(conf):.2f}" if isinstance(conf, (int, float)) else "", outcome, str(r.get("reason") or ""))
    console.print(t)
    applied = sum(1 for r in results if r.get("applied"))
    if not results:
        console.print("没有需要推断的说话人（本人和已手动标注的不推断；语句太少的也跳过）。")
    else:
        console.print(f"共 {len(results)} 人，写入 {applied} 个关系。手动标注：audio-note speakers relation <姓名> <关系>")
    return 0


def _scan_voices(cfg: Config, db: DB, voices: Any, engine_factory: Callable[[], Any], quiet: bool = False) -> dict[str, int]:
    """backfill_cluster_embeddings for the segments that still lack cluster embeddings (loads the engine only when needed).
    `quiet` drops the progress lines (the --json listing reports the numbers in its document instead)."""
    missing = voices.missing_segments(db)
    if not missing:
        return {"segments": 0, "clusters": 0, "skipped": 0}
    if not quiet:
        console.print(f"正在扫描 {len(missing)} 段录音的声音…")
    stats = dict(voices.backfill_cluster_embeddings(db, cfg, engine_factory()))
    if not quiet:
        console.print(f"扫描完成：{stats['segments']} 段 / {stats['clusters']} 个声音簇，{stats['skipped']} 段跳过。")
    return stats


KEY_SHOWN = 6   # characters of a voice group's stable key shown in the table (the full key is accepted too)


def _print_voice_groups(res: dict[str, Any]) -> None:
    groups = res.get("groups") or []
    if not groups:
        console.print("没有待确认的声音（所有语句都已识别，或还没有已转写的记录）。")
        return
    t = Table(title=f"未识别的声音（{len(groups)} 组，{res.get('unknown_utterances', 0)} 句未识别）")
    for col in ("#", "编号", "句数", "时长", "日期", "可能是", "示例"):
        t.add_column(col)
    for i, g in enumerate(groups, start=1):
        days = g.get("days") or []
        span = days[0] if len(days) <= 1 else f"{days[0]} ~ {days[-1]}（{len(days)} 天）"
        closest = g.get("closest") or {}
        maybe = f"{closest['name']}（{closest['score']:.2f}）" if closest and float(closest.get("score") or 0) >= 0.3 else ""
        samples = "\n".join(f"{s.get('time', '')} {s.get('text', '')}".strip() for s in (g.get("samples") or [])[:3])
        t.add_row(str(i), str(g.get("key") or "")[:KEY_SHOWN], str(g.get("utterances", 0)), f"{float(g.get('speech_s', 0)):.0f}s",
                  span or "", maybe, samples)
    console.print(t)
    console.print("命名：audio-note speakers voices --assign <#> <姓名> [--self] [--relation 同事]（<#> 也可以写编号，编号在重新扫描后不变）")


def _voice_groups_json(res: dict[str, Any], scan: dict[str, int]) -> dict[str, Any]:
    """The `speakers voices --json` document: the grouping exactly as speech.voices returns it, each group
    with the 1-based `index` that `--assign` accepts, plus what the automatic scan did."""
    out = {k: v for k, v in res.items() if k != "groups"}
    out["scan"] = dict(scan)
    out["groups"] = [dict(g, index=i) for i, g in enumerate(res.get("groups") or [], start=1)]
    return out


def _resolve_voice_group(groups: list[dict[str, Any]], ref: Any) -> tuple[int, dict[str, Any]]:
    """`--assign` takes the 1-based index of the listing or the group's stable key (full or a prefix of at
    least a few characters). Returns (index, group)."""
    ref = str(ref or "").strip().lower()
    if not ref:
        raise CLIError("--assign 需要给出序号或编号：speakers voices --assign <#或编号> <姓名>")
    if ref.isdigit() and 1 <= int(ref) <= len(groups):
        return int(ref), groups[int(ref) - 1]
    hits = [(i, g) for i, g in enumerate(groups, start=1) if len(ref) >= 3 and str(g.get("key") or "").lower().startswith(ref)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise CLIError(f"编号 {ref} 不唯一（匹配 {len(hits)} 组），请多写几位")
    if ref.isdigit():
        raise CLIError(f"没有第 {int(ref)} 组（共 {len(groups)} 组，先运行 audio-note speakers voices 查看）")
    raise CLIError(f"没有编号为 {ref} 的声音组（共 {len(groups)} 组，先运行 audio-note speakers voices 查看）")


def _voice_engine_factory(cfg: Config) -> Callable[[], Any]:
    """Lazy SpeakerEngine for `voices` / `scan`: only built when a scan or a naming needs it; a missing
    model file or a failing load is a CLIError (Chinese message) instead of a traceback."""
    def engine() -> Any:
        if not cfg.model_path("embedding_model").exists():
            raise CLIError(f"模型文件缺失，无法提取声纹（运行 scripts/download_models.ps1）：{cfg.model_path('embedding_model')}")
        from ..speech.speakers import SpeakerEngine

        try:
            return SpeakerEngine(cfg)
        except CLIError:
            raise
        except Exception as e:  # noqa: BLE001 - onnxruntime / model file problems
            raise CLIError(f"声纹模型加载失败：{e}（检查 models/ 目录，或重新运行 scripts/download_models.ps1）") from e

    return engine


def cmd_speakers_voices(cfg: Config, args: argparse.Namespace) -> int:
    """List the unidentified voices grouped across recordings; `--assign <#|key> <name>` names one group."""
    voices = _lazy_module("speech.voices")
    as_json = wants_json(args)
    if as_json and args.assign is not None:
        raise CLIError("--json 只用于列出声音，不能和 --assign 一起用")
    db = _db(cfg)
    engine = _voice_engine_factory(cfg)
    scan = _scan_voices(cfg, db, voices, engine, quiet=as_json)
    res = voices.group_unknown_voices(db, cfg)
    if as_json:
        emit_json(_voice_groups_json(res, scan))
        return 0
    if args.assign is None:
        _print_voice_groups(res)
        return 0
    if not args.name:
        raise CLIError("--assign 需要给出姓名：speakers voices --assign <#或编号> <姓名>")
    groups = res.get("groups") or []
    if scan["clusters"]:
        _print_voice_groups(res)
        console.print("[yellow]刚扫描到新的录音，分组可能变了：请按上面的新序号或编号重新运行 --assign。[/yellow]")
        return 1
    idx, g = _resolve_voice_group(groups, args.assign)
    console.print(f"第 {idx} 组 = 编号 {g.get('key', '')}（{g.get('utterances', 0)} 句）")
    try:
        out = voices.name_voice(db, cfg, engine(), g["members"], args.name, is_self=bool(args.self),
                                relation=(args.relation or "").strip() or None, **_force_kwargs(args))
    except ValueError as e:
        raise CLIError(str(e) + ("" if getattr(args, "force", False) or "不像" not in str(e) else "（命令行用 --force）")) from e
    console.print(f"已标记为 [bold]{out['name']}[/bold] (id {out['id']})：{out['utterances_assigned']} 句，"
                  f"补充 {out['samples_added']} 个声纹样本（{len(out['segments'])} 段 / {len(out['days'])} 天）。")
    if out.get("samples_rejected"):
        console.print(f"[yellow]{out['samples_rejected']} 个样本和 {out['name']} 已有的样本不像，没有加入（确定是同一个人可加 --force）。[/yellow]")
    if out.get("members_inconsistent"):
        console.print(f"[yellow]这组里有 {out['members_inconsistent']} 段声音和其他成员不像，没有用来做样本。[/yellow]")
    if out.get("samples_from_clusters"):
        console.print(f"这组的录音已过期，{out['samples_from_clusters']} 个样本直接取自扫描时保存的声纹。")
    if out.get("skipped_user_labels"):
        console.print(f"{out['skipped_user_labels']} 句已手动标为别人，保持不变。")
    if args.self:
        console.print("已设为本人。")
    if args.relation:
        console.print(f"关系：{args.relation}")
    _reindex_after_enroll(cfg, db)
    _rescore_after(cfg, db)
    from ..audio.loudness import min_speech_dbfs

    left = voices.unknown_utterance_count(db, min_speech_dbfs(cfg))
    if left and out["samples_added"]:
        console.print(f"还有 {left} 句未识别：audio-note speakers relabel 可用新声纹重新识别其他记录。")
    return 0


def cmd_speakers_scan(cfg: Config, args: argparse.Namespace) -> int:
    """Only compute the missing cluster embeddings (what `voices` does before listing)."""
    voices = _lazy_module("speech.voices")
    db = _db(cfg)
    stats = _scan_voices(cfg, db, voices, _voice_engine_factory(cfg))
    if not stats["segments"] and not stats["skipped"]:
        console.print("没有需要扫描的录音段。")
    return 0


def cmd_speakers_reject(cfg: Config, args: argparse.Namespace) -> int:
    """「不是 TA」: the sentence at --at is not NAME (default: its current speaker). Vetoes the speaker for it (and with
    --cluster for the same voice in that recording), stores a negative sample, re-identifies the segment."""
    from ..speech import voices

    db = _db(cfg)
    u = utterance_at(db, _parse_datetime(args.at))
    if args.name:
        sp = _speaker_by_name(db, args.name)
    elif u["speaker_id"] is not None:
        sp = db.one("SELECT * FROM speakers WHERE id=?", (int(u["speaker_id"]),))
        if sp is None:
            raise CLIError("这句的说话人已不存在：用 --name 指定")
    else:
        raise CLIError("这句还没有说话人：用 --name 指定要否决的人")
    engine = _optional_engine(cfg)
    try:
        res = voices.reject_voice(db, cfg, engine, int(sp["id"]), [int(u["id"])], whole_cluster=bool(getattr(args, "cluster", False)))
    except (LookupError, ValueError) as e:
        raise CLIError(str(e)) from e
    console.print(f"{str(u['abs_start'])[:19].replace('T', ' ')} 「{(u['text'] or '').strip()[:40]}」：已标为不是 {sp['name']}"
                  f"（{res['rejected']} 句，{res['negatives_added']} 个负样本，重新识别后 {res['relabeled_changed']} 句有变化）。")
    if res.get("warning"):
        console.print(f"[yellow]{res['warning']}（只记下了否决，没有存成负样本）[/yellow]")
    elif engine is None:
        console.print("[yellow]声纹模型不可用：只记下了否决，没有存负样本。[/yellow]")
    _rescore_after(cfg, db)
    return 0


def cmd_speakers_negatives(cfg: Config, args: argparse.Namespace) -> int:
    """List the negative samples (all, or one person's); --delete ID removes one."""
    as_json = wants_json(args)
    if as_json and getattr(args, "delete", None) is not None:
        raise CLIError("--json 只用于列出负样本，不能和 --delete 一起用")
    db = _db(cfg)
    if getattr(args, "delete", None) is not None:
        n = db.delete_speaker_negative(int(args.delete))
        if not n:
            raise CLIError(f"没有 id={args.delete} 的负样本")
        console.print(f"已删除负样本 #{args.delete}。")
        _rescore_after(cfg, db)
        return 0
    where, params = "", []
    if args.name:
        sp = _speaker_by_name(db, args.name)
        where, params = " WHERE n.speaker_id=?", [int(sp["id"])]
    rows = db.query("SELECT n.id, n.speaker_id, n.utterance_id, s.name, n.duration_s, n.source_path, n.created_at, u.abs_start, u.text"
                    " FROM speaker_negatives n"
                    " JOIN speakers s ON s.id=n.speaker_id LEFT JOIN utterances u ON u.id=n.utterance_id"
                    f"{where} ORDER BY s.name, n.id", params)
    if as_json:
        emit_json({"negatives": [
            {"id": int(r["id"]), "speaker_id": int(r["speaker_id"]), "speaker": r["name"],
             "duration_s": float(r["duration_s"]) if r["duration_s"] is not None else None,
             "source_path": r["source_path"], "created_at": r["created_at"], "utterance_id": r["utterance_id"],
             "abs_start": r["abs_start"], "text": r["text"]} for r in rows]})
        return 0
    t = Table(title="负样本（不是 TA 的声音）")
    for col in ("id", "不是", "时长", "来源", "那句话", "加入时间"):
        t.add_column(col)
    for r in rows:
        src = Path(str(r["source_path"]).split("#")[0]).name if r["source_path"] else "-"
        dur = f"{float(r['duration_s']):.1f}s" if r["duration_s"] is not None else "-"
        said = f"{str(r['abs_start'])[:19].replace('T', ' ')} {(r['text'] or '').strip()[:30]}" if r["abs_start"] else "-"
        t.add_row(str(r["id"]), r["name"], dur, src, said, r["created_at"] or "")
    console.print(t)
    if not rows:
        console.print("（没有负样本：在记录里对认错的句子用 audio-note speakers reject --at \"时间\"）")
    return 0


def _relabel_segment_quietly(cfg: Config, db: DB, engine: Any, seg_id: int) -> None:
    """After a label, re-identify that segment's utterances so the transcript shows the new name."""
    try:
        from ..speech.pipeline import Processor

        changed = Processor(cfg, db, engine=engine).relabel_segment(seg_id)
        console.print(f"已按新声纹重新识别录音段 #{seg_id}：{changed} 条语句的说话人有变化。")
    except Exception as e:  # noqa: BLE001
        log.warning("relabel after label failed: %s", e)


def cmd_speakers_label(cfg: Config, args: argparse.Namespace) -> int:
    from ..speech.speakers import SpeakerEngine

    db = _db(cfg)
    seg_id: int | None = None
    if args.at:
        if args.segment:
            raise CLIError("--at 与 <segment_id|path> 二选一")
        seg, a, b = resolve_label_target(db, _parse_datetime(args.at), args.dur)
        seg_id = int(seg["id"])
        path = Path(seg["path"])
    else:
        if not args.segment:
            raise CLIError("请给出 <segment_id|path>（并用 --from/--to 指定秒数），或者用 --at \"YYYY-MM-DD HH:MM:SS\"")
        if args.start is None or args.end is None:
            raise CLIError("使用 <segment_id|path> 时必须给出 --from 和 --to（秒）")
        ref = str(args.segment)
        if ref.isdigit():
            row = db.get_segment(int(ref))
            if row is None:
                raise CLIError(f"没有 id={ref} 的录音段")
            seg_id = int(row["id"])
            path = Path(row["path"])
        else:
            path = Path(ref)
            row = db.one("SELECT id FROM segments WHERE path=?", (str(path),))
            seg_id = int(row["id"]) if row else None
        a, b = float(args.start), float(args.end)
    if not path.exists():
        raise CLIError(f"音频文件不存在（可能已过期删除）: {path}")
    if b <= a:
        raise CLIError("--to 必须大于 --from")
    engine = SpeakerEngine(cfg)
    try:
        min_s = float(cfg.speakers.get("min_label_sample_s", 2.5))
    except (TypeError, ValueError, AttributeError):
        min_s = 2.5
    try:
        sid = engine.enroll_from_segment_turns(db, args.name, path, [(a, b)], min_speech_s=min_s, **_force_kwargs(args))
    except ValueError as e:
        if getattr(e, "min_s", None) is not None:       # speakers.TooShortSample (checked by attribute: tests stub the module)
            raise CLIError(f"{e}（命令行的 label 只能提取声纹，请选 ≥ {min_s:g} 秒语音的范围，或在界面的「记录」页标记这句）") from e
        raise CLIError(str(e) + ("" if getattr(args, "force", False) or "不像" not in str(e) else "（命令行用 --force）")) from e
    console.print(f"已把 {path.name} 的 {a:.1f}s–{b:.1f}s 标记为 {args.name} (id {sid})。")
    if getattr(args, "self", False):
        db.set_self_speaker(int(sid))
        console.print(f"已把 {args.name} 设为本人。")
    relation = (getattr(args, "relation", None) or "").strip()
    if relation:
        db.set_relation(int(sid), relation, "user")
        console.print(f"关系：{relation}")
    _reindex_after_enroll(cfg, db)
    if seg_id is not None:
        _relabel_segment_quietly(cfg, db, engine, seg_id)
    _rescore_after(cfg, db)
    console.print("历史记录可用 audio-note speakers relabel [--day YYYY-MM-DD] 按新声纹重新识别（不重复转写）。")
    return 0


def cmd_speakers_relabel(cfg: Config, args: argparse.Namespace) -> int:
    """Re-identify speakers on already transcribed segments (no ASR, no cost)."""
    from ..speech.speakers import SpeakerEngine

    Processor = _lazy("speech.pipeline", "Processor")
    db = _db(cfg)
    if not db.speaker_embeddings():
        raise CLIError("还没有注册任何说话人（先 speakers enroll 或 speakers label）")
    day = _parse_day(args.day) if args.day else None
    proc = Processor(cfg, db, engine=SpeakerEngine(cfg))
    stats = proc.relabel(day=day, segment_id=args.segment)
    console.print(f"重新识别 {stats['segments']} 段：{stats['changed']} 条语句的说话人有变化，{stats['skipped']} 段跳过（音频已删除或失败）。")
    return 0


def cmd_speakers_calibrate(cfg: Config, args: argparse.Namespace) -> int:
    db = _db(cfg)
    embs = db.speaker_embeddings()
    if len(embs) < 2:
        raise CLIError("至少需要 2 个注册样本才能校准（建议每人 ≥ 3 个样本、≥ 2 个人）")
    r = calibrate_threshold(embs)
    t = Table(title="说话人相似度分布（cosine）")
    for col in ("类型", "对数", "最小", "平均", "最大"):
        t.add_column(col, justify="right")
    for label, key in (("同一人", "same"), ("不同人", "cross")):
        s = r[key]
        if s:
            t.add_row(label, str(s["n"]), f"{s['min']:.3f}", f"{s['mean']:.3f}", f"{s['max']:.3f}")
        else:
            t.add_row(label, "0", "-", "-", "-")
    console.print(t)
    console.print(f"样本 {r['n_embeddings']} 个 / 说话人 {r['n_speakers']} 人；当前阈值 identify_threshold = {cfg.speakers.identify_threshold}")
    if r["same"] is None:
        console.print("[yellow]没有同一人的多个样本，无法估计同人分布；请每人至少注册 2 个样本。[/yellow]")
    if r["cross"] is None:
        console.print("[yellow]只有一个说话人，无法估计不同人分布；请再注册一个人。[/yellow]")
    if r["overlap"]:
        console.print("[yellow]同人与异人分数区间重叠，识别会有误判；建议增加样本或检查录音质量。[/yellow]")
    if r["suggested"] is not None:
        console.print(f"建议阈值: [bold]{r['suggested']:.2f}[/bold]（写入 config.yaml 的 speakers.identify_threshold）")
    return 0


def _rescore_after(cfg: Config, db: DB, quiet: bool = False) -> int:
    """The reference voices changed (a sample, a negative sample, "this is me"): rebuild the centroids and
    re-decide the stored sentences from their stored vectors (speech.scoring.refresh: numpy only, no audio).
    Says so in one line when sentences changed; never fails the command. Returns how many changed."""
    try:
        from ..speech import scoring

        res = scoring.refresh(db, cfg, source="cli")
    except Exception as e:  # noqa: BLE001 - the command did its job; the service catches up on its next run
        log.warning("decisions not refreshed: %s", e)
        return 0
    changed = int((res.get("rescore") or {}).get("changed") or 0) if res.get("rescored") else 0
    if changed and not quiet:
        console.print(f"已按现在的声纹重新判断历史记录：{changed} 句有变化。")
    return changed


_STATE_LABELS = (("sure", "确定"), ("guess", "可能是"), ("unknown", "未知"))


def cmd_speakers_rescore(cfg: Config, args: argparse.Namespace) -> int:
    """Re-decide who spoke from the stored voice vectors (no audio, no model, no ASR): seconds for a whole
    history. --dry-run only reports what would change."""
    from ..speech import scoring

    db = _db(cfg)
    day = _parse_day(args.day) if args.day else None
    seg_ids = [int(args.segment)] if args.segment is not None else None
    dry = bool(getattr(args, "dry_run", False))
    out: dict[str, Any] = {}
    if not dry:
        rep_ = scoring.adapt(db, cfg)
        out["adapt"] = {"written": rep_.get("written", 0), "changed": bool(rep_.get("changed"))}
    res = scoring.rescore(db, cfg, day=day, segment_ids=seg_ids, dry_run=dry)
    out.update({k: res[k] for k in ("utterances", "scored", "no_vector", "faint", "user", "before", "after", "changed",
                                    "speaker_changed", "moves", "days", "dry_run", "version", "normalised", "wall_s")})
    if not dry and day is None and seg_ids is None:
        db.set_kv(scoring.KV_RESCORE, {"version": res["version"], "at": datetime.now().isoformat(timespec="seconds"),
                                       "summary": {k: out[k] for k in out if k != "adapt"}})
    if wants_json(args):
        emit_json(out)
        return 0
    t = Table(title="重判说话人" + ("（预演，没有写入）" if dry else ""))
    for col in ("", "之前", "之后"):
        t.add_column(col, justify="right")
    for key, label in _STATE_LABELS:
        t.add_row(label, str(res["before"][key]), str(res["after"][key]))
    console.print(t)
    console.print(f"共 {res['utterances']} 句：{res['scored']} 句用存下来的声纹重判，{res['user']} 句是你手标的（不动），"
                  f"{res['faint']} 句太轻、{res['no_vector']} 句没有存声纹（保持原样）。")
    verb = "会有变化" if dry else "有变化"
    console.print(f"{res['changed']} 句{verb}（其中 {res['speaker_changed']} 句换了说话人），用时 {res['wall_s']:.2f} 秒。")
    if res["after"]["guess"]:
        console.print(f"有 {res['after']['guess']} 句标成了“可能是某人”：在界面的「说话人 → 核对」里确认或否定，系统会跟着学。")
    return 0


def cmd_speakers_scoring(cfg: Config, args: argparse.Namespace) -> int:
    """How voices are being decided right now: the scoring mode, the impostor cohort, each speaker's samples and
    learned centroids, and how many sentences are sure / maybe / unknown."""
    from ..speech import scoring

    db = _db(cfg)
    g = scoring.Gallery.load(db, cfg)
    p = g.params
    cents: dict[int, list[Any]] = {}
    for r in db.query("SELECT speaker_id, condition, n, speech_s, updated_at FROM speaker_centroids ORDER BY speaker_id, condition"):
        cents.setdefault(int(r["speaker_id"]), []).append(r)
    st = db.one("SELECT SUM(speaker_source IS 'user' AND speaker_id IS NOT NULL) AS user,"
                " SUM(NOT (speaker_source IS 'user' AND speaker_id IS NOT NULL) AND speaker_id IS NOT NULL) AS sure,"
                " SUM(speaker_id IS NULL AND speaker_guess_id IS NOT NULL) AS guess,"
                " SUM(speaker_id IS NULL AND speaker_guess_id IS NULL) AS unknown FROM utterances")
    states = {k: int(st[k] or 0) for k in ("sure", "guess", "unknown", "user")} if st else {}
    last = db.get_kv(scoring.KV_RESCORE) or None
    data = {"scoring": p.scoring, "version": g.version, "normalised": bool(g.normalised),
            "cohort": int(len(g.cohort)) if g.cohort is not None else 0, "cohort_min": int(p.cohort_min),
            "thresholds": ({"sure": p.sure_norm, "guess": p.guess_norm, "unit": "norm"} if g.normalised else
                           {"sure": p.identify_threshold, "guess": p.identify_threshold - p.guess_raw_margin, "unit": "cosine"}),
            "speakers": [{"id": sp.sid, "name": sp.name, "samples": int(len(sp.every)), "outlier_samples": int(sp.outliers),
                          "centroids": [{"condition": c["condition"], "n": int(c["n"] or 0),
                                         "speech_s": round(float(c["speech_s"] or 0.0), 1), "updated_at": c["updated_at"]}
                                        for c in cents.get(sp.sid, [])]} for sp in g.speakers],
            "states": states, "last_rescore": last,
            "stale": bool(g.has_speakers and (last or {}).get("version") != g.version)}
    if wants_json(args):
        emit_json(data)
        return 0
    mode = "新评分层 v5" if p.scoring != "legacy" else "旧规则 legacy"
    if g.normalised:
        console.print(f"评分方式：{mode}；已存 {data['cohort']} 个“别人的声音”用来归一化，"
                      f"分数 ≥ {p.sure_norm:g} 自动认人，≥ {p.guess_norm:g} 标成“可能是”。")
    else:
        console.print(f"评分方式：{mode}；“别人的声音”还不够 {p.cohort_min} 个，暂时按原始相似度判断"
                      f"（≥ {p.identify_threshold:g} 自动认人）。")
    cond_names = {"near": "近", "far": "远", "system": "系统声音", "all": "综合"}
    t = Table(title="说话人的声音")
    for col in ("人", "样本", "不参与的离群样本", "学到的声音中心（证据段数）"):
        t.add_column(col)
    for sp in data["speakers"]:
        learned = "、".join(f"{cond_names.get(c['condition'], c['condition'])} {c['n']}" for c in sp["centroids"]) or "-"
        t.add_row(sp["name"], str(sp["samples"]), str(sp["outlier_samples"]) if sp["outlier_samples"] else "-", learned)
    console.print(t)
    if not data["speakers"]:
        console.print("（还没有登记任何说话人）")
    if states:
        console.print(f"记录里：确定 {states['sure']} 句，可能是 {states['guess']} 句，未知 {states['unknown']} 句，你手标的 {states['user']} 句。")
    if data["stale"]:
        console.print("[yellow]声纹库变了但历史还没重判：audio-note speakers rescore（服务在运行时会自动做）。[/yellow]")
    return 0


def register(sub: Any) -> None:
    sp = sub.add_parser("speakers", help="说话人注册 / 列表 / 标注 / 校准")
    ssub = sp.add_subparsers(dest="speakers_cmd", metavar="<sub>")
    ssub.required = True
    s = ssub.add_parser("enroll", help="用音频样本注册说话人")
    s.add_argument("name")
    s.add_argument("files", nargs="+", help="wav/mp3/... 样本文件")
    s.add_argument("--notes", default=None)
    s.add_argument("--force", action="store_true", help="样本和此人已有样本不像时仍然加入")
    s.set_defaults(func=handler(cmd_speakers_enroll))
    s = ssub.add_parser("record", help="用麦克风现场录一段样本并注册说话人（注册自己最方便）")
    s.add_argument("name")
    s.add_argument("--seconds", type=int, default=20, help="录制秒数（默认 20，至少 5）")
    s.add_argument("--notes", default=None)
    s.set_defaults(func=handler(cmd_speakers_record))
    add_json_flag(ssub.add_parser("list", help="列出说话人")).set_defaults(func=handler(cmd_speakers_list))
    s = ssub.add_parser("label", help="把某段录音的一个时间范围标记为某人（并重新识别该段）")
    s.add_argument("segment", nargs="?", default=None, help="segment id 或音频文件路径（记录文件的 ## 标题里有 #id 和文件名）")
    s.add_argument("name")
    s.add_argument("--from", dest="start", type=float, default=None, metavar="S", help="起始秒（配合 segment）")
    s.add_argument("--to", dest="end", type=float, default=None, metavar="S", help="结束秒（配合 segment）")
    s.add_argument("--at", default=None, metavar="TIME", help='记录里那句话的时间，如 "2026-10-31 08:19:32"（自动找到对应录音段与区间）')
    s.add_argument("--dur", type=float, default=None, metavar="S", help="配合 --at：从该句开始取多少秒（默认取整句）")
    s.add_argument("--self", dest="self", action="store_true", help="这个人就是我（本人）")
    s.add_argument("--relation", default=None, metavar="R", help="和本人的关系，如 同事 / 朋友 / 领导")
    s.add_argument("--force", action="store_true", help="这段声音和此人已有样本不像时仍然加入")
    s.set_defaults(func=handler(cmd_speakers_label))
    s = ssub.add_parser("voices", help="列出记录里还没认出来的声音（按声音分组），--assign 给一组起名字")
    s.add_argument("--assign", default=None, metavar="N|编号", help="给第 N 组（或编号为 N 的组）命名；序号和编号见不带参数的输出")
    s.add_argument("name", nargs="?", default=None, help="配合 --assign：这组声音是谁")
    s.add_argument("--self", dest="self", action="store_true", help="配合 --assign：这组声音是我自己")
    s.add_argument("--relation", default=None, metavar="R", help="配合 --assign：和本人的关系")
    s.add_argument("--force", action="store_true", help="配合 --assign：这组声音和此人已有样本不像时仍然加入")
    add_json_flag(s).set_defaults(func=handler(cmd_speakers_voices))
    ssub.add_parser("scan", help="只扫描还没分析过声音的录音段（voices 会自动做）").set_defaults(func=handler(cmd_speakers_scan))
    s = ssub.add_parser("set-self", help="标出哪位说话人是本人（日报 / 问答以本人视角）")
    s.add_argument("name")
    s.set_defaults(func=handler(cmd_speakers_set_self))
    s = ssub.add_parser("relation", help="标注某人和本人的关系（- 清除）")
    s.add_argument("name")
    s.add_argument("relation", help="如 同事 / 朋友 / 领导 / 导师，或 - 清除")
    s.add_argument("--note", default=None, help="备注")
    s.set_defaults(func=handler(cmd_speakers_relation))
    s = ssub.add_parser("relations", help="让模型从记录里推断各说话人和本人的关系（手动标过的不动）")
    s.add_argument("--force", action="store_true", help="忽略上次推断的缓存")
    s.add_argument("--speaker", default=None, metavar="NAME", help="只推断某一个人")
    s.set_defaults(func=handler(cmd_speakers_relations))
    s = ssub.add_parser("relabel", help="对已转写的录音段按当前声纹库重新识别说话人（不重复 ASR）")
    s.add_argument("--day", default=None, help="只处理某天 YYYY-MM-DD")
    s.add_argument("--segment", type=int, default=None, help="只处理某个 segment id")
    s.set_defaults(func=handler(cmd_speakers_relabel))
    ssub.add_parser("calibrate", help="根据注册样本估计识别阈值").set_defaults(func=handler(cmd_speakers_calibrate))
    s = ssub.add_parser("audit", help="检查每个声纹样本和同一人其他样本是否一致（--fix 删除离群样本）")
    s.add_argument("--fix", action="store_true", help="删除离群样本")
    s.set_defaults(func=handler(cmd_speakers_audit))
    s = ssub.add_parser("reject", help="某句话认错了人：标为「不是 TA」（存负样本，以后不再认成 TA）")
    s.add_argument("--at", required=True, metavar="TIME", help='记录里那句话的时间，如 "2026-11-04 13:17:05"')
    s.add_argument("--name", default=None, help="不是谁（默认这句当前的说话人）")
    s.add_argument("--cluster", action="store_true", help="这段录音里同一个声音的其他句子也一起")
    s.set_defaults(func=handler(cmd_speakers_reject))
    s = ssub.add_parser("negatives", help="列出负样本（不是 TA 的声音）；--delete ID 删除一个")
    s.add_argument("name", nargs="?", default=None, help="只看某人的")
    s.add_argument("--delete", type=int, default=None, metavar="ID", help="删除这个负样本")
    add_json_flag(s).set_defaults(func=handler(cmd_speakers_negatives))
    s = ssub.add_parser("rescore", help="按现在的声纹库重判历史记录的说话人（只用存下来的声纹，几秒钟，不读音频）")
    s.add_argument("--day", default=None, help="只处理某天 YYYY-MM-DD")
    s.add_argument("--segment", type=int, default=None, help="只处理某个 segment id")
    s.add_argument("--dry-run", dest="dry_run", action="store_true", help="只看会有什么变化，不写入")
    add_json_flag(s).set_defaults(func=handler(cmd_speakers_rescore))
    s = ssub.add_parser("scoring", help="查看现在是怎么认人的：评分方式、每个人学到的声音中心、确定 / 可能 / 未知各多少句")
    add_json_flag(s).set_defaults(func=handler(cmd_speakers_scoring), read_only=True)
