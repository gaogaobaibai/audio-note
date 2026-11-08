"""CLI command coverage for audio_note/cli.py: every sub-command driven through `cli.main([...])`.

No network (MimoClient is replaced by a fake whose methods raise), no audio devices (capture is
replaced by a fake module), no model loading (SpeakerEngine / Processor / analyzers are stubbed via
sys.modules or monkeypatch). All data goes to tmp_path through the `tmp_cfg` fixture.
"""
from __future__ import annotations

import io
import logging
import sys
import time
import types
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import yaml
from rich.console import Console

from audio_note import cli
from audio_note.config import load_config
from audio_note.db import DB

from tests.conftest import FakeVAD, MODELS_DIR, sine
from tests.test_cli import tmp_cfg  # noqa: F401  (fixture re-used from the existing CLI tests)

SR = 16000


# --- helpers ------------------------------------------------------------------------

class FakeClient:
    """Stands in for MimoClient: any call means a test tried to use the network."""

    usage = {"calls": 0, "audio_seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0}

    def chat(self, *a, **k): raise AssertionError("no network")
    def text(self, *a, **k): raise AssertionError("no network")
    def json(self, *a, **k): raise AssertionError("no network")
    def transcribe_bytes(self, *a, **k): raise AssertionError("no network")


@pytest.fixture
def out(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Capture everything the CLI prints through its module-level rich console."""
    buf = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=buf, width=400, force_terminal=False,
                                                 color_system=None, highlight=False, soft_wrap=True))
    return buf


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    c = FakeClient()
    monkeypatch.setattr(cli, "_client", lambda cfg: c)
    return c


@pytest.fixture
def make_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Like tmp_cfg but accepts nested overrides (e.g. memory.retrieval)."""
    def make(overrides: dict | None = None) -> Any:
        data = tmp_path / "data2"
        base = {"paths": {"data_dir": str(data), "models_dir": str(MODELS_DIR)}, "mimo": {"api_key": ""}}
        base.update(overrides or {})
        p = tmp_path / "config2.yaml"
        p.write_text(yaml.safe_dump(base), encoding="utf-8")
        monkeypatch.setenv("AUDIO_NOTE_CONFIG", str(p))
        monkeypatch.delenv("MIMO_API_KEY", raising=False)
        return load_config(p)

    yield make
    from audio_note.logging_setup import _remove_own_handlers
    _remove_own_handlers(logging.getLogger())


def stub_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: Any) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def write_wav(path: Path, seconds: float = 2.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), sine(seconds), SR, subtype="PCM_16")
    return path


class _FakeSilero(FakeVAD):
    """Accepts SileroVAD's constructor and reports speech everywhere."""

    def __init__(self, model_path: Any, threads: int = 1):
        super().__init__(default=1.0)


def add_done_segment(db: DB, cfg: Any, when: datetime, path: Path | None = None, status: str = "done") -> int:
    p = path or (cfg.audio_dir / f"seg_{when.strftime('%H%M%S')}.opus")
    sid = db.add_segment(p, when, 10.0)
    db.set_segment_status(sid, status)
    return sid


# --- argparse / main plumbing ------------------------------------------------------------

def test_no_subcommand_exits_2():
    with pytest.raises(SystemExit) as ex:
        cli.main([])
    assert ex.value.code == 2


def test_unknown_subcommand_exits_2():
    with pytest.raises(SystemExit) as ex:
        cli.main(["frobnicate"])
    assert ex.value.code == 2


def test_run_help_exits_zero():
    with pytest.raises(SystemExit) as ex:
        cli.main(["run", "--help"])
    assert ex.value.code == 0


def test_unreadable_config_returns_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, out: io.StringIO):
    bad = tmp_path / "bad.yaml"
    bad.write_text("mimo: [unclosed", encoding="utf-8")
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    assert cli.main(["--config", str(bad), "status"]) == 2
    assert "读取配置失败" in out.getvalue()


def test_cli_error_in_command_is_exit_2_with_message(tmp_cfg, monkeypatch: pytest.MonkeyPatch, out: io.StringIO):
    def boom(cfg, args):
        raise cli.CLIError("故意失败")

    monkeypatch.setattr(cli, "cmd_cleanup", boom)      # build_parser() resolves the name at call time
    assert cli.main(["cleanup"]) == 2
    text = out.getvalue()
    assert "错误" in text and "故意失败" in text


def test_unexpected_exception_is_exit_1(tmp_cfg, monkeypatch: pytest.MonkeyPatch, out: io.StringIO):
    def boom(cfg, args):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(cli, "cmd_cleanup", boom)
    assert cli.main(["cleanup"]) == 1
    assert "RuntimeError" in out.getvalue() and "kaboom" in out.getvalue()


def test_keyboard_interrupt_is_exit_130(tmp_cfg, monkeypatch: pytest.MonkeyPatch, out: io.StringIO):
    def boom(cfg, args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "cmd_cleanup", boom)
    assert cli.main(["cleanup"]) == 130
    assert "已中断" in out.getvalue()


def test_logging_setup_failure_does_not_block_command(tmp_cfg, monkeypatch: pytest.MonkeyPatch):
    import audio_note.logging_setup as ls

    def bad_setup(cfg, verbose=False):
        raise OSError("no log file")

    monkeypatch.setattr(ls, "setup_logging", bad_setup)
    monkeypatch.setattr(cli, "cmd_cleanup", lambda cfg, args: None)   # `None` result -> exit 0
    assert cli.main(["-v", "cleanup"]) == 0


# --- pure helpers ----------------------------------------------------------------------------

def test_lazy_missing_attribute_is_cli_error():
    with pytest.raises(cli.CLIError) as ex:
        cli._lazy("config", "NoSuchThing")
    assert "audio_note.config" in str(ex.value) and "NoSuchThing" in str(ex.value)


def test_expand_files_globs_and_errors(tmp_path: Path):
    for n in ("b.wav", "a.wav", "c.txt"):
        (tmp_path / n).write_bytes(b"")
    hits = cli._expand_files([str(tmp_path / "*.wav"), str(tmp_path / "c.txt")])
    assert [h.name for h in hits] == ["a.wav", "b.wav", "c.txt"]
    with pytest.raises(cli.CLIError) as ex:
        cli._expand_files([str(tmp_path / "*.mp3")])
    assert "没有匹配的文件" in str(ex.value)
    with pytest.raises(cli.CLIError) as ex:
        cli._expand_files([str(tmp_path / "missing.wav")])
    assert "文件不存在" in str(ex.value)


def test_parse_datetime_day_and_bytes():
    assert cli._parse_datetime("2026-10-31 09:00") == datetime(2026, 10, 31, 9, 0)
    assert cli._parse_datetime(" 2026-10-31T09:00:05 ") == datetime(2026, 10, 31, 9, 0, 5)
    assert cli._parse_datetime("2026-10-31") == datetime(2026, 10, 31)
    with pytest.raises(cli.CLIError):
        cli._parse_datetime("31/10/2026")
    assert cli._parse_day("2026-10-31") == "2026-10-31"
    assert cli._parse_day(None) == datetime.now().strftime("%Y-%m-%d")
    with pytest.raises(cli.CLIError):
        cli._parse_day("2026/10/31")
    assert cli._human_bytes(512) == "512 B"
    assert cli._human_bytes(1536) == "1.5 KB"
    assert cli._human_bytes(3 * 1024 ** 3) == "3.0 GB"
    assert cli._human_bytes(5 * 1024 ** 4) == "5.0 TB"


def test_invoke_edge_cases():
    def pos_only(a, /, b, *args, **kwargs):
        return (a, b, args, kwargs)

    assert cli._invoke(pos_only, a=1, b=2, c=3) == (1, 2, (), {})

    class NoSignature:
        @property
        def __signature__(self):
            raise ValueError("no signature")

        def __call__(self, **kw):
            return kw

    # inspect.signature() fails -> everything is passed through as keywords
    assert cli._invoke(NoSignature(), x=1, y=2) == {"x": 1, "y": 2}

    def needs(a, b):
        return a + b

    with pytest.raises(cli.CLIError) as ex:
        cli._invoke(needs, a=1)
    assert "缺少参数 'b'" in str(ex.value)


def test_print_result_variants(out: io.StringIO):
    class WithText:
        text = "对象文本"

    class Item:
        def __init__(self, text, speaker_name=None, start=None, start_s=None):
            self.text, self.speaker_name, self.start, self.start_s = text, speaker_name, start, start_s

    cli._print_result(None)
    assert out.getvalue() == ""
    cli._print_result("纯文本")
    cli._print_result("# 标题\n\n正文")
    cli._print_result(("元组首项", 123))
    cli._print_result(WithText())
    cli._print_result({"text": "字典文本", "x": 1})
    cli._print_result({"done": 2, "failed": 0})
    cli._print_result([
        {"text": "甲说", "speaker": "甲", "start": 1.5},
        {"text_raw": "无人", "start_s": 2},
        {"text": "无时间"},
    ])
    cli._print_result([Item("乙说", "乙", start=3.0), Item("丙说", None, start_s=4.0), Item("无", None), 42])
    cli._print_result(3.14)
    text = out.getvalue()
    for needle in ("纯文本", "标题", "正文", "元组首项", "对象文本", "字典文本", "done: 2", "failed: 0",
                   "[    1.5s] 甲：甲说", "[    2.0s] 无人", "无时间", "[    3.0s] 乙：乙说", "[    4.0s] 丙说", "42", "3.14"):
        assert needle in text, needle
    assert "123" not in text and "x: 1" not in text


# --- devices / import ---------------------------------------------------------------------------

def test_devices_with_fake_capture(tmp_cfg, monkeypatch: pytest.MonkeyPatch, out: io.StringIO):
    class Dev:
        index, name, hostapi, channels, default_sr = 3, "Fake Mic", "WASAPI", 2, 48000.0

    stub_module(monkeypatch, "audio_note.audio.capture", list_input_devices=lambda: [Dev()])
    assert cli.main(["devices"]) == 0
    text = out.getvalue()
    assert "Fake Mic" in text and "WASAPI" in text and "48000" in text and "audio.device" in text


def test_import_real_wav_through_real_importer(tmp_cfg, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, out: io.StringIO):
    import audio_note.audio.vad as vad_mod

    monkeypatch.setattr(vad_mod, "SileroVAD", _FakeSilero)
    wav = write_wav(tmp_path / "rec_20261031_091500.wav", seconds=6.0)
    assert cli.main(["import", str(wav)]) == 0
    text = out.getvalue()
    assert f"{wav.name}: 1 段" in text and "共导入 1 段" in text
    db = DB(tmp_cfg.db_path)
    seg = db.one("SELECT * FROM segments")
    assert seg is not None and seg["source"] == "import" and seg["origin"] == str(wav)
    assert seg["started_at"].startswith("2026-10-31T09:15:00")      # time taken from the file name
    assert Path(seg["path"]).exists() and tmp_cfg.data_dir in Path(seg["path"]).parents


def test_import_start_and_multi_file_paths(tmp_cfg, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, out: io.StringIO):
    import audio_note.audio.importer as imp
    import audio_note.audio.vad as vad_mod

    seen: list[tuple] = []

    def fake_import(cfg, db, path, start_time=None, vad=None):
        seen.append((Path(path).name, start_time))
        if Path(path).name.startswith("bad"):
            raise RuntimeError("解码失败")
        return [1, 2]

    monkeypatch.setattr(vad_mod, "SileroVAD", _FakeSilero)
    monkeypatch.setattr(imp, "import_file", fake_import)
    a = write_wav(tmp_path / "a.wav")
    b = write_wav(tmp_path / "bad.wav")
    assert cli.main(["import", "--start", "2026-10-31 09:00", str(a), str(b)]) == 0
    text = out.getvalue()
    assert "--start 指定了多个文件的同一起始时间" in text
    assert "a.wav: 2 段" in text and "导入失败" in text and "解码失败" in text and "共导入 2 段" in text
    assert seen == [("a.wav", datetime(2026, 10, 31, 9, 0)), ("bad.wav", datetime(2026, 10, 31, 9, 0))]
    # without --start the importer gets None (it guesses from the name / mtime itself)
    seen.clear()
    assert cli.main(["import", str(tmp_path / "*.wav")]) == 0
    assert [s[1] for s in seen] == [None, None]


def test_import_bad_start_is_cli_error(tmp_cfg, tmp_path: Path, out: io.StringIO):
    wav = write_wav(tmp_path / "a.wav")
    assert cli.main(["import", "--start", "昨天", str(wav)]) == 2
    assert "无法解析时间" in out.getvalue()


# --- transcribe / process -----------------------------------------------------------------------

def test_transcribe_missing_file_is_cli_error(tmp_cfg, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, out: io.StringIO):
    stub_module(monkeypatch, "audio_note.speech.pipeline", transcribe_file=lambda **k: [])
    assert cli.main(["transcribe", str(tmp_path / "nope.wav")]) == 2
    assert "文件不存在" in out.getvalue()


def test_transcribe_calls_stub_and_prints(tmp_cfg, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_client, out: io.StringIO):
    seen: dict = {}

    def transcribe_file(cfg, client, path, db=None, engine=None):
        seen.update(cfg=cfg, client=client, path=path, db=db)
        return [{"start_s": 0.0, "end_s": 1.0, "speaker": "甲", "text": "你好"}, {"start_s": 1.0, "text": "世界"}]

    stub_module(monkeypatch, "audio_note.speech.pipeline", transcribe_file=transcribe_file)
    wav = write_wav(tmp_path / "x.wav")
    assert cli.main(["transcribe", str(wav)]) == 0
    assert seen["client"] is fake_client and seen["path"] == wav and isinstance(seen["db"], DB)
    text = out.getvalue()
    assert "甲：你好" in text and "世界" in text


def test_process_without_run_method_is_cli_error(tmp_cfg, monkeypatch: pytest.MonkeyPatch, fake_client, out: io.StringIO):
    class NoRun:
        def __init__(self, cfg, db, client): ...

    stub_module(monkeypatch, "audio_note.speech.pipeline", Processor=NoRun)
    assert cli.main(["process"]) == 2
    assert "Processor 没有 run()" in out.getvalue()


def test_process_prints_usage_line(tmp_cfg, monkeypatch: pytest.MonkeyPatch, fake_client, out: io.StringIO):
    class Proc:
        def __init__(self, cfg, db, client): ...

        def process_pending(self, limit=None, retry_failed=False):
            return {"processed": 1}

    fake_client.usage = {"calls": 3, "audio_seconds": 12.4, "prompt_tokens": 10, "completion_tokens": 5}
    stub_module(monkeypatch, "audio_note.speech.pipeline", Processor=Proc)
    assert cli.main(["process"]) == 0
    text = out.getvalue()
    assert "processed: 1" in text and "API: 3 次调用" in text and "音频 12s" in text and "tokens 10+5" in text


# --- speakers -------------------------------------------------------------------------------------

class FakeEngine:
    """SpeakerEngine stand-in: writes real DB rows so the CLI's follow-up queries see them."""

    instances: list["FakeEngine"] = []

    def __init__(self, cfg):
        self.cfg = cfg
        self.calls: list[tuple] = []
        FakeEngine.instances.append(self)

    def enroll(self, db, name, sources, notes=None):
        self.calls.append(("enroll", name, [Path(s) for s in sources], notes))
        sid = db.upsert_speaker(name, notes)
        for s in sources:
            db.add_speaker_embedding(sid, np.ones(8, dtype=np.float32), str(s), 5.0)
        return sid

    def enroll_from_segment_turns(self, db, name, seg_path, turns, **kw):
        self.calls.append(("label", name, Path(seg_path), turns))
        if name == "太短":
            raise ValueError("选中的范围只有 0.5s 语音")
        sid = db.upsert_speaker(name)
        db.add_speaker_embedding(sid, np.ones(8, dtype=np.float32), f"{seg_path}#{turns}", 2.0)
        return sid

    def embed_speech_only(self, audio, max_s=60.0):
        self.calls.append(("embed", float(len(audio)) / SR))
        return np.ones(8, dtype=np.float32), 1.0 if float(np.abs(audio).max()) < 0.02 else 4.5


class FakeProcessor:
    """speech.pipeline.Processor stand-in for relabel / relabel_segment."""

    def __init__(self, cfg, db, client=None, engine=None):
        self.db, self.engine = db, engine

    def relabel_segment(self, seg_id):
        return 1

    def relabel(self, day=None, segment_id=None):
        FakeProcessor.last = (day, segment_id)
        return {"segments": 2, "changed": 3, "skipped": 1}


@pytest.fixture
def speakers_stubs(monkeypatch: pytest.MonkeyPatch):
    FakeEngine.instances.clear()
    stub_module(monkeypatch, "audio_note.speech.speakers", SpeakerEngine=FakeEngine)
    stub_module(monkeypatch, "audio_note.speech.pipeline", Processor=FakeProcessor)
    return FakeEngine


def test_speakers_enroll_and_list(tmp_cfg, tmp_path: Path, speakers_stubs, out: io.StringIO):
    db = DB(tmp_cfg.db_path)
    now = datetime(2026, 10, 31, 9, 0)
    seg = add_done_segment(db, tmp_cfg, now)
    db.add_utterances([{"segment_id": seg, "start_s": 0, "end_s": 1, "abs_start": now.isoformat(), "day": "2026-10-31",
                        "text_raw": "李雷来了", "text": "李雷来了"}])
    wavs = [write_wav(tmp_path / f"s{i}.wav") for i in range(2)]
    assert cli.main(["speakers", "enroll", "李雷", str(tmp_path / "s*.wav"), "--notes", "同事"]) == 0
    text = out.getvalue()
    assert "已注册 李雷 (id 1)，共 2 个样本" in text
    assert "已更新检索索引（1 条）" in text            # _reindex_after_enroll ran over the utterance
    assert "speakers relabel" in text                     # done segments exist -> hint
    assert FakeEngine.instances[0].calls == [("enroll", "李雷", wavs, "同事")]
    assert db.one("SELECT text_seg FROM utterance_tokens")["text_seg"]

    out.truncate(0); out.seek(0)
    assert cli.main(["speakers", "list"]) == 0
    text = out.getvalue()
    assert "李雷" in text and "同事" in text and "10s" in text and "（空）" not in text


def test_speakers_list_empty(tmp_cfg, out: io.StringIO):
    assert cli.main(["speakers", "list"]) == 0
    assert "已注册说话人" in out.getvalue() and "（空）" in out.getvalue()


def test_speakers_enroll_missing_file_is_cli_error(tmp_cfg, tmp_path: Path, speakers_stubs, out: io.StringIO):
    assert cli.main(["speakers", "enroll", "甲", str(tmp_path / "none.wav")]) == 2
    assert "文件不存在" in out.getvalue()


def test_speakers_label_by_segment_id(tmp_cfg, tmp_path: Path, speakers_stubs, out: io.StringIO):
    db = DB(tmp_cfg.db_path)
    wav = write_wav(tmp_cfg.audio_dir / "2026-10-31" / "seg.wav", 3.0)
    seg = add_done_segment(db, tmp_cfg, datetime(2026, 10, 31, 8, 0), path=wav)
    assert cli.main(["speakers", "label", str(seg), "韩梅梅", "--from", "0.5", "--to", "2.0"]) == 0
    text = out.getvalue()
    assert "已把 seg.wav 的 0.5s–2.0s 标记为 韩梅梅 (id 1)" in text
    assert f"已按新声纹重新识别录音段 #{seg}：1 条语句的说话人有变化" in text
    assert "speakers relabel" in text
    assert FakeEngine.instances[0].calls[0] == ("label", "韩梅梅", wav, [(0.5, 2.0)])
    assert db.speaker_embeddings()[0][1] == "韩梅梅"


def test_speakers_label_by_path_not_in_db_skips_relabel(tmp_cfg, tmp_path: Path, speakers_stubs, out: io.StringIO):
    wav = write_wav(tmp_path / "loose.wav", 3.0)
    assert cli.main(["speakers", "label", str(wav), "甲", "--from", "0", "--to", "1"]) == 0
    text = out.getvalue()
    assert "标记为 甲" in text and "重新识别录音段" not in text


@pytest.mark.parametrize("argv,needle", [
    (["speakers", "label", "甲"], "请给出 <segment_id|path>"),
    (["speakers", "label", "1", "甲", "--from", "0"], "必须给出 --from 和 --to"),
    (["speakers", "label", "1", "甲", "--from", "0", "--to", "1", "--at", "2026-10-31 08:00:00"], "--at 与 <segment_id|path> 二选一"),
    (["speakers", "label", "999", "甲", "--from", "0", "--to", "1"], "没有 id=999 的录音段"),
    (["speakers", "label", "--at", "2026-10-31 08:00:00", "甲"], "没有已转写的语句"),
])
def test_speakers_label_argument_errors(tmp_cfg, speakers_stubs, out: io.StringIO, argv, needle):
    assert cli.main(argv) == 2
    assert needle in out.getvalue()


def test_speakers_label_range_and_engine_errors(tmp_cfg, tmp_path: Path, speakers_stubs, out: io.StringIO):
    db = DB(tmp_cfg.db_path)
    wav = write_wav(tmp_path / "seg.wav", 3.0)
    seg = add_done_segment(db, tmp_cfg, datetime(2026, 10, 31, 8, 0), path=wav)
    assert cli.main(["speakers", "label", str(seg), "甲", "--from", "2", "--to", "1"]) == 2
    assert "--to 必须大于 --from" in out.getvalue()
    assert cli.main(["speakers", "label", str(seg), "太短", "--from", "0", "--to", "1"]) == 2
    assert "0.5s 语音" in out.getvalue()
    # audio gone (expired) -> clear error
    gone = add_done_segment(db, tmp_cfg, datetime(2026, 10, 31, 9, 0), path=tmp_path / "gone.opus")
    assert cli.main(["speakers", "label", str(gone), "甲", "--from", "0", "--to", "1"]) == 2
    assert "音频文件不存在" in out.getvalue()


def test_speakers_label_at_time_resolves_utterance(tmp_cfg, tmp_path: Path, speakers_stubs, out: io.StringIO):
    db = DB(tmp_cfg.db_path)
    wav = write_wav(tmp_cfg.audio_dir / "seg.wav", 5.0)
    seg = add_done_segment(db, tmp_cfg, datetime(2026, 10, 31, 8, 19, 0), path=wav)
    db.add_utterances([
        {"segment_id": seg, "start_s": 0.0, "end_s": 2.0, "abs_start": "2026-10-31T08:19:00.000", "day": "2026-10-31", "text_raw": "a", "text": "a"},
        {"segment_id": seg, "start_s": 2.0, "end_s": 4.5, "abs_start": "2026-10-31T08:19:02.000", "day": "2026-10-31", "text_raw": "b", "text": "b"},
    ])
    # inside the second utterance -> its full range
    assert cli.main(["speakers", "label", "--at", "2026-10-31 08:19:03", "甲"]) == 0
    assert "的 2.0s–4.5s 标记为 甲" in out.getvalue()
    # --dur overrides the end; a time after every utterance picks the nearest one
    assert cli.main(["speakers", "label", "--at", "2026-10-31 08:19:30", "--dur", "1.5", "甲"]) == 0
    assert "的 2.0s–3.5s 标记为 甲" in out.getvalue()
    # direct call: the tolerance window bounds the search; a matching call returns the segment row
    row, a, b = cli.resolve_label_target(db, datetime(2026, 10, 31, 8, 19, 1))
    assert int(row["id"]) == seg and (a, b) == (0.0, 2.0)
    with pytest.raises(cli.CLIError) as ex:
        cli.resolve_label_target(db, datetime(2026, 10, 31, 8, 19, 30), tolerance_s=5.0)
    assert "附近 5s 内没有已转写的语句" in str(ex.value)


def test_speakers_relabel(tmp_cfg, speakers_stubs, out: io.StringIO):
    assert cli.main(["speakers", "relabel"]) == 2
    assert "还没有注册任何说话人" in out.getvalue()
    db = DB(tmp_cfg.db_path)
    db.add_speaker_embedding(db.upsert_speaker("甲"), np.ones(4, dtype=np.float32), None, 5.0)
    assert cli.main(["speakers", "relabel", "--day", "2026-10-31", "--segment", "7"]) == 0
    assert FakeProcessor.last == ("2026-10-31", 7)
    assert "重新识别 2 段：3 条语句的说话人有变化，1 段跳过" in out.getvalue()
    assert cli.main(["speakers", "relabel", "--day", "20261031"]) == 2
    assert "日期格式应为 YYYY-MM-DD" in out.getvalue()


def test_speakers_calibrate_warnings(tmp_cfg, out: io.StringIO):
    db = DB(tmp_cfg.db_path)
    rng = np.random.default_rng(3)
    sid = db.upsert_speaker("甲")
    base = rng.normal(size=16).astype(np.float32)
    for _ in range(3):
        db.add_speaker_embedding(sid, base + 0.1 * rng.normal(size=16).astype(np.float32), None, 5.0)
    assert cli.main(["speakers", "calibrate"]) == 0
    text = out.getvalue()
    assert "只有一个说话人" in text and "建议阈值" in text and "identify_threshold = 0.45" in text
    # a second speaker with the *same* voice -> overlapping distributions
    sid2 = db.upsert_speaker("乙")
    for _ in range(2):
        db.add_speaker_embedding(sid2, base + 0.1 * rng.normal(size=16).astype(np.float32), None, 5.0)
    out.truncate(0); out.seek(0)
    assert cli.main(["speakers", "calibrate"]) == 0
    text = out.getvalue()
    assert "同人与异人分数区间重叠" in text and "样本 5 个 / 说话人 2 人" in text


def test_speakers_calibrate_one_sample_each(tmp_cfg, out: io.StringIO):
    db = DB(tmp_cfg.db_path)
    db.add_speaker_embedding(db.upsert_speaker("甲"), np.array([1, 0, 0, 0], dtype=np.float32), None, 5.0)
    db.add_speaker_embedding(db.upsert_speaker("乙"), np.array([0, 1, 0, 0], dtype=np.float32), None, 5.0)
    assert cli.main(["speakers", "calibrate"]) == 0
    assert "没有同一人的多个样本" in out.getvalue()


class FakeMic:
    """MicCapture stand-in: hands out `blocks` then None."""

    blocks: list[np.ndarray | None] = []

    def __init__(self, device=None, block_ms=32, gain=1.0):
        self._it = iter(FakeMic.blocks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def read(self, timeout=1.0):
        try:
            return next(self._it)
        except StopIteration:
            return None


@pytest.fixture
def record_stubs(monkeypatch: pytest.MonkeyPatch, speakers_stubs):
    stub_module(monkeypatch, "audio_note.audio.capture", MicCapture=FakeMic)
    monkeypatch.setattr(time, "sleep", lambda s: None)       # skip the 3-2-1 countdown
    return FakeMic


def test_speakers_record_enrolls_sample(tmp_cfg, record_stubs, out: io.StringIO):
    FakeMic.blocks = [sine(1.0) for _ in range(7)]
    assert cli.main(["speakers", "record", "我", "--seconds", "5", "--notes", "本人"]) == 0
    text = out.getvalue()
    assert "将录制 5 秒" in text and "开始录音" in text and "已录 5s" in text
    assert "样本已保存" in text and "已注册 我 (id 1)，语音 4.5s，共 1 个样本" in text
    saved = list((tmp_cfg.speakers_dir / "我").glob("record_*.wav"))
    assert len(saved) == 1
    audio, sr = sf.read(str(saved[0]))
    assert sr == SR and len(audio) == 5 * SR                 # trimmed to exactly `seconds`
    db = DB(tmp_cfg.db_path)
    assert db.one("SELECT notes FROM speakers")["notes"] == "本人"
    assert FakeEngine.instances[0].calls == [("embed", 5.0)]


def test_speakers_record_silent_sample_is_rejected(tmp_cfg, record_stubs, out: io.StringIO):
    FakeMic.blocks = [np.zeros(SR, dtype=np.float32) for _ in range(5)]
    assert cli.main(["speakers", "record", "我", "--seconds", "1"]) == 2     # --seconds is raised to 5
    text = out.getvalue()
    assert "音量很低" in text and "只检测到 1.0 秒语音" in text
    assert DB(tmp_cfg.db_path).one("SELECT COUNT(*) AS n FROM speakers")["n"] == 0


def test_speakers_record_without_mic_data_times_out(tmp_cfg, record_stubs, monkeypatch: pytest.MonkeyPatch, out: io.StringIO):
    FakeMic.blocks = []
    clock = iter(range(0, 10_000, 20))
    monkeypatch.setattr(time, "time", lambda: float(next(clock)))
    assert cli.main(["speakers", "record", "我"]) == 2
    assert "麦克风没有数据" in out.getvalue()


# --- daily / push-test ------------------------------------------------------------------------

class FakeDaily:
    runs: list[tuple] = []
    pushes: list[str] = []
    result: Any = ("# 日报\n\n- 事项一", {"todos": []})

    def __init__(self, cfg, db, client):
        assert isinstance(client, FakeClient)

    def run(self, day, force=False):
        FakeDaily.runs.append((day, force))
        return FakeDaily.result

    def push(self, day):
        FakeDaily.pushes.append(day)
        return {"toast": None}


@pytest.fixture
def daily_stub(monkeypatch: pytest.MonkeyPatch, fake_client):
    FakeDaily.runs.clear(); FakeDaily.pushes.clear()
    FakeDaily.result = ("# 日报\n\n- 事项一", {"todos": []})
    stub_module(monkeypatch, "audio_note.analysis.daily", DailyAnalyzer=FakeDaily)
    return FakeDaily


def test_daily_generates_and_pushes(tmp_cfg, daily_stub, out: io.StringIO):
    assert cli.main(["daily", "--day", "2026-10-30"]) == 0
    assert FakeDaily.runs == [("2026-10-30", False)] and FakeDaily.pushes == []
    assert "事项一" in out.getvalue()
    assert cli.main(["daily", "--day", "2026-10-30", "--force", "--push"]) == 0
    assert FakeDaily.runs[-1] == ("2026-10-30", True) and FakeDaily.pushes == ["2026-10-30"]
    assert "已推送 2026-10-30 的日报" in out.getvalue()


def test_daily_existing_summary_is_reused(tmp_cfg, daily_stub, out: io.StringIO):
    db = DB(tmp_cfg.db_path)
    db.execute("INSERT INTO daily_summaries(day, summary_md, created_at) VALUES (?,?,?)",
               ("2026-10-29", "# 旧日报\n\n已有内容", "2026-10-29 22:30:00"))
    assert cli.main(["daily", "--day", "2026-10-29"]) == 0
    text = out.getvalue()
    assert "2026-10-29 的日报已存在（2026-10-29 22:30:00）" in text and "已有内容" in text
    assert FakeDaily.runs == []
    # --push on an existing report builds the analyzer only for the push
    assert cli.main(["daily", "--day", "2026-10-29", "--push"]) == 0
    assert FakeDaily.runs == [] and FakeDaily.pushes == ["2026-10-29"]


def test_daily_non_markdown_result_and_bad_day(tmp_cfg, daily_stub, out: io.StringIO):
    FakeDaily.result = {"text": "非 Markdown 结果"}
    assert cli.main(["daily", "--day", "2026-10-28"]) == 0
    assert "非 Markdown 结果" in out.getvalue()
    assert cli.main(["daily", "--day", "10-28"]) == 2
    assert "日期格式应为 YYYY-MM-DD" in out.getvalue()


def test_daily_without_api_key_is_cli_error(tmp_cfg, monkeypatch: pytest.MonkeyPatch, out: io.StringIO):
    stub_module(monkeypatch, "audio_note.analysis.daily", DailyAnalyzer=FakeDaily)
    assert cli.main(["daily"]) == 2
    assert "MIMO_API_KEY" in out.getvalue()


class FakeMessage:
    def __init__(self, title, body_md, url=None):
        self.title, self.body_md, self.url = title, body_md, url


def _push_stub(monkeypatch: pytest.MonkeyPatch, results: Any, sent: list):
    class FakeNotifier:
        def __init__(self, cfg): ...

        def send(self, msg):
            sent.append(msg)
            return results

    stub_module(monkeypatch, "audio_note.push.base", Notifier=FakeNotifier, Message=FakeMessage)


def test_push_test_reports_per_channel(tmp_cfg, monkeypatch: pytest.MonkeyPatch, out: io.StringIO):
    sent: list = []
    _push_stub(monkeypatch, {"toast": None, "feishu": "HTTP 500"}, sent)
    assert cli.main(["push-test", "你好", "世界"]) == 1
    text = out.getvalue()
    assert "toast: 已发送" in text and "feishu: HTTP 500" in text
    assert sent[0].title == "audio-note 测试" and sent[0].body_md == "你好 世界"

    _push_stub(monkeypatch, {"toast": None}, sent)
    assert cli.main(["push-test", "ok"]) == 0


def test_push_test_non_dict_result(tmp_cfg, monkeypatch: pytest.MonkeyPatch, out: io.StringIO):
    _push_stub(monkeypatch, "sent-to-3-channels", [])
    assert cli.main(["push-test", "hi"]) == 0
    assert "sent-to-3-channels" in out.getvalue()


# --- chat ---------------------------------------------------------------------------------------

class FakeRetriever:
    def __init__(self, cfg, db, client, mode=None, expand=None):
        self.mode, self.expand = mode, expand


class FakeChat:
    last: "FakeChat | None" = None

    def __init__(self, cfg, db, client, retriever=None, strategy=None):
        self.retriever, self.strategy, self.asked, self.repl_called = retriever, strategy, [], False
        FakeChat.last = self

    def ask(self, question, history=None):
        self.asked.append((question, history))
        return f"回答：{question}"

    def repl(self):
        self.repl_called = True


@pytest.fixture
def chat_stubs(monkeypatch: pytest.MonkeyPatch, fake_client):
    FakeChat.last = None
    stub_module(monkeypatch, "audio_note.memory.chat", MemoryChat=FakeChat)
    stub_module(monkeypatch, "audio_note.memory.retrieval", Retriever=FakeRetriever)


def test_chat_one_shot_question(tmp_cfg, chat_stubs, out: io.StringIO):
    assert cli.main(["chat", "--strategy", "rag", "--retrieval", "fts", "--no-expand", "昨天", "谁来过"]) == 0
    chat = FakeChat.last
    assert chat.asked == [("昨天 谁来过", [])] and not chat.repl_called
    assert chat.strategy == "rag" and chat.retriever.mode == "fts" and chat.retriever.expand is False
    assert "回答：昨天 谁来过" in out.getvalue()


def test_chat_without_question_enters_repl(tmp_cfg, chat_stubs, out: io.StringIO):
    assert cli.main(["chat", "--expand"]) == 0
    chat = FakeChat.last
    assert chat.repl_called and chat.asked == []
    assert chat.strategy is None and chat.retriever.mode is None and chat.retriever.expand is True


def test_chat_without_api_key_is_cli_error(tmp_cfg, monkeypatch: pytest.MonkeyPatch, out: io.StringIO):
    stub_module(monkeypatch, "audio_note.memory.chat", MemoryChat=FakeChat)
    stub_module(monkeypatch, "audio_note.memory.retrieval", Retriever=FakeRetriever)
    assert cli.main(["chat", "问题"]) == 2
    assert "mimo.api_key" in out.getvalue()


# --- cleanup / status / reindex / init ------------------------------------------------------------

def test_cleanup_with_and_without_vad(tmp_cfg, monkeypatch: pytest.MonkeyPatch, out: io.StringIO):
    import audio_note.audio.storage as storage
    import audio_note.audio.vad as vad_mod

    seen: list = []

    def fake_cleanup(cfg, db, vad=None):
        seen.append(vad)
        return {"rechecked": 1, "no_speech": 0, "expired": 2, "disk_cap": 0}

    monkeypatch.setattr(storage, "cleanup", fake_cleanup)
    monkeypatch.setattr(vad_mod, "SileroVAD", _FakeSilero)
    assert cli.main(["cleanup"]) == 0
    assert isinstance(seen[0], _FakeSilero)
    assert "rechecked: 1" in out.getvalue() and "expired: 2" in out.getvalue()

    class Broken:
        def __init__(self, *a, **k):
            raise OSError("model missing")

    monkeypatch.setattr(vad_mod, "SileroVAD", Broken)
    assert cli.main(["cleanup"]) == 0
    assert seen[1] is None and "VAD 不可用，跳过语音复检" in out.getvalue()


def test_status_on_populated_db(tmp_cfg, out: io.StringIO):
    tmp_cfg.ensure_dirs()
    db = DB(tmp_cfg.db_path)
    today = datetime.now().replace(microsecond=0)
    done = add_done_segment(db, tmp_cfg, today)
    add_done_segment(db, tmp_cfg, today.replace(hour=1), status="failed")
    db.add_utterances([
        {"segment_id": done, "start_s": 0, "end_s": 1, "abs_start": today.isoformat(), "day": today.strftime("%Y-%m-%d"),
         "text_raw": "今天", "text": "今天"},
        {"segment_id": done, "start_s": 1, "end_s": 2, "abs_start": "2026-10-01T10:00:00.000", "day": "2026-10-01",
         "text_raw": "以前", "text": "以前"},
    ])
    db.execute("INSERT INTO daily_summaries(day, summary_md, created_at, pushed_at) VALUES (?,?,?,?)",
               ("2026-10-01", "# x", "2026-10-01 22:30:00", None))
    db.set_kv("last_daily_day", "2026-10-01")
    db.add_speaker_embedding(db.upsert_speaker("甲", "朋友"), np.ones(4, dtype=np.float32), None, 7.0)
    (tmp_cfg.audio_dir / "blob.bin").write_bytes(b"\0" * 2048)
    assert cli.main(["status"]) == 0
    text = out.getvalue()
    for needle in ("未运行", "未配置", "done=1", "failed=1", "今日语句", "2 条 / 2 天", "2.0 KB", "上限 20 GB",
                   "最近日报", "2026-10-01（生成 2026-10-01 22:30:00，推送 未推送）", "调度 last_daily_day",
                   "失败段", "--retry-failed", "已注册说话人", "甲", "朋友", "7s"):
        assert needle in text, needle
    assert "未注册说话人" not in text


def test_reindex_keywords_only(tmp_cfg, out: io.StringIO):
    import audio_note.memory.index  # noqa: F401  (gives the package its `index` attribute, see report)

    db = DB(tmp_cfg.db_path)
    seg = add_done_segment(db, tmp_cfg, datetime(2026, 10, 31, 9, 0))
    db.add_utterances([{"segment_id": seg, "start_s": 0, "end_s": 1, "abs_start": "2026-10-31T09:00:00.000",
                        "day": "2026-10-31", "text_raw": "测试", "text": "测试"} for _ in range(3)])
    assert cli.main(["reindex", "--no-vectors"]) == 0
    text = out.getvalue()
    assert "关键词索引已重建：3 条" in text and "向量索引跳过" in text
    assert db.one("SELECT COUNT(*) AS n FROM utterance_tokens")["n"] == 3


def test_reindex_fts_mode_skips_vectors(make_cfg, out: io.StringIO):
    import audio_note.memory.index  # noqa: F401

    make_cfg({"memory": {"retrieval": "fts"}})
    assert cli.main(["reindex"]) == 0
    assert "关键词索引已重建：0 条" in out.getvalue() and "向量索引跳过" in out.getvalue()


def test_reindex_vectors(tmp_cfg, monkeypatch: pytest.MonkeyPatch, out: io.StringIO):
    import audio_note.memory.embed as embed
    import audio_note.memory.index  # noqa: F401
    import audio_note.memory.vectors as vectors

    class Emb:
        name = "fake-embedder"

    missing = {"n": 0}
    monkeypatch.setattr(embed, "get_embedder", lambda cfg: Emb())
    monkeypatch.setattr(vectors, "missing_count", lambda db, model: missing["n"])
    monkeypatch.setattr(vectors, "index_missing", lambda db, emb: missing["n"])
    assert cli.main(["reindex"]) == 0
    assert "向量索引已是最新（模型 fake-embedder，0 条）" in out.getvalue()
    missing["n"] = 4
    assert cli.main(["reindex"]) == 0
    text = out.getvalue()
    assert "正在用 fake-embedder 嵌入 4 条记录" in text and "向量索引完成：新增 4 条" in text


def test_init_with_existing_config_does_not_overwrite(tmp_cfg, out: io.StringIO):
    cfg_path = Path(tmp_cfg["_config_path"]) if "_config_path" in tmp_cfg else None
    before = cfg_path.read_text(encoding="utf-8") if cfg_path else None
    rc = cli.main(["--config", str(cfg_path), "init"])       # without --config, init targets the project's own config.yaml
    text = out.getvalue()
    assert rc in (0, 1)
    assert "配置文件已存在" in text and "（不覆盖）" in text
    assert f"数据目录: {tmp_cfg.data_dir}" in text
    assert "mimo.api_key 未填写" in text
    assert "模型文件" in text
    assert (rc == 1) == ("缺失模型文件" in text)
    if cfg_path:
        assert cfg_path.read_text(encoding="utf-8") == before
    assert tmp_cfg.audio_dir.is_dir() and tmp_cfg.speakers_dir.is_dir()


def test_init_missing_models_returns_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, out: io.StringIO):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "paths": {"data_dir": str(tmp_path / "data"), "models_dir": str(tmp_path / "no_models")},
        "mimo": {"api_key": "sk-real-looking-key"},
    }), encoding="utf-8")
    monkeypatch.setenv("AUDIO_NOTE_CONFIG", str(cfg_path))
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    try:
        assert cli.main(["--config", str(cfg_path), "init"]) == 1
        text = out.getvalue()
        assert "缺失模型文件" in text and "sherpa-onnx releases" in text and "缺失" in text
        assert "mimo.api_key 未填写" not in text
    finally:
        from audio_note.logging_setup import _remove_own_handlers
        _remove_own_handlers(logging.getLogger())


# --- speakers v2: set-self / relation / relations / voices / scan ------------------------------------

def _unit_vec(seed: int, dim: int = 192) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class VoiceEngine:
    """SpeakerEngine stand-in for `speakers voices` / `scan`: the real voices module drives it
    (cluster_embedding for the backfill, enroll_from_audio for naming)."""

    instances: list["VoiceEngine"] = []

    def __init__(self, cfg):
        self.cfg = cfg
        self.calls: list[tuple] = []
        VoiceEngine.instances.append(self)

    def cluster_embedding(self, audio, turns):
        self.calls.append(("cluster", len(turns)))
        return _unit_vec(7)

    def enroll_from_audio(self, db, name, audio, source=None, copy_as_source=False, **kw):
        self.calls.append(("enroll", name, source))
        sid = db.upsert_speaker(name)
        db.add_speaker_embedding(sid, _unit_vec(8), source, len(audio) / SR)
        return sid


@pytest.fixture
def voice_engine(monkeypatch: pytest.MonkeyPatch):
    import audio_note.speech.speakers as spk

    VoiceEngine.instances.clear()
    monkeypatch.setattr(spk, "SpeakerEngine", VoiceEngine)
    return VoiceEngine


def _unknown_segment(db: DB, cfg: Any, when: datetime, texts: list[str], cluster: int = 0) -> int:
    """A done segment with a real wav file and unidentified utterances of 2 s each in one cluster."""
    wav = write_wav(cfg.audio_dir / when.strftime("%Y-%m-%d") / f"seg_{when.strftime('%H%M%S')}.wav", 2.0 * len(texts) + 1)
    seg = add_done_segment(db, cfg, when, path=wav)
    db.add_utterances([{"segment_id": seg, "start_s": 2.0 * i, "end_s": 2.0 * i + 2.0, "cluster_label": cluster,
                        "abs_start": (when + timedelta(seconds=2 * i)).isoformat(timespec="milliseconds"),
                        "day": when.strftime("%Y-%m-%d"), "text_raw": t, "text": t} for i, t in enumerate(texts)])
    return seg


def test_speakers_set_self_and_relation(tmp_cfg, out: io.StringIO):
    db = DB(tmp_cfg.db_path)
    a = db.upsert_speaker("张伟")
    b = db.upsert_speaker("王敏")
    assert cli.main(["speakers", "set-self", "李四"]) == 2
    assert "没有叫 李四 的说话人" in out.getvalue()
    assert cli.main(["speakers", "set-self", "张伟"]) == 0
    assert "已把 张伟 (id 1) 设为本人" in out.getvalue() and db.self_speaker()["id"] == a
    out.truncate(0); out.seek(0)
    assert cli.main(["speakers", "set-self", "王敏"]) == 0                      # moves: still exactly one self
    assert "本人已从 张伟 改为 王敏" in out.getvalue()
    assert [r["id"] for r in db.query("SELECT id FROM speakers WHERE is_self=1")] == [b]
    assert cli.main(["speakers", "relation", "王敏", "同事"]) == 2               # self needs no relation
    assert cli.main(["speakers", "relation", "张伟", "同事", "--note", "同一个组"]) == 0
    assert "已标注 张伟：同事（同一个组）" in out.getvalue()
    row = db.one("SELECT relation, relation_source, relation_note, relation_updated_at FROM speakers WHERE id=?", (a,))
    assert tuple(row)[:3] == ("同事", "user", "同一个组") and row["relation_updated_at"]
    out.truncate(0); out.seek(0)
    assert cli.main(["speakers", "list"]) == 0
    text = out.getvalue()
    assert "本人" in text and "关系" in text and "同事" in text and "我" in text
    assert text.index("王敏") < text.index("张伟")                                # self is listed first
    assert cli.main(["speakers", "relation", "张伟", "-"]) == 0
    assert "已清除 张伟 的关系标注" in out.getvalue()
    assert tuple(db.one("SELECT relation, relation_source, relation_note FROM speakers WHERE id=?", (a,))) == (None, None, None)
    db.set_relation(a, "朋友", "auto", "推断")
    out.truncate(0); out.seek(0)
    assert cli.main(["speakers", "list"]) == 0
    assert "朋友（自动推断）" in out.getvalue()
    # no self yet -> hint
    db.set_self_speaker(None)
    out.truncate(0); out.seek(0)
    assert cli.main(["speakers", "list"]) == 0
    assert "speakers set-self" in out.getvalue()


def test_speakers_relations_prints_table(tmp_cfg, fake_client, monkeypatch: pytest.MonkeyPatch, out: io.StringIO):
    from audio_note.analysis import relations

    db = DB(tmp_cfg.db_path)
    sid = db.upsert_speaker("王敏")
    seen: list[dict] = []

    def fake_infer(cfg, db_, client, speaker_ids=None, force=False):
        assert client is fake_client
        seen.append({"speaker_ids": speaker_ids, "force": force})
        return [{"id": sid, "name": "王敏", "relation": "同事", "confidence": 0.82, "reason": "叫他王总", "applied": True, "skipped": None},
                {"id": 9, "name": "李强", "relation": "不确定", "confidence": 0.3, "reason": "证据不足", "applied": False, "skipped": None},
                {"id": 10, "name": "小雨", "relation": None, "confidence": None, "reason": None, "applied": False, "skipped": "语句太少"}]

    monkeypatch.setattr(relations, "infer_relations", fake_infer)
    assert cli.main(["speakers", "relations"]) == 0
    text = out.getvalue()
    for needle in ("关系推断", "王敏", "同事", "0.82", "已写入", "叫他王总", "李强", "未写入", "小雨", "跳过：语句太少", "共 3 人，写入 1 个关系"):
        assert needle in text, needle
    assert cli.main(["speakers", "relations", "--force", "--speaker", "王敏"]) == 0
    assert seen == [{"speaker_ids": None, "force": False}, {"speaker_ids": [sid], "force": True}]
    assert cli.main(["speakers", "relations", "--speaker", "没有人"]) == 2
    assert "没有叫 没有人 的说话人" in out.getvalue() and len(seen) == 2
    monkeypatch.setattr(relations, "infer_relations", lambda *a, **k: [])
    out.truncate(0); out.seek(0)
    assert cli.main(["speakers", "relations"]) == 0
    assert "没有需要推断的说话人" in out.getvalue()


def test_speakers_relations_without_api_key_is_cli_error(tmp_cfg, out: io.StringIO):
    assert cli.main(["speakers", "relations"]) == 2
    assert "MIMO_API_KEY" in out.getvalue()


def test_speakers_voices_scan_list_and_assign(tmp_cfg, voice_engine, out: io.StringIO):
    db = DB(tmp_cfg.db_path)
    assert cli.main(["speakers", "voices"]) == 0
    assert "没有待确认的声音" in out.getvalue() and voice_engine.instances == []      # empty db: no engine loaded
    seg1 = _unknown_segment(db, tmp_cfg, datetime(2026, 10, 30, 9, 0), ["早上好", "今天开会", "下午再说"])
    seg2 = _unknown_segment(db, tmp_cfg, datetime(2026, 10, 31, 10, 0), ["明天见"])
    out.truncate(0); out.seek(0)
    assert cli.main(["speakers", "scan"]) == 0
    text = out.getvalue()
    assert "正在扫描 2 段录音" in text and "扫描完成：2 段 / 2 个声音簇" in text
    assert len(voice_engine.instances) == 1 and [c for c in voice_engine.instances[0].calls if c[0] == "cluster"] == [("cluster", 3), ("cluster", 1)]
    assert len(db.cluster_embeddings()) == 2
    out.truncate(0); out.seek(0)
    assert cli.main(["speakers", "scan"]) == 0
    assert "没有需要扫描的录音段" in out.getvalue() and len(voice_engine.instances) == 1
    # list: one group (identical fake embeddings), 4 sentences over 2 days, 3 example lines
    out.truncate(0); out.seek(0)
    assert cli.main(["speakers", "voices"]) == 0
    text = out.getvalue()
    assert "未识别的声音（1 组，4 句未识别）" in text and "2026-10-30 ~ 2026-10-31（2 天）" in text and "8s" in text
    assert "早上好" in text and "今天开会" in text and "--assign <#>" in text
    assert cli.main(["speakers", "voices", "--assign", "1"]) == 2
    assert "需要给出姓名" in out.getvalue()
    assert cli.main(["speakers", "voices", "--assign", "5", "张伟"]) == 2
    assert "没有第 5 组" in out.getvalue()
    out.truncate(0); out.seek(0)
    assert cli.main(["speakers", "voices", "--assign", "1", "张伟", "--self", "--relation", "同事"]) == 0
    text = out.getvalue()
    assert "已标记为 张伟 (id 1)：4 句，补充 2 个声纹样本（2 段 / 2 天）" in text and "已设为本人" in text and "关系：同事" in text
    assert "已更新检索索引（4 条）" in text and "speakers relabel" not in text    # nothing unidentified is left
    utts = db.query("SELECT speaker_id, speaker_name, speaker_score, speaker_source FROM utterances ORDER BY id")
    assert all(tuple(u) == (1, "张伟", None, "user") for u in utts) and len(utts) == 4
    me = db.self_speaker()
    assert me["name"] == "张伟" and me["relation"] == "同事" and me["relation_source"] == "user"
    enrolls = [c for e in voice_engine.instances for c in e.calls if c[0] == "enroll"]
    assert len(enrolls) == 2 and enrolls[0][1] == "张伟" and "#0.0-6.0" in enrolls[0][2]      # longest member first
    assert (tmp_cfg.transcripts_dir / "2026-10-30.md").exists() and "张伟" in (tmp_cfg.transcripts_dir / "2026-10-31.md").read_text(encoding="utf-8")
    out.truncate(0); out.seek(0)
    assert cli.main(["speakers", "voices"]) == 0
    assert "没有待确认的声音" in out.getvalue()
    assert seg1 != seg2


def test_speakers_voices_assign_after_a_scan_asks_for_a_fresh_number(tmp_cfg, voice_engine, out: io.StringIO):
    db = DB(tmp_cfg.db_path)
    _unknown_segment(db, tmp_cfg, datetime(2026, 10, 30, 9, 0), ["早上好"])
    assert cli.main(["speakers", "voices", "--assign", "1", "张伟"]) == 1       # the scan just stored new clusters
    text = out.getvalue()
    assert "正在扫描 1 段录音" in text and "分组可能变了" in text
    assert db.one("SELECT COUNT(*) AS n FROM speakers")["n"] == 0
    out.truncate(0); out.seek(0)
    assert cli.main(["speakers", "voices", "--assign", "1", "张伟"]) == 0        # nothing new: proceeds
    assert "已标记为 张伟" in out.getvalue()
    assert db.one("SELECT speaker_name FROM utterances")["speaker_name"] == "张伟"


def test_speakers_voices_assign_too_short_is_cli_error(tmp_cfg, voice_engine, monkeypatch: pytest.MonkeyPatch, out: io.StringIO):
    def too_short(self, db, name, audio, source=None):
        raise ValueError("选中的范围只有 0.3s 语音（至少需要 1s），未注册")

    monkeypatch.setattr(VoiceEngine, "enroll_from_audio", too_short)
    db = DB(tmp_cfg.db_path)
    _unknown_segment(db, tmp_cfg, datetime(2026, 10, 30, 9, 0), ["嗯"])
    assert cli.main(["speakers", "scan"]) == 0
    assert cli.main(["speakers", "voices", "--assign", "1", "新人"]) == 2
    assert "语音太短" in out.getvalue()
    assert db.one("SELECT COUNT(*) AS n FROM speakers")["n"] == 0
    assert db.one("SELECT speaker_id FROM utterances")["speaker_id"] is None


def test_speakers_label_accepts_self_and_relation(tmp_cfg, tmp_path: Path, speakers_stubs, out: io.StringIO):
    db = DB(tmp_cfg.db_path)
    wav = write_wav(tmp_cfg.audio_dir / "2026-10-31" / "seg.wav", 3.0)
    seg = add_done_segment(db, tmp_cfg, datetime(2026, 10, 31, 8, 0), path=wav)
    assert cli.main(["speakers", "label", str(seg), "张伟", "--from", "0.5", "--to", "2.0", "--self", "--relation", "导师"]) == 0
    text = out.getvalue()
    assert "已把 张伟 设为本人" in text and "关系：导师" in text
    me = db.self_speaker()
    assert me["name"] == "张伟" and me["relation"] == "导师" and me["relation_source"] == "user"
    # without the flags nothing changes for the next person
    assert cli.main(["speakers", "label", str(seg), "王敏", "--from", "0.5", "--to", "2.0"]) == 0
    assert db.self_speaker()["name"] == "张伟" and db.one("SELECT relation FROM speakers WHERE name='王敏'")["relation"] is None


# --- review fixes: stable keys in `speakers voices`, engine failures as CLIError ------------------------------

def test_speakers_voices_shows_keys_and_assign_accepts_index_or_key(tmp_cfg, voice_engine, out: io.StringIO):
    from audio_note.speech import voices

    db = DB(tmp_cfg.db_path)
    _unknown_segment(db, tmp_cfg, datetime(2026, 10, 30, 9, 0), ["早上好", "今天开会"])
    assert cli.main(["speakers", "scan"]) == 0
    key = voices.group_unknown_voices(db, tmp_cfg)["groups"][0]["key"]
    out.truncate(0); out.seek(0)
    assert cli.main(["speakers", "voices"]) == 0
    text = out.getvalue()
    assert "编号" in text and key[:6] in text and "也可以写编号" in text
    # an unknown key, a too-short prefix and an index out of range are CLIErrors; nothing is named
    assert cli.main(["speakers", "voices", "--assign", "zzzzzz", "张伟"]) == 2
    assert "没有编号为 zzzzzz 的声音组" in out.getvalue()
    assert cli.main(["speakers", "voices", "--assign", "ab", "张伟"]) == 2            # too short for a key
    assert cli.main(["speakers", "voices", "--assign", "7", "张伟"]) == 2
    assert "没有第 7 组" in out.getvalue()
    assert db.one("SELECT COUNT(*) AS n FROM speakers")["n"] == 0
    # by (short) key: the group it resolved to is printed
    out.truncate(0); out.seek(0)
    assert cli.main(["speakers", "voices", "--assign", key[:6], "张伟"]) == 0
    text = out.getvalue()
    assert f"第 1 组 = 编号 {key}" in text and "已标记为 张伟" in text
    assert db.one("SELECT speaker_name FROM utterances")["speaker_name"] == "张伟"
    # by index: the key it resolved to is printed; by full key works as well
    _unknown_segment(db, tmp_cfg, datetime(2026, 10, 31, 9, 0), ["明天见"])
    assert cli.main(["speakers", "scan"]) == 0
    key2 = voices.group_unknown_voices(db, tmp_cfg)["groups"][0]["key"]
    assert key2 != key
    out.truncate(0); out.seek(0)
    assert cli.main(["speakers", "voices", "--assign", "1", "王敏"]) == 0
    assert f"第 1 组 = 编号 {key2}" in out.getvalue() and "已标记为 王敏" in out.getvalue()
    _unknown_segment(db, tmp_cfg, datetime(2026, 11, 1, 9, 0), ["后天见"])
    assert cli.main(["speakers", "scan"]) == 0
    key3 = voices.group_unknown_voices(db, tmp_cfg)["groups"][0]["key"]
    assert cli.main(["speakers", "voices", "--assign", key3, "王敏"]) == 0
    assert db.one("SELECT COUNT(*) AS n FROM utterances WHERE speaker_id IS NULL")["n"] == 0


def test_speakers_voices_engine_failure_is_a_cli_error(tmp_cfg, voice_engine, monkeypatch: pytest.MonkeyPatch, out: io.StringIO):
    db = DB(tmp_cfg.db_path)
    _unknown_segment(db, tmp_cfg, datetime(2026, 10, 30, 9, 0), ["早上好"])

    def boom(self, cfg):
        raise RuntimeError("onnxruntime: model file is corrupt")

    monkeypatch.setattr(VoiceEngine, "__init__", boom)
    assert cli.main(["speakers", "voices"]) == 2
    text = out.getvalue()
    assert "错误" in text and "声纹模型加载失败" in text and "Traceback" not in text
    # a missing model file is reported before the engine is even built
    cfg_path = Path(tmp_cfg["_config_path"])
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data.setdefault("speakers", {})["embedding_model"] = "missing-model.onnx"
    cfg_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr(VoiceEngine, "__init__", lambda self, cfg: pytest.fail("the engine must not be built"))
    out.truncate(0); out.seek(0)
    assert cli.main(["speakers", "scan"]) == 2
    assert "模型文件缺失" in out.getvalue() and "download_models" in out.getvalue()
    assert db.one("SELECT COUNT(*) AS n FROM cluster_embeddings")["n"] == 0


# --- speakers v3: audit [--fix], --force ------------------------------------------------------------

def test_speakers_audit_lists_outliers_and_fix_deletes_them(tmp_cfg, out: io.StringIO):
    db = DB(tmp_cfg.db_path)
    assert cli.main(["speakers", "audit"]) == 0
    assert "没有样本" in out.getvalue()
    a = db.upsert_speaker("甲")
    base = _unit_vec(1)
    good1 = db.add_speaker_embedding(a, base, "a.wav", 3.0)
    good2 = db.add_speaker_embedding(a, base * 0.9 + 0.1 * _unit_vec(2), "b.wav", 4.0)
    bad = db.add_speaker_embedding(a, _unit_vec(9), None, 5.0)
    b = db.upsert_speaker("乙")
    db.add_speaker_embedding(b, _unit_vec(3), "c.wav", 2.0)
    out.truncate(0); out.seek(0)
    assert cli.main(["speakers", "audit"]) == 0
    text = out.getvalue()
    assert "声纹样本一致性" in text and "a.wav" in text and "扫描时保存的声纹" in text and "c.wav" in text
    assert "1 个样本和其他样本不像" in text and "甲" in text and "audit --fix" in text
    assert db.one("SELECT COUNT(*) AS n FROM speaker_embeddings")["n"] == 4
    out.truncate(0); out.seek(0)
    assert cli.main(["speakers", "audit", "--fix"]) == 0
    assert "已删除 1 个离群样本" in out.getvalue()
    assert [r["id"] for r in db.query("SELECT id FROM speaker_embeddings WHERE speaker_id=? ORDER BY id", (a,))] == [good1, good2]
    assert bad not in [r["id"] for r in db.query("SELECT id FROM speaker_embeddings")]
    out.truncate(0); out.seek(0)
    assert cli.main(["speakers", "audit", "--fix"]) == 0
    assert "所有样本都" in out.getvalue()
    # two samples that disagree: the (longer) medoid is the reference and stays, the other one is the outlier
    db.add_speaker_embedding(b, _unit_vec(11), None, 1.0)
    out.truncate(0); out.seek(0)
    assert cli.main(["speakers", "audit", "--fix"]) == 0
    assert "已删除 1 个离群样本" in out.getvalue() and "乙 已没有样本" not in out.getvalue()
    assert [r["source_path"] for r in db.query("SELECT source_path FROM speaker_embeddings WHERE speaker_id=?", (b,))] == ["c.wav"]
    assert db.one("SELECT name FROM speakers WHERE id=?", (b,))["name"] == "乙"


def test_speakers_enroll_and_label_pass_force_only_when_asked(tmp_cfg, tmp_path: Path, speakers_stubs, monkeypatch: pytest.MonkeyPatch, out: io.StringIO):
    class SampleMismatch(ValueError):          # the CLI only relies on it being a ValueError whose message says 不像
        def __init__(self, name, score, threshold):
            super().__init__(f"这段声音和 {name} 已有的样本不像（相似度 {score:.2f} < {threshold:.2f}），没有加入")

    class ForceEngine(FakeEngine):
        def enroll(self, db, name, sources, notes=None, **kw):
            self.calls.append(("enroll-force", kw.get("force", False)))
            sid = db.upsert_speaker(name, notes)
            if kw.get("force"):
                db.add_speaker_embedding(sid, np.ones(8, dtype=np.float32), str(sources[0]), 5.0)
            return sid

        def enroll_from_segment_turns(self, db, name, seg_path, turns, **kw):
            self.calls.append(("label-force", kw.get("force", False)))
            if not kw.get("force"):
                raise SampleMismatch(name, 0.2, 0.5)
            sid = db.upsert_speaker(name)
            db.add_speaker_embedding(sid, np.ones(8, dtype=np.float32), None, 2.0)
            return sid

    stub_module(monkeypatch, "audio_note.speech.speakers", SpeakerEngine=ForceEngine)
    wav = write_wav(tmp_path / "s.wav")
    assert cli.main(["speakers", "enroll", "甲", str(wav)]) == 0
    assert "1 个文件没有加入" in out.getvalue() and "--force" in out.getvalue()
    assert cli.main(["speakers", "enroll", "甲", str(wav), "--force"]) == 0
    assert ForceEngine.instances[-1].calls[-1] == ("enroll-force", True) and ForceEngine.instances[0].calls[0] == ("enroll-force", False)
    seg_wav = write_wav(tmp_cfg.audio_dir / "seg.wav", 3.0)
    db = DB(tmp_cfg.db_path)
    seg = add_done_segment(db, tmp_cfg, datetime(2026, 10, 31, 8, 0), path=seg_wav)
    out.truncate(0); out.seek(0)
    assert cli.main(["speakers", "label", str(seg), "甲", "--from", "0", "--to", "2"]) == 2
    assert "不像" in out.getvalue() and "--force" in out.getvalue()
    assert cli.main(["speakers", "label", str(seg), "甲", "--from", "0", "--to", "2", "--force"]) == 0
    assert ForceEngine.instances[-1].calls[-1] == ("label-force", True)


# --- v3b: sample origins / ambiguity in `speakers audit`, mic origin of `speakers record` --------------------

def test_speakers_audit_shows_origin_and_fix_keeps_ambiguous_samples(tmp_cfg, out: io.StringIO):
    db = DB(tmp_cfg.db_path)
    rng = np.random.default_rng(0)

    def unit(v: np.ndarray) -> np.ndarray:
        return (v / np.linalg.norm(v)).astype(np.float32)

    def near(base: np.ndarray, seed: int) -> np.ndarray:
        return unit(base + 0.03 * np.random.default_rng(seed).standard_normal(base.shape[0]))

    a, b = unit(rng.standard_normal(192)), unit(rng.standard_normal(192))
    sid = db.upsert_speaker("谁")
    for i in range(3):
        db.add_speaker_embedding(sid, near(a, i), None, 10.0, origin="voice")
    for i in range(2):
        db.add_speaker_embedding(sid, near(b, 10 + i), f"l{i}.wav", 5.0, origin="label")
    lone = db.add_speaker_embedding(sid, unit(rng.standard_normal(192)), None, 5.0, origin="cluster")
    assert cli.main(["speakers", "audit"]) == 0
    text = out.getvalue()
    assert "来源" in text and "声音组" in text and "标记" in text and "簇" in text
    assert "分成两组互不相像" in text and "歧义" in text and "1 个样本和其他样本不像" in text
    out.truncate(0), out.seek(0)
    assert cli.main(["speakers", "audit", "--fix"]) == 0
    assert "已删除 1 个离群样本" in out.getvalue()
    assert db.one("SELECT COUNT(*) AS n FROM speaker_embeddings")["n"] == 5          # the two factions stay
    assert db.one("SELECT id FROM speaker_embeddings WHERE id=?", (lone,)) is None


def test_speakers_record_stores_mic_origin(tmp_cfg, record_stubs, out: io.StringIO):
    FakeMic.blocks = [sine(1.0) for _ in range(7)]
    assert cli.main(["speakers", "record", "我", "--seconds", "5"]) == 0
    assert DB(tmp_cfg.db_path).one("SELECT origin FROM speaker_embeddings")["origin"] == "mic"
    with pytest.raises(SystemExit) as ex:
        cli.main(["loudness", "--help"])
    assert ex.value.code == 0



# --- speakers reject / negatives (v4) ------------------------------------------------------------------------

def test_speakers_reject_and_negatives(tmp_cfg, monkeypatch: pytest.MonkeyPatch, out: io.StringIO):
    db = DB(tmp_cfg.db_path)
    now = datetime(2026, 11, 4, 13, 17, 5)
    seg = add_done_segment(db, tmp_cfg, now)                     # no audio file: veto only
    jia = db.upsert_speaker("甲")
    yi = db.upsert_speaker("乙")
    u1, u2 = db.add_utterances([
        {"segment_id": seg, "start_s": 0, "end_s": 2, "abs_start": now.isoformat(), "day": "2026-11-04", "cluster_label": 0,
         "speaker_id": jia, "speaker_name": "甲", "text_raw": "不是我说的", "text": "不是我说的"},
        {"segment_id": seg, "start_s": 2, "end_s": 4, "abs_start": (now + timedelta(seconds=2)).isoformat(), "day": "2026-11-04",
         "cluster_label": 0, "speaker_id": jia, "speaker_name": "甲", "text_raw": "也不是", "text": "也不是"}])
    monkeypatch.setattr(cli, "_optional_engine", lambda cfg: None)
    assert cli.main(["speakers", "reject", "--at", "2026-11-04 13:17:05"]) == 0
    assert "已标为不是 甲" in out.getvalue() and "只记下了否决" in out.getvalue()
    assert db.one("SELECT speaker_id FROM utterances WHERE id=?", (u1,))["speaker_id"] is None
    assert db.one("SELECT speaker_id FROM utterances WHERE id=?", (u2,))["speaker_id"] == jia
    assert cli.main(["speakers", "reject", "--at", "2026-11-04 13:17:07", "--name", "乙", "--cluster"]) == 0
    assert db.rejected_speakers([u1, u2]) == {u1: {jia}, u2: {yi}}         # --cluster only adds sentences that are 乙 now
    assert cli.main(["speakers", "reject", "--at", "2026-11-04 13:17:05"]) == 2   # nobody to reject any more
    assert "--name" in out.getvalue()
    assert cli.main(["speakers", "reject", "--at", "2026-11-04 13:17:05", "--name", "丙"]) == 2
    out.truncate(0); out.seek(0)
    assert cli.main(["speakers", "negatives"]) == 0 and "没有负样本" in out.getvalue()
    nid = db.add_speaker_negative(jia, np.ones(8, dtype=np.float32), "x.wav#0.0-2.0", 2.0, utterance_id=u1, segment_id=seg)
    out.truncate(0); out.seek(0)
    assert cli.main(["speakers", "negatives", "甲"]) == 0
    text = out.getvalue()
    assert "x.wav" in text and "不是我说的" in text and "2.0s" in text
    assert cli.main(["speakers", "negatives", "--delete", str(nid)]) == 0 and db.speaker_negatives() == []
    assert cli.main(["speakers", "negatives", "--delete", str(nid)]) == 2
