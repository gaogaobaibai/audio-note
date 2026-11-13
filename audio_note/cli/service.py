"""Long-running commands and devices: `devices`, `record`, `run`, `ui`.

`record`, `run` and `ui` are the service commands (see `logging_setup.SERVICE_COMMANDS`): they own the
main log file, every other command logs to the separate CLI log.
"""
from __future__ import annotations

import argparse
import logging
import threading
from typing import Any

from rich.table import Table

from ..config import Config
from ._late import Late, handler
from .helpers import CLIError, _db

log = logging.getLogger("audio_note.cli")
console = Late("console")


def cmd_devices(cfg: Config, args: argparse.Namespace) -> int:
    from ..audio.capture import list_input_devices

    devices = list_input_devices()
    t = Table(title="音频输入设备")
    t.add_column("#", justify="right")
    t.add_column("名称")
    t.add_column("Host API")
    t.add_column("声道", justify="right")
    t.add_column("默认采样率", justify="right")
    for d in devices:
        t.add_row(str(d.index), d.name, d.hostapi, str(d.channels), f"{d.default_sr:.0f}")
    console.print(t)
    console.print(f"配置 audio.device = {cfg.audio.device!r}（null = 系统默认）")
    from ..audio.loopback import list_loopback_devices

    lbs = list_loopback_devices()
    if lbs:
        t2 = Table(title="系统声音源（录会议 / 通话对方的声音：audio.system_audio: true）")
        t2.add_column("#", justify="right")
        t2.add_column("输出设备")
        t2.add_column("声道", justify="right")
        t2.add_column("采样率", justify="right")
        t2.add_column("默认")
        for d in lbs:
            t2.add_row(str(d.index), d.name, str(d.channels), f"{d.default_sr:.0f}", "✓" if d.is_default else "")
        console.print(t2)
        console.print(f"配置 audio.system_audio = {bool(cfg.audio.get('system_audio', False))}，"
                      f"audio.system_device = {cfg.audio.get('system_device')!r}（null = 默认输出设备）")
    else:
        console.print("[yellow]系统声音源不可用（需要 Windows 和 PyAudioWPatch：uv pip install PyAudioWPatch）[/yellow]")
    return 0


def _wait_with_ctrl_c(stop_event: threading.Event, threads: list[threading.Thread]) -> None:
    try:
        while not stop_event.is_set() and any(t.is_alive() for t in threads):
            stop_event.wait(0.5)
    except KeyboardInterrupt:
        console.print("\n正在停止…")
    stop_event.set()
    for t in threads:
        t.join(15.0)


def cmd_record(cfg: Config, args: argparse.Namespace) -> int:
    from ..audio.recorder import Recorder

    db = _db(cfg)
    stop_event = threading.Event()
    rec = Recorder(cfg, db, stop_event)
    errors: list[BaseException] = []

    def target() -> None:
        try:
            rec.run()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
            log.exception("recorder failed: %s", e)
            stop_event.set()

    t = threading.Thread(target=target, name="recorder", daemon=True)
    t.start()
    console.print("录音中（只保存检测到语音的片段），Ctrl+C 停止。")
    _wait_with_ctrl_c(stop_event, [t])
    console.print(f"本次写入 {rec.segments_written} 段，共 {rec.speech_total_s:.0f}s 语音。")
    return 1 if errors else 0


def cmd_run(cfg: Config, args: argparse.Namespace) -> int:
    from ..service import run_service

    if args.port:
        cfg["ui"]["port"] = int(args.port)
    return run_service(cfg, tray=bool(args.tray), ui=False if args.no_ui else None, open_browser=True if args.open else None)


def cmd_ui(cfg: Config, args: argparse.Namespace) -> int:
    """Web UI without recording (recording can be started from the page). When another process already
    runs the service, the page works read-only (browse + chat)."""
    from ..runtime import Runtime, RuntimeHolder, profile_label
    from ..service import AlreadyRunning, _install_signal_handlers, start_ui

    if args.port:
        cfg["ui"]["port"] = int(args.port)
    rt = Runtime(cfg)
    try:
        rt.start_service(record=bool(args.record))
    except AlreadyRunning as e:
        console.print(f"[yellow]{e}[/yellow]\n界面以只读模式启动：可以看记录和问答，录音和处理请在那个进程的界面操作。")
    holder = RuntimeHolder(rt)
    srv = start_ui(holder, open_browser=not args.no_browser)
    if srv is None:
        rt.shutdown()
        raise CLIError("界面启动失败（端口被占用？看上面的日志）")
    console.print(f"界面地址 [bold]{srv.url}[/bold]  档案：{profile_label(rt.profile)}"
                  f"  （Ctrl+C 退出{'' if args.record else '；录音可在页面上开始'}）")
    stop = threading.Event()
    _install_signal_handlers(stop)
    try:
        while not stop.is_set():
            stop.wait(0.5)
    except KeyboardInterrupt:
        console.print("\n正在停止…")
    srv.stop()
    holder.rt.shutdown()
    return 0


def register(sub: Any) -> None:
    sub.add_parser("devices", help="列出音频输入设备").set_defaults(func=handler(cmd_devices))
    sub.add_parser("record", help="只录音（VAD 门控，只存语音段）").set_defaults(func=handler(cmd_record))
    s = sub.add_parser("run", help="常驻服务：录音 + 定时处理/清理/日报 + 网页界面")
    s.add_argument("--tray", action="store_true", help="显示系统托盘图标")
    s.add_argument("--no-ui", action="store_true", help="不提供网页界面")
    s.add_argument("--open", action="store_true", help="启动后在浏览器打开界面")
    s.add_argument("--port", type=int, default=None, help="界面端口（默认 ui.port = 8765）")
    s.set_defaults(func=handler(cmd_run))
    s = sub.add_parser("ui", help="打开网页界面（不自动录音；录音可在页面上开始）")
    s.add_argument("--record", action="store_true", help="启动时就开始录音（等于 run --open）")
    s.add_argument("--port", type=int, default=None, help="界面端口（默认 ui.port = 8765）")
    s.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    s.set_defaults(func=handler(cmd_ui))
