"""Always-on service: recorder thread + scheduler thread, single instance, optional tray icon.

    audio-note run [--tray]   ->  run_service(cfg, tray=...)

Single instance: <data_dir>/service.lock holds the pid of the running service, the time it was
written and the process creation time (kernel32 GetProcessTimes on Windows, /proc stat elsewhere).
On start the pid is checked for liveness (ctypes OpenProcess on Windows, `tasklist` as fallback,
os.kill(pid, 0) elsewhere) AND its creation time is compared with the recorded one, so a pid that
was recycled by an unrelated process after a crash or reboot does not count as a running
instance. A stale lock is overwritten.

Ctrl+C / SIGTERM set the shared stop_event; threads are joined with a timeout. The pystray tray
icon (pillow-drawn) is optional: any failure there is logged and the service keeps running.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Config
from .db import DB

log = logging.getLogger(__name__)

LOCK_NAME = "service.lock"
RECORDER_RESTART_DELAY_S = 30.0
JOIN_TIMEOUT_S = 15.0


# --- process liveness / lock file -------------------------------------------

def pid_alive(pid: int) -> bool:
    """Best-effort check whether a process with this pid exists (no psutil)."""
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.OpenProcess.restype = wintypes.HANDLE
            k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            ERROR_ACCESS_DENIED = 5
            h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not h:
                return ctypes.get_last_error() == ERROR_ACCESS_DENIED  # exists but not ours
            try:
                code = wintypes.DWORD()
                k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
                if k32.GetExitCodeProcess(h, ctypes.byref(code)):
                    return code.value == STILL_ACTIVE
                return True
            finally:
                k32.CloseHandle(h)
        except Exception as e:  # noqa: BLE001
            log.debug("ctypes liveness check failed (%s); using tasklist", e)
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                                 capture_output=True, text=True, timeout=10, check=False).stdout
            return f'"{pid}"' in out
        except Exception as e:  # noqa: BLE001
            log.debug("tasklist failed: %s", e)
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def process_start_time(pid: int) -> float | None:
    """Creation time of a process as a POSIX timestamp (None when unknown). Windows: GetProcessTimes;
    Linux: /proc/<pid>/stat starttime; otherwise None. A pid plus its start time identifies a process
    uniquely even after the pid number has been reused."""
    if pid <= 0:
        return None
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.OpenProcess.restype = wintypes.HANDLE
            k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not h:
                return None
            try:
                creation, exit_, kernel, user = (wintypes.FILETIME() for _ in range(4))
                k32.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
                if not k32.GetProcessTimes(h, ctypes.byref(creation), ctypes.byref(exit_), ctypes.byref(kernel), ctypes.byref(user)):
                    return None
                ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime   # 100 ns since 1601-01-01
                return ticks / 1e7 - 11644473600.0
            finally:
                k32.CloseHandle(h)
        except Exception as e:  # noqa: BLE001
            log.debug("GetProcessTimes failed for pid %d: %s", pid, e)
            return None
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        start_ticks = int(fields[19])
        with open("/proc/uptime", "r", encoding="utf-8") as f:
            uptime = float(f.read().split()[0])
        hz = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        return time.time() - uptime + start_ticks / float(hz)
    except Exception:  # noqa: BLE001
        return None


def lock_path(cfg: Config) -> Path:
    return cfg.data_dir / LOCK_NAME


def read_lock(cfg: Config) -> int | None:
    """pid stored in the lock file, or None when absent/unreadable."""
    return read_lock_info(cfg)[0]


def read_lock_info(cfg: Config) -> tuple[int | None, float | None]:
    """(pid, process creation time) from the lock file; either may be None."""
    p = lock_path(cfg)
    try:
        lines = p.read_text(encoding="utf-8").split()
    except OSError:
        return None, None
    pid = int(lines[0]) if lines and lines[0].isdigit() else None
    ctime: float | None = None
    if len(lines) >= 3:
        try:
            ctime = float(lines[2])
        except ValueError:
            ctime = None
    return pid, ctime


def _same_process(pid: int, recorded_ctime: float | None) -> bool:
    """True when `pid` is alive and (if both are known) its creation time matches the recorded one."""
    if not pid_alive(pid):
        return False
    if recorded_ctime is None:
        return True   # old-format lock: cannot tell, assume it is the same process
    actual = process_start_time(pid)
    if actual is None:
        return True
    return abs(actual - recorded_ctime) < 2.0


def running_pid(cfg: Config) -> int | None:
    """pid of a live service instance (lock present, process alive and not a recycled pid), else None."""
    pid, ctime = read_lock_info(cfg)
    if pid is not None and _same_process(pid, ctime):
        return pid
    return None


class AlreadyRunning(RuntimeError):
    def __init__(self, pid: int):
        super().__init__(f"audio-note 服务已在运行 (pid {pid})")
        self.pid = pid


def acquire_lock(cfg: Config) -> Path:
    """Write our pid into the lock file; raise AlreadyRunning when another live instance holds it."""
    p = lock_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    other, other_ctime = read_lock_info(cfg)
    if other is not None and other != os.getpid() and _same_process(other, other_ctime):
        raise AlreadyRunning(other)
    if other is not None:
        log.info("removing stale lock (pid %d not running or recycled)", other)
    mine = process_start_time(os.getpid())
    p.write_text(f"{os.getpid()}\n{datetime.now().isoformat(timespec='seconds')}\n{mine if mine is not None else ''}\n",
                 encoding="utf-8")
    return p


def release_lock(cfg: Config) -> None:
    p = lock_path(cfg)
    try:
        if read_lock(cfg) == os.getpid():
            p.unlink()
    except OSError as e:
        log.debug("release lock failed: %s", e)


# --- tray icon ----------------------------------------------------------------

def _make_icon_image(size: int = 64, color: tuple[int, int, int] = (220, 60, 60)):
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    m = size // 8
    # microphone capsule + stand
    d.rounded_rectangle((size * 3 // 8, m, size * 5 // 8, size // 2 + m), radius=size // 8, fill=color)
    d.arc((size // 4, size // 4, size * 3 // 4, size * 3 // 4 + m), start=0, end=180, fill=color, width=max(2, size // 16))
    d.line((size // 2, size * 3 // 4 + m // 2, size // 2, size - m), fill=color, width=max(2, size // 16))
    d.line((size * 3 // 8, size - m, size * 5 // 8, size - m), fill=color, width=max(2, size // 16))
    return img


def _status_text(cfg: Config, db: DB, recorder: Any) -> str:
    rows = db.query("SELECT status, COUNT(*) AS n FROM segments GROUP BY status")
    counts = ", ".join(f"{r['status']}={r['n']}" for r in rows) or "无录音段"
    today = datetime.now().strftime("%Y-%m-%d")
    utt = db.one("SELECT COUNT(*) AS n FROM utterances WHERE day=?", (today,))
    parts = [f"段落: {counts}", f"今日语句: {utt['n'] if utt else 0}"]
    if recorder is not None:
        parts.append(f"本次录音: {getattr(recorder, 'segments_written', 0)} 段 / {getattr(recorder, 'speech_total_s', 0.0):.0f}s 语音")
        parts.append("正在说话" if getattr(recorder, "is_speaking", False) else "静音")
    return "\n".join(parts)


def start_tray(cfg: Config, db: DB, stop_event: threading.Event, recorder_ref: dict[str, Any],
               ui_url: str | None = None, runtime_getter: Any = None) -> Any | None:
    """Start a pystray icon in a daemon thread. Returns the icon (or None when unavailable).
    `runtime_getter` (optional) returns the current Runtime so the status reflects a switched profile."""
    try:
        import pystray
    except Exception as e:  # noqa: BLE001
        log.warning("tray unavailable (pystray import failed: %s)", e)
        return None
    try:
        image = _make_icon_image()

        def on_status(icon, item):  # noqa: ANN001
            try:
                if runtime_getter is not None:
                    cur = runtime_getter()
                    text = _status_text(cur.cfg, cur.db, cur.recorder)
                else:
                    text = _status_text(cfg, db, recorder_ref.get("recorder"))
                log.info("status:\n%s", text)
                if getattr(icon, "HAS_NOTIFICATION", False):
                    icon.notify(text, "audio-note 状态")
            except Exception as e:  # noqa: BLE001
                log.warning("tray status failed: %s", e)

        def on_quit(icon, item):  # noqa: ANN001
            log.info("quit requested from tray")
            stop_event.set()
            try:
                icon.stop()
            except Exception:  # noqa: BLE001
                pass

        items = []
        if ui_url:
            import webbrowser

            items.append(pystray.MenuItem("打开界面", lambda icon, item: webbrowser.open(ui_url), default=True))
        items += [pystray.MenuItem("Status", on_status), pystray.MenuItem("Quit", on_quit)]
        menu = pystray.Menu(*items)
        icon = pystray.Icon("audio-note", image, "audio-note 录音中", menu)

        def runner() -> None:
            try:
                icon.run()
            except Exception as e:  # noqa: BLE001
                log.warning("tray icon stopped: %s", e)

        threading.Thread(target=runner, name="tray", daemon=True).start()
        log.info("tray icon started")
        return icon
    except Exception as e:  # noqa: BLE001
        log.warning("tray icon failed: %s", e)
        return None


# --- threads ------------------------------------------------------------------

def _recorder_loop(cfg: Config, db: DB, stop_event: threading.Event, ref: dict[str, Any]) -> None:
    """Run the Recorder; on crash, wait and restart until stop_event is set.

    `ref` carries state both ways: the loop stores the live "recorder" and the "last_error"; the Runtime may
    put an "events" bus and an "on_segment" callback into it, which every Recorder built here receives
    (events for the page, the callback to wake the scheduler when a segment file is closed)."""
    from .audio.recorder import Recorder

    restarts = 0
    while not stop_event.is_set():
        try:
            if restarts:
                # PortAudio's device table is a snapshot from process start: without a refresh a mic that
                # was unplugged/replugged or a headset that became the default is never found again
                from .audio.capture import refresh_devices

                refresh_devices()
            restarts += 1
            rec = Recorder(cfg, db, stop_event)
            for key in ("events", "on_segment"):
                if ref.get(key) is not None:
                    try:
                        setattr(rec, key, ref[key])
                    except Exception:  # noqa: BLE001 - a stand-in recorder that takes no attributes
                        pass
            ref["recorder"] = rec
            rec.run()
            if not stop_event.is_set():
                log.warning("recorder exited unexpectedly; restarting in %.0fs", RECORDER_RESTART_DELAY_S)
        except Exception as e:  # noqa: BLE001
            ref["last_error"] = repr(e)
            log.exception("recorder crashed: %s; restarting in %.0fs", e, RECORDER_RESTART_DELAY_S)
        if stop_event.wait(RECORDER_RESTART_DELAY_S):
            break
    log.info("recorder thread finished")


def _install_signal_handlers(stop_event: threading.Event) -> None:
    def handler(signum, frame):  # noqa: ANN001
        log.info("signal %s received; stopping", signum)
        stop_event.set()

    for name in ("SIGTERM", "SIGBREAK", "SIGINT"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            if name == "SIGINT":
                continue  # keep KeyboardInterrupt semantics for Ctrl+C
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass  # not main thread / unsupported


def _apply_priority(cfg: Config) -> None:
    """performance.background_priority: run this (long-lived) process below normal priority. Never raises."""
    try:
        from . import governor

        governor.apply_process_priority(cfg)
    except Exception as e:  # noqa: BLE001
        log.debug("process priority not changed: %s", e)


def start_ui(rt: Any, open_browser: bool = False) -> Any | None:
    """Serve the web UI for a Runtime (or a RuntimeHolder, so the page can switch profiles) on a
    background thread. Returns the UIServer, or None when it could not start (port taken, missing
    dependency): the service keeps running without it."""
    try:
        from .runtime import RuntimeHolder
        from .ui.app import create_app
        from .ui.server import UIServer

        holder = rt if isinstance(rt, RuntimeHolder) else RuntimeHolder(rt)
        rt = holder.rt
        _apply_priority(rt.cfg)      # a process that serves the UI is the long-running one (`run` and `ui` both get here)
        srv = UIServer(create_app(holder), str(rt.cfg.ui.host or "127.0.0.1"), int(rt.cfg.ui.port or 8765))
        rt.ui_url = srv.start()
        if open_browser:
            srv.open_browser()
        return srv
    except Exception as e:  # noqa: BLE001
        log.error("web UI not started: %s", e)
        return None


def run_service(cfg: Config, tray: bool = False, stop_event: threading.Event | None = None,
                record: bool = True, schedule: bool = True, ui: bool | None = None,
                open_browser: bool | None = None) -> int:
    """Blocking service loop (`audio-note run`). Returns a process exit code.

    Built on runtime.Runtime: lock + scheduler (+ recorder) threads, plus the web UI when
    `ui` (default cfg.ui.enabled) is on. The recorder can be stopped and restarted from the UI
    while the process keeps running; the loop only gives up when the service's own threads all
    died while the service is still supposed to run."""
    from .runtime import Runtime, RuntimeHolder

    stop_event = stop_event or threading.Event()
    cfg.ensure_dirs()
    _apply_priority(cfg)
    rt = Runtime(cfg)
    try:
        rt.start_service(record=record, schedule=schedule)
    except AlreadyRunning as e:
        log.error("%s", e)
        print(str(e), file=sys.stderr)
        rt.shutdown()
        return 1

    holder = RuntimeHolder(rt)   # the UI may replace holder.rt when the user switches profiles
    icon = None
    server = None
    try:
        _install_signal_handlers(stop_event)
        ui_enabled = bool(cfg.ui.enabled) if ui is None else bool(ui)
        if ui_enabled:
            server = start_ui(holder, open_browser=bool(cfg.ui.open_browser) if open_browser is None else bool(open_browser))
        if tray:
            icon = start_tray(cfg, rt.db, stop_event, rt._rec_ref, ui_url=rt.ui_url, runtime_getter=lambda: holder.rt)
        log.info("service started (pid %d)%s; Ctrl+C to stop", os.getpid(), f", UI at {rt.ui_url}" if rt.ui_url else "")
        try:
            while not stop_event.is_set():
                stop_event.wait(1.0)
                cur = holder.rt
                if (record or schedule) and cur.service_running() and not any(t.is_alive() for t in cur.threads()):
                    log.error("all worker threads exited; stopping")
                    break
        except KeyboardInterrupt:
            log.info("Ctrl+C received; stopping")
        stop_event.set()
        holder.rt.stop_service()
        if icon is not None:
            try:
                icon.stop()
            except Exception:  # noqa: BLE001
                pass
        log.info("service stopped")
        return 0
    finally:
        stop_event.set()
        if server is not None:
            server.stop()
        holder.rt.shutdown()
