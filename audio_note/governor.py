"""Resource governor: how much of the machine the local models may use, and when to leave it alone.

    threads(cfg)                 -> onnxruntime threads for the speaker models
    apply_process_priority(cfg)  -> run the whole process below normal priority (Windows)
    boost_current_thread()       -> called by the recorder thread: back to a normal level inside that process
    on_battery()                 -> True / False / None (unknown)
    defer_reason(cfg)            -> why heavy local work should wait right now (a short Chinese phrase), or None

Everything is plain ctypes against kernel32 and fails silently: on another platform, or when a call is
refused, the functions are no-ops (False / None). Nothing here imports a third-party package.

Measured on the owner's laptop (SPEC_v1 section 1): diarization with 4 threads is only 1.6x faster than with 1
but burns 2.6x the CPU seconds, hence `balanced` = 2 threads and `eco` = 1.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

log = logging.getLogger(__name__)

#: performance.mode -> threads of the local models
MODE_THREADS = {"eco": 1, "balanced": 2, "fast": 4}
DEFAULT_MODE = "balanced"
MAX_THREADS = 64

BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
NORMAL_PRIORITY_CLASS = 0x00000020
THREAD_PRIORITY_NORMAL = 0
THREAD_PRIORITY_HIGHEST = 2      # +2 relative to the process class

BATTERY_REASON = "正在用电池供电，本机处理已推迟（插上电源后自动继续）"


def _section(cfg: Any, name: str) -> dict:
    try:
        sec = cfg.get(name) if hasattr(cfg, "get") else None
    except Exception:  # noqa: BLE001
        sec = None
    return sec if isinstance(sec, dict) else {}


def _int(v: Any) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def mode(cfg: Any) -> str:
    """performance.mode, normalised; an unknown value counts as 'balanced'."""
    m = str(_section(cfg, "performance").get("mode") or DEFAULT_MODE).strip().lower()
    return m if m in MODE_THREADS else DEFAULT_MODE


def threads(cfg: Any) -> int:
    """Threads for the local speaker models: speakers.num_threads when it is set (non-zero), else
    performance.threads, else the number that goes with performance.mode (eco 1 / balanced 2 / fast 4)."""
    n = _int(_section(cfg, "speakers").get("num_threads"))
    if n <= 0:
        n = _int(_section(cfg, "performance").get("threads"))
    if n <= 0:
        n = MODE_THREADS[mode(cfg)]
    return max(1, min(MAX_THREADS, n))


# --- Windows plumbing ---------------------------------------------------------------------------------------

def _kernel32() -> Any | None:
    """kernel32 with the prototypes this module needs, or None off Windows / when ctypes cannot load it."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.GetCurrentProcess.argtypes = []
        k.GetCurrentThread.restype = wintypes.HANDLE
        k.GetCurrentThread.argtypes = []
        k.SetPriorityClass.restype = wintypes.BOOL
        k.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k.SetThreadPriority.restype = wintypes.BOOL
        k.SetThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
        k.GetSystemPowerStatus.restype = wintypes.BOOL
        k.GetSystemPowerStatus.argtypes = [ctypes.c_void_p]
        return k
    except Exception as e:  # noqa: BLE001
        log.debug("kernel32 unavailable: %s", e)
        return None


def apply_process_priority(cfg: Any) -> bool:
    """When performance.background_priority is on: put this process into the BELOW_NORMAL priority class so
    transcription never competes with what the user is doing. Returns True when the class was set.
    Only the long-running service calls this (service.run_service / start_ui), never a one-off command."""
    if not bool(_section(cfg, "performance").get("background_priority", False)):
        return False
    k = _kernel32()
    if k is None:
        return False
    try:
        ok = bool(k.SetPriorityClass(k.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS))
    except Exception as e:  # noqa: BLE001
        log.debug("SetPriorityClass failed: %s", e)
        return False
    if ok:
        log.info("process priority: below normal (performance.background_priority)")
    return ok


def _set_thread_priority(level: int) -> bool:
    k = _kernel32()
    if k is None:
        return False
    try:
        return bool(k.SetThreadPriority(k.GetCurrentThread(), int(level)))
    except Exception as e:  # noqa: BLE001
        log.debug("SetThreadPriority failed: %s", e)
        return False


def boost_current_thread() -> None:
    """Raise the calling thread two steps above its process class. The recorder thread calls it: inside a
    BELOW_NORMAL process that puts audio capture back at a normal level, so a busy machine does not make
    it drop blocks."""
    _set_thread_priority(THREAD_PRIORITY_HIGHEST)


def restore_current_thread() -> None:
    """Undo boost_current_thread() (the recorder calls it when its loop ends)."""
    _set_thread_priority(THREAD_PRIORITY_NORMAL)


def on_battery() -> bool | None:
    """True on battery, False on mains, None when unknown (desktop without the API, other platforms)."""
    k = _kernel32()
    if k is None:
        return None
    try:
        import ctypes

        class _PowerStatus(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

        st = _PowerStatus()
        if not k.GetSystemPowerStatus(ctypes.byref(st)):
            return None
        if st.ACLineStatus == 0:
            return True
        if st.ACLineStatus == 1:
            return False
        return None
    except Exception as e:  # noqa: BLE001
        log.debug("GetSystemPowerStatus failed: %s", e)
        return None


def defer_reason(cfg: Any) -> str | None:
    """Why heavy local work (diarization, voice re-scoring) should not run right now, or None.
    Today the only reason is performance.defer_on_battery while the machine runs on battery."""
    if not bool(_section(cfg, "performance").get("defer_on_battery", False)):
        return None
    try:
        return BATTERY_REASON if on_battery() is True else None
    except Exception:  # noqa: BLE001 - never let a power query stop the pipeline
        return None


__all__ = ["threads", "mode", "apply_process_priority", "boost_current_thread", "restore_current_thread",
           "on_battery", "defer_reason", "MODE_THREADS", "BATTERY_REASON"]
