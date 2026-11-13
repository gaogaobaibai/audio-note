"""Shared plumbing of the bench scripts: locating the code under test, the API key (never printed, never
written), read-only database access, JSON I/O and a small exclusive stage timer.

Which ``audio_note`` gets imported (the code under test):
    1. ``AUDIO_NOTE_SRC`` (a checkout root), when set;
    2. else a ``PYTHONPATH`` entry that holds an ``audio_note`` package (how the pre-upgrade baseline is run:
       cwd and PYTHONPATH point at a detached checkout of the old commit);
    3. else the repository this file lives in.
``code_info()`` records which one it was in every result file.

The MiMo key: ``MIMO_API_KEY`` from the environment, else read in memory from the YAML named by ``--key-config`` /
``BENCH_KEY_CONFIG`` and exported to this process only. A bench profile's own config.yaml never holds a key.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SR = 16000


# ------------------------------------------------------------------------------------------------ locations
def default_root() -> Path:
    """Where caches, rendered audio and bench profiles live: $BENCH_ROOT, else <repo>/tmp_bench (git-ignored)."""
    env = os.environ.get("BENCH_ROOT")
    return Path(env) if env else REPO_ROOT / "tmp_bench"


def profile_dir(root: Path | str, name: str) -> Path:
    return Path(root) / "profiles" / name


def resolve_profile(profile: str | os.PathLike, root: Path | str | None = None) -> Path:
    """A bench profile directory from a path or a bare name under <root>/profiles."""
    p = Path(profile)
    if (p / "config.yaml").exists() or p.is_absolute():
        return p
    return profile_dir(root or default_root(), str(profile))


def profile_db_path(pdir: Path | str) -> Path:
    return Path(pdir) / "data" / "audio_note.sqlite3"


# ------------------------------------------------------------------------------------------------ code under test
def _pythonpath_src() -> Path | None:
    for entry in (os.environ.get("PYTHONPATH") or "").split(os.pathsep):
        if entry and (Path(entry) / "audio_note" / "__init__.py").exists():
            return Path(entry)
    return None


def use_project(src: str | os.PathLike | None = None) -> Path:
    """Make ``import audio_note`` resolve to the code under test; returns that checkout's root."""
    chosen = Path(src) if src else None
    if chosen is None and os.environ.get("AUDIO_NOTE_SRC"):
        chosen = Path(os.environ["AUDIO_NOTE_SRC"])
    if chosen is None:
        chosen = _pythonpath_src() or REPO_ROOT
    chosen = chosen.resolve()
    if "audio_note" in sys.modules:
        loaded = Path(sys.modules["audio_note"].__file__).resolve().parent.parent
        if loaded != chosen:
            raise RuntimeError(f"audio_note already imported from {loaded}, wanted {chosen}")
        return loaded
    s = str(chosen)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)
    return chosen


def _git_head(root: Path) -> str | None:
    """Commit hash of a checkout / worktree read from the .git files (no git process)."""
    try:
        git = root / ".git"
        if git.is_file():
            txt = git.read_text(encoding="utf-8").strip()
            if not txt.startswith("gitdir:"):
                return None
            git = Path(txt.split(":", 1)[1].strip())
            if not git.is_absolute():
                git = (root / git).resolve()
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head[:12]
        ref = head.split(":", 1)[1].strip()
        common = git
        cd = git / "commondir"
        if cd.exists():
            common = (git / cd.read_text(encoding="utf-8").strip()).resolve()
        for base in (git, common):
            f = base / ref
            if f.exists():
                return f.read_text(encoding="utf-8").strip()[:12]
        packed = common / "packed-refs"
        if packed.exists():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(" " + ref):
                    return line.split(" ", 1)[0][:12]
    except OSError:
        return None
    return None


def code_info() -> dict[str, Any]:
    """Which audio_note is loaded: checkout root, version, commit (best effort)."""
    use_project()
    import audio_note

    root = Path(audio_note.__file__).resolve().parent.parent
    return {"src": str(root), "version": getattr(audio_note, "__version__", None), "commit": _git_head(root)}


# ------------------------------------------------------------------------------------------------ API key
def load_key(key_config: str | os.PathLike | None = None) -> dict[str, Any]:
    """Make MIMO_API_KEY available to this process and return the NON-secret mimo settings worth copying into
    a bench profile ({"base_url": ...} when the key file names one). The key itself is never returned."""
    extra: dict[str, Any] = {}
    path = key_config or os.environ.get("BENCH_KEY_CONFIG")
    if path:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        mimo = raw.get("mimo") if isinstance(raw, dict) else None
        if isinstance(mimo, dict):
            if mimo.get("base_url"):
                extra["base_url"] = str(mimo["base_url"])
            if not os.environ.get("MIMO_API_KEY") and mimo.get("api_key"):
                os.environ["MIMO_API_KEY"] = str(mimo["api_key"])
    if not os.environ.get("MIMO_API_KEY"):
        raise SystemExit("no MiMo API key: set MIMO_API_KEY, or pass --key-config <config.yaml> / BENCH_KEY_CONFIG")
    return extra


def scrub(text: str) -> str:
    """Remove the API key from a message before it is logged."""
    key = os.environ.get("MIMO_API_KEY")
    return text.replace(key, "***") if key else text


# ------------------------------------------------------------------------------------------------ files
def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return p


def load_truth(pdir_or_file: Path | str) -> dict[str, Any]:
    """truth.json of a bench profile directory (or the file itself)."""
    p = Path(pdir_or_file)
    if p.is_dir():
        p = p / "truth.json"
    return read_json(p)


# ------------------------------------------------------------------------------------------------ sqlite (read only)
def open_ro(db_path: Path | str) -> sqlite3.Connection:
    """Read-only connection: a bench must never migrate or modify the database it scores."""
    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(p)
    uri = "file:" + p.as_posix().replace("?", "%3f").replace("#", "%23")
    conn = sqlite3.connect(uri + "?mode=ro", uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
    except sqlite3.OperationalError:
        conn.close()
        conn = sqlite3.connect(uri + "?mode=ro&immutable=1", uri=True, timeout=10.0)
        conn.row_factory = sqlite3.Row
    return conn


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?", (name,)).fetchone() is not None


def columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


# ------------------------------------------------------------------------------------------------ small maths
def rate(n: float, d: float) -> float | None:
    return round(n / d, 4) if d else None


def pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{100 * x:.{digits}f}%"


# ------------------------------------------------------------------------------------------------ timing
class StageTimer:
    """Exclusive wall and CPU seconds per named stage. Stages nest (per thread): the time of an inner stage is
    subtracted from the stage around it, so ``identify`` does not include the embeddings it asks for.
    CPU is the whole process (time.process_time), which is what the owner's machine pays."""

    def __init__(self) -> None:
        self.totals: dict[str, dict[str, float]] = {}
        self._local = threading.local()
        self._lock = threading.Lock()

    def _stack(self) -> list[list[float]]:
        st = getattr(self._local, "stack", None)
        if st is None:
            st = self._local.stack = []
        return st

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        st = self._stack()
        frame = [0.0, 0.0]            # wall / cpu spent in nested stages
        st.append(frame)
        w0, c0 = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            w, c = time.perf_counter() - w0, time.process_time() - c0
            st.pop()
            if st:
                st[-1][0] += w
                st[-1][1] += c
            with self._lock:
                t = self.totals.setdefault(name, {"n": 0, "wall_s": 0.0, "cpu_s": 0.0})
                t["n"] += 1
                t["wall_s"] += max(0.0, w - frame[0])
                t["cpu_s"] += max(0.0, c - frame[1])

    def wrap(self, obj: Any, attr: str, name: str) -> bool:
        """Replace ``obj.attr`` (a bound method / function) by a timed version. False when it is absent."""
        fn = getattr(obj, attr, None)
        if not callable(fn):
            return False

        def timed(*a: Any, **kw: Any) -> Any:
            with self.stage(name):
                return fn(*a, **kw)

        timed.__name__ = getattr(fn, "__name__", attr)
        timed.__wrapped__ = fn          # inspect.signature() follows it: signature probing still works
        try:
            setattr(obj, attr, timed)
        except (AttributeError, TypeError):
            return False
        return True

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {k: {"n": int(v["n"]), "wall_s": round(v["wall_s"], 4), "cpu_s": round(v["cpu_s"], 4)}
                    for k, v in sorted(self.totals.items())}


#: engine attribute -> stage name (identify = the matching logic around the embeddings)
ENGINE_STAGES: dict[str, str] = {
    "diarize": "diarize",
    "embed": "embed",
    "identify_turns": "identify",
    "identify_utterances": "identify",
    "cluster_embedding": "identify",
    "match": "identify",
}


def time_engine(engine: Any, timer: StageTimer, stages: dict[str, str] | None = None) -> list[str]:
    """Wrap the speaker engine's stage methods; returns the attribute names that were wrapped."""
    return [a for a, n in (stages or ENGINE_STAGES).items() if timer.wrap(engine, a, n)]


def call_supported(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` with only the keyword arguments its signature accepts (duck-typed seams differ by version)."""
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    return fn(*args, **{k: v for k, v in kwargs.items() if k in params})
