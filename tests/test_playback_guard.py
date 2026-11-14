"""Self-playback guard: listening to old recordings in the page must not record them again as new speech.

The page's player reports play / stop (POST /api/playback, heartbeat while playing); the recorder leaves the blocks
captured meanwhile out (audio.playback_guard: all = silence, system = drop only the system audio, off). No real
audio devices: scripted fake microphone / loopback, scripted VAD, guards with an injected clock."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from audio_note.audio import playback
from audio_note.audio import recorder as rec_mod
from audio_note.audio.capture import MixedCapture
from audio_note.audio.playback import LEAD_S, TAIL_S, TTL_S, FileWindows, PlaybackGuard, covered, guard_mode
from audio_note.audio.recorder import Recorder
from audio_note.audio.vad import SAMPLE_RATE, WINDOW

from .conftest import FakeVAD, noise

BLOCK_S = WINDOW / SAMPLE_RATE
STATIC = Path(__file__).resolve().parents[1] / "audio_note" / "ui" / "static" / "js"


@pytest.fixture(autouse=True)
def _clean_global_guard():
    playback.GUARD.clear()
    yield
    playback.GUARD.clear()


class Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


# --- the guard ------------------------------------------------------------------------------------------

def test_reported_playback_becomes_a_window_with_lead_and_tail():
    clk = Clock()
    g = PlaybackGuard(clock=clk)
    assert not g.covers(1000.0)
    assert g.report("tab1", True) is True
    clk.t = 1004.0
    g.report("tab1", True)                       # heartbeat
    clk.t = 1007.5
    assert g.report("tab1", False) is False
    assert g.windows() == [(1000.0, 1007.5)]
    assert g.covers(1000.0 - LEAD_S + 0.01) and not g.covers(1000.0 - LEAD_S - 0.01)
    assert g.covers(1007.5 + TAIL_S - 0.01) and not g.covers(1007.5 + TAIL_S + 0.01)
    assert g.covers(1003.0)


def test_a_tab_that_stops_reporting_counts_as_stopped_after_the_ttl():
    clk = Clock()
    g = PlaybackGuard(clock=clk)
    g.report("tab", True)
    clk.t = 1000.0 + TTL_S - 0.5
    assert g.playing() and g.covers(clk.t)       # still within its promise to report again
    clk.t = 1000.0 + TTL_S + 0.5                 # closed / crashed / asleep: no more reports
    assert not g.playing()
    assert g.windows() == [(1000.0, 1000.0 + TTL_S)]
    assert not g.covers(1000.0 + TTL_S + TAIL_S + 0.1)


def test_reports_that_overtake_each_other_cannot_end_a_newer_playback():
    clk = Clock()
    g = PlaybackGuard(clock=clk)
    g.report("tab", True, seq=1)
    clk.t = 1001.0
    g.report("tab", True, seq=3)                 # "play" of the next clip arrived first ...
    assert g.report("tab", False, seq=2) is True  # ... the older "pause" is ignored
    assert g.stale == 1 and g.playing()
    clk.t = 1002.0
    assert g.report("tab", False, seq=4) is False


def test_tabs_are_independent_and_history_is_bounded():
    clk = Clock()
    g = PlaybackGuard(clock=clk)
    g.report("a", True)
    g.report("b", True)
    clk.t = 1001.0
    g.report("a", False)
    assert g.playing()                           # b still plays
    clk.t = 1002.0
    g.report("b", False)
    assert len(g.windows()) == 2
    clk.t = 1002.0 + playback.HISTORY_S + 1
    assert g.windows() == []                     # old windows are forgotten
    for i in range(playback.MAX_TABS + 5):
        g.report(f"t{i}", True)
    assert len(g._open) == playback.MAX_TABS


def test_stop_without_play_and_odd_tab_ids_are_harmless():
    g = PlaybackGuard(clock=Clock())
    assert g.report("never-played", False) is False
    assert g.report(None, True) is True and g.report("", False) is False   # both mean the same anonymous tab
    assert g.report("x" * 500, True) and max(len(k) for k in g._open) == 64


@pytest.mark.parametrize("value,mode", [(None, "all"), ("all", "all"), ("System", "system"), ("off", "off"),
                                        (False, "off"), (True, "all"), ("false", "off"), ("bogus", "all")])
def test_guard_mode_parsing(cfg_factory, value, mode):
    cfg = cfg_factory({"audio": {"playback_guard": value}})
    assert guard_mode(cfg) == mode


def test_default_mode_is_all(cfg_factory):
    assert guard_mode(cfg_factory()) == "all"
    assert guard_mode(SimpleNamespace()) == "all"


def test_windows_mirrored_to_a_file_for_a_recorder_in_another_process(tmp_path):
    p = tmp_path / playback.FILE_NAME
    clk = Clock(50.0)
    fw = FileWindows(p, clock=clk)
    assert fw.windows() == [] and not fw.covers(10.0)        # no file: nothing playing
    playback.write_file(p, [(10.0, 12.0), (20.0, 26.0)])
    clk.t += 1
    assert fw.windows() == [(10.0, 12.0), (20.0, 26.0)] and fw.covers(11.0) and not fw.covers(15.0)
    p.write_text("{not json", encoding="utf-8")
    clk.t += 1
    assert fw.windows() == []                                 # half-written / foreign file: ignored
    p.write_text(json.dumps({"windows": [[1, 2], ["x", 3], [5, 4], [7]]}), encoding="utf-8")
    clk.t += 1
    assert fw.windows() == [(1.0, 2.0)]
    p.unlink()
    clk.t += 1
    assert fw.windows() == []


def test_covered_helper():
    assert covered([(10.0, 11.0)], 11.0 + TAIL_S) and not covered([], 1.0)


# --- system audio is dropped inside the capture ---------------------------------------------------------

class Mic:
    overflows = dropped = 0
    last_callback_at = 0.0

    def __init__(self, blocks):
        self.blocks = list(blocks)
        self.last_block_wall = 0.0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return None

    def is_active(self):
        return True

    def read(self, timeout=1.0):
        if not self.blocks:
            return None
        b, self.last_block_wall = self.blocks.pop(0)
        return b


class Loop:
    dropped = 0
    device = SimpleNamespace(name="spk")

    def __init__(self, chunks):
        self.chunks = list(chunks)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return None

    def read_available(self):
        return self.chunks.pop(0) if self.chunks else np.zeros(0, dtype=np.float32)


def blk(v, n=WINDOW):
    return np.full(n, v, dtype=np.float32)


def test_mixed_capture_drops_system_audio_while_the_page_plays():
    mic = Mic([(blk(0.1), 1.0), (blk(0.1), 2.0), (blk(0.1), 3.0)])
    loop = Loop([blk(0.3), blk(0.3, 3000), blk(0.3)])        # the second read also brings a backlog
    m = MixedCapture(mic, loop)
    m.drop_system = lambda wall: wall == 2.0
    with m:
        assert float(m.read()[0]) == pytest.approx(0.4)       # mixed
        b = m.read()
        assert float(b.max()) == pytest.approx(0.1) and m.last_channels == (pytest.approx(0.1), 0.0)
        assert m.system_dropped_samples == 3000 and len(m._buf) == 0      # the backlog went with it
        assert float(m.read()[0]) == pytest.approx(0.4)       # mixing resumes with fresh loopback audio only


def test_a_failing_drop_predicate_never_stops_the_capture():
    mic = Mic([(blk(0.1), 1.0)])

    def boom(wall):
        raise RuntimeError("guard broke")

    m = MixedCapture(mic, Loop([blk(0.3)]))
    m.drop_system = boom
    with m:
        assert float(m.read()[0]) == pytest.approx(0.4)


# --- the recorder ---------------------------------------------------------------------------------------

class ScriptMic(Mic):
    def __init__(self, blocks, stop):
        super().__init__(blocks)
        self.stop = stop

    def read(self, timeout=1.0):
        if not self.blocks:
            self.stop.set()
            return None
        return super().read(timeout)


class Bus:
    def __init__(self):
        self.events = []

    def publish(self, kind, **data):
        self.events.append((kind, data))


def _blocks(n, t0, amp=0.05):
    a = noise(n * BLOCK_S, amp=amp, seed=3)
    return [(a[i * WINDOW:(i + 1) * WINDOW], t0 + (i + 1) * BLOCK_S) for i in range(n)]


@pytest.fixture
def gate_cfg(cfg_factory):
    def make(mode="all", **audio):
        return cfg_factory({"audio": {"min_speech_s": 0.1, "min_silence_s": 0.5, "pre_roll_s": 0.1, "post_roll_s": 0.1,
                                      "playback_guard": mode, **audio}})
    return make


def _recorder(cfg, db, monkeypatch, blocks, probs, guard, loop_chunks=None):
    stop = threading.Event()
    mic = ScriptMic(blocks, stop)
    monkeypatch.setattr(rec_mod, "MicCapture", lambda *a, **k: mic)
    if loop_chunks is not None:
        monkeypatch.setattr(rec_mod, "open_capture", lambda c, m: MixedCapture(m, Loop(loop_chunks)))
    r = Recorder(cfg, db, stop)
    r.vad = FakeVAD(probs, default=0.0)
    r.playback = guard
    r.events = Bus()
    return r


def _guard_window(t0, a_s, b_s):
    """A guard that saw the page play from t0 + a_s to t0 + b_s."""
    clk = Clock(t0 + a_s)
    g = PlaybackGuard(clock=clk)
    g.report("tab", True)
    clk.t = t0 + b_s
    g.report("tab", False)
    clk.t = t0 + 20.0                             # "now" while the recorder works through the blocks (within HISTORY_S)
    return g


def test_speech_played_by_the_page_is_not_recorded(gate_cfg, db, monkeypatch):
    t0 = time.time() - 3600
    n = 260                                       # 8.3 s
    probs = [0.0] * n
    for i in range(40, 90):                       # "speech" 1.3-2.9 s: the clip the page plays
        probs[i] = 1.0
    for i in range(170, 210):                     # real speech 5.4-6.7 s, after the playback
        probs[i] = 1.0
    g = _guard_window(t0, 1.0, 3.0)
    r = _recorder(gate_cfg("all"), db, monkeypatch, _blocks(n, t0), probs, g)
    r.run()
    rows = db.query("SELECT started_at, duration_s FROM segments")
    assert len(rows) == 1                          # only the speech after the playback
    from datetime import datetime

    start = datetime.fromisoformat(rows[0]["started_at"]).timestamp() - t0
    assert 5.0 < start < 5.6
    muted = (3.0 + TAIL_S) - (1.0 - LEAD_S)
    assert r.playback_muted_s == pytest.approx(muted, abs=2 * BLOCK_S)
    paused = [d["paused"] for k, d in r.events.events if k == "recording" and "paused" in d]
    assert paused[:2] == [True, False] and r.paused_for_playback is False


def test_a_segment_open_when_playback_starts_is_closed_and_kept(gate_cfg, db, monkeypatch):
    t0 = time.time() - 3600
    n = 200
    probs = [1.0] * 100 + [0.0] * 100             # someone talks from the start ... and the page starts playing at 2 s
    g = _guard_window(t0, 2.0, 5.0)
    r = _recorder(gate_cfg("all"), db, monkeypatch, _blocks(n, t0), probs, g)
    r.run()
    rows = db.query("SELECT duration_s FROM segments")
    assert len(rows) == 1
    assert rows[0]["duration_s"] < 2.0 + 0.6       # ends at the playback (+ hangover), not at 3.2 s


def test_system_mode_keeps_the_microphone_and_drops_the_clip(gate_cfg, db, monkeypatch):
    t0 = time.time() - 3600
    n = 160
    probs = [0.0] * 20 + [1.0] * 60 + [0.0] * 80   # the user talks 0.6-2.6 s while a clip plays 1-2 s
    loud = set(range(31, 63))
    chunks = [blk(0.3) if i in loud else np.zeros(0, dtype=np.float32) for i in range(n)]
    g = _guard_window(t0, 1.0, 2.0)
    r = _recorder(gate_cfg("system", system_audio=True), db, monkeypatch, _blocks(n, t0), probs, g, loop_chunks=chunks)
    r.run()
    row = db.one("SELECT channel_rms, duration_s FROM segments")
    assert row is not None and row["duration_s"] > 1.5          # the microphone was kept
    pairs = np.frombuffer(row["channel_rms"], dtype=np.float16).astype(np.float32).reshape(-1, 2)
    assert float(pairs[:, 1].max()) == 0.0                     # none of the played clip got in
    assert r.paused_for_playback is False and r.playback_muted_s == 0.0
    assert not [d for k, d in r.events.events if d.get("paused")]   # "system" never says "paused"


def test_without_the_guard_the_clip_would_have_been_recorded(gate_cfg, db, monkeypatch):
    # the bug: mode "off" (the old behaviour) records the played clip through the system-audio channel
    t0 = time.time() - 3600
    n = 160
    probs = [0.0] * 30 + [1.0] * 40 + [0.0] * 90
    chunks = [blk(0.3) if 30 <= i < 70 else np.zeros(0, dtype=np.float32) for i in range(n)]
    g = _guard_window(t0, 0.9, 2.3)
    r = _recorder(gate_cfg("off", system_audio=True), db, monkeypatch, _blocks(n, t0, amp=0.0005), probs, g, loop_chunks=chunks)
    r.run()
    row = db.one("SELECT channel_rms FROM segments")
    assert row is not None
    pairs = np.frombuffer(row["channel_rms"], dtype=np.float16).astype(np.float32).reshape(-1, 2)
    assert float(pairs[:, 1].max()) > 0.2


def test_all_mode_with_system_audio_records_nothing_of_the_clip(gate_cfg, db, monkeypatch):
    t0 = time.time() - 3600
    n = 160
    probs = [0.0] * 30 + [1.0] * 40 + [0.0] * 90
    chunks = [blk(0.3) if 30 <= i < 70 else np.zeros(0, dtype=np.float32) for i in range(n)]
    g = _guard_window(t0, 0.9, 2.3)
    r = _recorder(gate_cfg("all", system_audio=True), db, monkeypatch, _blocks(n, t0, amp=0.0005), probs, g, loop_chunks=chunks)
    r.run()
    assert db.one("SELECT COUNT(*) AS n FROM segments")["n"] == 0


def test_a_read_only_page_in_another_process_is_honoured(gate_cfg, db, monkeypatch):
    t0 = time.time() - 3600
    n = 160
    probs = [0.0] * 30 + [1.0] * 40 + [0.0] * 90
    cfg = gate_cfg("all")
    playback.write_file(playback.file_path(cfg.data_dir), [(t0 + 0.9, t0 + 2.3)])
    r = _recorder(cfg, db, monkeypatch, _blocks(n, t0), probs, PlaybackGuard(clock=Clock(t0 + 10_000)))
    r.run()
    assert db.one("SELECT COUNT(*) AS n FROM segments")["n"] == 0


def test_a_broken_guard_never_stops_the_recording(gate_cfg, db, monkeypatch):
    t0 = time.time() - 3600

    class Broken:
        def covers(self, t):
            raise RuntimeError("boom")

    probs = [0.0] * 10 + [1.0] * 40 + [0.0] * 40
    r = _recorder(gate_cfg("all"), db, monkeypatch, _blocks(90, t0), probs, Broken())
    r.run()
    assert db.one("SELECT COUNT(*) AS n FROM segments")["n"] == 1


# --- API, status, settings, page ------------------------------------------------------------------------

@pytest.fixture
def client(cfg_factory):
    from fastapi.testclient import TestClient

    from audio_note.runtime import Runtime
    from audio_note.ui.app import create_app

    rt = Runtime(cfg_factory({"memory": {"retrieval": "fts", "embedding_model": "hash"}}))
    try:
        with TestClient(create_app(rt), raise_server_exceptions=False) as c:
            yield c, rt
    finally:
        rt.shutdown()


def test_api_playback_reports_reach_the_guard(client):
    c, rt = client
    r = c.post("/api/playback", json={"tab": "t1", "playing": True, "seq": 1})
    assert r.status_code == 200 and r.json()["playing"] is True and r.json()["mode"] == "all"
    assert playback.GUARD.playing() and playback.GUARD.covers(time.time())
    assert c.post("/api/playback", json={"tab": "t1", "playing": False, "seq": 2}).json()["playing"] is False
    assert not playback.GUARD.playing() and playback.GUARD.covers(time.time())     # the tail is still covered
    assert c.post("/api/playback", json={"tab": "t1", "playing": "maybe"}).status_code == 400
    assert c.post("/api/playback", json={"tab": ["x"], "playing": True}).status_code == 400
    assert c.post("/api/playback", json={"tab": "t1", "playing": True, "seq": "x"}).status_code == 400
    assert not (rt.cfg.data_dir / playback.FILE_NAME).exists()        # this process records: nothing mirrored


def test_api_playback_is_mirrored_when_another_process_records(client, monkeypatch):
    c, rt = client
    monkeypatch.setattr(rt, "other_pid", lambda: 4242)
    assert c.post("/api/playback", json={"tab": "t1", "playing": True}).status_code == 200
    data = json.loads((rt.cfg.data_dir / playback.FILE_NAME).read_text(encoding="utf-8"))
    (s, e), = data["windows"]
    assert s <= time.time() <= e


def test_api_playback_refuses_other_sites(client):
    c, _ = client
    r = c.post("/api/playback", json={"tab": "t", "playing": True}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403 and not playback.GUARD.playing()


def test_status_reports_the_pause_and_the_mode(client):
    c, rt = client
    s = c.get("/api/status").json()["service"]
    assert s["paused_for_playback"] is False and s["playback_guard"] == "all"
    rt._rec_ref["recorder"] = SimpleNamespace(is_speaking=False, paused_for_playback=True, segments_written=0,
                                              speech_total_s=0.0, started_at=None, capture=None)
    assert rt.status()["service"]["paused_for_playback"] is True
    assert rt.status_lite()["service"]["paused_for_playback"] is True


def test_settings_page_can_change_the_mode(client):
    c, rt = client
    r = c.put("/api/config", json={"audio": {"playback_guard": "system"}})
    assert r.status_code == 200 and "audio.playback_guard" in r.json()["saved"]
    assert guard_mode(rt.cfg) == "system"
    assert c.put("/api/config", json={"audio": {"playback_guard": "sometimes"}}).status_code == 400
    sch = c.get("/api/config").json()["schema"]["keys"]["audio.playback_guard"]
    assert sch["choices"] == ["all", "system", "off"]


def test_the_player_reports_play_and_stop_with_a_heartbeat():
    js = (STATIC / "ui" / "player.js").read_text(encoding="utf-8")
    assert '"/api/playback"' in js and "keepalive: true" in js and "seq: ++seq" in js
    assert f"HEARTBEAT_MS = {int(playback.HEARTBEAT_S * 1000)}" in js
    for ev, fn in (('"play"', "guardOn()"), ('"pause"', "guardOff()"), ('"error"', "guardOff()"), ('"pagehide"', "guardOff()")):
        line = next(l for l in js.splitlines() if f"addEventListener({ev}" in l)
        assert fn in line or fn in js.split(f"addEventListener({ev}", 1)[1][:80], ev
    schema = (STATIC / "pages" / "settings" / "schema.js").read_text(encoding="utf-8")
    assert '"audio.playback_guard"' in schema
    shell = (STATIC / "ui" / "shell.js").read_text(encoding="utf-8")
    assert "s.paused_for_playback" in shell


def test_forget_can_keep_the_backups(client):
    # cleaning recorded playback up must not cost the pre-upgrade database backup
    c, rt = client
    seg = rt.db.add_segment("2026-11-13-x.opus", __import__("datetime").datetime(2026, 11, 13, 19, 39, 31), 9.0)
    ids = rt.db.add_utterances([{"segment_id": seg, "start_s": 0.0, "end_s": 2.0, "abs_start": "2026-11-13T19:39:31.000",
                                 "day": "2026-11-13", "speaker_name": None, "speaker_id": None, "text_raw": "放出来的一句",
                                 "text": "放出来的一句"}])
    bdir = Path(rt.db.path).parent / "backups"
    bdir.mkdir(exist_ok=True)
    backup = bdir / "audio_note-v0-20261113-193739.sqlite3"
    backup.write_bytes(b"x")
    dry = c.post("/api/memory/forget", json={"utterance_ids": ids}).json()
    assert dry["backups"] == [backup.name]                                    # the default still purges them
    rep = c.post("/api/memory/forget", json={"utterance_ids": ids, "dry_run": False, "purge_backups": False}).json()
    assert rep["utterances"] == 1 and rep["backups"] == [] and backup.exists()
    assert c.post("/api/memory/forget", json={"utterance_ids": ids, "purge_backups": "maybe"}).status_code == 400
