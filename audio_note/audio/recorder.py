"""Always-on recorder: mic -> VAD gate -> speech-only segment files.

Run in a thread (see service.py) or directly via `audio-note record`.

Timing: segment start times are derived from absolute sample indices (stream_t0 + n/16000).
Because blocks can be dropped (queue overflow, PortAudio overflow) and the machine can sleep,
``stream_t0`` is re-anchored whenever the wall-clock stamp of the newest block (taken inside the
PortAudio callback, so a queue backlog does not matter) drifts more than REANCHOR_THRESHOLD_S
from the sample clock. A segment that is already open keeps the start time its partial file was
named with, so the partial and final names always agree.

Robustness: if no block arrives for STALL_TIMEOUT_S (device unplugged, Bluetooth headset gone,
WASAPI endpoint invalidated by sleep/resume) or the stream reports itself inactive, run() raises
RuntimeError after flushing the open segment; service._recorder_loop then reopens the device.
A failing partial-file write degrades to memory-only capture instead of killing the thread.

Channel levels: when the capture mixes microphone + system audio (MixedCapture) every block comes
with ``last_channels`` = (mic_rms, system_rms). The recorder keeps one entry per block, indexed by
absolute sample position, for as long as the gate may still put that block into a segment (open
segment or pre-roll ring), and stores the pairs of the blocks overlapping each emitted segment in
segments.channel_rms / channel_block_s. Microphone-only capture leaves them NULL.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from .. import events as events_mod
from ..config import Config
from ..db import DB
from .capture import MicCapture, open_capture
from .storage import PartialWriter, finalize_segment, recover_partials, stale_partials
from .vad import SAMPLE_RATE, WINDOW, GateConfig, SileroVAD, SpeechGate

log = logging.getLogger(__name__)

STALL_TIMEOUT_S = 10.0          # no block for this long -> the input stream is considered dead
REANCHOR_THRESHOLD_S = 1.0      # |wall clock - sample clock| above this -> re-anchor stream_t0
PARTIAL_RECOVERY_INTERVAL_S = 300.0   # while running, look for crash leftovers this often (not only at start)


def gate_config_from(cfg: Config) -> GateConfig:
    a = cfg.audio
    return GateConfig(threshold=float(a.vad_threshold), min_silence_s=float(a.min_silence_s),
                      min_speech_s=float(a.min_speech_s), max_segment_s=float(a.max_segment_s),
                      pre_roll_s=float(a.pre_roll_s), post_roll_s=float(a.post_roll_s))


class Recorder:
    def __init__(self, cfg: Config, db: DB, stop_event: threading.Event | None = None):
        self.cfg = cfg
        self.db = db
        self.stop_event = stop_event or threading.Event()
        self.vad = SileroVAD(cfg.model_path("vad_model"), threads=1)
        self.gate = SpeechGate(gate_config_from(cfg))
        self.segments_written = 0
        self.speech_total_s = 0.0
        self.reanchors = 0
        self.started_at: datetime | None = None
        self.stream_t0: datetime | None = None   # wall time of absolute sample 0
        self._partial: PartialWriter | None = None
        self._partial_start_sample = 0
        self._consumed = 0                        # absolute samples fed to the gate
        self._delivered = 0                       # absolute samples received from the mic
        self.last_block_at = 0.0                  # time.time() when the last block was read
        self.last_prob = 0.0
        self.is_speaking = False
        self._recover_vad: SileroVAD | None = None   # separate instance: recovery must not reset the live VAD state
        self.partials_recovered = 0
        self.capture: Any = None                  # the open MicCapture / MixedCapture while run() is inside it
        #: event bus for `segment` / `recording` events (the service sets it; NULL drops everything)
        self.events: Any = events_mod.NULL
        #: called with the segment id right after a segment was stored (the service wakes its scheduler with it)
        self.on_segment: Any = None
        # (start sample, n samples, mic_rms, system_rms) per delivered block, oldest first; only while mixing
        self._channels: deque[tuple[int, int, float, float]] = deque()

    # wall-clock time of an absolute sample index
    def _time_of(self, sample: int) -> datetime:
        assert self.stream_t0 is not None
        return self.stream_t0 + timedelta(seconds=sample / SAMPLE_RATE)

    def run(self) -> None:
        """Record until the stop event is set. While it runs, the calling thread is raised above the process's
        (possibly below-normal, see performance.background_priority) priority so capture never starves."""
        boosted = False
        try:
            from .. import governor

            if bool((self.cfg.get("performance") or {}).get("background_priority", False)):
                governor.boost_current_thread()
                boosted = True
        except Exception as e:  # noqa: BLE001 - priorities are a nicety
            log.debug("recorder thread priority not raised: %s", e)
        try:
            self._run()
        finally:
            if boosted:
                try:
                    governor.restore_current_thread()
                except Exception:  # noqa: BLE001
                    pass

    def _run(self) -> None:
        cfg = self.cfg
        cfg.ensure_dirs()
        recover_partials(cfg, self.db, self.vad)
        self.started_at = datetime.now()
        carry = np.zeros(0, dtype=np.float32)
        with open_capture(cfg, MicCapture(cfg.audio.device, int(cfg.audio.block_ms), float(cfg.audio.gain))) as mic:
            self.capture = mic
            if getattr(mic, "system_error", None):
                log.warning("system audio: %s", mic.system_error)
            elif getattr(mic, "system_active", False):
                log.info("recording microphone + system audio (%s)", getattr(mic, "system_name", "?"))
            self.stream_t0 = datetime.now()
            self._consumed = 0
            self._delivered = 0
            self._channels.clear()
            first_block = True
            last_log = time.time()
            last_recover = time.time()
            self.last_block_at = time.time()
            try:
                while not self.stop_event.is_set():
                    if time.time() - last_recover >= PARTIAL_RECOVERY_INTERVAL_S:
                        last_recover = time.time()
                        self._recover_stale_partials()
                    block = mic.read(timeout=0.5)
                    if block is None:
                        self._check_stall(mic)
                        continue
                    self.last_block_at = time.time()
                    self._note_channels(getattr(mic, "last_channels", None), len(block))
                    self._delivered += len(block)
                    self._reanchor(getattr(mic, "last_block_wall", 0.0) or self.last_block_at, force=first_block)
                    first_block = False
                    buf = np.concatenate([carry, block]) if len(carry) else block
                    n = len(buf) // WINDOW
                    for i in range(n):
                        self._feed_window(buf[i * WINDOW:(i + 1) * WINDOW])
                    carry = buf[n * WINDOW:]
                    self._prune_channels()
                    if time.time() - last_log > 300:
                        last_log = time.time()
                        log.info("recorder alive: %d segments, %.0fs speech, overflows=%d dropped=%d reanchors=%d",
                                 self.segments_written, self.speech_total_s, mic.overflows, mic.dropped, self.reanchors)
            finally:
                # never lose the open segment, whatever ended the loop
                for seg in self.gate.flush():
                    self._emit(seg)
                self._abandon_partial()

    # --- per-window work -------------------------------------------------------
    def _feed_window(self, w: np.ndarray) -> None:
        p = self.vad.prob(w)
        self.last_prob = p
        was_active = self.gate.active
        segs = self.gate.push(w, p)
        self._consumed += WINDOW
        if self.gate.active and not was_active:
            self._open_partial(self._consumed - self.gate._buf_len)
        if self._partial is not None and self.gate.active:
            try:
                self._partial.write(w)
            except Exception as e:  # noqa: BLE001 - disk full / dir unwritable: keep going in memory
                log.warning("partial write failed (%s); continuing without partial file", e)
                self._abandon_partial()
        for seg in segs:
            self._emit(seg)
        if self.gate.active != self.is_speaking:
            self.is_speaking = self.gate.active
            self.events.publish("recording", on=True, speaking=bool(self.is_speaking))

    # --- channel levels (mic + system audio) -----------------------------------
    def _note_channels(self, pair: Any, n: int) -> None:
        """Remember the (mic_rms, system_rms) of the block about to be delivered (absolute samples
        [self._delivered, self._delivered + n)). Blocks without a pair (microphone only) are not kept,
        so a segment that is not fully covered gets no channel data."""
        if pair is None or n <= 0:
            return
        try:
            mic_rms, sys_rms = float(pair[0]), float(pair[1])
        except (TypeError, ValueError, IndexError):
            return
        self._channels.append((self._delivered, n, mic_rms, sys_rms))

    def _prune_channels(self) -> None:
        """Drop pairs of blocks the gate can no longer put into a segment: everything before the open
        segment's first sample, or before the pre-roll ring while idle."""
        if not self._channels:
            return
        g = self.gate
        held = g._buf_len if g.active else g._pre_len
        keep_from = self._consumed - held
        while self._channels and self._channels[0][0] + self._channels[0][1] <= keep_from:
            self._channels.popleft()

    def _segment_channels(self, seg) -> tuple[np.ndarray | None, float | None]:
        """(n_blocks x 2 float32 array, seconds per block) for the blocks overlapping `seg`, or
        (None, None) when the segment is not covered end to end by recorded pairs."""
        if not self._channels or len(seg.audio) == 0:
            return None, None
        # gate positions and the recorder's consumed counter differ by a constant within one run
        start = seg.start_sample + (self._consumed - self.gate._pos)
        end = start + len(seg.audio)
        sel = [c for c in self._channels if c[0] < end and c[0] + c[1] > start]
        if not sel or sel[0][0] > start or sel[-1][0] + sel[-1][1] < end:
            log.debug("segment [%d, %d) not fully covered by channel levels; storing none", start, end)
            return None, None
        for a, b in zip(sel, sel[1:]):
            if b[0] != a[0] + a[1]:
                log.debug("gap in channel levels inside segment [%d, %d); storing none", start, end)
                return None, None
        pairs = np.array([(c[2], c[3]) for c in sel], dtype=np.float32)
        block_s = float(np.median([c[1] for c in sel])) / SAMPLE_RATE
        return pairs, block_s

    # --- clock -----------------------------------------------------------------
    def _reanchor(self, block_wall: float, force: bool = False) -> None:
        """Keep stream_t0 consistent with the callback wall-clock stamp of the newest block."""
        if not block_wall:
            return
        expected = self._time_of(self._delivered)
        actual = datetime.fromtimestamp(block_wall)
        drift = (actual - expected).total_seconds()
        if force:
            self.stream_t0 = actual - timedelta(seconds=self._delivered / SAMPLE_RATE)
            return
        if abs(drift) >= REANCHOR_THRESHOLD_S:
            self.stream_t0 = actual - timedelta(seconds=self._delivered / SAMPLE_RATE)
            self.reanchors += 1
            log.warning("audio clock drifted %.1fs from wall clock (sleep / dropped blocks?); re-anchored", drift)

    def _check_stall(self, mic: MicCapture) -> None:
        silent_for = time.time() - self.last_block_at
        active = mic.is_active() if hasattr(mic, "is_active") else True
        if silent_for > STALL_TIMEOUT_S or not active:
            raise RuntimeError(f"input stream stalled: no audio for {silent_for:.0f}s (stream active={active}); reopening device")

    # --- partial file ----------------------------------------------------------
    def _recover_stale_partials(self) -> int:
        """Finish *_partial.wav leftovers that appeared (or were too fresh to touch) after start-up,
        e.g. from a hard kill followed by a restart within seconds. Our own open partial is skipped."""
        own = self._partial.path if self._partial is not None else None
        try:
            if not stale_partials(self.cfg, exclude=own):
                return 0
            if self._recover_vad is None:
                self._recover_vad = SileroVAD(self.cfg.model_path("vad_model"), threads=1)
            n = recover_partials(self.cfg, self.db, self._recover_vad, exclude=own)
            self.partials_recovered += n
            return n
        except Exception as e:  # noqa: BLE001 - housekeeping must never stop the recorder
            log.warning("partial recovery failed: %s", e)
            return 0

    def _open_partial(self, start_sample: int) -> None:
        self._partial_start_sample = start_sample
        started = self._time_of(start_sample)
        try:
            self._partial = PartialWriter(self.cfg, started)
            # pre-roll already sits in the gate buffer; write it now
            if self.gate._buf:
                for b in self.gate._buf[:-1]:
                    self._partial.write(b)
        except Exception as e:  # noqa: BLE001
            log.warning("partial writer failed: %s", e)
            self._abandon_partial()

    def _abandon_partial(self) -> None:
        p = self._partial
        self._partial = None
        if p is None:
            return
        try:
            p.abandon()
        except Exception:  # noqa: BLE001
            pass

    def _emit(self, seg) -> None:
        partial_path = None
        if self._partial is not None:
            started = self._partial.started_at   # keep the name the partial file was created with
            try:
                self._partial.close()
            except Exception as e:  # noqa: BLE001
                log.warning("partial close failed: %s", e)
            partial_path = self._partial.path
            self._partial = None
        else:
            started = self._time_of(seg.start_sample)
        try:
            channels, block_s = self._segment_channels(seg)
        except Exception as e:  # noqa: BLE001 - channel levels are optional; the audio must still be saved
            log.warning("channel levels unavailable for this segment: %s", e)
            channels, block_s = None, None
        try:
            seg_id = finalize_segment(self.cfg, self.db, seg.audio, started, seg.speech_s, partial=partial_path,
                                      channel_rms=channels, channel_block_s=block_s)
        except Exception as e:  # noqa: BLE001
            log.exception("finalize failed: %s", e)
            return
        if seg_id is not None:
            self.segments_written += 1
            self.speech_total_s += seg.speech_s
            self._announce(seg_id, started, seg)

    def _announce(self, seg_id: int, started: datetime, seg: Any) -> None:
        """Tell the page (a `segment` event) and the scheduler (on_segment) that a segment is ready.
        Neither may ever disturb the recording."""
        try:
            self.events.publish("segment", id=int(seg_id), status="new", day=started.strftime("%Y-%m-%d"),
                                duration_s=round(len(seg.audio) / SAMPLE_RATE, 2), speech_s=round(float(seg.speech_s), 2))
        except Exception as e:  # noqa: BLE001
            log.debug("segment event not published: %s", e)
        cb = self.on_segment
        if cb is not None:
            try:
                cb(int(seg_id))
            except Exception as e:  # noqa: BLE001
                log.debug("on_segment callback failed: %s", e)


def run_recorder(cfg: Config, db: DB, stop_event: threading.Event | None = None) -> Recorder:
    rec = Recorder(cfg, db, stop_event)
    rec.run()
    return rec
