"""Speaker diarization (who spoke when) + enrollment-based identification (which known person).

Models (sherpa-onnx, CPU, no torch):
  segmentation : pyannote segmentation-3.0 (ONNX)
  embedding    : 3D-Speaker CAM++ zh-cn 16k (192-d)

Enrollment: `enroll(name, wav_paths)` extracts one embedding per sample (speech-gated) and stores
them in speaker_embeddings. Identification: cosine similarity between a cluster's mean embedding
and each enrolled speaker's embeddings (max over samples); accepted if >= identify_threshold.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sherpa_onnx

from ..config import Config
from ..db import DB
from ..audio.storage import read_segment
from ..audio.vad import SAMPLE_RATE, GateConfig, SileroVAD, segment_audio

log = logging.getLogger(__name__)


@dataclass
class Turn:
    start: float
    end: float
    cluster: int                 # diarization cluster within this audio
    speaker_id: int | None = None
    speaker_name: str | None = None
    score: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


class SpeakerEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        sp = cfg.speakers
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(cfg.model_path("embedding_model")), num_threads=int(sp.num_threads))
        )
        self.dim = self.extractor.dim
        self._diar: sherpa_onnx.OfflineSpeakerDiarization | None = None
        self._vad: SileroVAD | None = None

    # --- lazy heavy objects ---------------------------------------------
    @property
    def diarizer(self) -> sherpa_onnx.OfflineSpeakerDiarization:
        if self._diar is None:
            sp = self.cfg.speakers
            dcfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(self.cfg.model_path("segmentation_model"))),
                    num_threads=int(sp.num_threads)),
                embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(self.cfg.model_path("embedding_model")), num_threads=int(sp.num_threads)),
                clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=float(sp.cluster_threshold)),
                min_duration_on=float(sp.min_turn_s), min_duration_off=0.5,
            )
            self._diar = sherpa_onnx.OfflineSpeakerDiarization(dcfg)
        return self._diar

    @property
    def vad(self) -> SileroVAD:
        if self._vad is None:
            self._vad = SileroVAD(self.cfg.model_path("vad_model"))
        return self._vad

    # --- embeddings --------------------------------------------------------
    def embed(self, audio: np.ndarray) -> np.ndarray:
        st = self.extractor.create_stream()
        st.accept_waveform(SAMPLE_RATE, audio.astype(np.float32, copy=False))
        st.input_finished()
        e = np.asarray(self.extractor.compute(st), dtype=np.float32)
        n = np.linalg.norm(e)
        return e / n if n > 0 else e

    def embed_speech_only(self, audio: np.ndarray, max_s: float = 60.0) -> tuple[np.ndarray, float]:
        """Concatenate VAD speech parts (up to max_s) before embedding. Returns (embedding, speech_seconds)."""
        segs = segment_audio(audio, self.vad, GateConfig(min_silence_s=0.3, pre_roll_s=0.1, post_roll_s=0.1, max_segment_s=max_s))
        parts = [s.audio for s in segs]
        speech = np.concatenate(parts) if parts else audio
        speech = speech[: int(max_s * SAMPLE_RATE)]
        return self.embed(speech), len(speech) / SAMPLE_RATE

    # --- diarization -------------------------------------------------------
    def diarize(self, audio: np.ndarray) -> list[Turn]:
        if len(audio) < SAMPLE_RATE:  # < 1 s: single turn
            return [Turn(0.0, len(audio) / SAMPLE_RATE, 0)]
        res = self.diarizer.process(audio.astype(np.float32, copy=False)).sort_by_start_time()
        return [Turn(float(r.start), float(r.end), int(r.speaker)) for r in res]

    # --- identification ----------------------------------------------------
    def identify_turns(self, audio: np.ndarray, turns: list[Turn], db: DB) -> list[Turn]:
        """Embed each cluster (all its turns concatenated, speech up to 60 s) and match against enrolled speakers."""
        enrolled = db.speaker_embeddings()
        thr = float(self.cfg.speakers.identify_threshold)
        unknown = self.cfg.speakers.unknown_label
        clusters: dict[int, list[Turn]] = {}
        for t in turns:
            clusters.setdefault(t.cluster, []).append(t)
        for cid, ts in clusters.items():
            parts = [audio[int(t.start * SAMPLE_RATE): int(t.end * SAMPLE_RATE)] for t in ts]
            speech = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
            if len(speech) < int(0.5 * SAMPLE_RATE):
                name, sid, score = unknown, None, None
            else:
                emb = self.embed(speech[: 60 * SAMPLE_RATE])
                sid, name, score = self.match(emb, enrolled, thr)
                if sid is None:
                    name = unknown
            for t in ts:
                t.speaker_id, t.speaker_name, t.score = sid, name, score
        return turns

    @staticmethod
    def match(emb: np.ndarray, enrolled: list[tuple[int, str, np.ndarray]], threshold: float) -> tuple[int | None, str | None, float | None]:
        if not enrolled:
            return None, None, None
        best: dict[int, tuple[str, float]] = {}
        for sid, name, e in enrolled:
            e = e / (np.linalg.norm(e) or 1.0)
            s = float(np.dot(emb, e))
            if sid not in best or s > best[sid][1]:
                best[sid] = (name, s)
        sid, (name, score) = max(best.items(), key=lambda kv: kv[1][1])
        if score >= threshold:
            return sid, name, score
        return None, None, score

    # --- enrollment ----------------------------------------------------------
    def enroll(self, db: DB, name: str, sources: list[Path | str], notes: str | None = None) -> int:
        """Register `name` with one embedding per source file. Files are decoded with libsndfile/ffmpeg."""
        from ..audio.importer import decode_any

        sid = db.upsert_speaker(name, notes)
        n = 0
        for src in sources:
            src = Path(src)
            audio = decode_any(src)
            emb, speech_s = self.embed_speech_only(audio)
            if speech_s < 1.0:
                log.warning("skip %s: only %.1fs speech", src, speech_s)
                continue
            db.add_speaker_embedding(sid, emb, str(src), speech_s)
            n += 1
        log.info("enrolled %s with %d samples", name, n)
        return sid

    def enroll_from_audio(self, db: DB, name: str, audio: np.ndarray, source: str | None = None) -> int:
        sid = db.upsert_speaker(name)
        emb, speech_s = self.embed_speech_only(audio)
        db.add_speaker_embedding(sid, emb, source, speech_s)
        return sid

    def enroll_from_segment_turns(self, db: DB, name: str, seg_path: Path | str, turns: list[tuple[float, float]]) -> int:
        """Label an existing recording's time ranges as `name` (used by `audio-note speakers label`)."""
        audio = read_segment(seg_path)
        parts = [audio[int(a * SAMPLE_RATE): int(b * SAMPLE_RATE)] for a, b in turns]
        return self.enroll_from_audio(db, name, np.concatenate(parts), f"{seg_path}#{turns[0][0]:.1f}-{turns[-1][1]:.1f}")
