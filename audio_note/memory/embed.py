"""Sentence embeddings for vector retrieval.

Embedder     fastembed (ONNX Runtime, CPU) wrapper around a HuggingFace model such as BAAI/bge-small-zh-v1.5,
             loaded lazily on first use and cached under models/embeddings (~90 MB download once).
HashEmbedder dependency-free character n-gram hashing: deterministic, no model file. Used by the tests and
             selectable with memory.embedding_model: hash.

Both return L2-normalised float32 vectors so cosine similarity is a dot product.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


def normalise(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        n = float(np.linalg.norm(a))
        return a / n if n > 0 else a
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return a / n


class BaseEmbedder:
    name: str = "base"
    dim: int = 0

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_passages([text])[0]


def stable_hash(s: str) -> int:
    """FNV-1a over UTF-8 bytes (Python's hash() is salted per process)."""
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


class HashEmbedder(BaseEmbedder):
    """Char unigram + bigram hashing into `dim` buckets with sign hashing. Overlapping characters ->
    similar vectors, which is all the unit tests need."""

    name = "hash"

    def __init__(self, dim: int = 256):
        self.dim = int(dim)

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        s = "".join((text or "").split())
        grams = list(s) + [s[i:i + 2] for i in range(len(s) - 1)]
        for g in grams:
            h = stable_hash(g)
            v[h % self.dim] += 1.0 if (h >> 20) & 1 else -1.0
        return normalise(v)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts])


class Embedder(BaseEmbedder):
    """fastembed TextEmbedding, loaded on first call (model load 2-8 s; ~1000 sentences/s on CPU)."""

    def __init__(self, model_name: str, cache_dir: Path | str, threads: int = 2):
        self.name = model_name
        self.cache_dir = Path(cache_dir)
        self.threads = max(1, int(threads))
        self._model = None
        self._lock = threading.Lock()
        self.dim = 0

    def load(self):
        with self._lock:
            if self._model is not None:
                return self._model
            from fastembed import TextEmbedding

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            kw = dict(model_name=self.name, cache_dir=str(self.cache_dir), threads=self.threads)
            try:  # cached copy first: no network round-trip, works offline
                m = TextEmbedding(local_files_only=True, **kw)
            except Exception as e:  # noqa: BLE001 - not cached yet -> download
                log.info("embedding model %s not cached (%s); downloading to %s", self.name, e, self.cache_dir)
                m = TextEmbedding(**kw)
            probe = np.asarray(list(m.embed(["你好"]))[0])
            self.dim = int(probe.shape[0])
            self._model = m
            log.info("embedding model %s ready (dim %d)", self.name, self.dim)
            return m

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        m = self.load()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return normalise(np.asarray(list(m.embed(list(texts), batch_size=64)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        m = self.load()
        return normalise(np.asarray(list(m.query_embed([text]))[0], dtype=np.float32))


def get_embedder(cfg) -> BaseEmbedder:
    """Embedder per config: memory.embedding_model 'hash' -> HashEmbedder, else a fastembed model id."""
    name = str(cfg.memory.embedding_model or "").strip() or DEFAULT_MODEL
    if name == "hash":
        return HashEmbedder()
    return Embedder(name, cfg.embeddings_dir, threads=int(cfg.memory.get("embedding_threads", 2)))
