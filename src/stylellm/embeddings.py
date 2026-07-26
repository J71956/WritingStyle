"""Shared sentence-embedding helpers (R3/R4).

One cached SentenceTransformer per model name, normalized embeddings so a dot
product is a cosine. Kept separate from `style_analyzer` (which embeds whole
documents) because the index, retriever, and query path must all embed in the
*same* space; this is the single place that space is defined.

Embeddings are optional at import time: `embed_texts` returns None if the model
can't be loaded (offline, not installed), and callers degrade gracefully.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=2)
def _model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str) -> np.ndarray | None:
    """Encode texts to L2-normalized row vectors, or None if unavailable."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    try:
        model = _model(model_name)
    except Exception:
        return None
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def cosine_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine of one (already-normalized) query vector against index rows.

    Inputs are assumed L2-normalized (embed_texts normalizes), so this is a
    plain dot product; we renormalize defensively to stay correct if not.
    """
    if matrix.size == 0:
        return np.empty((0,), dtype=np.float32)
    q = query / (np.linalg.norm(query) + 1e-12)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return m @ q
