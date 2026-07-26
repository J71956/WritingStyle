"""Dense style-exemplar retrieval (R4).

Brute-force cosine over the flat in-memory index, with an optional `doc_type`
filter and MMR (Maximal Marginal Relevance) diversity so the top-k exemplars are
representative rather than five near-copies. Near-duplicate chunks are collapsed
first (P8). No BM25 and no cross-encoder reranker in v1 — unjustified at a few
hundred chunks (spec §1, R4.4).

Properties: P7 (doc_type filter soundness), P8 (retrieval bound & diversity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Settings
from .embeddings import cosine_matrix, embed_texts
from .index import StyleIndex
from .models import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity to the query


def _dedupe_near_duplicates(
    order: list[int], embeddings: np.ndarray, threshold: float
) -> list[int]:
    """Drop candidates whose cosine to an already-kept candidate exceeds threshold."""
    kept: list[int] = []
    for i in order:
        vi = embeddings[i]
        if any(float(np.dot(vi, embeddings[j])) >= threshold for j in kept):
            continue
        kept.append(i)
    return kept


def _mmr(
    candidates: list[int],
    query_sims: np.ndarray,
    embeddings: np.ndarray,
    k: int,
    lam: float,
) -> list[int]:
    """Maximal Marginal Relevance selection (R4.3)."""
    selected: list[int] = []
    pool = list(candidates)
    while pool and len(selected) < k:
        best_idx = None
        best_val = -np.inf
        for i in pool:
            relevance = query_sims[i]
            diversity = max((float(np.dot(embeddings[i], embeddings[j])) for j in selected), default=0.0)
            val = lam * relevance - (1 - lam) * diversity
            if val > best_val:
                best_val, best_idx = val, i
        selected.append(best_idx)
        pool.remove(best_idx)
    return selected


def retrieve(
    query: str,
    index: StyleIndex,
    settings: Settings,
    doc_type: str | None = None,
    k: int | None = None,
) -> list[RetrievedChunk]:
    """Return up to k diverse, on-type exemplars for a style request (R4.1-4.3)."""
    k = k or settings.retrieve.top_k
    if len(index) == 0 or index.embeddings is None:
        return []

    # 1. doc_type filter (P7): restrict the candidate set up front.
    cand = [
        i for i, c in enumerate(index.chunks)
        if doc_type is None or c.doc_type == doc_type
    ]
    if not cand:
        return []

    # 2. Embed the query in the index's space; degrade if unavailable.
    q = embed_texts([query], settings.index.embedding_model)
    if q is None or q.size == 0:
        return []
    sims = cosine_matrix(q[0], index.embeddings)

    # 3. Rank candidates, collapse near-duplicates, then MMR for diversity.
    ranked = sorted(cand, key=lambda i: float(sims[i]), reverse=True)
    deduped = _dedupe_near_duplicates(ranked, index.embeddings, settings.retrieve.dedupe_threshold)
    chosen = _mmr(deduped, sims, index.embeddings, k, settings.retrieve.mmr_lambda)

    return [RetrievedChunk(chunk=index.chunks[i], score=float(sims[i])) for i in chosen]
