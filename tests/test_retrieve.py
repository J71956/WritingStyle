"""Retriever tests: doc_type filter (P7), bound & diversity (P8), MMR/dedupe."""

from __future__ import annotations

import numpy as np

from stylellm.config import Settings
from stylellm.index import StyleIndex
from stylellm.models import Chunk, content_hash
from stylellm.retrieve import _dedupe_near_duplicates, _mmr, retrieve


def _chunk(cid: str, doc_type: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id=f"{cid}.pdf", doc_type=doc_type, chunk_index=0,
        token_count=10, text=text, content_hash=content_hash(text),
    )


def _unit(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    return v / np.linalg.norm(v)


def _make_index(rows):
    chunks, embs = [], []
    for cid, dt, text, vec in rows:
        chunks.append(_chunk(cid, dt, text))
        embs.append(_unit(vec))
    return StyleIndex(chunks=chunks, embeddings=np.stack(embs))


def _patch_query(monkeypatch, vec):
    monkeypatch.setattr(
        "stylellm.retrieve.embed_texts",
        lambda texts, model: np.stack([_unit(vec)]),
    )


def test_p7_doc_type_filter_soundness(monkeypatch):
    index = _make_index([
        ("a", "cover_letter", "cover text", [1, 0, 0]),
        ("b", "personal_statement", "ps text", [0.9, 0.1, 0]),
        ("c", "cover_letter", "cover two", [0, 1, 0]),
    ])
    _patch_query(monkeypatch, [1, 0, 0])
    out = retrieve("q", index, Settings(), doc_type="cover_letter", k=5)
    assert out, "expected results"
    assert all(r.chunk.doc_type == "cover_letter" for r in out)


def test_p8_bound_and_no_near_duplicates(monkeypatch):
    # Two near-identical vectors + one distinct; dedupe should drop a near-copy.
    index = _make_index([
        ("a", "assignment", "one", [1, 0, 0]),
        ("a2", "assignment", "one-copy", [0.999, 0.001, 0]),
        ("b", "assignment", "two", [0, 1, 0]),
    ])
    _patch_query(monkeypatch, [1, 0, 0])
    s = Settings()
    out = retrieve("q", index, s, k=5)
    assert len(out) <= 5
    ids = {r.chunk.chunk_id for r in out}
    # a and a2 are near-duplicates (cosine ~1) — only one survives.
    assert not ({"a", "a2"} <= ids)


def test_dedupe_helper_threshold():
    embs = np.stack([_unit([1, 0]), _unit([0.99, 0.01]), _unit([0, 1])])
    kept = _dedupe_near_duplicates([0, 1, 2], embs, threshold=0.95)
    assert kept == [0, 2]


def test_mmr_prefers_diversity_over_redundant_relevance():
    # idx0 most relevant; idx1 nearly identical to idx0; idx2 relevant but distinct.
    embs = np.stack([_unit([1, 0]), _unit([0.98, 0.02]), _unit([0.6, 0.8])])
    sims = embs @ _unit([1, 0])
    # lam < 0.5 tips MMR toward diversity: the redundant idx1 (near-copy of idx0)
    # loses to the relevant-but-distinct idx2.
    chosen = _mmr([0, 1, 2], sims, embs, k=2, lam=0.3)
    assert chosen[0] == 0
    assert chosen[1] == 2


def test_empty_index_returns_nothing():
    empty = StyleIndex(chunks=[], embeddings=None)
    assert retrieve("q", empty, Settings()) == []
