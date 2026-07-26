"""Chunking & local in-memory index (R3).

Paragraph-based chunking with a token cap (spec §7): consecutive paragraphs are
packed up to ~256 tokens; a lone paragraph over the 512-token hard max is split
on sentence boundaries so a chunk never splits mid-sentence. Each chunk is
embedded once and stored in a flat, in-memory numpy index — brute-force cosine
over a few hundred chunks is sub-millisecond, so no FAISS/Chroma server is
warranted at this scale.

Re-indexing is idempotent: chunks are keyed on `content_hash`, so identical
corpus text yields identical chunks with no duplicates (R3.4, P6).

Persisted under artifacts/index/: chunks.jsonl (metadata + text) and
embeddings.npy (row-aligned to chunks.jsonl).

Properties: P5 (chunk invariants), P6 (idempotent indexing).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Settings
from .embeddings import embed_texts
from .ingest import count_tokens
from .models import Chunk, Document, content_hash

# Sentence boundary: end punctuation followed by whitespace. Coarse but
# deterministic and dependency-free — good enough to sub-split oversize
# paragraphs without pulling spaCy into the indexing hot path.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_paragraphs(text: str) -> list[str]:
    """Split cleaned text into paragraphs on blank lines."""
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if p.strip()]


def _split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(paragraph) if s.strip()]


def _pack_oversize(paragraph: str, max_tokens: int) -> list[str]:
    """Sentence-pack a paragraph that alone exceeds the hard max into <=max pieces."""
    pieces: list[str] = []
    current: list[str] = []
    current_tok = 0
    for sent in _split_sentences(paragraph):
        tok = count_tokens(sent)
        if current and current_tok + tok > max_tokens:
            pieces.append(" ".join(current))
            current, current_tok = [], 0
        current.append(sent)
        current_tok += tok
    if current:
        pieces.append(" ".join(current))
    return pieces or [paragraph]


def chunk_document(doc: Document, target_tokens: int, max_tokens: int) -> list[Chunk]:
    """Paragraph-pack a document into Chunks obeying the token cap (R3.1)."""
    tags = [doc.doc_type] + (["dominant"] if doc.is_dominant else [])
    texts: list[str] = []
    buf: list[str] = []
    buf_tok = 0

    def flush() -> None:
        nonlocal buf, buf_tok
        if buf:
            texts.append("\n\n".join(buf))
            buf, buf_tok = [], 0

    for para in _split_paragraphs(doc.text):
        ptok = count_tokens(para)
        if ptok > max_tokens:
            # A single paragraph too big to keep whole: flush, then sentence-split.
            flush()
            texts.extend(_pack_oversize(para, max_tokens))
            continue
        if buf and buf_tok + ptok > target_tokens:
            flush()
        buf.append(para)
        buf_tok += ptok
    flush()

    chunks: list[Chunk] = []
    for i, t in enumerate(texts):
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{i}",
                doc_id=doc.doc_id,
                doc_type=doc.doc_type,
                chunk_index=i,
                token_count=count_tokens(t),
                style_tags=tags,
                text=t,
                content_hash=content_hash(t),
            )
        )
    return chunks


def chunk_corpus(documents: list[Document], settings: Settings) -> list[Chunk]:
    """Chunk every eligible document; dedupe on content_hash (idempotent, P6)."""
    min_tok = settings.analyzer.min_doc_tokens
    exclude = settings.index.exclude_below_min_tokens
    target = settings.index.chunk_target_tokens
    hard_max = settings.index.chunk_max_tokens

    seen: set[str] = set()
    out: list[Chunk] = []
    for doc in documents:
        if exclude and doc.token_count < min_tok:
            continue
        for ch in chunk_document(doc, target, hard_max):
            if ch.content_hash in seen:
                continue  # idempotent: identical text indexed once (R3.4)
            seen.add(ch.content_hash)
            out.append(ch)
    return out


@dataclass
class StyleIndex:
    """Flat in-memory index: chunks row-aligned to an embedding matrix."""

    chunks: list[Chunk]
    embeddings: np.ndarray | None  # shape (n_chunks, dim), or None if unavailable

    def __len__(self) -> int:
        return len(self.chunks)


def build_index(documents: list[Document], settings: Settings) -> StyleIndex:
    """Chunk, embed, and assemble the in-memory index (R3.3)."""
    chunks = chunk_corpus(documents, settings)
    emb = embed_texts([c.text for c in chunks], settings.index.embedding_model)
    if emb is not None:
        for c, v in zip(chunks, emb):
            c.embedding = [float(x) for x in v]
    return StyleIndex(chunks=chunks, embeddings=emb)


# --- persistence ------------------------------------------------------------


def _index_dir(settings: Settings) -> Path:
    return settings.artifacts_path / "index"


def persist_index(index: StyleIndex, settings: Settings) -> Path:
    """Write chunks.jsonl + embeddings.npy to artifacts/index/. Returns the dir."""
    d = _index_dir(settings)
    d.mkdir(parents=True, exist_ok=True)
    with (d / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in index.chunks:
            # embedding lives in the .npy; keep the jsonl compact and readable.
            f.write(c.model_dump_json(exclude={"embedding"}) + "\n")
    if index.embeddings is not None:
        np.save(d / "embeddings.npy", index.embeddings)
    return d


def load_index(settings: Settings) -> StyleIndex:
    """Load a previously persisted index from artifacts/index/."""
    d = _index_dir(settings)
    jsonl = d / "chunks.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"No index found at {jsonl}. Run `stylellm index` first.")
    chunks = [
        Chunk.model_validate_json(line)
        for line in jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    npy = d / "embeddings.npy"
    emb = np.load(npy) if npy.exists() else None
    if emb is not None:
        for c, v in zip(chunks, emb):
            c.embedding = [float(x) for x in v]
    return StyleIndex(chunks=chunks, embeddings=emb)
