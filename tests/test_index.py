"""Chunking & idempotent-indexing tests (P5, P6)."""

from __future__ import annotations

from stylellm.config import Settings
from stylellm.index import chunk_corpus, chunk_document
from stylellm.models import Document


def _para(n_words: int, word: str = "word") -> str:
    return " ".join([word] * n_words)


def test_p5_chunk_token_cap_and_metadata():
    # A paragraph well over the hard max must be split so every chunk <= 512 tok.
    settings = Settings()
    big = _para(2000)
    # give it sentence boundaries so the oversize splitter has something to cut on
    big = ". ".join(_para(50) for _ in range(40)) + "."
    doc = Document(doc_id="d.pdf", doc_type="assignment", text=big, token_count=2000)
    chunks = chunk_document(doc, settings.index.chunk_target_tokens, settings.index.chunk_max_tokens)
    assert chunks, "expected at least one chunk"
    for c in chunks:
        assert c.token_count <= settings.index.chunk_max_tokens
        assert c.doc_id == "d.pdf"
        assert c.doc_type == "assignment"
        assert c.content_hash
        assert c.text.strip()


def test_p5_paragraphs_packed_not_split_midway():
    # Small paragraphs are packed to the target; boundaries stay on paragraphs.
    text = "\n\n".join(_para(30, w) for w in ["alpha", "beta", "gamma", "delta"])
    doc = Document(doc_id="d.pdf", doc_type="reflection", text=text, token_count=120)
    chunks = chunk_document(doc, target_tokens=64, max_tokens=512)
    # Each ~30-word paragraph fits; packing to 64 groups two per chunk => 2 chunks.
    assert len(chunks) == 2
    # No chunk contains a partial paragraph: each single-word vocabulary stays intact.
    joined = " ".join(c.text for c in chunks)
    for w in ["alpha", "beta", "gamma", "delta"]:
        assert joined.count(w) == 30


def test_p6_idempotent_dedupe_on_content_hash():
    # Two documents with identical text produce one chunk each pre-dedupe, but
    # chunk_corpus dedupes on content_hash across the corpus.
    settings = Settings()
    settings.index.exclude_below_min_tokens = False
    text = _para(40, "same")
    docs = [
        Document(doc_id="a.pdf", doc_type="assignment", text=text, token_count=40),
        Document(doc_id="b.pdf", doc_type="assignment", text=text, token_count=40),
    ]
    chunks = chunk_corpus(docs, settings)
    hashes = [c.content_hash for c in chunks]
    assert len(hashes) == len(set(hashes))  # no duplicate content
    assert len(chunks) == 1


def test_exclude_below_min_tokens():
    settings = Settings()  # exclude_below_min_tokens True, min_doc_tokens 50
    docs = [
        Document(doc_id="short.pdf", doc_type="cover_letter", text=_para(10), token_count=10),
        Document(doc_id="ok.pdf", doc_type="cover_letter", text=_para(80), token_count=80),
    ]
    chunks = chunk_corpus(docs, settings)
    assert {c.doc_id for c in chunks} == {"ok.pdf"}
