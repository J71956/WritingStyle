"""End-to-end integration on the real Dataset/ corpus (slow)."""

from __future__ import annotations

import pytest

from stylellm.config import load_settings
from stylellm.ingest import ingest_corpus
from stylellm.models import StyleProfile
from stylellm.style_analyzer import analyze

pytestmark = pytest.mark.slow


def test_ingest_real_corpus_flags_and_counts():
    settings = load_settings()
    if not settings.dataset_path.exists():
        pytest.skip("Dataset/ not present")
    result = ingest_corpus(settings)
    assert len(result.documents) == 17
    # Every doc has a valid type and a token count.
    assert all(d.token_count >= 0 for d in result.documents)
    # EE.pdf yields almost no prose (its essay body is image-based) and is
    # therefore flagged as an extraction warning.
    assert any("EE.pdf" in w for w in result.warnings)


def test_analyze_real_corpus_produces_complete_profile():
    settings = load_settings()
    if not settings.dataset_path.exists():
        pytest.skip("Dataset/ not present")
    result = ingest_corpus(settings)
    docs = [d for d in result.documents if d.token_count >= settings.analyzer.min_doc_tokens]
    profile = analyze(docs, settings)

    assert isinstance(profile, StyleProfile)
    # Corpus-level: all four dimensions populated.
    assert profile.lexical and profile.syntactic and profile.pragmatic
    # Per-type: each present type carries all four dimensions.
    for dims in profile.per_type.values():
        assert set(dims) == {"lexical", "syntactic", "semantic", "pragmatic"}
    # Several real doc types survive the min-token filter.
    assert len(profile.per_type) >= 4
