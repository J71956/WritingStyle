"""Hypothesis property tests: P2 (dominance), P3 (skew invariance), P4 (round-trip)."""

from __future__ import annotations

from datetime import datetime

from hypothesis import given
from hypothesis import strategies as st

from stylellm.ingest import flag_dominant
from stylellm.models import Document, StyleProfile
from stylellm.style_analyzer import _mean_dicts

DOC_TYPES = ["personal_statement", "cover_letter", "academic_ia", "assignment", "reflection"]


# --- P2: dominant-doc flagging ---------------------------------------------


@given(counts=st.lists(st.integers(min_value=1, max_value=100_000), min_size=1, max_size=12),
       threshold=st.floats(min_value=0.1, max_value=0.9))
def test_p2_dominance_flag_matches_definition(counts, threshold):
    docs = [
        Document(doc_id=f"d{i}.pdf", doc_type=DOC_TYPES[i % len(DOC_TYPES)],
                 text="x", token_count=c)
        for i, c in enumerate(counts)
    ]
    flag_dominant(docs, threshold)
    total = sum(counts)
    for doc in docs:
        assert doc.is_dominant == (doc.token_count > threshold * total)


# --- P3: skew-aware aggregation invariance ---------------------------------

_num_dict = st.dictionaries(
    keys=st.sampled_from(["ttr", "mtld", "len", "depth"]),
    values=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
    min_size=1, max_size=4,
)


@given(d=_num_dict, k=st.integers(min_value=1, max_value=8))
def test_p3_mean_of_identical_duplicates_is_identity(d, k):
    # Duplicating a doc's features within its type leaves the within-type mean
    # unchanged, so the corpus aggregate is invariant to duplication (P3 core).
    out = _mean_dicts([d] * k)
    for key, val in d.items():
        assert abs(out[key] - val) < 1e-9


@given(dominant=_num_dict, other=_num_dict, extra=st.integers(min_value=0, max_value=10))
def test_p3_corpus_aggregate_invariant_to_dominant_duplication(dominant, other, extra):
    # Two types: one "dominant" type whose doc is duplicated `extra` extra times.
    # Equal-weight-by-type aggregate must not move.
    def corpus(dom_copies):
        type_dom = _mean_dicts([dominant] * dom_copies)
        type_other = _mean_dicts([other])
        return _mean_dicts([type_dom, type_other])

    base = corpus(1)
    dup = corpus(1 + extra)
    keys = set(base) | set(dup)
    for key in keys:
        assert abs(base.get(key, 0.0) - dup.get(key, 0.0)) < 1e-9


# --- P4: profile completeness & round-trip ---------------------------------

_feature_dict = st.dictionaries(
    keys=st.text(min_size=1, max_size=8),
    values=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    max_size=5,
)


@given(lex=_feature_dict, syn=_feature_dict, prag=_feature_dict)
def test_p4_profile_json_round_trip(lex, syn, prag):
    profile = StyleProfile(
        version="1.0",
        computed_at=datetime(2026, 7, 26, 12, 0, 0),
        lexical=lex, syntactic=syn, semantic={"skew_mode": "equal_weight_by_type"},
        pragmatic=prag,
        per_type={"cover_letter": {"lexical": lex, "syntactic": syn,
                                   "semantic": {}, "pragmatic": prag}},
    )
    restored = StyleProfile.model_validate_json(profile.model_dump_json())
    assert restored.version == profile.version
    for key, val in lex.items():
        assert restored.lexical[key] == val
    # all four dimensions present per type
    assert set(restored.per_type["cover_letter"]) == {"lexical", "syntactic", "semantic", "pragmatic"}
