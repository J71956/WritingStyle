"""Feature-math unit tests (lexical/vector). spaCy-dependent tests are marked slow."""

from __future__ import annotations

import pytest

from stylellm.features import (
    VECTOR_KEYS,
    extract_features,
    feature_vector,
    mtld,
    type_token_ratio,
    vector_distance,
)


def test_ttr_basic():
    assert type_token_ratio(["a", "b", "c", "a"]) == 3 / 4
    assert type_token_ratio([]) == 0.0


def test_mtld_all_unique_is_high_all_same_is_low():
    unique = [f"w{i}" for i in range(100)]
    repeated = ["w"] * 100
    assert mtld(unique) > mtld(repeated)
    # a single repeated type collapses diversity toward the sequence length floor
    assert mtld(repeated) <= len(repeated)


def test_mtld_short_input():
    assert mtld([]) == 0.0
    assert mtld(["only"]) == 1.0


def test_feature_vector_shape_and_order_stable():
    feats = {
        "lexical": {"ttr": 0.5, "mtld": 40.0},
        "syntactic": {"mean_sentence_len": 20.0, "mean_parse_depth": 4.0,
                      "clause_density": 2.0, "pos_dist": {"NOUN": 0.3, "VERB": 0.2}},
        "pragmatic": {"flesch_kincaid_grade": 12.0, "formality": 60.0,
                      "mean_sentiment": 0.0, "rhetorical_marker_rate": 0.01},
    }
    vec = feature_vector(feats)
    assert len(vec) == len(VECTOR_KEYS)
    assert vec[0] == 0.5  # lexical.ttr first, stable order
    assert vector_distance(vec, vec) == 0.0


@pytest.mark.slow
def test_extract_features_full_pipeline_smoke():
    text = ("I study engineering. However, my curiosity drives me. "
            "Therefore, I research problems and solve them effectively.")
    feats = extract_features(text)
    assert set(feats) == {"lexical", "syntactic", "pragmatic"}
    assert feats["syntactic"]["mean_sentence_len"] > 0
    assert 0.0 <= feats["lexical"]["ttr"] <= 1.0
    assert sum(feats["syntactic"]["pos_dist"].values()) == pytest.approx(1.0, abs=1e-6)
