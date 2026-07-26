"""Style-analyzer aggregation tests (mean/skew) and profile round-trip (P4)."""

from __future__ import annotations

from datetime import datetime

import pytest

from stylellm.models import StyleProfile
from stylellm.style_analyzer import _mean_dicts


def test_mean_dicts_scalar_average_union_semantics():
    # Missing keys count as 0 across the full list (union semantics).
    dicts = [{"a": 2.0, "b": 4.0}, {"a": 4.0}]
    out = _mean_dicts(dicts)
    assert out["a"] == 3.0  # (2+4)/2
    assert out["b"] == 2.0  # (4+0)/2


def test_mean_dicts_recurses_into_subdicts():
    dicts = [{"pos": {"NOUN": 0.4, "VERB": 0.2}}, {"pos": {"NOUN": 0.2}}]
    out = _mean_dicts(dicts)
    assert out["pos"]["NOUN"] == pytest.approx(0.3)
    assert out["pos"]["VERB"] == pytest.approx(0.1)


def test_profile_json_round_trip_within_tolerance():
    # P4: serialize -> deserialize reproduces the profile within float tolerance.
    profile = StyleProfile(
        version="1.0",
        computed_at=datetime(2026, 7, 26, 12, 0, 0),
        lexical={"ttr": 0.398123, "mtld": 77.665},
        syntactic={"mean_sentence_len": 29.31, "pos_dist": {"NOUN": 0.25}},
        semantic={"skew_mode": "equal_weight_by_type"},
        pragmatic={"formality": 65.899},
        per_type={"cover_letter": {"lexical": {"ttr": 0.523}}},
    )
    blob = profile.model_dump_json()
    restored = StyleProfile.model_validate_json(blob)
    assert restored.lexical["ttr"] == profile.lexical["ttr"]
    assert restored.per_type["cover_letter"]["lexical"]["ttr"] == 0.523
    assert restored.version == "1.0"
