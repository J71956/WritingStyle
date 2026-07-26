"""Evaluator tests: stylometric similarity monotonicity + A/B logging."""

from __future__ import annotations

from datetime import datetime

from stylellm.config import Settings
from stylellm.models import ABTrial, StyleProfile
from stylellm import evaluate


def _profile() -> StyleProfile:
    return StyleProfile(
        version="1.0",
        computed_at=datetime(2026, 7, 26, 12, 0, 0),
        lexical={"ttr": 0.45, "mtld": 80.0},
        syntactic={"mean_sentence_len": 22.0, "mean_parse_depth": 4.0,
                   "clause_density": 1.5, "pos_dist": {"NOUN": 0.25, "VERB": 0.15}},
        semantic={},
        pragmatic={"formality": 66.0, "flesch_kincaid_grade": 12.0,
                   "mean_sentiment": 0.01, "rhetorical_marker_rate": 0.03},
        per_type={},
    )


def _fake_features(vec_dict):
    """Return a features dict extract_features would produce, from explicit values."""
    return {
        "lexical": {"ttr": vec_dict["ttr"], "mtld": vec_dict["mtld"]},
        "syntactic": {
            "mean_sentence_len": vec_dict["msl"], "mean_parse_depth": vec_dict["depth"],
            "clause_density": vec_dict["clause"], "pos_dist": vec_dict["pos"],
        },
        "pragmatic": {
            "flesch_kincaid_grade": vec_dict["fk"], "formality": vec_dict["formality"],
            "mean_sentiment": vec_dict["sent"], "rhetorical_marker_rate": vec_dict["rmr"],
        },
    }


def test_similarity_identical_is_higher_than_divergent(monkeypatch):
    settings = Settings()
    profile = _profile()

    # A text whose features match the corpus-level profile exactly.
    same = _fake_features({"ttr": 0.45, "mtld": 80.0, "msl": 22.0, "depth": 4.0,
                           "clause": 1.5, "fk": 12.0, "formality": 66.0, "sent": 0.01,
                           "rmr": 0.03, "pos": {"NOUN": 0.25, "VERB": 0.15}})
    far = _fake_features({"ttr": 0.9, "mtld": 10.0, "msl": 5.0, "depth": 1.0,
                          "clause": 0.1, "fk": 2.0, "formality": 20.0, "sent": -0.5,
                          "rmr": 0.0, "pos": {"NOUN": 0.05, "VERB": 0.5}})

    def fake_extract(text):
        return same if text == "same" else far

    monkeypatch.setattr(evaluate, "extract_features", fake_extract)

    sim_same = evaluate.stylometric_similarity("same", profile, None, settings)
    sim_far = evaluate.stylometric_similarity("far", profile, None, settings)
    assert 0.0 <= sim_far < sim_same <= 1.0
    assert sim_same > 0.95  # near-identical vectors score near 1


def test_record_preference_resolves_blind_label_and_logs(tmp_path):
    settings = Settings()
    settings.paths.artifacts_dir = str(tmp_path)

    trial = ABTrial(trial_id="t1", prompt="p", output_styled="S", output_plain="P")
    presentation = {"A": "plain", "B": "styled", "exemplars_used": []}

    rated = evaluate.record_preference(trial, presentation, "B", settings)
    assert rated.preferred == "styled"
    assert rated.rated_at is not None

    log = tmp_path / settings.evaluate.log_file
    assert log.exists()
    assert "ab_trial" in log.read_text(encoding="utf-8")


def test_record_preference_rejects_bad_label(tmp_path):
    settings = Settings()
    settings.paths.artifacts_dir = str(tmp_path)
    trial = ABTrial(trial_id="t1", prompt="p", output_styled="S", output_plain="P")
    presentation = {"A": "styled", "B": "plain"}
    try:
        evaluate.record_preference(trial, presentation, "C", settings)
        assert False, "expected ValueError"
    except ValueError:
        pass
