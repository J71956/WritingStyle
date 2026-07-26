"""Generator tests: prompt completeness (P9) + fake-backend orchestration."""

from __future__ import annotations

from datetime import datetime

from stylellm.config import Settings
from stylellm.generate import FakeBackend, build_prompt, generate, style_summary
from stylellm.models import Chunk, StyleProfile, content_hash
from stylellm.retrieve import RetrievedChunk


def _profile() -> StyleProfile:
    return StyleProfile(
        version="1.0",
        computed_at=datetime(2026, 7, 26, 12, 0, 0),
        lexical={"ttr": 0.45, "mtld": 80.0},
        syntactic={"mean_sentence_len": 22.0, "pos_dist": {"NOUN": 0.25}},
        semantic={"skew_mode": "equal_weight_by_type"},
        pragmatic={"formality": 66.0, "flesch_kincaid_grade": 12.0,
                   "rhetorical_markers": {"however": 0.01, "therefore": 0.02}},
        per_type={"cover_letter": {
            "lexical": {"ttr": 0.52, "mtld": 70.0},
            "syntactic": {"mean_sentence_len": 18.0, "pos_dist": {"NOUN": 0.22}},
            "semantic": {},
            "pragmatic": {"formality": 60.0, "flesch_kincaid_grade": 11.0, "rhetorical_markers": {}},
        }},
    )


def _exemplar(text: str) -> RetrievedChunk:
    ch = Chunk(chunk_id="c0", doc_id="d.pdf", doc_type="cover_letter", chunk_index=0,
               token_count=10, text=text, content_hash=content_hash(text))
    return RetrievedChunk(chunk=ch, score=0.9)


def test_p9_prompt_always_has_brief_and_user_prompt():
    prompt = build_prompt("write a cover letter", _profile(), exemplars=[], doc_type=None)
    assert "Style brief" in prompt
    assert "write a cover letter" in prompt


def test_p9_exemplars_included_only_when_present():
    without = build_prompt("do X", _profile(), exemplars=[], doc_type="cover_letter")
    assert "Examples of the author's voice" not in without

    with_ex = build_prompt("do X", _profile(), [_exemplar("Dear hiring manager, ...")], "cover_letter")
    assert "Examples of the author's voice" in with_ex
    assert "Dear hiring manager" in with_ex


def test_style_summary_uses_per_type_when_available():
    s = style_summary(_profile(), "cover_letter")
    assert "cover_letter" in s
    assert "formality" in s.lower()


def test_generate_with_fake_backend_no_similarity():
    settings = Settings()
    result = generate(
        "write a thank-you note", _profile(), exemplars=[_exemplar("Thanks so much")],
        settings=settings, doc_type="cover_letter",
        backend=FakeBackend(), compute_similarity=False,
    )
    assert result.generated_text.startswith("[fake-backend draft]")
    assert result.exemplars_used == ["c0"]
    assert result.latency_ms >= 0
    assert result.style_similarity == 0.0
