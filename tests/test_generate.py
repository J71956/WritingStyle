"""Generator tests: prompt completeness (P9), ablation assembly, backends."""

from __future__ import annotations

from datetime import datetime

import pytest

from stylellm.config import Settings
from stylellm.generate import (
    _SYSTEM_INSTRUCTION,
    FakeBackend,
    HFBackend,
    _split_system,
    assemble_prompt,
    build_prompt,
    generate,
    make_backend,
    strip_reasoning,
    style_summary,
)
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
    prompt = build_prompt(
        "write a cover letter", _profile(), exemplars=[], doc_type=None
    )
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


# --- ablation ladder (§2 of the test plan) ----------------------------------


@pytest.mark.parametrize(
    "use_profile,use_exemplars,wants_brief,wants_examples",
    [
        (False, False, False, False),  # plain
        (True, False, True, False),    # profile_only
        (False, True, False, True),    # exemplars_only
        (True, True, True, True),      # full
    ],
)
def test_assemble_prompt_toggles_components_independently(
    use_profile, use_exemplars, wants_brief, wants_examples
):
    prompt = assemble_prompt(
        "write a cover letter", _profile(), [_exemplar("Dear hiring manager, ...")],
        doc_type="cover_letter", use_profile=use_profile, use_exemplars=use_exemplars,
    )
    assert ("Style brief" in prompt) is wants_brief
    assert ("Examples of the author's voice" in prompt) is wants_examples
    # The request itself survives every condition.
    assert "write a cover letter" in prompt


def test_plain_condition_leaks_no_style_information():
    """The `plain` arm must carry neither the brief, the exemplars, nor the persona."""
    prompt = assemble_prompt(
        "do X", _profile(), [_exemplar("Dear hiring manager, ...")],
        doc_type="cover_letter", use_profile=False, use_exemplars=False,
    )
    assert "Dear hiring manager" not in prompt
    assert "formality" not in prompt.lower()
    assert _SYSTEM_INSTRUCTION not in prompt


def test_build_prompt_is_the_full_condition():
    both = assemble_prompt("do X", _profile(), [_exemplar("Sample")], "cover_letter",
                           use_profile=True, use_exemplars=True)
    assert build_prompt("do X", _profile(), [_exemplar("Sample")], "cover_letter") == both


# --- hf backend --------------------------------------------------------------


def test_strip_reasoning_removes_think_span():
    assert strip_reasoning("<think>weighing options</think>Dear sir,") == "Dear sir,"
    assert strip_reasoning("plain text") == "plain text"
    # An unclosed span means generation died mid-thought — no usable answer.
    assert strip_reasoning("<think>still going and then it stopped") == ""


def test_split_system_lifts_the_instruction_for_chat_templates():
    prompt = build_prompt("do X", _profile(), [], "cover_letter")
    system, user = _split_system(prompt)
    assert system == _SYSTEM_INSTRUCTION
    assert "Style brief" in user and "do X" in user
    assert _SYSTEM_INSTRUCTION not in user

    # The plain condition has no system turn to lift out.
    plain = assemble_prompt("do X", _profile(), [], None, False, False)
    assert _split_system(plain) == ("", plain)


class _NoSystemTokenizer:
    """Stub of a Gemma-style template: raises if given a system role."""

    def apply_chat_template(self, messages, **kwargs):
        if any(m["role"] == "system" for m in messages):
            raise ValueError("System role not supported")
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


class _SystemOkTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


def _bare_backend(tokenizer):
    backend = HFBackend.__new__(HFBackend)  # bypass __init__: no weights in tests
    backend.tokenizer = tokenizer
    backend.enable_thinking = None
    backend.merged_system = False
    backend.model_id = "stub"
    return backend


def test_render_folds_system_into_user_when_template_rejects_it():
    """Gemma has no system role — the instruction must survive, not be dropped."""
    backend = _bare_backend(_NoSystemTokenizer())
    rendered = backend._render(build_prompt("do X", _profile(), [], "cover_letter"))

    assert "system:" not in rendered
    assert _SYSTEM_INSTRUCTION in rendered  # folded in, not lost
    assert "do X" in rendered
    assert backend.merged_system is True


def test_render_keeps_system_turn_when_supported():
    backend = _bare_backend(_SystemOkTokenizer())
    rendered = backend._render(build_prompt("do X", _profile(), [], "cover_letter"))

    assert rendered.startswith("system:")
    assert backend.merged_system is False


def test_make_backend_dispatches_to_hf(monkeypatch):
    """Dispatch only — no weights are downloaded in the test suite."""
    captured = {}

    def fake_init(self, model_id, **kwargs):
        captured["model_id"] = model_id
        captured.update(kwargs)

    monkeypatch.setattr(HFBackend, "__init__", fake_init)
    settings = Settings()
    settings.generate.backend = "hf"
    settings.generate.model = "Qwen/Qwen2.5-7B-Instruct"

    backend = make_backend(settings)
    assert isinstance(backend, HFBackend)
    assert captured["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert captured["enable_thinking"] is False


def test_make_backend_rejects_unknown():
    settings = Settings()
    settings.generate.backend = "not-a-backend"
    with pytest.raises(ValueError):
        make_backend(settings)


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
