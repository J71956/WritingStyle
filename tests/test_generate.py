"""Generator tests: prompt completeness (P9), ablation assembly, backends."""

from __future__ import annotations

from datetime import datetime

import pytest

from stylellm.config import Settings
from stylellm.generate import (
    _REWRITE_INSTRUCTION,
    _SYSTEM_INSTRUCTION,
    FakeBackend,
    HFBackend,
    LlamaCppBackend,
    _split_system,
    assemble_prompt,
    build_prompt,
    build_rewrite_prompt,
    content_retention,
    expansion_ratio,
    generate,
    rewrite,
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


def test_rewrite_prompt_carries_the_draft_and_forbids_invention():
    prompt = build_rewrite_prompt(
        "I did an internship at Acme in 2024.", _profile(),
        [_exemplar("Throughout my studies, I have endeavoured to...")], "cover_letter",
    )

    assert prompt.startswith(_REWRITE_INSTRUCTION)
    assert "not add any information" in prompt
    assert "I did an internship at Acme in 2024." in prompt
    assert "Throughout my studies" in prompt  # exemplar as a tone model
    assert "Style brief" in prompt


def test_rewrite_prompt_splits_into_system_and_user_turns():
    """The rewrite instruction must reach the model as a system turn, like R5.2's."""
    prompt = build_rewrite_prompt("draft text", _profile(), [], "cover_letter")
    system, user = _split_system(prompt)

    assert system == _REWRITE_INSTRUCTION
    assert "draft text" in user
    assert _REWRITE_INSTRUCTION not in user


@pytest.mark.parametrize(
    "source, rewritten, expected",
    [
        # Trailing punctuation must not split a word off from its bare form.
        ("Acme 2024 internship", "My 2024 internship at Acme.", 1.0),
        ("Acme 2024 internship", "I had a lovely time somewhere.", 0.0),
        ("", "anything at all", 1.0),  # nothing to lose
        ("Acme 2024 internship", "My internship at Acme.", 2 / 3),  # dropped the year
    ],
)
def test_content_retention(source, rewritten, expected):
    assert content_retention(source, rewritten) == pytest.approx(expected)


def test_content_retention_keeps_numbers():
    """A dropped figure is exactly the failure this guard exists to catch."""
    assert content_retention("revenue rose 15 percent", "revenue rose") < 1.0


def test_content_retention_penalises_inflection():
    """Documents the metric's known floor: it matches surface forms, not lemmas.

    A faithful rewrite that re-inflects ("internship" -> "interned") loses
    points it does not deserve, which is why the CLI warns only well below 1.0
    and why this number is a smoke alarm, not a fidelity score.
    """
    assert content_retention("a 2024 internship", "interned during 2024") < 1.0


def test_expansion_ratio_catches_what_retention_cannot():
    """The observed failure: a one-line draft inflated into paragraphs.

    Every source word survives, so retention looks healthy; only the expansion
    ratio shows that the model wrote a document of its own.
    """
    source = "group project on recycling went ok we got a B+"
    inflated = source + " " + " ".join(f"invented{i} material{i}" for i in range(40))

    assert content_retention(source, inflated) == pytest.approx(1.0)
    assert expansion_ratio(source, inflated) > 2.0


@pytest.mark.parametrize(
    "source, rewritten, expected",
    [
        ("alpha beta gamma", "alpha beta gamma", 1.0),
        ("alpha beta gamma", "alpha beta gamma delta epsilon zeta", 2.0),
        ("", "", 0.0),
        ("", "invented facts appeared", float("inf")),  # unbounded invention
    ],
)
def test_expansion_ratio(source, rewritten, expected):
    assert expansion_ratio(source, rewritten) == expected


def test_rewrite_scores_the_source_as_a_baseline():
    settings = Settings()
    result = rewrite(
        "i done a internship at acme it was good",
        _profile(), exemplars=[_exemplar("Thanks so much")], settings=settings,
        doc_type="cover_letter", backend=FakeBackend(),
    )

    assert result.source_text == "i done a internship at acme it was good"
    assert result.rewritten_text.startswith("[fake-backend rewrite]")
    assert 0.0 < result.source_similarity <= 1.0
    assert result.style_delta == result.style_similarity - result.source_similarity
    assert result.exemplars_used == ["c0"]


def test_rewrite_reports_zero_retention_when_the_model_returns_nothing():
    class _Empty:
        def complete(self, prompt, max_tokens, temperature):
            return ""

    result = rewrite(
        "Acme 2024 internship", _profile(), [], Settings(),
        doc_type="cover_letter", backend=_Empty(), compute_similarity=False,
    )
    assert result.rewritten_text == ""
    assert result.content_retention == 0.0
    assert result.expansion_ratio == 0.0
    assert result.style_similarity == 0.0


def _llamacpp(model_path: str, gguf_file: str | None = "Qwen3.5-9B-Q4_K_M.gguf"):
    """Build the backend with /props stubbed — no server runs in the suite."""
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend.host = "http://localhost:8080"
    backend.gguf_file = gguf_file
    backend.enable_thinking = False
    backend.timeout = 600
    backend.last_prompt_tokens = 0
    backend.last_new_tokens = 0
    backend.merged_system = False
    backend.served_model = ""
    backend.revision = "unknown"
    backend.seed = None
    backend._get = lambda path: {"model_path": model_path}
    return backend


_CACHED = (
    r"C:\Users\user\.cache\huggingface\hub\models--unsloth--Qwen3.5-9B-GGUF"
    r"\snapshots\3885219b6810b007914f3a7950a8d1b469d598a5\Qwen3.5-9B-Q4_K_M.gguf"
)


def test_llamacpp_records_served_model_and_revision():
    backend = _llamacpp(_CACHED)
    backend._check_served_model()

    assert backend.served_model == "Qwen3.5-9B-Q4_K_M.gguf"
    assert backend.revision == "3885219b6810b007914f3a7950a8d1b469d598a5"


def test_llamacpp_rejects_a_stale_server():
    """A server left running with another model must not mislabel a campaign row."""
    backend = _llamacpp(_CACHED.replace("Qwen3.5-9B-Q4_K_M", "gemma-4-E4B-it-Q4_K_M"))
    with pytest.raises(RuntimeError, match="gemma-4-E4B-it-Q4_K_M.gguf"):
        backend._check_served_model()


def test_llamacpp_accepts_any_model_when_gguf_file_is_unset():
    backend = _llamacpp("/models/whatever.gguf", gguf_file=None)
    backend._check_served_model()
    assert backend.served_model == "whatever.gguf"


def test_llamacpp_sends_system_turn_and_strips_reasoning():
    backend = _llamacpp(_CACHED)
    sent = {}

    def fake_post(path, body):
        sent["path"], sent["body"] = path, body
        return {
            "choices": [{"message": {"content": "<think>hmm</think>Dear Sir,"}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 7},
        }

    backend._post = fake_post
    text = backend.complete(build_prompt("do X", _profile(), [], "cover_letter"), 256, 0.7)

    assert text == "Dear Sir,"
    assert sent["path"] == "/v1/chat/completions"
    roles = [m["role"] for m in sent["body"]["messages"]]
    assert roles == ["system", "user"]
    assert sent["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert backend.last_prompt_tokens == 120
    assert backend.last_new_tokens == 7
    assert backend.merged_system is False
    assert "seed" not in sent["body"]  # unset seed must not pin the sampler


def test_llamacpp_sends_the_seed_when_the_campaign_pins_it():
    """llama.cpp samples server-side, so P10 rides on the request, not set_seed."""
    backend = _llamacpp(_CACHED)
    sent = {}
    backend._post = lambda path, body: (
        sent.update(body) or {"choices": [{"message": {"content": "x"}}], "usage": {}}
    )
    backend.seed = 1234
    backend.complete(build_prompt("do X", _profile(), [], "cover_letter"), 256, 0.7)

    assert sent["seed"] == 1234


def test_llamacpp_folds_system_when_the_template_rejects_it():
    """Gemma's template has no system role; the instruction must survive a 400."""
    import urllib.error

    backend = _llamacpp(_CACHED)
    calls = []

    def fake_post(path, body):
        calls.append(body)
        if len(calls) == 1:
            raise urllib.error.HTTPError(path, 400, "no system role", {}, None)
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    backend._post = fake_post
    text = backend.complete(build_prompt("do X", _profile(), [], "cover_letter"), 256, 0.7)

    assert text == "ok"
    assert [m["role"] for m in calls[1]["messages"]] == ["user"]
    assert _SYSTEM_INSTRUCTION in calls[1]["messages"][0]["content"]
    assert backend.merged_system is True


def test_make_backend_dispatches_to_llamacpp(monkeypatch):
    captured = {}

    def fake_init(self, host, **kwargs):
        captured["host"] = host
        captured.update(kwargs)

    monkeypatch.setattr(LlamaCppBackend, "__init__", fake_init)
    settings = Settings()  # llamacpp is the default backend

    backend = make_backend(settings)
    assert isinstance(backend, LlamaCppBackend)
    assert captured["host"] == "http://localhost:8080"
    assert captured["gguf_file"] == "Qwen3.5-9B-Q4_K_M.gguf"
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
