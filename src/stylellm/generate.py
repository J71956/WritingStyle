"""Prompt-based generation in the author's voice (R5).

The PromptOrchestrator assembles three parts (R5.2): a natural-language
**style-profile summary** (tone, formality, complexity, preferred structures),
the top-k retrieved chunks as **few-shot stylistic examples**, and the user
prompt. If retrieval is empty, generation proceeds with the profile summary
alone (R5.4).

Generation runs a **local instruct model**. The default path is Hugging Face
`transformers` running the weights on-device (`HFBackend`); an Ollama HTTP
backend is kept for the original workflow, and a deterministic `fake` backend
lets the orchestration and tests run without a model installed. In every case
inference is local — no hosted inference API, so no author text leaves the
machine.

Properties: P9 (prompt completeness).
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Protocol

from .config import Settings
from .models import GenerationResult, StyleProfile
from .retrieve import RetrievedChunk


# --- style-profile summary --------------------------------------------------


def _fmt_markers(markers: dict) -> str:
    if not markers:
        return "few explicit discourse markers"
    top = sorted(markers.items(), key=lambda kv: -kv[1])[:4]
    return "connectives such as " + ", ".join(f"'{m}'" for m, _ in top)


def style_summary(profile: StyleProfile, doc_type: str | None) -> str:
    """Render a compact natural-language style brief for the system prompt (R5.2)."""
    dims = profile.per_type.get(doc_type) if doc_type else None
    if not dims:  # fall back to the skew-aware corpus profile
        dims = {
            "lexical": profile.lexical,
            "syntactic": profile.syntactic,
            "pragmatic": profile.pragmatic,
        }
    lex = dims.get("lexical", {})
    syn = dims.get("syntactic", {})
    prag = dims.get("pragmatic", {})

    formality = prag.get("formality", 50.0)
    formality_word = (
        "highly formal" if formality >= 65
        else "formal" if formality >= 55
        else "neutral" if formality >= 45
        else "conversational"
    )
    grade = prag.get("flesch_kincaid_grade", 0.0)
    msl = syn.get("mean_sentence_len", 0.0)
    ttr = lex.get("ttr", 0.0)
    markers = prag.get("rhetorical_markers", {})

    scope = f"the author's *{doc_type}* writing" if doc_type else "the author's overall writing"
    lines = [
        f"Style brief for {scope}:",
        f"- Register: {formality_word} (formality score ~{formality:.0f}/100).",
        (f"- Complexity: reads at roughly grade {grade:.0f}; "
         f"mean sentence length ~{msl:.0f} words."),
        f"- Lexical variety: type-token ratio ~{ttr:.2f}.",
        f"- Rhetorical habits: {_fmt_markers(markers)}.",
    ]
    return "\n".join(lines)


# --- prompt assembly (R5.2) -------------------------------------------------

_SYSTEM_INSTRUCTION = (
    "You are a writing assistant that drafts text in one specific author's "
    "personal voice. Match the register, sentence rhythm, and rhetorical habits "
    "described in the style brief, and take the examples as models of tone and "
    "structure — not as content to copy. Write only the requested text."
)


def assemble_prompt(
    user_prompt: str,
    profile: StyleProfile,
    exemplars: list[RetrievedChunk],
    doc_type: str | None,
    use_profile: bool = True,
    use_exemplars: bool = True,
) -> str:
    """Assemble the prompt with the two style components independently toggled.

    The ablation ladder needs four combinations; `build_prompt` (both on) is the
    production path, and the experiment harness uses the other three. With both
    components off the result is a bare request, matching `evaluate._PLAIN_TEMPLATE`
    in spirit: no style information reaches the model.
    """
    if not use_profile and not use_exemplars:
        return f"Write the following:\n{user_prompt.strip()}"

    parts = [_SYSTEM_INSTRUCTION]
    if use_profile:
        parts += ["", style_summary(profile, doc_type)]
    if use_exemplars and exemplars:
        parts.append("")
        parts.append("Examples of the author's voice:")
        for i, ex in enumerate(exemplars, 1):
            parts.append(f"[Example {i} — {ex.chunk.doc_type}]")
            parts.append(ex.chunk.text.strip())
            parts.append("")
    parts.append("Now write the following in the author's voice:")
    parts.append(user_prompt.strip())
    return "\n".join(parts).strip()


def build_prompt(
    user_prompt: str,
    profile: StyleProfile,
    exemplars: list[RetrievedChunk],
    doc_type: str | None,
) -> str:
    """Assemble system instruction + style brief + few-shot exemplars + request.

    Always contains the style brief and the user prompt; exemplars are included
    only when retrieval was non-empty (P9, R5.4).
    """
    return assemble_prompt(user_prompt, profile, exemplars, doc_type)


# --- backends ---------------------------------------------------------------


class Backend(Protocol):
    def complete(self, prompt: str, max_tokens: int, temperature: float) -> str: ...


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove a reasoning model's <think>…</think> span from decoded output.

    Hybrid-reasoning models emit chain-of-thought inline. Scoring it as prose
    would corrupt the stylometric vector, so it is cut before the text is
    returned. An unclosed span (generation stopped mid-thought) leaves no usable
    answer, so it is dropped entirely and surfaces as an empty draft.
    """
    text = _THINK_RE.sub("", text)
    text = _OPEN_THINK_RE.sub("", text)
    return text.strip()


def _split_system(prompt: str) -> tuple[str, str]:
    """Split an assembled prompt into (system turn, user turn) for chat templates."""
    if prompt.startswith(_SYSTEM_INSTRUCTION):
        return _SYSTEM_INSTRUCTION, prompt[len(_SYSTEM_INSTRUCTION):].strip()
    return "", prompt  # plain condition: no system instruction to lift out


class HFBackend:
    """Local Hugging Face `transformers` backend (R5.1).

    Weights are loaded from the local Hub cache once per instance and reused for
    every call — reloading an 8B model per generation would dominate a campaign's
    runtime. Set `HF_HUB_OFFLINE=1` after the first download to guarantee the
    offline property.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        dtype: str = "auto",
        enable_thinking: bool | None = False,
        load_in_4bit: bool = False,
        revision: str | None = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.enable_thinking = enable_thinking
        self.last_prompt_tokens = 0
        self.last_new_tokens = 0

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        kwargs: dict = {"dtype": dtype, "device_map": device, "revision": revision}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()

    @property
    def revision(self) -> str:
        """Resolved commit sha of the loaded weights, for the paper's setup table."""
        return getattr(self.model.config, "_commit_hash", None) or "unknown"

    def _render(self, prompt: str) -> str:
        system, user = _split_system(prompt)
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
        if self.enable_thinking is not None:
            # Only some templates accept this; fall back cleanly when they don't.
            try:
                return self.tokenizer.apply_chat_template(
                    messages, enable_thinking=self.enable_thinking, **kwargs
                )
            except (TypeError, ValueError):
                pass
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def complete(self, prompt: str, max_tokens: int, temperature: float) -> str:
        import torch

        rendered = self._render(prompt)
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        n_prompt = int(inputs["input_ids"].shape[-1])

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        # Slice off the prompt so only the completion is decoded.
        new_ids = out[0][n_prompt:]
        self.last_prompt_tokens = n_prompt
        self.last_new_tokens = int(new_ids.shape[-1])
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return strip_reasoning(text)


class OllamaBackend:
    """Local Ollama HTTP backend (R5.1) using the standard library only."""

    def __init__(self, host: str, model: str, think: bool | None = None):
        self.host = host.rstrip("/")
        self.model = model
        self.think = think

    def complete(self, prompt: str, max_tokens: int, temperature: float) -> str:
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        # Only send `think` for models that support it; omit for the rest so
        # non-reasoning models (llama3.1) don't reject the request.
        if self.think is not None:
            body["think"] = self.think
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise RuntimeError(
                f"Could not reach a local Ollama model at {self.host} "
                f"(model {self.model!r}). Is `ollama serve` running and the model pulled? "
                f"Original error: {e}"
            ) from e
        return data.get("response", "")


class FakeBackend:
    """Deterministic offline backend for tests / no-model runs.

    Echoes the requested task with a marker so the orchestration path is fully
    exercisable without a local model. Not a style model — only plumbing.
    """

    def complete(self, prompt: str, max_tokens: int, temperature: float) -> str:
        request = prompt.rsplit("in the author's voice:", 1)[-1].strip()
        return f"[fake-backend draft] {request}"


def make_backend(settings: Settings) -> Backend:
    backend = settings.generate.backend.lower()
    if backend == "fake":
        return FakeBackend()
    if backend == "hf":
        cfg = settings.generate
        return HFBackend(
            cfg.model,
            device=cfg.device,
            dtype=cfg.dtype,
            enable_thinking=cfg.enable_thinking,
            load_in_4bit=cfg.load_in_4bit,
            revision=cfg.revision,
        )
    if backend == "ollama":
        return OllamaBackend(
            settings.generate.ollama_host, settings.generate.model, settings.generate.think
        )
    raise ValueError(f"Unknown generation backend: {settings.generate.backend!r}")


# --- orchestration (R5) -----------------------------------------------------


def generate(
    user_prompt: str,
    profile: StyleProfile,
    exemplars: list[RetrievedChunk],
    settings: Settings,
    doc_type: str | None = None,
    backend: Backend | None = None,
    compute_similarity: bool = True,
) -> GenerationResult:
    """Assemble the prompt, call the local model, and score style similarity (R5.3)."""
    backend = backend or make_backend(settings)
    prompt = build_prompt(user_prompt, profile, exemplars, doc_type)

    start = time.perf_counter()
    text = backend.complete(prompt, settings.generate.max_tokens, settings.generate.temperature)
    latency_ms = int((time.perf_counter() - start) * 1000)

    similarity = 0.0
    if compute_similarity and text.strip():
        from .evaluate import stylometric_similarity  # lazy: avoids import cycle

        similarity = stylometric_similarity(text, profile, doc_type, settings)

    return GenerationResult(
        generated_text=text,
        style_similarity=similarity,
        exemplars_used=[ex.chunk.chunk_id for ex in exemplars],
        latency_ms=latency_ms,
    )
