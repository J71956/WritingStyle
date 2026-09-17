"""Prompt-based generation in the author's voice (R5).

The PromptOrchestrator assembles three parts (R5.2): a natural-language
**style-profile summary** (tone, formality, complexity, preferred structures),
the top-k retrieved chunks as **few-shot stylistic examples**, and the user
prompt. If retrieval is empty, generation proceeds with the profile summary
alone (R5.4).

Generation runs a **local instruct model**. The default path is llama.cpp's
`llama-server` holding a quantized GGUF (`LlamaCppBackend`), which is what makes
a 9B model fit an 8 GB card; `HFBackend` runs unquantized weights in-process via
Hugging Face `transformers`, an Ollama HTTP backend is kept for the original
workflow, and a deterministic `fake` backend lets the orchestration and tests
run without a model installed. In every case inference is local — no hosted
inference API, so no author text leaves the machine.

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
from .models import GenerationResult, RewriteResult, StyleProfile
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


_REWRITE_INSTRUCTION = (
    "You are a rewriting assistant that recasts a draft into one specific "
    "author's personal voice. Preserve the draft's content exactly: every fact, "
    "name, number, and claim in it must survive, in the same order, and you must "
    "not add any information that is not already there. Change only the wording, "
    "sentence rhythm, and register, to match the style brief and the examples — "
    "take the examples as models of tone, not as content to borrow. Return only "
    "the rewritten text."
)


def build_rewrite_prompt(
    source_text: str,
    profile: StyleProfile,
    exemplars: list[RetrievedChunk],
    doc_type: str | None,
) -> str:
    """Assemble the restyling prompt: instruction + brief + exemplars + draft (R5.5).

    Mirrors `build_prompt` but inverts the task. Generation is told to invent
    content in the author's voice; rewriting is told the content is fixed and
    only the voice may move.
    """
    parts = [_REWRITE_INSTRUCTION, "", style_summary(profile, doc_type)]
    if exemplars:
        parts.append("")
        parts.append("Examples of the author's voice:")
        for i, ex in enumerate(exemplars, 1):
            parts.append(f"[Example {i} — {ex.chunk.doc_type}]")
            parts.append(ex.chunk.text.strip())
            parts.append("")
    parts.append("Rewrite the following draft in the author's voice:")
    parts.append(source_text.strip())
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
    for instruction in (_SYSTEM_INSTRUCTION, _REWRITE_INSTRUCTION):
        if prompt.startswith(instruction):
            return instruction, prompt[len(instruction):].strip()
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
        # True once _render has had to fold the system instruction into the user
        # turn because this model's template rejects a system role (e.g. Gemma).
        self.merged_system = False

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

        # Some families (Gemma) have no system role and their template raises on
        # one, so fall back to folding the instruction into the user turn. The
        # prompt content is identical either way; only the framing differs, and
        # `merged_system` records which framing a model actually received so the
        # paper can report it rather than implying all three were prompted alike.
        candidates = []
        if system:
            candidates.append(
                [{"role": "system", "content": system}, {"role": "user", "content": user}]
            )
            candidates.append([{"role": "user", "content": f"{system}\n\n{user}"}])
        else:
            candidates.append([{"role": "user", "content": user}])

        last_error: Exception | None = None
        for i, messages in enumerate(candidates):
            try:
                rendered = self._apply_template(messages)
            except Exception as e:  # jinja2 TemplateError isn't importable here
                last_error = e
                continue
            self.merged_system = bool(system) and i > 0
            return rendered

        raise RuntimeError(
            f"No chat-template form worked for {self.model_id!r}: {last_error}"
        ) from last_error

    def _apply_template(self, messages: list[dict]) -> str:
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


class LlamaCppBackend:
    """Local llama.cpp backend over `llama-server`'s OpenAI-compatible API (R5.1).

    The server owns the weights, so a 9B model runs from a ~5.7 GB Q4_K_M GGUF
    on a card that cannot hold it at bf16. It also applies the chat template
    baked into the GGUF, which is why this backend sends structured messages
    rather than the flat rendered string `HFBackend` builds.

    Quantization is a reportable experimental condition: `served_model` records
    the exact file the server has loaded so the paper cites what actually ran.
    """

    def __init__(
        self,
        host: str,
        gguf_file: str | None = None,
        enable_thinking: bool | None = False,
        timeout: int = 600,
    ):
        self.host = host.rstrip("/")
        self.gguf_file = gguf_file
        self.enable_thinking = enable_thinking
        self.timeout = timeout
        self.last_prompt_tokens = 0
        self.last_new_tokens = 0
        # Mirrors HFBackend: true once the server's template has rejected a
        # system role and the instruction had to be folded into the user turn.
        self.merged_system = False
        self.served_model = ""
        self.revision = "unknown"
        # Sampling seed, sent per request. The sampler lives in the server
        # process, so `transformers.set_seed` cannot reach it — the campaign's
        # reproducibility claim (P10) depends on setting this instead.
        self.seed: int | None = None
        self._check_served_model()

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(f"{self.host}{path}")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _check_served_model(self) -> None:
        """Read the loaded model from /props and fail loudly on a mismatch.

        Also recovers the Hub snapshot sha from the cache path, which is the
        revision the paper's setup table pins.
        """
        try:
            props = self._get("/props")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise RuntimeError(
                f"Could not reach llama-server at {self.host}. Start it with:\n"
                f"  llama-server -hf {self.gguf_file or '<repo>'} --port "
                f"{self.host.rsplit(':', 1)[-1]}\n"
                f"Original error: {e}"
            ) from e

        path = (props.get("model_path") or props.get("model") or "").replace("\\", "/")
        parts = path.split("/")
        served = parts[-1] if parts else ""
        if self.gguf_file and served and served != self.gguf_file:
            raise RuntimeError(
                f"llama-server at {self.host} has {served!r} loaded, but the config "
                f"asks for {self.gguf_file!r}. Restart the server with the right "
                f"GGUF, or clear generate.gguf_file to accept whatever is served."
            )
        self.served_model = served or "unknown"
        # .../snapshots/<sha>/<file>.gguf when served from the Hub cache.
        if len(parts) >= 3 and parts[-3] == "snapshots":
            self.revision = parts[-2]

    def _messages(self, prompt: str, fold_system: bool) -> list[dict]:
        system, user = _split_system(prompt)
        if not system:
            return [{"role": "user", "content": user}]
        if fold_system:
            return [{"role": "user", "content": f"{system}\n\n{user}"}]
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def complete(self, prompt: str, max_tokens: int, temperature: float) -> str:
        # Templates with no system role (Gemma) make the server 400 the request;
        # retry with the instruction folded into the user turn so it survives.
        last_error: Exception | None = None
        for fold in (False, True):
            body = {
                "messages": self._messages(prompt, fold),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }
            if self.enable_thinking is not None:
                body["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
            if self.seed is not None:
                body["seed"] = self.seed
            try:
                data = self._post("/v1/chat/completions", body)
            except urllib.error.HTTPError as e:
                last_error = e
                if e.code == 400 and not fold:
                    continue  # probably the system-role rejection; try folded
                raise RuntimeError(f"llama-server rejected the request: {e}") from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                raise RuntimeError(
                    f"Lost the llama-server connection at {self.host}: {e}"
                ) from e

            self.merged_system = fold
            usage = data.get("usage") or {}
            self.last_prompt_tokens = int(usage.get("prompt_tokens", 0))
            self.last_new_tokens = int(usage.get("completion_tokens", 0))
            choices = data.get("choices") or [{}]
            text = (choices[0].get("message") or {}).get("content") or ""
            return strip_reasoning(text)

        raise RuntimeError(
            f"No message form worked against llama-server: {last_error}"
        ) from last_error

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            f"{self.host}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


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
        # Echo the draft back for rewrites, so a dry run exercises the content
        # -retention path instead of scoring it 0 against a marker string.
        if prompt.startswith(_REWRITE_INSTRUCTION):
            return f"[fake-backend rewrite] {request}"
        return f"[fake-backend draft] {request}"


def make_backend(settings: Settings) -> Backend:
    backend = settings.generate.backend.lower()
    if backend == "fake":
        return FakeBackend()
    if backend == "llamacpp":
        cfg = settings.generate
        return LlamaCppBackend(
            cfg.llamacpp_host,
            gguf_file=cfg.gguf_file,
            enable_thinking=cfg.enable_thinking,
        )
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


# --- rewriting (R5.5) -------------------------------------------------------

# Internal apostrophes/hyphens/dots are part of the word ("don't", "U.S.",
# "data-driven"); a trailing one is punctuation, so each separator must be
# followed by more word characters to be kept.
_WORD_RE = re.compile(r"[a-z0-9]+(?:['\-.][a-z0-9]+)*")


def content_words(text: str) -> set[str]:
    """Content-bearing tokens of `text`: lowercased, stopwords removed.

    Numbers are kept deliberately — a dropped or invented figure is the kind of
    content loss a restyling pass must not be allowed to hide.
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return {
        w for w in _WORD_RE.findall(text.lower())
        if w not in ENGLISH_STOP_WORDS and len(w) > 1
    }


def content_retention(source: str, rewritten: str) -> float:
    """Share of the source's content words that survive the rewrite, in [0, 1].

    A blunt instrument: it cannot tell a faithful paraphrase from a lucky
    vocabulary overlap, and legitimate synonym choices cost it points. It is here
    to catch the gross failure — a model that ignored the draft and wrote its own
    thing — not to certify factual fidelity. An empty source scores 1.0 (nothing
    to lose).
    """
    src = content_words(source)
    if not src:
        return 1.0
    return len(src & content_words(rewritten)) / len(src)


def expansion_ratio(source: str, rewritten: str) -> float:
    """Content words in the rewrite per content word in the source.

    ~1.0 is a faithful restyle. Well above that means the model padded the draft
    with material of its own — the failure `content_retention` cannot see, since
    inventing text does not remove any of the original words. An empty source
    with a non-empty rewrite is unbounded invention, reported as `inf`.
    """
    src = len(content_words(source))
    out = len(content_words(rewritten))
    if src == 0:
        return 0.0 if out == 0 else float("inf")
    return out / src


def rewrite(
    source_text: str,
    profile: StyleProfile,
    exemplars: list[RetrievedChunk],
    settings: Settings,
    doc_type: str | None = None,
    backend: Backend | None = None,
    compute_similarity: bool = True,
) -> RewriteResult:
    """Restyle an existing draft into the author's voice, keeping its content (R5.5)."""
    backend = backend or make_backend(settings)
    prompt = build_rewrite_prompt(source_text, profile, exemplars, doc_type)

    start = time.perf_counter()
    text = backend.complete(prompt, settings.generate.max_tokens, settings.generate.temperature)
    latency_ms = int((time.perf_counter() - start) * 1000)

    similarity = source_sim = 0.0
    if compute_similarity:
        from .evaluate import stylometric_similarity  # lazy: avoids import cycle

        # The source is scored too: without that baseline there is no way to say
        # whether the rewrite actually moved the text toward the author's voice.
        source_sim = stylometric_similarity(source_text, profile, doc_type, settings)
        if text.strip():
            similarity = stylometric_similarity(text, profile, doc_type, settings)

    return RewriteResult(
        rewritten_text=text,
        source_text=source_text,
        style_similarity=similarity,
        source_similarity=source_sim,
        content_retention=content_retention(source_text, text),
        expansion_ratio=expansion_ratio(source_text, text),
        exemplars_used=[ex.chunk.chunk_id for ex in exemplars],
        latency_ms=latency_ms,
    )
