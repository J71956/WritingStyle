"""Evaluation: stylometric similarity + blind A/B harness (R6).

The **primary automated metric** is a stylometric similarity (R6.1): the
closeness between a generated text's lexical/syntactic/pragmatic feature vector
and the author's profile vector for the target `doc_type`. This tracks *how*
text is written — not BLEU (which rewards memorization) and not embedding cosine
(which tracks topic).

The **primary subjective metric** is a blind A/B (R6.2): the same prompt drafted
*with* the style profile + exemplars vs. *without* (plain prompt), labels hidden;
the author records a preference. Trials are appended to a local JSONL log (R6.4).

Retrieval quality is judged indirectly via A/B outcomes, not a Hit@K target
(R6.3).
"""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import Settings
from .features import VECTOR_KEYS, extract_features, feature_vector
from .index import StyleIndex
from .models import ABTrial, Document, GenerationResult, RewriteResult, StyleProfile
from .retrieve import retrieve


# --- stylometric similarity (R6.1) ------------------------------------------


def _profile_target_vector(profile: StyleProfile, doc_type: str | None) -> list[float]:
    """The author's feature vector for a doc_type (or the skew-aware corpus one)."""
    dims = profile.per_type.get(doc_type) if doc_type else None
    if not dims:
        dims = {
            "lexical": profile.lexical,
            "syntactic": profile.syntactic,
            "pragmatic": profile.pragmatic,
        }
    return feature_vector(dims)


def feature_scales(documents: list[Document], settings: Settings) -> list[float]:
    """Per-feature std across per-document vectors, floored (for standardized distance).

    Gives each stylometric feature comparable weight regardless of its natural
    magnitude (e.g. MTLD ~80 vs. TTR ~0.4), so the similarity isn't dominated by
    large-valued features.
    """
    min_tok = settings.analyzer.min_doc_tokens
    vecs = [
        feature_vector(extract_features(d.text))
        for d in documents
        if d.token_count >= min_tok
    ]
    if not vecs:
        return [1.0] * len(VECTOR_KEYS)
    std = np.asarray(vecs, dtype=float).std(axis=0)
    return np.maximum(std, settings.evaluate.scale_floor).tolist()


def stylometric_similarity(
    text: str,
    profile: StyleProfile,
    doc_type: str | None,
    settings: Settings,
    scales: list[float] | None = None,
) -> float:
    """Similarity in (0, 1]: 1.0 = identical style vector, →0 as styles diverge (R6.1).

    With `scales` (per-feature std from the corpus) the distance is a standardized
    Euclidean distance mapped through 1/(1+d). Without them, a self-contained
    per-feature relative distance is used, so the metric still works at generation
    time when the corpus isn't loaded.
    """
    gen = np.asarray(feature_vector(extract_features(text)), dtype=float)
    tgt = np.asarray(_profile_target_vector(profile, doc_type), dtype=float)

    if scales is not None:
        s = np.asarray(scales, dtype=float)
        d = float(np.linalg.norm((gen - tgt) / s) / np.sqrt(len(s)))
        return 1.0 / (1.0 + d)

    # Self-contained fallback: mean per-feature relative difference.
    floor = settings.evaluate.scale_floor
    denom = np.abs(gen) + np.abs(tgt) + floor
    rel = np.abs(gen - tgt) / denom
    return float(max(0.0, 1.0 - rel.mean()))


# --- logging (R6.4) ---------------------------------------------------------


def _log_path(settings: Settings) -> Path:
    return settings.artifacts_path / settings.evaluate.log_file


def append_log(record: dict, settings: Settings) -> Path:
    """Append one JSON record (a trial or a generation) to the local eval log."""
    path = _log_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return path


# --- blind A/B harness (R6.2) -----------------------------------------------

_PLAIN_TEMPLATE = "Write the following:\n{prompt}"


def run_ab_trial(
    prompt: str,
    profile: StyleProfile,
    index: StyleIndex,
    settings: Settings,
    doc_type: str | None = None,
    backend=None,
) -> tuple[ABTrial, dict]:
    """Draft styled (profile + exemplars) vs. plain for the same prompt (R6.2).

    Returns the (unrated) ABTrial plus a `presentation` dict mapping the blind
    labels A/B to which arm they are, so the caller can show them without leaking
    the condition and then record which arm the author preferred.
    """
    from .generate import build_prompt, make_backend  # lazy: avoids import cycle

    backend = backend or make_backend(settings)

    exemplars = retrieve(prompt, index, settings, doc_type=doc_type)
    styled_prompt = build_prompt(prompt, profile, exemplars, doc_type)
    styled = backend.complete(
        styled_prompt, settings.generate.max_tokens, settings.generate.temperature
    )
    plain = backend.complete(
        _PLAIN_TEMPLATE.format(prompt=prompt.strip()),
        settings.generate.max_tokens,
        settings.generate.temperature,
    )

    trial = ABTrial(
        trial_id=uuid.uuid4().hex[:12],
        prompt=prompt,
        output_styled=styled,
        output_plain=plain,
    )

    # Blind presentation: randomly assign the two arms to labels A and B.
    arms = ["styled", "plain"]
    random.shuffle(arms)
    presentation = {"A": arms[0], "B": arms[1], "exemplars_used": [e.chunk.chunk_id for e in exemplars]}
    return trial, presentation


def record_preference(
    trial: ABTrial,
    presentation: dict,
    chosen_label: str,
    settings: Settings,
) -> ABTrial:
    """Resolve a blind label ('A'/'B') to the preferred arm, log, and return the trial."""
    label = chosen_label.strip().upper()
    if label not in ("A", "B"):
        raise ValueError("chosen_label must be 'A' or 'B'")
    trial.preferred = presentation[label]  # type: ignore[assignment]
    trial.rated_at = datetime.now()
    append_log({"type": "ab_trial", **trial.model_dump(), "presentation": presentation}, settings)
    return trial


def log_generation(result: GenerationResult, prompt: str, doc_type: str | None, settings: Settings) -> None:
    """Append a single generation's outcome to the eval log (R6.4)."""
    append_log(
        {"type": "generation", "prompt": prompt, "doc_type": doc_type, **result.model_dump()},
        settings,
    )


def log_rewrite(result: RewriteResult, doc_type: str | None, settings: Settings) -> None:
    """Append a single rewrite's outcome to the eval log (R6.4).

    `style_delta` is stored rather than recomputed at read time, so the log is
    self-describing when it is read back months later.
    """
    append_log(
        {
            "type": "rewrite",
            "doc_type": doc_type,
            "style_delta": result.style_delta,
            **result.model_dump(),
        },
        settings,
    )
