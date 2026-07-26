"""Shared stylometric feature extraction (R2, reused by the deferred R6).

Four dimensions per the spec:
  - lexical:   type-token ratio, MTLD, top n-grams
  - syntactic: mean sentence length, POS distribution, mean parse depth, clause density
  - semantic:  (topic clusters / centroid handled in style_analyzer, corpus-level)
  - pragmatic: Flesch-Kincaid, formality, mean sentiment, rhetorical-marker rates

`feature_vector()` produces a stable-ordered numeric vector so the deferred
evaluator (R6.1) can compute a stylometric distance against a profile.

spaCy is loaded lazily and cached; this module works on already-cleaned text.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

_WORD_RE = re.compile(r"[A-Za-z']+")

# Rhetorical / discourse markers (pragmatic dimension).
RHETORICAL_MARKERS = (
    "however", "therefore", "moreover", "furthermore", "nevertheless",
    "thus", "consequently", "in addition", "for example", "for instance",
    "on the other hand", "in conclusion", "firstly", "secondly", "finally",
)

# Heylighen & Dewaele formality F-score uses these coarse POS classes.
_FORMAL_POS = {"NOUN", "PROPN", "ADJ", "ADP", "DET"}
_CONTEXTUAL_POS = {"PRON", "VERB", "ADV", "INTJ"}


@lru_cache(maxsize=1)
def _nlp():
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except OSError as e:  # pragma: no cover - environment guard
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm"
        ) from e
    # Long docs (EE) exceed the default max_length; raise it.
    nlp.max_length = 3_000_000
    return nlp


# --- lexical ----------------------------------------------------------------


def type_token_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def mtld(tokens: list[str], threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010).

    Length-robust alternative to TTR: mean number of tokens it takes for the
    running TTR to fall to `threshold`, averaged over forward and backward
    passes. Hand-rolled because no trustworthy pip package exists (unit-tested).
    """
    if len(tokens) < 2:
        return float(len(tokens))

    def _one_pass(seq: list[str]) -> float:
        factors = 0.0
        types: set[str] = set()
        count = 0
        for tok in seq:
            types.add(tok)
            count += 1
            ttr = len(types) / count
            if ttr <= threshold:
                factors += 1
                types.clear()
                count = 0
        if count > 0:  # partial factor for the trailing segment
            ttr = len(types) / count
            factors += (1 - ttr) / (1 - threshold)
        return len(seq) / factors if factors > 0 else float(len(seq))

    forward = _one_pass(tokens)
    backward = _one_pass(list(reversed(tokens)))
    return (forward + backward) / 2


def top_ngrams(tokens: list[str], n: int = 2, k: int = 20) -> dict[str, int]:
    grams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return dict(Counter(grams).most_common(k))


def lexical_features(text: str, top_k: int = 20) -> dict:
    tokens = [t.lower() for t in _WORD_RE.findall(text)]
    return {
        "n_tokens": len(tokens),
        "ttr": type_token_ratio(tokens),
        "mtld": mtld(tokens),
        "top_bigrams": top_ngrams(tokens, n=2, k=top_k),
    }


# --- syntactic & pragmatic (spaCy) ------------------------------------------


def _parse_depth(sent) -> int:
    """Max root-to-token depth in a dependency parse."""
    depths: dict[int, int] = {}

    def depth(tok) -> int:
        if tok.i in depths:
            return depths[tok.i]
        d = 0 if tok.head == tok else depth(tok.head) + 1
        depths[tok.i] = d
        return d

    return max((depth(t) for t in sent), default=0)


def syntactic_features(text: str) -> dict:
    doc = _nlp()(text)
    sents = list(doc.sents)
    n_sents = len(sents) or 1
    pos_counts: Counter[str] = Counter()
    depths: list[int] = []
    clause_heads = 0
    n_tokens = 0
    for sent in sents:
        depths.append(_parse_depth(sent))
        for tok in sent:
            n_tokens += 1
            pos_counts[tok.pos_] += 1
            # clause markers: verbs acting as clause roots/heads
            if tok.dep_ in {"ROOT", "advcl", "ccomp", "xcomp", "relcl", "acl", "conj"} and tok.pos_ in {"VERB", "AUX"}:
                clause_heads += 1
    total_pos = sum(pos_counts.values()) or 1
    pos_dist = {p: pos_counts[p] / total_pos for p in sorted(pos_counts)}
    return {
        "mean_sentence_len": n_tokens / n_sents,
        "mean_parse_depth": sum(depths) / len(depths) if depths else 0.0,
        "clause_density": clause_heads / n_sents,
        "pos_dist": pos_dist,
    }


def _formality_score(pos_dist: dict[str, float]) -> float:
    """Heylighen F-score in [0,100]; higher = more formal."""
    f = sum(pos_dist.get(p, 0.0) for p in _FORMAL_POS)
    c = sum(pos_dist.get(p, 0.0) for p in _CONTEXTUAL_POS)
    return 50.0 * ((f - c) + 1.0)


def _mean_sentiment(text: str) -> float:
    """Lightweight lexicon-free proxy: net polarity via a tiny word list.

    Kept dependency-free and deterministic; a coarse signal is enough for a
    style profile (not a sentiment product).
    """
    pos_words = {"good", "great", "excellent", "positive", "benefit", "improve",
                 "success", "effective", "strong", "valuable", "advantage"}
    neg_words = {"bad", "poor", "negative", "problem", "fail", "weak", "issue",
                 "difficult", "limitation", "disadvantage", "risk"}
    toks = [t.lower() for t in _WORD_RE.findall(text)]
    if not toks:
        return 0.0
    score = sum(t in pos_words for t in toks) - sum(t in neg_words for t in toks)
    return score / len(toks)


def pragmatic_features(text: str, pos_dist: dict[str, float]) -> dict:
    import textstat

    lower = text.lower()
    n_words = len(_WORD_RE.findall(text)) or 1
    markers = {m: lower.count(m) / n_words for m in RHETORICAL_MARKERS if lower.count(m)}
    return {
        "flesch_kincaid_grade": float(textstat.flesch_kincaid_grade(text)) if text.strip() else 0.0,
        "formality": _formality_score(pos_dist),
        "mean_sentiment": _mean_sentiment(text),
        "rhetorical_marker_rate": sum(markers.values()),
        "rhetorical_markers": markers,
    }


# --- top-level assembly -----------------------------------------------------


def extract_features(text: str, top_k: int = 20) -> dict:
    """Compute lexical + syntactic + pragmatic dimensions for one text block.

    The semantic dimension (topic clusters / embedding centroid) is computed at
    corpus/type level in style_analyzer, not here.
    """
    lexical = lexical_features(text, top_k=top_k)
    syntactic = syntactic_features(text)
    pragmatic = pragmatic_features(text, syntactic["pos_dist"])
    return {"lexical": lexical, "syntactic": syntactic, "pragmatic": pragmatic}


# Stable feature order for the stylometric vector (R6.1). POS fractions appended
# in a fixed tag order so vectors are comparable across texts.
_POS_ORDER = ("NOUN", "VERB", "ADJ", "ADV", "ADP", "PRON", "DET", "PROPN",
              "AUX", "CCONJ", "SCONJ", "NUM", "PART", "INTJ")

VECTOR_KEYS = (
    "lexical.ttr",
    "lexical.mtld",
    "syntactic.mean_sentence_len",
    "syntactic.mean_parse_depth",
    "syntactic.clause_density",
    "pragmatic.flesch_kincaid_grade",
    "pragmatic.formality",
    "pragmatic.mean_sentiment",
    "pragmatic.rhetorical_marker_rate",
) + tuple(f"pos.{p}" for p in _POS_ORDER)


def feature_vector(features: dict) -> list[float]:
    """Flatten a features dict into a stable-ordered numeric vector (R6.1)."""
    lex = features.get("lexical", {})
    syn = features.get("syntactic", {})
    prag = features.get("pragmatic", {})
    pos = syn.get("pos_dist", {})
    vec = [
        lex.get("ttr", 0.0),
        lex.get("mtld", 0.0),
        syn.get("mean_sentence_len", 0.0),
        syn.get("mean_parse_depth", 0.0),
        syn.get("clause_density", 0.0),
        prag.get("flesch_kincaid_grade", 0.0),
        prag.get("formality", 0.0),
        prag.get("mean_sentiment", 0.0),
        prag.get("rhetorical_marker_rate", 0.0),
    ]
    vec.extend(pos.get(p, 0.0) for p in _POS_ORDER)
    return [float(x) for x in vec]


def vector_distance(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two feature vectors (used by R6 later)."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
