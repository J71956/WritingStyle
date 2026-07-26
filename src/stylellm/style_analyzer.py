"""Multi-dimensional, skew-aware style analysis (R2) — the primary v1 deliverable.

Computes lexical/syntactic/semantic/pragmatic features per doc_type and at
corpus level. The corpus aggregate is **skew-aware**: features are computed per
document, averaged within each doc_type, then averaged across types with equal
weight (skew_mode = equal_weight_by_type). This guarantees the dominant Extended
Essay cannot swamp the profile, and that duplicating it leaves the aggregate
unchanged (property P3).

Properties: P3 (skew-aware aggregation), P4 (profile completeness & round-trip).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import numpy as np

from .config import Settings
from .embeddings import embed_texts
from .features import extract_features
from .models import Document, StyleProfile


# --- numeric aggregation over nested feature dicts --------------------------


def _mean_dicts(dicts: list[dict]) -> dict:
    """Recursively average a list of feature dicts.

    Numeric leaves are averaged over the union of keys (missing treated as 0).
    Sub-dicts recurse. Non-numeric leaves take the first value.
    """
    if not dicts:
        return {}
    keys: list = []
    for d in dicts:
        for k in d:
            if k not in keys:
                keys.append(k)
    out: dict = {}
    for k in keys:
        values = [d[k] for d in dicts if k in d]
        sample = values[0]
        if isinstance(sample, dict):
            # average over the union of sub-keys, missing = 0
            out[k] = _mean_dicts([v for v in values if isinstance(v, dict)])
        elif isinstance(sample, (int, float)) and not isinstance(sample, bool):
            # divide by full count so absent keys count as 0 (union semantics)
            out[k] = sum(float(v) for v in values) / len(dicts)
        else:
            out[k] = sample
    return out


# --- semantic dimension (embeddings) ----------------------------------------
#
# Document embeddings come from the shared `embeddings.embed_texts` so the
# analyzer, index, and query path all embed in the same space (returns None if
# the model can't be loaded, and callers degrade gracefully).


def _topic_clusters(embeddings: np.ndarray, k: int, seed: int) -> dict:
    from sklearn.cluster import KMeans

    k = min(k, len(embeddings))
    if k < 2:
        return {"n_clusters": len(embeddings), "cluster_sizes": [len(embeddings)]}
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(embeddings)
    sizes = np.bincount(km.labels_, minlength=k).tolist()
    return {"n_clusters": int(k), "cluster_sizes": sizes}


def _centroid(vecs: np.ndarray) -> list[float]:
    return [round(float(x), 5) for x in vecs.mean(axis=0)]


# --- main analysis ----------------------------------------------------------


def analyze(documents: list[Document], settings: Settings) -> StyleProfile:
    if not documents:
        raise ValueError("No documents to analyze")

    top_k = settings.analyzer.top_ngrams
    model_name = settings.analyzer.embedding_model
    seed = settings.seed

    # 1. Per-document textual features (lexical/syntactic/pragmatic).
    per_doc: dict[str, dict] = {d.doc_id: extract_features(d.text, top_k=top_k) for d in documents}

    # 2. Document embeddings for the semantic dimension (graceful if unavailable).
    doc_ids = [d.doc_id for d in documents]
    emb = embed_texts([d.text for d in documents], model_name)
    emb_by_id = {doc_ids[i]: emb[i] for i in range(len(doc_ids))} if emb is not None else {}

    # 3. Group by doc_type.
    by_type: dict[str, list[Document]] = defaultdict(list)
    for d in documents:
        by_type[d.doc_type].append(d)

    # 4. Per-type profiles.
    per_type: dict[str, dict] = {}
    type_semantic: dict[str, dict] = {}
    for dt, docs in by_type.items():
        feats = [per_doc[d.doc_id] for d in docs]
        merged = _mean_dicts(feats)
        # semantic per type
        if emb_by_id:
            vecs = np.stack([emb_by_id[d.doc_id] for d in docs])
            sem = {"style_embedding_centroid": _centroid(vecs), "n_docs": len(docs)}
        else:
            sem = {"note": "embeddings unavailable", "n_docs": len(docs)}
        type_semantic[dt] = sem
        per_type[dt] = {
            "lexical": merged.get("lexical", {}),
            "syntactic": merged.get("syntactic", {}),
            "semantic": sem,
            "pragmatic": merged.get("pragmatic", {}),
        }

    # 5. Corpus-level, skew-aware aggregate: equal weight per doc_type.
    #    Average the per-type dimension dicts across types (each type once).
    corpus_lexical = _mean_dicts([per_type[dt]["lexical"] for dt in per_type])
    corpus_syntactic = _mean_dicts([per_type[dt]["syntactic"] for dt in per_type])
    corpus_pragmatic = _mean_dicts([per_type[dt]["pragmatic"] for dt in per_type])

    # corpus semantic: equal-weight mean of per-type centroids + topic clusters
    if emb_by_id:
        type_centroids = np.stack(
            [np.asarray(type_semantic[dt]["style_embedding_centroid"]) for dt in per_type]
        )
        corpus_semantic = {
            "style_embedding_centroid": _centroid(type_centroids),
            "topic_clusters": _topic_clusters(
                np.stack([emb_by_id[i] for i in doc_ids]), settings.analyzer.topic_clusters, seed
            ),
            "skew_mode": settings.analyzer.skew_mode,
        }
    else:
        corpus_semantic = {"note": "embeddings unavailable", "skew_mode": settings.analyzer.skew_mode}

    return StyleProfile(
        version=settings.analyzer.profile_version,
        computed_at=datetime.now(),
        lexical=corpus_lexical,
        syntactic=corpus_syntactic,
        semantic=corpus_semantic,
        pragmatic=corpus_pragmatic,
        per_type=per_type,
    )


def persist_profile(profile: StyleProfile, settings: Settings) -> tuple:
    """Write style_profile.json and style_report.md. Returns (json_path, md_path)."""
    from .report import render_style_report

    art = settings.artifacts_path
    art.mkdir(parents=True, exist_ok=True)
    json_path = art / "style_profile.json"
    md_path = art / "style_report.md"
    json_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(render_style_report(profile), encoding="utf-8")
    return json_path, md_path
