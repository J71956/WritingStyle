"""Markdown rendering for the ingestion report and the style report (R1.5, R2.3).

The style report is the primary human-readable v1 deliverable.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from .ingest import IngestResult
from .models import StyleProfile


def render_ingestion_report(result: IngestResult) -> str:
    docs = result.documents
    total_tokens = sum(d.token_count for d in docs) or 1

    per_type: dict[str, list] = defaultdict(list)
    for d in docs:
        per_type[d.doc_type].append(d)

    lines: list[str] = []
    lines.append("# Ingestion Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Documents: **{len(docs)}**")
    lines.append(f"- Total tokens (approx, word-based): **{total_tokens:,}**")
    lines.append("")

    lines.append("## Per-document-type token distribution")
    lines.append("")
    lines.append("| doc_type | docs | tokens | % of corpus |")
    lines.append("|---|---:|---:|---:|")
    for dt in sorted(per_type, key=lambda k: -sum(x.token_count for x in per_type[k])):
        group = per_type[dt]
        tok = sum(x.token_count for x in group)
        lines.append(f"| {dt} | {len(group)} | {tok:,} | {100 * tok / total_tokens:.1f}% |")
    lines.append("")

    dominant = [d for d in docs if d.is_dominant]
    lines.append("## Dominant documents (R1.3, >40% of corpus tokens)")
    lines.append("")
    if dominant:
        for d in dominant:
            lines.append(f"- **{d.doc_id}** — {d.token_count:,} tokens "
                         f"({100 * d.token_count / total_tokens:.1f}%) → `is_dominant`")
    else:
        lines.append("- None.")
    lines.append("")

    lines.append("## Per-document detail")
    lines.append("")
    lines.append("| doc_id | doc_type | tokens | dominant |")
    lines.append("|---|---|---:|:---:|")
    for d in sorted(docs, key=lambda x: -x.token_count):
        lines.append(f"| {d.doc_id} | {d.doc_type} | {d.token_count:,} | "
                     f"{'✅' if d.is_dominant else ''} |")
    lines.append("")

    lines.append("## Extraction warnings")
    lines.append("")
    if result.warnings:
        for w in result.warnings:
            lines.append(f"- ⚠️ {w}")
    else:
        lines.append("- None.")
    lines.append("")

    if result.pii_redactions:
        lines.append("## PII redactions (soft, logged)")
        lines.append("")
        for doc_id, n in sorted(result.pii_redactions.items()):
            lines.append(f"- {doc_id}: {n} redaction(s)")
        lines.append("")

    return "\n".join(lines)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, dict):
        # show top few key=value pairs compactly
        items = list(v.items())[:6]
        return ", ".join(f"{k}={_fmt(val)}" for k, val in items)
    if isinstance(v, list):
        return ", ".join(str(x) for x in v[:8])
    return str(v)


def _scalar_rows(dim: dict) -> list[tuple[str, str]]:
    rows = []
    for k, val in dim.items():
        if isinstance(val, (int, float, str)):
            rows.append((k, _fmt(val)))
    return rows


def render_style_report(profile: StyleProfile) -> str:
    lines: list[str] = []
    lines.append("# Style Report")
    lines.append("")
    lines.append(f"- Profile version: `{profile.version}`")
    lines.append(f"- Computed at: {profile.computed_at.isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("> Corpus-level aggregate is **skew-aware** (equal-weight by doc_type), "
                 "so the dominant Extended Essay cannot swamp the profile (R2.2).")
    lines.append("")

    lines.append("## Corpus-level profile (skew-aware)")
    lines.append("")
    for dim in ("lexical", "syntactic", "semantic", "pragmatic"):
        data = getattr(profile, dim)
        lines.append(f"### {dim.capitalize()}")
        lines.append("")
        rows = _scalar_rows(data)
        if rows:
            lines.append("| feature | value |")
            lines.append("|---|---|")
            for k, v in rows:
                lines.append(f"| {k} | {v} |")
        # non-scalar extras (ngrams, pos_dist, topics)
        for k, val in data.items():
            if isinstance(val, (dict, list)):
                lines.append(f"- **{k}**: {_fmt(val)}")
        lines.append("")

    lines.append("## Per-document-type profiles")
    lines.append("")
    for dt in sorted(profile.per_type):
        lines.append(f"### {dt}")
        lines.append("")
        dims = profile.per_type[dt]
        lines.append("| dimension | features |")
        lines.append("|---|---|")
        for dim in ("lexical", "syntactic", "semantic", "pragmatic"):
            lines.append(f"| {dim} | {_fmt(dims.get(dim, {}))} |")
        lines.append("")

    return "\n".join(lines)
