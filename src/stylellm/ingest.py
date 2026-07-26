"""Corpus ingestion & cleaning (R1).

PDF -> UTF-8 text, boilerplate stripped (running headers/footers, page numbers,
Extended-Essay reference section), tagged with a curated doc_type, flagged for
token dominance, and summarized in an ingestion report.

Properties: P1 (cleaning removes boilerplate), P2 (dominant-doc flagging).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from .config import Settings
from .models import Document, DocType

# --- token counting ---------------------------------------------------------

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def count_tokens(text: str) -> int:
    """Approximate token count as word count.

    Whitespace/punctuation word count is proportional to true model tokens, so
    it is sufficient for the dominance ratio (R1.3) and per-type distribution.
    """
    return len(_WORD_RE.findall(text))


# --- cleaning ---------------------------------------------------------------

_PAGE_NUM_RE = re.compile(r"^\s*(?:page\s+)?\d+\s*(?:/\s*\d+)?\s*$", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Conservative, single-line phone shape: no newlines, no decimal-devouring.
# Applied only to lines that contain letters (contact lines) so numeric data
# tables in the IAs/EE are never touched.
_PHONE_RE = re.compile(r"(?<![\w.])\+?\d[\d ()\-]{5,13}\d(?![\w.])")
_ALPHA_RE = re.compile(r"[A-Za-z]")


def extract_pages(pdf_path: Path) -> list[str]:
    """Return per-page plain text using PyMuPDF."""
    pages: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pages.append(page.get_text("text"))
    return pages


def _find_repeated_lines(pages: list[str], min_pages: int) -> set[str]:
    """Lines appearing (normalized) on >= min_pages distinct pages = boilerplate."""
    seen: Counter[str] = Counter()
    for page in pages:
        unique = {ln.strip() for ln in page.splitlines() if ln.strip()}
        for ln in unique:
            seen[ln] += 1
    return {ln for ln, n in seen.items() if n >= min_pages and len(ln) < 120}


def _strip_reference_section(text: str, pattern: str) -> str:
    """Drop everything from the first bibliography/references heading to EOF (P1)."""
    ref_re = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    m = ref_re.search(text)
    return text[: m.start()].rstrip() if m else text


def clean(pages: list[str], settings: Settings, strip_references: bool) -> str:
    """Remove running headers/footers, page numbers, and (optionally) references."""
    repeated = _find_repeated_lines(pages, settings.ingest.header_footer_min_pages)
    kept: list[str] = []
    for page in pages:
        for line in page.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped in repeated:  # running header/footer
                continue
            if _PAGE_NUM_RE.match(stripped):  # page number
                continue
            if settings.ingest.drop_nonprose_lines and not _ALPHA_RE.search(stripped):
                continue  # numeric data-table row / symbol-only line
            kept.append(stripped)
    text = "\n".join(kept)
    if strip_references:
        text = _strip_reference_section(text, settings.ingest.reference_heading_pattern)
    # Normalize whitespace: collapse >2 blank lines, unify to UTF-8 already.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def redact_pii(text: str) -> tuple[str, int]:
    """Soft, logged PII redaction of obvious emails/phones (R1.4). Returns count.

    Emails are always redacted. Phone numbers are redacted only on lines that
    contain letters — pure-numeric data-table rows (common in the IAs and the
    Extended Essay) are left untouched so the corpus isn't corrupted.
    """
    n = 0
    out_lines: list[str] = []
    for line in text.splitlines():
        line, k = _EMAIL_RE.subn("[EMAIL]", line)
        n += k
        if _ALPHA_RE.search(line):
            line, k = _PHONE_RE.subn("[PHONE]", line)
            n += k
        out_lines.append(line)
    return "\n".join(out_lines), n


# --- classification ---------------------------------------------------------

_VALID_TYPES = {
    "personal_statement",
    "cover_letter",
    "academic_ia",
    "assignment",
    "reflection",
    "extended_essay",
}


def classify_doc_type(filename: str, doc_type_map: dict[str, str]) -> DocType:
    """Curated filename -> doc_type lookup (reliable at 17 files)."""
    dt = doc_type_map.get(filename)
    if dt is None:
        raise KeyError(f"No doc_type mapping for {filename!r}; add it to config/default.yaml")
    if dt not in _VALID_TYPES:
        raise ValueError(f"Invalid doc_type {dt!r} for {filename!r}")
    return dt  # type: ignore[return-value]


# --- orchestration ----------------------------------------------------------


@dataclass
class IngestResult:
    documents: list[Document]
    warnings: list[str] = field(default_factory=list)
    pii_redactions: dict[str, int] = field(default_factory=dict)


def _extraction_warning(doc_id: str, text: str, file_bytes: int) -> str | None:
    """Flag likely-scanned/image PDFs (low text yield for a large file)."""
    words = count_tokens(text)
    if not text:
        return f"{doc_id}: extracted no text (likely a scanned/image PDF)"
    alpha = sum(c.isalpha() for c in text)
    alpha_ratio = alpha / max(len(text), 1)
    if file_bytes > 1_000_000 and words < 500:
        return f"{doc_id}: {words} words from {file_bytes // 1024} KB file — possible scanned PDF / low text yield"
    if alpha_ratio < 0.5:
        return f"{doc_id}: low alphabetic ratio ({alpha_ratio:.2f}) — extraction may be garbled"
    return None


def flag_dominant(documents: list[Document], threshold: float) -> list[Document]:
    """Set is_dominant on any doc exceeding `threshold` of corpus tokens (P2)."""
    total = sum(d.token_count for d in documents) or 1
    for doc in documents:
        doc.is_dominant = doc.token_count > threshold * total
    return documents


def ingest_corpus(settings: Settings) -> IngestResult:
    """Extract, clean, tag, and flag all PDFs in the dataset directory (R1)."""
    dataset = settings.dataset_path
    pdfs = sorted(p for p in dataset.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {dataset}")

    result = IngestResult(documents=[])

    for pdf in pdfs:
        doc_id = pdf.name
        doc_type = classify_doc_type(doc_id, settings.doc_type_map)
        pages = extract_pages(pdf)
        strip_refs = doc_type == "extended_essay"
        text = clean(pages, settings, strip_references=strip_refs)

        if settings.ingest.redact_pii:
            text, n = redact_pii(text)
            if n:
                result.pii_redactions[doc_id] = n

        warn = _extraction_warning(doc_id, text, pdf.stat().st_size)
        if warn:
            result.warnings.append(warn)

        tokens = count_tokens(text)
        result.documents.append(
            Document(doc_id=doc_id, doc_type=doc_type, text=text, token_count=tokens)
        )

    # Dominance flag (P2): any doc > threshold * corpus total tokens.
    flag_dominant(result.documents, settings.ingest.dominance_threshold)
    return result


def persist_corpus(result: IngestResult, settings: Settings) -> Path:
    """Write cleaned text + documents.jsonl to artifacts/corpus/. Returns dir."""
    import json

    corpus_dir = settings.artifacts_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    jsonl = corpus_dir / "documents.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for doc in result.documents:
            (corpus_dir / f"{doc.doc_id}.txt").write_text(doc.text, encoding="utf-8")
            f.write(doc.model_dump_json(exclude={"text"}) + "\n")
    return corpus_dir
