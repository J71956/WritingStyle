"""Ingest unit tests: cleaning (P1), doc_type, dominance (P2), PII safety."""

from __future__ import annotations

from stylellm.config import load_settings
from stylellm.ingest import (
    _strip_reference_section,
    classify_doc_type,
    clean,
    flag_dominant,
    redact_pii,
)
from stylellm.models import Document

SETTINGS = load_settings()


def test_clean_strips_running_headers_and_page_numbers():
    # "My Name" appears on all 3 pages (running header); page numbers vary.
    pages = [
        "My Name\nIntroduction paragraph one.\n1",
        "My Name\nBody paragraph two here.\n2",
        "My Name\nConclusion paragraph three.\n3",
    ]
    out = clean(pages, SETTINGS, strip_references=False)
    assert "My Name" not in out  # P1: running header removed
    assert "1" not in out.split()  # page numbers removed
    assert "Introduction paragraph one." in out
    assert "Conclusion paragraph three." in out


def test_clean_drops_nonprose_numeric_lines():
    pages = ["Real prose sentence here.\n0.0 811585.124 353.150\n42\nAnother sentence."]
    out = clean(pages, SETTINGS, strip_references=False)
    assert "Real prose sentence here." in out
    assert "811585" not in out  # numeric data-table row dropped


def test_strip_reference_section():
    text = "Body of the essay.\nReferences\nSmith, J. (2020). A paper."
    out = _strip_reference_section(text, SETTINGS.ingest.reference_heading_pattern)
    assert "Body of the essay." in out
    assert "Smith" not in out


def test_classify_doc_type_valid_and_missing():
    dtmap = {"a.pdf": "cover_letter"}
    assert classify_doc_type("a.pdf", dtmap) == "cover_letter"
    try:
        classify_doc_type("missing.pdf", dtmap)
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_redact_pii_email_and_contact_phone():
    text, n = redact_pii("Contact me at john.doe@example.com or call +852 1234 5678.")
    assert "[EMAIL]" in text and "[PHONE]" in text
    assert n == 2


def test_redact_pii_leaves_numeric_tables_intact():
    # The exact failure that collapsed EE: numeric rows must survive untouched.
    table = "0.0 811585.124 353.150\n1.0 811509.537 353.145\n2.0 811433.958"
    out, n = redact_pii(table)
    assert out == table  # no alphabetic content -> phone rule never fires
    assert n == 0


def test_flag_dominant():
    docs = [
        Document(doc_id="big.pdf", doc_type="extended_essay", text="x", token_count=800),
        Document(doc_id="s1.pdf", doc_type="cover_letter", text="x", token_count=100),
        Document(doc_id="s2.pdf", doc_type="cover_letter", text="x", token_count=100),
    ]
    flag_dominant(docs, 0.40)
    assert docs[0].is_dominant is True  # 800/1000 = 80% > 40%
    assert docs[1].is_dominant is False
