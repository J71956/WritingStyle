"""Shared fixtures: a tiny synthetic corpus that needs no PDFs or models."""

from __future__ import annotations

import pytest

from stylellm.models import Document


@pytest.fixture
def tiny_docs() -> list[Document]:
    return [
        Document(
            doc_id="ps1.pdf",
            doc_type="personal_statement",
            text=(
                "I am passionate about engineering. However, my journey began with "
                "curiosity. Therefore, I pursued research diligently and effectively."
            ),
            token_count=20,
        ),
        Document(
            doc_id="cl1.pdf",
            doc_type="cover_letter",
            text=(
                "Dear hiring manager, I write to apply for the analyst role. "
                "Moreover, my experience aligns strongly with your requirements."
            ),
            token_count=20,
        ),
        Document(
            doc_id="ia1.pdf",
            doc_type="academic_ia",
            text=(
                "This investigation examines thermal conduction. The temperature "
                "gradient drives heat transfer. Consequently, the model predicts loss."
            ),
            token_count=20,
        ),
    ]
