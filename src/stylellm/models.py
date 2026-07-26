"""Data models (spec §5).

Pydantic v2 models so serialized artifacts get a versioned schema and a
float-tolerant JSON round-trip for free (property P4). Field names are kept
identical to the spec.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

DocType = Literal[
    "personal_statement",
    "cover_letter",
    "academic_ia",
    "assignment",
    "reflection",
    "extended_essay",
]

# The four style dimensions, in stable order (used for reports and vectors).
DIMENSIONS = ("lexical", "syntactic", "semantic", "pragmatic")


def content_hash(text: str) -> str:
    """SHA-256 of normalized text — used for idempotent indexing (R3.4)."""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class Document(BaseModel):
    doc_id: str
    doc_type: DocType
    text: str  # cleaned, UTF-8
    token_count: int
    is_dominant: bool = False  # token_count > 40% of corpus total (R1.3)


class StyleProfile(BaseModel):
    version: str
    computed_at: datetime
    # Corpus-level (skew-aware) features per dimension.
    lexical: dict = Field(default_factory=dict)
    syntactic: dict = Field(default_factory=dict)
    semantic: dict = Field(default_factory=dict)
    pragmatic: dict = Field(default_factory=dict)
    # doc_type -> {lexical, syntactic, semantic, pragmatic}
    per_type: dict[str, dict] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_type: DocType
    chunk_index: int
    token_count: int  # <= 512
    style_tags: list[str] = Field(default_factory=list)
    text: str
    content_hash: str  # SHA-256, for idempotent indexing
    embedding: Optional[list[float]] = None


class GenerationResult(BaseModel):
    generated_text: str
    style_similarity: float
    exemplars_used: list[str] = Field(default_factory=list)  # chunk_ids
    latency_ms: int


class ABTrial(BaseModel):
    trial_id: str
    prompt: str
    output_styled: str  # with profile + exemplars (label hidden)
    output_plain: str  # plain prompt (label hidden)
    preferred: Optional[Literal["styled", "plain"]] = None
    rated_at: Optional[datetime] = None
