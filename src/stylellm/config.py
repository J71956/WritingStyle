"""Configuration (R7): config-as-code with fixed seeds.

A single YAML file (config/default.yaml) is the source of truth. `Settings`
is a typed view over it; `set_seeds()` pins randomness for reproducibility
(P10).
"""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, Field

# Repo root = two levels up from this file (src/stylellm/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


class PathsCfg(BaseModel):
    dataset_dir: str = "Dataset"
    artifacts_dir: str = "artifacts"


class IngestCfg(BaseModel):
    dominance_threshold: float = 0.40
    header_footer_min_pages: int = 3
    redact_pii: bool = True
    drop_nonprose_lines: bool = True
    reference_heading_pattern: str = r"^\s*(works cited|bibliography|references|reference list)\s*$"


class AnalyzerCfg(BaseModel):
    skew_mode: str = "equal_weight_by_type"
    # Documents below this token count can't yield reliable style features and
    # are excluded from analysis (e.g. EE.pdf, whose prose isn't in its text
    # layer — only a numeric appendix is extractable).
    min_doc_tokens: int = 50
    top_ngrams: int = 20
    topic_clusters: int = 6
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    profile_version: str = "1.0"


class Settings(BaseModel):
    seed: int = 42
    paths: PathsCfg = Field(default_factory=PathsCfg)
    ingest: IngestCfg = Field(default_factory=IngestCfg)
    analyzer: AnalyzerCfg = Field(default_factory=AnalyzerCfg)
    doc_type_map: dict[str, str] = Field(default_factory=dict)

    # Resolved absolute paths (repo-root-relative inputs resolved on load).
    @property
    def dataset_path(self) -> Path:
        return _resolve(self.paths.dataset_dir)

    @property
    def artifacts_path(self) -> Path:
        return _resolve(self.paths.artifacts_dir)


def _resolve(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)


@lru_cache(maxsize=None)
def load_settings(config_path: str | Path | None = None) -> Settings:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return Settings()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Settings(**data)


def set_seeds(seed: int) -> None:
    """Pin RNGs so the same config + corpus produces identical outputs (P10)."""
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional at analysis time; seed it if present.
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass
