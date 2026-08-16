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


class IndexCfg(BaseModel):
    # Paragraph-based chunking with a token cap (R3.1): pack whole paragraphs up
    # to the target; a single paragraph over the hard max is sentence-split.
    chunk_target_tokens: int = 256
    chunk_max_tokens: int = 512
    # Embedding model for chunks; keep aligned with the analyzer's so query and
    # index share a space (verify tier in spec §8).
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    # Skip documents too short to yield usable prose exemplars (e.g. EE.pdf's
    # numeric appendix), mirroring the analyzer's min_doc_tokens gate.
    exclude_below_min_tokens: bool = True


class RetrieveCfg(BaseModel):
    top_k: int = 5
    # MMR trade-off: 1.0 = pure relevance, 0.0 = pure diversity (R4.3).
    mmr_lambda: float = 0.5
    # Cosine above this = near-duplicate; such chunks are collapsed (P8).
    dedupe_threshold: float = 0.95


class GenerateCfg(BaseModel):
    # Local instruct model via Hugging Face transformers by default (R5.1).
    # "ollama" is the original HTTP path; "fake" is an offline, deterministic
    # backend for tests / when no local model is installed.
    backend: str = "hf"  # hf | ollama | fake
    # HF repo id (backend=hf) or Ollama tag (backend=ollama).
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    max_tokens: int = 512
    temperature: float = 0.7

    # --- hf backend ---------------------------------------------------------
    # Pin a commit sha for reproducibility; None tracks the repo's main branch.
    revision: str | None = None
    device: str = "auto"  # passed to device_map
    dtype: str = "auto"
    # Suppress chain-of-thought on hybrid-reasoning models so max_new_tokens is
    # spent on the answer. None = don't pass the flag at all.
    enable_thinking: bool | None = False
    # 4-bit weights (needs bitsandbytes) for models that won't fit in VRAM.
    load_in_4bit: bool = False

    # --- ollama backend -----------------------------------------------------
    ollama_host: str = "http://localhost:11434"
    # Ollama's equivalent of enable_thinking; null for non-reasoning models
    # (e.g. llama3.1) that reject the `think` parameter.
    think: bool | None = False


class EvalCfg(BaseModel):
    # A/B + stylometric-similarity trials are appended here (under artifacts/).
    log_file: str = "eval_log.jsonl"
    # Floor on per-feature std when standardizing the stylometric vector, so a
    # constant feature can't produce a divide-by-zero (R6.1).
    scale_floor: float = 1e-6


class Settings(BaseModel):
    seed: int = 42
    paths: PathsCfg = Field(default_factory=PathsCfg)
    ingest: IngestCfg = Field(default_factory=IngestCfg)
    analyzer: AnalyzerCfg = Field(default_factory=AnalyzerCfg)
    index: IndexCfg = Field(default_factory=IndexCfg)
    retrieve: RetrieveCfg = Field(default_factory=RetrieveCfg)
    generate: GenerateCfg = Field(default_factory=GenerateCfg)
    evaluate: EvalCfg = Field(default_factory=EvalCfg)
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
