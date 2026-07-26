# Personal Style LLM

A local, single-user, fully offline pipeline that analyzes a personal corpus of
writing and produces a multi-dimensional **style profile** — a structured,
per-document-type characterization of *how* the author writes (lexical richness,
sentence structure, formality, readability, rhetorical habits).

The corpus never leaves the machine: no third-party APIs, no uploads, no
services. Everything runs from a single CLI against local files.

## Status

This repository implements **v1, phase A** — the project's primary deliverable:

- **Ingestion (R1):** PDF → clean UTF-8 text, per-document-type tagging, corpus
  reporting.
- **Style analysis (R2):** lexical / syntactic / semantic / pragmatic features
  computed per document type and aggregated into a skew-aware corpus profile.

The generation half — local-model drafting in the author's voice, exemplar
retrieval, and blind A/B evaluation — is specified in `personal-style-llm-spec.md`
and is the planned next phase.

## How it works

```
PDFs ──► Ingest ──► Cleaned corpus ──► Style Analyzer ──► StyleProfile (JSON)
         (R1)       + ingestion report   (R2)              + style report (Markdown)
```

**Ingestion** extracts text with PyMuPDF, strips boilerplate (running
headers/footers, page numbers, bibliography sections) and non-prose lines
(numeric data tables), applies a soft/logged PII redaction pass, tags each
document with a curated type (personal statement, cover letter, academic report,
assignment, reflection, extended essay), and flags any document large enough to
dominate the corpus.

**Style analysis** computes four dimensions of features:

| Dimension  | Example features |
|------------|------------------|
| Lexical    | type-token ratio, MTLD lexical diversity, top n-grams |
| Syntactic  | mean sentence length, POS distribution, parse depth, clause density |
| Semantic   | topic clusters, style-embedding centroid |
| Pragmatic  | Flesch-Kincaid grade, formality (Heylighen F-score), sentiment, rhetorical-marker rates |

Features are computed **per document type** and then combined into a
corpus-level profile using **equal weighting by type**, so a single unusually
long document cannot dominate the overall picture. Outputs are a versioned
`StyleProfile` JSON and a human-readable Markdown report.

## Setup (Windows, conda)

```powershell
conda create -n style-llm python=3.11 -y
conda activate style-llm
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

The first `analyze` run downloads the sentence-transformers embedding model
(~420 MB) once; thereafter the pipeline runs fully offline.

## Usage

Place the source PDFs in `Dataset/` (kept local and git-ignored) and map each
filename to a document type in `config/default.yaml`, then:

```powershell
stylellm ingest     # -> artifacts/corpus/ + artifacts/ingestion_report.md
stylellm analyze    # -> artifacts/style_profile.json + artifacts/style_report.md
```

All derived outputs land under `artifacts/` (git-ignored). Configuration —
random seed, document-type map, cleaning thresholds, embedding model — lives in
`config/default.yaml`.

## Testing

```powershell
pytest -m "not slow"   # fast unit + Hypothesis property tests
pytest -m slow         # spaCy + full pipeline on the real corpus
```

Property tests cover dominant-document flagging, skew-aware aggregation
invariance, and profile JSON round-tripping.

## Layout

```
src/stylellm/   config, models, ingest, features, style_analyzer, report, cli
config/         default.yaml (config-as-code)
tests/          unit + property/ (Hypothesis) + integration
Dataset/        source PDFs (local, git-ignored)
artifacts/      derived outputs (git-ignored)
```

## Design principles

- **Local and private by default** — the corpus and every derived artifact stay
  on local disk.
- **Right-sized** — dense in-memory analysis over a small corpus; no servers,
  databases, or distributed infrastructure.
- **Reproducible** — config-as-code with fixed seeds; the same inputs produce
  identical profiles.

See `personal-style-llm-spec.md` for the full specification, requirements
(R1–R7), and correctness properties (P1–P10).
