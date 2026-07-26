# Personal Style LLM

Local, single-user, offline analysis of a personal writing corpus. This is **v1
scope A**: corpus ingestion (R1) and a multi-dimensional, skew-aware **style
profile + report** (R2) — the spec's primary deliverable. Retrieval, generation,
and evaluation are the gated follow-on (see `personal-style-llm-spec.md` §11–12).

## Setup (Windows, conda)

```powershell
conda create -n style-llm python=3.11 -y
conda activate style-llm
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

The first `analyze` run downloads the sentence-transformers embedding model
(~420 MB) once, then works offline.

## Usage

```powershell
stylellm ingest     # -> artifacts/corpus/ + artifacts/ingestion_report.md
stylellm analyze    # -> artifacts/style_profile.json + artifacts/style_report.md
```

Everything under `artifacts/` is derived and git-ignored. Configuration
(seed, doc_type map, thresholds) lives in `config/default.yaml`.

## Tests

```powershell
pytest -m "not slow"   # fast unit + Hypothesis property tests
pytest -m slow         # spaCy + full pipeline on the real corpus
```

## Key data finding — EE.pdf

`EE.pdf` is **entirely a numeric data appendix** — 395 pages of simulation
data tables, with **no prose anywhere in the file** (verified: 0 pages contain
more than 15 alphabetic words; no embedded images; page renders show only
numeric columns). Its ≈64.8k extractable "words" are decimal data values.

- The essay's actual prose (introduction, methodology, analysis, conclusion) is
  **not in this PDF** — it is a separate document not present in `Dataset/`.
  OCR cannot recover prose that the file does not contain.
- The spec's premise that `EE.pdf` dominates the *style* signal therefore does
  not hold. Numeric data-table lines are dropped (`ingest.drop_nonprose_lines`),
  leaving ~21 tokens, so EE is excluded by `analyzer.min_doc_tokens`.
- The style profile is built from the **16 real-prose documents**. To include
  the Extended Essay's writing, add the real essay document to `Dataset/`.

## Layout

```
src/stylellm/   config, models, ingest, features, style_analyzer, report, cli
config/         default.yaml (config-as-code)
tests/          unit + property/ (Hypothesis) + integration
artifacts/      derived outputs (git-ignored)
```
