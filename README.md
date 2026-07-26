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

The Extended Essay's **prose is not in its PDF text layer**: only a large numeric
data appendix is extractable (≈64.8k numeric "words", 8 lines with any letters).
The essay body is image-based and would require OCR to recover. Consequences:

- The spec's premise that `EE.pdf` dominates the *style* signal does not hold —
  once numeric data-table lines are dropped (`ingest.drop_nonprose_lines`), EE
  contributes ~21 words and is excluded by `analyzer.min_doc_tokens`.
- The style profile is therefore built from the **16 real-prose documents**.
- To include EE's actual writing, add an OCR pass (deferred; Tesseract).

## Layout

```
src/stylellm/   config, models, ingest, features, style_analyzer, report, cli
config/         default.yaml (config-as-code)
tests/          unit + property/ (Hypothesis) + integration
artifacts/      derived outputs (git-ignored)
```
