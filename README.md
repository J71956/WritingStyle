# Personal Style LLM

A local, single-user, fully offline pipeline that analyzes a personal corpus of
writing and produces a multi-dimensional **style profile** — a structured,
per-document-type characterization of *how* the author writes (lexical richness,
sentence structure, formality, readability, rhetorical habits).

The corpus never leaves the machine: no third-party APIs, no uploads, no
services. Everything runs from a single CLI against local files.

## Status

This repository implements the full **v1** pipeline — analysis and generation:

- **Ingestion (R1):** PDF → clean UTF-8 text, per-document-type tagging, corpus
  reporting.
- **Style analysis (R2):** lexical / syntactic / semantic / pragmatic features
  computed per document type and aggregated into a skew-aware corpus profile.
- **Chunk & index (R3):** paragraph-based chunking (≤512 tokens), local
  in-memory embedding index, idempotent rebuild.
- **Retrieval (R4):** dense cosine exemplar search with `doc_type` filtering and
  MMR diversity.
- **Generation (R5):** prompt-based drafting in the author's voice via a local
  instruct model (llama.cpp, on-device), combining the style profile with
  retrieved exemplars — either writing from a request (`generate`) or restyling
  an existing draft in place (`rewrite`, R5.5).
- **Evaluation (R6):** stylometric similarity + a blind A/B harness, logged
  locally.

Fine-tuning, hybrid retrieval, and a served endpoint remain deferred to v2 and
gated on v1 evidence (see `personal-style-llm-spec.md` §12).

## How it works

```
PDFs ──► Ingest ──► Cleaned corpus ──► Style Analyzer ──► StyleProfile (JSON)
         (R1)       + ingestion report   (R2)              + style report (Markdown)
                          │
                          ├──► Chunk + Embed ──► Local index (R3)
                          │                          │
                          │              Dense retrieval + MMR (R4)
                          │                          │
                          └──► Prompt Orchestrator ──┴──► Local model ──► Draft (R5)
                                     (profile summary + exemplars)          │
                                                              Stylometric similarity
                                                              + blind A/B (R6)
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
stylellm index      # -> artifacts/index/ (chunks.jsonl + embeddings.npy)
```

Then generate and evaluate:

```powershell
# Draft in the author's voice (needs a local model; see below):
stylellm generate "a cover letter for a data analyst internship" --doc-type cover_letter

# Restyle a draft you already wrote, keeping its content:
stylellm rewrite --file draft.txt --doc-type cover_letter --out styled.txt

# Blind A/B: styled (profile + exemplars) vs. plain, logged to artifacts/eval_log.jsonl:
stylellm ab "a short personal statement about engineering" --doc-type personal_statement
```

`generate` invents content in your voice; `rewrite` takes your draft as fixed
and moves only the voice. Rewrite reports three numbers: the style similarity
before and after (so you can see whether it actually moved toward your voice),
**retention** — how much of the draft's vocabulary survived — and **expansion** —
how much longer the result is in content words. The two content numbers are
one-sided guards that only work as a pair: retention catches a rewrite that
dropped your substance, expansion catches one that padded it with invention,
which retention cannot see. Neither checks facts; read the output.

Short drafts provoke invention badly — a one-line note reliably comes back as a
fully structured document with details you never wrote. Give it a few sentences
at least, and treat a warning as a reason to re-read rather than a veto.

Generation defaults to a local **llama.cpp** server holding a quantized GGUF —
`unsloth/Qwen3.5-9B-GGUF` at `Q4_K_M` (`config/default.yaml` → `generate.model`).
At ~5.7 GB it fits an 8 GB card that cannot hold the same model at bf16.
Pre-download the weights once, then start the server and leave it running:

```powershell
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf
llama-server -hf unsloth/Qwen3.5-9B-GGUF:Q4_K_M --no-mmproj --port 8080 -ngl 99 -c 8192
```

`stylellm generate` then talks to it over localhost. It checks which GGUF the
server actually has loaded and refuses to run on a mismatch, so a server left
over from another model cannot silently mislabel a result. Set
`generate.gguf_file: null` to accept whatever is being served.

Set `HF_HUB_OFFLINE=1` after the download and everything runs on-device — the
GGUF is fetched from the Hub once, inference is a local process bound to
localhost, and no hosted inference API is used, so no corpus text ever leaves
the machine. `generate.enable_thinking` defaults to `false` so
hybrid-reasoning models spend `max_tokens` on the answer rather than a
chain-of-thought; set it to `null` for non-reasoning models.

Three other backends are available via `--backend`: `hf` (unquantized weights
in-process via `transformers` — `pip install -e ".[hf]"`, plus
`generate.load_in_4bit: true` and `".[quant]"` if they won't fit VRAM),
`ollama` (the original HTTP path — `ollama serve` plus an `ollama pull`ed tag
in `generate.model`, with `generate.think` in place of `enable_thinking`), and
`fake` for an offline dry-run of the orchestration with no model at all.

All derived outputs land under `artifacts/` (git-ignored). Configuration —
random seed, document-type map, cleaning thresholds, embedding model, chunking,
retrieval, and generation settings — lives in `config/default.yaml`.

## Experiments and the paper

`experiments/` holds the test campaign that feeds the write-up in `paper/`:

```powershell
python experiments/run_campaign.py --backend fake --limit 8   # offline dry run
python experiments/run_campaign.py                            # the real grid
python experiments/run_ab_session.py --trials 20              # human blind A/B
python experiments/analyze_results.py                         # tables + figures
```

Unlike the CLI, the campaign starts and stops its own `llama-server` — one per
model slice, since a server holds a single GGUF — so port 8080 must be free
when it runs. Point `generate.llamacpp_bin` at the executable if it isn't on
PATH.

The grid sweeps 2 models × 18 prompts × 4 ablation conditions × 3 repetitions,
scoring every draft with the standardized stylometric metric and appending rows
to `artifacts/eval/campaign.jsonl`. It is resumable — a re-run skips `run_id`s
already logged. See `paper/README.md` for the LaTeX build.

## Testing

```powershell
pytest -m "not slow"   # fast unit + Hypothesis property tests
pytest -m slow         # spaCy + full pipeline on the real corpus
```

Property tests cover dominant-document flagging, skew-aware aggregation
invariance, profile JSON round-tripping, chunk token-cap invariants (P5), and
retrieval bound + diversity (P8).

## Layout

```
src/stylellm/   config, models, ingest, features, style_analyzer, report,
                embeddings, index, retrieve, generate, evaluate, cli
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
